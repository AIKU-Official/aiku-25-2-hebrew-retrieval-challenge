# /app/project_code/taejin/model_taejin_01.py
# Zero-shot retriever: (Knesset-)DictaBERT (embedding only)
# Zero-shot reranker: BGE reranker (cross-encoder)
# Public API: preprocess(corpus_dict) -> dict, predict(query_dict, preprocessed_data) -> list[{'paragraph_uuid','score'}]

import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# ===================== Config (env로 변경 가능) ==============================
KD_DEFAULT_ID       = os.getenv("KD_BERT_ID", "GiliGold/Knesset-DictaBERT")
BGE_DEFAULT_ID      = os.getenv("BGE_RERANKER_ID", "BAAI/bge-reranker-v2-m3")
EMBED_MAX_LEN       = int(os.getenv("EMBED_MAX_LEN", "512"))
EMBED_BS            = int(os.getenv("EMBED_BS", "32"))
RERANK_BS           = int(os.getenv("RERANK_BS", "4"))
TOPK_CANDIDATES     = int(os.getenv("TOPK_CANDS", "100"))
TOPK_RETURN         = int(os.getenv("TOPK_RETURN", "20"))
USE_FP16            = os.getenv("USE_FP16", "1") != "0"
# ============================================================================

# --------------------- KD retriever (네가 준 코드 + 경로해결) -----------------
def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def _resolve_knesset_dictabert_path():
    # 1) 프로젝트 루트의 models/
    cands = [
        "models/dictabert",
        "models/Knesset-DictaBERT",
        # 2) baseline 폴더 안 models/
        "Dataset and Baseline/baseline_submission/models/dictabert",
        "Dataset and Baseline/baseline_submission/models/Knesset-DictaBERT",
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    # 3) 로컬에 없으면 HF에서 내려받아 쓰기
    return KD_DEFAULT_ID

class KDZeroShotRetriever:
    """
    Zero-shot (Knesset-)DictaBERT 임베딩 기반 retriever.
    - E5처럼 prefix 필요 없음.
    - embed_texts(...) 시그니처를 E5와 동일하게 맞춰서 기존 코드 손대지 않도록 구현.
    """
    def __init__(self, model_name=None, device=None, use_fp16: bool = USE_FP16):
        model_id = model_name or _resolve_knesset_dictabert_path()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        base = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float32
        self.model = base.to(self.device, dtype=dtype).eval()
        self.corpus_ids: List[str] = []
        self.corpus_embeddings: np.ndarray | None = None

    @torch.no_grad()
    def _encode(self, texts, batch_size=EMBED_BS, max_length=EMBED_MAX_LEN):
        outs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[i:i+batch_size],
                padding=True, truncation=True, max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            out = self.model(**enc)
            emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            outs.append(emb.detach().cpu())
            del enc, out, emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0).numpy()

    # E5 인터페이스와 동일한 시그니처 유지
    def embed_texts(self, texts, is_query=False, batch_size=EMBED_BS):
        return self._encode(texts, batch_size=batch_size)

# --------------------- BGE reranker (로컬 우선 경로 해결) ---------------------
def _resolve_bge_path():
    cands = [
        "models/bge-reranker-v2-m3",
        "Dataset and Baseline/baseline_submission/models/bge-reranker-v2-m3",
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    return BGE_DEFAULT_ID

class BGEReranker:
    """Zero-shot BGE reranker (cross-encoder, sequence classification logits 사용)"""
    def __init__(self, model_name=None, device=None, use_fp16: bool = USE_FP16):
        model_id = model_name or _resolve_bge_path()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
        ).to(self.device, dtype=dtype).eval()

    @torch.no_grad()
    def rerank(self, query_text: str, passages: List[str], passage_ids: List[str], top_k: int = TOPK_RETURN) -> List[Tuple[str, float]]:
        if not passages:
            return []
        scores: List[float] = []
        bs = max(1, RERANK_BS)
        for i in range(0, len(passages), bs):
            batch_p = passages[i:i+bs]
            batch_q = [query_text] * len(batch_p)
            inputs = self.tok(batch_q, batch_p, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            # 모형별 로짓 형태 대응
            if logits.ndim == 1:
                s = logits
            elif logits.shape[1] == 1:
                s = logits.squeeze(-1)
            else:
                s = logits[:, 1]  # positive class
            scores.extend(s.detach().cpu().numpy().tolist())
            del inputs, outputs, logits, s
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pairs = list(zip(passage_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

# --------------------- Globals (preprocess 결과 캐시) -------------------------
_retriever: KDZeroShotRetriever | None = None
_reranker:  BGEReranker | None = None
_corpus_texts: Dict[str, str] = {}
# -----------------------------------------------------------------------------

# ============================= Public API ====================================
def preprocess(corpus_dict: Dict[str, Dict]) -> Dict:
    """코퍼스 임베딩 계산 및 캐시 구성"""
    global _retriever, _reranker, _corpus_texts

    print("="*60)
    print("PREPROCESSING: KD-BERT (zero-shot) + BGE reranker")
    print("="*60)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # 1) 모델 초기화 (로컬 폴더 우선, 없으면 HF에서 받음)
    print(f"Loading KD-BERT retriever: {_resolve_knesset_dictabert_path()}")
    _retriever = KDZeroShotRetriever()
    print(f"Loading BGE reranker: {_resolve_bge_path()}")
    _reranker  = BGEReranker()

    # 2) 텍스트 수집
    ids = list(corpus_dict.keys())
    passages = [doc.get("passage", doc.get("text", "")) for doc in corpus_dict.values()]
    _corpus_texts = {ids[i]: passages[i] for i in range(len(ids))}

    # 3) 임베딩 계산
    print(f"Embedding {len(ids)} passages...")
    emb = _retriever.embed_texts(passages, is_query=False, batch_size=EMBED_BS)
    print("✓ embeddings:", emb.shape)

    _retriever.corpus_ids = ids
    _retriever.corpus_embeddings = emb

    return {
        "retriever": _retriever,
        "reranker": _reranker,
        "corpus_ids": ids,
        "corpus_embeddings": emb,
        "corpus_texts": _corpus_texts,
        "num_documents": len(ids),
    }

def predict(query: Dict[str, str], preprocessed_data: Dict) -> List[Dict[str, float]]:
    """KD-BERT cosine retrieval → BGE rerank → 상위 결과 반환"""
    global _retriever, _reranker, _corpus_texts
    q = query.get("query", "")
    if not q:
        return []

    # preprocess로부터 주입 or 글로벌 캐시 사용
    retr = _retriever or preprocessed_data.get("retriever")
    rerk = _reranker  or preprocessed_data.get("reranker")
    _corpus_texts = _corpus_texts or preprocessed_data.get("corpus_texts", {})
    if retr is None or rerk is None:
        print("Error: retriever/reranker not initialized")
        return []

    # Stage 1: cosine retrieval
    q_emb = retr.embed_texts([q], is_query=True, batch_size=max(1, EMBED_BS // 2))
    # L2 normalized이므로 내적이 곧 cosine
    scores = (q_emb @ retr.corpus_embeddings.T)[0]  # (N,)
    top_idx = np.argsort(scores)[::-1][:TOPK_CANDIDATES]
    cand_ids = [retr.corpus_ids[i] for i in top_idx]
    cand_txt = [_corpus_texts[i] for i in cand_ids]

    # Stage 2: BGE rerank
    reranked = rerk.rerank(q, cand_txt, cand_ids, top_k=TOPK_RETURN)
    return [{"paragraph_uuid": pid, "score": float(s)} for pid, s in reranked]

