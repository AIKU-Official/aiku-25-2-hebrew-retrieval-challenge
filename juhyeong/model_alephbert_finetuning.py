
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ================= Env knobs =================
QWEN_LOCAL_DIRS = [

    "models/Qwen3-Embedding-4B",
    "Dataset and Baseline/baseline_submission/models/Qwen3-Embedding-4B",
]
QWEN_ENV_VARS = ["QWEN_EMB_DIR", "QWEN_EMBEDDING_DIR", "QWEN_DIR"]


QWEN_QUERY_PREFIX = os.getenv("QWEN_QUERY_PREFIX", "")   
QWEN_PASS_PREFIX  = os.getenv("QWEN_PASS_PREFIX", "")    

RETRIEVER_ENV_VARS = ["DICTA_MODEL_DIR", "DICTABERT_DIR", "HEB_BERT_DIR"]
RETRIEVER_LOCAL_DIRS = [
    "models/alephbert_fine_tuning",
]

EMBED_MAX_LEN = int(os.getenv("EMBED_MAX_LEN", "512"))
EMBED_BS      = int(os.getenv("EMBED_BS", "32"))
QWEN_BS       = int(os.getenv("QWEN_BS", "32"))
USE_FP16      = os.getenv("USE_FP16", "1") != "0"
TOPK_CAND     = int(os.getenv("TOPK_CAND", "100"))
TOPK_RETURN   = int(os.getenv("TOPK_RETURN", "20"))

# ============================================================
# 공통 유틸
# ============================================================
def _dtype_by_device(device: str) -> torch.dtype:
    return torch.float16 if (device.startswith("cuda") and USE_FP16) else torch.float32

def _first_dir(cands: List[str]) -> str | None:
    for c in cands:
        if c and Path(c).is_dir():
            return c
    return None

def _resolve_model_dir(*, env_vars: List[str], local_candidates: List[str], tag: str) -> str:
    for var in env_vars:
        val = os.getenv(var, "").strip()
        if val and Path(val).is_dir():
            print(f"[{tag}] Using env path: {val}")
            return val
    hit = _first_dir(local_candidates)
    if hit:
        print(f"[{tag}] Using local model: {hit}")
        return hit
    tried = "\n - ".join([*env_vars, *local_candidates])
    raise FileNotFoundError(
        f"[{tag}] 모델 폴더를 찾지 못했습니다. 다음 후보를 확인하세요:\n - {tried}\n"
        f"※ 최종 폴더( config.json / tokenizer.* / model.safetensors|pytorch_model.bin )를 준비하세요."
    )

# ============================================================
# AlephBERT (DictaBERT 계열) Retriever  
# ============================================================
class DictaRetriever:
    """
    네가 파인튜닝한 AlephBERT 체크포인트 포함. mean-pooling + L2 normalize.
    """
    def __init__(self, model_name: str | None = None, device: str | None = None):
        if model_name is None:
            here = Path(os.path.dirname(os.path.abspath(__file__)))
            # 여기 기준 상대경로 후보도 추가하고 싶으면 여기에 더 넣어도 됨
            model_name = _resolve_model_dir(
                env_vars=RETRIEVER_ENV_VARS,
                local_candidates=RETRIEVER_LOCAL_DIRS,
                tag="RETRIEVER"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RETRIEVER] Loading on device: {self.device}")

        dtype = _dtype_by_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, local_files_only=True).to(self.device)
        self.model.eval()

        self.hidden = getattr(self.model.config, "hidden_size", 768)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.corpus_ids: List[str] = []
        self.corpus_embeddings: np.ndarray | None = None

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = EMBED_BS, max_length: int = EMBED_MAX_LEN) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.hidden), dtype=np.float32)

        outs: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = [(t or "").strip() for t in texts[i:i + batch_size]]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(self.device)

            out = self.model(**enc)
            last = out.last_hidden_state               # [B, T, H]
            attn = enc["attention_mask"]               # [B, T]

            mask = attn.unsqueeze(-1).expand(last.size()).float()
            summed = (last * mask).sum(dim=1)          # [B, H]
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / denom                    # mean pooling
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(pooled.detach().cpu())

            del enc, out, last, attn, mask, summed, denom, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(outs, dim=0).numpy().astype(np.float32)

# ============================================================
# Qwen3-Embedding-4B 기반 임베딩 재순위기(dual-encoder rerank)
# ============================================================
class QwenEmbeddingReranker:
    """
    Cross-encoder가 아님. 질의/문단을 각각 임베딩하여 코사인 점수로 재순위.
    BGE cross-encoder 대체로 '임시 2단계 dense rerank' 용으로 사용.
    """
    def __init__(self, model_dir: str | None = None, device: str | None = None):
        if model_dir is None:
            model_dir = _resolve_model_dir(
                env_vars=QWEN_ENV_VARS,
                local_candidates=QWEN_LOCAL_DIRS,
                tag="QWEN-EMB"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[QWEN-EMB] Loading on device: {self.device}")

        dtype = _dtype_by_device(self.device)
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype, local_files_only=True).to(self.device)
        self.model.eval()

        self.hidden = getattr(self.model.config, "hidden_size", 1536)  # Qwen3-Embedding-4B는 1536 차원(모델에 따라 다를 수 있음)

    @torch.no_grad()
    def _embed(self, texts: List[str], batch_size: int = QWEN_BS, max_length: int = EMBED_MAX_LEN) -> np.ndarray:
        outs: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tok(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(self.device)   
            out = self.model(**enc)
            last = out.last_hidden_state
            attn = enc["attention_mask"]
            mask = attn.unsqueeze(-1).expand(last.size()).float()
            pooled = (last * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(pooled.detach().cpu())

            del enc, out, last, attn, mask, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0).numpy().astype(np.float32)

    @torch.no_grad()
    def rerank(self, query_text: str, passages: List[str], passage_ids: List[str], top_k: int = TOPK_RETURN) -> List[Tuple[str, float]]:
        if not passages:
            return [] 
        # prefix 옵션
        q = QWEN_QUERY_PREFIX + (query_text or "")
        ps = [QWEN_PASS_PREFIX + (p or "") for p in passages]

        q_emb = self._embed([q], batch_size=1)            # [1, d]
        p_emb = self._embed(ps, batch_size=QWEN_BS)       # [N, d]
        scores = (q_emb @ p_emb.T)[0]                     # 내적 == cosine (정규화済)

        idx = np.argsort(scores)[::-1][:min(top_k, len(scores))]
        return [(passage_ids[i], float(scores[i])) for i in idx]

# ============================================================
# 파이프라인 API (preprocess / predict)
# ============================================================
retriever: DictaRetriever | None = None
reranker:  QwenEmbeddingReranker | None = None
corpus_texts: Dict[str, str] = {}

def preprocess(corpus_dict: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    AlephBERT 임베딩 + Qwen(임베딩 재순위) 준비
    """
    global retriever, reranker, corpus_texts

    print("=" * 60)
    print("PREPROCESSING: AlephBERT (fine-tuned) + Qwen3-Embedding-4B rerank")
    print("=" * 60)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print("Loading retriever (AlephBERT fine-tuned preferred)...")
    retriever = DictaRetriever()

    print("Loading reranker (Qwen3-Embedding-4B)...")
    reranker = QwenEmbeddingReranker()

    print(f"Preparing corpus with {len(corpus_dict)} documents...")

    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]
    corpus_texts.clear()
    corpus_texts.update({doc_id: passages[i] for i, doc_id in enumerate(retriever.corpus_ids)})

    print("Computing retriever embeddings...")
    retriever.corpus_embeddings = retriever.embed_texts(passages, batch_size=EMBED_BS, max_length=EMBED_MAX_LEN)

    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated embeddings for {len(retriever.corpus_ids)} documents")
    print(f"✓ Embedding matrix shape: {retriever.corpus_embeddings.shape}")

    return {
        'retriever': retriever,
        'reranker': reranker,
        'corpus_ids': retriever.corpus_ids,
        'corpus_embeddings': retriever.corpus_embeddings,
        'corpus_texts': corpus_texts,
        'num_documents': len(corpus_dict)
    }

def predict(query: Dict[str, str], preprocessed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    1) AlephBERT 코사인 top-K 후보
    2) Qwen 임베딩 기반 재순위(top-K 최종 20)
    """
    global retriever, reranker, corpus_texts

    query_text = (query or {}).get('query', '').strip()
    if not query_text:
        return []

    if retriever is None or reranker is None:
        retriever = preprocessed_data.get('retriever')
        reranker  = preprocessed_data.get('reranker')
        corpus_texts = preprocessed_data.get('corpus_texts', {})
        if retriever is None or reranker is None:
            print("Error: Missing retriever or reranker in preprocessed data")
            return []

    try:
        # Stage 1: dense retrieval
        q_emb = retriever.embed_texts([query_text], batch_size=1, max_length=EMBED_MAX_LEN)
        scores = cosine_similarity(q_emb, retriever.corpus_embeddings)[0]

        K = min(TOPK_CAND, len(scores))
        top_idx = np.argsort(scores)[::-1][:K]
        cand_ids = [retriever.corpus_ids[i] for i in top_idx]
        cand_passages = [corpus_texts.get(cid, '') for cid in cand_ids]

        # Stage 2: Qwen embedding rerank
        reranked = reranker.rerank(query_text, cand_passages, cand_ids, top_k=TOPK_RETURN)

        results = [{'paragraph_uuid': pid, 'score': float(s)} for pid, s in reranked]
        return results

    except Exception as e:
        print(f"Error in prediction: {e}")
        # Fallback: dense-only
        try:
            q_emb = retriever.embed_texts([query_text], batch_size=1, max_length=EMBED_MAX_LEN)
            scores = cosine_similarity(q_emb, retriever.corpus_embeddings)[0]
            idx = np.argsort(scores)[::-1][:TOPK_RETURN]
            return [{'paragraph_uuid': retriever.corpus_ids[i], 'score': float(scores[i])} for i in idx]
        except Exception:
            return []
