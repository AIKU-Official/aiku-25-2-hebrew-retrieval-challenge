# -*- coding: utf-8 -*-
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 공통 유틸
# ============================================================
def _dtype_by_device(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32

def _first_dir(cands: List[str]) -> str | None:
    for c in cands:
        if c and Path(c).is_dir():
            return c
    return None

def _resolve_model_dir(
    *,
    env_vars: List[str],
    local_candidates: List[str],
    tag: str
) -> str:
    # 1) 환경변수 우선
    for var in env_vars:
        val = os.getenv(var, "").strip()
        if val and Path(val).is_dir():
            print(f"[{tag}] Using env path: {val}")
            return val
    # 2) 로컬 후보
    hit = _first_dir(local_candidates)
    if hit:
        print(f"[{tag}] Using local model: {hit}")
        return hit
    # 3) 실패 시 명시적 오류
    tried = "\n - ".join([*env_vars, *local_candidates])
    raise FileNotFoundError(
        f"[{tag}] 모델 폴더를 찾지 못했습니다. 다음 후보를 확인하세요:\n - {tried}\n"
        f"※ 최종 폴더( config.json / tokenizer.* / model.safetensors|pytorch_model.bin 이 바로 들어있는 폴더 )를 "
        f"{env_vars[0]} 환경변수로 지정하세요."
    )

# ============================================================
# DictaBERT Retriever (E5 대체)
# ============================================================
class DictaRetriever:
    """
    DictaBERT(Knesset-DictaBERT / dictabert-splinter 등)로 문장 임베딩 생성.
    E5처럼 'query:'/'passage:' prefix 불필요. mean-pooling + L2 normalize.
    """
    def __init__(self, model_name: str | None = None, device: str | None = None):
        if model_name is None:
            here = Path(os.path.dirname(os.path.abspath(__file__)))
            candidates = [
                str(here / "models" / "Knesset-DictaBERT"),
                str(here / "models" / "dictabert"),
                str(here / "models" / "dictabert-splinter"),
            ]
            model_name = _resolve_model_dir(
                env_vars=["DICTA_MODEL_DIR", "DICTABERT_DIR", "HEB_BERT_DIR"],
                local_candidates=candidates,
                tag="DICTA"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DICTA] Loading on device: {self.device}")

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
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.hidden), dtype=np.float32)

        chunks: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = [(t or "").strip() for t in texts[i:i + batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**enc)
            last = out.last_hidden_state          # [B, T, H]
            attn = enc["attention_mask"]          # [B, T]

            mask = attn.unsqueeze(-1).expand(last.size()).float()
            summed = (last * mask).sum(dim=1)     # [B, H]
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / denom               # mean pooling

            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.cpu())

            del enc, out, last, attn, mask, summed, denom, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        embs = torch.cat(chunks, dim=0).numpy().astype(np.float32)
        return embs

# ============================================================
# BGE Reranker (그대로, 안전 옵션 추가)
# ============================================================
class BGEReranker:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        # 로컬 우선
        if model_name is None:
            here = Path(os.path.dirname(os.path.abspath(__file__)))
            candidates = [str(here / "models" / "bge-reranker-v2-m3")]
            model_name = _resolve_model_dir(
                env_vars=["BGE_MODEL_DIR", "BGE_RERANKER_DIR"],
                local_candidates=candidates,
                tag="BGE"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BGE] Loading reranker on device: {self.device}")

        dtype = _dtype_by_device(self.device)
        from transformers import AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True
        ).to(self.device)
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def rerank(self, query_text: str, passages: List[str], passage_ids: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        if not passages:
            return []
        scores: List[float] = []
        bs = 4

        for i in range(0, len(passages), bs):
            batch_passages = passages[i:i + bs]
            batch_queries = [query_text] * len(batch_passages)

            inputs = self.tokenizer(
                batch_queries,
                batch_passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits
            if logits.ndim == 1:
                batch_scores = logits.cpu().numpy()
            elif logits.shape[1] == 1:
                batch_scores = logits.squeeze(-1).cpu().numpy()
            else:
                batch_scores = logits[:, 1].cpu().numpy()

            scores.extend(batch_scores.tolist())

            del inputs, outputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        paired = list(zip(passage_ids, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:top_k]

# ============================================================
# 파이프라인 API (preprocess / predict)
# ============================================================
retriever: DictaRetriever | None = None
reranker: BGEReranker | None = None
corpus_texts: Dict[str, str] = {}

def preprocess(corpus_dict: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    DictaBERT 임베딩 + BGE 재정렬 준비
    """
    global retriever, reranker, corpus_texts

    print("=" * 60)
    print("PREPROCESSING: Initializing DictaBERT + BGE Reranker Pipeline...")
    print("=" * 60)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print("Loading DictaBERT retriever...")
    retriever = DictaRetriever()

    print("Loading BGE reranker...")
    reranker = BGEReranker()

    print(f"Preparing corpus with {len(corpus_dict)} documents...")

    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]
    corpus_texts.clear()
    corpus_texts.update({doc_id: passages[i] for i, doc_id in enumerate(retriever.corpus_ids)})

    print("Computing DictaBERT embeddings...")
    retriever.corpus_embeddings = retriever.embed_texts(passages, batch_size=32)

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
    1) DictaBERT 코사인 유사도 top-K 후보
    2) BGE reranker로 최종 재정렬
    """
    global retriever, reranker, corpus_texts

    query_text = (query or {}).get('query', '').strip()
    if not query_text:
        return []

    if retriever is None or reranker is None:
        retriever = preprocessed_data.get('retriever')
        reranker = preprocessed_data.get('reranker')
        corpus_texts = preprocessed_data.get('corpus_texts', {})
        if retriever is None or reranker is None:
            print("Error: Missing retriever or reranker in preprocessed data")
            return []

    try:
        print("Stage 1: DictaBERT retrieval...")
        q_emb = retriever.embed_texts([query_text], batch_size=1)
        scores = cosine_similarity(q_emb, retriever.corpus_embeddings)[0]

        K = min(100, len(scores))
        top_idx = np.argsort(scores)[::-1][:K]
        cand_ids = [retriever.corpus_ids[i] for i in top_idx]
        cand_passages = [corpus_texts.get(cid, '') for cid in cand_ids]

        print("Stage 2: BGE reranking...")
        reranked = reranker.rerank(query_text, cand_passages, cand_ids, top_k=20)

        results = [{'paragraph_uuid': pid, 'score': float(s)} for pid, s in reranked]
        print(f"✓ Returned {len(results)} results with reranker scores")
        return results

    except Exception as e:
        print(f"Error in prediction: {e}")
        # Fallback: DictaBERT-only
        try:
            q_emb = retriever.embed_texts([query_text], batch_size=1)
            scores = cosine_similarity(q_emb, retriever.corpus_embeddings)[0]
            idx = np.argsort(scores)[::-1][:20]
            return [{'paragraph_uuid': retriever.corpus_ids[i], 'score': float(scores[i])} for i in idx]
        except Exception:
            return []
