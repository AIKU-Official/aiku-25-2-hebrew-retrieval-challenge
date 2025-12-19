# model2.py
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 공통: 모델 경로 해결 유틸리티
# ---------------------------
def _resolve_model_dir(
    *,
    env_var_names: List[str],
    model_subdir_name: str,
    extra_candidates: List[str] | None = None,
    must_exist: bool = True,
) -> str:
    """
    로컬 모델 디렉토리를 다음 우선순위로 탐색:
      1) 환경변수 목록에서 첫 번째로 존재하는 경로
      2) __file__/models/<name>
      3) CWD/models/<name>
      4) 상위 디렉토리들에서 models/<name>, baseline_submission/models/<name>
      5) extra_candidates 목록
    존재하지 않으면 FileNotFoundError(어디를 찾았는지 로그 포함).
    """
    tried: List[Path] = []

    # 1) ENV 우선
    for var in env_var_names:
        val = os.getenv(var, "").strip()
        if val:
            p = Path(val).expanduser().resolve()
            if p.is_dir():
                return str(p)
            tried.append(p)

    # 2) __file__/models/<name>
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "models" / model_subdir_name,
        Path.cwd() / "models" / model_subdir_name,
    ]

    # 3) 상위 디렉토리 쭉 훑기
    for parent in here.parents:
        candidates.append(parent / "models" / model_subdir_name)
        candidates.append(parent / "baseline_submission" / "models" / model_subdir_name)

    # 4) 추가 후보
    if extra_candidates:
        for c in extra_candidates:
            candidates.append(Path(c).expanduser().resolve())

    for c in candidates:
        if c.is_dir():
            return str(c)
        tried.append(c)

    if must_exist:
        tried_str = "\n - ".join(str(x) for x in tried)
        raise FileNotFoundError(
            f"[MODEL PATH ERROR] '{model_subdir_name}' 폴더를 찾지 못함.\n"
            f"다음 경로들을 시도했음:\n - {tried_str}\n\n"
            "해결:\n"
            f"  - 환경변수 {env_var_names} 중 하나에 절대경로를 지정하거나,\n"
            "  - models/<name> 폴더 구조를 올바른 위치에 두세요 (압축 해제된 실제 파일들!).\n"
            "  - 폴더 안에 config.json, tokenizer 파일들, model.safetensors/pytorch_model.bin 이 있어야 합니다."
        )
    return ""


def _choose_dtype(device: str) -> torch.dtype:
    return torch.float16 if device.startswith("cuda") else torch.float32


def _enforce_local_only_kwargs(local_only: bool = True) -> dict:
    # 로컬 파일만 사용하도록 강제 (네트워크/로그인으로 튀는 것 방지)
    return {"local_files_only": bool(local_only)}


# ---------------------------
# E5 Retriever
# ---------------------------
class E5Retriever:
    def __init__(self, model_name: str | None = None, device: str | None = None, local_only: bool = True):
        """
        Multilingual E5 로컬 로딩 전용. 경로를 못 찾으면 명시적으로 에러.
        ENV 우선순위: E5_MODEL_DIR, E5_DIR, E5_PATH
        """
        if model_name is None:
            model_name = _resolve_model_dir(
                env_var_names=["E5_MODEL_DIR", "E5_DIR", "E5_PATH"],
                model_subdir_name="multilingual-e5-base",
            )
            print(f"[E5] Using local model dir: {model_name}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[E5] Loading on device: {self.device}")

        dtype = _choose_dtype(self.device)
        kwargs = _enforce_local_only_kwargs(local_only)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, **kwargs).to(self.device)
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.corpus_ids: List[str] = []
        self.corpus_embeddings: np.ndarray | None = None

    @torch.no_grad()
    def embed_texts(self, texts: List[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        # E5는 prefix 필수
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + (t or "").strip() for t in texts]

        all_chunks: List[torch.Tensor] = []
        total = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**enc)
            attn = enc["attention_mask"]
            hidden = out.last_hidden_state

            # mean pooling
            mask = attn.unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / denom

            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            all_chunks.append(pooled.cpu())

            del enc, out, attn, hidden, mask, summed, denom, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        embs = torch.cat(all_chunks, dim=0).numpy()
        return embs.astype(np.float32)


# ---------------------------
# BGE Reranker
# ---------------------------
class BGEReranker:
    def __init__(self, model_name: str | None = None, device: str | None = None, local_only: bool = True):
        """
        BGE reranker 로컬 로딩. ENV 우선순위: BGE_MODEL_DIR, BGE_DIR, BGE_PATH
        """
        from transformers import AutoModelForSequenceClassification

        if model_name is None:
            model_name = _resolve_model_dir(
                env_var_names=["BGE_MODEL_DIR", "BGE_DIR", "BGE_PATH"],
                model_subdir_name="bge-reranker-v2-m3",
            )
            print(f"[BGE] Using local model dir: {model_name}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BGE] Loading on device: {self.device}")

        dtype = _choose_dtype(self.device)
        kwargs = _enforce_local_only_kwargs(local_only)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True, **kwargs
        ).to(self.device)
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def rerank(self, query_text: str, passages: List[str], passage_ids: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        if not passages:
            return []

        scores: List[float] = []
        batch_size = 4
        qbatch = [query_text] * batch_size

        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i + batch_size]
            batch_queries = qbatch[:len(batch_passages)]

            enc = self.tokenizer(
                batch_queries,
                batch_passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**enc)
            logits = out.logits
            if logits.ndim == 1:
                bscores = logits.cpu().numpy()
            elif logits.shape[1] == 1:
                bscores = logits.squeeze(-1).cpu().numpy()
            else:
                bscores = logits[:, 1].cpu().numpy()

            scores.extend(bscores.tolist())

            del enc, out, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        paired = list(zip(passage_ids, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:top_k]


# ---------------------------
# 파이프라인 API
# ---------------------------
retriever: E5Retriever | None = None
reranker: BGEReranker | None = None
corpus_texts: Dict[str, str] = {}


def preprocess(corpus_dict: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    E5 임베딩 + BGE 준비
    """
    global retriever, reranker, corpus_texts

    print("=" * 60)
    print("PREPROCESSING: Initializing E5 + BGE Reranker Pipeline...")
    print("=" * 60)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("Loading E5 retriever...")
    retriever = E5Retriever(local_only=True)

    print("Loading BGE reranker...")
    reranker = BGEReranker(local_only=True)

    print(f"Preparing corpus with {len(corpus_dict)} documents...")

    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [doc.get("passage", doc.get("text", "")) for doc in corpus_dict.values()]
    corpus_texts = {doc_id: passages[i] for i, doc_id in enumerate(retriever.corpus_ids)}

    print("Computing E5 embeddings...")
    retriever.corpus_embeddings = retriever.embed_texts(passages, is_query=False, batch_size=32)

    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated embeddings for {len(retriever.corpus_ids)} documents")
    print(f"✓ Embedding matrix shape: {retriever.corpus_embeddings.shape}")

    return {
        "retriever": retriever,
        "reranker": reranker,
        "corpus_ids": retriever.corpus_ids,
        "corpus_embeddings": retriever.corpus_embeddings,
        "corpus_texts": corpus_texts,
        "num_documents": len(corpus_dict),
    }


def predict(query: Dict[str, str], preprocessed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    E5 1차 검색 → BGE 재정렬
    """
    global retriever, reranker, corpus_texts

    query_text = (query or {}).get("query", "").strip()
    if not query_text:
        return []

    if retriever is None or reranker is None:
        retriever = preprocessed_data.get("retriever")
        reranker = preprocessed_data.get("reranker")
        corpus_texts = preprocessed_data.get("corpus_texts", {})
        if retriever is None or reranker is None:
            print("[ERROR] retriever/reranker not found in preprocessed_data")
            return []

    try:
        print("Stage 1: E5 retrieval...")
        q_emb = retriever.embed_texts([query_text], is_query=True, batch_size=1)
        e5_scores = cosine_similarity(q_emb, retriever.corpus_embeddings)[0]

        top_k_candidates = min(100, len(e5_scores))
        top_idx = np.argsort(e5_scores)[::-1][:top_k_candidates]

        cand_ids = [retriever.corpus_ids[i] for i in top_idx]
        cand_texts = [corpus_texts.get(cid, "") for cid in cand_ids]

        print("Stage 2: BGE reranking...")
        reranked = reranker.rerank(query_text, cand_texts, cand_ids, top_k=20)

        results = [{"paragraph_uuid": pid, "score": float(score)} for pid, score in reranked]
        print(f"✓ Returned {len(results)} results with reranker scores")
        return results

    except Exception as e:
        print(f"[WARN] Rerank failed, fallback to E5-only: {e}")
        try:
            q_emb = retriever.embed_texts([query_text], is_query=True, batch_size=1)
            e5_scores = cosine_similarity(q_emb, retriever.corpus_embeddings)[0]
            top_idx = np.argsort(e5_scores)[::-1][:20]
            return [{"paragraph_uuid": retriever.corpus_ids[i], "score": float(e5_scores[i])} for i in top_idx]
        except Exception:
            return []
