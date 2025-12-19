# Retriever: HeBERT (Hebrew-specific, encoder-only, zero-shot dense)
# Reranker : BGE reranker (cross-encoder, zero-shot)
# Public API: preprocess(corpus_dict), predict(query, preprocessed_data)

from __future__ import annotations
import os, re
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# ====================== Config ======================
HEBERT_ID_DEFAULT     = os.getenv("HEBERT_ID", "avichr/heBERT")            # Apache-2.0
BGE_ID_DEFAULT        = os.getenv("BGE_ID", "BAAI/bge-reranker-v2-m3")     # Apache-2.0
EMBED_MAX_LEN         = int(os.getenv("EMBED_MAX_LEN", "512"))
EMBED_BS              = int(os.getenv("EMBED_BS", "32"))
RERANK_BS             = int(os.getenv("RERANK_BS", "4"))
TOPK_CANDIDATES       = int(os.getenv("TOPK_CANDS", "150"))
TOPK_RETURN           = int(os.getenv("TOPK_RETURN", "20"))
USE_FP16              = os.getenv("USE_FP16", "1") != "0"
# ====================================================

# -------- Hebrew normalization (niqqud/quotes/dash) --------
_HEBREW_NIQQUD = re.compile(r"[\u0591-\u05C7]")  # cantillation + niqqud
def normalize_he(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = _HEBREW_NIQQUD.sub("", t)
    t = t.replace("״", '"').replace("׳", "'").replace("־", "-")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _prep_text(s: str) -> str:
    return normalize_he(s)

# ---------------- local path resolvers ----------------
def _resolve_hebert_path() -> str:
    cands = [
        "models/heBERT",
        "models/HeBERT",
        "Dataset and Baseline/baseline_submission/models/heBERT",
        "Dataset and Baseline/baseline_submission/models/HeBERT",
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    return HEBERT_ID_DEFAULT  # dev 편의. 제출 시엔 로컬 폴더 포함 필수.

def _resolve_bge_path() -> str:
    cands = [
        "models/bge-reranker-v2-m3",
        "Dataset and Baseline/baseline_submission/models/bge-reranker-v2-m3",
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    return BGE_ID_DEFAULT     # dev 편의. 제출 시엔 로컬 폴더 포함 필수.

# ---------------- HeBERT retriever -------------------
class HeBERTRetriever:
    """Hebrew-specific dense retriever (HeBERT, mean-pool + L2)."""
    def __init__(self, model_id: str | None = None, device: str | None = None, use_fp16: bool = USE_FP16):
        model_id = model_id or _resolve_hebert_path()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        base = AutoModel.from_pretrained(model_id)
        dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float32
        self.model = base.to(self.device, dtype=dtype).eval()
        self.corpus_ids: List[str] = []
        self.corpus_embeddings: np.ndarray | None = None

    @torch.no_grad()
    def embed_texts(self, texts: List[str], is_query: bool = False, batch_size: int = EMBED_BS) -> np.ndarray:
        texts = [_prep_text(t) for t in texts]
        outs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i+batch_size], padding=True, truncation=True,
                           max_length=EMBED_MAX_LEN, return_tensors="pt").to(self.device)
            out = self.model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            outs.append(emb.detach().cpu())
            del enc, out, hidden, mask, emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0).numpy()

# ---------------- BGE reranker -----------------------
class BGEReranker:
    """BGE cross-encoder reranker (logit-based score)."""
    def __init__(self, model_id: str | None = None, device: str | None = None, use_fp16: bool = USE_FP16):
        model_id = model_id or _resolve_bge_path()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
        ).to(self.device, dtype=dtype).eval()

    @torch.no_grad()
    def rerank(self, query_text: str, passages: List[str], passage_ids: List[str], top_k: int = TOPK_RETURN) -> List[Tuple[str, float]]:
        if not passages:
            return []
        q = _prep_text(query_text)
        scores: List[float] = []
        bs = max(1, RERANK_BS)
        for i in range(0, len(passages), bs):
            batch_p = [_prep_text(p) for p in passages[i:i+bs]]
            batch_q = [q] * len(batch_p)
            inputs = self.tok(batch_q, batch_p, padding=True, truncation=True,
                              max_length=512, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            if logits.ndim == 1:
                s = logits
            elif logits.shape[1] == 1:
                s = logits.squeeze(-1)
            else:
                s = logits[:, 1]
            scores.extend(s.detach().cpu().numpy().tolist())
            del inputs, outputs, logits, s
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pairs = list(zip(passage_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

# ---------------- Globals (cache) --------------------
_retriever: HeBERTRetriever | None = None
_reranker:  BGEReranker | None = None
_corpus_texts: Dict[str, str] = {}

# ================= Public API =======================
def preprocess(corpus_dict: Dict[str, Dict]) -> Dict:
    """
    corpus_dict: {doc_id: {'passage': str, 'text': str, ...}, ...}
    returns: dict with retriever/reranker objects and embeddings
    """
    global _retriever, _reranker, _corpus_texts
    print("="*60)
    print("PREPROCESSING: HeBERT retriever (zero-shot) + BGE reranker")
    print("="*60)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    hebert_path = _resolve_hebert_path()
    bge_path = _resolve_bge_path()
    print(f"[init] HeBERT from: {hebert_path}")
    _retriever = HeBERTRetriever(hebert_path)
    print(f"[init] BGE reranker from: {bge_path}")
    _reranker  = BGEReranker(bge_path)

    ids = list(corpus_dict.keys())
    passages = [_prep_text(corpus_dict[i].get("passage", corpus_dict[i].get("text", ""))) for i in ids]
    _corpus_texts = {ids[i]: passages[i] for i in range(len(ids))}

    print(f"[embed] encoding {len(ids)} passages...")
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
    """
    query: {'query': '...'}
    preprocessed_data: output of preprocess()
    returns: [{'paragraph_uuid': doc_id, 'score': float}, ...] (desc)
    """
    global _retriever, _reranker, _corpus_texts
    q = query.get("query", "")
    if not q:
        return []

    retr = _retriever or preprocessed_data.get("retriever")
    rerk = _reranker  or preprocessed_data.get("reranker")
    _corpus_texts = _corpus_texts or preprocessed_data.get("corpus_texts", {})
    if retr is None or rerk is None:
        print("Error: retriever/reranker not initialized")
        return []

    q_emb = retr.embed_texts([_prep_text(q)], is_query=True, batch_size=max(1, EMBED_BS // 2))
    scores = (q_emb @ retr.corpus_embeddings.T)[0]  # cosine (L2-normalized)
    top_idx = np.argsort(scores)[::-1][:TOPK_CANDIDATES]
    cand_ids = [retr.corpus_ids[i] for i in top_idx]
    cand_txt = [_corpus_texts[i] for i in cand_ids]

    reranked = rerk.rerank(q, cand_txt, cand_ids, top_k=TOPK_RETURN)
    return [{"paragraph_uuid": pid, "score": float(s)} for pid, s in reranked]

# ---------------- Smoke test ------------------------
if __name__ == "__main__":
    corpus = {
        "p1": {"text": "שלום עולם זה טקסט דוגמה"},
        "p2": {"text": "הכנסת ישראל מקיימת דיונים חשובים"},
        "p3": {"text": "מודל חיפוש בעברית עם HeBERT ובורר BGE"},
    }
    q = {"query": "דיונים בכנסת ישראל"}
    pp = preprocess(corpus)
    res = predict(q, pp)
    print("Top results:", res[:3])
