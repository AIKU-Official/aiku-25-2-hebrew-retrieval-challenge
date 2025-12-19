# /app/project_code/taejin/model_taejin_04.py
# HeBERT retriever (mean-pool) + HeBERT MLM-based cross-encoder reranker (PLL scoring)
# Public API: preprocess(corpus_dict), predict(query, preprocessed_data)

from __future__ import annotations
import os, re
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,                    # for retriever (encoder-only)
    AutoModelForMaskedLM,         # for reranker (MLM head)
)

# ====================== Config ======================
HEBERT_LOCAL_CAND_PATHS = [
    "models/avichr-heBERT",
    "models/heBERT_local",
    "models/heBERT",
    "models/HeBERT",
    "Dataset and Baseline/baseline_submission/models/heBERT",
    "Dataset and Baseline/baseline_submission/models/HeBERT",
]
HEBERT_HF_FALLBACK     = os.getenv("HEBERT_ID", "avichr/heBERT")  # 제출시엔 호출되면 안 됨

EMBED_MAX_LEN          = int(os.getenv("EMBED_MAX_LEN", "512"))
EMBED_BS               = int(os.getenv("EMBED_BS", "32"))
USE_FP16               = os.getenv("USE_FP16", "1") != "0"

TOPK_CANDIDATES        = int(os.getenv("TOPK_CANDS", "120"))   # reranker 부담 고려
TOPK_RETURN            = int(os.getenv("TOPK_RETURN", "20"))

RERANK_MAX_QUERY_TOK   = int(os.getenv("RERANK_MAX_QTOK", "64"))  # PLL 비용 제한
RERANK_PAIR_BS         = int(os.getenv("RERANK_PAIR_BS", "16"))   # 한 번에 평가할 마스킹 시퀀스 배치
RERANK_MAX_LEN         = int(os.getenv("RERANK_MAX_LEN", "512"))

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

# ---------------- local path resolver ----------------
def _resolve_hebert_path() -> str:
    for p in HEBERT_LOCAL_CAND_PATHS:
        if os.path.isdir(p):
            return p
    return HEBERT_HF_FALLBACK  # dev fallback; 제출시에는 로컬 경로가 반드시 잡혀야 함.

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
            hidden = out.last_hidden_state                              # (B, L, H)
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # mean pool
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            outs.append(emb.detach().cpu())
            del enc, out, hidden, mask, emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0).numpy()

# ------------- HeBERT MLM-based reranker (PLL) ---------------
class HeBERTMLMReranker:
    """
    Cross-encoder reranker using HeBERT's MLM head.
    Score = average log-probability of query tokens when masked,
            given the passage context: [CLS] passage [SEP] query [SEP]
    Zero-shot, but much stronger than naive CLS-dot heuristics.
    """
    def __init__(self, model_id: str | None = None, device: str | None = None, use_fp16: bool = USE_FP16):
        model_id = model_id or _resolve_hebert_path()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
        self.tok = AutoTokenizer.from_pretrained(model_id)
        mlm = AutoModelForMaskedLM.from_pretrained(model_id)
        dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float32
        self.model = mlm.to(self.device, dtype=dtype).eval()

    @torch.no_grad()
    def _pll_score_batch(self, q_texts: List[str], p_texts: List[str]) -> List[float]:
        """
        Compute PLL scores for multiple (q,p) pairs.
        NOTE: Expensive. We cap query tokens per pair to RERANK_MAX_QUERY_TOK.
        """
        scores: List[float] = []
        for q, p in zip(q_texts, p_texts):
            # Normalize
            q = _prep_text(q)
            p = _prep_text(p)
            # Tokenize once to know segmentation
            enc = self.tok(p, q, padding=False, truncation=True,
                           max_length=RERANK_MAX_LEN, return_tensors="pt")
            # Figure indices range for query tokens
            # For BERT tokenizer, input_ids = [CLS] ...passage... [SEP] ...query... [SEP]
            input_ids = enc["input_ids"][0]
            token_type_ids = enc.get("token_type_ids", None)
            attention_mask = enc["attention_mask"][0]
            # Locate the first [SEP]
            sep_id = self.tok.sep_token_id
            cls_id = self.tok.cls_token_id
            # Find sep positions
            sep_positions = (input_ids == sep_id).nonzero().flatten().tolist()
            if len(sep_positions) < 1:
                # fallback: treat all as one segment
                p_end = (attention_mask == 1).nonzero().flatten()[-1].item()
            else:
                p_end = sep_positions[0]  # end of passage segment
            # query tokens start from p_end+1 until next [SEP] (or end-1)
            q_start = p_end + 1
            if len(sep_positions) >= 2:
                q_end = sep_positions[1]
            else:
                q_end = (attention_mask == 1).nonzero().flatten()[-1].item()
            # Clamp to max-qtok
            q_token_positions = list(range(q_start, min(q_end, q_start + RERANK_MAX_QUERY_TOK)))

            if len(q_token_positions) == 0:
                scores.append(float("-inf"))
                continue

            # Build masked variants
            masked_inputs = []
            labels_list = []
            for pos in q_token_positions:
                mi = {k: v.clone() for k, v in enc.items()}  # input_ids, attention_mask, token_type_ids?
                mi["input_ids"][0, pos] = self.tok.mask_token_id
                labels = -100 * torch.ones_like(mi["input_ids"])
                labels[0, pos] = input_ids[pos]  # supervise only the masked position
                masked_inputs.append(mi)
                labels_list.append(labels)

            # Run in mini-batches
            logprobs = []
            for i in range(0, len(masked_inputs), RERANK_PAIR_BS):
                batch = masked_inputs[i:i+RERANK_PAIR_BS]
                labels_b = labels_list[i:i+RERANK_PAIR_BS]
                # Stack into a batch
                keys = batch[0].keys()
                batch_stacked = {k: torch.cat([b[k] for b in batch], dim=0).to(self.device) for k in keys}
                labels_stacked = torch.cat(labels_b, dim=0).to(self.device)
                outputs = self.model(**batch_stacked)
                # Cross-entropy only on masked positions
                # logits: (B, L, V)
                logits = outputs.logits  # float
                log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
                # gather at true token ids where labels != -100
                mask = (labels_stacked != -100)
                true_ids = labels_stacked[mask]
                pred_logs = log_softmax[mask, true_ids]
                logprobs.append(pred_logs.detach().cpu())
                del outputs, logits, log_softmax, mask, true_ids, pred_logs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if logprobs:
                lp = torch.cat(logprobs).mean().item()  # average log-prob across masked query tokens
            else:
                lp = float("-inf")
            scores.append(lp)
            del enc, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores

    def rerank(self, query_text: str, passages: List[str], passage_ids: List[str], top_k: int = TOPK_RETURN) -> List[Tuple[str, float]]:
        if not passages:
            return []
        # PLL scores (higher is better)
        scores = self._pll_score_batch([query_text]*len(passages), passages)
        pairs = list(zip(passage_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

# ---------------- Globals (cache) --------------------
_retriever: HeBERTRetriever | None = None
_reranker:  HeBERTMLMReranker | None = None
_corpus_texts: Dict[str, str] = {}

# ================= Public API =======================
def preprocess(corpus_dict: Dict[str, Dict]) -> Dict:
    """
    corpus_dict: {doc_id: {'passage': str, 'text': str, ...}, ...}
    returns: dict with retriever/reranker objects and embeddings
    """
    global _retriever, _reranker, _corpus_texts
    print("="*60)
    print("PREPROCESSING: HeBERT retriever + HeBERT-MLM reranker (zero-shot PLL)")
    print("="*60)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    hebert_path = _resolve_hebert_path()
    print(f"[init] HeBERT from: {hebert_path}")
    _retriever = HeBERTRetriever(hebert_path)
    _reranker  = HeBERTMLMReranker(hebert_path)

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

    # Stage 1: dense retrieval via cosine (embeddings already L2-normalized)
    q_emb = retr.embed_texts([_prep_text(q)], is_query=True, batch_size=max(1, EMBED_BS // 2))
    scores = (q_emb @ retr.corpus_embeddings.T)[0]  # cosine (due to L2 norm)
    top_idx = np.argsort(scores)[::-1][:TOPK_CANDIDATES]
    cand_ids = [retr.corpus_ids[i] for i in top_idx]
    cand_txt = [_corpus_texts[i] for i in cand_ids]

    # Stage 2: HeBERT-MLM rerank (PLL)
    reranked = rerk.rerank(q, cand_txt, cand_ids, top_k=TOPK_RETURN)
    return [{"paragraph_uuid": pid, "score": float(s)} for pid, s in reranked]

# ---------------- Smoke test ------------------------
if __name__ == "__main__":
    corpus = {
        "p1": {"text": "שלום עולם זה טקסט דוגמה"},
        "p2": {"text": "הכנסת ישראל מקיימת דיונים חשובים"},
        "p3": {"text": "מודל חיפוש בעברית עם HeBERT ובורר MLM"},
    }
    q = {"query": "דיונים בכנסת ישראל"}
    pp = preprocess(corpus)
    res = predict(q, pp)
    print("Top results:", res[:3])
