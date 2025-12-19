# file: train_reranker_bge_v2m3.py
import os, json, math, time, random, hashlib
import re, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics.pairwise import cosine_similarity

# 환경
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ──────────────────────────────────────────────────────────────────────────────
# 히브리어 정규화 (리트리버와 완전히 동일)
# ──────────────────────────────────────────────────────────────────────────────
_NIQQUD = r'[\u0591-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C7]'
_BIDI   = r'[\u200E\u200F\u202A-\u202E]'

def normalize_he(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = (t.replace('ם','מ').replace('ן','נ').replace('ץ','צ')
           .replace('ף','פ').replace('ך','כ'))
    t = re.sub(_NIQQUD, '', t)
    t = re.sub(_BIDI, '', t)
    return t.strip()

# ──────────────────────────────────────────────────────────────────────────────
# 경로/설정
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./hsrc"
TRAIN_PATH = os.path.join(DATA_DIR, "hsrc_train.jsonl")
CORPUS_PATH = os.path.join(DATA_DIR, "hsrc_corpus.jsonl")

OUT_DIR = "./outputs_dicta_ndcg2030"
SPLIT_DIR = os.path.join(OUT_DIR, "splits")
RERANK_OUT = os.path.join(OUT_DIR, "reranker_bge_v2_m3")
os.makedirs(RERANK_OUT, exist_ok=True)

MODELS_DIR = "./models"
BGE_DIR = os.path.join(MODELS_DIR, "bge-reranker-v2-m3")

RETRIEVER_DIR_CANDIDATES = [
    os.path.join(OUT_DIR, "best_biencoder"),
    os.path.join(MODELS_DIR, "alephbert_gimmel_base_512"),
]

# 하이퍼파라미터
SEED = 42
BATCH_SIZE = 2
NUM_NEG = 4
GRAD_ACCUM = 4
EPOCHS = 30
LR = 2e-5
WARMUP_RATIO = 0.1
EARLYSTOP_PATIENCE = 3
IMPROVE_EPS = 1e-6

Q_MAX_LEN = 96
P_MAX_LEN = 256
CE_MAX_LEN = 384

TOPK_MINING = 50
EVAL_K = 20
EVAL_FULL_EVERY = 3
EVAL_DURING_TRAIN = True

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def device_dtype():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    return dev, dtype

# ──────────────────────────────────────────────────────────────────────────────
# IO 유틸
# ──────────────────────────────────────────────────────────────────────────────
def assert_file_exists(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"필수 파일 없음/빈 파일: {path}")

def read_records_any(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048); f.seek(0)
        if head.lstrip().startswith("["):
            arr = json.load(f)
            if isinstance(arr, list): return arr
            raise ValueError("JSON 최상위가 리스트가 아님")
        recs=[]
        for i,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            recs.append(json.loads(line))
        return recs

def get_query_text(entry: Dict[str, Any]) -> str:
    for k in ["query","question","text","q","title","query_text","question_text"]:
        v = entry.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
    return ""

def get_query_id(entry: Dict[str, Any]) -> str:
    qid = entry.get("query_uuid") or entry.get("id") or entry.get("qid")
    if not qid:
        qtxt = get_query_text(entry)
        qid = hashlib.md5(qtxt.encode("utf-8")).hexdigest()
    return str(qid)

def normalize_hsrc_label(v: Any) -> int:
    try: iv = int(v)
    except Exception: return 0
    if iv == 1: return 3
    if iv == 3: return 2
    return 0

def load_corpus_flexible(path: str) -> Dict[str, Dict[str, Any]]:
    raw = read_records_any(path)
    id_keys = ["paragraph_id","pid","id","para_id","doc_id","uuid"]
    text_keys = ["passage","text","paragraph","content","body","context"]
    corpus={}
    for obj in raw:
        if isinstance(obj.get("paragraphs"), dict):
            for pkey,pobj in obj["paragraphs"].items():
                if not isinstance(pobj, dict): continue
                pid = pobj.get("uuid") or pobj.get("id") or f"{obj.get('query_uuid','u')}_{pkey}"
                txt = pobj.get("passage") or pobj.get("text") or pobj.get("content")
                if not pid or not txt: continue
                corpus[str(pid)] = {"text": str(txt), **pobj}
            continue
        pid = next((str(obj[k]) for k in id_keys if k in obj and obj[k] not in (None,"")), None)
        txt = next((str(obj[k]) for k in text_keys if k in obj and obj[k] not in (None,"")), None)
        if pid and txt: corpus[pid] = {"text": str(txt), **obj}
    return corpus

def parse_entry_to_relmap(entry: Dict[str,Any]) -> Tuple[str, Dict[str,int]]:
    q = get_query_text(entry)
    relmap={}
    if isinstance(entry.get("paragraphs"), dict) and isinstance(entry.get("target_actions"), dict):
        key2pid={}
        for pkey, pobj in entry["paragraphs"].items():
            if not isinstance(pobj, dict): continue
            pid = pobj.get("uuid") or pobj.get("id") or f"{entry.get('query_uuid','u')}_{pkey}"
            key2pid[str(pkey)] = str(pid)
        for ta_key, label in entry["target_actions"].items():
            idx = ta_key.split("_")[-1] if "_" in ta_key else ta_key
            pkey = f"paragraph_{idx}" if "_" in ta_key else ta_key
            pid = key2pid.get(pkey)
            if pid: relmap[pid] = normalize_hsrc_label(label)
        return q, relmap

    if isinstance(entry.get("annotations"), list):
        ann_id_keys = ["paragraph_id","pid","id","para_id","doc_id","uuid","passage_id","paragraphId","paraId","p_id"]
        for a in entry["annotations"]:
            pid = next((str(a[k]) for k in ann_id_keys if k in a and a[k] not in (None,"")), None)
            rel_raw = a.get("relevance", a.get("rel", 0))
            if pid:
                try: relmap[pid] = int(rel_raw)
                except: relmap[pid] = 0
    elif isinstance(entry.get("relevances"), dict):
        for pid, rel_raw in entry["relevances"].items():
            try: relmap[str(pid)] = int(rel_raw)
            except: relmap[str(pid)] = 0
    else:
        if isinstance(entry.get("positives"), (list,tuple)):
            for pid in entry["positives"]: relmap[str(pid)] = 3
        if isinstance(entry.get("hard_negative_passages"), list):
            ann_id_keys = ["paragraph_id","pid","id","para_id","doc_id","uuid","passage_id"]
            for p in entry["hard_negative_passages"]:
                pid = next((str(p[k]) for k in ann_id_keys if k in p and p[k] not in (None,"")), None)
                if pid: relmap[pid] = min(0, relmap.get(pid,0))
    return q, relmap

def load_manifest_ids(split_dir: str) -> Dict[str,set]:
    out={}
    for name in ("train","val","test"):
        p=os.path.join(split_dir, f"{name}_query_ids.txt")
        if not os.path.exists(p): return {}
        with open(p,"r",encoding="utf-8") as f:
            out[name] = {line.strip() for line in f if line.strip()}
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Retriever (bi-encoder) – 임베딩
# ──────────────────────────────────────────────────────────────────────────────
class BiEncoderForEmbed(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.dir=model_dir
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.base= AutoModel.from_pretrained(model_dir, local_files_only=True)
    def encode(self, texts: List[str], device, max_len=256):
        texts = [normalize_he(x) for x in texts]
        with torch.no_grad():
            enc=self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out=self.base(**enc).last_hidden_state
            mask=enc["attention_mask"].unsqueeze(-1).expand(out.size()).float()
            pooled=(out*mask).sum(1)/torch.clamp(mask.sum(1), min=1e-9)
            pooled = pooled/ (pooled.norm(p=2, dim=-1, keepdim=True)+1e-9)
        return pooled

def pick_retriever_dir() -> str:
    for d in RETRIEVER_DIR_CANDIDATES:
        if Path(d).is_dir():
            print(f"[RETR] using: {d}")
            return d
    raise FileNotFoundError("리트리버 모델 폴더를 찾지 못함")

# ──────────────────────────────────────────────────────────────────────────────
# Cross-Encoder (BGE reranker) – 입력 직전 정규화
# ──────────────────────────────────────────────────────────────────────────────
class CrossEncoder(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.dir=model_dir
        self.tok=AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        )
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        torch.backends.cuda.matmul.allow_tf32 = True

    def score_pairs(self, queries: List[str], passages: List[str], device, max_len=512):
        queries  = [normalize_he(q) for q in queries]
        passages = [normalize_he(p) for p in passages]
        enc=self.tok(queries, passages, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            logits=self.model(**enc).logits.squeeze(-1)
        return logits

    def forward_pairs(self, queries: List[str], passages: List[str], device, max_len=512):
        queries  = [normalize_he(q) for q in queries]
        passages = [normalize_he(p) for p in passages]
        enc=self.tok(queries, passages, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out=self.model(**enc).logits.squeeze(-1)
        return out

# ──────────────────────────────────────────────────────────────────────────────
# nDCG
# ──────────────────────────────────────────────────────────────────────────────
def ndcg_at_k(rels: List[int], k:int=20) -> float:
    rels = rels[:k]
    dcg=0.0
    for i,rel in enumerate(rels,1):
        dcg += (2**rel - 1)/math.log2(i+1)
    idcg=0.0
    for i,rel in enumerate(sorted(rels, reverse=True),1):
        idcg += (2**rel - 1)/math.log2(i+1)
    return 0.0 if idcg==0 else dcg/idcg

# ──────────────────────────────────────────────────────────────────────────────
# 마이닝 (retriever top-k)
# ──────────────────────────────────────────────────────────────────────────────
def precompute_corpus_embs(retriever: BiEncoderForEmbed, corpus: Dict[str,Dict[str,Any]], device, max_len=P_MAX_LEN):
    ids=list(corpus.keys())
    texts=[corpus[i]["text"] for i in ids]
    embs=[]
    bs=256
    for i in range(0,len(texts),bs):
        embs.append(retriever.encode(texts[i:i+bs], device, max_len).cpu().numpy())
    embs=np.vstack(embs).astype(np.float32)
    return embs, ids

def mine_candidates_for_query(qtext: str, retriever: BiEncoderForEmbed, corpus_embs: np.ndarray, corpus_ids: List[str], device, topk=TOPK_MINING):
    qtext = normalize_he(qtext)
    q_emb = retriever.encode([qtext], device, max_len=Q_MAX_LEN).cpu().numpy()
    sims = cosine_similarity(q_emb, corpus_embs)[0]
    idx = np.argsort(sims)[::-1][:topk]
    return [corpus_ids[i] for i in idx]

# ──────────────────────────────────────────────────────────────────────────────
# 데이터셋 (pos + N negs) – 정규화 일관 적용
# ──────────────────────────────────────────────────────────────────────────────
class RerankerTrainSet(Dataset):
    def __init__(self, 
                 train_entries: List[Dict[str,Any]],
                 corpus: Dict[str,Dict[str,Any]],
                 retriever: BiEncoderForEmbed,
                 corpus_embs: np.ndarray,
                 corpus_ids: List[str],
                 device):
        self.items=[]
        for e in train_entries:
            q = get_query_text(e)
            if not q: continue
            _, relmap = parse_entry_to_relmap(e)
            pos = [pid for pid, r in relmap.items() if r>=2 and pid in corpus]
            if not pos: continue

            cand = mine_candidates_for_query(q, retriever, corpus_embs, corpus_ids, device, topk=TOPK_MINING)
            neg_pool = [pid for pid in cand if pid not in pos]
            if not neg_pool: continue

            pos_pid = random.choice(pos)
            if len(neg_pool) >= NUM_NEG:
                negs = random.sample(neg_pool, NUM_NEG)
            else:
                negs = neg_pool + random.choices(neg_pool, k=NUM_NEG-len(neg_pool))
            self.items.append((q, pos_pid, negs))
        print(f"[TRAINSET] {len(self.items)} instances (each = 1 pos + {NUM_NEG} negs)")

        self.corpus = corpus

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        q, pos_pid, negs = self.items[i]
        q = normalize_he(q)
        pos_text = normalize_he(self.corpus[pos_pid]["text"])
        neg_texts = [normalize_he(self.corpus[n]["text"]) for n in negs]
        return q, pos_text, neg_texts

def collate_rerank(batch):
    queries=[]; passages=[]; group_slices=[]
    for q, pos, negs in batch:
        start=len(passages)
        queries.append(q); passages.append(pos)  # label index = 0
        for ng in negs:
            queries.append(q); passages.append(ng)
        end=len(passages)
        group_slices.append((start, end))
    return queries, passages, group_slices

def group_softmax_ce(scores: torch.Tensor, group_slices: List[Tuple[int,int]]) -> torch.Tensor:
    losses=[]
    for (s,e) in group_slices:
        g = scores[s:e]
        logp = g - torch.logsumexp(g, dim=0)
        losses.append(-logp[0])
    return torch.stack(losses).mean()

# ──────────────────────────────────────────────────────────────────────────────
# 평가: retriever→rerank→nDCG@K
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_ndcg_cross(
    cross: CrossEncoder,
    retriever: BiEncoderForEmbed,
    items: List[Dict[str,Any]],
    corpus: Dict[str,Dict[str,Any]],
    corpus_embs: np.ndarray,
    corpus_ids: List[str],
    device,
    k=EVAL_K
) -> float:
    ndcgs=[]
    cross.eval()
    for e in items:
        qtxt = normalize_he(get_query_text(e))
        if not qtxt: continue
        _, relmap = parse_entry_to_relmap(e)

        cand = mine_candidates_for_query(qtxt, retriever, corpus_embs, corpus_ids, device, topk=max(100,k))
        cand_texts = [normalize_he(corpus[pid]["text"]) for pid in cand]
        scores = []
        bs=32
        for i in range(0, len(cand), bs):
            qs = [qtxt]*min(bs, len(cand)-i)
            ps = cand_texts[i:i+bs]
            s = cross.score_pairs(qs, ps, device, max_len=CE_MAX_LEN).cpu().tolist()
            scores.extend(s)
        order = np.argsort(np.array(scores))[::-1][:k]
        top_pids = [cand[i] for i in order]
        rels = [int(relmap.get(pid, 0)) for pid in top_pids]
        ndcgs.append(ndcg_at_k(rels, k=k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────────────────────────────────────
def main():
    set_seed()
    dev, _ = device_dtype()

    assert_file_exists(TRAIN_PATH)
    assert_file_exists(CORPUS_PATH)
    if not Path(BGE_DIR).is_dir():
        raise FileNotFoundError(f"bge-reranker-v2-m3 폴더가 없음: {BGE_DIR}")

    corpus = load_corpus_flexible(CORPUS_PATH)
    all_entries = read_records_any(TRAIN_PATH)
    manifest = load_manifest_ids(SPLIT_DIR)
    if not manifest:
        raise RuntimeError(f"split manifest가 없습니다: {SPLIT_DIR}")

    def filter_split(name):
        ids = manifest[name]
        return [e for e in all_entries if get_query_id(e) in ids]
    train_entries = filter_split("train")
    val_entries   = filter_split("val")
    test_entries  = filter_split("test")

    print(f"[SPLIT] train={len(train_entries)}  val={len(val_entries)}  test={len(test_entries)}")

    retr_dir = pick_retriever_dir()
    retr = BiEncoderForEmbed(retr_dir).to(dev)
    corpus_embs, corpus_ids = precompute_corpus_embs(retr, corpus, dev, max_len=P_MAX_LEN)
    print(f"[EMB] corpus_embs: {corpus_embs.shape}")

    cross = CrossEncoder(BGE_DIR).to(dev)

    train_set = RerankerTrainSet(train_entries, corpus, retr, corpus_embs, corpus_ids, dev)
    loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_rerank,
        drop_last=True
    )

    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * EPOCHS // max(1, GRAD_ACCUM)
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(cross.parameters(), lr=LR)
    from transformers.optimization import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=(dev=="cuda"))

    best_val = -1.0
    epochs_wo_imp = 0

    for epoch in range(1, EPOCHS+1):
        torch.cuda.empty_cache()
        cross.train()
        running=0.0
        t0=time.time()
        optimizer.zero_grad(set_to_none=True)
        micro_count = 0

        for step, (qs, ps, groups) in enumerate(loader, 1):
            for (s,e) in groups:
                with torch.amp.autocast('cuda', enabled=(dev=="cuda")):
                    _qs = qs[s:e]
                    _ps = ps[s:e]
                    scores = cross.forward_pairs(_qs, _ps, dev, max_len=CE_MAX_LEN)
                    loss = group_softmax_ce(scores, [(0, len(_qs))]) / max(1, GRAD_ACCUM)

                scaler.scale(loss).backward()
                running += loss.item() * max(1, GRAD_ACCUM)
                micro_count += 1

                if micro_count % GRAD_ACCUM == 0:
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            if step % 50 == 0:
                print(f"[E{epoch}] step {step}/{steps_per_epoch} loss={running/step:.4f}")

        print(f"[E{epoch}] train_loss={running/max(1,steps_per_epoch):.4f}  ({(time.time()-t0)/60:.1f} min)")

        if EVAL_DURING_TRAIN and ((epoch % EVAL_FULL_EVERY == 0) or epoch==1):
            val_ndcg = evaluate_ndcg_cross(cross, retr, val_entries, corpus, corpus_embs, corpus_ids, dev, k=EVAL_K)
            print(f"[E{epoch}] [FULL EVAL] VAL nDCG@{EVAL_K} (retriever→rerank) = {val_ndcg:.4f}")

            improved = val_ndcg > best_val + IMPROVE_EPS
            if improved:
                best_val = val_ndcg
                epochs_wo_imp = 0
                save_dir = os.path.join(RERANK_OUT, "best")
                cross.model.save_pretrained(save_dir)
                cross.tok.save_pretrained(save_dir)
                with open(os.path.join(RERANK_OUT, "best_val.txt"), "w") as f:
                    f.write(f"{best_val:.6f}\n")
                print(f"[E{epoch}] ✓ saved best reranker → {save_dir}")
            else:
                epochs_wo_imp += 1
                print(f"[E{epoch}] No improvement ({epochs_wo_imp}/{EARLYSTOP_PATIENCE})")
                if epochs_wo_imp >= EARLYSTOP_PATIENCE:
                    print("[EARLY STOP] 연속 개선 실패로 중단")
                    break
        else:
            print(f"[E{epoch}] (검증 생략) {EVAL_FULL_EVERY}에폭마다 평가")

    best_dir = os.path.join(RERANK_OUT, "best")
    if Path(best_dir).is_dir():
        print("[LOAD] best reranker ckpt")
        cross = CrossEncoder(best_dir).to(dev)

    if val_entries:
        val_ndcg = evaluate_ndcg_cross(cross, retr, val_entries, corpus, corpus_embs, corpus_ids, dev, k=EVAL_K)
        print(f"[FINAL] VAL nDCG@{EVAL_K} = {val_ndcg:.4f}")
    if test_entries:
        test_ndcg = evaluate_ndcg_cross(cross, retr, test_entries, corpus, corpus_embs, corpus_ids, dev, k=EVAL_K)
        print(f"[FINAL] TEST nDCG@{EVAL_K} = {test_ndcg:.4f}")

if __name__=="__main__":
    main()
