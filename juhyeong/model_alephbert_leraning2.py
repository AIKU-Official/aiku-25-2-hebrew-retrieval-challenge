import os
import json
import math
import time
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from sklearn.metrics.pairwise import cosine_similarity

# 노이즈 로그 줄이기(토크나이저 포크 경고 방지)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =========================
# 설정
# =========================
DATA_DIR = "./hsrc"
TRAIN_PATH = os.path.join(DATA_DIR, "hsrc_train.jsonl")
CORPUS_PATH = os.path.join(DATA_DIR, "hsrc_corpus.jsonl")

OUT_DIR = "./outputs_dicta_ndcg2027"
os.makedirs(OUT_DIR, exist_ok=True)
SPLIT_DIR = os.path.join(OUT_DIR, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 120

# 길이 분리: 질의는 짧게, 문단은 256(기본). 필요시 조절.
Q_MAX_LEN = 96
P_MAX_LEN = 256  # ← 256/512 바꿔가며 실험

WARMUP_RATIO = 0.1
GRAD_ACCUM = 1
TEMPERATURE = 0.05

EARLYSTOP_PATIENCE = 3
IMPROVE_EPS = 1e-6

TOP_K_FOR_RERANK = 100
USE_RERANKER = False

# 학습 중 검증 관련(요청대로 3에폭마다 Full eval 실행)
EVAL_FULL_EVERY = 3
EVAL_DURING_TRAIN = True

# 학습 중엔 캐시를 쓰지 않고(allow_cache=False) 항상 최신 가중치로 임베딩 재계산
# 최종 평가에선 allow_cache=True로 저장/재사용 가능
EVAL_CACHE_DURING_TRAIN = False

# =========================
# 공통 유틸
# =========================
def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def device_dtype():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    return dev, dtype

def resolve_model_dir(envs: List[str], candidates: List[str], tag: str) -> str:
    for e in envs:
        val = os.getenv(e, "").strip()
        if val and Path(val).is_dir():
            print(f"[{tag}] Using env: {val}")
            return val
    for c in candidates:
        if Path(c).is_dir():
            print(f"[{tag}] Using local: {c}")
            return c
    tried = "\n - ".join([*envs, *candidates])
    raise FileNotFoundError(
        f"[{tag}] 모델 폴더를 못 찾음. 후보:\n - {tried}\n"
        f"최종 폴더( config.json/tokenizer.* & model.safetensors|pytorch_model.bin )를 {envs[0]}에 지정하세요."
    )

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom

def l2norm(x, dim=-1, eps=1e-9):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

def ndcg_at_k(rels: List[int], k: int = 20) -> float:
    rels = rels[:k]
    dcg = 0.0
    for i, rel in enumerate(rels, start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    rels_sorted = sorted(rels, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(rels_sorted, start=1):
        idcg += (2**rel - 1) / math.log2(i + 1)
    return 0.0 if idcg == 0 else dcg / idcg

def assert_file_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[DATA] 파일이 없습니다: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"[DATA] 파일이 비어 있습니다: {path}")

# =========================
# 입출력(유연 로더)
# =========================
def read_records_any(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[WARN] 파일 없음: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith('['):
            arr = json.load(f)
            if isinstance(arr, list):
                return arr
            raise ValueError("JSON 최상위가 리스트가 아닙니다.")
        recs = []
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"[DATA] JSONL 파싱 실패 ({path}:{ln}): {e}")
        return recs

# =========================
# HSRC 라벨 매핑
# =========================
def normalize_hsrc_label(v: Any) -> int:
    try:
        iv = int(v)
    except Exception:
        return 0
    if iv == 1:
        return 3
    if iv == 3:
        return 2
    return 0  # 0,4 등은 음성

# =========================
# 코퍼스 로더
# =========================
def load_corpus_flexible(path: str) -> Dict[str, Dict[str, Any]]:
    raw = read_records_any(path)
    if not raw:
        print("[CORPUS] 입력 레코드 0개")
        return {}

    id_keys = ["paragraph_id", "pid", "id", "para_id", "doc_id", "uuid"]
    text_keys = ["passage", "text", "paragraph", "content", "body", "context"]

    corpus: Dict[str, Dict[str, Any]] = {}
    missing_id = missing_text = 0

    for obj in raw:
        # HSRC 형태
        if isinstance(obj.get("paragraphs"), dict):
            for pkey, pobj in obj["paragraphs"].items():
                if not isinstance(pobj, dict):
                    continue
                pid = pobj.get("uuid") or pobj.get("id")
                txt = pobj.get("passage") or pobj.get("text") or pobj.get("content")
                if not pid:
                    pid = f"{obj.get('query_uuid','unknown')}_{pkey}"
                if not txt:
                    missing_text += 1
                    continue
                corpus[str(pid)] = {"text": str(txt), **pobj}
            continue

        # 평평한 레코드
        pid = next((str(obj[k]) for k in id_keys if k in obj and obj[k] not in (None, "")), None)
        txt = next((str(obj[k]) for k in text_keys if k in obj and obj[k] not in (None, "")), None)
        if pid is None:
            missing_id += 1
            continue
        if txt is None:
            missing_text += 1
            continue
        corpus[pid] = {"text": txt, **obj}

    print(f"[CORPUS] Loaded {len(corpus)} paragraphs. (missing_id={missing_id}, missing_text={missing_text})")
    if len(corpus) == 0:
        print("[HINT] HSRC 포맷이면 'paragraphs' 내부의 'uuid'/'passage' 키를 확인하세요.")
    return corpus

# =========================
# 쿼리 텍스트/ID 추출 + manifest 분할/저장/로드
# =========================
def get_query_text(entry: Dict[str, Any]) -> str:
    q_keys = ["query", "question", "text", "q", "title", "query_text", "question_text"]
    return next((entry[k].strip() for k in q_keys if isinstance(entry.get(k), str) and entry[k].strip()), "")

def get_query_id(entry: Dict[str, Any]) -> str:
    qid = entry.get("query_uuid") or entry.get("id") or entry.get("qid")
    if not qid:
        qtxt = get_query_text(entry)
        qid = hashlib.md5(qtxt.encode("utf-8")).hexdigest()
    return str(qid)

def save_manifest(split_ids: Dict[str, list], split_dir: str = SPLIT_DIR):
    os.makedirs(split_dir, exist_ok=True)
    for name, ids in split_ids.items():
        with open(os.path.join(split_dir, f"{name}_query_ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(ids))

def load_manifest(split_dir: str = SPLIT_DIR) -> Dict[str, set]:
    out = {}
    for name in ("train", "val", "test"):
        p = os.path.join(split_dir, f"{name}_query_ids.txt")
        if not os.path.exists(p):
            return {}
        with open(p, "r", encoding="utf-8") as f:
            out[name] = {line.strip() for line in f if line.strip()}
    return out

def split_and_persist(raw_entries: List[Dict[str, Any]], seed: int = SEED, split_dir: str = SPLIT_DIR) -> Dict[str, set]:
    # 결정적 정렬 후 시드 기반 셔플 → 70/15/15
    entries = sorted(raw_entries, key=lambda e: get_query_id(e))
    rng = random.Random(seed)
    idx = list(range(len(entries)))
    rng.shuffle(idx)
    n = len(entries)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    split_ids = {
        "train": [get_query_id(entries[i]) for i in train_idx],
        "val":   [get_query_id(entries[i]) for i in val_idx],
        "test":  [get_query_id(entries[i]) for i in test_idx],
    }
    save_manifest(split_ids, split_dir)
    return {k: set(v) for k, v in split_ids.items()}

def apply_manifest_split(raw_entries: List[Dict[str, Any]], manifest: Dict[str, set]) -> Tuple[list, list, list]:
    train, val, test = [], [], []
    for e in raw_entries:
        qid = get_query_id(e)
        if qid in manifest.get("train", set()):
            train.append(e)
        elif qid in manifest.get("val", set()):
            val.append(e)
        elif qid in manifest.get("test", set()):
            test.append(e)
    return train, val, test

def dump_split_jsonl_for_reranker(train_entries, val_entries, test_entries, corpus, out_dir=SPLIT_DIR):
    """
    리랭커 입력용 쿼리 JSONL 생성 (하드네거 없음).
    포맷: {"query_id": str, "query": str, "positives": [pid,...]}
    """
    os.makedirs(out_dir, exist_ok=True)

    def parse_training_entry_flexible(entry: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        q_keys = ["query", "question", "text", "q", "title", "query_text", "question_text"]
        q = next((entry[k].strip() for k in q_keys if isinstance(entry.get(k), str) and entry[k].strip()), "")
        relmap: Dict[str, int] = {}

        if isinstance(entry.get("paragraphs"), dict) and isinstance(entry.get("target_actions"), dict):
            key2pid: Dict[str, str] = {}
            for pkey, pobj in entry["paragraphs"].items():
                if not isinstance(pobj, dict):
                    continue
                pid = pobj.get("uuid") or pobj.get("id") or f"{entry.get('query_uuid','unknown')}_{pkey}"
                key2pid[str(pkey)] = str(pid)
            for ta_key, label in entry["target_actions"].items():
                idx = ta_key.split("_")[-1] if "_" in ta_key else ta_key
                pkey = f"paragraph_{idx}" if "_" in ta_key else ta_key
                pid = key2pid.get(pkey)
                if not pid:
                    continue
                relmap[pid] = normalize_hsrc_label(label)
            return q, relmap

        if isinstance(entry.get("annotations"), list):
            ann_id_keys = ["paragraph_id","pid","id","para_id","doc_id","uuid","passage_id","paragraphId","paraId","p_id"]
            for a in entry["annotations"]:
                pid = next((str(a[k]) for k in ann_id_keys if k in a and a[k] not in (None,"")), None)
                rel_raw = a.get("relevance", a.get("rel", 0))
                if pid:
                    try:
                        relmap[pid] = int(rel_raw)
                    except Exception:
                        relmap[pid] = 0
        elif isinstance(entry.get("relevances"), dict):
            for pid, rel_raw in entry["relevances"].items():
                try:
                    relmap[str(pid)] = int(rel_raw)
                except Exception:
                    relmap[str(pid)] = 0
        else:
            if isinstance(entry.get("positives"), (list, tuple)):
                for pid in entry["positives"]:
                    relmap[str(pid)] = max(3, relmap.get(str(pid), 3))
            if isinstance(entry.get("negatives"), (list, tuple)):
                for pid in entry["negatives"]:
                    relmap[str(pid)] = min(0, relmap.get(str(pid), 0))
            if isinstance(entry.get("positive_passages"), list):
                ann_id_keys = ["paragraph_id","pid","id","para_id","doc_id","uuid","passage_id"]
                for p in entry["positive_passages"]:
                    pid = next((str(p[k]) for k in ann_id_keys if k in p and p[k] not in (None,"")), None)
                    if pid:
                        relmap[pid] = max(3, relmap.get(pid, 3))
            if isinstance(entry.get("hard_negative_passages"), list):
                ann_id_keys = ["paragraph_id","pid","id","para_id","doc_id","uuid","passage_id"]
                for p in entry["hard_negative_passages"]:
                    pid = next((str(p[k]) for k in ann_id_keys if k in p and p[k] not in (None,"")), None)
                    if pid:
                        relmap[pid] = min(0, relmap.get(pid, 0))

        return q, relmap

    def to_items(entries):
        items = []
        for e in entries:
            q = get_query_text(e)
            qid = get_query_id(e)
            # 내부 파서로 positives 추출 (라벨>=2)
            qtxt, relmap = parse_training_entry_flexible(e)
            # qtxt가 비어있으면 get_query_text 사용
            if not qtxt:
                qtxt = q
            pos = [pid for pid, r in relmap.items() if r >= 2 and pid in corpus]
            if qtxt and pos:
                items.append({"query_id": qid, "query": qtxt, "positives": pos})
        return items

    for name, es in [("train", train_entries), ("val", val_entries), ("test", test_entries)]:
        path = os.path.join(out_dir, f"{name}_queries.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for obj in to_items(es):
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# =========================
# 학습용 파서 (리트리버용 (q, relmap) 리스트 생성)
# =========================
def parse_training_entry_flexible(entry: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    q_keys = ["query", "question", "text", "q", "title", "query_text", "question_text"]
    q = next((entry[k].strip() for k in q_keys if isinstance(entry.get(k), str) and entry[k].strip()), "")
    relmap: Dict[str, int] = {}

    if isinstance(entry.get("paragraphs"), dict) and isinstance(entry.get("target_actions"), dict):
        key2pid: Dict[str, str] = {}
        for pkey, pobj in entry["paragraphs"].items():
            if not isinstance(pobj, dict):
                continue
            pid = pobj.get("uuid") or pobj.get("id") or f"{entry.get('query_uuid','unknown')}_{pkey}"
            key2pid[str(pkey)] = str(pid)
        for ta_key, label in entry["target_actions"].items():
            idx = ta_key.split("_")[-1] if "_" in ta_key else ta_key
            pkey = f"paragraph_{idx}" if "_" in ta_key else ta_key
            pid = key2pid.get(pkey)
            if not pid:
                continue
            relmap[pid] = normalize_hsrc_label(label)
        return q, relmap

    if isinstance(entry.get("annotations"), list):
        ann_id_keys = ["paragraph_id","pid","id","para_id","doc_id","uuid","passage_id","paragraphId","paraId","p_id"]
        for a in entry["annotations"]:
            pid = next((str(a[k]) for k in ann_id_keys if k in a and a[k] not in (None,"")), None)
            rel_raw = a.get("relevance", a.get("rel", 0))
            if pid:
                try:
                    relmap[pid] = int(rel_raw)
                except Exception:
                    relmap[pid] = 0
    elif isinstance(entry.get("relevances"), dict):
        for pid, rel_raw in entry["relevances"].items():
            try:
                relmap[str(pid)] = int(rel_raw)
            except Exception:
                relmap[str(pid)] = 0
    else:
        if isinstance(entry.get("positives"), (list, tuple)):
            for pid in entry["positives"]:
                relmap[str(pid)] = max(3, relmap.get(str(pid), 3))
        if isinstance(entry.get("negatives"), (list, tuple)):
            for pid in entry["negatives"]:
                relmap[str(pid)] = min(0, relmap.get(str(pid), 0))
        if isinstance(entry.get("positive_passages"), list):
            ann_id_keys = ["paragraph_id","pid","id","para_id","doc_id","uuid","passage_id"]
            for p in entry["positive_passages"]:
                pid = next((str(p[k]) for k in ann_id_keys if k in p and p[k] not in (None,"")), None)
                if pid:
                    relmap[pid] = max(3, relmap.get(pid, 3))
        if isinstance(entry.get("hard_negative_passages"), list):
            ann_id_keys = ["paragraph_id","pid","id","para_id","doc_id","uuid","passage_id"]
            for p in entry["hard_negative_passages"]:
                pid = next((str(p[k]) for k in ann_id_keys if k in p and p[k] not in (None,"")), None)
                if pid:
                    relmap[pid] = min(0, relmap.get(pid, 0))

    return q, relmap

def to_q_rel_list(entries_raw: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, int]]]:
    out = []
    for e in entries_raw:
        q, relmap = parse_training_entry_flexible(e)
        if q and any(v > 0 for v in relmap.values()):
            out.append((q, relmap))
    return out

# =========================
# 데이터셋
# =========================
class PairDataset(Dataset):
    def __init__(self, q_rel_list: List[Tuple[str, Dict[str, int]]], corpus: Dict[str, Dict[str, Any]]):
        self.samples: List[Tuple[str, str]] = []
        for q, rels in q_rel_list:
            pos_ids = [pid for pid, r in rels.items() if r >= 3]
            if not pos_ids:
                pos_ids = [pid for pid, r in rels.items() if r == 2]
            for pid in pos_ids:
                text = corpus.get(pid, {}).get("text", "")
                if text:
                    self.samples.append((q, text))
        print(f"[DATASET] Built {len(self.samples)} (query, positive) pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

# =========================
# 인코더 (멀티 GPU 지원)
# =========================
class BiEncoder(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        base = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.hidden = getattr(base.config, "hidden_size", 768)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"[DP] Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.encoder = nn.DataParallel(base)
        else:
            self.encoder = base

    def encode_texts(self, texts: List[str], device: str, max_len: int = 256) -> torch.Tensor:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        ).to(device)
        out = self.encoder(**enc)
        last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = mean_pool(last_hidden, enc["attention_mask"])
        return l2norm(pooled)

# =========================
# 손실
# =========================
def info_nce_loss(q_emb: torch.Tensor, p_emb: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    logits = (q_emb @ p_emb.t()) / temperature
    labels = torch.arange(q_emb.size(0), device=q_emb.device)
    return nn.CrossEntropyLoss()(logits, labels)

# =========================
# (참고용) Reranker는 비활성화
# =========================
class BGEReranker:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Reranker disabled: USE_RERANKER=False")

# =========================
# 스케줄러
# =========================
def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    from transformers.optimization import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# =========================
# 학습 루프
# =========================
def train_loop(model: BiEncoder, train_ds: PairDataset, val_items: List[Tuple[str, Dict[str,int]]],
               corpus: Dict[str, Dict[str, Any]], device: str):
    if len(train_ds) == 0:
        print("[TRAIN] 학습 샘플 0개 → 학습을 건너뜁니다.")
        return

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / BATCH_SIZE))
    t_total = steps_per_epoch * EPOCHS // max(1, GRAD_ACCUM)
    warmup = int(t_total * WARMUP_RATIO)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup, t_total)

    def collate(batch):
        qs, ps = zip(*batch)
        return list(qs), list(ps)

    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate,
        drop_last=True
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    best_val = -1.0
    epochs_without_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        t0 = time.time()

        for step, (qs, ps) in enumerate(loader, start=1):
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                q_emb = model.encode_texts(qs, device, Q_MAX_LEN)
                p_emb = model.encode_texts(ps, device, P_MAX_LEN)
                loss = info_nce_loss(q_emb, p_emb, TEMPERATURE) / max(1, GRAD_ACCUM)

            scaler.scale(loss).backward()
            if step % max(1, GRAD_ACCUM) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += loss.item() * max(1, GRAD_ACCUM)

            if step % 50 == 0:
                print(f"[E{epoch}] step {step}/{steps_per_epoch} loss={running/step:.4f}")

        dur = time.time() - t0
        print(f"[E{epoch}] train_loss={running/max(1, step):.4f}  ({dur/60:.1f} min)")

        # ----- 3에폭마다 전체 평가 + 얼리스톱 -----
        if val_items and EVAL_DURING_TRAIN:
            do_full_eval = (epoch % EVAL_FULL_EVERY == 0) or (epoch == 1)
            if do_full_eval:
                val_ndcg = evaluate_ndcg(
                    model, val_items, corpus, device,
                    use_reranker=False,
                    allow_cache=EVAL_CACHE_DURING_TRAIN  # False 권장
                )
                print(f"[E{epoch}] [FULL EVAL] VAL nDCG@20 (dense only) = {val_ndcg:.4f}")

                improved = val_ndcg > best_val + IMPROVE_EPS
                if improved:
                    best_val = val_ndcg
                    epochs_without_improve = 0
                    save_path = os.path.join(OUT_DIR, "best_biencoder")
                    os.makedirs(save_path, exist_ok=True)
                    enc_to_save = model.encoder.module if isinstance(model.encoder, nn.DataParallel) else model.encoder
                    enc_to_save.save_pretrained(save_path)
                    model.tokenizer.save_pretrained(save_path)
                    with open(os.path.join(OUT_DIR, "best_val.txt"), "w") as f:
                        f.write(f"{best_val:.6f}\n")
                    print(f"[E{epoch}] ✓ Saved best model → {save_path}")
                else:
                    epochs_without_improve += 1
                    print(f"[E{epoch}] No improvement ({epochs_without_improve}/{EARLYSTOP_PATIENCE})")
                    if epochs_without_improve >= EARLYSTOP_PATIENCE:
                        print(f"[EARLY STOP] {EVAL_FULL_EVERY}에폭 주기의 풀 평가에서 {EARLYSTOP_PATIENCE}회 연속 개선 실패 → 중단")
                        break
            else:
                print(f"[E{epoch}] (검증 생략) 3에폭마다 전체 평가를 수행합니다.")
        elif not val_items:
            print("[VAL] 검증 세트가 없어 nDCG 평가/early stopping을 생략합니다.")

# =========================
# 임베딩/검색/평가
# =========================
def embed_corpus(model: BiEncoder, corpus: Dict[str, Dict[str, Any]], device: str,
                 cache_path: str | None) -> Tuple[np.ndarray, List[str]]:
    ids = list(corpus.keys())
    if len(ids) == 0:
        return np.zeros((0, getattr(model, "hidden", 768)), dtype=np.float32), []

    # 캐시 사용: allow_cache=True인 경우만 로드
    if cache_path and os.path.exists(cache_path) and os.path.exists(cache_path + ".ids"):
        print("[EMB] Loading cached corpus embeddings…")
        embs = np.load(cache_path)
        with open(cache_path + ".ids", "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f]
        return embs, ids

    print("[EMB] Computing corpus embeddings…")
    model.eval()
    texts = [corpus[i]["text"] for i in ids]
    embs = []
    bs = 256
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        with torch.no_grad():
            emb = model.encode_texts(batch, device, P_MAX_LEN).cpu().numpy()
        embs.append(emb)
        if (i // bs) % 20 == 0:
            print(f"  {i}/{len(texts)}")
    embs = np.vstack(embs).astype(np.float32)

    if cache_path:
        np.save(cache_path, embs)
        with open(cache_path + ".ids", "w", encoding="utf-8") as f:
            f.write("\n".join(ids))
    return embs, ids

def evaluate_ndcg(model: BiEncoder, items: List[Tuple[str, Dict[str,int]]],
                  corpus: Dict[str, Dict[str, Any]], device: str,
                  use_reranker: bool = False, bge_dir: str | None = None,
                  allow_cache: bool = True) -> float:
    if not items:
        return 0.0

    model_tag = Path(getattr(model, "model_dir", "unknown")).name
    cache = os.path.join(OUT_DIR, f"corpus_emb__{model_tag}__len{P_MAX_LEN}.npy") if allow_cache else None

    corpus_embs, corpus_ids = embed_corpus(model, corpus, device, cache)
    if corpus_embs.shape[0] == 0:
        print("[EVAL] 코퍼스 임베딩 0개 → nDCG=0.0 반환")
        return 0.0

    if use_reranker:
        raise RuntimeError("Reranker disabled. Set USE_RERANKER=True and implement training-time policy if needed.")

    ndcgs = []
    model.eval()
    for q, relmap in items:
        with torch.no_grad():
            q_emb = model.encode_texts([q], device, Q_MAX_LEN).cpu().numpy()
        sims = cosine_similarity(q_emb, corpus_embs)[0]
        top_idx = np.argsort(sims)[::-1][:20]
        top_pids = [corpus_ids[i] for i in top_idx]
        rels = [int(relmap.get(pid, 0)) for pid in top_pids[:20]]
        ndcgs.append(ndcg_at_k(rels, k=20))

    return float(np.mean(ndcgs)) if ndcgs else 0.0

# =========================
# 메인
# =========================
def main():
    set_seed(SEED)
    dev, _ = device_dtype()

    # 파일 존재 확인
    assert_file_exists(TRAIN_PATH)
    assert_file_exists(CORPUS_PATH)

    here = Path(__file__).parent
    aleph_dir = resolve_model_dir(
        envs=["ALEPHBERT_DIR", "ALEPHBERT_GIMMEL_DIR", "BASE_MODEL_DIR"],
        candidates=[str(here / "models" / "alephbert_gimmel_base_512")],
        tag="ALEPH"
    )

    # 데이터 로딩
    corpus = load_corpus_flexible(CORPUS_PATH)
    raw_entries = read_records_any(TRAIN_PATH)

    if len(corpus) == 0:
        raise ValueError("[DATA] 코퍼스가 0개입니다. HSRC 포맷의 'paragraphs' 안 'uuid'/'passage' 확인 필요.")
    if len(raw_entries) == 0:
        raise ValueError("[DATA] 학습 항목이 0개입니다. HSRC 'target_actions' 또는 표준 라벨 구조 확인.")

    # --- 고정 split 적용: manifest 로드 or 생성+저장 ---
    manifest = load_manifest(SPLIT_DIR)
    if not manifest:
        print("[SPLIT] manifest not found → creating and saving splits…")
        manifest = split_and_persist(raw_entries, seed=SEED, split_dir=SPLIT_DIR)
    else:
        print("[SPLIT] loaded existing manifest from splits/")

    # 동일 split으로 raw_entries 분할
    train_entries_raw, val_entries_raw, test_entries_raw = apply_manifest_split(raw_entries, manifest)
    print(f"[SPLIT] train={len(train_entries_raw)}  val={len(val_entries_raw)}  test={len(test_entries_raw)}")

    # 리랭커 입력용 JSONL 덤프 (하드네거 없이, positives만)
    dump_split_jsonl_for_reranker(train_entries_raw, val_entries_raw, test_entries_raw, corpus, out_dir=SPLIT_DIR)
    print(f"[RERANKER] dumped split jsonl to: {SPLIT_DIR}")

    # 리트리버 학습 입력 형식으로 변환
    train_items = to_q_rel_list(train_entries_raw)
    val_items   = to_q_rel_list(val_entries_raw)
    test_items  = to_q_rel_list(test_entries_raw)

    # 데이터셋/모델
    train_ds = PairDataset(train_items, corpus)
    model = BiEncoder(model_dir=aleph_dir)

    # 학습 (3에폭마다 전체 평가 + 얼리스톱)
    train_loop(model, train_ds, val_items, corpus, dev)

    # 베스트 체크포인트 로드(있으면)
    best_dir = os.path.join(OUT_DIR, "best_biencoder")
    if Path(best_dir).is_dir():
        print("[LOAD] Loading best checkpoint…")
        model = BiEncoder(model_dir=best_dir)

    # 최종 평가(전체 코퍼스, 캐시 사용 허용)
    if val_items:
        val_ndcg = evaluate_ndcg(model, val_items, corpus, dev, use_reranker=False, allow_cache=True)
        print(f"[FINAL] VAL nDCG@20 = {val_ndcg:.4f}")
    if test_items:
        test_ndcg = evaluate_ndcg(model, test_items, corpus, dev, use_reranker=False, allow_cache=True)
        print(f"[FINAL] TEST nDCG@20 = {test_ndcg:.4f}")

if __name__ == "__main__":
    main()
