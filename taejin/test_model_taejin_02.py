# /app/project_code/taejin/test_local_eval_ndcg.py
# Local harness to (1) call preprocess/predict, (2) compute nDCG@20,
# (3) write run.txt, predictions.json, scores.txt for sanity-check.

import os, sys, json, math, time
from typing import Dict, List, Tuple
from datetime import datetime

# === 선택: 어느 모델을 테스트할지에 맞게 import 경로/이름을 바꿔 ===
# from taejin.model_taejin_01 import preprocess, predict    # KD-BERT + BGE
# from taejin.model_taejin_02 import preprocess, predict    # mT5 encoder + BGE
from model_taejin_02 import preprocess, predict      # HeBERT + BGE

# ------------------ 작은 샘플 데이터셋 ------------------
def make_tiny_dataset() -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]], Dict[Tuple[str,str], int]]:
    """
    returns: (corpus_dict, queries, qrels)
      - corpus_dict: {doc_id: {"passage": "...", "text":"..."}}
      - queries: [{"qid":"Q1","query":"..."}, ...]
      - qrels: {(qid, doc_id): rel (0..4)}
    """
    corpus = {
        "d0": {"passage": "דן קם בבוקר בשעה שבע. הוא שתה קפה ונסע לעבודה באופניים.", "text": ""},
        "d1": {"passage": "בירושלים יש היסטוריה עתיקה. אנשים מבקרים בכותל המערבי.", "text": ""},
        "d2": {"passage": "הכנסת ישראל מקיימת דיונים ונאומים בנושאים חשובים.", "text": ""},
        "d3": {"passage": "מדעי המחשב עוסקים באלגוריתמים ומודלים של למידה חישובית.", "text": ""},
        "d4": {"passage": "שוק מחנה יהודה בירושלים מלא באורות וקולות בערב.", "text": ""},
    }
    queries = [
        {"qid": "Q1", "query": "באיזו שעה דן קם בבוקר?"},
        {"qid": "Q2", "query": "איפה אנשים מבקרים בירושלים?"},
        {"qid": "Q3", "query": "דיונים ונאומים בכנסת ישראל"},
    ]
    # 0..4 relevance (간단 예시)
    qrels = {
        ("Q1","d0"): 4, ("Q1","d2"): 0, ("Q1","d3"): 0, ("Q1","d1"): 0, ("Q1","d4"): 0,
        ("Q2","d1"): 4, ("Q2","d4"): 2, ("Q2","d0"): 0, ("Q2","d2"): 0, ("Q2","d3"): 0,
        ("Q3","d2"): 4, ("Q3","d1"): 1, ("Q3","d3"): 1, ("Q3","d0"): 0, ("Q3","d4"): 0,
    }
    return corpus, queries, qrels

# ------------------ nDCG 계산 ------------------
def dcg_at_k(gains: List[float], p: int) -> float:
    """gains: relevance grades already ordered by rank desc."""
    s = 0.0
    for i, g in enumerate(gains[:p], start=1):
        denom = math.log2(i + 1)
        s += (2**g - 1) / denom
    return s

def ndcg_at_k(pred: List[Tuple[str,float]], qrels: Dict[Tuple[str,str], int], qid: str, k: int = 20) -> float:
    # pred: list of (doc_id, score) sorted desc by score
    gains = [qrels.get((qid, doc_id), 0) for doc_id, _ in pred]
    dcg  = dcg_at_k(gains, k)
    # Ideal order
    all_truth = [rel for (qq, _), rel in qrels.items() if qq == qid]
    all_truth.sort(reverse=True)
    idcg = dcg_at_k(all_truth, k) if all_truth else 0.0
    return (dcg / idcg) if idcg > 0 else 0.0

# ------------------ 메인 러너 ------------------
def main(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    corpus, queries, qrels = make_tiny_dataset()

    print("\n" + "="*80)
    print("[Local Eval] START")
    print("="*80)

    # 1) preprocess
    t0 = time.time()
    pp = preprocess(corpus)
    print(f"[preprocess] done in {(time.time()-t0):.2f}s")

    # 파일들
    run_path   = os.path.join(out_dir, "run.txt")            # trec run
    pred_path  = os.path.join(out_dir, "predictions.json")   # json predictions
    score_path = os.path.join(out_dir, "scores.txt")         # single-line ndcg

    run_lines = []
    all_preds_json = []
    ndcgs = []

    # 2) predict per query
    for q in queries:
        qid = q["qid"]
        t1 = time.time()
        res = predict({"query": q["query"]}, pp)  # [{'paragraph_uuid', 'score'}, ...]
        dt = (time.time()-t1)*1000

        # keep top20 only
        res = sorted(res, key=lambda x: x["score"], reverse=True)[:20]

        # accumulate run lines (TREC format)
        for rank, item in enumerate(res, start=1):
            docid = item["paragraph_uuid"]
            score = item["score"]
            run_lines.append(f"{qid} Q0 {docid} {rank} {score:.6f} hebert_bge_local")

        # json predictions for inspection
        all_preds_json.append({
            "qid": qid,
            "query": q["query"],
            "results": res,
            "time_ms": dt
        })

        # compute ndcg@20
        nd = ndcg_at_k([(r["paragraph_uuid"], r["score"]) for r in res], qrels, qid, k=20)
        ndcgs.append(nd)
        print(f"[{qid}] nDCG@20={nd:.4f} | {len(res)} results | {dt:.1f} ms")

    # 3) write files
    with open(run_path, "w", encoding="utf-8") as f:
        f.write("\n".join(run_lines) + "\n")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(all_preds_json, f, ensure_ascii=False, indent=2)

    mean_ndcg = sum(ndcgs)/len(ndcgs) if ndcgs else 0.0
    with open(score_path, "w", encoding="utf-8") as f:
        f.write(f"ndcg@20 {mean_ndcg:.6f}\n")

    print("\n" + "-"*80)
    print(f"[Local Eval] mean nDCG@20 = {mean_ndcg:.4f}")
    print(f"run.txt        -> {run_path}")
    print(f"predictions    -> {pred_path}")
    print(f"scores.txt     -> {score_path}")
    print("-"*80)

if __name__ == "__main__":
    # 기본 출력 폴더
    OUT_DIR = "/app/project_code/taejin/test_results_local"
    main(OUT_DIR)
