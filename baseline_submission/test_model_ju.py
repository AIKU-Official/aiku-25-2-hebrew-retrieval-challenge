import os
import sys
import json
import time
import threading
from typing import Dict, Any, List, Tuple
from datetime import datetime

# ---- 즉시 출력 & 시끄러운 로그 차단 ----
os.environ.setdefault("PYTHONUNBUFFERED", "1")   # 표준출력 버퍼링 비활성화
os.environ.setdefault("TRANSFORMERS_NO_TF", "1") # TF 경로 차단(혼선/경고 감소)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ─────────────────────────────────────────────────────────────
# 20초마다 진행 상황을 찍는 하트비트 유틸
# ─────────────────────────────────────────────────────────────
class Heartbeat:
    """interval 초마다 '살아있다' 메시지 + 누적 시간(+선택적 상태) 출력"""
    def __init__(self, label: str, interval: float = 20.0, msg_fn=None):
        self.label = label
        self.interval = interval
        self.msg_fn = msg_fn  # 상태 문자열을 반환하는 콜백(optional)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self.start_time = time.time()

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        try:
            self._t.join(timeout=0.2)
        except RuntimeError:
            pass

    def _run(self):
        # 첫 출력은 interval 대기 없이 바로 찍고, 이후 interval 간격
        next_time = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now - self.start_time >= next_time:
                elapsed = now - self.start_time
                extra = ""
                if callable(self.msg_fn):
                    try:
                        extra = " | " + str(self.msg_fn())
                    except Exception:
                        extra = ""
                print(f"[hb] {self.label} running... elapsed={elapsed:.1f}s{extra}", flush=True)
                next_time += self.interval
            # 짧게 자자
            time.sleep(0.5)

# NumPy와 Scikit-learn 임포트
try:
    import numpy as np
    from sklearn.metrics import ndcg_score
except ImportError:
    print("Error: Required libraries (NumPy, Scikit-learn) not found.", file=sys.stderr, flush=True)
    sys.exit(1)

# --- 모델 임포트 설정 ---
# 1) 모델 파일 경로를 sys.path에 우선 등록
MODEL_DIR = "/app/project_code/Dataset and Baseline/baseline_submission/"
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

# 2) 모델 파일에서 preprocess 및 predict 함수 임포트
try:
    from juh_model import preprocess, predict
    print(f"Successfully imported preprocess and predict from juh_model.py in {MODEL_DIR}", flush=True)
except ImportError as e:
    print(f"Error: Failed to import juh_model: {e}", file=sys.stderr, flush=True)
    print(f"Check if '{MODEL_DIR}/juh_model.py' exists and is named correctly.", file=sys.stderr, flush=True)
    sys.exit(1)
# --- 임포트 설정 끝 ---


def load_data(
    corpus_file: str,
    train_file: str
) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]], Dict[int, Dict[str, float]]]:
    """
    hsrc_corpus.jsonl과 hsrc_train.jsonl 파일을 로드하여
    코퍼스, 쿼리 목록, 정답(QREL) 데이터를 생성합니다.
    """
    print("Loading corpus (hsrc_corpus.jsonl)...", flush=True)
    corpus_dict = {}
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                doc_id = entry.get('uuid')
                passage = entry.get('passage', entry.get('text', ''))
                if doc_id and passage:
                    corpus_dict[doc_id] = {'passage': passage, 'text': passage}
        print(f"✓ Loaded {len(corpus_dict)} documents into corpus_dict.", flush=True)
    except Exception as e:
        print(f"Error loading corpus file {corpus_file}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    print("Loading queries and QREL (hsrc_train.jsonl)...", flush=True)
    queries_list: List[Dict[str, str]] = []
    qrel: Dict[int, Dict[str, float]] = {}

    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                query_text = entry.get('query')
                if not query_text:
                    continue
                queries_list.append({'query': query_text})

                target_actions = entry.get('target_actions', {})
                current_qrel: Dict[str, float] = {}

                paragraphs = entry.get('paragraphs', {})
                # target_action_k  → paragraph_k  → paragraphs[paragraph_k]['uuid']
                for key, score in target_actions.items():
                    try:
                        paragraph_key = key.replace('target_action', 'paragraph')
                        doc_id = paragraphs.get(paragraph_key, {}).get('uuid')
                        if doc_id:
                            current_qrel[doc_id] = float(score)
                    except Exception as ee:
                        print(f"Error parsing QREL data for query {i}: {ee}", file=sys.stderr, flush=True)
                        continue

                if current_qrel:
                    qrel[i] = current_qrel

        print(f"✓ Loaded {len(queries_list)} queries and {len(qrel)} QREL entries.", flush=True)
    except Exception as e:
        print(f"Error loading train file {train_file}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    return corpus_dict, queries_list, qrel


def run_evaluation(data_dir: str, output_path: str):
    """
    데이터를 로드하고 모델을 평가하여 nDCG@20을 측정합니다.
    """

    CORPUS_FILE = os.path.join(data_dir, 'hsrc_corpus.jsonl')
    TRAIN_FILE  = os.path.join(data_dir, 'hsrc_train.jsonl')

    print("\n" + "="*80, flush=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING MODEL EVALUATION", flush=True)
    print(f"Using Corpus: {CORPUS_FILE}", flush=True)
    print(f"Using Train Data/QREL: {TRAIN_FILE}", flush=True)
    print("="*80 + "\n", flush=True)

    # 1) 데이터 로드
    corpus, queries, qrel = load_data(CORPUS_FILE, TRAIN_FILE)
    if not corpus or not queries or not qrel:
        print("[FATAL] Required data for evaluation is missing.", file=sys.stderr, flush=True)
        return

    # 2) Preprocess 실행 (하트비트로 20초마다 누적 시간 출력)
    try:
        print("\nRunning preprocess function...", flush=True)
        preprocess_start_time = time.time()

        pre_hb = Heartbeat(label="preprocess", interval=20.0)  # 퍼센트 모를 때도 '살아있다+경과시간' 보장
        pre_hb.start()
        try:
            preprocessed_data = preprocess(corpus)
        finally:
            pre_hb.stop()

        preprocess_end_time = time.time()
        print(f"Preprocessing completed in {(preprocess_end_time - preprocess_start_time):.2f} seconds.", flush=True)

        # 전체 문서 ID
        all_doc_ids = preprocessed_data.get('corpus_ids', list(corpus.keys()))
    except Exception as e:
        print(f"\n[FATAL] Preprocessing failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # 3) 예측 + nDCG 계산
    all_results: List[Dict[str, Any]] = []
    total_ndcg_score = 0.0
    evaluated_queries_count = 0

    print("\n" + "="*80, flush=True)
    print("STARTING PREDICTION AND NDCG CALCULATION", flush=True)
    print("="*80, flush=True)

    # 예측 하트비트: 처리 개수/누적 시간/평균 시간
    pred_stats = {"done": 0, "sum_ms": 0.0, "total": len(queries)}

    def pred_msg():
        done = pred_stats["done"]
        total = pred_stats["total"]
        sum_ms = pred_stats["sum_ms"]
        avg_ms = (sum_ms / max(1, done)) if done else 0.0
        pct = (100.0 * done / max(1, total))
        return f"pred {done}/{total} ({pct:.1f}%) | cum={sum_ms:.0f}ms | avg={avg_ms:.1f}ms/q"

    pred_hb = Heartbeat(label="predict", interval=20.0, msg_fn=pred_msg)
    pred_hb.start()

    try:
        for i, query in enumerate(queries):
            if i not in qrel:
                continue

            start_time = time.time()
            prediction_result = predict(query, preprocessed_data)
            end_time = time.time()

            # 3.1 y_score 구성
            y_score = np.zeros(len(all_doc_ids))
            for result_item in prediction_result:
                try:
                    doc_id = result_item['paragraph_uuid']
                    doc_idx = all_doc_ids.index(doc_id)
                    y_score[doc_idx] = result_item['score']
                except ValueError:
                    continue

            # 3.2 y_true 구성
            y_true = np.zeros(len(all_doc_ids))
            for doc_id, true_score in qrel[i].items():
                try:
                    doc_idx = all_doc_ids.index(doc_id)
                    y_true[doc_idx] = true_score
                except ValueError:
                    continue

            # 3.3 nDCG@20
            ndcg_20 = 0.0
            try:
                ndcg_20 = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=20)
                total_ndcg_score += ndcg_20
                evaluated_queries_count += 1
            except Exception as e_ndcg:
                print(f"Error calculating nDCG for query {i}: {e_ndcg}", file=sys.stderr, flush=True)

            # 3.4 결과 요약
            result_entry = {
                'query_index': i,
                'query': query['query'],
                'qrel_entries': len(qrel[i]),
                'results_count': len(prediction_result),
                'ndcg@20': float(ndcg_20),
                'time_ms': (end_time - start_time) * 1000,
                'top_results': [r['paragraph_uuid'] for r in prediction_result[:5]],
            }
            all_results.append(result_entry)

            print(
                f"Query {i+1}/{len(queries)} completed in {result_entry['time_ms']:.2f}ms. | NDCG@20: {ndcg_20:.4f}",
                flush=True
            )

            # 하트비트가 읽어갈 누적 통계 갱신
            pred_stats["done"] += 1
            pred_stats["sum_ms"] += result_entry['time_ms']
    finally:
        pred_hb.stop()

    # 4) 최종 결과
    final_avg_ndcg = total_ndcg_score / evaluated_queries_count if evaluated_queries_count > 0 else 0.0

    print("\n" + "="*80, flush=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] EVALUATION COMPLETE", flush=True)
    print(f"Total Queries: {len(queries)}", flush=True)
    print(f"Evaluated Queries: {evaluated_queries_count}", flush=True)
    print(f"FINAL AVERAGE nDCG@20: {final_avg_ndcg:.4f}", flush=True)
    print("="*80, flush=True)

    # 4.1 저장
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'total_queries': len(queries),
        'evaluated_queries': evaluated_queries_count,
        'average_ndcg@20': final_avg_ndcg,
        'individual_query_results': all_results
    }

    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_path, f"test_jeyun_007_{timestamp}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)
        print(f"Detailed results saved to: {output_filename}", flush=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to save results to file: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    DATA_DIR = os.path.join(
        '/app/project_code',
        'Dataset and Baseline',
        'baseline_submission',
        'hsrc'
    )

    TEST_OUTPUT_DIR = '/app/project_code/juhyeong/test_results'

    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {TEST_OUTPUT_DIR}", flush=True)

    run_evaluation(DATA_DIR, TEST_OUTPUT_DIR)
