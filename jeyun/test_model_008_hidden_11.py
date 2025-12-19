import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime

# NumPy와 Scikit-learn 임포트
try:
    import numpy as np
    from sklearn.metrics import ndcg_score
except ImportError:
    print("Error: Required libraries (NumPy, Scikit-learn) not found.", file=sys.stderr)
    sys.exit(1)

# --- 모델 임포트 설정 ---
# 1. model_jeyun_008_hidden_11.py 파일이 
MODEL_DIR = "/app/project_code/jeyun/" 

if MODEL_DIR not in sys.path:
    # 가장 먼저 검색하도록 0번 인덱스에 추가
    sys.path.insert(0, MODEL_DIR) 
    
# 2. 모델 파일에서 preprocess 및 predict 함수를 임포트합니다.
try:
    from model_jeyun_008_hidden_11 import preprocess, predict
    print(f"Successfully imported preprocess and predict from model_jeyun_008_hidden_11.py in {MODEL_DIR}")
except ImportError as e:
    print(f"Error: Failed to import model_jeyun_008_hidden_11: {e}", file=sys.stderr)
    print(f"Check if '{MODEL_DIR}/model_jeyun_008_hidden_11.py' exists and is named correctly.", file=sys.stderr)
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
    print("Loading corpus (hsrc_corpus.jsonl)...")
    corpus_dict = {}
    
    # 1. 코퍼스 데이터 로드 (hsrc_corpus.jsonl)
    # 형식: {'uuid': '...', 'passage': '...'}
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                doc_id = entry.get('uuid')
                passage = entry.get('passage', entry.get('text', ''))
                
                if doc_id and passage:
                    # 'text' 필드가 존재하지 않으므로 'passage'를 'text'에도 저장
                    corpus_dict[doc_id] = {'passage': passage, 'text': passage}
        print(f"✓ Loaded {len(corpus_dict)} documents into corpus_dict.")
    except Exception as e:
        print(f"Error loading corpus file {corpus_file}: {e}", file=sys.stderr)
        sys.exit(1)


    print("Loading queries and QREL (hsrc_train.jsonl)...")
    queries_list = []
    qrel = {} # QREL: {query_index: {doc_id: relevance_score, ...}}
    
    # 2. 학습(쿼리/정답) 데이터 로드 (hsrc_train.jsonl)
    # 형식: {'query_uuid': '...', 'query': '...', 'target_actions': {'target_action_0': score, ...}}
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                query_text = entry.get('query')
                
                if not query_text:
                    continue

                # 쿼리 목록에 추가
                queries_list.append({'query': query_text})
                
                # QREL 데이터 추출
                target_actions = entry.get('target_actions', {})
                
                current_qrel = {}
                # target_actions의 키는 'target_action_0' 형식이며, 값은 관련성 점수(0~4)임.
                for key, score in target_actions.items():
                    # target_actions에는 관련 문단 uuid와 점수가 바로 맵핑되어 있지 않음.
                    # 'paragraphs' 필드를 통해 uuid를 찾아야 하나, 
                    # 제공된 이미지 정보에는 'paragraphs'가 object 타입으로만 명시되어 있고 
                    # 'target_actions'는 relevance label만 명시되어 있습니다.
                    
                    # **[중요 가정]** 현재 시점에서는 target_actions의 키(target_action_N)를
                    # 직접 문단 UUID로 사용할 수 없으므로, 
                    # hsrc_train.jsonl의 'paragraphs' 필드 구조를 기반으로 UUID를 추출해야 합니다.
                    
                    # *임시 조치: 'paragraphs' 필드의 구조를 가정하고 UUID를 가져옵니다.*
                    # 실제 데이터셋 구조에 따라 아래 로직은 수정이 필요할 수 있습니다.
                    paragraphs = entry.get('paragraphs', {})
                    
                    # target_actions 키와 paragraphs 키가 일치한다고 가정하고 UUID를 가져옴.
                    # 예: target_action_0 -> paragraphs의 0번째 문단의 uuid
                    try:
                        # target_actions의 키(예: target_action_0)를 paragraphs의 키(예: paragraph_0)와 연결
                        paragraph_key = key.replace('target_action', 'paragraph')
                        
                        # paragraphs의 값은 {uuid: ..., passage: ...} 이므로, uuid를 추출
                        doc_id = paragraphs.get(paragraph_key, {}).get('uuid')
                        
                        if doc_id:
                            # QREL에 추가 (관련성 점수 float로 변환)
                            current_qrel[doc_id] = float(score) 
                    except Exception as e:
                        print(f"Error parsing QREL data for query {i}: {e}", file=sys.stderr)
                        continue

                # QREL에 현재 쿼리 추가
                if current_qrel:
                    qrel[i] = current_qrel
        
        print(f"✓ Loaded {len(queries_list)} queries and {len(qrel)} QREL entries.")
        
    except Exception as e:
        print(f"Error loading train file {train_file}: {e}", file=sys.stderr)
        sys.exit(1)


    return corpus_dict, queries_list, qrel


def run_evaluation(data_dir: str, output_path: str):
    """
    데이터를 로드하고 모델을 평가하여 nDCG@20을 측정합니다.
    """
    
    # 데이터 파일 경로 설정
    CORPUS_FILE = os.path.join(data_dir, 'hsrc_corpus.jsonl')
    TRAIN_FILE = os.path.join(data_dir, 'hsrc_train.jsonl')
    
    print("\n" + "="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING MODEL EVALUATION")
    print(f"Using Corpus: {CORPUS_FILE}")
    print(f"Using Train Data/QREL: {TRAIN_FILE}")
    print("="*80 + "\n")

    # 1. 데이터 로드 (코퍼스, 쿼리, QREL)
    corpus, queries, qrel = load_data(CORPUS_FILE, TRAIN_FILE)
    
    if not corpus or not queries or not qrel:
        print("[FATAL] Required data for evaluation is missing.", file=sys.stderr)
        return

    # 2. Preprocess 함수 실행 (모델 로드 및 임베딩 계산)
    try:
        print("\nRunning preprocess function...")
        preprocess_start_time = time.time()
        preprocessed_data = preprocess(corpus)
        preprocess_end_time = time.time()
        print(f"Preprocessing completed in {(preprocess_end_time - preprocess_start_time):.2f} seconds.")
        
        all_doc_ids = preprocessed_data.get('corpus_ids', list(corpus.keys())) # 전체 문서 ID 목록
    except Exception as e:
        print(f"\n[FATAL] Preprocessing failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Predict 함수 순차적 실행 및 nDCG 계산
    all_results = []
    total_ndcg_score = 0.0
    evaluated_queries_count = 0

    print("\n" + "="*80)
    print("STARTING PREDICTION AND NDCG CALCULATION")
    print("="*80)

    for i, query in enumerate(queries):
        if i not in qrel:
            # 해당 쿼리에 대한 정답(QREL)이 없으면 평가에서 제외
            continue
            
        start_time = time.time()
        
        # predict 함수 호출 (모델 검색)
        prediction_result = predict(query, preprocessed_data)
        
        end_time = time.time()
        
        # 3.1 Predicted Scores (y_score) 벡터 생성 (NumPy)
        # 전체 문서 ID 순서(all_doc_ids)에 맞게 예측 점수 벡터를 초기화
        y_score = np.zeros(len(all_doc_ids))
        
        # predict 결과(최대 20개 문서)를 득점 벡터에 반영 (리랭커/유사도 점수 사용)
        for result_item in prediction_result:
            try:
                doc_id = result_item['paragraph_uuid']
                # 예측된 문서 ID가 전체 목록에서 몇 번째 인덱스인지 찾음
                doc_idx = all_doc_ids.index(doc_id)
                # 예측 점수 반영
                y_score[doc_idx] = result_item['score']
            except ValueError:
                # 결과 문서가 코퍼스에 없는 경우 (이상 케이스)
                continue 

        # 3.2 True Relevance (y_true) 벡터 생성 (NumPy)
        # 전체 문서 ID 순서에 맞춰 정답(qrel) 점수를 벡터에 반영
        y_true = np.zeros(len(all_doc_ids))
        for doc_id, true_score in qrel[i].items():
            try:
                doc_idx = all_doc_ids.index(doc_id)
                y_true[doc_idx] = true_score
            except ValueError:
                # QREL에 있는 문서가 코퍼스에 없는 경우 (이상 케이스)
                continue
        
        # 3.3 nDCG@20 계산
        ndcg_20 = 0.0
        try:
            # y_true와 y_score는 2D 배열 형태(shape (1, N))로 reshape하여 전달
            # k=20에 대해 평가
            ndcg_20 = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=20)
            
            # 총점에 누적
            total_ndcg_score += ndcg_20
            evaluated_queries_count += 1
            
        except Exception as e_ndcg:
            print(f"Error calculating nDCG for query {i}: {e_ndcg}", file=sys.stderr)
            # ndcg_20은 0.0으로 유지

        # 3.4 결과 정리 및 저장
        result_entry = {
            'query_index': i,
            'query': query['query'],
            'qrel_entries': len(qrel[i]),
            'results_count': len(prediction_result),
            'ndcg@20': float(ndcg_20),
            'time_ms': (end_time - start_time) * 1000,
            'top_results': [r['paragraph_uuid'] for r in prediction_result[:5]], # 상위 5개 문서 ID
        }
        all_results.append(result_entry)
        
        # 결과 요약 출력
        print(f"Query {i+1}/{len(queries)} completed in {result_entry['time_ms']:.2f}ms. | NDCG@20: {ndcg_20:.4f}")

    # 4. 최종 결과 요약 및 저장
    
    final_avg_ndcg = total_ndcg_score / evaluated_queries_count if evaluated_queries_count > 0 else 0.0
    
    print("\n" + "="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] EVALUATION COMPLETE")
    print(f"Total Queries: {len(queries)}")
    print(f"Evaluated Queries: {evaluated_queries_count}")
    print(f"FINAL AVERAGE nDCG@20: {final_avg_ndcg:.4f}")
    print("="*80)

    # 4.1 최종 요약 데이터를 파일에 저장
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'total_queries': len(queries),
        'evaluated_queries': evaluated_queries_count,
        'average_ndcg@20': final_avg_ndcg,
        'individual_query_results': all_results
    }
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_path, f"test_jeyun_008_hidden_11_{timestamp}.json")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)
        
        print(f"Detailed results saved to: {output_filename}")

    except Exception as e:
        print(f"\n[ERROR] Failed to save results to file: {e}", file=sys.stderr)


if __name__ == "__main__":
    # 데이터 파일이 있는 디렉토리 (컨테이너 내부 경로 기준)
    # 실제 데이터셋 경로에 맞게 수정해야 합니다.
    DATA_DIR = os.path.join(
        '/app/project_code',
        'Dataset and Baseline', 
        'baseline_submission', 
        'hsrc'
    ) 
    
    # 테스트 결과 저장 디렉토리 (호스트에 마운트된 경로 아래에 생성)
    TEST_OUTPUT_DIR = '/app/project_code/jeyun/test_results' 
    
    # 결과 디렉토리 생성 확인
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {TEST_OUTPUT_DIR}")
        
    run_evaluation(DATA_DIR, TEST_OUTPUT_DIR)
