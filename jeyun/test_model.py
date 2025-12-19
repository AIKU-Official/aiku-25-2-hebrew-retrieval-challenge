import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime

# NumPy와 Scikit-learn 임포트
try:
    import numpy as np
    from sklearn.metrics import ndcg_score # nDCG 계산을 위해 추가
except ImportError:
    print("Error: Required libraries (NumPy, Scikit-learn) not found.", file=sys.stderr)
    sys.exit(1)



import sys
import os

# 1. 파일이 존재하는 정확한 디렉토리 경로를 변수에 할당합니다.
#    (예: model_jeyun_010_GGUF.py 파일이 /app/jeyun/model_jeyun/ 폴더 안에 있다고 가정)
MODEL_DIR = "/app/project_code/jeyun/" 
# 2. 해당 디렉토리를 Python 검색 경로(sys.path)에 추가합니다.
if MODEL_DIR not in sys.path:
    # 가장 먼저 검색하도록 0번 인덱스에 추가
    sys.path.insert(0, MODEL_DIR) 
    
# 3. 파일 이름(model_jeyun_010_GGUF)에서 함수를 직접 임포트합니다.
#    (파일 이름에 .py 확장자를 제외한 모듈 이름 사용)
try:
    from model_jeyun_010_GGUF import preprocess, predict
except ImportError as e:
    # 경로가 잘못되었거나 파일 이름이 틀렸을 가능성이 높습니다.
    print(f"Error: {e}")
    print(f"Check if '{MODEL_DIR}/model_jeyun_010_GGUF.py' exists.")
    sys.exit(1)


def setup_test_data() -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]], Dict[int, Dict[str, float]]]:
    """
    로컬 테스트를 위한 가상 코퍼스, 쿼리, 정답(QREL) 데이터를 생성합니다.
    (실제 테스트를 위해서는 QREL 부분을 실제 정답 데이터로 대체해야 합니다.)
    """
    
    # 1. 테스트용 코퍼스 (200개 문서 가정)
    test_corpus_dict = {}
    for i in range(200):
        doc_id = f"doc_{i+1}"
        test_corpus_dict[doc_id] = {
            'passage': f"This is the document passage number {i+1}. It contains information about machine learning models and large language models (LLMs).",
            'text': f"Full text for document {i+1}."
        }
    test_corpus_dict["doc_0"] = {
        'passage': f"דן קם בבוקר בשעה שבע. הוא שתה קפה חם ואכל פרוסת לחם. לאחר מכן, הוא יצא לעבודה באופניים שלו. הוא אוהב מאוד את השקט של הבוקר המוקדם. doc number is {0}. 단은 아침 7시에 일어났습니다. 그는 따뜻한 커피를 마시고 빵 한 조각을 먹었습니다. 그 후, 그는 자신의 자전거를 타고 출근했습니다. 그는 이른 아침의 고요함을 매우 좋아합니다.",
        'text': f"Full text for document {0}."
    }
    test_corpus_dict["doc_1"] = {
        'passage': f"ירושלים היא עיר עם היסטוריה עתיקה מאוד. אנשים מכל העולם מגיעים לבקר בכותל המערבי. העיר מחברת בין מסורת לחיים מודרניים. בערב, השווקים בעיר מלאים באורות וקולות.	 doc number is {2}. 예루살렘은 매우 오래된 역사를 가진 도시입니다. 전 세계의 사람들이 통곡의 벽(서쪽 벽)을 방문하기 위해 옵니다. 이 도시는 전통과 현대 생활을 연결합니다. 저녁에는 도시의 시장들이 빛과 소리로 가득 찹니다.",
        'text': f"Full text for document {0}."
    }

    # 2. 테스트용 쿼리 (2개 가정)
    test_queries = [
        {'query': '1. באיזו שעה דן קם בבוקר?1. 단은 아침 몇 시에 일어났습니까?'},
        {'query': '2. באמצעות מה דן יצא לעבודה?	2. 단은 무엇을 이용하여 출근했습니까?'},
        {'query': '1. איזה מקום מפורסם אנשים מכל העולם באים לבקר בו?	1. 전 세계의 사람들이 방문하러 오는 유명한 장소는 어디입니까?'},
        {'query': '2. מה מאפיין את השווקים בירושלים בערב?	2. 저녁의 예루살렘 시장들을 특징짓는 것은 무엇입니까?'}
    ]
    
    # 3. 쿼리 관련성 데이터 (정답, QREL) - 임시 가상 데이터
    # { 쿼리_인덱스: { 문서_ID: 실제_관련성_점수 (float), ... } }
    # 예: 쿼리 0은 doc_10에 3점, doc_15에 2점의 관련성이 있음
    qrel = {
        0: {"doc_0": 4.0, "doc_15": 2.0, "doc_3": 1.0, "doc_50": 0.0}, 
        1: {"doc_0": 4.0, "doc_101": 3.0, "doc_150": 1.0, "doc_2": 0.0},
        2: {"doc_1": 4.0, "doc_5": 2.0, "doc_20": 1.0, "doc_30": 0.0},
        3: {"doc_1": 4.0, "doc_8": 2.0, "doc_25": 1.0, "doc_40": 0.0}
    }
    
    return test_corpus_dict, test_queries, qrel


def run_local_test(output_path: str):
    """
    로컬 테스트를 실행하고 결과를 JSON 파일로 저장합니다.
    """
    print("\n" + "="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING CODALAB LOCAL TEST SIMULATION")
    print("="*80 + "\n")

    # 1. 테스트 데이터 준비 (QREL 포함)
    corpus, queries, qrel = setup_test_data()
    
    # 2. Preprocess 함수 실행 (모델 로드 및 임베딩 계산)
    try:
        print("Running preprocess function...")
        preprocessed_data = preprocess(corpus)
        all_doc_ids = preprocessed_data['corpus_ids'] # 전체 문서 ID 목록
    except Exception as e:
        print(f"\n[FATAL] Preprocessing failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Predict 함수 순차적 실행 및 결과 수집
    all_results = []
    for i, query in enumerate(queries):
        start_time = time.time()
        print(f"\n--- Running Query {i+1}/{len(queries)} ---")
        
        # predict 함수 호출
        prediction_result = predict(query, preprocessed_data)
        
        end_time = time.time()
        
        # 3.1 Predicted Scores (y_score) 벡터 생성 (NumPy)
        #    전체 문서 ID 순서(all_doc_ids)에 맞게 예측 점수 벡터를 초기화
        y_score = np.zeros(len(all_doc_ids))
        
        # predict 결과(20개 문서)를 득점 벡터에 반영 (리랭커 점수 사용)
        for result_item in prediction_result:
            try:
                doc_id = result_item['paragraph_uuid']
                # 예측된 문서 ID가 전체 목록에서 몇 번째 인덱스인지 찾음
                doc_idx = all_doc_ids.index(doc_id)
                # 예측 점수 반영
                y_score[doc_idx] = result_item['score']
            except ValueError:
                # 마운트된 코퍼스 ID와 결과 ID가 불일치하는 경우
                continue 

        # 3.2 True Relevance (y_true) 벡터 생성 (NumPy)
        #    전체 문서 ID 순서에 맞춰 정답(qrel) 점수를 벡터에 반영
        y_true = np.zeros(len(all_doc_ids))
        if i in qrel: # 현재 쿼리 인덱스에 대한 정답(qrel)이 있다면
            for doc_id, true_score in qrel[i].items():
                try:
                    doc_idx = all_doc_ids.index(doc_id)
                    y_true[doc_idx] = true_score
                except ValueError:
                    # QREL에 있는 문서가 코퍼스에 없는 경우
                    continue
        
        # 3.3 nDCG@20 계산 (k=20)
        ndcg_20 = 0.0
        try:
            # y_true와 y_score는 2D 배열 형태(shape (1, N))로 reshape하여 전달
            # k=20에 대해 평가
            ndcg_20 = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=20)
        except Exception as e_ndcg:
            print(f"Error calculating nDCG: {e_ndcg}", file=sys.stderr)
            # ndcg_20은 0.0으로 유지

        # 3.4 결과 정리 및 저장
        result_entry = {
            'query': query['query'],
            'results': prediction_result,
            'time_ms': (end_time - start_time) * 1000,
            'ndcg@20': float(ndcg_20) # nDCG 점수 추가
        }
        all_results.append(result_entry)
        
        # 결과 요약 출력
        if prediction_result:
            print(f"Query {i+1} completed in {result_entry['time_ms']:.2f}ms.")
            print(f"nDCG@20 Score: {ndcg_20:.4f}") # nDCG 점수 출력
            print(f"Top 3 results: {[r['paragraph_uuid'] for r in prediction_result[:3]]}")
        else:
            print(f"Query {i+1} failed or returned no results.")

    # 4. 전체 결과 JSON 파일로 저장
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_path, f"local_test_results_{timestamp}.json")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        
        print("\n" + "="*80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] LOCAL TEST COMPLETE!")
        print(f"Final results saved to: {output_filename}")
        print("="*80)

    except Exception as e:
        print(f"\n[ERROR] Failed to save results to file: {e}", file=sys.stderr)


if __name__ == "__main__":
    TEST_OUTPUT_DIR = '/app/project_code/jeyun/test_results' 
    
    # 결과 디렉토리 생성 확인
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {TEST_OUTPUT_DIR}")
        
    run_local_test(TEST_OUTPUT_DIR)
