import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime

# NumPy가 필요하므로 임포트합니다. (코퍼스 데이터 생성용)
try:
    import numpy as np
except ImportError:
    print("Error: NumPy not found. Please ensure all dependencies are installed.", file=sys.stderr)
    sys.exit(1)

# model.py 에서 필요한 핵심 함수들을 임포트합니다.
# model.py가 다른 경로에 있을 경우 해당 경로를 sys.path에 추가합니다.
MODEL_PATH = os.environ.get("/home/aikusrv01/aiku/25_2_hebrew_retrieval_challenge/Dataset and Baseline/baseline_submission/")
if MODEL_PATH and MODEL_PATH not in sys.path:
    sys.path.insert(0, MODEL_PATH)

try:###model이름###
    from model_taejin_01 import preprocess, predict
except ImportError:
    print("Error: Could not import preprocess/predict from model.py.", file=sys.stderr)
    print("Ensure model.py is in the same directory.", file=sys.stderr)
    sys.exit(1)


def setup_test_data() -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]]]:
    """로컬 테스트를 위한 가상 코퍼스와 쿼리를 생성합니다. (실제 데이터와 구조 일치)"""
    
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
    return test_corpus_dict, test_queries

def run_local_test(output_path: str):
    """
    로컬 테스트를 실행하고 결과를 JSON 파일로 저장합니다.
    """
    print("\n" + "="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING CODALAB LOCAL TEST SIMULATION")
    print("="*80 + "\n")

    # 1. 테스트 데이터 준비
    corpus, queries = setup_test_data()
    
    # 2. Preprocess 함수 실행 (모델 로드 및 임베딩 계산)
    try:
        print("Running preprocess function...")
        # preprocess 함수는 모델 인스턴스와 임베딩을 반환합니다.
        preprocessed_data = preprocess(corpus)
    except Exception as e:
        print(f"\n[FATAL] Preprocessing failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Predict 함수 순차적 실행 및 결과 수집
    all_results = []
    for i, query in enumerate(queries):
        start_time = time.time()
        print(f"\n--- Running Query {i+1}/{len(queries)} ---")
        
        # predict 함수 호출 (preprocessed_data를 인수로 전달)
        prediction_result = predict(query, preprocessed_data)
        
        end_time = time.time()
        
        # 결과에 쿼리 정보와 소요 시간 추가
        result_entry = {
            'query': query['query'],
            'results': prediction_result,
            'time_ms': (end_time - start_time) * 1000
        }
        all_results.append(result_entry)
        
        # 결과 요약 출력
        if prediction_result:
            print(f"Query {i+1} completed in {result_entry['time_ms']:.2f}ms.")
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
    #  테스트 결과 파일을 저장할 컨테이너 내부 경로
    # 이 경로는 도커 run 시 -v 옵션으로 마운트된 호스트 폴더 하위에 존재해야 합니다.
    TEST_OUTPUT_DIR = '/app/project_code/taejin/test_results' 
    
    # 결과 디렉토리 생성 확인 (없으면 생성)
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {TEST_OUTPUT_DIR}")
        
    run_local_test(TEST_OUTPUT_DIR)

