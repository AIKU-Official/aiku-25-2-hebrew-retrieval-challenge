# aiku-25-2-hebrew-retrieval-challenge


# aiku-25-2-beating_contest

## 소개
Coda bench 에서 열린 challenge 로서 제한된 상황 속에서 수행하는 
히브리어(이스라엘 언어) 문서 검색 challenge 입니다.

대회 링크: https://www.codabench.org/competitions/9950/?ref=mlcontests#/pages-tab
시도한 모델 노션 링크 : https://www.notion.so/aiku/287a7930e09c802fb0cae9d7f5e943a1?source=copy_link

# WORK DISTRIBUTION

- 짧은 시간 안에 제출을 완성해야했기에,
- 개발 효율을 위해 각자 submission 을 제출하되, 개발한 모델의 결과를 주기적으로 공유하는 식으로 프로젝트 진행.

# 대회 제한 사항
- `g5.xlarge` 인스턴스는 추론 코드 실행 및 점수 계산에 사용됩니다. 쿼리당 `predict` 함수 실행 시간이 2.0초 이하인 제출물만 최종 단계에서 평가됩니다.
- `preprocess` 함수의 최대 실행 시간은 1.5시간입니다.
- 제출 zip 파일의 최대 크기는 7GB를 초과할 수 없습니다. 제출된 모델은 재현 가능해야 하므로, 훈련 시간은 합리적이어야 하며 사용된 모든 추가 훈련 데이터는 다른 모든 참가자에게 제공되어야 합니다.
- 제출된 모델은 외부 머신러닝 도구에 의존하거나 어떤 종류의 외부 서비스 호출 없이 완전히 로컬, 오프라인 환경에서 실행되어야 합니다.


# Dataset

- train dataset
출처 :
Kol-Zchut (a website providing information on legal and civil rights), 
the Hebrew edition of Wikipedia, 
and protocols from Knesset (Israeli parliament) committees.

![image.png](attachment:cbd6fcb6-bc18-493e-97da-ace4c8fe3e12:image.png)

<img width="1077" height="507" alt="image" src="https://github.com/user-attachments/assets/fd4f938b-37e5-4157-92df-0ef323c06d7c" />


0 - paragraph has no relevance to the query
1 - paragraph has weak relevance to the query
2 - paragraph has partial relevance to the query
3 - paragraph has strong relevance to the query
4 - paragraph has complete relevance to the query


**주요 실험**

실험 흐름

1. **base line 선정 : retrieve-then-rerank 구조 사용.**
2. **어떤 모델들을 사용해볼 것인가?**
    1. 기준 1 : 도메인 sota 급 논문 레퍼런스 
    2. 기준 2 : multilingual Retrieval leaderboard 내 상위 모델 사용
3. **어떻게 성능을 끌어올릴 것인가.**  
    1. finetuning
        1. train dataset 의 함정
        2. DPR, Contrastive Learning. 
    2. 앙상블
        1. 두 Dense Model 앙상블
        2. Sparse Model( e.g. BM25 ) 를 앙상블 혹은 0단계로 사용
    3. 중간레이어 추출
        1. BERT attention layer 의 중간 히든 레이어를 직접 사용 시도.
    4. 번역 (잡기술)
        1. 히브리어 → 영어 / 영어 retrieve 수행
    5. 작은 llm 사용 해보기
        1. Qwen-0.6B 사용해보기.

4. 실험시에 중요하게 고려되었던 제한 조건

1. Commercial Liicence 허용되는 모델만 허용 / API calling 금지 
    1. 이로 인해서 Gemini/voyage등 성능이 좋았지만 사용 불가
    2. Qwen 시리즈 LM을 적극적으로 활용
2. 모델은 무조건 로컬로 offline으로 돌아가야하므로 hf모델을 다 다운받아서 환경에 업로드 후 학습
3. 최대 총 용량은 7GB
4. 총 전처리 시간 1.5 h
5. Test 시에 inference time 제한 query당 2초 이내 
    
    

## 훈련 및 평가

**훈련 시도 1**

**사용 모델 :  ( retriever )** Qwen3-embedding 0.6B + **( reranker )** Qwen3-reranker 0.6B

**결과 평가**  : inference time 2초 초과로 제한 조건 초과 / training time(전처리 시간도 초과 위험)

→ 더 가벼운 모델 선정으로 방향성 지정

**훈련 시도 2**

**사용 모델 :  ( retriever )** Knesset-multi-E5  + **( reranker ) BGE reranker**

**결과 평가**  : retreiver를 hebrew law domain으로 pretrained된 모델을 사용하였지만 결과 하락

→ reranker문제인지 확인 필요

**훈련 시도 3**

**사용 모델 :  ( retriever )** Knesset-multi-E5  + **( reranker ) reranker 제거**

**결과 평가**  : 0.41의 성능 → reranker제거시 성능 개선 retreiver에 집중해서 개선

**훈련 시도 4**

**사용 모델 :  ( retriever )** AlephBERT  DPR  + **( reranker ) reranker finetuning**

**결과 평가**  :  AlephBERT 히브리어 기반의 BERT 에 DPR + Coress-encoder reranker 파인튜닝

→ val set NDCG 값 폭증 / passage 설정 문제/ 서버 이슈로 시도 불가

**훈련 시도 4**

**사용 모델 :  ( retriever )** AlephBERT  gimmel 512  + **( reranker ) Qwen3 Embedding 4B/0.6B**

**결과 평가**  :  AlephBERT bert에서 pre process를 다르게 한 pretrained model을 retrevier로 사용 

→ Qwen 모델 크기로 인해 inference time 초과/ pre-process model쪽 모양에 맞춰줄려고 전처리를 다시하다보니 training 시간 초과

**훈련 시도 5**

**사용 모델 :  ( retriever )** BGE Retreiver의 tokenizer를 그대로 활용한 BM25  + E5-large(multilingual) 두 모델 결과 concat하여 top-100 문서 추출 

**( reranker ) BGE Reranker 사용** 

**결과 평가**  : 
<img width="1800" height="355" alt="image" src="https://github.com/user-attachments/assets/ee2b178a-39e5-49af-847a-f6e16cb5c38e" />



0.417로 baseline 보다 개선 및 13등 등록

**훈련 시도 6 *(SOTA)**

troubleshooting
<img width="623" height="648" alt="image" src="https://github.com/user-attachments/assets/eddc3f25-f038-441f-9081-a2012324bf74" />

train data 에 대해서 아예 점수가 0.0000이 나와서 NDCG@20 점수가  안나옴을 확인

→Lexical ( 어휘 단순 일치) 도 못하는 것으로 파악

→ TF-IDF / BM25 를 retriever 에 추가

**< 최종 결과 >**
<img width="1110" height="261" alt="image" src="https://github.com/user-attachments/assets/b0feda62-28b1-4af7-ada5-edc5ee829531" />

## 결과

Private score (nDCG@20) : 0.6399 (8등)
<img width="876" height="2637" alt="image" src="https://github.com/user-attachments/assets/64c3fdfa-9dd1-4aca-9e5a-644d322fdd56" />

## 한계 분석

시도해보지 못했던 추가적인 개선 방안 아이디어

< 모델 양자화  / 경량화  > 

이번 대회에서의 inference time이 query당 2초로 생각보다 시간 초과가 발생하는 경우가 많았음
<img width="943" height="784" alt="image" src="https://github.com/user-attachments/assets/b2aeed6b-a925-484f-a228-6713a9e19583" />

이번 대회의 경우 

<LLM을 이용한 데이터 증강 및 finetuning> 
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/2e31cdc0-c7b2-43e4-8dab-8d43a607967d" />

결국 retriever 는 bi-encoder 이고 e5의 경우 어떤지 모르겠지만 일단 A B 양 측 parameter 를 공유한다고 가정.
 여기에다가 우리가 <Document, Query> pair 를 많이 많이 만들어야 제대로 fine-tuning 할 수 있음.

생각해볼 수 있는것은 우리의 hsrc_corpus.jsonl 에 127,000개의 문서들이 있으니

chatGPT같은 LLM api로  
“다음 문서로부터 대답할 수 있는 질문을 히브리어의 문장으로 생성해라. 문서 : {passage}”
라고 프롬프팅 해버리면 당연히 꽤 좋은 합성 데이터를 만들 수 있고 이를 통해 학습하는 것은 무리 없음.



## 팀 역할 ( 기여도 %)
- 권태진(20%) : data 전처리 및 baseline 작성
- *이제윤(20%) : 팀장, 아이디어 제시, 모델링 시도 및 기타 행정 
- 이성은(20%) : Finetuning 집중, 추가 모델링 시도
- 이주형(20%) : Finetuning 집중, 추가 모델링 시도
- 장국영(20%) : 앙상블, 추가 모델링 시도, ppt 등
