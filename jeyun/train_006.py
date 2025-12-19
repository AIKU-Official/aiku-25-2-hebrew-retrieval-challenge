import os
import torch
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import random
import json
import traceback
# 🚨 PEFT 및 Hugging Face 필수 import
from transformers import ( 
    AutoModel, AutoTokenizer, 
    Trainer, TrainingArguments as HfTrainingArguments,
    SchedulerType
)
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SimilarityFunction
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss # Loss 함수는 유지


# --- 0. 설정 및 경로 정의 ---
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'bge-reranker-v2-m3') 
DATA_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hsrc', 'hsrc_train.jsonl') 
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'bge-reranker-v2-m3_006')

# 최적화 및 LoRA 하이퍼파라미터
LEARNING_RATE = 2e-5      
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05       
BATCH_SIZE = 8           # 🚨 메모리 절약을 위해 BATCH_SIZE는 낮게 설정
GRAD_ACCUM_STEPS = 4     # 🚨 Gradient Accumulation을 사용하여 실질 배치 크기(32) 유지
EVAL_STEPS = 500          
ES_PATIENCE = 3           
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RELEVANCE_THRESHOLD = 3 

# LoRA 설정
LORA_R = 8
LORA_ALPHA = 16


# --- 1. 데이터 로드 및 형식 변환 (변경 없음) ---
# ... (load_and_prepare_data 함수 코드는 동일하게 유지) ...
def load_and_prepare_data(file_path, val_ratio=0.1, max_val_queries=2000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    print(f"Loading data from: {file_path}")
    dataset = load_dataset('json', data_files={'train': file_path}, split='train')
    
    split_dataset = dataset.train_test_split(test_size=val_ratio, seed=42)
    train_set = split_dataset['train']
    val_set_full = split_dataset['test']
    val_set = val_set_full.select(range(min(len(val_set_full), max_val_queries)))
    
    # --- 훈련 데이터 준비: datasets.Dataset 객체로 유지 ---
    # PEFT Trainer는 datasets.Dataset을 직접 사용
    
    # 훈련에 필요한 필드를 포함하는 리스트 생성 (쿼리, 긍정 문서)
    train_data_list = []
    
    for item in train_set:
        query = item['query']
        paragraphs = item['paragraphs']
        target_actions = item['target_actions']
        
        for action_key, relevance_score_raw in target_actions.items():
            paragraph_key = action_key.replace('target_action', 'paragraph')
            
            try:
                relevance_score = int(relevance_score_raw)
            except (ValueError, TypeError):
                continue
            
            if relevance_score >= RELEVANCE_THRESHOLD and paragraph_key in paragraphs:
                positive_passage = paragraphs[paragraph_key]['passage']
                train_data_list.append({
                    'query': query,
                    'positive_passage': positive_passage
                })
    
    # 리스트를 Hugging Face Dataset으로 변환
    train_dataset = datasets.Dataset.from_list(train_data_list)
                
    # --- 검증 데이터 준비 (IR Evaluator용) ---
    queries, corpus, relevant_docs = {}, {}, defaultdict(dict)

    eval_data_list = []

    # ... (검증 데이터셋 구성 로직은 이전과 동일) ...
    for idx, item in enumerate(val_set):
        query = item['query']
        paragraphs = item['paragraphs']
        target_actions = item['target_actions']
        
        q_id = item.get('query_uuid', f'q_{idx}')
        queries[q_id] = query
        
        passage_map = {} 
        for key, p_data in paragraphs.items():
            p_uuid = p_data['uuid']
            corpus[p_uuid] = p_data['passage']
            passage_map[key] = p_uuid

        for action_key, relevance_score_raw in target_actions.items():
            paragraph_key = action_key.replace('target_action', 'paragraph')
            
            try:
                relevance_score = int(relevance_score_raw)
            except (ValueError, TypeError):
                continue
            
            if paragraph_key in passage_map:
                p_uuid = passage_map[paragraph_key]
                relevant_docs[q_id][p_uuid] = relevance_score 

                # 추가: 긍정 문서(score >= RELEVANCE_THRESHOLD)만 eval_dataset에 추가
                if relevance_score >= RELEVANCE_THRESHOLD:
                    positive_passage = paragraphs[paragraph_key]['passage']
                    eval_data_list.append({
                        'query': query,
                        'positive_passage': positive_passage
                    })

    eval_dataset = datasets.Dataset.from_list(eval_data_list)

    print(f"Loaded {len(train_data_list)} (Query, Positive) pairs for training.")
    print(f"Prepared {len(queries)} validation queries.")
    print(f"Prepared {len(corpus)} documents in validation corpus (via UUIDs).")
    
    return train_dataset, eval_dataset, queries, corpus, relevant_docs


# --- 2. 커스텀 Trainer 및 Data Collator 정의 ---

class CustomDPRCollator:
    """훈련 데이터를 토큰화하고 Loss 계산에 필요한 딕셔너리를 반환합니다."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        queries = [f['query'] for f in features]
        passages = [f['positive_passage'] for f in features]
        
        # 쿼리와 문단을 따로 토큰화 (Bi-Encoder 구조)
        q_inputs = self.tokenizer(queries, padding=True, truncation=True, return_tensors='pt')
        p_inputs = self.tokenizer(passages, padding=True, truncation=True, return_tensors='pt')
        
        return {
            'q_input_ids': q_inputs['input_ids'],
            'q_attention_mask': q_inputs['attention_mask'],
            'p_input_ids': p_inputs['input_ids'],
            'p_attention_mask': p_inputs['attention_mask'],
        }

class CustomDPRTrainer(Trainer):
    """MultipleNegativesRankingLoss를 사용하여 Loss를 계산하는 커스텀 Trainer."""
    def __init__(self, model, args, data_collator, train_dataset,eval_dataset, tokenizer):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
        )

    # 🚨 compute_loss 로직을 완전히 재작성하여 S-T Loss를 우회
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        # 1. BGE 임베딩 생성 함수 (GPU/PEFT 모델에서 실행)
        def get_embedding(input_ids, attention_mask):
            # model(AutoModel + LoRA)의 출력
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # BGE/RoBERTa의 Mean Pooling 구현
            embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            # L2 정규화 (BGE 필수)
            return F.normalize(mean_embeddings, p=2, dim=1)
        
        # 2. 쿼리와 문단 임베딩 생성 (Q_emb, P_emb)
        # inputs 딕셔너리가 CUDA로 이동했음을 가정
        q_emb = get_embedding(inputs['q_input_ids'], inputs['q_attention_mask'])
        p_emb = get_embedding(inputs['p_input_ids'], inputs['p_attention_mask'])
        
        # 3. MNR Loss 핵심 로직: Dot Product (유사도) 및 Softmax 계산
        # Batch Size: N
        # 쿼리 임베딩: (N, D), 문단 임베딩: (N, D)
        # 유사도 행렬: (N, N)
        
        # 내적(Dot Product)을 통한 유사도 계산
        # 🚨 Cross-Entropy / MNR Loss에 사용하기 위해 log_softmax를 적용
        similarity_matrix = torch.matmul(q_emb, p_emb.transpose(0, 1)) 
        
        # 4. Loss 계산
        # MNR Loss는 In-Batch Negatives를 사용하며, 
        # 정답(Positive)은 대각선(Diagonal) 위치에 존재함 (Q_i vs P_i)
        
        # Log Softmax 계산 (Softmax(Sim)에 Negative Log Likelihood Loss를 적용)
        log_softmax_scores = F.log_softmax(similarity_matrix, dim=1)
        
        # 정답 인덱스는 [0, 1, 2, ..., N-1] 대각선
        target_labels = torch.arange(len(q_emb), device=q_emb.device)
        
        # Negative Log Likelihood Loss 적용 (대각선 위치의 로그 확률을 최대화)
        loss = F.nll_loss(log_softmax_scores, target_labels)

        return (loss, {'loss': loss.detach()}) if return_outputs else loss

# --- 3. 트레이너 설정 및 실행 ---
def setup_lora_trainer(train_dataset, eval_dataset, queries, corpus, relevant_docs, model_path):
    # 3.1. 토크나이저 및 기본 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    base_model.to(DEVICE)

    # 3.2. LoRA 설정 및 적용
    lora_config = LoraConfig(
        r=LORA_R, 
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value", "key"], # XLM-R/RoBERTa 기반 모델의 일반적인 타겟
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(base_model, lora_config)
    print("LoRA 모델로 변환 완료. 학습 가능한 파라미터:")
    model.print_trainable_parameters() 

    # 3.3. Training Arguments 설정
    training_args = HfTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS, # 🚨 그래디언트 누적 설정
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=SchedulerType.COSINE, 
        optim="adamw_torch", 
        fp16=True, 
        logging_steps=50,
        remove_unused_columns=False, # <-- 이 줄을 추가합니다.

        # 🌟 Evaluation/Save Strategy 🌟
        eval_strategy="steps",    
        eval_steps=EVAL_STEPS,          
        save_strategy="steps",
        save_steps=EVAL_STEPS,          
        save_total_limit=3,             
        load_best_model_at_end=True,    
        metric_for_best_model="eval_ndcg@10", # 🚨 Metric 이름을 'eval_' 접두사와 함께 명시
        greater_is_better=True,
    )

    # 3.4. Data Collator 및 Evaluator 설정
    data_collator = CustomDPRCollator(tokenizer)
    
    # IR Evaluator는 Trainer와 분리하여 Callbacks으로 처리
    # Note: transformers.Trainer는 EvaluationStrategy.STEPS를 지원합니다.
    
    # 3.5. Custom Trainer 인스턴스화
    trainer = CustomDPRTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # 검증 데이터셋 추가
        tokenizer=tokenizer,
    )
    
    # 3.6. Callbacks 추가 (Early Stopping 및 IR Evaluator)
    from transformers import TrainerCallback
    
    # IR Evaluator를 Callback 형태로 만들어줘야 함 (Trainer에 직접 Evaluator 인자가 없기 때문)
    class IREvaluatorCallback(TrainerCallback):
        def __init__(self, evaluator):
            self.evaluator = evaluator
        
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % args.eval_steps == 0:
                print(f"\n--- Running IR Evaluation at step {state.global_step} ---")
                
                # 🚨 S-T Evaluator 실행 및 결과 얻기
                output_scores = self.evaluator(model.module if hasattr(model, 'module') else model)
                
                # NDCG@10을 Trainer가 인식하도록 로그에 추가
                ndcg_score = output_scores.get('ndcg@10')
                if ndcg_score is not None:
                    state.log_history.append({
                        'global_step': state.global_step,
                        'eval_ndcg@10': ndcg_score
                    })
                
                # Early Stopping을 위한 컨트롤 반환 (복잡하므로 일단 로그만 사용)
                return control 
            
    trainer.add_callback(IREvaluatorCallback(
        InformationRetrievalEvaluator(
            queries, corpus, relevant_docs, name='ndcg@10', main_score_function=SimilarityFunction.COSINE
        )
    ))

    return trainer


# --- 4. 메인 실행 ---
if __name__ == '__main__':
    try:
        # 1. 데이터 준비
        train_data, eval_data, val_queries, val_corpus, val_qrels = load_and_prepare_data(DATA_FILE_PATH)
        
        if not train_data:
            print("FATAL ERROR: Training data is empty. Exiting.")
            exit()
            
        # 2. 트레이너 설정
        trainer = setup_lora_trainer(train_data, eval_data, val_queries, val_corpus, val_qrels, LOCAL_MODEL_PATH)
        
        # 3. 훈련 시작
        print("\n" + "="*50)
        print("Starting BGE-M3 LoRA Fine-tuning (Custom Trainer Mode)...")
        print("="*50)
        
        trainer.train()
        
        # 4. 가중치 저장 (PEFT 모델 가중치 저장)
        final_save_path = os.path.join(OUTPUT_DIR, "final_best_lora_model")
        
        # load_best_model_at_end=True에 따라 최적 모델이 로드된 상태에서 저장
        trainer.save_model(final_save_path) 
        
        # 🚨 LoRA 모델만 저장 (디스크 공간 절약)
        # model.save_pretrained(final_save_path) # Trainer가 처리하므로 불필요
        
        print(f"\n✓ Fine-tuning complete. Best LoRA weights saved to {final_save_path}")

    except Exception as e:
        print(f"\n\n🚨 CRITICAL ERROR DURING TRAINING: {e}")
        traceback.print_exc()
        print("Please check file paths, GPU memory, and data keys.")