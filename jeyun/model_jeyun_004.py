import json
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

"""
    Retriever : Qwen3-embedding-0.6B bi-encoder
    Reranker : Qwen3-embedding-0.6B cross-encoder 망함.
"""

class QwenRetriever:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the Qwen retriever using the qwen model.
        """
        # Use local model
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'Qwen3-Embedding-0.6B')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local qwen model from: {model_name}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Loading Qwen model on device: {self.device}")
        
        # old model
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        
        try:
            self.model = SentenceTransformer(
                model_name, 
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"}
            )
        except Exception as e:
            print(f"Warning: Failed to load Qwen with specific args ({e}). Trying standard load.")
            self.model = SentenceTransformer(model_name, device=self.device)
            
        self.model.eval()
        
        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.corpus_ids = []
        self.corpus_embeddings = None

    def embed_texts(self, texts, is_query=False, batch_size=4): 
        """
        Generates embeddings for texts using Qwen3 model with proper instruction format.
        """
        # Qwen models benefit from using the 'query' prompt for queries.
        prompt_name = "query" if is_query else None
        
        # Use SentenceTransformer's encode method
        embeddings = self.model.encode(
            sentences=texts,
            prompt_name=prompt_name, # Use the Qwen-specific prompt if available
            batch_size=batch_size,
            show_progress_bar=not is_query,
            convert_to_numpy=True,
            # Ensure no unnecessary L2 normalization if the model already handles it
            normalize_embeddings=True 
        )
        
        print(f"Processed {len(texts)} texts.")
        
        return embeddings
            

class QwenReranker:
    # Qwen Reranker에 필요한 상수 및 헬퍼 함수를 클래스 변수/메서드로 정의
    
    # 1. Qwen Reranker의 Prompt 정의
    # Instruction/Query/Document를 위한 시스템 프롬프트
    QWEN_RERANK_PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    # Assistant의 응답을 위한 접미사 (모델이 <think>를 건너뛰고 바로 "yes" 또는 "no"를 예측하도록 유도)
    QWEN_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    # 기본 태스크 지시 사항
    QWEN_RERANK_TASK = 'Given a web search query, retrieve relevant passages that answer the query'
    # 최대 길이
    MAX_LENGTH = 8192
    
    def __init__(self, model_name=None, device=None):
        """
        Initializes the Qwen reranker (Cross-Encoder) for fine-grained relevance scoring.
        """
        # Use local model 
        if model_name is None:
            # Qwen Reranker 경로로 수정
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'Qwen3-Reranker-0.6B')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local Qwen reranker model from: {model_name}")
            else:
                model_name = "Qwen/Qwen3-Reranker-0.6B"
                print(f"Local model not found. Using remote Qwen reranker: {model_name}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Qwen reranker on device: {self.device}")
        
        # Qwen Reranker는 AutoModelForCausalLM을 사용합니다.
        # 주의: Qwen reranker는 패딩을 'left'로 설정하는 것이 권장됩니다.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reranker 로직에 필요한 토큰 ID를 미리 저장
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # Prefix/Suffix 토큰 인코딩
        self.prefix_tokens = self.tokenizer.encode(self.QWEN_RERANK_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.QWEN_RERANK_SUFFIX, add_special_tokens=False)

    def _format_instruction(self, instruction, query, doc):
        """쿼리-문서 쌍에 Qwen의 Instruct 프롬프트를 적용합니다."""
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, 
            query=query, 
            doc=doc
        )
        return output

    def _process_inputs(self, pairs):
        """주어진 프롬프트 쌍에 Prefix/Suffix를 추가하고 패딩을 처리합니다."""
        
        # 1. 쿼리-문서 쌍 토큰화 (특수 토큰/패딩 없이)
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, # 나중에 패딩 시 생성됨
            max_length=self.MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        # 2. Prefix/Suffix 토큰 추가
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        
        # 3. 패딩 및 텐서 변환 (padding_side='left'는 __init__에서 설정됨)
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.MAX_LENGTH)
        
        # 4. 디바이스 이동
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
            
        return inputs

    def _compute_logits(self, inputs):
        """모델을 실행하고 'yes' 로짓에서 최종 점수를 계산합니다."""
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 마지막 토큰의 로짓만 사용
        batch_scores = outputs.logits[:, -1, :]
        
        # 'true'와 'false' 토큰의 로짓 추출
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        
        # 두 로짓만 스택하고 Log Softmax를 적용하여 확률처럼 만듭니다.
        # Log Softmax(False, True) -> [Prob(False), Prob(True)]
        stacked_logits = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(stacked_logits, dim=1)
        
        # 'True' 클래스(인덱스 1)의 지수(exp)를 취하여 확률(Score)을 얻습니다.
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self, query_text, passages, passage_ids, top_k=20):
        """
        Rerank the passages using Qwen reranker (Cross-Encoder).
        기존 BGE reranker의 배치 루프 구조를 유지합니다.
        """
        if not passages:
            return []
        
        scores = []
        # Conservative batch size는 유지
        batch_size = 4 
        
        total_batches = (len(passages) + batch_size - 1) // batch_size
        
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i : i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # 1. Qwen 프롬프트 형식으로 쌍을 구성
                pairs = [
                    self._format_instruction(self.QWEN_RERANK_TASK, query_text, doc) 
                    for doc in batch_passages
                ]
                
                # 2. 입력 처리 (토크나이징, Prefix/Suffix 추가, 패딩, 디바이스 이동)
                inputs = self._process_inputs(pairs)
                
                # 3. 로짓 계산 및 점수 변환
                batch_scores = self._compute_logits(inputs)
                
                scores.extend(batch_scores)
                
                # Cleanup
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in reranking batch {batch_num}/{total_batches}: {e}")
                # Fallback: Use neutral scores for this batch
                fallback_scores = [0.5] * len(batch_passages)
                scores.extend(fallback_scores)
        
        # Combine results and sort by reranking score
        results = list(zip(passage_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


retriever = None
reranker = None
corpus_texts = {} 

def preprocess(corpus_dict):
    """
    PREPROCESSING: Initializes the QwenRetriever + QwenReranker Pipeline.
    
    1. Initializes QwenRetriever (for passage embeddings) and QwenReranker.
    2. Computes and stores Qwen embeddings for the entire corpus.
    """
    global retriever, reranker, corpus_texts
    
    print("=" * 60)
    print("PREPROCESSING: Initializing Qwen Retriever + Qwen Reranker Pipeline...") 
    print("=" * 60)
    
    # Set GPU memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("Loading Qwen retriever (for passage embeddings)...")
    retriever = QwenRetriever()
    
    print("Loading Qwen reranker...")
    reranker = QwenReranker()
    
    print(f"Preparing corpus with {len(corpus_dict)} documents...")
    
    # Store corpus IDs, passages, and original texts
    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]
    
    # Store original texts for reranking
    corpus_texts = {doc_id: passages[i] for i, doc_id in enumerate(retriever.corpus_ids)}
    
    # Compute embeddings with conservative batch size for retrieval
    print("Computing Qwen corpus embeddings...")
    retriever.corpus_embeddings = retriever.embed_texts(passages, is_query=False, batch_size=4)
    
    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated Qwen embeddings for {len(retriever.corpus_ids)} documents")
    print(f"✓ Embedding matrix shape: {retriever.corpus_embeddings.shape}")
    
    return {
        'retriever': retriever,
        'reranker': reranker,
        'corpus_ids': retriever.corpus_ids,
        'corpus_embeddings': retriever.corpus_embeddings,
        'corpus_texts': corpus_texts,
        'num_documents': len(corpus_dict)
    }

def predict(query, preprocessed_data):
    """
    Two-stage prediction: Qwen retrieval + Qwen reranking.
    
    Input/Output format remains the same.
    """
    global retriever, reranker, corpus_texts
    
    # Extract query text
    query_text = query.get('query', '')
    if not query_text:
        return []
    
    # Use global instances or get from preprocessed_data
    if retriever is None:
        retriever = preprocessed_data.get('retriever')
        reranker = preprocessed_data.get('reranker')
        corpus_texts = preprocessed_data.get('corpus_texts', {})
        
        if retriever is None or reranker is None:
            print("Error: Missing retriever or reranker in preprocessed data")
            return []
    
    try:
        # STAGE 1: Qwen Retrieval (get top 100 candidates)
        print("Stage 1: Qwen retrieval...") 
        
        query_embedding = retriever.embed_texts([query_text], is_query=True, batch_size=1)
        
        # Compute cosine similarity with precomputed corpus embeddings
        qwen_scores = cosine_similarity(query_embedding, retriever.corpus_embeddings)[0]
        
        # Get top 100 candidates for reranking
        top_100_indices = np.argsort(qwen_scores)[::-1][:100]
        # Get passages and IDs for reranking
        candidate_ids = [retriever.corpus_ids[idx] for idx in top_100_indices]
        candidate_passages = [corpus_texts.get(doc_id, '') for doc_id in candidate_ids]
        
        # STAGE 2: Qwen Reranking (rerank top 100 -> top 20)
        print("Stage 2: Qwen reranking...")
        reranked_results = reranker.rerank(
            query_text, 
            candidate_passages, 
            candidate_ids, 
            top_k=20
        )
        
        # Build final results with ACTUAL reranking scores
        results = []
        for rank, (passage_id, rerank_score) in enumerate(reranked_results):
            results.append({
                'paragraph_uuid': passage_id,
                'score': float(rerank_score) 
            })
        
        print(f"✓ Returned {len(results)} results with reranker scores")
        return results
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Fallback to Qwen-only retrieval with Qwen scores
        try:
            print("Falling back to Qwen-only retrieval...")
            
            query_embedding = retriever.embed_texts([query_text], is_query=True, batch_size=1)
            qwen_scores = cosine_similarity(query_embedding, retriever.corpus_embeddings)[0]
            top_indices = np.argsort(qwen_scores)[::-1][:20]
            
            results = []
            for idx in top_indices:
                results.append({
                    'paragraph_uuid': retriever.corpus_ids[idx],
                    'score': float(qwen_scores[idx]) 
                })
            
            return results
        except:
            return []