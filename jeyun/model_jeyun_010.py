import json
import torch
import numpy as np
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity

"""
    Retriever : Qwen3-embedding-0.6B bi-encoder -> 0.35 정도 train validation 나옴
    Only.
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
        
        try:
            self.model = SentenceTransformer(
                model_name, 
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"}  
            )
        except Exception as e:
            print(f"Warning: Failed to load Qwen with specific args ({e}). Trying standard load.")
            # 표준 로드 시에도 device를 명시합니다.
            self.model = SentenceTransformer(model_name, device=self.device)
            
        self.model.eval()
        
        # 모델이 정확히 원하는 디바이스(cuda:0)에 있는지 확인하고 이동 (안정성 확보)
        if self.device.startswith('cuda'):
            self.model.to(self.device)
        
        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.corpus_ids = []
        self.corpus_embeddings = None

    def embed_texts(self, texts, is_query=False, batch_size=8): 
        """
        Generates embeddings for texts using Qwen3 model.
        """
        # Qwen models benefit from using the 'query' prompt for queries.
        prompt_name = "query" if is_query else None
        
        # Use SentenceTransformer's encode method
        # DataParallel 관련 인자가 없으므로, 모델이 로드된 단일 GPU를 사용합니다.
        embeddings = self.model.encode(
            sentences=texts,
            prompt_name=prompt_name, # Use the Qwen-specific prompt if available
            batch_size=batch_size,
            show_progress_bar=not is_query,
            convert_to_numpy=True,
            normalize_embeddings=True 
        )
        
        print(f"Processed {len(texts)} texts.")
        
        return embeddings
        
# --- 나머지 함수는 변경 사항 없음 ---
# (preprocess와 predict 함수는 단일 Bi-Encoder 검색 로직을 유지합니다.)

retriever = None
reranker = None
corpus_texts = {} 
def preprocess(corpus_dict):
    """
    PREPROCESSING: Initializes the QwenRetriever (Bi-Encoder) only.
    2. Computes and stores Qwen embeddings for the entire corpus.
    """
    global retriever
    start_time = datetime.now()
    print("=" * 60)
    print("PREPROCESSING: Initializing Qwen Retriever Pipeline (Bi-Encoder Only)...") 
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("=" * 60)
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("Loading Qwen retriever (for passage embeddings)...")
    retriever = QwenRetriever()
    
    # Reranker 로딩 제거
    
    print(f"Preparing corpus with {len(corpus_dict)} documents...")
    
    # Store corpus IDs
    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]
    
    # corpus_texts 저장 제거
    
    print("Computing Qwen corpus embeddings...")
    # 배치 사이즈를 64로 증가 (retriever.embed_texts 기본값 사용)
    retriever.corpus_embeddings = retriever.embed_texts(passages, is_query=False, batch_size=8) 
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated Qwen embeddings for {len(retriever.corpus_ids)} documents")
    print(f"✓ Embedding matrix shape: {retriever.corpus_embeddings.shape}")
    print("=" * 60)
    print(f"완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {duration.total_seconds():.0f}초 ({duration.total_seconds()//60:.0f}분 {duration.total_seconds()%60:.0f}초)")
    print("=" * 60)
    # 반환 객체에서 reranker 및 corpus_texts 제거
    return {
        'retriever': retriever,
        'corpus_ids': retriever.corpus_ids,
        'corpus_embeddings': retriever.corpus_embeddings,
        'num_documents': len(corpus_dict)
    }
def predict(query, preprocessed_data):
    """
    Single-stage prediction: Qwen retrieval (Bi-Encoder) only.
    Returns Top 20 results immediately based on cosine similarity score.
    """
    global retriever
    
    query_text = query.get('query', '')
    if not query_text:
        return []
    
    # Use global instances or get from preprocessed_data
    if retriever is None:
        retriever = preprocessed_data.get('retriever')
        
        if retriever is None:
            print("Error: Missing retriever in preprocessed data")
            return []
    
    try:
        # STAGE 1: Qwen Retrieval (get top 20 candidates directly)
        print("Stage 1: Qwen retrieval (Bi-Encoder Only)...") 
        
        # 1. 쿼리 임베딩 생성
        query_embedding = retriever.embed_texts([query_text], is_query=True, batch_size=1)
        
        # 2. 코사인 유사도 계산
        # compute cosine similarity is generally faster with numpy than torch for this final step
        qwen_scores = cosine_similarity(query_embedding, retriever.corpus_embeddings)[0]
        
        # 3. Top 20 인덱스 추출 (20개만 필요하므로 100 대신 20 사용)
        top_indices = np.argsort(qwen_scores)[::-1][:20]
        
        # 4. 최종 결과 구성 (점수는 Qwen 유사도 점수 사용)
        results = []
        for idx in top_indices:
            results.append({
                'paragraph_uuid': retriever.corpus_ids[idx],
                'score': float(qwen_scores[idx]) 
            })
        
        print(f"✓ Returned {len(results)} results with Bi-Encoder scores")

        return results
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return []