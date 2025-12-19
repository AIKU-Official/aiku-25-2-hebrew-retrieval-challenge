import json
import torch
import numpy as np
from bm25s import BM25
from datetime import datetime

from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification

"""
    Retriever B : 
        - BM25 (tokenizer 는 bge-reranker-v2-m3 내부 tokenizer 사용)
        - multilingual-e5-large 
        각 두 모델의 top 100 의 합집합을 reranker 로 전달. 그럼 100 ~  200 개의 passage 가 reranker 로 전달되겠지.
    Reranker : BGE reranker 그대로.
"""


class E5LargeRetriever:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the  E5 retriever using the multilingual  E5 base model.
        """
        # Use local model
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'multilingual-e5-large')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local  E5 model from: {model_name}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Loading  E5 multilingual model on device: {self.device}")
        print("AutoTokenizer.from_pretrained() starts...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device) starts...")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        print("model.eval() starts...")
        self.model.eval()
        
        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.corpus_ids = []
        self.corpus_embeddings = None

    def embed_texts(self, texts, is_query=False, batch_size=64): 
        """
        Generates embeddings for texts using  E5 model with proper prefixes.
         E5 requires specific prefixes for queries vs passages.
        """
        #  E5 model requires specific prefixes
        if is_query:
            # Add query prefix for  E5
            prefixed_texts = [f"query: {text.strip()}" for text in texts]
        else:
            # Add passage prefix for  E5
            prefixed_texts = [f"passage: {text.strip()}" for text in texts]

        all_embeddings = []
        total_batches = (len(prefixed_texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(prefixed_texts), batch_size):
            batch_num = i // batch_size + 1
            if not is_query and batch_num % 50 == 0:
                print(f"Processing batch {batch_num}/{total_batches} ({(batch_num/total_batches)*100:.1f}%)")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            batch_texts = prefixed_texts[i:i + batch_size]
            
            try:
                encoded = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    model_output = self.model(**encoded)
                    
                    #  E5 uses mean pooling with attention mask
                    attention_mask = encoded['attention_mask']
                    embeddings = model_output.last_hidden_state
                    
                    # Mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                    
                    # L2 normalize embeddings (important for  E5)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                # Move to CPU immediately
                all_embeddings.append(embeddings.cpu())
                
                # Clear GPU memory
                del encoded, model_output, embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM at batch {batch_num}, reducing batch size...")
                # Process one item at a time
                for single_text in batch_texts:
                    try:
                        encoded = self.tokenizer(
                            [single_text], 
                            padding=True, 
                            truncation=True, 
                            max_length=512,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        with torch.no_grad():
                            model_output = self.model(**encoded)
                            attention_mask = encoded['attention_mask']
                            embeddings = model_output.last_hidden_state
                            
                            # Mean pooling
                            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            embeddings = sum_embeddings / sum_mask
                            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                            
                        all_embeddings.append(embeddings.cpu())
                        
                        del encoded, model_output, embeddings
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e2:
                        print(f"Failed to process single text: {e2}")
                        #  E5-base has 768 dimensions
                        zero_embedding = torch.zeros(1, 768).float()
                        all_embeddings.append(zero_embedding)

        return torch.cat(all_embeddings, dim=0).numpy()

class BM25Retriever:
    def __init__(self, tokenizer_model_name=None):
        """
        BM25 Retriever를 초기화합니다.
        BGE M3 토크나이저를 사용하여 Subword 토큰화 및 정규화를 수행합니다.
        """
        tokenizer_model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'bge-reranker-v2-m3') if tokenizer_model_name is None else tokenizer_model_name
        print(f"Loading BM25 Tokenizer: {tokenizer_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        self.bm25_model = BM25()
        self.corpus_texts = {} # 문서 ID와 원본 텍스트 매핑
        self.corpus_ids = []   # 문서 ID 리스트
        
        # BGE M3 토크나이저에서 CLS, SEP 토큰 ID를 가져와 토큰화 시 제외합니다.
        # 대부분의 BPE/SentencePiece 기반 모델은 특수 토큰을 사용합니다.
        self.special_token_ids = self.tokenizer.all_special_ids
        
        print("BM25 Retriever initialized.")

    def _tokenize_hebrew_subwords(self, text):
        """
        BGE M3 토크나이저를 사용하여 Subword 토큰을 추출하고 특수 토큰을 제거합니다.
        BM25 검색을 위한 토큰(문자열 리스트)을 반환합니다.
        """
        
        # 1. 토큰 ID 획득 및 특수 토큰 제거
        encoded = self.tokenizer(
            text, 
            truncation=True, 
            padding=False,
            return_tensors='pt'
        )['input_ids'].squeeze(0).tolist()
        
        # 특수 토큰 제거 (CLS, SEP, UNK 등)
        token_ids = [id for id in encoded if id not in self.special_token_ids]
        
        # 2. 토큰 ID를 다시 Subword 문자열로 변환
        # BPE 토큰에는 단어 경계를 나타내는 ' ' (U+2581) 기호가 포함될 수 있습니다.
        subword_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # (선택 사항: 영어가 아닌 경우 불용어 제거 로직 추가 가능)
        
        return subword_tokens

    def index_corpus(self, corpus_dict):
        """
        코퍼스를 토큰화하고 BM25 인덱스를 구축합니다.
        """
        self.corpus_ids = list(corpus_dict.keys())
        passages = []
        
        for doc_id, doc in corpus_dict.items():
            text = doc.get('passage', doc.get('text', ''))
            passages.append(text)
            self.corpus_texts[doc_id] = text # 텍스트 저장

        print(f"Tokenizing {len(self.corpus_ids)} documents using BGE M3 tokenizer...")
        
        # 코퍼스 토큰화
        tokenized_corpus = [self._tokenize_hebrew_subwords(doc) for doc in tqdm(passages, desc="Tokenizing")]
        
        # BM25 인덱스 구축
        print("Building BM25 index...")
        self.bm25_model.index(tokenized_corpus)
        
        print(f"✓ BM25 index built with {len(self.corpus_ids)} documents.")

    def retrieve(self, query_text, top_k=100):
        """
        쿼리를 BM25 인덱스로 검색하고 상위 K개 문서의 ID와 점수를 반환합니다.
        """
        # 1. 쿼리 토큰화
        tokenized_query = self._tokenize_hebrew_subwords(query_text)
        
        # -----------------------------------------------------------------------
        # [수정 필요 부분]: bm25s는 쿼리 리스트를 기대합니다.
        # 단일 쿼리를 검색하더라도 [tokenized_query] 형태로 리스트로 감싸야 합니다.
        tokenized_queries = [tokenized_query] 
        # -----------------------------------------------------------------------

        # 2. BM25 검색 실행
        # retrieve() 메서드는 (인덱스, 점수) 튜플을 반환합니다.
        # 인덱스는 코퍼스 리스트의 위치에 해당합니다.
        
        # 수정: tokenized_query 대신 tokenized_queries (리스트의 리스트) 전달
        doc_indices, scores = self.bm25_model.retrieve(tokenized_queries, k=top_k) 
        
        # *********** 기존 코드와 동일 (doc_indices와 scores가 쿼리 수만큼 래핑되어 있음) ***********
        # BM25Retriever.retrieve는 단일 쿼리만 처리하므로 [0] 인덱스를 사용하여 래핑을 해제합니다.
        
        results = []
        for index, score in zip(doc_indices.tolist()[0], scores.tolist()[0]): 
             # 인덱스를 문서 ID로 변환
            doc_id = self.corpus_ids[index]
            doc_text = self.corpus_texts[doc_id]
            
            results.append({
                'paragraph_uuid': doc_id,
                'score': float(score),
                'text': doc_text # 나중에 reranker에 전달하기 위해 텍스트도 포함
            })
            
        return results

class BGEReranker:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the BGE reranker for fine-grained relevance scoring.
        """
        # Use local model 
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'bge-reranker-v2-m3')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local BGE model from: {model_name}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading BGE reranker on device: {self.device}")
        
        # BGE reranker is actually a special model type
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def rerank(self, query_text, passages, passage_ids, top_k=20):
        """
        Rerank the passages using BGE reranker - CORRECTED VERSION.
        """
        if not passages:
            return []
        
        scores = []
        batch_size = 64  # Conservative batch size
        
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i : i + batch_size]
            
            try:
                # BGE reranker expects SEPARATE query and passage inputs
                # NOT concatenated strings
                batch_queries = [query_text] * len(batch_passages)
                
                # Tokenize query-passage pairs properly
                with torch.no_grad():
                    inputs = self.tokenizer(
                        batch_queries,
                        batch_passages,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # Get relevance scores from sequence classification model
                    outputs = self.model(**inputs)
                    
                    # BGE reranker outputs logits for relevance classification
                    logits = outputs.logits
                    
                    # Handle different output shapes
                    if len(logits.shape) == 1:
                        # Single score per pair
                        batch_scores = logits.cpu().numpy()
                    elif logits.shape[1] == 1:
                        # Single column output
                        batch_scores = logits.squeeze(-1).cpu().numpy()
                    else:
                        # Binary classification - take positive class (index 1)
                        batch_scores = logits[:, 1].cpu().numpy()
                
                scores.extend(batch_scores.tolist())
                
                # Cleanup
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in reranking batch {i//batch_size + 1}: {e}")
                # Fallback: Use neutral scores for this batch
                fallback_scores = [0.5] * len(batch_passages)
                scores.extend(fallback_scores)
        
        # Combine results and sort by reranking score
        results = list(zip(passage_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]

# Global instances
retriever = None
reranker = None
corpus_texts = {}  # Store original passage texts for reranking

def preprocess(corpus_dict):
    """
    Preprocessing function using E5 multilingual model + BGE reranker.
    
    Input: corpus_dict - dict mapping document IDs to document objects with 'passage'/'text' field
    Output: dict containing initialized models, embeddings, and corpus data
    
    Note: Uses global variables (retriever, reranker, corpus_texts) for efficiency,
    but also returns all required data via preprocessed_data for function interface.
    """
    global e5_retriever, reranker, bm25_retriever, corpus_texts
    start_time = datetime.now()

    print("=" * 60)
    print("PREPROCESSING: Initializing E5 , BM25 Pipeline...")
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("=" * 60)
    
    # Set GPU memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize E5 retriever
    print("Loading E5 retriever...")
    e5_retriever  = E5LargeRetriever()
    
    # Initialize BGE reranker   
    print("Loading BGE reranker...")
    reranker = BGEReranker()
    

    bm25_retriever = BM25Retriever()
    bm25_retriever.index_corpus(corpus_dict) # BM25 인덱싱 수행




    print(f"Preparing corpus with {len(corpus_dict)} documents...")
    
    e5_retriever.corpus_ids = bm25_retriever.corpus_ids # ID 리스트 통일
    passages = [bm25_retriever.corpus_texts[doc_id] for doc_id in e5_retriever.corpus_ids]
    corpus_texts = bm25_retriever.corpus_texts # 텍스트 저장소 통일
    e5_retriever.corpus_embeddings = e5_retriever.embed_texts(passages, is_query=False, batch_size=64)

    end_time = datetime.now()
    duration = end_time - start_time
    
    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated embeddings for {len(e5_retriever.corpus_ids)} documents")
    print(f"✓ Embedding matrix shape: {e5_retriever.corpus_embeddings.shape}")
    print("=" * 60)
    print(f"완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {duration.total_seconds():.0f}초 ({duration.total_seconds()//60:.0f}분 {duration.total_seconds()%60:.0f}초)")
    print("=" * 60)
    
    return {
        'e5_retriever': e5_retriever,
        'bm25_retriever': bm25_retriever,
        'reranker': reranker,
        'corpus_ids': e5_retriever.corpus_ids,
        'corpus_embeddings': e5_retriever.corpus_embeddings,
        'corpus_texts': corpus_texts,
        'num_documents': len(corpus_dict)
    }
def predict(query, preprocessed_data):
    """
    Hybrid Prediction: (BM25 + KnessetE5) Retrieval -> Reranking
    """
    global e5_retriever, bm25_retriever, reranker, corpus_texts
    
    query_text = query.get('query', '')
    
    # ... (생략: 전역 변수 로딩 및 에러 처리) ...

    # STAGE 1: Hybrid Retrieval (BM25 + Dense) - 상위 200개 후보 추출 (예시)
    # ----------------------------------------------------------------------
    TOP_K_CANDIDATES = 200
    
    # 1. BM25 Retrieval (Sparse) - 100개
    print("Stage 1a: BM25 retrieval...")
    bm25_results = bm25_retriever.retrieve(query_text, top_k=100)
    
    # 2. E5 Retrieval (Dense) - 100개
    print("Stage 1b: E5 retrieval...")
    query_embedding = e5_retriever.embed_texts([query_text], is_query=True, batch_size=1)
    e5_scores = cosine_similarity(query_embedding, e5_retriever.corpus_embeddings)[0]
    e5_top_indices = np.argsort(e5_scores)[::-1][:100]
    
    e5_results = []
    for idx in e5_top_indices:
        doc_id = e5_retriever.corpus_ids[idx]
        e5_results.append({
            'paragraph_uuid': doc_id,
            'score': float(e5_scores[idx]),
            'text': corpus_texts[doc_id]
        })
    
    # 3. 결과 통합 (Reciprocal Rank Fusion - RRF)
    print("Stage 1c: Fusing results with Reciprocal Rank Fusion...")
    fused_scores = {}
    k = 60  # RRF constant

    # Process BM25 results
    for rank, res in enumerate(bm25_results):
        doc_id = res['paragraph_uuid']
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {'score': 0, 'text': res['text']}
        fused_scores[doc_id]['score'] += 1 / (k + rank + 1)

    # Process E5 results
    for rank, res in enumerate(e5_results):
        doc_id = res['paragraph_uuid']
        if doc_id not in fused_scores:
            fused_scores[doc_id] = {'score': 0, 'text': res['text']}
        fused_scores[doc_id]['score'] += 1 / (k + rank + 1)

    # Sort candidates by the new RRF score
    sorted_candidates = sorted(fused_scores.items(), key=lambda item: item[1]['score'], reverse=True)

    # Get the top candidates for reranking
    candidate_ids = [doc_id for doc_id, data in sorted_candidates[:TOP_K_CANDIDATES]]
    candidate_passages = [data['text'] for doc_id, data in sorted_candidates[:TOP_K_CANDIDATES]]
    
    # STAGE 2: BGE Reranking (rerank candidates -> top 20)
    # ----------------------------------------------------------------------
    print(f"Stage 2: BGE reranking {len(candidate_ids)} candidates...")
    reranked_results = reranker.rerank(
        query_text, 
        candidate_passages, 
        candidate_ids, 
        top_k=20
    )
    
    # ... (후략: 결과 빌드 및 반환) ...
    results = []
    for rank, (passage_id, rerank_score) in enumerate(reranked_results):
        results.append({
            'paragraph_uuid': passage_id,
            'score': float(rerank_score)
        })
    
    print(f"✓ Returned {len(results)} final results with reranker scores")
    return results