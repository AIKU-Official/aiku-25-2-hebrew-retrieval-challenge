import json
import torch
from datetime import datetime
import numpy as np
import faiss
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity

"""
    Retriever : GiliGold/Knesset-multi-e5-large
    Reranker : BGE reranker 그대로.
"""

class KnessetE5LargeRetriever:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the KnessetE5 retriever using the multilingual KnessetE5 base model.
        """
        # Use local model
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'Knesset-multi-e5-large')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local KnessetE5 model from: {model_name}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Loading KnessetE5 multilingual model on device: {self.device}")
        print("AutoTokenizer.from_pretrained() starts...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("AutoModel.from_pretrained(model_name, quantization_config=... ).to(self.device) starts...")
        
        self.model = AutoModel.from_pretrained(model_name,torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        # 양자화...
        # self.model = AutoModel.from_pretrained(
        #     model_name, 
        #     #torch_dtype=torch.float16
        #     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        #     device_map='cuda:0' # 자동 GPU 로드
        # )
        print("model.eval() starts...")
        self.model.eval()
        
        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.corpus_ids = []
        self.corpus_embeddings = None

    def embed_texts(self, texts, is_query=False, batch_size=64): 
        """
        Generates embeddings for texts using KnessetE5 model with proper prefixes.
        KnessetE5 requires specific prefixes for queries vs passages.
        """
        # KnessetE5 model requires specific prefixes
        if is_query:
            # Add query prefix for KnessetE5
            prefixed_texts = [f"query: {text.strip()}" for text in texts]
        else:
            # Add passage prefix for KnessetE5
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
                    
                    # KnessetE5 uses mean pooling with attention mask
                    attention_mask = encoded['attention_mask']
                    embeddings = model_output.last_hidden_state
                    
                    # Mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                    
                    # L2 normalize embeddings (important for KnessetE5)
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
                        # KnessetE5-base has 768 dimensions
                        zero_embedding = torch.zeros(1, 768).float()
                        all_embeddings.append(zero_embedding)

        return torch.cat(all_embeddings, dim=0).numpy()
 
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
        from transformers import AutoModelForSequenceClassification
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.float16, trust_remote_code=True).to(self.device)

        # 양자화... 하면 .to(device) 필요 없음. 근데 성능이 깨져버리네...
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     model_name, 
        #     #torch_dtype=torch.float16,
        #     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        #     trust_remote_code=True,
        #     device_map='cuda:0' # 자동 GPU 로드
        # )
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
    Preprocessing function using KnessetE5 multilingual model + BGE reranker.
    
    Input: corpus_dict - dict mapping document IDs to document objects with 'passage'/'text' field
    Output: dict containing initialized models, embeddings, and corpus data
    
    Note: Uses global variables (retriever, reranker, corpus_texts) for efficiency,
    but also returns all required data via preprocessed_data for function interface.
    """
    global retriever, reranker, corpus_texts
    
    start_time = datetime.now()
    print("=" * 60)
    print("PREPROCESSING: Initializing KnessetE5 + BGE Reranker Pipeline...")
    print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("=" * 60)
    
    # Set GPU memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize KnessetE5 retriever
    print("Loading KnessetE5 retriever...")
    retriever = KnessetE5LargeRetriever()
    
    # Initialize BGE reranker   
    print("Loading BGE reranker...")
    reranker = BGEReranker()
    
    print(f"Preparing corpus with {len(corpus_dict)} documents...")
    
    # Store corpus IDs, passages, and original texts
    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]
    
    # Store original texts for reranking
    corpus_texts = {doc_id: passages[i] for i, doc_id in enumerate(retriever.corpus_ids)}
    
    # Compute embeddings with conservative batch size for retrieval
    print("Computing KnessetE5 embeddings...")
    retriever.corpus_embeddings = retriever.embed_texts(passages, is_query=False, batch_size=32)
    # Faiss Index 생성 및 GPU에 로드 (GPU가 있다면)
    D = retriever.corpus_embeddings.shape[1] # 임베딩 차원 (KnessetE5: 768)
    # Cosine Similarity는 L2 Distance를 사용하며, 벡터가 L2 정규화되어 있을 때 Inner Product와 동치
    index = faiss.IndexFlatIP(D) # Inner Product (Dot Product) Index 사용
    index.add(retriever.corpus_embeddings.astype('float32'))

    # GPU 사용 시:
    if retriever.device != 'cpu' and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index) # GPU 0번에 인덱스 로드
        print("✓ Faiss index successfully loaded onto GPU.")

    retriever.faiss_index = index # Faiss 인덱스를 retriever 객체에 저장

    end_time = datetime.now()
    duration = end_time - start_time
    
    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated embeddings for {len(retriever.corpus_ids)} documents")
    print(f"✓ Embedding matrix shape: {retriever.corpus_embeddings.shape}")
    print("=" * 60)
    print(f"완료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {duration.total_seconds():.0f}초 ({duration.total_seconds()//60:.0f}분 {duration.total_seconds()%60:.0f}초)")
    print("=" * 60)
    return {
        'retriever': retriever,
        'reranker': reranker,
        'corpus_ids': retriever.corpus_ids,
        'corpus_embeddings': retriever.corpus_embeddings,
        'corpus_texts': corpus_texts,
        'num_documents': len(corpus_dict),
        'faiss_index': retriever.faiss_index 
    }

def predict(query, preprocessed_data):
    """
    Two-stage prediction: KnessetE5 retrieval + BGE reranking.
    
    Input: 
    - query: dict with 'query' field containing query text
    - preprocessed_data: dict from preprocess() containing models and corpus data
    
    Output: list of dicts with 'paragraph_uuid' and 'score' fields, ranked by relevance
    
    Note: Uses global variables for efficiency but can also extract required data
    from preprocessed_data parameter for proper function interface.
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
        # STAGE 1: KnessetE5 Retrieval (get top 100 candidates)
        print("Stage 1: KnessetE5 retrieval...")
        query_embedding = retriever.embed_texts([query_text], is_query=True, batch_size=1)
        
        # Faiss 검색: D = 거리(유사도), I = 인덱스
        # top_k=100
        faiss_index = preprocessed_data.get('faiss_index')
        D, I = faiss_index.search(query_embedding.astype('float32'), 100) 

        # D는 유사도 점수 (Dot Product), I는 인덱스
        candidate_indices = I[0]
        candidate_scores = D[0] # 유사도 점수를 그대로 사용

        # Get passages and IDs for reranking
        candidate_ids = [retriever.corpus_ids[idx] for idx in candidate_indices]
        candidate_passages = [corpus_texts.get(doc_id, '') for doc_id in candidate_ids]
        
        # Compute cosine similarity with precomputed corpus embeddings
        KnessetE5_scores = cosine_similarity(query_embedding, retriever.corpus_embeddings)[0]
        # Get top 100 candidates for reranking
        top_100_indices = np.argsort(KnessetE5_scores)[::-1][:100]
        # Get passages and IDs for reranking
        candidate_ids = [retriever.corpus_ids[idx] for idx in top_100_indices]
        candidate_passages = [corpus_texts.get(doc_id, '') for doc_id in candidate_ids]
        
        # STAGE 2: BGE Reranking (rerank top 100 -> top 20)
        print("Stage 2: BGE reranking...")
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
                'score': float(rerank_score)  # Use actual BGE reranker score!
            })
        
        print(f"✓ Returned {len(results)} results with reranker scores")
        return results
        
    except Exception as e:
        try:
            print(f"Error in reranking: {e}. Falling back to Faiss retrieval scores.")
            
            results = []
            for i in range(min(20, len(candidate_ids))): 
                results.append({
                    'paragraph_uuid': candidate_ids[i],
                    'score': float(candidate_scores[i]) 
                })
            return results
        except:
            return []