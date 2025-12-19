import json
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class QwenRetriever:
    """
    Qwen3-Embedding-4B 모델을 사용하여 쿼리와 코퍼스(Passage) 모두의 임베딩을 생성합니다.
    ( 2560D 임베딩 차원을 사용합니다.)
    """
    def __init__(self, model_name=None, device=None):
        """
        Initializes the Qwen3-Embedding-4B model.
        """
        # Use local model 
        if model_name is None:
            # Qwen 모델 경로를 지정합니다.
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'Qwen3-Embedding-4B')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local Qwen model from: {model_name}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Loading Qwen3-Embedding-4B model on device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, # 메모리 효율을 위해 float16 사용
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.corpus_ids = []
        self.corpus_embeddings = None

    # Qwen은 Qwen3 달리 is_query 여부에 따른 접두사가 필요 없으므로, is_query 파라미터를 제거했습니다.
    def embed_texts(self, texts, batch_size=16): 
        """
        Generates embeddings for texts using Qwen model (no prefixes needed).
        """
        # Qwen은 Qwen3 달리 접두사(prefix)를 사용하지 않습니다.
        # texts는 이미 전처리된 원본 텍스트 리스트입니다.
        prefixed_texts = [text.strip() for text in texts]

        all_embeddings = []
        total_batches = (len(prefixed_texts) + batch_size - 1) // batch_size
        
        # is_query 여부를 알 수 없으므로, 모든 텍스트에 대해 진행률을 표시하지 않습니다. (선택적)
        for i in range(0, len(prefixed_texts), batch_size):
            batch_num = i // batch_size + 1
            # 코퍼스 임베딩 시에만 진행률을 보고하도록 preprocess에서 제어합니다.

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
                    
                    # Qwen uses mean pooling (Qwen3 유사한 Mean pooling 구현)
                    attention_mask = encoded['attention_mask']
                    embeddings = model_output.last_hidden_state
                    
                    # Mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                    
                    # L2 normalize embeddings (Qwen에서도 중요)
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
                        # Qwen3-Embedding-4B는 2560D
                        zero_embedding = torch.zeros(1, 2560).float()
                        all_embeddings.append(zero_embedding)

        return torch.cat(all_embeddings, dim=0).numpy()



class QwenRetriever:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the Qwen3-Embedding-4B model for high-quality query embedding.
        """
        # Use local model 
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'Qwen3-Embedding-4B')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local Qwen model from: {model_name}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Qwen3-Embedding-4B model on device: {self.device}")
        
        # Qwen3-Embedding-4B uses AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, # Use float16 for memory efficiency
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Qwen embedding method (simplified since it's only used for the query)
    def embed_query(self, query_text):
        """
        Generates a single embedding for the query text using Qwen model.
        Qwen does NOT require 'query:' or 'passage:' prefixes.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                [query_text], 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Qwen uses mean pooling, similar to Qwen3's implementation
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean pooling implementation
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
            # L2 normalize embeddings (important for Qwen and Qwen3)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy()


# Global instances - RENAMED 'reranker' to 'qwen_model'
retriever = None
qwen_model = None
corpus_texts = {} 

# --- Preprocessing Function (수정됨) ---
def preprocess(corpus_dict):
    """
    Prepares the Qwen3-Qwen Dual-Encoder pipeline.
    1. Initializes Qwen3 Retriever and Qwen Model.
    2. Computes and stores Qwen3 embeddings for the entire corpus.
    Qwen is only used for query-side embedding during prediction.
    """
    global retriever, qwen_model, corpus_texts
    
    print("=" * 60)
    print("PREPROCESSING: Initializing Qwen3 + Qwen3-Embedding-4B Pipeline...")
    print("=" * 60)
    
    # Set GPU memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize Qwen3 retriever (for passage embeddings)
    print("Loading Qwen3 retriever (for passage embeddings)...")
    retriever = QwenRetriever()
    
    # Initialize Qwen model (for query embedding) - Renamed from 'reranker'
    print("Loading Qwen model (for query embedding)...")
    qwen_model = QwenRetriever()
    
    print(f"Preparing corpus with {len(corpus_dict)} documents...")
    
    # Store corpus IDs, passages, and original texts
    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]
    
    # Store original texts (kept for possible future use, though not needed for this dual-encoder approach)
    corpus_texts = {doc_id: passages[i] for i, doc_id in enumerate(retriever.corpus_ids)}
    
    # Compute Qwen3 embeddings for the corpus
    print("Computing Qwen3 corpus embeddings...")
    retriever.corpus_embeddings = retriever.embed_texts(passages, is_query=False, batch_size=16)
    
    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated Qwen3 embeddings for {len(retriever.corpus_ids)} documents")
    print(f"✓ Embedding matrix shape: {retriever.corpus_embeddings.shape}")
    
    return {
        'retriever': retriever,
        'qwen_model': qwen_model, # Changed key name
        'corpus_ids': retriever.corpus_ids,
        'corpus_embeddings': retriever.corpus_embeddings,
        'corpus_texts': corpus_texts,
        'num_documents': len(corpus_dict)
    }

# --- Predict Function (크게 수정됨) ---
def predict(query, preprocessed_data):

    global retriever, qwen_model
    
    # Extract query text
    query_text = query.get('query', '')
    if not query_text:
        return []
    
    # Use global instances or get from preprocessed_data
    if retriever is None:
        retriever = preprocessed_data.get('retriever')
        qwen_model = preprocessed_data.get('qwen_model') # Changed key name
        
        if retriever is None or qwen_model is None:
            print("Error: Missing retriever or Qwen model in preprocessed data")
            return []
    
    # Check for precomputed corpus embeddings
    if retriever.corpus_embeddings is None:
        print("Error: Qwen3 corpus embeddings not computed.")
        return []
        
    # --- Dual-Encoder Retrieval ---
    try:
        # STAGE 1: Qwen Query Embedding
        print("Stage 1: Qwen query embedding...")
        # Use Qwen to get the query embedding
        qwen_query_embedding = qwen_model.embed_query(query_text)
        
        # STAGE 2: Cosine Similarity (Qwen Query vs Qwen3 Passages)
        # Note: This is an unusual combination (Qwen query, Qwen3 passage) but is a viable dual-encoder strategy.
        # If the passages were embedded with Qwen, the retrieval would be Qwen vs Qwen.
        # For a true Reranker replacement, we stick to the 2-stage idea.
        print("Stage 2: Cosine similarity (Qwen Query vs Qwen3 Passages)...")
        
        # Compute cosine similarity
        # Use Qwen query embedding against Qwen3 corpus embeddings
        dual_encoder_scores = cosine_similarity(qwen_query_embedding, retriever.corpus_embeddings)[0]
        
        # Get top 20 results based on the dual-encoder score
        top_indices = np.argsort(dual_encoder_scores)[::-1][:20]
        
        # Build final results with the dual-encoder scores
        results = []
        for idx in top_indices:
            results.append({
                'paragraph_uuid': retriever.corpus_ids[idx],
                'score': float(dual_encoder_scores[idx]) # Use actual dual-encoder cosine similarity score
            })
        
        print(f"✓ Returned {len(results)} results with Dual-Encoder scores")
        return results
        
    except Exception as e:
        print(f"Error in Qwen Dual-Encoder prediction: {e}")
        # Fallback: Fallback to Qwen3-only retrieval (Qwen3 Query vs Qwen3 Passages)
        try:
            print("Falling back to Qwen3-only retrieval...")
            Qwen3_query_embedding = retriever.embed_texts([query_text], is_query=True, batch_size=1)
            Qwen3_scores = cosine_similarity(Qwen3_query_embedding, retriever.corpus_embeddings)[0]
            top_indices = np.argsort(Qwen3_scores)[::-1][:20]
            
            results = []
            for idx in top_indices:
                results.append({
                    'paragraph_uuid': retriever.corpus_ids[idx],
                    'score': float(Qwen3_scores[idx]) 
                })
            
            print("✓ Returned results using Qwen3-only scores")
            return results
        except Exception as e_fallback:
            print(f"Error in Qwen3 fallback: {e_fallback}")
            return []