import json
import torch
import torch.nn as nn # New import for the Linear layer
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class E5Retriever:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the E5 retriever using the multilingual E5 base model.
        """
        # Use local model
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'multilingual-e5-base')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local E5 model from: {model_name}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Loading E5 multilingual model on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.model.eval()
        
        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.corpus_ids = []
        self.corpus_embeddings = None

    def embed_texts(self, texts, is_query=False, batch_size=32): 
        """
        Generates embeddings for texts using E5 model with proper prefixes.
        E5 requires specific prefixes for queries vs passages.
        """
        # E5 model requires specific prefixes
        if is_query:
            # Add query prefix for E5
            prefixed_texts = [f"query: {text.strip()}" for text in texts]
        else:
            # Add passage prefix for E5
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
                    
                    # E5 uses mean pooling with attention mask
                    attention_mask = encoded['attention_mask']
                    embeddings = model_output.last_hidden_state
                    
                    # Mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                    
                    # L2 normalize embeddings (important for E5)
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
                        # E5-base has 768 dimensions
                        zero_embedding = torch.zeros(1, 768).float()
                        all_embeddings.append(zero_embedding)

        return torch.cat(all_embeddings, dim=0).numpy()

class QwenRetriever:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the Qwen3-Embedding-4B model for high-quality query embedding 
        and adds a linear layer to project the 2560-dim embedding to 768-dim.
        """
        # Use local model 
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'Qwen3-Embedding-4B')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local Qwen model from: {model_name}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Qwen3-Embedding-4B model on device: {self.device}")
        

        # Qwen3-Embedding-4B uses AutoModel and outputs 2560 dimensions
        self.input_embed_dim = 2560
        self.output_embed_dim = 768 # Target dimension to match E5
        


        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, # Use float16 for memory efficiency
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        # ⭐ Instead of linear projection, use truncation: Qwen's 2560-dim embedding will be truncated to 768-dim.
        # No linear layer is needed; truncation will be done in embed_query().
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Qwen embedding method (simplified since it's only used for the query)
    def embed_query(self, query_text):
        """
        Generates a single embedding for the query text using Qwen model,
        then projects it from 2560 dimensions down to 768 dimensions.
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
            
            # Qwen uses mean pooling, similar to E5's implementation
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean pooling implementation (results in [1, 2560] tensor)
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask # [1, 2560]
            
            #  ADDED: Linear projection (2560 -> 768)
            embedding = self.linear_projection(embedding) # [1, 768]
            
            # L2 normalize embeddings (important for Qwen and E5)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1) # [1, 768]
            
            return embedding.cpu().numpy()

# Global instances - RENAMED 'reranker' to 'qwen_model'
retriever = None
qwen_model = None
corpus_texts = {} 

# --- Preprocessing Function (수정됨) ---
def preprocess(corpus_dict):
    """
    Prepares the E5-Qwen Dual-Encoder pipeline.
    1. Initializes E5 Retriever and Qwen Model.
    2. Computes and stores E5 embeddings for the entire corpus.
    Qwen is only used for query-side embedding during prediction.
    """
    global retriever, qwen_model, corpus_texts
    
    print("=" * 60)
    print("PREPROCESSING: Initializing E5 + Qwen3-Embedding-4B Pipeline...")
    print("=" * 60)
    
    # Set GPU memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize E5 retriever (for passage embeddings)
    print("Loading E5 retriever (for passage embeddings)...")
    retriever = E5Retriever()
    
    # Initialize Qwen model (for query embedding) - Renamed from 'reranker'
    print("Loading Qwen model (for query embedding)...")
    qwen_model = QwenRetriever()
    
    print(f"Preparing corpus with {len(corpus_dict)} documents...")
    
    # Store corpus IDs, passages, and original texts
    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]
    
    # Store original texts (kept for possible future use, though not needed for this dual-encoder approach)
    corpus_texts = {doc_id: passages[i] for i, doc_id in enumerate(retriever.corpus_ids)}
    
    # Compute E5 embeddings for the corpus
    print("Computing E5 corpus embeddings...")
    retriever.corpus_embeddings = retriever.embed_texts(passages, is_query=False, batch_size=32)
    
    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated E5 embeddings for {len(retriever.corpus_ids)} documents")
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
    """
    Dual-Encoder prediction: E5 Passage Embeddings + Qwen Query Embedding + Cosine Similarity.
    
    Input: 
    - query: dict with 'query' field containing query text
    - preprocessed_data: dict from preprocess() containing models and corpus data
    
    Output: list of dicts with 'paragraph_uuid' and 'score' fields, ranked by relevance
    """
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
        print("Error: E5 corpus embeddings not computed.")
        return []
        
    # --- Dual-Encoder Retrieval ---
    try:
        # STAGE 1: Qwen Query Embedding
        print("Stage 1: Qwen query embedding...")
        # Use Qwen to get the query embedding
        qwen_query_embedding = qwen_model.embed_query(query_text)
        
        # STAGE 2: Cosine Similarity (Qwen Query vs E5 Passages)
        # Note: This is an unusual combination (Qwen query, E5 passage) but is a viable dual-encoder strategy.
        # If the passages were embedded with Qwen, the retrieval would be Qwen vs Qwen.
        # For a true Reranker replacement, we stick to the 2-stage idea.
        print("Stage 2: Cosine similarity (Qwen Query vs E5 Passages)...")
        
        # Compute cosine similarity
        # Use Qwen query embedding against E5 corpus embeddings
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
        # Fallback: Fallback to E5-only retrieval (E5 Query vs E5 Passages)
        try:
            print("Falling back to E5-only retrieval...")
            e5_query_embedding = retriever.embed_texts([query_text], is_query=True, batch_size=1)
            e5_scores = cosine_similarity(e5_query_embedding, retriever.corpus_embeddings)[0]
            top_indices = np.argsort(e5_scores)[::-1][:20]
            
            results = []
            for idx in top_indices:
                results.append({
                    'paragraph_uuid': retriever.corpus_ids[idx],
                    'score': float(e5_scores[idx]) 
                })
            
            print("✓ Returned results using E5-only scores")
            return results
        except Exception as e_fallback:
            print(f"Error in E5 fallback: {e_fallback}")
            return []