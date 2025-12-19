import json
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm # For progress bars during translation
import time
import sys 

"""
    Translator: english-hebrew - mt5-base 를finetuning 한 모델인듯 ... 
    나머지: baseline 과 동일. 
"""

class Translator:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the qwen3 translator model.
        """
        local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'Qwen3-0.6B')
        model_name = local_model_path
        print(f"Using local qwen3 model from: {model_name}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        print(f"Loading Qwen3 translator on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def translate(self, texts, src_lang="he", tgt_lang="en", batch_size=16, max_new_tokens=256):
        """
        Translates a list of texts from source to target language using Qwen3.
        """
        translated_texts = []
        
        # Prepare messages for the chat template
        all_messages = []
        for text in texts:
            prompt = f"Translate the following {src_lang} text to {tgt_lang}: \"{text.strip()}\""
            messages = [
                {"role": "system", "content": "You are a helpful assistant that translates text."},
                {"role": "user", "content": prompt}
            ]
            all_messages.append(messages)

        for i in tqdm(range(0, len(all_messages), batch_size), desc=f"Translating {src_lang.upper()} to {tgt_lang.upper()}"):
            batch_messages = all_messages[i:i + batch_size]
            
            try:
                prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]

                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=max_new_tokens,
                    )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]

                batch_translations = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                translated_texts.extend(batch_translations)

                del inputs, generated_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM during translation at batch {i//batch_size + 1}, processing one by one...")
                for single_messages in batch_messages:
                    try:
                        prompt = self.tokenizer.apply_chat_template(single_messages, tokenize=False, add_generation_prompt=True)
                        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

                        with torch.no_grad():
                            generated_ids = self.model.generate(
                                inputs.input_ids,
                                attention_mask=inputs.attention_mask,
                                max_new_tokens=max_new_tokens,
                            )
                        
                        generated_ids = [
                            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        
                        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        translated_texts.append(response)

                        del inputs, generated_ids
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e2:
                        print(f"Failed to translate single item: {e2}")
                        translated_texts.append("") # Append empty string on failure
            except Exception as e:
                print(f"Error during translation batch: {e}")
                translated_texts.extend([""] * len(batch_messages))

        return translated_texts


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

    def embed_texts(self, texts, is_query=False, batch_size=64): 
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
        
        for i in tqdm(range(0, len(prefixed_texts), batch_size), desc=f"Embedding {'queries' if is_query else 'passages'}"):
            # Removed the print statement, tqdm handles progress
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
                print(f"CUDA OOM at batch {i//batch_size + 1} during embedding, reducing batch size...")
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
                        print(f"Failed to process single text for embedding: {e2}")
                        # E5-base has 768 dimensions
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
        
        for i in tqdm(range(0, len(passages), batch_size), desc="Reranking"):
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
translator = None # New global instance for translator
retriever = None
reranker = None
corpus_original_texts = {}  # Store original passage texts
corpus_translated_texts = {} # Store translated passage texts for embedding and reranking


#[수정됨] data_dir 인수 추가 (번역된 코퍼스 저장 경로로 사용)
def preprocess(corpus_dict):
    """
    Preprocessing function using Qwen3 translation, then E5 multilingual model + BGE reranker.
    
    Input: 
    - corpus_dict: dict mapping document IDs to document objects with 'passage'/'text' field
    - data_dir: str, path to the directory containing corpus (for saving translated file)
    Output: dict containing initialized models, embeddings, and corpus data
    """
    global translator, retriever, reranker, corpus_original_texts, corpus_translated_texts
    
    print("=" * 60)
    print("PREPROCESSING: Initializing Qwen3 Translator + E5 + BGE Reranker Pipeline...")
    print("=" * 60)
    start_time = time.time() 
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize Qwen3 translator
    print("Loading Qwen3 Translator...")
    translator = Translator()
    
    # Store corpus IDs and original passage texts
    retriever_corpus_ids = list(corpus_dict.keys())
    original_passages = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]
    corpus_original_texts = {doc_id: original_passages[i] for i, doc_id in enumerate(retriever_corpus_ids)}
    
    # STAGE 0: Translate the entire corpus to English
    print(f"Translating corpus of {len(original_passages)} documents from Hebrew to English...")
    translated_passages = translator.translate(original_passages, src_lang="he", tgt_lang="en", batch_size=64)
    
    # Store translated texts
    corpus_translated_texts = {doc_id: translated_passages[i] for i, doc_id in enumerate(retriever_corpus_ids)}
    

    # Initialize E5 retriever
    print("\nLoading E5 retriever...")
    retriever = E5Retriever() # Now E5 will embed *English* texts
    
    # Initialize BGE reranker   
    print("Loading BGE reranker...")
    reranker = BGEReranker() # Now BGE will rerank *English* texts
    
    print(f"Preparing corpus with {len(retriever_corpus_ids)} documents...")
    
    # Compute embeddings for TRANSLATED passages
    print("Computing E5 embeddings for translated corpus...")
    retriever.corpus_ids = retriever_corpus_ids # Assign corpus IDs
    retriever.corpus_embeddings = retriever.embed_texts(translated_passages, is_query=False, batch_size=256)
    
    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated embeddings for {len(retriever.corpus_ids)} documents")
    print(f"✓ Embedding matrix shape: {retriever.corpus_embeddings.shape}")
    
    end_time = time.time() # End timer
    elapsed_time = end_time - start_time
    print(f"✓ Total preprocessing time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    
    return {
        'translator': translator,
        'retriever': retriever,
        'reranker': reranker,
        'corpus_ids': retriever.corpus_ids,
        'corpus_embeddings': retriever.corpus_embeddings,
        'corpus_original_texts': corpus_original_texts, # Keep original texts if needed later
        'corpus_translated_texts': corpus_translated_texts, # Pass translated texts for reranking
        'num_documents': len(corpus_dict)
    }

def predict(query, preprocessed_data):
    """
    Two-stage prediction with Qwen3 translation: E5 retrieval + BGE reranking.
    
    Input: 
    - query: dict with 'query' field containing query text (assumed Hebrew)
    - preprocessed_data: dict from preprocess() containing models and corpus data
    
    Output: list of dicts with 'paragraph_uuid' and 'score' fields, ranked by relevance
    """
    global translator, retriever, reranker, corpus_original_texts, corpus_translated_texts
    
    # Extract query text (assumed to be in original language, e.g., Hebrew)
    original_query_text = query.get('query', '')
    if not original_query_text:
        return []
    
    # Use global instances or get from preprocessed_data
    if translator is None or retriever is None or reranker is None:
        translator = preprocessed_data.get('translator')
        retriever = preprocessed_data.get('retriever')
        reranker = preprocessed_data.get('reranker')
        corpus_original_texts = preprocessed_data.get('corpus_original_texts', {})
        corpus_translated_texts = preprocessed_data.get('corpus_translated_texts', {}) # Get translated texts here
        
        if translator is None or retriever is None or reranker is None:
            print("Error: Missing translator, retriever or reranker in preprocessed data", file=sys.stderr)
            return []
    
    try:
        # STAGE 0: Translate the query from Hebrew to English
        # print("Stage 0: Translating query...") # [수정됨] 아래에서 더 자세히 출력
        translated_query_texts = translator.translate([original_query_text], src_lang="he", tgt_lang="en", batch_size=1)
        translated_query_text = translated_query_texts[0] if translated_query_texts else ""
        
        # --- #[추가됨] Request 3: 쿼리 번역 결과 출력 ---
        print("\n" + "-" * 40)
        print(f"Original Query (HE):   {original_query_text}")
        print(f"Translated Query (EN): {translated_query_text}")
        print("-" * 40)
        # --- #[추가됨] End ---

        if not translated_query_text:
            print("Warning: Query translation failed, falling back to original query text for E5 (may not work as intended).", file=sys.stderr)
            # If translation fails, use original query but results will likely be poor for English-centric E5
            translated_query_text = original_query_text 
            
        # STAGE 1: E5 Retrieval (get top 100 candidates)
        print("Stage 1: E5 retrieval (using translated query)...")
        query_embedding = retriever.embed_texts([translated_query_text], is_query=True, batch_size=1)
        
        # Compute cosine similarity with precomputed corpus embeddings (which are for translated texts)
        E5_scores = cosine_similarity(query_embedding, retriever.corpus_embeddings)[0]
        
        # Get top 100 candidates for reranking
        top_100_indices = np.argsort(E5_scores)[::-1][:100]
        
        # Get passages (translated versions) and IDs for reranking
        candidate_ids = [retriever.corpus_ids[idx] for idx in top_100_indices]
        candidate_passages = [corpus_translated_texts.get(doc_id, '') for doc_id in candidate_ids] # Use translated texts for reranking
        
        # STAGE 2: BGE Reranking (rerank top 100 -> top 20)
        print("Stage 2: BGE reranking (using translated query and translated candidates)...")
        reranked_results = reranker.rerank(
            translated_query_text, # Use translated query
            candidate_passages,    # Use translated candidate passages
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
        
        # [수정됨] "print" -> "print" (오타 수정), stderr 대신 stdout 사용
        print(f"✓ Returned {len(results)} results with reranker scores")
        return results
        
    except Exception as e:
        print(f"Fatal error in prediction pipeline: {e}", file=sys.stderr)
        # Fallback in case of a complete failure
        return []