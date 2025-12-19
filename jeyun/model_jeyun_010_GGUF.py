import json
import torch
import numpy as np
import os
import warnings
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
# Add GGUF support
from llama_cpp import Llama

# GGUF embedding 관련 경고 메시지 억제
warnings.filterwarnings("ignore", message=".*embeddings required but some input tokens were not marked as outputs.*")

class QwenRetriever:
    def __init__(self, model_name=None, device=None):
        """
        Initializes the Qwen retriever using the GGUF model.
        """
        # Use GGUF model
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'Qwen3-Embedding-0.6B-GGUF')
            if os.path.isdir(local_model_path):
                gguf_files = [f for f in os.listdir(local_model_path) if f.endswith('.gguf')]
                if gguf_files:
                    model_name = os.path.join(local_model_path, gguf_files[0])

        if model_name is None:
            raise ValueError("No GGUF model path provided and no local model found.")

        print(f"Using GGUF model: {model_name}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Loading Qwen GGUF model on device: {self.device}")

        # Load GGUF model using llama-cpp-python with common params
        try:
            self.model = Llama(
                model_path=model_name,
                embedding=True,
                n_ctx=32768,
                n_gpu_layers=-1 if torch.cuda.is_available() else 0,
                n_threads=8,
                n_parallel = 4,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
                f16_kv=True,
                low_vram=False,
            )
            print("✓ Successfully loaded GGUF model")
        except Exception as e:
            print(f"Error loading GGUF model: {e}")
            raise

        self.corpus_ids = []
        self.corpus_embeddings = None

    def _call_embed_api(self, texts):
        """
        Try several possible llama-cpp-python embedding call signatures and parse the result.
        Returns: list of embedding vectors (list of list of floats)
        """
        # 1) try .embed (common newer wrapper)
        try:
            if hasattr(self.model, "embed"):
                res = self.model.embed(texts)
                # parse different possible shapes
                # Case: dict with 'data' like {'data': [{'embedding': [...]}, ...]}
                if isinstance(res, dict) and "data" in res:
                    embeddings = [d.get("embedding") for d in res["data"]]
                    return embeddings
                # Case: list of dicts or list of lists
                if isinstance(res, list):
                    # if list of dicts
                    if len(res) > 0 and isinstance(res[0], dict) and "embedding" in res[0]:
                        return [r["embedding"] for r in res]
                    # assume list of vectors
                    return res
                # fallback: try attribute
                if hasattr(res, "data") and isinstance(res.data, list):
                    return [d.get("embedding") for d in res.data]
        except Exception as e:
            # don't fail yet, try next option
            print(f"embed() call failed with: {e}")

        # 2) try create_embedding (some wrappers expose this)
        try:
            if hasattr(self.model, "create_embedding"):
                # some create_embedding signatures accept input= or inputs=
                try:
                    res = self.model.create_embedding(input=texts)
                except TypeError:
                    res = self.model.create_embedding(texts)
                if isinstance(res, dict) and "data" in res:
                    return [d.get("embedding") for d in res["data"]]
                if isinstance(res, list):
                    return res
        except Exception as e:
            print(f"create_embedding() call failed with: {e}")

        # 3) try embed_batch or embeddings
        try:
            if hasattr(self.model, "embed_batch"):
                res = self.model.embed_batch(texts)
                if isinstance(res, list):
                    return res
        except Exception as e:
            print(f"embed_batch() call failed with: {e}")

        # 4) last resort: try calling model with single inputs iteratively
        try:
            single_embeddings = []
            for t in texts:
                # try embed for single
                if hasattr(self.model, "embed"):
                    r = self.model.embed([t])
                    if isinstance(r, dict) and "data" in r:
                        single_embeddings.append(r["data"][0].get("embedding"))
                        continue
                    if isinstance(r, list):
                        single_embeddings.append(r[0] if len(r) == 1 else r)
                        continue
                # try create_embedding single
                if hasattr(self.model, "create_embedding"):
                    try:
                        r = self.model.create_embedding(input=t)
                    except TypeError:
                        r = self.model.create_embedding(t)
                    if isinstance(r, dict) and "data" in r:
                        single_embeddings.append(r["data"][0].get("embedding"))
                        continue
                    if isinstance(r, list):
                        single_embeddings.append(r[0] if len(r) == 1 else r)
                        continue
                # if none worked, raise
                raise RuntimeError("No embedding API available for single call fallback.")
            return single_embeddings
        except Exception as e:
            print(f"Iterative single-call fallback failed: {e}")

        # If we reach here, we couldn't get embeddings
        raise RuntimeError("Failed to obtain embeddings from llama-cpp model (all attempts failed).")

    def embed_texts(self, texts, is_query=False, batch_size=128):
        """
        Generates embeddings for texts using Qwen3 GGUF model via batch encoding.
        Args:
            texts (list[str]): The list of texts to embed.
            is_query (bool): Whether the texts are queries (affects prompt/logging).
            batch_size (int): The number of texts to process in a single batch.
        Returns:
            np.ndarray: normalized embeddings (num_texts x dim)
        """
        embeddings = []
        total_docs = len(texts)

        for i in range(0, total_docs, batch_size):
            batch_texts = texts[i:i+batch_size]
            current_batch_size = len(batch_texts)

            batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            if not is_query:
                print(f"Processing batch {batch_num}/{total_batches} ({min(i + current_batch_size, total_docs)}/{total_docs} total)")

            try:
                batch_results = self._call_embed_api(batch_texts)

                # Normalize returned structure: ensure list-of-lists
                if batch_results is None:
                    raise RuntimeError("Received None from embed API")

                # Some APIs may return nested structures; coerce to python lists of floats
                cleaned = []
                for item in batch_results:
                    if item is None:
                        cleaned.append(None)
                        continue
                    # convert numpy arrays to lists
                    if hasattr(item, "tolist"):
                        cleaned.append(item.tolist())
                    else:
                        cleaned.append(list(item))

                # If any embedding is None, fill with zeros of determined dim later
                embeddings.extend(cleaned)

            except Exception as e:
                print(f"Error in GGUF batch encoding (Batch {batch_num}): {e}. Falling back to zero vectors for this batch.")
                # Determine embedding_dim from previously obtained embeddings if possible
                if len(embeddings) > 0:
                    embedding_dim = len(embeddings[0])
                else:
                    embedding_dim = 1024  # fallback guess
                for _ in range(current_batch_size):
                    embeddings.append([0.0] * embedding_dim)

        # Convert to numpy array and handle any None entries
        # Determine embedding dim from first non-None
        first_non_none = None
        for e in embeddings:
            if e is not None:
                first_non_none = e
                break
        if first_non_none is None:
            raise RuntimeError("All embeddings are None / failed.")

        embedding_dim = len(first_non_none)
        normalized_embeddings = []
        for e in embeddings:
            if e is None:
                vec = np.zeros(embedding_dim, dtype=float)
            else:
                vec = np.array(e, dtype=float)
                # If vector length differs, pad or trim
                if vec.shape[0] < embedding_dim:
                    pad = np.zeros(embedding_dim - vec.shape[0], dtype=float)
                    vec = np.concatenate([vec, pad])
                elif vec.shape[0] > embedding_dim:
                    vec = vec[:embedding_dim]
            normalized_embeddings.append(vec)

        embeddings_np = np.vstack(normalized_embeddings)

        # Normalize embeddings (cosine similarity를 위해)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_np = embeddings_np / norms

        print(f"✓ Processed {total_docs} texts with GGUF batching (final shape: {embeddings_np.shape}).")

        return embeddings_np


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
    retriever.corpus_embeddings = retriever.embed_texts(passages, is_query=False, batch_size=128) 
    
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