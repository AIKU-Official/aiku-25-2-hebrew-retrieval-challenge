import json
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import unicodedata

##################
# To.DO.
# 1. dictabert + bge-reranker 확인해보고 
# 2. preprocess ->  SPLINTER 알고리즘 적용해보기 
###################

_NIQQUD_RE = re.compile(r'[\u0591-\u05C7]') 

def normalize_he(text: str) -> str:

    t = unicodedata.normalize("NFC", text or "")
    t = _NIQQUD_RE.sub("", t)
    t = " ".join(t.split())
    return t
class DictaBertRetriever:
    def __init__(self, model_name=None, device=None):
        """
        Initializes a Hebrew-first retriever using DictaBERT.
        Expect a local checkpoint at ./models/dictabert  (recommended).
        """
        # Prefer local model inside the submission zip
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'dictabert')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local DictaBERT from: {model_name}")
            else:

                model_name = local_model_path  

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Loading DictaBERT on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.corpus_ids = []
        self.corpus_embeddings = None

    @torch.no_grad()
    def _batch_embed(self, batch_texts):
        encoded = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state          # [B, L, H]
        attention_mask = encoded['attention_mask'].unsqueeze(-1).float()  # [B, L, 1]


        summed = (token_embeddings * attention_mask).sum(dim=1)          # [B, H]
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)               # [B, 1]
        sent_embeddings = summed / counts                                # [B, H]


        sent_embeddings = torch.nn.functional.normalize(sent_embeddings, p=2, dim=1)
        return sent_embeddings.cpu()

    def embed_texts(self, texts, is_query=False, batch_size=32):
        """
        Generate embeddings for texts using DictaBERT.
        (No E5-style prefixes; we apply a light Hebrew normalization.)
        """
        texts = [normalize_he(t) for t in texts]
        all_embeds = []
        total = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                embs = self._batch_embed(batch)
                all_embeds.append(embs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                # 매우 보수적으로 1개씩 처리
                for s in batch:
                    try:
                        embs = self._batch_embed([s])
                        all_embeds.append(embs)
                    except Exception as e2:
                        print(f"Embed fail, fallback zero vector: {e2}")
                        # DictaBERT hidden size 보편값 768
                        all_embeds.append(torch.zeros(1, 768))

        return torch.cat(all_embeds, dim=0).numpy()

class BGEReranker:
    def __init__(self, model_name=None, device=None):
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'bge-reranker-v2-m3')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local BGE model from: {model_name}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BGE reranker on device: {self.device}")

        from transformers import AutoModelForSequenceClassification
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
        if not passages:
            return []
        query_text = normalize_he(query_text)
        passages = [normalize_he(p) for p in passages]

        scores = []
        batch_size = 4
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i+batch_size]
            try:
                batch_queries = [query_text] * len(batch_passages)
                with torch.no_grad():
                    inputs = self.tokenizer(
                        batch_queries, batch_passages,
                        padding=True, truncation=True, max_length=512,
                        return_tensors='pt'
                    ).to(self.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    if len(logits.shape) == 1:
                        batch_scores = logits.cpu().numpy()
                    elif logits.shape[1] == 1:
                        batch_scores = logits.squeeze(-1).cpu().numpy()
                    else:
                        batch_scores = logits[:, 1].cpu().numpy()
                scores.extend(batch_scores.tolist())
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error in reranking batch {i//batch_size + 1}: {e}")
                scores.extend([0.5] * len(batch_passages))

        results = list(zip(passage_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)  # 안정 정렬

        seen = set()
        uniq = []
        for pid, sc in results:
            if pid in seen:
                continue
            seen.add(pid)
            uniq.append((pid, sc))
            if len(uniq) >= top_k:
                break
        return uniq


retriever = None
reranker = None
corpus_texts = {}

def preprocess(corpus_dict):
    global retriever, reranker, corpus_texts

    print("=" * 60)
    print("PREPROCESSING: Initializing DictaBERT + BGE Reranker Pipeline...")
    print("=" * 60)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print("Loading DictaBERT retriever...")
    retriever = DictaBertRetriever()

    print("Loading BGE reranker...")
    reranker = BGEReranker()

    print(f"Preparing corpus with {len(corpus_dict)} documents...")

    retriever.corpus_ids = list(corpus_dict.keys())
    passages_raw = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]

    corpus_texts = {doc_id: passages_raw[i] for i, doc_id in enumerate(retriever.corpus_ids)}

    print("Computing DictaBERT embeddings...")
    passages_norm = [normalize_he(t) for t in passages_raw]
    retriever.corpus_embeddings = retriever.embed_texts(passages_norm, is_query=False, batch_size=32)

    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated embeddings for {len(retriever.corpus_ids)} documents")
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
    global retriever, reranker, corpus_texts

    query_text = query.get('query', '')
    if not query_text:
        return []

    if retriever is None:
        retriever = preprocessed_data.get('retriever')
        reranker = preprocessed_data.get('reranker')
        corpus_texts = preprocessed_data.get('corpus_texts', {})
        if retriever is None or reranker is None:
            print("Error: Missing retriever or reranker in preprocessed data")
            return []

    try:
        print("Stage 1: DictaBERT retrieval...")
        q_emb = retriever.embed_texts([query_text], is_query=True, batch_size=1)
        sims = cosine_similarity(q_emb, retriever.corpus_embeddings)[0]

        top_100 = np.argsort(sims)[::-1][:100]
        cand_ids = [retriever.corpus_ids[idx] for idx in top_100]
        cand_passages = [corpus_texts.get(doc_id, '') for doc_id in cand_ids]

        print("Stage 2: BGE reranking...")
        reranked = reranker.rerank(query_text, cand_passages, cand_ids, top_k=20)

        results = [{'paragraph_uuid': pid, 'score': float(score)} for pid, score in reranked]
        print(f"✓ Returned {len(results)} results with reranker scores")
        return results

    except Exception as e:
        print(f"Error in prediction: {e}")
        try:
            q_emb = retriever.embed_texts([query_text], is_query=True, batch_size=1)
            sims = cosine_similarity(q_emb, retriever.corpus_embeddings)[0]
            top_idx = np.argsort(sims)[::-1][:20]
            
            out = []
            seen = set()
            for idx in top_idx:
                pid = retriever.corpus_ids[idx]
                if pid in seen:
                    continue
                seen.add(pid)
                out.append({'paragraph_uuid': pid, 'score': float(sims[idx])})
                if len(out) >= 20:
                    break
            return out
        except:
            return []
