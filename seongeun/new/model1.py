import json
import torch
from typing import Optional
import numpy as np
import os
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

class AlephBertDPRRetriever:
    def __init__(
        self,
        query_model_name: Optional[str] = None,
        passage_model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 256,
        pooling: str = "cls",          # "cls" or "mean"
        proj_dim: Optional[int] = None, # e.g., 384
        use_fp16_if_cuda: bool = True,
    ):
        """
        AlephBERT 기반 DPR-style 바이인코더 (가중치 비공유).
        - embed_texts(texts, is_query=False, batch_size=32) 시그니처 유지하여 E5 대체 가능.
        - query/passage 각각 전용 인코더를 사용.
        """
        # 로컬 경로 자동 탐색 or 기본 모델명
        def _resolve_model_name(name, fallback):
            if name is not None:
                return name
            local_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models", fallback.replace("/", "-")
            )
            if os.path.isdir(local_model_path):
                print(f"Using local model from: {local_model_path}")
                return local_model_path
            return fallback

        query_model_name  = _resolve_model_name(query_model_name,  "onlplab/alephbert-base")
        passage_model_name = _resolve_model_name(passage_model_name, "onlplab/alephbert-base")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.pooling = pooling.lower()
        self.proj_dim = proj_dim

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[DPR] Loading query encoder:   {query_model_name}")
        print(f"[DPR] Loading passage encoder: {passage_model_name}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(passage_model_name, use_fast=True)
        dtype = torch.float16 if (use_fp16_if_cuda and self.device.startswith("cuda")) else torch.float32

        # 가중치 비공유: 서로 다른 AutoModel 인스턴스
        self.query_encoder   = AutoModel.from_pretrained(query_model_name,   torch_dtype=dtype).to(self.device)
        self.passage_encoder = AutoModel.from_pretrained(passage_model_name, torch_dtype=dtype).to(self.device)

        # 선택적 투영(공유/비공유 중 택1: 여기선 공유 projection 사용)
        if self.proj_dim is not None:
            hid_q = self.query_encoder.config.hidden_size
            hid_d = self.passage_encoder.config.hidden_size
            assert hid_q == hid_d, "query/passsage hidden size must match for shared projection"
            self.proj = torch.nn.Linear(hid_q, self.proj_dim, bias=False).to(self.device)
        else:
            self.proj = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.corpus_ids: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None

    def to(self, device: str):
        self.device = device
        self.query_encoder.to(device)
        self.passage_encoder.to(device)
        if self.proj is not None:
            self.proj.to(device)
        return self
    
    def load(self, load_dir: str):
        self.query_encoder = AutoModel.from_pretrained(os.path.join(load_dir, "query_encoder")).to(self.device)
        self.passage_encoder = AutoModel.from_pretrained(os.path.join(load_dir, "passage_encoder")).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_dir, "tokenizer"), use_fast=True)
        if os.path.exists(os.path.join(load_dir, "projection.pt")):
            self.load_projection(os.path.join(load_dir, "projection.pt"))
        print(f"[DPR] Loaded model from {load_dir}")

    def _pool(self, outputs, attention_mask):
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_emb = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            emb = sum_emb / denom
        elif self.pooling == "cls_pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            emb = last_hidden[:, 0]  # [CLS]
        if self.proj is not None:
            emb = self.proj(emb)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def _encode(self, texts: List[str], is_query: bool) -> torch.Tensor:
        """
        내부 인코딩 (한 번에 모든 텍스트를 처리하지 않음 — embed_texts에서 배치로 분할)
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        encoder = self.query_encoder if is_query else self.passage_encoder
        outputs = encoder(**enc)
        embs = self._pool(outputs, enc["attention_mask"])
        return embs  # [B,H]

    def embed_texts(self, texts: List[str], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        """
        - texts: List[str]
        - is_query: 쿼리/패시지 선택
        - return: np.ndarray [N, dim]
        """
        if len(texts) == 0:
            dim = self.proj_dim or self.passage_encoder.config.hidden_size
            return np.zeros((0, dim), dtype=np.float32)

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            if (not is_query) and (batch_num % 200 == 0):
                print(f"[DPR embed] batch {batch_num}/{total_batches} "
                      f"({(batch_num/total_batches)*100:.1f}%)")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            batch_texts = [t if isinstance(t, str) else str(t) for t in texts[i:i+batch_size]]

            try:
                with torch.no_grad():
                    embs = self._encode(batch_texts, is_query=is_query)  # [B,H]
                all_embeddings.append(embs.cpu())

                del embs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"[DPR] CUDA OOM at batch {batch_num}, fallback to single encoding…")
                for t in batch_texts:
                    try:
                        embs = self._encode([t], is_query=is_query)  # [1,H]
                        all_embeddings.append(embs.cpu())
                        del embs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e2:
                        print(f"Failed to process a text: {e2}")
                        dim = self.proj_dim or self.passage_encoder.config.hidden_size
                        all_embeddings.append(torch.zeros(1, dim, dtype=torch.float32))

        return torch.cat(all_embeddings, dim=0).cpu().to(torch.float32).numpy()

    # 학습용: 단일 쿼리 배치에 대해 후보 문단 점수 계산 (GRADIENT 필요)
    def forward(self, query_inputs: Dict[str, torch.Tensor],
                doc_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        query_inputs: [B, L] 텐서 가능
        doc_inputs:   [B, N, L] 혹은 [N, L]
        return: scores
            - 입력이 (B, N)라면 [B, N] 스코어
            - 입력이 (1, N)라면 [N]
        """
        # Encoder 모드는 바깥에서 제어 (train/eval)
        q_out = self.query_encoder(**query_inputs)              # [B,L,H]
        q_vec = self._pool(q_out, query_inputs["attention_mask"])  # [B,H]

        # doc_inputs가 [N,L]이면 B=1로 간주
        if doc_inputs["input_ids"].dim() == 2:
            d_out = self.passage_encoder(**doc_inputs)          # [N,L,H]
            d_vecs = self._pool(d_out, doc_inputs["attention_mask"])   # [N,H]
            scores = q_vec @ d_vecs.T                           # [B,N]
            return scores.squeeze(0) if q_vec.size(0) == 1 else scores

        # [B,N,L] 형태 지원
        B, N, L = doc_inputs["input_ids"].shape
        flat_docs = {k: v.view(B*N, L) for k, v in doc_inputs.items()}
        d_out = self.passage_encoder(**flat_docs)               # [B*N,L,H]
        d_vecs = self._pool(d_out, flat_docs["attention_mask"]) # [B*N,H]
        d_vecs = d_vecs.view(B, N, -1)                          # [B,N,H]
        # batched matmul → [B,N]
        scores = torch.matmul(d_vecs, q_vec.unsqueeze(-1)).squeeze(-1)
        return scores

    # 상태 저장/로딩
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.query_encoder.save_pretrained(os.path.join(save_dir, "query_encoder"))
        self.passage_encoder.save_pretrained(os.path.join(save_dir, "passage_encoder"))
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        if self.proj is not None:
            torch.save(self.proj.state_dict(), os.path.join(save_dir, "projection.pt"))

    def load_projection(self, path: str):
        if self.proj is None:
            raise ValueError("proj_dim is None; initialize with proj_dim to use projection.")
        state = torch.load(path, map_location=self.device)
        self.proj.load_state_dict(state)


class AlephBertReranker:
    def __init__(self, model_name=None, device=None):
        # 로컬 모델 자동 탐색 (원래 경로와 유사한 구조 유지)
        def _resolve_model_name(name, fallback):
            if name is not None:
                return name
            local_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models", fallback.replace("/", "-")
            )
            if os.path.isdir(local_model_path):
                print(f"Using local model from: {local_model_path}")
                return local_model_path
            return fallback
        
        model_name = _resolve_model_name(model_name, "onlplab/alephbert-base")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading AlephBERT reranker on device: {self.device}")

        # tokenizer & model (sequence classification head, regression-style: num_labels=1)
        dtype = torch.float16 if (self.device.startswith("cuda") and torch.cuda.is_available()) else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,                 # 단일 로짓 출력 (연속 점수)
            torch_dtype=dtype,
            trust_remote_code=False
        ).to(self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def rerank(self, query_text, passages, passage_ids, top_k=20, batch_size=4):
        """
        AlephBERT cross-encoder로 패시지 재랭킹.
        - 반환: [(passage_id, score)] 내림차순 상위 top_k
        """
        if not passages:
            return []

        scores: List[float] = []

        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i + batch_size]

            try:
                batch_queries = [query_text] * len(batch_passages)

                with torch.no_grad():
                    inputs = self.tokenizer(
                        batch_queries,
                        batch_passages,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    logits = outputs.logits  # [B, 1] or [B] depending on HF version

                    if logits.ndim == 1:
                        batch_scores = logits.detach().float().cpu().numpy()
                    elif logits.shape[1] == 1:
                        batch_scores = logits.squeeze(-1).detach().float().cpu().numpy()
                    else:
                        # 혹시 2-class head가 로드된 커스텀 체크포인트라면 양성 클래스 사용
                        batch_scores = logits[:, 1].detach().float().cpu().numpy()

                scores.extend(batch_scores.tolist())

                # 메모리 정리
                del inputs, outputs, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in reranking batch {i // batch_size + 1}: {e}")
                # 실패 시 중립 점수 부여 (원 코드와 동일한 폴백)
                fallback_scores = [0.5] * len(batch_passages)
                scores.extend(fallback_scores)

        # (passage_id, score)로 묶고 내림차순 정렬
        results = list(zip(passage_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# Global instances
retriever = None
reranker = None
corpus_texts = {}  # Store original passage texts for reranking

def preprocess(corpus_dict):
    """
    Preprocessing using AlephBERT DPR retriever + AlephBERT reranker.

    Input:
      - corpus_dict: dict[str, dict], 각 값에는 최소 'passage' 또는 'text' 중 하나가 있어야 함.
        예) { "doc_id": {"passage": "..."} } 또는 { "doc_id": {"text": "..."} }

    Output:
      - dict: {'retriever', 'reranker', 'corpus_ids', 'corpus_embeddings', 'corpus_texts', 'num_documents'}
    """
    global retriever, reranker, corpus_texts

    print("=" * 60)
    print("PREPROCESSING: Initializing AlephBERT DPR + AlephBERT Reranker Pipeline...")
    print("=" * 60)

    # GPU 메모리 할당 전략(선택)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 1) 모델 초기화 (전역 인스턴스 재사용)
    if retriever is None:
        print("Loading AlephBERT DPR retriever...")
        retriever = AlephBertDPRRetriever()
    else:
        print("Reusing existing AlephBertDPRRetriever instance.")

    if reranker is None:
        print("Loading AlephBERT reranker...")
        reranker = AlephBertReranker()
    else:
        print("Reusing existing AlephBertReranker instance.")

    # 2) 코퍼스 준비 (ID와 텍스트 매핑을 '동일 루프'에서 생성해 정렬 문제 방지)
    doc_ids = []
    passages = []
    for doc_id, doc in corpus_dict.items():
        # passage 우선, 없으면 text, 둘 다 없으면 빈 문자열
        text = doc.get("passage", doc.get("text", ""))
        # 혹시 None/비문자 타입 방지
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        doc_ids.append(doc_id)
        passages.append(text)

    print(f"Preparing corpus with {len(doc_ids)} documents...")

    retriever.corpus_ids = doc_ids
    corpus_texts = {d: p for d, p in zip(doc_ids, passages)}

    # 3) 임베딩 계산
    # 배치 사이즈는 메모리 상황에 맞게 조절 가능 (기본 32)
    print("Computing AlephBERT DPR embeddings...")
    retriever.corpus_embeddings = retriever.embed_texts(passages, is_query=False, batch_size=32)

    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated embeddings for {len(retriever.corpus_ids)} documents")
    print(f"✓ Embedding matrix shape: {retriever.corpus_embeddings.shape}")

    return {
        "retriever": retriever,
        "reranker": reranker,
        "corpus_ids": retriever.corpus_ids,
        "corpus_embeddings": retriever.corpus_embeddings,
        "corpus_texts": corpus_texts,
        "num_documents": len(doc_ids),
    }

def predict(query, preprocessed_data, top_k_retrieve: int = 100, top_k_return: int = 20, batch_size_query: int = 1):
    """
    Two-stage prediction: AlephBERT DPR retrieval + AlephBERT reranking.

    Input:
      - query: dict with key 'query' (str)
      - preprocessed_data: dict returned by preprocess()
      - top_k_retrieve: DPR 단계에서 재랭킹 후보로 보낼 개수
      - top_k_return: 최종 반환 개수
      - batch_size_query: 쿼리 임베딩 배치 크기 (보통 1)

    Output:
      - list[{'paragraph_uuid': str, 'score': float}]
    """
    global retriever, reranker, corpus_texts

    # 0) 쿼리 텍스트 추출
    query_text = (query or {}).get('query', '')
    if not isinstance(query_text, str) or not query_text.strip():
        return []

    query_text = query_text.strip()

    # 1) 전역 인스턴스 / 프리프로세스 데이터에서 안전하게 꺼내기
    if retriever is None:
        retriever = (preprocessed_data or {}).get('retriever')
    if reranker is None:
        reranker = (preprocessed_data or {}).get('reranker')
    if not corpus_texts:
        corpus_texts = (preprocessed_data or {}).get('corpus_texts', {})

    if retriever is None:
        print("Error: retriever is not initialized.")
        return []

    if getattr(retriever, "corpus_embeddings", None) is None or len(getattr(retriever, "corpus_ids", [])) == 0:
        print("Error: corpus is not preprocessed or embeddings are missing.")
        return []

    try:
        # ===== STAGE 1: DPR Retrieval =====
        print("Stage 1: AlephBERT DPR retrieval...")
        # (옵션) 추론 모드 보장
        if hasattr(retriever.query_encoder, "eval"):
            retriever.query_encoder.eval()
        if hasattr(retriever.passage_encoder, "eval"):
            retriever.passage_encoder.eval()

        q_emb = retriever.embed_texts([query_text], is_query=True, batch_size=batch_size_query)  # [1, H]
        dense_scores = cosine_similarity(q_emb, retriever.corpus_embeddings)[0]                  # [N]

        top_k_retrieve = min(top_k_retrieve, dense_scores.shape[0])
        top_idx = np.argsort(dense_scores)[::-1][:top_k_retrieve]

        candidate_ids = [retriever.corpus_ids[i] for i in top_idx]
        candidate_passages = [corpus_texts.get(cid, "") for cid in candidate_ids]

        # ===== STAGE 2: Reranking =====
        if reranker is not None:
            print("Stage 2: AlephBERT reranking...")
            # (옵션) 추론 모드 보장
            if hasattr(reranker, "model") and hasattr(reranker.model, "eval"):
                reranker.model.eval()

            reranked = reranker.rerank(
                query_text=query_text,
                passages=candidate_passages,
                passage_ids=candidate_ids,
                top_k=top_k_return
            )
            # 결과 포맷 정리
            results = [
                {"paragraph_uuid": pid, "score": float(score)}
                for (pid, score) in reranked
            ]
            print(f"✓ Returned {len(results)} results with reranker scores")
            return results

        else:
            # reranker가 없으면 DPR 점수로 바로 반환
            print("Reranker not available. Returning DPR results only.")
            top_k_return = min(top_k_return, len(candidate_ids))
            out = []
            for i in range(top_k_return):
                idx = top_idx[i]
                out.append({
                    "paragraph_uuid": retriever.corpus_ids[idx],
                    "score": float(dense_scores[idx])
                })
            return out

    except Exception as e:
        print(f"Error in prediction: {e}")
        # 폴백: DPR-only (reranker 스킵)
        try:
            q_emb = retriever.embed_texts([query_text], is_query=True, batch_size=batch_size_query)
            dense_scores = cosine_similarity(q_emb, retriever.corpus_embeddings)[0]
            order = np.argsort(dense_scores)[::-1][:top_k_return]
            return [
                {
                    "paragraph_uuid": retriever.corpus_ids[i],
                    "score": float(dense_scores[i])
                }
                for i in order
            ]
        except Exception as e2:
            print(f"Secondary fallback failed: {e2}")
            return []