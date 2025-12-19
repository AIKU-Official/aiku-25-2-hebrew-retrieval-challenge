#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_dpr_retriever.py
- Codabench baseline E5Retriever 의 retriever 품질 sanity check
- 동일 문장 / 무관 문장 cosine 비교로 정상 작동 여부 점검
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model import E5Retriever  # baseline의 retriever 클래스 import

# ==============================================
# 1. Load retriever (same as in model.py)
# ==============================================
retriever = E5Retriever()
print(f"\n✅ Retriever loaded on device: {retriever.device}")

# ==============================================
# 2. Define test sentences (Hebrew)
# ==============================================
same_text = "דן קם בבוקר בשעה שש והולך לעבודה."        # “Dan gets up at 6 and goes to work.”
diff_text = "החתול יושב על הכיסא בסלון."              # “The cat is sitting on the chair.”

# ==============================================
# 3. Embed as query and passage
# ==============================================
qv = retriever.embed_texts([same_text], is_query=True)
dv_same = retriever.embed_texts([same_text], is_query=False)
dv_diff = retriever.embed_texts([diff_text], is_query=False)

# ==============================================
# 4. Compute cosine similarities
# ==============================================
cos_same = cosine_similarity(qv, dv_same)[0][0]
cos_diff = cosine_similarity(qv, dv_diff)[0][0]

print("\n===== DPR Retriever Sanity Check =====")
print(f"Query:  {same_text}")
print(f"Pos:    {same_text}")
print(f"Neg:    {diff_text}")
print("---------------------------------------")
print(f"Cosine(query, same passage) = {cos_same:.4f}")
print(f"Cosine(query, diff passage) = {cos_diff:.4f}")

if cos_same > cos_diff:
    print("✅ DPR retriever seems to work (positive > negative).")
else:
    print("⚠️ DPR retriever embeddings may be untrained or misaligned.")
