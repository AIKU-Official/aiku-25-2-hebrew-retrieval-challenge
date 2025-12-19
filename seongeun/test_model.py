# test_model.py
import json
from model import preprocess, predict

# ====== 1️⃣ 간단한 히브리어 코퍼스 ======
corpus = {
    "doc_1": {"passage": "דן קם בבוקר בשעה שש והולך לעבודה."},
    "doc_2": {"passage": "מיכל לומדת באוניברסיטה בתל אביב."},
    "doc_3": {"passage": "החתול יושב על הכיסא בסלון."},
    "doc_4": {"passage": "הילדים משחקים בכדור בפארק אחר הצהריים."},
    "doc_5": {"passage": "יורד גשם חזק ברחובות ירושלים."},
}

# ====== 2️⃣ preprocess (임베딩 생성) ======
print("🔹 Building corpus embeddings...")
preproc = preprocess(corpus)
print(json.dumps(preproc, indent=2, ensure_ascii=False))

# ====== 3️⃣ 질의 (히브리어 QA 스타일) ======
queries = [
    {"query": "באיזו שעה דן קם בבוקר?"},        # 단은 아침 몇시에 일어났나
    {"query": "איפה מיכל לומדת?"},              # 미할은 어디서 공부하나
    {"query": "מה עושה החתול?"},                # 고양이는 무엇을 하고 있나
]

# ====== 4️⃣ 예측 ======
for q in queries:
    print("\n============================")
    print(f"🟢 Query: {q['query']}")
    results = predict(q, preproc, top_k_retrieve=5, top_k_return=3)
    for r in results:
        docid, score = r["paragraph_uuid"], r["score"]
        print(f"  {docid:>6} | score={score:8.4f} | text={corpus[docid]['passage']}")
