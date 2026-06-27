import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# Folder
# =====================================

input_file = "data/processed/cases.json"
output_folder = "data/results"
eval_folder = "data/eval"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(eval_folder, exist_ok=True)

# =====================================
# Load Dataset
# =====================================

with open(input_file, "r", encoding="utf-8") as f:
    cases = json.load(f)

df = pd.DataFrame(cases)

print(f"Jumlah kasus : {len(df)}")

# =====================================
# Membuat Dokumen
# =====================================

df["document"] = (
    df["pasal"].fillna("").astype(str)
    + " "
    + df["ringkasan_fakta"].fillna("").astype(str)
)

# =====================================
# Train Test Split (80:20)
# =====================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"Jumlah Train : {len(train_df)}")
print(f"Jumlah Test  : {len(test_df)}")

# =====================================
# TF-IDF
# =====================================

vectorizer = TfidfVectorizer()

train_vectors = vectorizer.fit_transform(train_df["document"])

print("Ukuran TF-IDF :", train_vectors.shape)

# =====================================
# Fungsi Retrieval
# =====================================

def retrieve(query: str, k: int = 5):
    """
    Melakukan retrieval Top-k kasus paling mirip
    """

    # 1. Pre-processing Query
    query = str(query).lower().strip()

    # 2. Vector Query
    query_vector = vectorizer.transform([query])

    # 3. Cosine Similarity
    similarity = cosine_similarity(
        query_vector,
        train_vectors
    ).flatten()

    # 4. Ranking
    result = train_df.copy()

    result["similarity"] = similarity

    result = result.sort_values(
        by="similarity",
        ascending=False
    )

    # 5. Return Top-k
    return result.head(k)

# =====================================
# Membuat queries.json
# =====================================

queries = []

for _, row in test_df.iterrows():

    queries.append({

        "query": row["document"],

        "ground_truth_case_id": int(row["case_id"])

    })

queries_file = os.path.join(
    eval_folder,
    "queries.json"
)

with open(
    queries_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        queries,
        f,
        indent=4,
        ensure_ascii=False
    )

print(f"queries.json berhasil dibuat : {queries_file}")

# =====================================
# Retrieval Semua Query Test
# =====================================

hasil = []

for _, row in test_df.iterrows():

    retrieved = retrieve(
        query=row["document"],
        k=5
    )

    for _, r in retrieved.iterrows():

        hasil.append({

            "query_case": row["case_id"],

            "query_no_perkara": row["no_perkara"],

            "ground_truth_case_id": row["case_id"],

            "similar_case": r["case_id"],

            "similar_no_perkara": r["no_perkara"],

            "similarity": round(r["similarity"],4)

        })

# =====================================
# Simpan Retrieval
# =====================================

hasil_df = pd.DataFrame(hasil)

output_file = os.path.join(
    output_folder,
    "retrieval.csv"
)

hasil_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# =====================================
# Tampilkan
# =====================================

print("\n")
print("="*60)
print("CASE RETRIEVAL SELESAI")
print("="*60)

print(hasil_df.head())

print(f"\nJumlah hasil : {len(hasil_df)}")
print(f"Hasil Retrieval : {output_file}")
print(f"File Query      : {queries_file}")