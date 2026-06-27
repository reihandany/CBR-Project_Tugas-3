import os
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# Folder
# ==========================================

case_file = "data/processed/cases.json"
query_file = "data/eval/queries.json"
output_folder = "data/results"

os.makedirs(output_folder, exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

with open(case_file, "r", encoding="utf-8") as f:
    cases = json.load(f)

df = pd.DataFrame(cases)

df["document"] = (
    df["pasal"].fillna("").astype(str)
    + " "
    + df["ringkasan_fakta"].fillna("").astype(str)
)

# ==========================================
# TF-IDF
# ==========================================

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(df["document"])

# ==========================================
# Retrieval Function
# ==========================================

def retrieve(query, k=5):

    query_vector = vectorizer.transform([str(query)])

    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    result = df.copy()

    result["similarity"] = similarity

    result = result.sort_values(
        by="similarity",
        ascending=False
    )

    return result.head(k)

# ==========================================
# Load Query Evaluation
# ==========================================

with open(query_file, "r", encoding="utf-8") as f:

    queries = json.load(f)

# ==========================================
# Evaluation
# ==========================================

hasil = []

total_hit = 0
total_precision = 0
total_recall = 0

for item in queries:

    query = item["query"]

    ground_truth = item["ground_truth_case_id"]

    retrieved = retrieve(query, k=5)

    retrieved_ids = retrieved["case_id"].tolist()

    # Hit
    hit = 1 if ground_truth in retrieved_ids else 0

    # Precision@5
    precision = hit / 5

    # Recall
    recall = hit

    total_hit += hit
    total_precision += precision
    total_recall += recall

    hasil.append({

        "query": query,

        "ground_truth": ground_truth,

        "retrieved": ",".join(map(str, retrieved_ids)),

        "hit": hit,

        "precision@5": round(precision,3),

        "recall": recall

    })

# ==========================================
# Summary
# ==========================================

n = len(queries)

hit_rate = total_hit / n
mean_precision = total_precision / n
mean_recall = total_recall / n

hasil_df = pd.DataFrame(hasil)

# ==========================================
# Simpan CSV
# ==========================================

evaluation_file = os.path.join(
    output_folder,
    "evaluation.csv"
)

hasil_df.to_csv(
    evaluation_file,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# Simpan Summary
# ==========================================

summary_file = os.path.join(
    output_folder,
    "summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("===== HASIL EVALUASI =====\n\n")

    f.write(f"Jumlah Query      : {n}\n")

    f.write(f"Hit Rate          : {hit_rate:.3f}\n")

    f.write(f"Precision@5       : {mean_precision:.3f}\n")

    f.write(f"Recall            : {mean_recall:.3f}\n")

# ==========================================
# Print
# ==========================================

print("="*60)
print("EVALUATION SELESAI")
print("="*60)

print(hasil_df)

print("\n")

print(f"Hit Rate    : {hit_rate:.3f}")
print(f"Precision@5 : {mean_precision:.3f}")
print(f"Recall      : {mean_recall:.3f}")

print("\n")

print("Output:")
print(evaluation_file)
print(summary_file)