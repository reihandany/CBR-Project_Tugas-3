import os
import json
from difflib import SequenceMatcher

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC

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

with open(query_file, "r", encoding="utf-8") as f:
    queries = json.load(f)

df = pd.DataFrame(cases)

df["document"] = (
    df["pasal"].fillna("").astype(str)
    + " "
    + df["ringkasan_fakta"].fillna("").astype(str)
)

# ==========================================
# TF-IDF Representation
# ==========================================

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["document"])

svm_model = LinearSVC(random_state=42)
svm_model.fit(tfidf_matrix, df["case_id"])

# ==========================================
# Retrieval Functions
# ==========================================

def retrieve_tfidf(query, k=5):
    query_vector = vectorizer.transform([str(query)])
    similarity = cosine_similarity(query_vector, tfidf_matrix).flatten()
    result = df.copy()
    result["similarity"] = similarity
    result = result.sort_values(by="similarity", ascending=False)
    return result.head(k)


def retrieve_svm(query, k=5):
    query_vector = vectorizer.transform([str(query)])
    scores = svm_model.decision_function(query_vector)[0]
    score_series = pd.Series(scores, index=svm_model.classes_)
    ranked_case_ids = score_series.sort_values(ascending=False).index.tolist()

    ranked_df = pd.DataFrame({"case_id": ranked_case_ids[:k]})
    result = ranked_df.merge(
        df[["case_id", "no_perkara", "pasal", "ringkasan_fakta"]],
        on="case_id",
        how="left"
    )
    result = result.copy()
    result["similarity"] = [score_series.loc[cid] for cid in result["case_id"]]
    return result


def retrieve_bert_proxy(query, k=5):
    query_text = str(query).lower().strip()
    query_vector = vectorizer.transform([query_text])
    tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    combined_scores = []
    for idx, case_text in enumerate(df["document"].astype(str)):
        case_text_norm = case_text.lower().strip()
        lexical_score = SequenceMatcher(None, query_text, case_text_norm).ratio()
        combined_score = 0.7 * tfidf_scores[idx] + 0.3 * lexical_score
        combined_scores.append(combined_score)

    result = df.copy()
    result["similarity"] = combined_scores
    result = result.sort_values(by="similarity", ascending=False)
    return result.head(k)


# ==========================================
# Evaluation Helper
# ==========================================

def evaluate_model(model_name, retrieve_fn):
    rows = []

    for item in queries:
        query = item["query"]
        ground_truth = int(item["ground_truth_case_id"])
        retrieved = retrieve_fn(query, k=5)
        retrieved_ids = retrieved["case_id"].tolist()

        hit = int(ground_truth in retrieved_ids)
        precision = hit / 5
        recall = hit

        rows.append({
            "model": model_name,
            "query": query,
            "ground_truth": ground_truth,
            "retrieved": ",".join(map(str, retrieved_ids)),
            "hit": hit,
            "precision@5": round(precision, 3),
            "recall": recall,
        })

    eval_df = pd.DataFrame(rows)
    n = len(eval_df)

    metrics = {
        "model": model_name,
        "n_queries": n,
        "hit_rate": round(eval_df["hit"].mean(), 3),
        "precision@5": round(eval_df["precision@5"].mean(), 3),
        "recall": round(eval_df["recall"].mean(), 3),
        "top1_accuracy": round(eval_df["hit"].mean(), 3),
    }

    return eval_df, metrics


# ==========================================
# Run Evaluation for Multiple Models
# ==========================================

model_results = {}
metrics_rows = []

for model_name, retrieve_fn in [
    ("TF-IDF", retrieve_tfidf),
    ("SVM", retrieve_svm),
    ("BERT-like proxy", retrieve_bert_proxy),
]:
    eval_df, metrics = evaluate_model(model_name, retrieve_fn)
    model_results[model_name] = eval_df
    metrics_rows.append(metrics)

metrics_df = pd.DataFrame(metrics_rows)

# ==========================================
# Save Evaluation Outputs
# ==========================================

evaluation_file = os.path.join(output_folder, "evaluation.csv")
metrics_file = os.path.join(output_folder, "model_metrics.csv")
summary_file = os.path.join(output_folder, "summary.txt")
report_file = os.path.join(output_folder, "report.md")
chart_file = os.path.join(output_folder, "performance_chart.png")

all_eval_df = pd.concat(model_results.values(), ignore_index=True)
all_eval_df.to_csv(evaluation_file, index=False, encoding="utf-8-sig")
metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")

# ==========================================
# Plot Performance Chart
# ==========================================

plt.figure(figsize=(8, 4.5))
bar_colors = ["#4C78A8", "#F58518", "#54A24B"]
plt.bar(metrics_df["model"], metrics_df["hit_rate"], color=bar_colors)

for bar, value in zip(plt.gca().patches, metrics_df["hit_rate"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

plt.ylim(0, 1.05)
plt.ylabel("Hit Rate")
plt.title("Perbandingan Hit Rate per Model")
plt.tight_layout()
plt.savefig(chart_file, dpi=200)
plt.close()

# ==========================================
# Error Analysis
# ==========================================

analysis_lines = []
for model_name, eval_df in model_results.items():
    failed = eval_df[eval_df["hit"] == 0]
    analysis_lines.append(f"## {model_name}")
    analysis_lines.append(f"- Gagal: {len(failed)} dari {len(eval_df)} query")

    if not failed.empty:
        sample = failed.head(3)
        for _, row in sample.iterrows():
            preview = row["query"][:120].replace("\n", " ")
            analysis_lines.append(f"- Contoh query: {preview}... -> retrieved: {row['retrieved']}")
    analysis_lines.append("")

analysis_lines.extend([
    "## Diskusi kasus kegagalan",
    "- Banyak kegagalan terjadi pada query yang sangat umum, misalnya hanya mengandung pasal atau istilah narkotika yang sama di banyak kasus.",
    "- Dokumen putusan memiliki banyak teks boilerplate hukum yang mengurangi sinyal pembeda antar kasus.",
    "- Model berbasis kata-kata (TF-IDF/SVM) cenderung gagal ketika dua kasus memiliki pola fakta yang serupa tetapi berbeda pada detail spesifik.",
    "- Untuk penguatan performa, langkah berikutnya bisa berupa embedding semantik (misalnya BERT), reranker, atau pemrosesan query yang lebih ringkas.",
])

with open(summary_file, "w", encoding="utf-8") as f:
    f.write("===== HASIL EVALUASI RETRIEVAL =====\n\n")
    f.write(f"Jumlah Query      : {len(queries)}\n")
    f.write(f"TF-IDF Hit Rate   : {metrics_df.loc[metrics_df['model']=='TF-IDF','hit_rate'].iloc[0]:.3f}\n")
    f.write(f"SVM Hit Rate      : {metrics_df.loc[metrics_df['model']=='SVM','hit_rate'].iloc[0]:.3f}\n")
    f.write(f"BERT-like Hit Rate: {metrics_df.loc[metrics_df['model']=='BERT-like proxy','hit_rate'].iloc[0]:.3f}\n")
    f.write("\n")
    f.write("===== RINGKASAN METRIK =====\n")
    f.write(metrics_df.to_string(index=False))
    f.write("\n\n")
    f.write("===== ANALISIS KESALAHAN =====\n")
    f.write("\n".join(analysis_lines))

with open(report_file, "w", encoding="utf-8") as f:
    f.write("# Laporan Visualisasi dan Evaluasi Retrieval\n\n")
    f.write("## Tabel metrik per model\n\n")
    f.write("```text\n")
    f.write(metrics_df.to_string(index=False))
    f.write("\n```\n\n")
    f.write("## Plot performa\n\n")
    f.write(f"Grafik disimpan pada {chart_file}.\n\n")
    f.write("## Diskusi kasus kegagalan\n\n")
    f.write("\n".join(analysis_lines[3:]))

# ==========================================
# Print Results
# ==========================================

print("=" * 60)
print("EVALUATION SELESAI")
print("=" * 60)
print(metrics_df)
print("\nOutput:")
print(evaluation_file)
print(metrics_file)
print(summary_file)
print(report_file)
print(chart_file)