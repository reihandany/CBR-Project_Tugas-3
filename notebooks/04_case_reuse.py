import os
import json
import pandas as pd

# =====================================
# Folder
# =====================================

cases_file = "data/processed/cases.json"
retrieval_file = "data/results/retrieval.csv"

output_folder = "data/results"

os.makedirs(output_folder, exist_ok=True)

# =====================================
# Load Data
# =====================================

with open(cases_file, "r", encoding="utf-8") as f:
    cases = json.load(f)

cases_df = pd.DataFrame(cases)

retrieval_df = pd.read_csv(retrieval_file)

print(f"Jumlah Case Base : {len(cases_df)}")
print(f"Jumlah Retrieval : {len(retrieval_df)}")

# =====================================
# Ambil Top-1 Retrieval
# =====================================

top1 = retrieval_df.sort_values(
    by=["query_case", "similarity"],
    ascending=[True, False]
).drop_duplicates(
    subset="query_case",
    keep="first"
)

print(f"Jumlah Query : {len(top1)}")

# =====================================
# Reuse
# =====================================

hasil = []

for _, row in top1.iterrows():

    similar_case = row["similar_case"]

    solusi = cases_df[
        cases_df["case_id"] == similar_case
    ]

    if len(solusi) == 0:

        amar = ""

        pidana = ""

        pasal = ""

        jenis = ""

    else:

        solusi = solusi.iloc[0]

        amar = solusi["amar_putusan"]

        pidana = solusi["lama_pidana"]

        pasal = solusi["pasal"]

        jenis = solusi["jenis_perkara"]

    hasil.append({

        "query_case": row["query_case"],

        "query_no_perkara": row["query_no_perkara"],

        "similar_case": similar_case,

        "similarity": row["similarity"],

        "jenis_perkara": jenis,

        "pasal_rekomendasi": pasal,

        "rekomendasi_putusan": amar,

        "rekomendasi_pidana": pidana

    })

# =====================================
# DataFrame
# =====================================

reuse_df = pd.DataFrame(hasil)

# =====================================
# Simpan
# =====================================

output_file = os.path.join(
    output_folder,
    "reuse.csv"
)

reuse_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# =====================================
# Tampilkan
# =====================================

print("\n")
print("="*60)
print("CASE REUSE SELESAI")
print("="*60)

print(reuse_df)

print("\nJumlah rekomendasi :", len(reuse_df))

print("Output :", output_file)