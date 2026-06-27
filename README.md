# TUGAS 3

Repository untuk tugas Computer-Based Reasoning (CBR) menggunakan putusan pidana dari Pengadilan Negeri Surabaya.

## Struktur

- `data/raw/` - file `.txt` hasil ekstraksi dari PDF putusan
- `data/processed/` - file `cases.json` yang berisi representasi kasus terstruktur
- `data/eval/` - file `queries.json` untuk evaluasi retrieval
- `data/results/` - hasil `retrieval.csv`, `reuse.csv`, dan `evaluation.csv`
- `notebooks/` - skrip Python untuk setiap tahap CBR

## Alur Notebook

1. `01_case_base.py` - menyiapkan basis kasus dan menyimpan data mentah
2. `02_case_representation.py` - ekstraksi metadata, ringkasan, dan pembuatan `cases.json`
3. `03_case_retrieval.py` - membuat model retrieval TF-IDF dan `queries.json`
4. `04_case_reuse.py` - mengambil kasus serupa dan merekomendasikan putusan
5. `05_evaluation.py` - evaluasi retrieval menggunakan query dan hit rate

## Penggunaan

1. Aktifkan virtual environment:

```bash
source venv/Scripts/activate
```

2. Pasang dependensi:

```bash
pip install -r requirements.txt
```

3. Jalankan masing-masing notebook sesuai urutan:

```bash
python notebooks/01_case_base.py
python notebooks/02_case_representation.py
python notebooks/03_case_retrieval.py
python notebooks/04_case_reuse.py
python notebooks/05_evaluation.py
```

## Catatan

- Dataset utama sekarang menggunakan `data/processed/cases.json`.
- File `cases.csv` tidak lagi digunakan dalam alur utama.
- Tambahkan `venv/`, `__pycache__/`, dan file notebook checkpoint ke `.gitignore`.
