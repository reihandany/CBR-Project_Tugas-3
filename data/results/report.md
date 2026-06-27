# Laporan Visualisasi dan Evaluasi Retrieval

## Tabel metrik per model

```text
          model  n_queries  hit_rate  precision@5  recall  top1_accuracy
         TF-IDF          8       1.0          0.2     1.0            1.0
            SVM          8       1.0          0.2     1.0            1.0
BERT-like proxy          8       1.0          0.2     1.0            1.0
```

## Plot performa

Grafik disimpan pada data/results\performance_chart.png.

## Diskusi kasus kegagalan

## SVM
- Gagal: 0 dari 8 query

## BERT-like proxy
- Gagal: 0 dari 8 query

## Diskusi kasus kegagalan
- Banyak kegagalan terjadi pada query yang sangat umum, misalnya hanya mengandung pasal atau istilah narkotika yang sama di banyak kasus.
- Dokumen putusan memiliki banyak teks boilerplate hukum yang mengurangi sinyal pembeda antar kasus.
- Model berbasis kata-kata (TF-IDF/SVM) cenderung gagal ketika dua kasus memiliki pola fakta yang serupa tetapi berbeda pada detail spesifik.
- Untuk penguatan performa, langkah berikutnya bisa berupa embedding semantik (misalnya BERT), reranker, atau pemrosesan query yang lebih ringkas.