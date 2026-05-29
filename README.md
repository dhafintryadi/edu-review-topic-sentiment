# Edu Review Topic & Sentiment Analysis

Analisis ulasan pengguna aplikasi edukasi digital untuk mengidentifikasi hambatan belajar, pola sentimen, dan implikasi desain sistem pembelajaran adaptif pada konteks Sekolah Rakyat.

## Background

Platform pembelajaran digital sering dinilai hanya dari rating aplikasi atau jumlah pengguna. Untuk kebutuhan pendidikan, metrik tersebut belum cukup. Ulasan pengguna dapat memuat sinyal penting tentang kesulitan memahami materi, gangguan teknis, kebutuhan latihan soal, masalah akses, harga, dan kualitas konten.

Proyek ini menggunakan Natural Language Processing (NLP) Bahasa Indonesia untuk mengubah ulasan pengguna menjadi insight yang lebih terstruktur. Fokus utamanya bukan hanya klasifikasi sentimen, tetapi juga pemetaan topik keluhan dan kebutuhan pengguna menjadi learning barrier yang dapat digunakan sebagai dasar desain sistem pembelajaran adaptif.

## Problem Statement

Bagaimana ulasan pengguna aplikasi edukasi dapat digunakan untuk mengidentifikasi hambatan belajar dan prioritas fitur sistem pembelajaran adaptif secara lebih sistematis?

## Objectives

1. Mengumpulkan dan memvalidasi ulasan aplikasi edukasi dari Google Play Store.
2. Membersihkan dan menormalisasi teks Bahasa Indonesia agar siap digunakan untuk sentiment analysis dan topic modelling.
3. Membangun baseline sentiment analysis untuk memahami polaritas ulasan.
4. Menggunakan topic modelling untuk menemukan kelompok isu utama dalam pengalaman belajar digital.
5. Memetakan topic dan sentiment menjadi learning barrier serta rekomendasi desain untuk sistem Sekolah Rakyat.

## Dataset

Sumber data berasal dari hasil crawling ulasan aplikasi edukasi digital di Google Play Store.

| Dataset | Keterangan |
|---|---|
| Raw review | 15.324 baris ulasan berhasil dimuat |
| Kolom utama | `reviewId`, `userName`, `score`, `content`, `at`, `thumbsUpCount`, `replyContent`, `repliedAt`, `appVersion` |
| Target sentiment | Diturunkan dari rating/label ulasan |
| Input topic modelling | Teks ulasan yang sudah dibersihkan dan diproses |

Catatan kualitas data:

- Terdapat missing value pada `replyContent`, `repliedAt`, dan `appVersion`.
- Distribusi rating tidak seimbang, dengan rating 5 mendominasi.
- Dataset hasil preprocessing memiliki duplikasi yang perlu diperhatikan pada tahap validasi.
- File CSV besar tidak disimpan di GitHub biasa dan perlu dikelola melalui storage eksternal atau Git LFS.

## Methodology

Workflow proyek mengikuti alur NLP end-to-end:

```text
Google Play Store Reviews
-> Crawling
-> Dataset Validation & Merge
-> Text Cleaning & Normalization
-> Sentiment Analysis
-> Topic Modelling
-> Learning Barrier Mapping
-> Design Implication for Adaptive Learning System
```

## Repository Structure

```text
edu-review-topic-sentiment/
|-- Crawler/
|   `-- google-play-scraper/       # Review crawling and app ID utilities
|-- Datasets/                      # Dataset validation and merge scripts
|-- Preprocessing/
|   `-- hybrid_preprocessing/      # Indonesian text preprocessing pipeline
|-- Sentiment-Analysis/
|   |-- nn-based/                  # TF-IDF / classic ML sentiment baseline
|   `-- transformer-based/         # Transformer experiment utilities
|-- Topic-Modelling/               # LDA, barrier mapping, and design synthesis
|-- docs/                          # GitHub upload and project notes
|-- dataset_quality_report.json
|-- validation_metrics.json
|-- requirements.txt
`-- README.md
```

## Preprocessing

Pipeline preprocessing berada di `Preprocessing/hybrid_preprocessing/` dan mencakup:

- case folding,
- pembersihan URL, simbol, noise, dan karakter tidak relevan,
- normalisasi kata informal/slang,
- tokenisasi,
- stopword handling,
- stemming Bahasa Indonesia dengan Sastrawi,
- pemisahan output untuk kebutuhan sentiment analysis dan topic modelling.

Preprocessing dibuat terpisah karena kebutuhan sentiment analysis dan topic modelling tidak selalu sama. Sentiment analysis perlu menjaga sinyal polaritas, sedangkan topic modelling perlu representasi kata yang lebih stabil untuk pembentukan topik.

## Sentiment Analysis

Baseline sentiment analysis menggunakan pendekatan klasik berbasis fitur teks. Hasil baseline utama:

| Metric | Value |
|---|---:|
| Accuracy | 0.7951 |
| Macro Precision | 0.5902 |
| Macro Recall | 0.6385 |
| Macro F1 | 0.6050 |

Performa per kelas menunjukkan bahwa kelas positif jauh lebih mudah diprediksi dibanding kelas netral. Ini konsisten dengan distribusi rating yang tidak seimbang dan menjadi alasan mengapa macro-F1 lebih penting daripada accuracy saja.

Terdapat juga eksperimen tambahan berbasis FastText dengan metrik validasi:

| Metric | Value |
|---|---:|
| Accuracy | 0.8933 |
| Precision | 0.8131 |
| Recall | 0.6665 |
| Macro F1 | 0.7044 |

## Topic Modelling

Topic modelling menggunakan LDA untuk mengidentifikasi kelompok isu utama dari ulasan pengguna.

Ringkasan hasil final:

| Item | Value |
|---|---:|
| Model | LDA |
| Optimal topic count | 8 |
| Vocabulary size | 2.689 |
| Documents used | 12.017 |
| LDA passes | 30 |
| LDA iterations | 150 |

Artefak topic modelling tersimpan di beberapa folder output:

- `Topic-Modelling/phase3a_outputs/`
- `Topic-Modelling/phase3b_outputs/`
- `Topic-Modelling/phase3c_outputs/`
- `Topic-Modelling/phase4_outputs/`
- `Topic-Modelling/phase5_outputs/`
- `Topic-Modelling/research_output/`

## Learning Barrier Findings

Hasil topic modelling dipetakan menjadi taxonomy learning barrier:

| Topic | Learning Barrier | Category |
|---:|---|---|
| 0 | Incomplete Material & Content Gaps | Content Quality Mismatch |
| 1 | Learning Comprehension & Clarity | Cognitive Difficulty |
| 2 | Pricing & Package Affordability | Cost / Affordability Barrier |
| 3 | Need for Scaffolding & Question Banks | Lack of Learning Support |
| 4 | Learner Motivation & Engagement | Engagement & Motivation Problem |
| 5 | Core App Crashes & Login Failures | System Usability Issues |
| 6 | Performance Lag & Update Errors | System Usability Issues |
| 7 | Authentication & System Access Errors | System Usability Issues |

Prioritas desain yang paling kuat dari hasil analisis:

1. `TB-4 System Usability Issues`: prioritas kritis karena gangguan teknis menghalangi akses belajar.
2. `TB-7 Content Quality Mismatch`: prioritas kritis karena materi yang tidak sesuai level membuat personalisasi tidak efektif.
3. `TB-6 Cost / Affordability Barrier`: prioritas tinggi karena akses ekonomi memengaruhi keberlanjutan belajar.
4. `TB-3 Lack of Learning Support`: kebutuhan terhadap AI tutor, hint, dan perluasan bank soal.

## Key Insights

- Masalah teknis seperti crash, login failure, loading, dan update error bukan sekadar masalah aplikasi, tetapi hambatan akses belajar.
- Kualitas dan kesesuaian materi menjadi isu besar; sistem adaptif perlu diagnostic placement dan sequencing berbasis prerequisite.
- Pengguna membutuhkan scaffolding berupa pembahasan soal, bank soal, hint, dan dukungan belajar yang lebih kontekstual.
- Sentiment analysis berguna sebagai sinyal pendukung, tetapi topic modelling memberikan insight yang lebih actionable untuk desain sistem.
- Dataset rating cenderung tidak seimbang, sehingga evaluasi model harus melihat macro-F1 dan performa per kelas, bukan accuracy saja.

## Recommendations

Rekomendasi desain untuk sistem pembelajaran adaptif:

- Bangun fondasi reliabilitas sistem lebih dulu: offline-first mode, session restore, dan graceful degradation.
- Tambahkan diagnostic placement test agar siswa masuk ke level materi yang sesuai.
- Gunakan prerequisite knowledge graph untuk menghindari mismatch materi.
- Sediakan AI tutor kontekstual, scaffolded hints, dan dynamic question bank.
- Pertimbangkan freemium core learning path atau subsidi untuk konteks akses pendidikan publik.
- Tambahkan monitoring sentimen dan topic drift agar masalah baru bisa terdeteksi secara berkala.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python Preprocessing/hybrid_preprocessing/run_preprocessing.py --input Datasets/raw_review.csv --output-dir Datasets --chunksize 2000
```

Run sentiment baseline:

```bash
python Sentiment-Analysis/nn-based/train_baseline_model.py
python Sentiment-Analysis/nn-based/batch_inference.py
```

Run topic modelling phases:

```bash
python Topic-Modelling/phase1_preprocessing.py
python Topic-Modelling/phase2_dictionary_corpus.py
python Topic-Modelling/phase3a_lda_exploration.py
python Topic-Modelling/phase3b_final_lda_training.py
python Topic-Modelling/run_phase3c.py
python Topic-Modelling/run_phase4.py
python Topic-Modelling/run_phase5.py
```

## Outputs

Important output files include:

- `dataset_quality_report.json`
- `validation_metrics.json`
- `Sentiment-Analysis/nn-based/baseline_metrics.json`
- `Topic-Modelling/phase3b_outputs/phase3b_summary.json`
- `Topic-Modelling/research_output/learning_barrier_identification.json`
- `Topic-Modelling/phase4_outputs/sr_design_implication_matrix.json`
- `Topic-Modelling/phase5_outputs/sr_system_logic_specification.json`

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- imbalanced-learn
- gensim
- Sastrawi
- NLTK
- google-play-scraper
- FastText / transformer experiment utilities

## Limitations

- Ulasan Google Play tidak selalu merepresentasikan seluruh populasi siswa.
- Rating pengguna tidak selalu sejajar dengan kualitas pengalaman belajar.
- Label sentimen berbasis rating dapat mengandung noise.
- Topic modelling LDA membutuhkan interpretasi manusia agar topik tidak dibaca secara terlalu literal.
- Dataset besar dan model artefact tidak seluruhnya disimpan di repository GitHub.

## Future Improvements

- Tambahkan dashboard visual untuk distribusi topic, sentiment, dan severity.
- Gunakan BERTopic atau embedding-based clustering sebagai pembanding LDA.
- Tambahkan SHAP atau error analysis untuk model sentiment.
- Buat pipeline reproducible end-to-end dengan satu entrypoint CLI.
- Tambahkan sample dataset kecil agar reviewer bisa menjalankan demo tanpa file CSV besar.
- Dokumentasikan data schema dan artifact lineage dengan lebih formal.

## Project Scope

Scope paling aman untuk presentasi portfolio atau laporan PKL:

1. crawling dan validasi ulasan pengguna,
2. preprocessing NLP Bahasa Indonesia,
3. sentiment analysis sebagai sinyal pendukung,
4. topic modelling sebagai analisis utama,
5. mapping learning barrier ke rekomendasi desain sistem Sekolah Rakyat.


