# Edu Review Topic & Sentiment Analysis

Analisis ulasan pengguna aplikasi edukasi digital untuk mengidentifikasi hambatan belajar, pola sentimen, dan implikasi desain sistem pembelajaran adaptif pada konteks Sekolah Rakyat.

---

## 📖 Background

Platform pembelajaran digital sering dinilai hanya dari rating aplikasi atau jumlah pengguna. Untuk kebutuhan pendidikan, metrik tersebut belum cukup. Ulasan pengguna dapat memuat sinyal penting tentang kesulitan memahami materi, gangguan teknis, kebutuhan latihan soal, masalah akses, harga, dan kualitas konten.

Proyek ini menggunakan Natural Language Processing (NLP) Bahasa Indonesia untuk mengubah ulasan pengguna menjadi insight yang lebih terstruktur. Fokus utamanya bukan hanya klasifikasi sentimen, tetapi juga pemetaan topik keluhan dan kebutuhan pengguna menjadi learning barrier yang dapat digunakan sebagai dasar desain sistem pembelajaran adaptif Sekolah Rakyat.

---

## ❓ Problem Statement

Bagaimana ulasan pengguna aplikasi edukasi dapat digunakan untuk mengidentifikasi hambatan belajar dan prioritas fitur sistem pembelajaran adaptif secara lebih sistematis?

---

## 🎯 Objectives

1. Mengumpulkan dan memvalidasi ulasan aplikasi edukasi dari Google Play Store.
2. Membersihkan dan menormalisasi teks Bahasa Indonesia agar siap digunakan untuk sentiment analysis dan topic modelling.
3. Membangun baseline sentiment analysis untuk memahami polaritas ulasan.
4. Menggunakan topic modelling untuk menemukan kelompok isu utama dalam pengalaman belajar digital.
5. Memetakan topic dan sentiment menjadi learning barrier serta rekomendasi desain untuk sistem Sekolah Rakyat.

---

## 📊 Dataset

Sumber data berasal dari hasil crawling ulasan aplikasi edukasi digital di Google Play Store.

| Dataset | Keterangan |
|---|---|
| Raw review | 15.324 baris ulasan berhasil dimuat |
| Kolom utama | `reviewId`, `userName`, `score`, `content`, `at`, `thumbsUpCount`, `replyContent`, `repliedAt`, `appVersion` |
| Target sentiment | Diturunkan dari rating/label ulasan |
| Input topic modelling | Teks ulasan yang sudah dibersihkan dan diproses |

**Catatan kualitas data:**
- Terdapat missing value pada `replyContent`, `repliedAt`, dan `appVersion`.
- Distribusi rating tidak seimbang, dengan rating 5 mendominasi.
- Dataset hasil preprocessing memiliki duplikasi yang perlu diperhatikan pada tahap validasi.
- File CSV besar disimpan secara lokal di direktori `assets/` dan tidak di-commit langsung ke GitHub.

---

## ⚙️ Methodology

Workflow proyek mengikuti alur NLP end-to-end:

```text
Google Play Store Reviews
-> Crawling
-> Dataset Validation & Merge
-> Text Cleaning & Normalization
-> Sentiment Analysis (TF-IDF + Logistic Regression)
-> Topic Modelling (LDA K=8)
-> Learning Barrier Severity Mapping
-> Design Implication for Sekolah Rakyat Adaptive Learning System
```

---

## 📂 Struktur Repositori (Production-Grade Layout)

Repositori ini telah dimodularisasi menjadi arsitektur yang bersih dan decoupled untuk kebutuhan production:

```text
edu-review-topic-sentiment/
├── core/                           # ← Paket kode utama (Production Engine)
│   ├── preprocessor.py             #   Wrapper orchestrator untuk preprocessing
│   ├── sentiment_engine.py         #   Engine inferensi TF-IDF + Logistic Regression
│   ├── topic_engine.py             #   Engine inferensi LDA K=8 (assets-anchored)
│   ├── severity_analyzer.py        #   Engine analisis severitas & implikasi sistem (Phase 4 & 5)
│   ├── preprocessing/              #   Modul tokenisasi, pembersihan, dan normalisasi teks
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   ├── normalization.py
│   │   ├── pipeline.py
│   │   └── tokenizer.py
│   ├── resources/                  #   Kamus slang (slang_dict.json) & stopwords (stopwords.txt)
│   │   ├── slang_dict.json
│   │   └── stopwords.txt
│   └── training/                   #   Script training historis (hanya untuk kebutuhan audit)
│       ├── train_baseline_model.py
│       └── phase3b_final_lda_training.py
├── assets/                         # ← Model binaries, dataset statis, dan acuan taksonomi
│   ├── raw_review.csv
│   ├── baseline_model.pkl          #   Pickle model sentiment & vectorizer
│   ├── lda_model_final_k8.gensim   #   Model LDA final beserta file state pendukungnya
│   ├── lda_dictionary.gensim
│   ├── lda_ready_corpus.csv
│   ├── phase3b_topic_taxonomy.json #   Sistem taksonomi topik keluhan
│   └── sr_blueprint_validation.json #  Matriks validasi blueprint Sekolah Rakyat
├── artifacts/                      # ← Output matriks dan spesifikasi sistem hasil eksekusi pipeline
├── run_pipeline.py                 # ← Orchestrator pipeline utama berbasis native Python
├── setup_assets.py                 # ← Script inisialisasi & penyalinan aset model biner
├── requirements.txt                # ← Dependensi sistem
└── README.md                       # ← Dokumentasi proyek
```

---

## 🛠️ Preprocessing

Pipeline preprocessing berada di `core/preprocessing/` dan diakses melalui `core/preprocessor.py` yang mencakup:
- Case folding.
- Pembersihan URL, simbol, noise, dan karakter tidak relevan.
- Normalisasi kata informal/slang (menggunakan kamus di `core/resources/slang_dict.json`).
- Tokenisasi.
- Stopword handling (menggunakan daftar di `core/resources/stopwords.txt`).
- Stemming Bahasa Indonesia menggunakan library Sastrawi.
- Pemisahan output untuk kebutuhan sentiment analysis dan topic modelling.

Preprocessing dibuat terpisah karena kebutuhan sentiment analysis dan topic modelling tidak selalu sama. Sentiment analysis perlu menjaga sinyal polaritas, sedangkan topic modelling perlu representasi kata yang lebih stabil untuk pembentukan topik.

---

## 📈 Sentiment Analysis

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

---

## 🤖 Topic Modelling

Topic modelling menggunakan LDA untuk mengidentifikasi kelompok isu utama dari ulasan pengguna.

Ringkasan hasil final:

| Item | Value |
|---|---|
| Model | LDA |
| Optimal topic count | 8 |
| Vocabulary size | 2.689 |
| Documents used | 12.017 |
| LDA passes | 30 |
| LDA iterations | 150 |

---

## 🔍 Learning Barrier Findings

Hasil topic modelling dipetakan menjadi taxonomy learning barrier:

| Topic | Learning Barrier | Category |
|---:|---|---|
| 0 | Incomplete Material & Content Gaps | TB-7 Content Quality Mismatch |
| 1 | Learning Comprehension & Clarity | TB-1 Cognitive Difficulty |
| 2 | Pricing & Package Affordability | TB-6 Cost / Affordability Barrier |
| 3 | Need for Scaffolding & Question Banks | TB-3 Lack of Learning Support |
| 4 | Learner Motivation & Engagement | TB-2 Engagement & Motivation Problem |
| 5 | Core App Crashes & Login Failures | TB-4 System Usability Issues |
| 6 | Performance Lag & Update Errors | TB-4 System Usability Issues |
| 7 | Authentication & System Access Errors | TB-4 System Usability Issues |

Prioritas desain yang paling kuat dari hasil analisis:
1. **`TB-4 System Usability Issues`**: Prioritas kritis karena gangguan teknis menghalangi seluruh akses belajar pengguna.
2. **`TB-7 Content Quality Mismatch`**: Prioritas kritis karena materi yang tidak sesuai level membuat personalisasi tidak efektif.
3. **`TB-6 Cost / Affordability Barrier`**: Prioritas tinggi karena akses ekonomi memengaruhi keberlanjutan belajar.
4. **`TB-3 Lack of Learning Support`**: Kebutuhan terhadap AI tutor, hint, dan perluasan bank soal.

---

## 💡 Key Insights

- Masalah teknis seperti crash, login failure, loading, dan update error bukan sekadar masalah aplikasi, tetapi hambatan akses belajar.
- Kualitas dan kesesuaian materi menjadi isu besar; sistem adaptif perlu diagnostic placement dan sequencing berbasis prerequisite.
- Pengguna membutuhkan scaffolding berupa pembahasan soal, bank soal, hint, dan dukungan belajar yang lebih kontekstual.
- Sentiment analysis berguna sebagai sinyal pendukung, tetapi topic modelling memberikan insight yang lebih actionable untuk desain sistem.
- Dataset rating cenderung tidak seimbang, sehingga evaluasi model harus melihat macro-F1 dan performa per kelas, bukan accuracy saja.

---

## 📋 Recommendations

Rekomendasi desain untuk sistem pembelajaran adaptif:
- Bangun fondasi reliabilitas sistem lebih dulu: offline-first mode, session restore, dan graceful degradation.
- Tambahkan diagnostic placement test agar siswa masuk ke level materi yang sesuai.
- Gunakan prerequisite knowledge graph untuk menghindari mismatch materi.
- Sediakan AI tutor kontekstual, scaffolded hints, dan dynamic question bank.
- Pertimbangkan freemium core learning path atau subsidi untuk konteks akses pendidikan publik.
- Tambahkan monitoring sentimen dan topic drift agar masalah baru bisa terdeteksi secara berkala.

---

## 📐 Arsitektur Sistem & Pemetaan Disiplin (DE vs DS/A)

Untuk menjamin skalabilitas dan stabilitas dalam fase production, kode sistem dipisahkan berdasarkan fokus Data Engineering (infrastruktur & reliabilitas) dan Data Science/Analytics (metrik statistik & model logika).

### 1. Data Engineering (DE) Focus Details
* **Optimasi Memori (Chunking):** Proses pembersihan data ulasan dilakukan secara bertahap dalam ukuran chunk `5000` baris menggunakan iterator `pd.read_csv`. Hal ini menjaga penggunaan RAM tetap rendah dan mencegah crash akibat kehabisan memori (OOM).
* **Idempotent I/O Guards:** Guna menghindari duplikasi data ketika pipeline dijalankan berulang kali, sistem secara otomatis menghapus file output lama sebelum mulai memproses chunk data yang baru:
  ```python
  for _stale in (sentiment_path, topic_path):
      if os.path.exists(_stale):
          os.remove(_stale)
  ```
* **Portabilitas Lokasi Sistem (`pathlib`):** Menggunakan pustaka `pathlib` untuk mendeteksi jalur folder secara dinamis berdasarkan letak modul yang dieksekusi, sehingga pipeline berjalan tanpa hambatan baik di Windows maupun Linux.

### 2. Data Science & Analytics (DS/A) Focus Details
* **Inference Sentimen NLP:** Sistem menggunakan `TfidfVectorizer` untuk mengekstrak bobot fitur teks ulasan, lalu melakukan klasifikasi menggunakan `LogisticRegression` secara batch untuk memprediksi sentimen (`predicted_label_name`) lengkap dengan confidence score terkalibrasi.
* **Keamanan Operasi Matematika (Zero-Division Guards):** Perhitungan skor keparahan menggabungkan nilai frekuensi topik ($f$) dan rasio negatif ($r_n$). Pengaman bersyarat dipasang untuk mencegah eror pembagian dengan nol (`ZeroDivisionError`) jika suatu topik memiliki frekuensi nol:
  ```python
  negative_ratio = len(neg_df) / freq if freq > 0 else 0
  mean_neg_conf  = neg_df["calibrated_confidence"].mean() if len(neg_df) > 0 else 0.0
  ```
* **Deterministic Rule Engine:** Menggunakan parser logika bawaan python di `core/severity_analyzer.py` untuk mengolah ranking keparahan dan merumuskan spesifikasi perilaku sistem adaptif Sekolah Rakyat secara deterministik.

### 3. Matriks Sintesis Disiplin

| Disiplin | File Modul Target | Pola Kode Konkret / Konteks Baris | Nilai Strategis Stabilitas Proyek |
| :--- | :--- | :--- | :--- |
| **Data Engineering** | `core/preprocessing/pipeline.py` | `pd.read_csv(..., chunksize=chunksize)` | Mencegah konsumsi RAM berlebih (OOM) saat memproses ulasan mentah skala besar. |
| **Data Engineering** | `core/preprocessing/pipeline.py` | `if os.path.exists(_stale): os.remove(_stale)` | Menjamin sifat idempotent pipeline, menghindari duplikasi data ulasan pada penulisan ulang. |
| **Data Engineering** | `core/preprocessor.py` | `Path(__file__).resolve().parent` | Mengeliminasi path absolut manual, membuat proyek portable lintas OS. |
| **Data Science & Analytics** | `core/sentiment_engine.py` | `vectorizer.transform(texts)` & `model.predict()` | Mengubah teks normal menjadi representasi bobot matematika dan mengklasifikasikan sentimen. |
| **Data Science & Analytics** | `core/topic_engine.py` | `negative_ratio = len(neg_df) / freq if freq > 0 else 0` | Mengamankan perhitungan statistik dari eror pembagian matematika. |
| **Data Science & Analytics** | `core/severity_analyzer.py` | `feature_requirements = { "TB-4 ...": ... }` | Menerjemahkan bobot severitas menjadi aturan kontrol sistem Sekolah Rakyat (FSM). |

---

## ⚡ Cara Menjalankan Pipeline

### Inisialisasi Dependensi
Pasang seluruh pustaka Python yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### Langkah 1: Setup Aset & Model
Salin data mentah dan model klasifikasi baseline ke struktur folder proyek:
```bash
python setup_assets.py
```

### Langkah 2: Jalankan Pipeline Analisis
Eksekusi pipeline analisis sentimen, pemodelan topik, dan perumusan implikasi desain secara end-to-end:
```bash
python run_pipeline.py
```

### Output Telemetri Pipeline yang Diharapkan:
```text
[CHECK] Verifying optimal model binaries and datasets... [OK]
[DEMO] 1. Preprocessing Raw Reviews... [SUCCESS]
[DEMO] 2. Running Sentiment Inference (Verified Pre-trained Model)... [SUCCESS]
[DEMO] 3. Computing Topic-Sentiment Severity Matrix... [SUCCESS]
[DEMO] 4. Generating Sekolah Rakyat Validation Artifacts... [SUCCESS]
```

Seluruh berkas luaran spesifikasi adaptif (implikasi desain, protokol resiliensi, and FSM aturan behavior) akan disimpan secara otomatis dalam format JSON di folder `artifacts/`.

---

## ⚠️ Limitations

- Ulasan Google Play tidak selalu merepresentasikan seluruh populasi siswa.
- Rating pengguna tidak selalu sejajar dengan kualitas pengalaman belajar.
- Label sentimen berbasis rating dapat mengandung noise.
- Topic modelling LDA membutuhkan interpretasi manusia agar topik tidak dibaca secara terlalu literal.
- Dataset besar dan model biner berukuran besar dikelola secara lokal pada folder `assets/`.

---

## 🚀 Future Improvements

- Tambahkan dashboard visual untuk distribusi topic, sentiment, dan severity.
- Gunakan BERTopic atau embedding-based clustering sebagai pembanding LDA.
- Tambahkan SHAP atau error analysis untuk model sentiment.
- Buat pipeline reproducible end-to-end dengan satu entrypoint CLI.
- Tambahkan sample dataset kecil agar reviewer bisa menjalankan demo tanpa file CSV besar.
- Dokumentasikan data schema dan artifact lineage dengan lebih formal.

---

## 🎯 Project Scope

Scope paling aman untuk presentasi portfolio atau laporan PKL:
1. Crawling dan validasi ulasan pengguna.
2. Preprocessing NLP Bahasa Indonesia.
3. Sentiment analysis sebagai sinyal pendukung.
4. Topic modelling sebagai analisis utama.
5. Mapping learning barrier ke rekomendasi desain sistem Sekolah Rakyat.
