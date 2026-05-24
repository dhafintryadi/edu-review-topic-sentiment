# AITF-2026 Sentiment Analysis Pipeline: End-to-End Report (Updated)

## Project Overview
**Research Focus**: Sentiment Analysis for Learning Barrier Identification in Indonesian Educational App Reviews
**Objective**: Establish comprehensive sentiment analysis pipeline to identify and quantify learning barriers through user feedback analysis.
**Methodology**: Pendekatan hybrid menggunakan teknik preprocessing teks tingkat lanjut, klasifikasi sentimen berbasis Logistic Regression (Baseline), dan integrasi dengan Topic Modeling untuk kuantifikasi *Severity* (tingkat keparahan) masalah.

**Update Date**: May 12, 2026
**Total Reviews Processed**: 13,951 (Clean Dataset)
**Pipeline Status**: Completed & Validated

---

## Data Integrity Audit & Cleanup
*Penting: Laporan ini telah diperbarui setelah melalui proses audit data pada 12 Mei 2026.*

### Findings
Ditemukan inkonsistensi jumlah data di mana dataset membengkak dari 15k menjadi 90k baris. Investigasi menunjukkan adanya redundansi data akibat penggunaan mode `append` yang berulang pada tahap preprocessing.

### Actions Taken
1.  **Cleanup**: Menghapus seluruh file hasil preprocessing yang terduplikasi.
2.  **Reprocessing**: Menjalankan ulang pipeline preprocessing langsung dari data mentah asli (`raw_review.csv`).
3.  **Normalization**: Memastikan dataset final hanya berisi ulasan unik berdasarkan `reviewId`.

---

## Phase 1: Data Preparation & Preprocessing
### Results
- **Dataset Size**: 13,951 unique reviews
- **Preprocessing steps**: 
    - Case folding & Cleaning (URLs, Emojis, Symbols)
    - Slang normalization & Indonesian Stemming
    - Stopword removal
- **Output**: `sentiment_preprocessed.csv`

---

## Phase 2: Baseline Model Training
### Model Configuration
- **Algorithm**: Logistic Regression with TF-IDF (1-2 n-grams)
- **Class Balancing**: Balanced Class Weights (Adjusted for severe imbalance)
- **Device**: CPU-based execution

### Performance Results (Real Baseline)
Metrik ini mencerminkan performa model pada data nyata tanpa gangguan duplikasi.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 79.51% |
| **Macro F1-Score** | 60.47% |
| **Macro Precision** | 59.00% |
| **Macro Recall** | 63.80% |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Positive** | 0.96 | 0.86 | 0.91 | 2,110 |
| **Negative** | 0.61 | 0.70 | 0.65 | 485 |
| **Neutral** | 0.20 | 0.36 | 0.26 | 196 |

*Catatan: Kelas Neutral tetap menjadi tantangan terbesar karena jumlah sampel yang sangat terbatas (~7% dari total data).*

---

## Phase 3: Topic Integration
Setiap ulasan dikategorikan ke dalam 8 learning barriers. Berikut adalah distribusi sentimen per topik pada dataset bersih:

| Topic Label | Total Reviews | Negative % | Neutral % | Positive % |
| :--- | :--- | :--- | :--- | :--- |
| Need for Scaffolding & Question Banks | 2,399 | 11.0% | 7.6% | 81.4% |
| Learning Comprehension & Clarity | 2,216 | 17.6% | 11.1% | 71.3% |
| Learner Motivation & Engagement | 2,081 | 13.7% | 8.9% | 77.4% |
| Incomplete Material & Content Gaps | 1,979 | 16.8% | 18.1% | 65.1% |
| Performance Lag & Update Errors | 1,490 | 29.0% | 13.7% | 57.3% |
| Core App Crashes & Login Failures | 1,343 | 33.2% | 11.4% | 55.4% |
| Pricing & Package Affordability | 1,307 | 27.1% | 11.8% | 61.1% |
| Authentication & Access Errors | 1,136 | 23.4% | 12.3% | 64.3% |

---

## Phase 4: Severity Analysis
Menghitung tingkat urgensi perbaikan menggunakan formula: `severity = negative_ratio × topic_frequency`.

### Top 5 Most Severe Learning Barriers
| Rank | Topic | Barrier | Severity | Level |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Core App Crashes & Login Failures** | TB-4 | 0.031969 | **High** |
| 2 | **Performance Lag & Update Errors** | TB-4 | 0.030966 | **High** |
| 3 | **Learning Comprehension & Clarity** | TB-1 | 0.027883 | Medium |
| 4 | **Pricing & Package Affordability** | TB-6 | 0.025375 | Medium |
| 5 | **Incomplete Material & Content Gaps** | TB-7 | 0.023869 | Medium |

---

## Research Insights & Conclusions
1.  **Technical Barrier Dominance**: Masalah teknis (Usability - TB-4) memiliki skor keparahan tertinggi. Hal ini menunjukkan bahwa pengguna seringkali terhambat sebelum sempat mengakses konten pembelajaran.
2.  **Data Quality Matters**: Audit data membuktikan bahwa akurasi tinggi (>90%) sebelumnya adalah semu akibat duplikasi. Akurasi 79.5% saat ini jauh lebih reliabel untuk pengambilan keputusan.
3.  **Neutral Class Ambiguity**: Diperlukan strategi pengumpulan data lebih lanjut untuk kelas netral guna meningkatkan kemampuan pembedaan model pada ulasan moderat.

---
**Report Finalized.**
*Signed by Antigravity AI Coding Assistant.*