# AITF UB x Komdigi 2026 - PKL Learning Barrier Analysis

Repository ini berisi pipeline analisis ulasan pengguna platform edukasi digital untuk mendukung penyusunan laporan PKL dan validasi konseptual kebutuhan sistem pembelajaran adaptif pada konteks Sekolah Rakyat.

## Ringkasan Pipeline

```text
Google Play Store
-> Crawler/google-play-scraper/
-> Datasets/raw_review.csv
-> Preprocessing/hybrid_preprocessing/
-> Datasets/sentiment_preprocessed.csv
-> Datasets/topic_preprocessed.csv
-> Sentiment-Analysis/nn-based/
-> Topic-Modelling/
-> Output topic-sentiment dan severity
```

## Struktur Utama

| Folder | Fungsi |
|---|---|
| `Crawler/google-play-scraper/` | Script crawling ulasan Google Play Store untuk aplikasi edukasi digital. |
| `Datasets/` | Script validasi/merge dataset. File CSV besar diabaikan oleh Git. |
| `Preprocessing/hybrid_preprocessing/` | Pipeline preprocessing teks Bahasa Indonesia untuk sentimen dan topic modelling. |
| `Sentiment-Analysis/nn-based/` | Baseline analisis sentimen berbasis Logistic Regression + TF-IDF. |
| `Topic-Modelling/` | Pipeline LDA, eksplorasi jumlah topik, interpretasi topik, dan mapping learning barriers. |
| `docs/` | Catatan repository dan panduan upload GitHub. |

## Referensi Repository Eksternal

Selain folder pipeline utama, workspace pengembangan lokal juga memanfaatkan beberapa repository eksternal sebagai referensi teknis, baseline implementasi, atau pembanding eksperimen. Folder-folder tersebut bukan seluruhnya merupakan kode inti yang dikembangkan dari awal dalam proyek PKL ini. Beberapa folder diabaikan oleh Git melalui `.gitignore` karena berukuran besar, berupa clone repository lain, atau hanya digunakan sebagai rujukan lokal selama pengembangan pipeline.

| Folder Referensi | Peran dalam Pengembangan Pipeline | Status Upload |
|---|---|---|
| `Crawler/google-play-scraper/` | Referensi dan basis crawler Google Play Store yang kemudian disesuaikan untuk kebutuhan pengambilan ulasan aplikasi edukasi digital. | Script crawler proyek tetap diunggah; dataset hasil crawling diabaikan. |
| `Preprocessing/indonlu/` | Referensi praktik NLP Bahasa Indonesia dan benchmark pemrosesan teks. | Diabaikan dari Git karena merupakan repository eksternal. |
| `Preprocessing/PySastrawi/` | Referensi stemming Bahasa Indonesia. Pipeline utama menggunakan dependency `Sastrawi`, bukan menyimpan seluruh clone repository. | Diabaikan dari Git. |
| `Preprocessing/text-normalization/` | Referensi normalisasi teks informal/slang. | Diabaikan dari Git. |
| `Sentiment-Analysis/fastText/` | Referensi dan eksperimen pembanding untuk klasifikasi teks ringan. | Diabaikan dari Git. |
| `Sentiment-Analysis/IndoBERTweet/` | Referensi eksperimen berbasis transformer untuk Bahasa Indonesia. | Diabaikan dari Git. |
| `Sentiment-Analysis/simpletransformers/` | Referensi framework eksperimen transformer. | Diabaikan dari Git. |
| `Topic-Modelling/gensim/` | Referensi pustaka topic modelling. Pipeline utama memakai package `gensim`, bukan menyimpan seluruh source repository. | Diabaikan dari Git. |
| `Topic-Modelling/pyLDAvis/` | Referensi visualisasi topic modelling. | Diabaikan dari Git. |

Dengan pemisahan ini, repository GitHub difokuskan pada kode pipeline PKL, konfigurasi, dokumentasi, dan ringkasan output yang relevan. Repository eksternal tetap disebut sebagai referensi pengembangan, tetapi tidak ikut diunggah sebagai vendored dependency agar ukuran repository tetap wajar dan riwayat kode lebih bersih.

## Catatan Data dan Artefak

File dataset mentah, model, virtual environment, checkpoint, dan dokumen laporan tidak dimasukkan ke Git karena ukuran besar atau bersifat lokal/pribadi. Daftar aturan ignore ada di `.gitignore`.

Artefak yang perlu disimpan di luar GitHub biasa:

- `cc.id.300.bin` dan `cc.id.300.bin.gz`
- file CSV besar di `Datasets/`
- checkpoint/model transformer
- file laporan `.docx`, `.pdf`, dan `.pptx`

Jika artefak besar perlu dipublikasikan, gunakan GitHub Releases, Google Drive, Zenodo, atau Git LFS.

## Cara Menjalankan Bagian Utama

Preprocessing:

```bash
python Preprocessing/hybrid_preprocessing/run_preprocessing.py --input Datasets/raw_review.csv --output-dir Datasets --chunksize 2000
```

Training baseline sentimen:

```bash
python Sentiment-Analysis/nn-based/train_baseline_model.py
python Sentiment-Analysis/nn-based/batch_inference.py
```

Topic modelling:

```bash
python Topic-Modelling/phase1_preprocessing.py
python Topic-Modelling/phase2_dictionary_corpus.py
python Topic-Modelling/phase3a_lda_exploration.py
python Topic-Modelling/phase3b_final_lda_training.py
```

## Status Scope

Untuk laporan PKL, scope yang paling aman adalah:

1. crawling dan preprocessing ulasan pengguna,
2. topic modelling sebagai analisis utama,
3. sentimen berbasis rating/model sederhana sebagai pendukung,
4. pemetaan learning barriers ke kebutuhan konseptual Sekolah Rakyat.
