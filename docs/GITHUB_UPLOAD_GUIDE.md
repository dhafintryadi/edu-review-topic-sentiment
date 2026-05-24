# GitHub Upload Guide

Dokumen ini menjelaskan apa yang aman diunggah ke GitHub dari direktori PKL ini.

## File yang Aman Diunggah

- Kode Python pipeline:
  - `Crawler/google-play-scraper/*.py`
  - `Preprocessing/hybrid_preprocessing/**/*.py`
  - `Sentiment-Analysis/nn-based/*.py`
  - `Topic-Modelling/*.py`
- Konfigurasi dan dokumentasi:
  - `README.md`
  - `.gitignore`
  - `.gitattributes`
  - `requirements.txt`
  - file `.md` yang relevan
- Output ringkasan kecil:
  - file `.json` ringkasan eksperimen
  - file `.csv` kecil yang tidak memuat data mentah sensitif

## File yang Jangan Diunggah ke GitHub Biasa

- Virtual environment:
  - `.venv/`
  - `.venv-1/`
  - `venv/`
- Dataset besar:
  - `Datasets/*.csv`
  - `Crawler/google-play-scraper/datasets/*.csv`
- Model dan checkpoint:
  - `*.bin`
  - `*.bin.gz`
  - `*.pkl`
  - `*.safetensors`
  - `*.npy`
  - folder `models/` dan `checkpoints/`
- Dokumen pribadi/laporan:
  - `*.docx`
  - `*.pdf`
  - `*.pptx`
- Clone dependency/vendor:
  - `Topic-Modelling/gensim/`
  - `Topic-Modelling/pyLDAvis/`
  - `Sentiment-Analysis/fastText/`
  - `Sentiment-Analysis/simpletransformers/`
  - `Preprocessing/indonlu/`

## Langkah Upload yang Disarankan

1. Inisialisasi repository:

   ```bash
   git init
   ```

2. Cek file yang akan masuk staging:

   ```bash
   git status --short
   ```

3. Tambahkan file:

   ```bash
   git add .
   ```

4. Pastikan tidak ada file besar ikut ter-stage:

   ```bash
   git status --short
   ```

5. Commit:

   ```bash
   git commit -m "Initial PKL analysis pipeline"
   ```

6. Hubungkan ke GitHub:

   ```bash
   git branch -M main
   git remote add origin <URL_REPOSITORY_GITHUB>
   git push -u origin main
   ```

## Catatan Penting

Direktori ini sebelumnya belum berupa git repository. Jangan menjalankan `git add .` sebelum memastikan `.gitignore` sudah aktif dan file besar tidak masuk staging.

Jika dataset atau model perlu dibagikan, gunakan salah satu opsi berikut:

- GitHub Releases
- Git LFS
- Google Drive
- Zenodo
- Hugging Face Hub untuk model

