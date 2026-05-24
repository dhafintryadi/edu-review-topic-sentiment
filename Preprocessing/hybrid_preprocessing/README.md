# Hybrid Preprocessing for Indonesian Google Play Reviews

## Tujuan
Repositori ini menyatukan kekuatan ketiga repo preprocessing yang ada di `PKL/Preprocessing`:
- `text-normalization` untuk basic cleaning, repeated character normalization, emoticon handling, dan tokenization.
- `PySastrawi` untuk stemming Bahasa Indonesia pada output topic modelling.
- `IndoNLU` sebagai referensi praktik NLU Indonesia dan sumber data benchmark.

## Pendekatan
Pipeline dirancang agar:
- modular dan bisa dikonfigurasi untuk dataset terpisah.
- konsisten untuk semua dataset Google Play Store.
- tidak menggabungkan dataset untuk analisis.
- menjaga informasi sentimen penting (emoji, emoticon, negasi).
- mendukung dua jalur output: sentiment dan topic modelling.

## Arsitektur
- `preprocessing/cleaning.py`: basic text cleaning, whitespace normalization, URL/HTML removal.
- `preprocessing/normalization.py`: repeated character reduction, emoji-to-text, slang/typo normalization, stemming.
- `preprocessing/tokenizer.py`: tokenizer Bahasa Indonesia (NLTK TweetTokenizer) dan stopword filtering.
- `preprocessing/pipeline.py`: orchestrator untuk preprocessing sentiment dan topic.
- `resources/slang_dict.json`: custom slang dictionary yang mudah diperluas.
- `resources/stopwords.txt`: custom stopword list tanpa kata negasi.

## Cara pakai
1. Pasang dependencies:
```bash
pip install -r PKL/Preprocessing/hybrid_preprocessing/requirements.txt
```
2. Jalankan preprocessing:
```bash
python PKL/Preprocessing/hybrid_preprocessing/run_preprocessing.py --input PKL/Datasets/main_review.csv --output-dir PKL/Preprocessing/hybrid_preprocessing/output/main
```

Output:
- `sentiment_preprocessed.csv`
- `topic_preprocessed.csv`

## Tradeoff
- Sentiment output: tidak di-stem, mempertahankan nuance kata lengkap, emoji, dan negasi.
- Topic output: stemming Sastrawi digunakan untuk meningkatkan kesamaan topik dan mengurangi sparsity.
- Normalisasi agresif dilakukan pada teks noisy, tetapi kata negasi dan emoji tetap dipertahankan untuk sentiment.

## Ekstensi
- Tambah kosakata slang di `resources/slang_dict.json`.
- Ubah stopword di `resources/stopwords.txt`.
- Tambah kamus typo khusus jika diperlukan.
