# Sentiment Analysis Model - Phase 1: Data Preparation

Comprehensive pipeline untuk mempersiapkan dataset sentiment analysis dengan weak labeling dan stratified split.

## 📋 Project Overview

Project ini mengimplementasikan **Phase 1** dari sentiment analysis end-to-end pipeline:
- Loading dataset (benchmark + main reviews)
- Data validation dan cleaning
- Weak labeling berbasis score (1-2→negative, 3→neutral, 4-5→positive)
- Stratified train/validation/test split
- Dataset summary reporting

## 📁 Project Structure

```
sentiment-analysis-model/
├── data/
│   ├── raw/              # Raw datasets (before processing)
│   │   ├── dataset1.csv
│   │   ├── dataset2.csv
│   │   └── combined.csv
│   └── processed/        # Processed splits (ready for training)
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── src/                  # Source code
│   ├── __init__.py
│   ├── config.py         # Configuration & constants
│   ├── data_loader.py    # Dataset loading & normalization
│   ├── data_validator.py # Validation & quality checks
│   ├── data_processor.py # Weak labeling & splitting
│   ├── dataset_summary.py # Summary reporting
│   └── prepare_data.py   # Main pipeline orchestrator
├── models/               # Model artifacts (Phase 2+)
├── results/              # Summary reports & outputs
│   ├── dataset_summary.json
│   └── dataset_summary.txt
├── visualizations/       # Charts & plots (Phase 2+)
├── notebooks/            # Jupyter notebooks
├── logs/                 # Pipeline logs
├── venv/                 # Virtual environment
├── requirements.txt      # Python dependencies
└── PHASE1_AUDIT_REPORT.md # Completion audit
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to project directory
cd sentiment-analysis-model

# Activate virtual environment
venv\Scripts\activate    # Windows
# or
source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Pipeline

```bash
# Execute complete data preparation
python src/prepare_data.py
```

**Output:**
- Raw datasets: `data/raw/`
- Processed splits: `data/processed/`
- Summary reports: `results/`
- Logs: `logs/data_preparation.log`

### 3. View Results

```bash
# Human-readable summary
cat results/dataset_summary.txt

# JSON format (for programmatic access)
cat results/dataset_summary.json
```

## 📊 Dataset Summary

| Metric | Value |
|--------|-------|
| **Total Samples** | 42,367 |
| **Features** | 14 columns |
| **Source** | Benchmark (3,512) + Main (48,234) reviews |
| **Missing Scores** | 9,379 removed (18.13%) |

### Label Distribution

| Label | Count | Percentage |
|-------|-------|-----------|
| Positive (2) | 30,482 | 71.95% |
| Negative (0) | 8,858 | 20.91% |
| Neutral (1) | 3,027 | 7.14% |

### Train/Val/Test Split

| Set | Samples | % | Negative | Neutral | Positive |
|-----|---------|---|----------|---------|----------|
| Train | 29,656 | 70% | 6,200 | 2,119 | 21,337 |
| Validation | 6,355 | 15% | 1,329 | 454 | 4,572 |
| Test | 6,356 | 15% | 1,329 | 454 | 4,573 |

## 📝 Module Documentation

### `config.py`
Configuration file dengan constants:
- Path configuration
- Sentiment mappings (0→negative, 1→neutral, 2→positive)
- Score to sentiment mapping (1-2→negative, 3→neutral, 4-5→positive)
- Split ratios (70/15/15)
- Random state untuk reproducibility (42)

### `data_loader.py`
Loading dan normalisasi dataset:
- `load_dataset()`: Load CSV dengan error handling
- `normalize_columns()`: Handle kolom alternatif (content_clean→content)
- `load_both_datasets()`: Load dan normalize kedua dataset
- `combine_datasets()`: Combine dengan tracking source

### `data_validator.py`
Validasi kualitas data:
- `DataValidator` class dengan methods:
  - `validate_columns()`: Check required columns
  - `check_missing_values()`: Detect NaN
  - `check_duplicates()`: Find duplicate rows
  - `validate_score_column()`: Numeric conversion & range check
  - `validate_content_column()`: Content validation
  - `validate_all()`: Comprehensive validation

### `data_processor.py`
Processing dan splitting:
- `WeakLabeler` class:
  - `apply_weak_labeling()`: Score→sentiment mapping
  - `get_label_distribution()`: Distribution stats
  - `print_label_distribution()`: Console output
- `StratifiedSplitter` class:
  - `split_data()`: Stratified split (70/15/15)
  - `save_splits()`: Save to CSV
  - `print_split_report()`: Console report

### `dataset_summary.py`
Report generation:
- `DatasetSummaryGenerator` class:
  - `generate_summary()`: Comprehensive summary dict
  - `save_json()`: JSON format
  - `save_txt()`: Human-readable format
  - `print_summary()`: Console output

### `prepare_data.py`
Main orchestrator:
- 11-step pipeline
- Error handling dan logging
- Complete workflow dari load hingga summary

## 🔑 Key Features

### ✅ Data Quality
- Validation untuk missing values, duplicates, invalid scores
- Automatic removal of rows dengan score=NaN
- Content validation (no empty content)
- Score range validation (1-5)

### ✅ Weak Labeling
- Simple rule-based labeling
- Konsisten dengan common sentiment ranges
- Menghasilkan sentiment_label (text) dan sentiment_id (numeric)

### ✅ Reproducibility
- Fixed random_state=42
- Stratified split mempertahankan class distribution
- Dokumentasi lengkap
- Requirements dengan exact versions

### ✅ Modular Design
- Separate concerns (loading, validation, processing)
- Reusable functions
- Easy to extend untuk Phase 2

### ✅ Comprehensive Logging
- File logging ke logs/data_preparation.log
- Console output untuk progress tracking
- Detailed validation reports

## 📈 Output Files

### Train/Val/Test Sets
```
data/processed/
├── train.csv         # 29,656 samples (70%)
├── validation.csv    # 6,355 samples (15%)
└── test.csv          # 6,356 samples (15%)
```

**Columns:**
- `reviewId`: Unique review identifier
- `userName`: User who posted
- `score`: Rating (1-5)
- `content`: Cleaned review text (primary feature)
- `sentiment_label`: Weak label (negative/neutral/positive)
- `sentiment_id`: Numeric label (0/1/2)
- `source`: Dataset origin (dataset1/dataset2)
- Plus other metadata columns

### Summary Reports
```
results/
├── dataset_summary.json  # Machine-readable
└── dataset_summary.txt   # Human-readable
```

## 🔧 Configuration

Modify `src/config.py` untuk customize:

```python
# Split ratios
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducibility
RANDOM_STATE = 42

# Sentiment mapping
SENTIMENT_MAPPING = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

# Score to sentiment
SCORE_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive"
}
```

## 📋 Pipeline Steps

```
1. Load datasets
2. Combine datasets
3. Save raw datasets
4. Validate dataset
5. Apply weak labeling
6. Clean (remove NaN scores)
7. Display samples
8. Show score distribution
9. Stratified split
10. Save splits
11. Generate summary
12. Final verification
```

## 🚨 Error Handling

Pipeline handles:
- ✅ Missing files (FileNotFoundError)
- ✅ Invalid CSV (ValueError)
- ✅ Missing columns (validation)
- ✅ Non-numeric scores (type conversion)
- ✅ NaN values (dropping rows)
- ✅ Invalid score ranges (detection)

## 📝 Logging

Logs tersimpan di `logs/data_preparation.log`:

```
2026-05-09 20:48:06,394 - prepare_data - INFO - STARTING DATA PREPARATION PIPELINE
2026-05-09 20:48:06,515 - data_loader - INFO - Successfully loaded dataset from ...
2026-05-09 20:48:19,749 - prepare_data - INFO - Dataset size after cleaning: 42367
...
```

## 🔄 Reproducibility

Untuk reproduce hasil:

```bash
# Clear old output
rm -rf data/processed/* results/* logs/*

# Run pipeline
python src/prepare_data.py

# Hasil akan identik dengan seed random_state=42
```

## 🎯 Next Steps (Phase 2)

Phase 1 completion enables Phase 2:
- ✅ Tokenization dengan IndoBERTweet
- ✅ Model training (IndoBERTweet-base-uncased)
- ✅ Fine-tuning pada sentiment classification
- ✅ Evaluation metrics
- ✅ Model deployment

## 📚 References

- **Dataset:** PKL/Datasets/{benchmark,main}_review_preprocessed/sentiment_preprocessed.csv
- **Model:** https://huggingface.co/indolem/indobertweet-base-uncased
- **Framework:** scikit-learn for stratified split, pandas for data processing

## ✅ Verification

Untuk verify setup:

```bash
# Activate venv
venv\Scripts\activate

# Run pipeline
python src/prepare_data.py

# Check output files exist
ls data/processed/
ls results/

# View summary
cat results/dataset_summary.txt
```

## 📞 Support

For issues:
1. Check `logs/data_preparation.log` untuk error details
2. Verify `requirements.txt` packages installed
3. Ensure dataset files exist di `PKL/Datasets/`
4. Check column names dalam dataset

---

**Phase 1 Status:** ✅ COMPLETE  
**Last Updated:** May 9, 2026  
**Ready for:** Phase 2 (Tokenization & Model Training)
