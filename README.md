# Sekolah Rakyat: Empirical Learning Barrier Analysis & Adaptive Architecture Specification

An end-to-end, deterministic Natural Language Processing (NLP) pipeline designed to extract user pain points, analyze learning barriers, and synthesize software architecture specifications for the adaptive learning platform **Sekolah Rakyat**.

---

## 📖 Executive Project Summary

This project establishes a direct, empirical connection between raw, unstructured user feedback and critical software architecture decisions. By analyzing **15,324 user reviews** of digital learning platforms in Indonesia, the system leverages NLP (TF-IDF + Logistic Regression for sentiment classification and Latent Dirichlet Allocation (LDA) with $K=8$ for topic modeling) to map subjective complaints into prioritized system requirements. 

The empirical findings reveal that learning engagement is primarily blocked by **system instability** (TB-4) and **curriculum mismatch** (TB-7), rather than learner motivation deficits. These insights are programmatically synthesized into system blueprints, finite state machines (FSM), and behavioral policies to guide the construction of a resilient, offline-first adaptive learning architecture.

---

## 📂 Production Directory Tree

The repository has been modularized and decoupled into a clean, production-grade layout. All experimental directories and legacy scripts have been purged:

```text
edu-review-topic-sentiment/
├── core/                           # ← Production logic package
│   ├── preprocessor.py             #   Orchestration wrapper for preprocessing
│   ├── sentiment_engine.py         #   TF-IDF + Logistic Regression inference
│   ├── topic_engine.py             #   LDA inference (k=8, assets-anchored)
│   ├── severity_analyzer.py        #   Phase 4 + Phase 5 native synthesis engine
│   ├── preprocessing/              #   Migrated text cleaning and tokenization modules
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   ├── normalization.py
│   │   ├── pipeline.py
│   │   └── tokenizer.py
│   ├── resources/                  #   Pre-packaged slang & stopword dictionaries
│   │   ├── slang_dict.json
│   │   └── stopwords.txt
│   └── training/                   #   Research-grade training scripts (auditable baseline)
│       ├── train_baseline_model.py
│       └── phase3b_final_lda_training.py
├── assets/                         # ← Model binaries, static taxonomy, and reference datasets
│   ├── raw_review.csv
│   ├── baseline_model.pkl          #   Serialized vectorizer and sentiment model
│   ├── lda_model_final_k8.gensim   #   Serialized LDA topic model (with companion files)
│   ├── lda_dictionary.gensim
│   ├── lda_ready_corpus.csv
│   ├── phase3b_topic_taxonomy.json #   Topic-to-barrier category mapping
│   └── sr_blueprint_validation.json #  Sekolah Rakyat feature validation reference
├── artifacts/                      # ← Dynamically generated pipeline outputs (JSON matrices)
├── run_pipeline.py                 # ← Unified native Python pipeline orchestrator
├── setup_assets.py                 # ← Bootstrap script to populate the assets directory
├── requirements.txt                # ← System dependencies
└── README.md                       # ← Project documentation
```

---

## ⚡ Quick Start & Execution Guide

### Prerequisites
Install all pipeline dependencies:
```bash
pip install -r requirements.txt
```

### Step 1: Bootstrapping Assets
Initialize the versioned models, reference taxonomies, and slang dictionaries in the correct structure:
```bash
python setup_assets.py
```

### Step 2: Running the Pipeline
Execute the end-to-end, native Python orchestration pipeline:
```bash
python run_pipeline.py
```

### Expected Telemetry Log Output
```text
[CHECK] Verifying optimal model binaries and datasets... [OK]
[DEMO] 1. Preprocessing Raw Reviews... [SUCCESS]
[DEMO] 2. Running Sentiment Inference (Verified Pre-trained Model)... [SUCCESS]
[DEMO] 3. Computing Topic-Sentiment Severity Matrix... [SUCCESS]
[DEMO] 4. Generating Sekolah Rakyat Validation Artifacts... [SUCCESS]
```

All synthesized outputs—including the design implication matrix, system resilience protocols, finite state machines, and curriculum alignment gates—are dynamically written as JSON structures to the `artifacts/` folder.

---

## 📐 System Architecture & Discipline Mapping (DE vs DS/A)

To transition from an experimental research state to a production-grade codebase, the system architecture separates operational engineering safeguards (Data Engineering) from statistical inference and rule logic (Data Science & Analytics).

### 1. Data Engineering (DE) Focus Details
* **Memory Optimization (Chunking):** The text cleaning execution loop streams the raw reviews in isolated chunks of `5,000` rows using pandas generator iterators (`chunksize`). This enforces a near-constant memory footprint, preventing Out-Of-Memory (OOM) crashes.
* **Idempotent I/O Guards:** To protect against duplicate records during execution, the preprocessing pipeline executes pre-run file unlinking to remove stale datasets before writing chunks via append (`a`) mode:
  ```python
  for _stale in (sentiment_path, topic_path):
      if os.path.exists(_stale):
          os.remove(_stale)
  ```
* **System-Agnostic Portability (`pathlib`):** Hardcoded paths are completely eliminated. Programmatic pathing using `pathlib` resolves all system targets relative to `Path(__file__).resolve()` to ensure pathing logic runs correctly across Windows and Unix environments regardless of the directory from which the script was invoked.

### 2. Data Science & Analytics (DS/A) Focus Details
* **NLP Feature Extraction:** The sentiment inference engine applies a pre-trained `TfidfVectorizer` to extract feature weights from clean texts, then runs batch inferences via a `LogisticRegression` classifier to generate prediction labels (`predicted_label_name`) and confidence scores.
* **Mathematical Safety (Zero-Division Guards):** Calculating severity scores requires combining document frequency ($f$) and negative sentiment ratio ($r_n$). Explicit guards are built-in to prevent `ZeroDivisionError` or `NaN` outputs when dealing with low-frequency or zero-negative topic subsets:
  ```python
  negative_ratio = len(neg_df) / freq if freq > 0 else 0
  mean_neg_conf  = neg_df["calibrated_confidence"].mean() if len(neg_df) > 0 else 0.0
  ```
* **Deterministic Rule Engine:** The calculated severity rankings are passed through an implication parser that translates metrics into concrete Sekolah Rakyat system specifications. High severity scores for System Usability (`TB-4` score: `1687.12`) trigger structural rules (e.g. offline-first caching), overriding secondary gamification and AI tutoring systems.

### 3. Synthesis Matrix

| Discipline | Target Module/File | Concrete Code Pattern / Line Context | Strategic Value for Project Stability |
| :--- | :--- | :--- | :--- |
| **Data Engineering** | `core/preprocessing/pipeline.py` | `pd.read_csv(..., chunksize=chunksize)` | Prevents RAM exhaustion/OOM crashes by processing massive datasets in isolated chunks. |
| **Data Engineering** | `core/preprocessing/pipeline.py` | `if os.path.exists(_stale): os.remove(_stale)` | Enforces pipeline idempotency, preventing duplicate records when appending chunks to outputs. |
| **Data Engineering** | `core/preprocessor.py` | `Path(__file__).resolve().parent` | Resolves cross-platform asset paths dynamically relative to module files, avoiding hardcoded dependencies. |
| **Data Science & Analytics** | `core/sentiment_engine.py` | `vectorizer.transform(texts)` & `model.predict()` | Extracts TF-IDF text features and executes inferences to categorize sentiment. |
| **Data Science & Analytics** | `core/topic_engine.py` | `negative_ratio = len(neg_df) / freq if freq > 0 else 0` | Guarantees mathematical stability and prevents program crashes due to division by zero on zero-negative topics. |
| **Data Science & Analytics** | `core/severity_analyzer.py` | `feature_requirements = { "TB-4 ...": ... }` | Evaluates empirical scores to programmatically synthesize design matrices and resilience protocols. |
