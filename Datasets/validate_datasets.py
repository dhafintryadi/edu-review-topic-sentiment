import pandas as pd
import os

base = r'c:\Users\ASUS\Documents\AITF-2026\PKL\Datasets'

FILES = {
    "raw_review":              f"{base}/raw_review.csv",
    "sentiment_preprocessed":  f"{base}/sentiment_preprocessed.csv",
    "topic_preprocessed":      f"{base}/topic_preprocessed.csv",
}

EXPECTED_RAW_COLS    = ["reviewId","userName","score","content","at",
                         "thumbsUpCount","replyContent","repliedAt","appVersion"]
EXPECTED_PREP_COLS   = ["reviewId","userName","score","content_raw","at",
                         "thumbsUpCount","replyContent","repliedAt","appVersion",
                         "content_clean","content_topic"]

SEP = "=" * 60

def validate(name, path, expected_cols, expected_rows=None):
    print(f"\n{SEP}")
    print(f"  FILE : {name}")
    print(SEP)

    # --- Load ---
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        print(f"  [OK] Loaded successfully  |  Rows: {len(df):,}  |  Cols: {len(df.columns)}")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)
            print(f"  [WARN] Loaded with latin-1 fallback  |  Rows: {len(df):,}")
        except Exception as e:
            print(f"  [ERR] Failed to load: {e}")
            return
    except Exception as e:
        print(f"  [ERR] Failed to load: {e}")
        return

    # --- Schema ---
    actual_cols = list(df.columns)
    print(f"\n  Columns detected: {actual_cols}")
    missing = [c for c in expected_cols if c not in actual_cols]
    extra   = [c for c in actual_cols  if c not in expected_cols]
    if not missing and not extra:
        print(f"  [OK] Schema matches expected columns exactly.")
    else:
        if missing: print(f"  [WARN] Missing expected cols : {missing}")
        if extra:   print(f"  [INFO] Extra cols (unexpected): {extra}")

    # --- Duplicates ---
    dups = df.duplicated().sum()
    print(f"\n  Duplicate rows : {dups:,}  {'[OK]' if dups == 0 else '[WARN]'}")

    # --- Nulls ---
    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if null_cols.empty:
        print(f"  Null values    : 0  [OK]")
    else:
        print(f"  Null values    : [WARN]")
        for col, cnt in null_cols.items():
            pct = cnt / len(df) * 100
            print(f"    - {col}: {cnt:,} ({pct:.2f}%)")

    # --- Basic stats ---
    print(f"\n  Row count      : {len(df):,}")
    print(f"  Column count   : {len(df.columns)}")
    if "score" in df.columns:
        score_series = pd.to_numeric(df["score"], errors="coerce")
        print(f"  Score dist     :\n{score_series.value_counts(dropna=False).sort_index().to_string()}")

    unique_rows = len(df) - dups
    if dups > 0:
        print(f"  Unique rows    : {unique_rows:,}")
    if expected_rows is not None:
        if len(df) == expected_rows:
            print(f"  Expected rows  : {expected_rows:,}  [OK]")
        else:
            print(f"  Expected rows  : {expected_rows:,}  [WARN] actual {len(df):,}")

    print()


# --- Source file row counts for cross-check ---
print(f"\n{SEP}")
print("  SOURCE FILE ROW COUNTS (for cross-validation)")
print(SEP)
sources = {
    "main_review.csv":                              f"{base}/main_review.csv",
    "benchmarking_review.csv":                      f"{base}/benchmarking_review.csv",
    "main_review_preprocessed/sentiment_preprocessed.csv":
        f"{base}/main_review_preprocessed/sentiment_preprocessed.csv",
    "benchmark_review_preprocessed/sentiment_preprocessed.csv":
        f"{base}/benchmark_review_preprocessed/sentiment_preprocessed.csv",
    "main_review_preprocessed/topic_preprocessed.csv":
        f"{base}/main_review_preprocessed/topic_preprocessed.csv",
    "benchmark_review_preprocessed/topic_preprocessed.csv":
        f"{base}/benchmark_review_preprocessed/topic_preprocessed.csv",
}
for label, path in sources.items():
    try:
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
        print(f"  {label}: {n:,} rows")
    except Exception as e:
        print(f"  {label}: ERROR - {e}")

# --- Validate merged files ---
expected_counts = {
    "raw_review": 13104 + 3526,
    "sentiment_preprocessed": 63153 + 3526,
    "topic_preprocessed": 63153 + 3526,
}
validate("raw_review",             FILES["raw_review"],             EXPECTED_RAW_COLS, expected_rows=expected_counts["raw_review"])
validate("sentiment_preprocessed", FILES["sentiment_preprocessed"], EXPECTED_PREP_COLS, expected_rows=expected_counts["sentiment_preprocessed"])
validate("topic_preprocessed",     FILES["topic_preprocessed"],     EXPECTED_PREP_COLS, expected_rows=expected_counts["topic_preprocessed"])

print(f"\n{SEP}")
print("  VALIDATION COMPLETE")
print(SEP)
