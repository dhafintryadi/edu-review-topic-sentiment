import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any

# Define paths
WORKING_DIR = Path("PKL/Sentiment-Analysis")
DATASETS_DIR = Path("PKL/Datasets")

RAW_PATH = DATASETS_DIR / "raw_review.csv"
SENTIMENT_PATH = DATASETS_DIR / "sentiment_preprocessed.csv"
TOPIC_PATH = DATASETS_DIR / "topic_preprocessed.csv"

OUTPUT_DIR = WORKING_DIR
REPORT_PATH = OUTPUT_DIR / "dataset_quality_report.json"
CLEAN_SENTIMENT_PATH = OUTPUT_DIR / "clean_loaded_sentiment_dataset.csv"
CLEAN_TOPIC_PATH = OUTPUT_DIR / "clean_loaded_topic_dataset.csv"

# Expected schemas
EXPECTED_RAW_COLS = ["reviewId", "userName", "score", "content", "at", "thumbsUpCount", "replyContent", "repliedAt", "appVersion"]
EXPECTED_PREP_COLS = ["reviewId", "userName", "score", "content_raw", "at", "thumbsUpCount", "replyContent", "repliedAt", "appVersion", "content_clean", "content_topic"]

def load_and_validate_dataset(path: Path, name: str, expected_cols: list) -> Dict[str, Any]:
    """Load dataset and perform validation checks."""
    report = {"name": name, "path": str(path), "status": "Failed", "errors": []}

    try:
        # Load with UTF-8
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        report["status"] = "Loaded Successfully"
        report["row_count"] = len(df)
        report["columns"] = list(df.columns)

        # Schema check
        actual_cols = set(df.columns)
        expected_set = set(expected_cols)
        missing = expected_set - actual_cols
        extra = actual_cols - expected_set

        if missing:
            report["errors"].append(f"Missing columns: {list(missing)}")
        if extra:
            report["errors"].append(f"Extra columns: {list(extra)}")

        # Nulls check
        null_counts = df.isnull().sum()
        report["null_counts"] = null_counts.to_dict()
        total_nulls = null_counts.sum()
        report["total_nulls"] = int(total_nulls)

        # Duplicates check
        dups = df.duplicated().sum()
        report["duplicate_rows"] = int(dups)

        # Score distribution if available
        if "score" in df.columns:
            score_dist = pd.to_numeric(df["score"], errors="coerce").value_counts(dropna=False).sort_index().to_dict()
            report["score_distribution"] = score_dist

        # Encoding check (basic - assume UTF-8 if loaded successfully)
        report["encoding"] = "utf-8"

    except Exception as e:
        report["status"] = "Failed"
        report["errors"].append(str(e))

    return report

def clean_dataset(df: pd.DataFrame, content_col: str) -> pd.DataFrame:
    """Clean dataset by removing duplicates based on content column."""
    # Remove duplicates based on content column
    df_clean = df.drop_duplicates(subset=[content_col], keep="first").copy()
    # Optionally handle nulls in content_col
    df_clean = df_clean.dropna(subset=[content_col])
    return df_clean

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and validate raw
    raw_report = load_and_validate_dataset(RAW_PATH, "raw_review", EXPECTED_RAW_COLS)

    # Load and validate sentiment
    sentiment_report = load_and_validate_dataset(SENTIMENT_PATH, "sentiment_preprocessed", EXPECTED_PREP_COLS)
    if sentiment_report["status"] == "Loaded Successfully":
        df_sentiment = pd.read_csv(SENTIMENT_PATH, encoding="utf-8", low_memory=False)
        df_sentiment_clean = clean_dataset(df_sentiment, "content_clean")
        df_sentiment_clean.to_csv(CLEAN_SENTIMENT_PATH, index=False, encoding="utf-8")
        sentiment_report["clean_row_count"] = len(df_sentiment_clean)

    # Load and validate topic
    topic_report = load_and_validate_dataset(TOPIC_PATH, "topic_preprocessed", EXPECTED_PREP_COLS)
    if topic_report["status"] == "Loaded Successfully":
        df_topic = pd.read_csv(TOPIC_PATH, encoding="utf-8", low_memory=False)
        df_topic_clean = clean_dataset(df_topic, "content_topic")
        df_topic_clean.to_csv(CLEAN_TOPIC_PATH, index=False, encoding="utf-8")
        topic_report["clean_row_count"] = len(df_topic_clean)

    # Compile full report
    full_report = {
        "phase": "data_preparation",
        "timestamp": pd.Timestamp.now().isoformat(),
        "datasets": {
            "raw": raw_report,
            "sentiment": sentiment_report,
            "topic": topic_report
        },
        "outputs": {
            "clean_sentiment_dataset": str(CLEAN_SENTIMENT_PATH),
            "clean_topic_dataset": str(CLEAN_TOPIC_PATH),
            "quality_report": str(REPORT_PATH)
        }
    }

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, ensure_ascii=False)

    print("Data preparation completed successfully!")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Clean sentiment dataset: {CLEAN_SENTIMENT_PATH}")
    print(f"Clean topic dataset: {CLEAN_TOPIC_PATH}")

if __name__ == "__main__":
    main()