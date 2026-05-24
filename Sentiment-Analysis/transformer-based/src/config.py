"""
Configuration and constants for sentiment analysis project
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset paths from PKL/Datasets
# Path: sentiment-analysis-model -> sentiment-analysis -> PKL -> Datasets
PKL_ROOT = PROJECT_ROOT.parent.parent / "Datasets"

# Check which datasets are available
BENCHMARK_REVIEW_PATH = PKL_ROOT / "benchmark_review_preprocessed" / "sentiment_preprocessed.csv"
MAIN_REVIEW_RAW_PATH = PKL_ROOT / "main_review.csv"
BENCHMARK_REVIEW_RAW_PATH = PKL_ROOT / "benchmarking_review.csv"

# Use preprocessed benchmark if available, otherwise use raw
if BENCHMARK_REVIEW_PATH.exists():
    DATASET_PATH_1 = BENCHMARK_REVIEW_PATH
else:
    DATASET_PATH_1 = BENCHMARK_REVIEW_RAW_PATH

# For main review, check if preprocessed exists, otherwise use raw
MAIN_REVIEW_PREPROCESSED = PKL_ROOT / "main_review_preprocessed" / "sentiment_preprocessed.csv"
if MAIN_REVIEW_PREPROCESSED.exists():
    DATASET_PATH_2 = MAIN_REVIEW_PREPROCESSED
else:
    DATASET_PATH_2 = MAIN_REVIEW_RAW_PATH

# Output paths
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"

# Sentiment label mapping
SENTIMENT_MAPPING = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

# Reverse mapping for labels
SENTIMENT_REVERSE_MAPPING = {v: k for k, v in SENTIMENT_MAPPING.items()}

# Score to sentiment mapping (weak labeling)
SCORE_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive"
}

# Dataset split ratios
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Random state for reproducibility
RANDOM_STATE = 42

# Required columns in dataset
REQUIRED_COLUMNS = ["content", "score"]

# Alternative column names mapping
COLUMN_ALIASES = {
    "content": ["content", "content_clean", "content_raw", "text", "review"],
    "score": ["score", "rating"]
}

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, 
                   VISUALIZATIONS_DIR, NOTEBOOKS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
