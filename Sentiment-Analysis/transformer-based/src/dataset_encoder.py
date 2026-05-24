"""
src/dataset_encoder.py
HuggingFace Dataset integration for IndoBERTweet sentiment analysis.
Phase 2B — CPU-efficient, lazy tokenization, label-consistent encoding.
"""

import logging
import os
from typing import Optional, Dict, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value, Sequence

from tokenizer import build_tokenizer, SentimentTokenizer

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
DEFAULT_TEXT_COL  = "content"
DEFAULT_LABEL_COL = "sentiment_id"
DEFAULT_MAX_LENGTH = 128


# ═══════════════════════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════════════════════

def load_csv_splits(
    data_dir: str,
    text_col:  str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    splits: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load train/validation/test CSV files from data_dir.

    Returns:
        dict mapping split name -> cleaned DataFrame with [text_col, label_col]
    """
    if splits is None:
        splits = {
            "train":      "train.csv",
            "validation": "validation.csv",
            "test":       "test.csv",
        }

    frames: Dict[str, pd.DataFrame] = {}
    for split, fname in splits.items():
        path = os.path.join(data_dir, fname)
        df   = pd.read_csv(path)

        # Keep only required columns
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in {fname}. Available: {df.columns.tolist()}")
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found in {fname}. Available: {df.columns.tolist()}")

        df = df[[text_col, label_col]].copy()
        df[text_col]  = df[text_col].fillna("").astype(str)
        df[label_col] = df[label_col].astype(int)

        n_before = len(df)
        df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
        n_after  = len(df)

        logger.info(f"  [{split}] Loaded {n_before:,} rows → {n_after:,} after dedup "
                    f"(removed {n_before - n_after:,})")
        frames[split] = df

    return frames


# ═══════════════════════════════════════════════════════════
# CSV → HuggingFace Dataset conversion
# ═══════════════════════════════════════════════════════════

def df_to_hf_dataset(
    df: pd.DataFrame,
    text_col:  str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> Dataset:
    """Convert a pandas DataFrame to a HuggingFace Dataset."""
    return Dataset.from_pandas(
        df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}),
        preserve_index=False,
    )


def build_dataset_dict(
    frames: Dict[str, pd.DataFrame],
    text_col:  str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> DatasetDict:
    """
    Build a HuggingFace DatasetDict from loaded DataFrames.

    Returns:
        DatasetDict with keys: "train", "validation", "test"
    """
    return DatasetDict({
        split: df_to_hf_dataset(df, text_col, label_col)
        for split, df in frames.items()
    })


# ═══════════════════════════════════════════════════════════
# Tokenization / Encoding
# ═══════════════════════════════════════════════════════════

def encode_dataset(
    dataset_dict: DatasetDict,
    tokenizer: SentimentTokenizer,
    text_col: str = "text",
    batch_size: int = 256,
    num_proc: Optional[int] = None,
    remove_text_col: bool = True,
) -> DatasetDict:
    """
    Apply tokenizer to all splits using Dataset.map() (lazy, CPU-efficient).

    Adds columns: input_ids, attention_mask (token_type_ids if available).
    Keeps column: label.

    Args:
        dataset_dict:    DatasetDict to encode.
        tokenizer:       SentimentTokenizer instance.
        text_col:        Name of the text column in dataset.
        batch_size:      Batch size for .map() processing.
        num_proc:        Number of CPU processes (1 = no multiprocessing).
        remove_text_col: Drop the raw text column after encoding.

    Returns:
        Encoded DatasetDict with tensor-ready columns.
    """
    def _tokenize_fn(examples):
        enc = tokenizer(examples[text_col])
        return enc

    logger.info(f"[encode_dataset] Encoding {list(dataset_dict.keys())} splits "
                f"(max_length={tokenizer.max_length}, batch_size={batch_size})")

    encoded = dataset_dict.map(
        _tokenize_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    if remove_text_col and text_col in encoded[list(encoded.keys())[0]].column_names:
        encoded = encoded.remove_columns([text_col])

    # Rename label column to "labels" (HuggingFace Trainer convention)
    if "label" in encoded[list(encoded.keys())[0]].column_names:
        encoded = encoded.rename_column("label", "labels")

    # Set PyTorch format
    encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    logger.info("[encode_dataset] Encoding complete.")
    for split, ds in encoded.items():
        logger.info(f"  [{split}] {len(ds):,} samples | columns: {ds.column_names}")

    return encoded


# ═══════════════════════════════════════════════════════════
# High-level pipeline entry point
# ═══════════════════════════════════════════════════════════

def build_encoded_datasets(
    data_dir: str,
    model_name:     str = "indolem/indobertweet-base-uncased",
    max_length:     int = DEFAULT_MAX_LENGTH,
    text_col:       str = DEFAULT_TEXT_COL,
    label_col:      str = DEFAULT_LABEL_COL,
    batch_size:     int = 256,
    dedup:          bool = True,
    splits:         Optional[Dict[str, str]] = None,
) -> Tuple[DatasetDict, SentimentTokenizer]:
    """
    Full pipeline: CSV → DataFrame → HuggingFace Dataset → Encoded DatasetDict.

    Args:
        data_dir:    Directory containing train.csv, validation.csv, test.csv.
        model_name:  HuggingFace model ID.
        max_length:  Tokenization max length.
        text_col:    Text column in CSV.
        label_col:   Label column in CSV.
        batch_size:  Encoding batch size.
        dedup:       Whether to deduplicate by text.

    Returns:
        (encoded_dataset_dict, tokenizer)

    Example:
        encoded, tok = build_encoded_datasets("data/processed", max_length=128)
        train_ds = encoded["train"]
    """
    logger.info("=" * 50)
    logger.info("Building encoded datasets pipeline")
    logger.info(f"  data_dir   : {data_dir}")
    logger.info(f"  model_name : {model_name}")
    logger.info(f"  max_length : {max_length}")
    logger.info("=" * 50)

    # 1. Load CSVs
    frames = load_csv_splits(data_dir, text_col=text_col, label_col=label_col, splits=splits)

    # 2. Build HF DatasetDict
    raw_datasets = build_dataset_dict(frames, text_col=text_col, label_col=label_col)
    logger.info(f"[pipeline] DatasetDict built: {raw_datasets}")

    # 3. Build tokenizer
    tok = build_tokenizer(model_name=model_name, max_length=max_length, padding="max_length")

    # 4. Encode
    encoded = encode_dataset(raw_datasets, tok, batch_size=batch_size)

    return encoded, tok


# ═══════════════════════════════════════════════════════════
# Quick validation helper
# ═══════════════════════════════════════════════════════════

def validate_encoded_dataset(encoded: DatasetDict, max_length: int = DEFAULT_MAX_LENGTH) -> dict:
    """
    Run structural validation checks on an encoded DatasetDict.

    Returns:
        dict of validation results per split.
    """
    report = {}
    for split, ds in encoded.items():
        checks = {}

        # Column presence
        checks["has_input_ids"]     = "input_ids"     in ds.column_names
        checks["has_attention_mask"] = "attention_mask" in ds.column_names
        checks["has_labels"]         = "labels"         in ds.column_names

        if checks["has_input_ids"]:
            sample = ds[0]
            checks["input_ids_length_correct"]     = len(sample["input_ids"])     == max_length
            checks["attention_mask_length_correct"] = len(sample["attention_mask"]) == max_length
            checks["label_is_int"] = isinstance(sample["labels"].item(), int)

        checks["all_pass"] = all(v for v in checks.values() if isinstance(v, bool))
        report[split] = checks
        status = "PASS" if checks["all_pass"] else "FAIL"
        logger.info(f"  [{split}] Validation: {status} | {checks}")

    return report


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    encoded, tok = build_encoded_datasets(data_dir, max_length=128)

    print("\n── Validation ──")
    report = validate_encoded_dataset(encoded, max_length=128)
    all_ok = all(v["all_pass"] for v in report.values())
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")

    print("\n── Sample batch ──")
    sample = encoded["train"][:4]
    print(f"input_ids shape   : {sample['input_ids'].shape}")
    print(f"attention_mask    : {sample['attention_mask'].shape}")
    print(f"labels            : {sample['labels'].tolist()}")
