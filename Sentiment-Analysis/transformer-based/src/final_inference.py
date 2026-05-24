"""
src/final_inference.py
Final Phase: Full-corpus sentiment inference using the production IndoBERTweet model.
- Loads BOTH datasets (main + benchmark)
- Applies Temperature Scaling (T=0.697) for calibrated probabilities
- Exports enriched CSV + Parquet for downstream topic modelling
"""

import os, sys, time, json, logging, datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

from inference_utils import extract_prediction_columns

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASETS_ROOT = os.path.join(ROOT, "..", "..", "Datasets")
MAIN_CSV      = os.path.join(DATASETS_ROOT, "main_review_preprocessed", "sentiment_preprocessed.csv")
BENCH_CSV     = os.path.join(DATASETS_ROOT, "benchmark_review_preprocessed", "sentiment_preprocessed.csv")
MODEL_DIR     = os.path.join(ROOT, "models", "final_baseline", "best_model")
RESULTS       = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_LENGTH  = 96
BATCH_SIZE  = 16   # CPU-safe
TEXT_COL    = "content_clean"
TEMPERATURE = 0.697

# ── Logging ─────────────────────────────────────────────────────────────────────
LOGS = os.path.join(ROOT, "logs")
os.makedirs(LOGS, exist_ok=True)
LOG_FILE = os.path.join(LOGS, "final_inference.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ── Simple Dataset wrapper ──────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def load_corpus():
    """Load and concatenate both datasets. Add a source column."""
    logger.info("Loading main dataset...")
    df_main = pd.read_csv(MAIN_CSV, encoding="utf-8")
    df_main["source"] = "main"
    logger.info(f"  Main: {len(df_main):,} rows")

    logger.info("Loading benchmark dataset...")
    df_bench = pd.read_csv(BENCH_CSV, encoding="utf-8")
    df_bench["source"] = "benchmark"
    logger.info(f"  Benchmark: {len(df_bench):,} rows")

    df = pd.concat([df_main, df_bench], ignore_index=True)
    logger.info(f"  Total corpus: {len(df):,} rows")
    return df


def run_inference(df: pd.DataFrame, model, tokenizer, device):
    """Run batched inference on the full corpus. Returns logit array."""
    # Handle missing text
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    logger.info(f"Running inference on {len(texts):,} texts (batch_size={BATCH_SIZE})...")

    dataset = TextDataset(texts, tokenizer, MAX_LENGTH)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_logits = []
    model.eval()
    with torch.no_grad():
        with open(LOG_FILE, "a", encoding="utf-8") as log_f:
            for batch in tqdm(loader, desc="Inferencing", unit="batch", file=log_f, mininterval=10):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                all_logits.append(outputs.logits.cpu().numpy())

    return np.vstack(all_logits)


def main():
    logger.info("=" * 60)
    logger.info("FINAL PHASE — FULL CORPUS SENTIMENT INFERENCE")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info(f"Model: {MODEL_DIR}")
    logger.info(f"Temperature Scaling: T={TEMPERATURE}")
    logger.info("=" * 60)

    start_time = time.time()

    # Load corpus
    df = load_corpus()

    # Load model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device    = torch.device("cpu")
    model.to(device)
    logger.info("Model loaded.")

    # Inference
    logits = run_inference(df, model, tokenizer, device)

    # Extract predictions
    preds = extract_prediction_columns(logits, temperature=TEMPERATURE)
    for col, values in preds.items():
        df[col] = values

    runtime = time.time() - start_time
    logger.info(f"Inference complete in {str(datetime.timedelta(seconds=int(runtime)))}.")

    # Save enriched dataset
    out_csv     = os.path.join(RESULTS, "final_sentiment_inference.csv")
    out_parquet = os.path.join(RESULTS, "final_sentiment_inference.parquet")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    df.to_parquet(out_parquet, index=False)
    logger.info(f"Saved: {out_csv}")
    logger.info(f"Saved: {out_parquet}")

    # Save runtime metadata
    meta = {
        "total_rows": len(df),
        "main_rows": int((df["source"] == "main").sum()),
        "benchmark_rows": int((df["source"] == "benchmark").sum()),
        "null_texts_handled": int(df[TEXT_COL].isna().sum()),
        "temperature": TEMPERATURE,
        "runtime_seconds": runtime,
        "runtime_formatted": str(datetime.timedelta(seconds=int(runtime))),
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(os.path.join(RESULTS, "inference_runtime.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("INFERENCE DONE.")
    return df


if __name__ == "__main__":
    main()
