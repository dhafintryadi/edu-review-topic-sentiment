"""
src/run_phase2c.py
Phase 2C Orchestrator
Executes data cleaning, leakage mitigation, token stats recalculation, and class weighting.
Generates clean outputs and comprehensive reports.
"""

import os, sys, json, logging, datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

DATA_PROC = os.path.join(ROOT, "data", "processed")
DATA_CLEAN = os.path.join(ROOT, "data", "cleaned")
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
VIZ = os.path.join(ROOT, "visualizations")

for d in [DATA_CLEAN, RESULTS, LOGS, VIZ]:
    os.makedirs(d, exist_ok=True)

# Logging Setup
LOG_FILE = os.path.join(LOGS, "phase2c.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

from data_cleaner import DataCleaner
from leakage_detector import LeakageMitigator
from class_weighting import ImbalanceHandler
from tokenizer import build_tokenizer

def main():
    logger.info("=" * 60)
    logger.info("PHASE 2C - Dataset Cleaning & Fine-Tuning Preparation")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info("=" * 60)

    # 1. Load Existing Processed Splits
    logger.info("\n[1] Loading Processed Splits")
    splits = {}
    for s in ["train", "validation", "test"]:
        path = os.path.join(DATA_PROC, f"{s}.csv")
        splits[s] = pd.read_csv(path)
        logger.info(f"Loaded {s}: {len(splits[s]):,} rows")

    initial_sizes = {s: len(df) for s, df in splits.items()}

    # 2. Exact Deduplication
    logger.info("\n[2] Exact Deduplication")
    cleaner = DataCleaner()
    dedup_stats = {}
    near_dup_stats = {}
    
    for s in ["train", "validation", "test"]:
        near_dup_stats[s] = cleaner.audit_near_duplicates(splits[s])
        splits[s], stats = cleaner.remove_exact_duplicates(splits[s], s)
        dedup_stats[s] = stats

    with open(os.path.join(RESULTS, "deduplication_report.json"), "w") as f:
        json.dump({"exact_duplicates": dedup_stats, "near_duplicates": near_dup_stats}, f, indent=2)

    # 3. Leakage Mitigation
    logger.info("\n[3] Leakage Mitigation")
    mitigator = LeakageMitigator()
    train_clean, val_clean, test_clean, leak_stats = mitigator.remove_cross_split_leakage(
        splits["train"], splits["validation"], splits["test"]
    )

    splits_clean = {
        "train": train_clean,
        "validation": val_clean,
        "test": test_clean
    }

    with open(os.path.join(RESULTS, "leakage_mitigation_report.json"), "w") as f:
        json.dump(leak_stats, f, indent=2)

    # 4. Save Clean Datasets
    logger.info("\n[4] Exporting Clean Datasets")
    for s, df in splits_clean.items():
        out_path = os.path.join(DATA_CLEAN, f"{s}_clean.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {s}_clean.csv ({len(df):,} rows)")

    # 5. Class Weights & Imbalance
    logger.info("\n[5] Class Weighting & Imbalance Statistics")
    handler = ImbalanceHandler()
    
    # Calculate for train set specifically as it's used for the loss function
    train_balance_stats = handler.compute_weights(splits_clean["train"])
    
    # Overall distribution for reporting
    all_clean_stats = {}
    for s, df in splits_clean.items():
        all_clean_stats[s] = handler.compute_weights(df)

    with open(os.path.join(RESULTS, "class_balance_report.json"), "w") as f:
        json.dump(all_clean_stats, f, indent=2)

    # 6. Recalculate Token Statistics
    logger.info("\n[6] Recalculating Token Statistics")
    tok = build_tokenizer(max_length=96)
    all_texts = pd.concat([df["content"] for df in splits_clean.values()]).dropna().astype(str).tolist()
    
    lengths = tok.get_token_lengths_batch(all_texts, batch_size=512)
    arr = np.array(lengths)
    
    token_stats = {
        "total_samples": len(arr),
        "mean": float(arr.mean()),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "pct_covered_by_96": float((arr <= 96).sum() / len(arr) * 100)
    }
    
    logger.info(f"New Token Stats: Mean={token_stats['mean']:.2f}, P95={token_stats['p95']}, Covered by 96={token_stats['pct_covered_by_96']:.2f}%")

    # 7. Visualizations
    logger.info("\n[7] Generating Visualizations")
    STYLE = {"bg": "#0f1117", "panel": "#1a1d27", "text": "#e0e0e0", "subtext": "#9099b0", "grid": "#252a3a"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": STYLE["text"],
        "axes.facecolor": STYLE["panel"], "axes.edgecolor": STYLE["grid"],
        "axes.labelcolor": STYLE["text"], "xtick.color": STYLE["subtext"],
        "ytick.color": STYLE["subtext"], "grid.color": STYLE["grid"],
        "figure.facecolor": STYLE["bg"],
    })

    # Viz 1: Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Negative", "Neutral", "Positive"]
    counts = [train_balance_stats["distribution_counts"]["negative"],
              train_balance_stats["distribution_counts"]["neutral"],
              train_balance_stats["distribution_counts"]["positive"]]
    colors = ["#f74f7a", "#f7914f", "#4ff7a0"]
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{int(bar.get_height()):,}", ha="center", color=STYLE["text"])
    
    ax.set_title("Cleaned Train Set Class Distribution", color=STYLE["text"], fontweight="bold")
    ax.set_ylabel("Samples")
    plt.savefig(os.path.join(VIZ, "class_distribution_cleaned.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Viz 2: Duplicate Removal Summary
    fig, ax = plt.subplots(figsize=(8, 5))
    splits_labels = ["train", "validation", "test"]
    final_sizes = [len(splits_clean[s]) for s in splits_labels]
    
    # We plot initial vs final
    x = np.arange(len(splits_labels))
    width = 0.35
    ax.bar(x - width/2, [initial_sizes[s] for s in splits_labels], width, label="Before Cleaning", color="#4f8ef7")
    ax.bar(x + width/2, final_sizes, width, label="After Cleaning & Leakage Removal", color="#4ff7a0")
    
    ax.set_xticks(x)
    ax.set_xticklabels(splits_labels)
    ax.set_title("Dataset Size Reduction", color=STYLE["text"], fontweight="bold")
    ax.legend(facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
    plt.savefig(os.path.join(VIZ, "duplicate_removal_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 8. Final Report Generation
    logger.info("\n[8] Generating Final Reports")
    
    impact_lines = [
        "===========================================================",
        "CLEANING IMPACT SUMMARY",
        "===========================================================",
        "DATASET SIZES",
        f"  Train      : {initial_sizes['train']:,} -> {len(splits_clean['train']):,} ({(len(splits_clean['train'])/initial_sizes['train'])*100:.1f}% remaining)",
        f"  Validation : {initial_sizes['validation']:,} -> {len(splits_clean['validation']):,} ({(len(splits_clean['validation'])/initial_sizes['validation'])*100:.1f}% remaining)",
        f"  Test       : {initial_sizes['test']:,} -> {len(splits_clean['test']):,} ({(len(splits_clean['test'])/initial_sizes['test'])*100:.1f}% remaining)",
        "",
        "CLASS DISTRIBUTION (TRAIN)",
        f"  Negative : {train_balance_stats['distribution_percentages']['negative']}%",
        f"  Neutral  : {train_balance_stats['distribution_percentages']['neutral']}%",
        f"  Positive : {train_balance_stats['distribution_percentages']['positive']}%",
        "",
        "RECOMMENDED CLASS WEIGHTS (Weighted Cross-Entropy)",
        f"  Weights : {train_balance_stats['class_weights']}",
        "==========================================================="
    ]
    
    with open(os.path.join(RESULTS, "cleaning_impact_summary.txt"), "w") as f:
        f.write("\n".join(impact_lines))

    summary_lines = [
        "===========================================================",
        "PHASE 2C SUMMARY - CPU-Efficient Fine-Tuning Preparation",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "===========================================================",
        "",
        "1. CLEANING RESULTS",
        f"  - Exact duplicates removed from train: {dedup_stats['train']['removed_exact_duplicates']:,}",
        f"  - Cross-split leakage removed: {leak_stats['removed_from_val_due_to_test'] + leak_stats['total_removed_from_train']:,} total samples purged.",
        "",
        "2. TOKEN STATISTICS (Cleaned Data)",
        f"  - max_length=96 covers {token_stats['pct_covered_by_96']:.2f}% of the cleaned dataset.",
        "  - Strategy: Proceed with max_length=96.",
        "",
        "3. CPU-EFFICIENT FINE-TUNING STRATEGY",
        "  - Device constraints: CPU only. Extreme optimization required.",
        "  - Batch Size: Recommend 8 or 16 per device to prevent OOM.",
        "  - Gradient Accumulation: Use steps=4 or 8 to achieve an effective batch size of 32 or 64.",
        "  - Layer Freezing: Recommend freezing the embedding layer and the first 6-8 encoder layers of IndoBERTweet.",
        "    (Only train the classification head and top 4 layers to drastically reduce backprop time).",
        "  - Optimizer: Use standard AdamW, avoid heavily memory-bound optimizers unless paged/8-bit is available (not usually on CPU).",
        "  - Epochs: Start with 3-5 epochs. Monitor validation loss closely.",
        "==========================================================="
    ]

    with open(os.path.join(RESULTS, "phase2c_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    logger.info("PHASE 2C COMPLETE. Clean datasets and reports generated.")

if __name__ == "__main__":
    main()
