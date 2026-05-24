"""
PHASE 2B - Step 1-4: Dataset Audit, Token Length Analysis,
Truncation Impact Analysis, and Visualization Generation
Model: indolem/indobertweet-base-uncased
"""

import os, sys, json, logging, datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from transformers import AutoTokenizer

# ── Paths ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
RESULTS  = os.path.join(ROOT, "results")
LOGS     = os.path.join(ROOT, "logs")
VIZ      = os.path.join(ROOT, "visualizations")
DATA_DIR = os.path.join(ROOT, "data", "processed")

for d in [RESULTS, LOGS, VIZ]:
    os.makedirs(d, exist_ok=True)

# ── Logging ────────────────────────────────────────────────
LOG_FILE = os.path.join(LOGS, "phase2b.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

MODEL_NAME   = "indolem/indobertweet-base-uncased"
TEXT_COL     = "content"
LABEL_COL    = "sentiment_id"
SPLITS       = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
MAX_LENGTHS  = [64, 96, 128, 256]

log.info("=" * 60)
log.info("PHASE 2B - Token Length Analysis & Dataset Audit")
log.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
log.info("=" * 60)

# ═══════════════════════════════════════════════════════════
# STEP 1 — Dataset Audit
# ═══════════════════════════════════════════════════════════
log.info("\n[STEP 1] Dataset Audit")

frames = {}
for split, fname in SPLITS.items():
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)
    frames[split] = df
    log.info(f"  Loaded {split}: {len(df):,} rows, {df.columns.tolist()}")

# Duplicate analysis per split
dup_report = {}
for split, df in frames.items():
    dup_mask   = df.duplicated(subset=[TEXT_COL], keep=False)
    dup_count  = df.duplicated(subset=[TEXT_COL], keep="first").sum()
    dup_report[split] = {
        "total_rows":      int(len(df)),
        "duplicate_rows":  int(dup_count),
        "duplicate_ratio": round(dup_count / len(df) * 100, 2),
        "unique_rows":     int(len(df) - dup_count),
    }
    log.info(f"  {split}: {dup_count:,} duplicates ({dup_report[split]['duplicate_ratio']}%)")

# Cross-split leakage check
train_texts = set(frames["train"][TEXT_COL].dropna())
val_texts   = set(frames["validation"][TEXT_COL].dropna())
test_texts  = set(frames["test"][TEXT_COL].dropna())
train_val_leak  = len(train_texts & val_texts)
train_test_leak = len(train_texts & test_texts)
val_test_leak   = len(val_texts  & test_texts)
log.info(f"  Leakage train↔val: {train_val_leak}, train↔test: {train_test_leak}, val↔test: {val_test_leak}")

dup_report["cross_split_leakage"] = {
    "train_val":  train_val_leak,
    "train_test": train_test_leak,
    "val_test":   val_test_leak,
}

with open(os.path.join(RESULTS, "duplicate_analysis.json"), "w") as f:
    json.dump(dup_report, f, indent=2)
log.info("  Saved: results/duplicate_analysis.json")

# ═══════════════════════════════════════════════════════════
# STEP 2 — Token Length Analysis
# ═══════════════════════════════════════════════════════════
log.info("\n[STEP 2] Token Length Analysis")

log.info(f"  Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

all_texts = pd.concat([
    frames["train"][[TEXT_COL]],
    frames["validation"][[TEXT_COL]],
    frames["test"][[TEXT_COL]],
], ignore_index=True)[TEXT_COL].dropna().astype(str).tolist()

log.info(f"  Tokenizing {len(all_texts):,} texts (this may take a while)...")

BATCH_SIZE = 512
token_lengths = []
for i in range(0, len(all_texts), BATCH_SIZE):
    batch = all_texts[i:i+BATCH_SIZE]
    enc   = tokenizer(batch, add_special_tokens=True, truncation=False)
    token_lengths.extend([len(ids) for ids in enc["input_ids"]])
    if (i // BATCH_SIZE) % 20 == 0:
        log.info(f"    Processed {min(i+BATCH_SIZE, len(all_texts)):,}/{len(all_texts):,}")

arr = np.array(token_lengths)

token_analysis = {
    "total_samples": int(len(arr)),
    "mean":    round(float(arr.mean()), 2),
    "median":  round(float(np.median(arr)), 2),
    "std":     round(float(arr.std()), 2),
    "min":     int(arr.min()),
    "max":     int(arr.max()),
    "p50":     int(np.percentile(arr, 50)),
    "p75":     int(np.percentile(arr, 75)),
    "p90":     int(np.percentile(arr, 90)),
    "p95":     int(np.percentile(arr, 95)),
    "p99":     int(np.percentile(arr, 99)),
    "p100":    int(np.percentile(arr, 100)),
}

log.info(f"  Mean={token_analysis['mean']}, Median={token_analysis['median']}, "
         f"P90={token_analysis['p90']}, P95={token_analysis['p95']}, "
         f"P99={token_analysis['p99']}, Max={token_analysis['max']}")

# Per-split token lengths
split_token_stats = {}
offset = 0
for split, df in frames.items():
    texts  = df[TEXT_COL].dropna().astype(str).tolist()
    n      = len(texts)
    slens  = np.array(token_lengths[offset:offset+n])
    offset += n
    split_token_stats[split] = {
        "mean":   round(float(slens.mean()), 2),
        "median": round(float(np.median(slens)), 2),
        "p90":    int(np.percentile(slens, 90)),
        "p95":    int(np.percentile(slens, 95)),
        "max":    int(slens.max()),
    }

token_analysis["per_split"] = split_token_stats

with open(os.path.join(RESULTS, "token_length_analysis.json"), "w") as f:
    json.dump(token_analysis, f, indent=2)
log.info("  Saved: results/token_length_analysis.json")

# ═══════════════════════════════════════════════════════════
# STEP 3 — Truncation Impact Analysis
# ═══════════════════════════════════════════════════════════
log.info("\n[STEP 3] Truncation Impact Analysis")

truncation_report = {}
recommended_max_length = 128  # default

for ml in MAX_LENGTHS:
    truncated     = int((arr > ml).sum())
    pct_truncated = round(truncated / len(arr) * 100, 2)
    pct_covered   = round(100 - pct_truncated, 2)
    truncation_report[str(ml)] = {
        "max_length":      ml,
        "truncated_count": truncated,
        "pct_truncated":   pct_truncated,
        "pct_covered":     pct_covered,
    }
    log.info(f"  max_length={ml:>4}: {pct_covered:.1f}% covered, {pct_truncated:.1f}% truncated ({truncated:,} samples)")

# Recommend: smallest max_length covering >= 95% samples
for ml in MAX_LENGTHS:
    if truncation_report[str(ml)]["pct_covered"] >= 95.0:
        recommended_max_length = ml
        break

truncation_report["recommendation"] = {
    "recommended_max_length":  recommended_max_length,
    "rationale": (
        f"Covers >= 95% of samples with max_length={recommended_max_length}, "
        f"optimal for CPU memory efficiency."
    ),
    "cpu_notes": [
        "Lower max_length reduces sequence padding overhead",
        "Smaller batches needed for longer sequences on CPU",
        "max_length=128 is standard for tweet-length texts",
    ]
}

with open(os.path.join(RESULTS, "truncation_impact_analysis.json"), "w") as f:
    json.dump(truncation_report, f, indent=2)
log.info(f"  Recommendation: max_length={recommended_max_length}")
log.info("  Saved: results/truncation_impact_analysis.json")

# ═══════════════════════════════════════════════════════════
# STEP 4 — Visualizations
# ═══════════════════════════════════════════════════════════
log.info("\n[STEP 4] Generating Visualizations")

STYLE = {
    "bg":       "#0f1117",
    "panel":    "#1a1d27",
    "accent1":  "#4f8ef7",
    "accent2":  "#f7914f",
    "accent3":  "#4ff7a0",
    "accent4":  "#f74f7a",
    "text":     "#e0e0e0",
    "subtext":  "#9099b0",
    "grid":     "#252a3a",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color":       STYLE["text"],
    "axes.facecolor":   STYLE["panel"],
    "axes.edgecolor":   STYLE["grid"],
    "axes.labelcolor":  STYLE["text"],
    "xtick.color":      STYLE["subtext"],
    "ytick.color":      STYLE["subtext"],
    "grid.color":       STYLE["grid"],
    "figure.facecolor": STYLE["bg"],
})

# ── VIZ 1: Token Length Distribution ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(STYLE["bg"])
fig.suptitle("Token Length Distribution — IndoBERTweet Tokenizer",
             fontsize=14, fontweight="bold", color=STYLE["text"], y=1.01)

ax = axes[0]
ax.set_facecolor(STYLE["panel"])
counts, bin_edges, patches = ax.hist(arr, bins=60, color=STYLE["accent1"], alpha=0.85, edgecolor="none")
for ml, col in zip([128, 256], [STYLE["accent2"], STYLE["accent3"]]):
    ax.axvline(ml, color=col, linewidth=1.5, linestyle="--", label=f"max={ml}")
ax.axvline(token_analysis["p95"], color=STYLE["accent4"], linewidth=1.5, linestyle=":", label=f"P95={token_analysis['p95']}")
ax.set_xlabel("Token Length", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Histogram (All Splits)", fontsize=11, color=STYLE["subtext"])
ax.legend(fontsize=9, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
ax.grid(axis="y", alpha=0.3)

stats_text = (
    f"n={token_analysis['total_samples']:,}\n"
    f"Mean={token_analysis['mean']:.1f}\n"
    f"Median={token_analysis['median']:.1f}\n"
    f"P90={token_analysis['p90']}\n"
    f"P95={token_analysis['p95']}\n"
    f"P99={token_analysis['p99']}\n"
    f"Max={token_analysis['max']}"
)
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", horizontalalignment="right",
        color=STYLE["text"], bbox=dict(boxstyle="round,pad=0.4",
        facecolor=STYLE["bg"], edgecolor=STYLE["grid"], alpha=0.8))

ax2 = axes[1]
ax2.set_facecolor(STYLE["panel"])
colors_split = [STYLE["accent1"], STYLE["accent2"], STYLE["accent3"]]
for (split, df), col in zip(frames.items(), colors_split):
    slens = np.array([len(ids) for ids in
                      tokenizer(df[TEXT_COL].dropna().astype(str).tolist()[:3000],
                                add_special_tokens=True, truncation=False)["input_ids"]])
    ax2.hist(slens, bins=40, color=col, alpha=0.6, label=split, edgecolor="none")
ax2.set_xlabel("Token Length", fontsize=11)
ax2.set_ylabel("Frequency", fontsize=11)
ax2.set_title("Per-Split Distribution (sample 3K each)", fontsize=11, color=STYLE["subtext"])
ax2.legend(fontsize=9, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out1 = os.path.join(VIZ, "token_length_distribution.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
plt.close()
log.info(f"  Saved: {out1}")

# ── VIZ 2: Truncation Tradeoff ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(STYLE["bg"])
ax.set_facecolor(STYLE["panel"])

mls    = MAX_LENGTHS
covs   = [truncation_report[str(ml)]["pct_covered"]   for ml in mls]
truncs = [truncation_report[str(ml)]["pct_truncated"]  for ml in mls]

x = np.arange(len(mls))
w = 0.35
bars1 = ax.bar(x - w/2, covs,   w, label="% Covered",   color=STYLE["accent3"], alpha=0.85)
bars2 = ax.bar(x + w/2, truncs, w, label="% Truncated", color=STYLE["accent4"], alpha=0.85)

for bar, val in zip(bars1, covs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=STYLE["text"])
for bar, val in zip(bars2, truncs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=STYLE["text"])

ax.axhline(95, color=STYLE["accent2"], linewidth=1.5, linestyle="--", label="95% threshold")
ax.set_xticks(x)
ax.set_xticklabels([f"max_length\n{ml}" for ml in mls], fontsize=10)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_title("Truncation Impact Analysis — Coverage vs Truncation by max_length",
             fontsize=12, fontweight="bold", color=STYLE["text"])
ax.legend(fontsize=9, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 110)

rec_x = mls.index(recommended_max_length)
ax.annotate(f"Recommended\nmax_length={recommended_max_length}",
            xy=(rec_x - w/2, covs[rec_x]),
            xytext=(rec_x - w/2 - 0.3, covs[rec_x] + 8),
            fontsize=8.5, color=STYLE["accent2"],
            arrowprops=dict(arrowstyle="->", color=STYLE["accent2"]))

plt.tight_layout()
out2 = os.path.join(VIZ, "truncation_tradeoff.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
plt.close()
log.info(f"  Saved: {out2}")

# ── VIZ 3: Duplicate Distribution ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(STYLE["bg"])
fig.suptitle("Dataset Audit — Duplicate & Sentiment Distribution",
             fontsize=13, fontweight="bold", color=STYLE["text"])

ax = axes[0]
ax.set_facecolor(STYLE["panel"])
split_names = list(dup_report.keys())
split_names = [s for s in split_names if s != "cross_split_leakage"]
uniques  = [dup_report[s]["unique_rows"]    for s in split_names]
dups_val = [dup_report[s]["duplicate_rows"] for s in split_names]
x = np.arange(len(split_names))
ax.bar(x, uniques,  label="Unique",     color=STYLE["accent1"], alpha=0.85)
ax.bar(x, dups_val, bottom=uniques, label="Duplicate", color=STYLE["accent4"], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(split_names, fontsize=11)
ax.set_ylabel("Row Count", fontsize=11)
ax.set_title("Duplicate Rows per Split", fontsize=11, color=STYLE["subtext"])
ax.legend(fontsize=9, facecolor=STYLE["panel"], edgecolor=STYLE["grid"])
ax.grid(axis="y", alpha=0.3)
for i, (s, d) in enumerate(zip(split_names, dups_val)):
    ax.text(i, uniques[i] + d + 100, f"{dup_report[s]['duplicate_ratio']}%",
            ha="center", fontsize=9, color=STYLE["accent4"])

ax2 = axes[1]
ax2.set_facecolor(STYLE["panel"])
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
label_colors = {0: STYLE["accent4"], 1: STYLE["accent2"], 2: STYLE["accent3"]}
train_df = frames["train"]
if LABEL_COL in train_df.columns:
    counts_lbl = train_df[LABEL_COL].value_counts().sort_index()
    labels_sorted = [label_map.get(k, str(k)) for k in counts_lbl.index]
    colors_sorted = [label_colors.get(k, STYLE["accent1"]) for k in counts_lbl.index]
    bars = ax2.bar(labels_sorted, counts_lbl.values, color=colors_sorted, alpha=0.85)
    for bar, val in zip(bars, counts_lbl.values):
        pct = val / counts_lbl.sum() * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{val:,}\n({pct:.1f}%)", ha="center", fontsize=9, color=STYLE["text"])
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Sentiment Distribution (Train Split)", fontsize=11, color=STYLE["subtext"])
    ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out3 = os.path.join(VIZ, "duplicate_distribution.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
plt.close()
log.info(f"  Saved: {out3}")

log.info("\n[STEP 1-4] COMPLETE")
log.info(f"  Recommended max_length: {recommended_max_length}")
log.info(f"  Duplicate ratio (train): {dup_report['train']['duplicate_ratio']}%")
log.info(f"  Cross-split leakage train↔val: {train_val_leak}")
