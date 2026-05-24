"""
PHASE_0: Exploratory Corpus Analysis — Sekolah Rakyat
======================================================
Objectives:
  1. Dataset audit & quality checks
  2. Token length distribution (content_topic)
  3. Vocabulary frequency & top-N terms
  4. Corpus sparsity estimate
  5. Score distribution & sentiment label breakdown
  6. Negative-review complaint vocabulary
  7. Severity-signal keyword frequency (learning barrier proxies)
  8. Export summary JSON + CSVs for downstream topic modelling

Execution: python eda_corpus_analysis.py
Outputs  : PKL/Topic-Modelling/eda_outputs/
"""

import os
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MAIN_CSV   = ROOT / "Datasets/main_review_preprocessed/topic_preprocessed.csv"
BENCH_CSV  = ROOT / "Datasets/benchmark_review_preprocessed/topic_preprocessed.csv"
SENTI_CSV  = ROOT / "Sentiment-Analysis/sentiment-analysis-model/results/final_sentiment_inference.csv"
OUT_DIR    = Path(__file__).parent / "eda_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  PHASE 0 — Exploratory Corpus Analysis")
print("=" * 65)

# ── 1. Load datasets ───────────────────────────────────────────────────────────
print("\n[1] Loading datasets …")
main_df  = pd.read_csv(MAIN_CSV,  low_memory=False)
bench_df = pd.read_csv(BENCH_CSV, low_memory=False)
senti_df = pd.read_csv(SENTI_CSV, low_memory=False)

# Isolate main-source sentiment rows
senti_main = senti_df[senti_df["source"] == "main"].copy()

print(f"  main_dataset  : {len(main_df):,} rows, {main_df.shape[1]} cols")
print(f"  bench_dataset : {len(bench_df):,} rows, {bench_df.shape[1]} cols")
print(f"  sentiment_inf : {len(senti_df):,} rows  (main={len(senti_main):,})")

# ── 2. Quality audit ───────────────────────────────────────────────────────────
print("\n[2] Quality audit …")

def audit(df, name):
    total = len(df)
    ct_null  = df["content_topic"].isna().sum()
    ct_empty = (df["content_topic"].fillna("").str.strip() == "").sum()
    cr_null  = df["content_raw"].isna().sum()
    dup      = df.duplicated(subset="reviewId").sum() if "reviewId" in df.columns else 0
    return {
        "dataset"            : name,
        "total_rows"         : total,
        "content_topic_null" : int(ct_null),
        "content_topic_empty": int(ct_empty),
        "content_raw_null"   : int(cr_null),
        "duplicate_reviewId" : int(dup),
        "usable_rows"        : int(total - ct_empty),
    }

audit_results = [audit(main_df, "main"), audit(bench_df, "benchmark")]
audit_df = pd.DataFrame(audit_results)
audit_df.to_csv(OUT_DIR / "audit_summary.csv", index=False)
print(audit_df.to_string(index=False))

# ── 3. Score distribution ──────────────────────────────────────────────────────
print("\n[3] Score distribution …")
score_dist = main_df["score"].value_counts().sort_index()
print(score_dist.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
score_dist.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
ax.set_title("Review Score Distribution — Main Dataset", fontsize=13, fontweight="bold")
ax.set_xlabel("Star Rating"); ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUT_DIR / "score_distribution.png", dpi=150)
plt.close()

# ── 4. Sentiment label breakdown (from inference) ──────────────────────────────
print("\n[4] Sentiment label breakdown (main source) …")
label_dist = senti_main["predicted_label_name"].value_counts()
pct = (label_dist / len(senti_main) * 100).round(2)
print(pd.concat([label_dist, pct], axis=1, keys=["count", "%"]).to_string())

fig, ax = plt.subplots(figsize=(6, 4))
colors = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
label_dist.plot(kind="bar", ax=ax,
                color=[colors.get(l, "#95a5a6") for l in label_dist.index],
                edgecolor="white")
ax.set_title("Predicted Sentiment Distribution — Main Dataset", fontsize=12, fontweight="bold")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUT_DIR / "sentiment_distribution.png", dpi=150)
plt.close()

# ── 5. Token length distribution ──────────────────────────────────────────────
print("\n[5] Token length distribution (content_topic) …")
# Use only non-empty rows
usable = main_df[main_df["content_topic"].notna() &
                 (main_df["content_topic"].str.strip() != "")].copy()
usable["token_count"] = usable["content_topic"].str.split().str.len()

stats = usable["token_count"].describe(percentiles=[.25, .5, .75, .90, .95]).round(2)
print(stats.to_string())
stats.to_csv(OUT_DIR / "token_length_stats.csv", header=["value"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Histogram
axes[0].hist(usable["token_count"], bins=50, color="#4C72B0", edgecolor="white")
axes[0].set_title("Token Length Distribution", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Tokens per review"); axes[0].set_ylabel("Frequency")
# Box plot
axes[1].boxplot(usable["token_count"], vert=True, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", color="white"),
                medianprops=dict(color="#f39c12", linewidth=2))
axes[1].set_title("Token Length Box Plot", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Tokens")
plt.tight_layout()
plt.savefig(OUT_DIR / "token_length_distribution.png", dpi=150)
plt.close()

# ── 6. Vocabulary frequency ────────────────────────────────────────────────────
print("\n[6] Vocabulary frequency analysis …")
all_tokens = []
for text in usable["content_topic"].dropna():
    all_tokens.extend(str(text).split())

vocab_counter = Counter(all_tokens)
vocab_size    = len(vocab_counter)
total_tokens  = sum(vocab_counter.values())

print(f"  Vocabulary size : {vocab_size:,}")
print(f"  Total tokens    : {total_tokens:,}")
print(f"  Unique/Total    : {vocab_size/total_tokens:.4f}")

# Top 50
top50 = vocab_counter.most_common(50)
top50_df = pd.DataFrame(top50, columns=["term", "frequency"])
top50_df["relative_freq"] = (top50_df["frequency"] / total_tokens * 100).round(4)
top50_df.to_csv(OUT_DIR / "top50_vocabulary.csv", index=False)
print("\n  Top 20 terms:")
print(top50_df.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 5))
ax.barh(top50_df["term"][:30][::-1], top50_df["frequency"][:30][::-1], color="#4C72B0")
ax.set_title("Top 30 Terms — Full Corpus (content_topic)", fontsize=12, fontweight="bold")
ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig(OUT_DIR / "top30_vocabulary.png", dpi=150)
plt.close()

# ── 7. Corpus sparsity ─────────────────────────────────────────────────────────
print("\n[7] Corpus sparsity estimate …")
n_docs  = len(usable)
# Sparsity via TF-DF: fraction of (doc, term) pairs that are zero
# Approximate: avg non-zero entries per doc / vocab size
avg_unique_per_doc = np.mean([len(set(str(t).split())) for t in usable["content_topic"].sample(min(5000, n_docs), random_state=42)])
sparsity = 1 - (avg_unique_per_doc / vocab_size)
print(f"  Docs           : {n_docs:,}")
print(f"  Vocab size     : {vocab_size:,}")
print(f"  Avg unique/doc : {avg_unique_per_doc:.1f}")
print(f"  Est. sparsity  : {sparsity:.4f} ({sparsity*100:.2f}%)")

sparsity_info = {
    "n_docs": n_docs, "vocab_size": vocab_size,
    "avg_unique_terms_per_doc": round(avg_unique_per_doc, 2),
    "estimated_sparsity": round(sparsity, 6),
    "sparsity_pct": round(sparsity * 100, 4)
}

# ── 8. Negative reviews — complaint vocabulary ─────────────────────────────────
print("\n[8] Negative-review complaint vocabulary …")
neg_ids = set(senti_main[senti_main["predicted_label_name"] == "negative"]["reviewId"])
neg_df  = usable[usable["reviewId"].isin(neg_ids)].copy()
print(f"  Negative docs matched : {len(neg_df):,}")

neg_tokens = []
for text in neg_df["content_topic"].dropna():
    neg_tokens.extend(str(text).split())

neg_counter = Counter(neg_tokens)
top_neg50 = pd.DataFrame(neg_counter.most_common(50), columns=["term", "frequency"])
top_neg50.to_csv(OUT_DIR / "top50_negative_vocabulary.csv", index=False)
print("\n  Top 20 negative terms:")
print(top_neg50.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 5))
ax.barh(top_neg50["term"][:30][::-1], top_neg50["frequency"][:30][::-1], color="#e74c3c")
ax.set_title("Top 30 Terms — Negative Reviews Only", fontsize=12, fontweight="bold")
ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig(OUT_DIR / "top30_negative_vocabulary.png", dpi=150)
plt.close()

# ── 9. Severity-oriented keyword signals ──────────────────────────────────────
print("\n[9] Severity-oriented learning barrier signals …")

BARRIER_KEYWORDS = {
    "Technical / App Crash"   : ["bug", "crash", "error", "eror", "lag", "ngelag", "lemot",
                                  "loading", "stuck", "freeze", "keluar", "force", "restart",
                                  "not responding", "hang", "blank", "putih"],
    "Paywall / Access Barrier" : ["premium", "berbayar", "bayar", "mahal", "langganan",
                                   "berlangganan", "gratis", "biaya", "harga", "free"],
    "Content Gap"             : ["kurang", "tidak ada", "gak ada", "belum ada", "kosong",
                                  "minim", "terbatas", "sedikit", "incomplete"],
    "Login / Auth Issues"     : ["login", "masuk", "daftar", "akun", "sandi", "password",
                                  "register", "verifikasi", "otp"],
    "UX / Navigation"         : ["bingung", "susah", "ribet", "rumit", "sulit", "tampilan",
                                  "antarmuka", "ui", "ux", "navigasi", "menu"],
    "Video / Media Failure"   : ["video", "nonton", "streaming", "buffering", "putar",
                                  "tidak muncul", "blank", "hitam", "audio", "suara"],
    "Curriculum Mismatch"     : ["kurikulum", "materi", "pelajaran", "silabus",
                                  "tidak sesuai", "beda", "salah materi"],
    "Notification Harassment" : ["chat", "wa", "whatsapp", "telepon", "telp", "dihubungi",
                                  "spam", "bawel", "ganggu"],
}

severity_rows = []
for category, kws in BARRIER_KEYWORDS.items():
    pattern = "|".join([re.escape(k) for k in kws])
    # Count in all usable docs
    all_hits = usable["content_topic"].str.contains(pattern, case=False, na=False).sum()
    # Count in negative docs
    neg_hits = neg_df["content_topic"].str.contains(pattern, case=False, na=False).sum()
    neg_rate = round(neg_hits / len(neg_df) * 100, 2) if len(neg_df) > 0 else 0
    severity_rows.append({
        "barrier_category"       : category,
        "total_corpus_mentions"  : int(all_hits),
        "negative_doc_mentions"  : int(neg_hits),
        "neg_doc_hit_rate_pct"   : neg_rate,
        "severity_signal"        : "HIGH" if neg_rate >= 20 else ("MEDIUM" if neg_rate >= 8 else "LOW"),
    })

severity_df = pd.DataFrame(severity_rows).sort_values("neg_doc_hit_rate_pct", ascending=False)
severity_df.to_csv(OUT_DIR / "severity_barrier_signals.csv", index=False)
print(severity_df.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
color_map = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}
bars = ax.barh(
    severity_df["barrier_category"],
    severity_df["neg_doc_hit_rate_pct"],
    color=[color_map[s] for s in severity_df["severity_signal"]],
)
ax.set_title("Severity Signals — Barrier Keyword Hit Rate in Negative Reviews",
             fontsize=11, fontweight="bold")
ax.set_xlabel("% of Negative Reviews Containing Keyword")
for bar, val in zip(bars, severity_df["neg_doc_hit_rate_pct"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "severity_barrier_signals.png", dpi=150)
plt.close()

# ── 10. Confidence score distribution (negative class) ────────────────────────
print("\n[10] Confidence score distribution (negative predictions) …")
neg_senti = senti_main[senti_main["predicted_label_name"] == "negative"]
conf_stats = neg_senti["confidence_score"].describe(percentiles=[.25, .5, .75]).round(4)
print(conf_stats.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(neg_senti["confidence_score"], bins=40, color="#e74c3c", edgecolor="white", alpha=0.85)
ax.axvline(neg_senti["confidence_score"].mean(), color="black", linestyle="--",
           label=f"Mean={neg_senti['confidence_score'].mean():.3f}")
ax.set_title("Confidence Score Distribution — Negative Predictions", fontsize=12, fontweight="bold")
ax.set_xlabel("Confidence Score"); ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_negative_distribution.png", dpi=150)
plt.close()

# ── 11. Export master summary JSON ────────────────────────────────────────────
print("\n[11] Writing master summary …")
summary = {
    "phase"   : "PHASE_0_EXPLORATORY_CORPUS_ANALYSIS",
    "datasets": {
        "main_total_rows"     : int(len(main_df)),
        "main_usable_rows"    : int(len(usable)),
        "benchmark_total_rows": int(len(bench_df)),
        "sentiment_main_rows" : int(len(senti_main)),
    },
    "score_distribution"  : score_dist.to_dict(),
    "sentiment_distribution": {
        "count"      : label_dist.to_dict(),
        "percentage" : pct.to_dict(),
    },
    "token_length_stats"  : stats.to_dict(),
    "vocabulary"          : {
        "vocab_size"   : vocab_size,
        "total_tokens" : total_tokens,
    },
    "sparsity"            : sparsity_info,
    "negative_vocab_top20": top_neg50.head(20).to_dict(orient="records"),
    "severity_signals"    : severity_df.to_dict(orient="records"),
}

with open(OUT_DIR / "eda_master_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

# ── Done ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 0 COMPLETE")
print(f"  Outputs written to: {OUT_DIR}")
print("=" * 65)
outputs = list(OUT_DIR.iterdir())
for o in sorted(outputs):
    size = o.stat().st_size
    print(f"    {o.name:<45} {size/1024:>7.1f} KB")
