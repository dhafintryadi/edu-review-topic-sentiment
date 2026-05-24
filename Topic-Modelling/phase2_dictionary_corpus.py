"""
PHASE_2: Dictionary & Corpus Engineering — Sekolah Rakyat
==========================================================
Input  : phase1_outputs/lda_token_lists.json
Output : phase2_outputs/
"""

import json, warnings
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gensim import corpora

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
PHASE1_DIR = Path(__file__).parent / "phase1_outputs"
OUT_DIR    = Path(__file__).parent / "phase2_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── A. Preserve Term Configuration ────────────────────────────────────────────
# Empirically validated from Phase 1 corpus analysis & sentiment review
OBSERVED_PRESERVE_TERMS = {
    "bug", "error", "crash", "lag", "loading", "stuck", "lambat",
    "login", "akun", "bayar", "mahal", "premium", "gratis",
    "materi", "soal", "video", "guru", "tidak",
    "belajar", "aplikasi",  # high-freq anchors with discriminative topic value
}
# Domain-informed heuristic terms (not directly observed in Phase 1 outputs)
HEURISTIC_PRESERVE_TERMS = {
    "freeze",      # application hang — related to crash/lag cluster
    "tidak_bisa",  # negated ability — compound complaint pattern
}
ALL_PRESERVE_TERMS = OBSERVED_PRESERVE_TERMS | HEURISTIC_PRESERVE_TERMS

# ── B. Barrier Phrase Configuration ───────────────────────────────────────────
# Confirmed by gensim Phrases model in Phase 1 (count >= threshold)
VALIDATED_DETECTED_BIGRAMS = [
    "ruang_guru", "mudah_paham", "latih_soal", "sering_error",
    "bank_soal", "kurang_lengkap", "bayar_mahal", "sering_keluar",
    "gabisa_buka", "sering_lag",
]
# Manually defined; confirmed present (count > 0) in candidate_phrase_validation.csv
VALIDATED_CANDIDATE_PHRASES = ["force_close", "stuck_loading"]

ALL_PROTECTED = ALL_PRESERVE_TERMS | set(VALIDATED_DETECTED_BIGRAMS) | set(VALIDATED_CANDIDATE_PHRASES)

# ── C. Filter Parameters ──────────────────────────────────────────────────────
NO_BELOW        = 5
NO_ABOVE        = 0.60
KEEP_N          = 8000
EMPTY_FLAG_PCT  = 5.0

print("=" * 65)
print("  PHASE 2 — Dictionary & Corpus Engineering")
print("=" * 65)

# ── 1. Load Phase 1 token lists ───────────────────────────────────────────────
print("\n[1] Loading Phase 1 token lists …")
with open(PHASE1_DIR / "lda_token_lists.json", "r", encoding="utf-8") as f:
    token_lists = json.load(f)
print(f"  Documents loaded   : {len(token_lists):,}")
print(f"  Total raw tokens   : {sum(len(d) for d in token_lists):,}")

# ── 2. Build raw gensim Dictionary ────────────────────────────────────────────
print("\n[2] Building raw gensim Dictionary …")
dictionary = corpora.Dictionary(token_lists)
pre_vocab   = len(dictionary)
print(f"  Pre-filter vocab   : {pre_vocab:,}")
print(f"  Num docs           : {dictionary.num_docs:,}")
print(f"  Num positions      : {dictionary.num_pos:,}")
print(f"  Raw bigrams        : {sum(1 for t in dictionary.token2id if '_' in t):,}")

# ── 3. Save pre-filter snapshot for whitelist restoration ─────────────────────
print("\n[3] Saving pre-filter snapshots …")
pre_stats = {tok: {"cfs": dictionary.cfs[tid], "dfs": dictionary.dfs[tid]}
             for tok, tid in dictionary.token2id.items()}
with open(OUT_DIR / "pre_filter_token2id_snapshot.json", "w", encoding="utf-8") as f:
    json.dump(dict(dictionary.token2id), f, ensure_ascii=False)

# Export pre-filter vocab CSV
pd.DataFrame([{
    "token": tok, "token_id": tid,
    "corpus_freq": dictionary.cfs[tid], "doc_freq": dictionary.dfs[tid],
    "doc_pct": round(dictionary.dfs[tid] / dictionary.num_docs * 100, 4),
    "is_protected": tok in ALL_PROTECTED, "is_bigram": "_" in tok,
} for tok, tid in dictionary.token2id.items()]
).sort_values("corpus_freq", ascending=False
).to_csv(OUT_DIR / "phase2_vocab_before.csv", index=False)
print("  pre_filter_token2id_snapshot.json  ->  saved")
print("  phase2_vocab_before.csv            ->  saved")

# ── 4. Apply conservative filter_extremes ────────────────────────────────────
print(f"\n[4] Applying filter_extremes (no_below={NO_BELOW}, no_above={NO_ABOVE}, keep_n={KEEP_N}) …")
dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=KEEP_N)
post_filter_vocab = len(dictionary)
print(f"  Post-filter vocab  : {post_filter_vocab:,}  (removed {pre_vocab - post_filter_vocab:,}, "
      f"retained {post_filter_vocab / pre_vocab * 100:.1f}%)")

# ── 5. Custom whitelist restoration ───────────────────────────────────────────
print("\n[5] Restoring removed protected tokens …")
restored, skipped = [], []
for token in sorted(ALL_PROTECTED):
    if token in dictionary.token2id:
        continue
    if token in pre_stats:
        new_id = max(dictionary.id2token, default=-1) + 1
        dictionary.token2id[token]  = new_id
        dictionary.id2token[new_id] = token
        dictionary.dfs[new_id]      = pre_stats[token]["dfs"]
        dictionary.cfs[new_id]      = pre_stats[token]["cfs"]
        category = ("OBSERVED" if token in OBSERVED_PRESERVE_TERMS
                    else "HEURISTIC" if token in HEURISTIC_PRESERVE_TERMS
                    else "BIGRAM" if token in VALIDATED_DETECTED_BIGRAMS
                    else "CANDIDATE")
        restored.append({"token": token, "category": category,
                         "dfs": pre_stats[token]["dfs"], "cfs": pre_stats[token]["cfs"]})
        print(f"  (R)  [{category:<10}] {token:<30} dfs={pre_stats[token]['dfs']}")
    else:
        skipped.append(token)
print(f"  Restored: {len(restored)}  |  Not in corpus (skipped): {len(skipped)}")
if skipped:
    print(f"  Skipped: {skipped}")

# ── 6. Compactify ─────────────────────────────────────────────────────────────
print("\n[6] Compactifying dictionary …")
dictionary.compactify()
final_vocab = len(dictionary)
print(f"  Final vocab size   : {final_vocab:,}")

# ── 7. Barrier survival audit ─────────────────────────────────────────────────
print("\n[7] Barrier survival audit …")
audit_rows = []
all_audit_ok = True
for group, terms in [
    ("VALIDATED_DETECTED_BIGRAMS",  VALIDATED_DETECTED_BIGRAMS),
    ("VALIDATED_CANDIDATE_PHRASES", VALIDATED_CANDIDATE_PHRASES),
    ("OBSERVED_PRESERVE_TERMS",     sorted(OBSERVED_PRESERVE_TERMS)),
    ("HEURISTIC_PRESERVE_TERMS",    sorted(HEURISTIC_PRESERVE_TERMS)),
]:
    survived = [t for t in terms if t in dictionary.token2id]
    missing  = [t for t in terms if t not in dictionary.token2id]
    if missing:
        all_audit_ok = False
    print(f"\n  [{group}] {len(survived)}/{len(terms)} survived")
    for t in survived:
        tid = dictionary.token2id[t]
        print(f"    [OK]  {t:<35} id={tid}  dfs={dictionary.dfs.get(tid,0)}")
    for t in missing:
        print(f"    [MISSING]  {t:<35}  *** MISSING ***")
    for t in terms:
        tid = dictionary.token2id.get(t)
        audit_rows.append({"token": t, "group": group, "survived": t in dictionary.token2id,
                           "final_id": tid, "doc_freq": dictionary.dfs.get(tid) if tid else None,
                           "corp_freq": dictionary.cfs.get(tid) if tid else None})

pd.DataFrame(audit_rows).to_csv(OUT_DIR / "phase2_barrier_survival.csv", index=False)
status_icon = "[PASS]" if all_audit_ok else "[WARN]"
print(f"\n  {status_icon}  Barrier audit: {'ALL PASSED' if all_audit_ok else 'FAILURES DETECTED'}")

# ── 8. Export post-filter vocabulary ─────────────────────────────────────────
print("\n[8] Exporting post-filter vocabulary …")
vocab_after_df = pd.DataFrame([{
    "token_id": tid, "token": tok,
    "corpus_freq": dictionary.cfs.get(tid, 0), "doc_freq": dictionary.dfs.get(tid, 0),
    "doc_pct": round(dictionary.dfs.get(tid, 0) / dictionary.num_docs * 100, 4),
    "is_protected": tok in ALL_PROTECTED, "is_bigram": "_" in tok,
    "is_validated_bigram": tok in VALIDATED_DETECTED_BIGRAMS,
    "is_candidate_phrase": tok in VALIDATED_CANDIDATE_PHRASES,
} for tok, tid in dictionary.token2id.items()]
).sort_values("corpus_freq", ascending=False)
vocab_after_df.to_csv(OUT_DIR / "phase2_vocab_after.csv", index=False)
print(f"  phase2_vocab_after.csv saved  |  bigrams in final vocab: {vocab_after_df['is_bigram'].sum()}")
print("\n  Top 30 terms (post-filter):")
print(vocab_after_df[["token", "corpus_freq", "doc_freq", "doc_pct",
                       "is_protected", "is_bigram"]].head(30).to_string(index=False))

# ── 9. Save gensim Dictionary ─────────────────────────────────────────────────
print("\n[9] Saving gensim Dictionary …")
dictionary.save(str(OUT_DIR / "lda_dictionary.gensim"))
print(f"  lda_dictionary.gensim  ->  saved")

# ── 10. Build BoW corpus ──────────────────────────────────────────────────────
print("\n[10] Building BoW corpus …")
bow_corpus    = [dictionary.doc2bow(tokens) for tokens in token_lists]
n_docs        = len(bow_corpus)
empty_docs    = sum(1 for d in bow_corpus if not d)
empty_pct     = empty_docs / n_docs * 100
bow_lengths   = [len(d) for d in bow_corpus]
total_nonzero = sum(bow_lengths)
sparsity      = (1 - total_nonzero / (n_docs * final_vocab)) * 100

print(f"  Total documents    : {n_docs:,}")
print(f"  Empty documents    : {empty_docs:,}  ({empty_pct:.2f}%)")
flag = "[WARN]" if empty_pct > EMPTY_FLAG_PCT else "[OK]"
print(f"  Empty doc check    : {flag}  (threshold={EMPTY_FLAG_PCT}%)")
print(f"  Sparsity (BoW)     : {sparsity:.4f}%")
print(f"  Matrix dims        : {n_docs:,} × {final_vocab:,}")
print(f"\n  BoW length distribution:")
print(pd.Series(bow_lengths).describe(percentiles=[.25,.5,.75,.90,.95]).round(2).to_string())

# ── 11. BoW length distribution plot ─────────────────────────────────────────
print("\n[11] Generating BoW length distribution plot …")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(bow_lengths, bins=40, color="#2980b9", edgecolor="white", alpha=0.85)
axes[0].axvline(np.median(bow_lengths), color="#e74c3c", linestyle="--",
                linewidth=1.5, label=f"Median={np.median(bow_lengths):.0f}")
axes[0].set_title("BoW Document Length Distribution", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Unique terms per doc (BoW)"); axes[0].set_ylabel("Frequency")
axes[0].legend(fontsize=9)

trimmed = [l for l in bow_lengths if l <= 50]
axes[1].hist(trimmed, bins=40, color="#27ae60", edgecolor="white", alpha=0.85)
axes[1].axvline(np.median(trimmed), color="#e74c3c", linestyle="--",
                linewidth=1.5, label=f"Median={np.median(trimmed):.0f}")
axes[1].set_title("BoW Length (zoomed: ≤50 terms)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Unique terms per doc (BoW)"); axes[1].legend(fontsize=9)

plt.suptitle("Phase 2 — BoW Corpus Document Length", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "phase2_bow_length_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  phase2_bow_length_distribution.png  ->  saved")

# ── 12. Export BoW corpus in Market Matrix format ────────────────────────────
print("\n[12] Saving BoW corpus (Market Matrix format) …")
mm_path = OUT_DIR / "lda_corpus.mm"
corpora.MmCorpus.serialize(str(mm_path), bow_corpus)
print(f"  lda_corpus.mm        ->  saved")
print(f"  lda_corpus.mm.index  ->  saved")

# ── 13. Export phase summary JSON ────────────────────────────────────────────
print("\n[13] Exporting phase summary …")
bow_len_series = pd.Series(bow_lengths)
summary = {
    "phase": "PHASE_2_DICTIONARY_CORPUS_ENGINEERING",
    "filter_params": {"no_below": NO_BELOW, "no_above": NO_ABOVE, "keep_n": KEEP_N},
    "vocab": {
        "pre_filter":       pre_vocab,
        "post_filter":      post_filter_vocab,
        "final":            final_vocab,
        "tokens_restored":  len(restored),
        "tokens_skipped":   len(skipped),
        "reduction_pct":    round((pre_vocab - final_vocab) / pre_vocab * 100, 2),
        "bigrams_in_final": int(vocab_after_df["is_bigram"].sum()),
    },
    "bow_corpus": {
        "total_docs":     n_docs,
        "empty_docs":     empty_docs,
        "empty_pct":      round(empty_pct, 4),
        "empty_flag":     empty_pct > EMPTY_FLAG_PCT,
        "total_nonzero":  total_nonzero,
        "sparsity_pct":   round(sparsity, 4),
        "bow_len_mean":   round(float(bow_len_series.mean()), 2),
        "bow_len_median": float(bow_len_series.median()),
        "bow_len_p75":    float(bow_len_series.quantile(0.75)),
        "bow_len_p90":    float(bow_len_series.quantile(0.90)),
        "bow_len_max":    int(bow_len_series.max()),
    },
    "barrier_audit": {
        "all_passed": all_audit_ok,
        "validated_bigrams_survived":  sum(1 for t in VALIDATED_DETECTED_BIGRAMS if t in dictionary.token2id),
        "candidate_phrases_survived":  sum(1 for t in VALIDATED_CANDIDATE_PHRASES if t in dictionary.token2id),
        "observed_terms_survived":     sum(1 for t in OBSERVED_PRESERVE_TERMS if t in dictionary.token2id),
        "heuristic_terms_survived":    sum(1 for t in HEURISTIC_PRESERVE_TERMS if t in dictionary.token2id),
    },
    "artifacts": [
        "lda_dictionary.gensim", "lda_corpus.mm", "lda_corpus.mm.index",
        "phase2_vocab_before.csv", "phase2_vocab_after.csv",
        "phase2_barrier_survival.csv", "phase2_bow_length_distribution.png",
        "pre_filter_token2id_snapshot.json", "phase2_summary.json",
    ],
}
with open(OUT_DIR / "phase2_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 65)
print("  PHASE 2 COMPLETE")
print(f"  Outputs -> {OUT_DIR}")
print("=" * 65)
for p in sorted(OUT_DIR.iterdir()):
    print(f"    {p.name:<50} {p.stat().st_size/1024:>8.1f} KB")
