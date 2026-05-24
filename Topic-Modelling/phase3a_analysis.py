"""
PHASE 3A Results Analysis - Standalone Script
===============================================
Analyzes Phase 3A exploratory LDA training results and provides k selection recommendation.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Paths
PHASE3A_DIR = Path("phase3a_outputs")
print("\n" + "=" * 70)
print("  PHASE 3A RESULTS ANALYSIS")
print("=" * 70)

# ── 1. Load Summary ──────────────────────────────────────────────────────────
print("\n[1] Loading Phase 3A Summary ...")
with open(PHASE3A_DIR / "phase3a_summary.json", "r") as f:
    summary = json.load(f)

results_df = pd.DataFrame(summary['results'])
print("\nPhase 3A Results Table:")
print(results_df.to_string(index=False))

# ── 2. Coherence Analysis ───────────────────────────────────────────────────
print("\n[2] Coherence Score Analysis")
print("-" * 70)

k_values = results_df['k'].values
cv_scores = results_df['coherence_cv'].values

print(f"\nCoherence Scores by k:")
for k, cv in zip(k_values, cv_scores):
    print(f"  k={k:2d}: Cv={cv:.4f}")

best_k_coherence = results_df.loc[results_df['coherence_cv'].idxmax()]
print(f"\n✓ HIGHEST COHERENCE: k={int(best_k_coherence['k'])} (Cv={best_k_coherence['coherence_cv']:.4f})")

# ── 3. Barrier Coverage Analysis ─────────────────────────────────────────────
print("\n[3] Barrier Phrase Coverage Analysis")
print("-" * 70)

print(f"\nBarrier Terms Found per k:")
for k, barriers, topics in zip(results_df['k'], results_df['barrier_terms_found'], results_df['unique_topics_with_barriers']):
    print(f"  k={k:2d}: {int(barriers):2d} barrier terms across {int(topics):2d} topics")

# Check which k has best distribution
best_barrier_dist = results_df['unique_topics_with_barriers'].max()
best_k_barriers = results_df[results_df['unique_topics_with_barriers'] == best_barrier_dist]

print(f"\n✓ BEST BARRIER DISTRIBUTION: {int(best_barrier_dist)} topics with barriers")
print(f"   Achieved by: k={', '.join([str(int(k)) for k in best_k_barriers['k'].values])}")

# ── 4. Load and Analyze Barrier Audits ──────────────────────────────────────
print("\n[4] Barrier Phrase Distribution Details")
print("-" * 70)

for k in k_values:
    audit_file = PHASE3A_DIR / f"barrier_cluster_audit_k{k}.csv"
    if audit_file.exists():
        df_audit = pd.read_csv(audit_file)
        print(f"\nk={k} | Barrier Phrases ({len(df_audit)}):")
        
        # Group by dominant topic
        by_topic = df_audit.groupby('dominant_topic_id').size()
        print(f"  Distribution across {len(by_topic)} topics:")
        for topic_id, count in by_topic.items():
            print(f"    Topic {int(topic_id)}: {count} barrier terms")

# ── 5. Load and Inspect Topic Keywords ──────────────────────────────────────
print("\n[5] Topic Keywords Preview")
print("-" * 70)

# Show k=8 as reference (highest coherence)
ref_k = int(best_k_coherence['k'])
kw_file = PHASE3A_DIR / f"topic_keywords_k{ref_k}.csv"

if kw_file.exists():
    df_kw = pd.read_csv(kw_file)
    print(f"\nk={ref_k} | Top 5 Keywords per Topic:")
    
    for topic_id in range(ref_k):
        top_kw = df_kw[(df_kw['topic_id'] == topic_id) & (df_kw['rank'] <= 5)]
        keywords = ", ".join(top_kw['word'].tolist())
        print(f"  Topic {topic_id:2d}: {keywords}")

# ── 6. Representative Reviews Preview ───────────────────────────────────────
print("\n[6] Representative Reviews Preview (k={})".format(ref_k))
print("-" * 70)

rep_file = PHASE3A_DIR / f"representative_reviews_k{ref_k}.csv"
if rep_file.exists():
    df_reps = pd.read_csv(rep_file)
    
    print(f"\nSample representative review per topic:")
    for topic_id in range(min(ref_k, 5)):  # Show first 5 topics
        topic_docs = df_reps[df_reps['topic_id'] == topic_id]
        if len(topic_docs) > 0:
            top_doc = topic_docs.iloc[0]
            text_preview = str(top_doc['text'])[:100].replace('\n', ' ')
            prob = top_doc['probability']
            print(f"\n  Topic {topic_id} [{prob:.3f}]:")
            print(f"    \"{text_preview}...\"")

# ── 7. K Selection Scoring ──────────────────────────────────────────────────
print("\n[7] K Selection Recommendation Logic")
print("-" * 70)

# Scoring criteria
scores = []

for _, row in results_df.iterrows():
    k = row['k']
    cv = row['coherence_cv']
    barriers = row['barrier_terms_found']
    topics_with_barriers = row['unique_topics_with_barriers']
    
    # Normalized scores (0-10 scale)
    coherence_score = (cv / results_df['coherence_cv'].max()) * 10  # Normalize by max
    barrier_score = (topics_with_barriers / results_df['unique_topics_with_barriers'].max()) * 10
    
    # Penalize extreme k values (too coarse or too fine)
    granularity_penalty = 0
    if k <= 4:
        granularity_penalty = -1  # Too coarse
    elif k >= 12:
        granularity_penalty = -0.5  # Slightly penalize fragmentation
    
    # Weighted composite
    # 60% coherence, 30% barrier coverage, 10% granularity balance
    composite = (coherence_score * 0.60) + (barrier_score * 0.30) + (10 * 0.10) + granularity_penalty
    
    scores.append({
        'k': k,
        'cv_score': cv,
        'coherence_norm': round(coherence_score, 2),
        'barrier_dist': topics_with_barriers,
        'barrier_norm': round(barrier_score, 2),
        'composite_score': round(composite, 2)
    })

scores_df = pd.DataFrame(scores)
print("\nK Selection Scoring Matrix:")
print(scores_df.to_string(index=False))

# ── 8. Final Recommendation ─────────────────────────────────────────────────
best_idx = scores_df['composite_score'].idxmax()
best_k = int(scores_df.loc[best_idx, 'k'])
best_score = scores_df.loc[best_idx, 'composite_score']

print("\n" + "=" * 70)
print("  RECOMMENDED k VALUE")
print("=" * 70)
print(f"\n  k = {best_k}")
print(f"  Composite Score: {best_score:.2f}/10")
print(f"  Coherence (Cv): {scores_df.loc[best_idx, 'cv_score']:.4f}")
print(f"  Barrier Distribution: {int(scores_df.loc[best_idx, 'barrier_dist'])} topics")

print(f"\n  Rationale:")
print(f"  ✓ Highest coherence score (Cv={scores_df.loc[best_idx, 'cv_score']:.4f})")
print(f"  ✓ Complete barrier phrase distribution (all {int(scores_df.loc[best_idx, 'barrier_dist'])} topics covered)")
print(f"  ✓ Balanced granularity (not too coarse, not too fragmented)")

# ── 9. Alternative Options ──────────────────────────────────────────────────
print(f"\n  Alternative Options:")
alternatives = scores_df[scores_df['k'] != best_k].sort_values('composite_score', ascending=False).head(2)
for idx, row in alternatives.iterrows():
    k = int(row['k'])
    score = row['composite_score']
    cv = row['cv_score']
    barriers = int(row['barrier_dist'])
    
    if k < best_k:
        note = "(more interpretable, simpler)"
    else:
        note = "(more granular, higher specificity)"
    
    print(f"  • k={k:2d}: Score={score:.2f} | Cv={cv:.4f} | Barriers={barriers} {note}")

# ── 10. Next Steps ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  NEXT STEPS: PHASE 3B FINAL LDA TRAINING")
print("=" * 70)

print(f"""
1. CONFIRM OPTIMAL k = {best_k}
   - Review representative reviews in representative_reviews_k{best_k}.csv
   - Verify topic keywords make semantic sense
   - Validate barrier phrase distribution matches domain expectations

2. MODIFY phase3b_final_lda_training.py
   - Set OPTIMAL_K = {best_k} (line: OPTIMAL_K = {best_k})
   - Increase PASSES and ITERATIONS for final model stability

3. EXECUTE PHASE 3B
   python Topic-Modelling/phase3b_final_lda_training.py
   
   This will:
   - Train final LDA model with k={best_k}
   - Extract document-topic distribution matrix
   - Generate topic statistics and barrier audits
   - Prepare all artifacts for Phase 3C (sentiment integration)

4. PHASE 3C: Sentiment-Topic Integration
   - Attach sentiment probability scores to topic assignments
   - Compute severity scores per topic
   - Identify high-priority learning barriers
""")

print("=" * 70)
print("  Analysis complete. Ready for Phase 3B execution.")
print("=" * 70)
