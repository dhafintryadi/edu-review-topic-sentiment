"""
PHASE_4: Severity Mapping — Correlate LDA Topics with Sentiment
==================================================================
Input  : phase3b_outputs/doc_topic_matrix_k{k}.pkl
         Datasets/main_review_preprocessed/sentiment_preprocessed.csv
         phase3b_outputs/topic_keywords_k{k}.csv
Output : phase4_outputs/ (barrier severity ranking, priority matrix)

WORKFLOW:
1. Load document-topic distributions from Phase 3B
2. Load sentiment probability scores (higher = more negative)
3. Compute topic-severity metric for each topic
4. Map topics to barrier phrases and categories
5. Generate priority ranking for remediation efforts
"""

import json
import warnings
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
PHASE3B_DIR= Path(__file__).parent / "phase3b_outputs"
DATASET_DIR= ROOT / "Datasets"
OUT_DIR    = Path(__file__).parent / "phase4_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Barrier categories for semantic mapping
BARRIER_CATEGORIES = {
    "Technical Issues": [
        "sering_error", "stuck_loading", "gabisa_buka", "force_close",
        "sering_lag", "bug", "crash", "login"
    ],
    "Pricing/Paywall": [
        "bayar", "bayar_mahal", "premium"
    ],
    "Content Quality": [
        "kurang_lengkap", "mudah_paham", "latih_soal", "bank_soal"
    ],
    "General Issues": [
        "sering_keluar", "aplikasi", "tidak", "ruang_guru"
    ]
}

print("=" * 70)
print("  PHASE 4 — Severity Mapping & Barrier Prioritization")
print("=" * 70)

if __name__ == "__main__":
    # ── 1. Load Phase 3B Artifacts ────────────────────────────────────────────────
    print("\n[1] Loading Phase 3B Artifacts ...")
    
    # Determine k from available files
    phase3b_files = list(PHASE3B_DIR.glob("lda_model_final_k*.gensim"))
    if not phase3b_files:
        raise FileNotFoundError(f"No final LDA model found in {PHASE3B_DIR}")
    
    # Extract k from filename
    model_file = phase3b_files[0]
    optimal_k = int(model_file.name.split("k")[-1].split(".")[0])
    print(f"  Detected optimal k = {optimal_k}")
    
    # Load document-topic matrix
    with open(PHASE3B_DIR / f"doc_topic_matrix_k{optimal_k}.pkl", "rb") as f:
        doc_topic_matrix = pickle.load(f)
    print(f"  Document-Topic matrix shape: {doc_topic_matrix.shape}")
    
    # Load topic keywords
    df_keywords = pd.read_csv(PHASE3B_DIR / f"topic_keywords_k{optimal_k}.csv")
    print(f"  Loaded {len(df_keywords)} keyword entries")
    
    # Load barrier audit
    barrier_audit_file = PHASE3B_DIR / f"barrier_phrase_audit_k{optimal_k}.csv"
    if barrier_audit_file.exists():
        df_barriers = pd.read_csv(barrier_audit_file)
        print(f"  Loaded {len(df_barriers)} barrier phrases")
    else:
        print("  [WARN] No barrier audit found")
        df_barriers = pd.DataFrame()

    # ── 2. Load Sentiment Data ────────────────────────────────────────────────────
    print("\n[2] Loading Sentiment Probability Scores ...")
    
    sentiment_file = DATASET_DIR / "main_review_preprocessed" / "sentiment_preprocessed.csv"
    if not sentiment_file.exists():
        raise FileNotFoundError(f"Sentiment file not found: {sentiment_file}")
    
    df_sentiment = pd.read_csv(sentiment_file, low_memory=False)
    
    # Identify sentiment probability column
    # Expected: 'negative_prob', 'sentiment_probability', 'prob_negative', etc.
    prob_cols = [c for c in df_sentiment.columns if 'prob' in c.lower()]
    
    if not prob_cols:
        print("  [ERROR] No probability column found. Available columns:")
        print(f"  {df_sentiment.columns.tolist()}")
        # Fallback: create synthetic scores from existing data
        print("  Using alternative approach...")
        if 'content_sentiment' in df_sentiment.columns:
            df_sentiment['negative_prob'] = (df_sentiment['content_sentiment'] == 'negative').astype(float)
        else:
            raise ValueError("Cannot determine sentiment column")
    else:
        prob_col = prob_cols[0]
        print(f"  Using probability column: {prob_col}")
        df_sentiment['negative_prob'] = df_sentiment[prob_col]
    
    print(f"  Sentiment data shape: {df_sentiment.shape}")
    print(f"  Negative probability range: [{df_sentiment['negative_prob'].min():.3f}, {df_sentiment['negative_prob'].max():.3f}]")
    
    sentiment_scores = df_sentiment['negative_prob'].values
    
    # Align with document-topic matrix
    if len(sentiment_scores) > len(doc_topic_matrix):
        sentiment_scores = sentiment_scores[:len(doc_topic_matrix)]
    elif len(sentiment_scores) < len(doc_topic_matrix):
        # Pad with mean score
        mean_score = sentiment_scores.mean()
        sentiment_scores = np.concatenate([
            sentiment_scores,
            np.full(len(doc_topic_matrix) - len(sentiment_scores), mean_score)
        ])
    
    print(f"  Aligned sentiment scores shape: {sentiment_scores.shape}")

    # ── 3. Compute Topic-Severity Metrics ─────────────────────────────────────────
    print("\n[3] Computing Topic-Severity Metrics ...")
    
    topic_metrics = []
    
    for topic_id in range(optimal_k):
        # Get documents assigned to this topic (with prob > 0)
        topic_docs = doc_topic_matrix[:, topic_id] > 0
        num_docs = topic_docs.sum()
        
        if num_docs == 0:
            continue
        
        # Get sentiment scores for docs assigned to this topic
        topic_sentiments = sentiment_scores[topic_docs]
        
        # Compute metrics
        avg_sentiment = topic_sentiments.mean()  # Higher = more negative
        neg_doc_count = (topic_sentiments > 0.5).sum()  # Docs with negative sentiment
        pct_negative = neg_doc_count / num_docs * 100
        
        # Severity = how often docs are negative × how strong that negativity is
        severity_score = avg_sentiment * (pct_negative / 100)
        
        # Get top keywords for this topic
        topic_kw = df_keywords[df_keywords['topic_id'] == topic_id].head(5)
        top_keywords = ", ".join(topic_kw['word'].tolist())
        
        topic_metrics.append({
            "topic_id": topic_id,
            "num_documents": num_docs,
            "pct_corpus": (num_docs / len(doc_topic_matrix)) * 100,
            "avg_sentiment_score": float(avg_sentiment),
            "pct_negative_docs": float(pct_negative),
            "severity_score": float(severity_score),
            "top_keywords": top_keywords
        })
    
    df_metrics = pd.DataFrame(topic_metrics)
    df_metrics = df_metrics.sort_values("severity_score", ascending=False)
    df_metrics.to_csv(OUT_DIR / f"topic_severity_ranking_k{optimal_k}.csv", index=False)
    
    print(f"\n  Top 5 Most Severe Topics:")
    for idx, row in df_metrics.head(5).iterrows():
        print(f"  Topic {int(row['topic_id']):2d}: Severity={row['severity_score']:.4f} | "
              f"Docs={int(row['num_documents']):6d} | "
              f"Negative={row['pct_negative_docs']:.1f}% | "
              f"{row['top_keywords']}")

    # ── 4. Map Topics to Barrier Categories ───────────────────────────────────────
    print("\n[4] Mapping Topics to Barrier Categories ...")
    
    topic_barrier_mapping = []
    
    for topic_id in range(optimal_k):
        topic_kw = df_keywords[df_keywords['topic_id'] == topic_id]
        
        # Find barrier terms in this topic
        if not df_barriers.empty:
            topic_barriers = df_barriers[df_barriers['dominant_topic_id'] == topic_id]
        else:
            topic_barriers = pd.DataFrame()
        
        barrier_terms = topic_barriers['barrier_term'].tolist() if len(topic_barriers) > 0 else []
        
        # Categorize barriers
        category = "Unclassified"
        for cat, terms in BARRIER_CATEGORIES.items():
            if any(term in barrier_terms for term in terms):
                category = cat
                break
        
        # Get severity from metrics
        severity = df_metrics[df_metrics['topic_id'] == topic_id]['severity_score'].values
        severity = severity[0] if len(severity) > 0 else 0
        
        topic_barrier_mapping.append({
            "topic_id": topic_id,
            "category": category,
            "barrier_terms": "|".join(barrier_terms),
            "num_barrier_terms": len(barrier_terms),
            "severity_score": float(severity)
        })
    
    df_mapping = pd.DataFrame(topic_barrier_mapping)
    df_mapping = df_mapping.sort_values("severity_score", ascending=False)
    df_mapping.to_csv(OUT_DIR / f"topic_barrier_mapping_k{optimal_k}.csv", index=False)
    
    print(f"  Topic-Barrier Mapping:")
    print(df_mapping[["topic_id", "category", "num_barrier_terms", "severity_score"]].to_string(index=False))

    # ── 5. Generate Barrier Priority Matrix ───────────────────────────────────────
    print("\n[5] Generating Barrier Priority Matrix ...")
    
    barrier_priority = []
    
    for topic_id in range(optimal_k):
        if not df_barriers.empty:
            topic_barriers = df_barriers[df_barriers['dominant_topic_id'] == topic_id]
            
            for _, barrier_row in topic_barriers.iterrows():
                term = barrier_row['barrier_term']
                
                # Find category
                category = "Unclassified"
                for cat, terms in BARRIER_CATEGORIES.items():
                    if term in terms:
                        category = cat
                        break
                
                # Get topic severity
                topic_severity = df_metrics[df_metrics['topic_id'] == topic_id]['severity_score'].values
                topic_severity = topic_severity[0] if len(topic_severity) > 0 else 0
                
                # Get barrier importance (probability in topic)
                barrier_importance = barrier_row['topic_probability']
                
                # Priority = topic_severity × barrier_importance
                priority = topic_severity * barrier_importance
                
                barrier_priority.append({
                    "barrier_term": term,
                    "category": category,
                    "topic_id": topic_id,
                    "topic_severity": float(topic_severity),
                    "barrier_importance": float(barrier_importance),
                    "priority_score": float(priority)
                })
    
    df_barrier_priority = pd.DataFrame(barrier_priority)
    if len(df_barrier_priority) > 0:
        df_barrier_priority = df_barrier_priority.sort_values("priority_score", ascending=False)
        df_barrier_priority.to_csv(OUT_DIR / f"barrier_priority_matrix_k{optimal_k}.csv", index=False)
        
        print(f"\n  Top 10 Highest Priority Barriers:")
        for idx, row in df_barrier_priority.head(10).iterrows():
            print(f"  {row['barrier_term']:15s} | Category: {row['category']:20s} | Priority: {row['priority_score']:.4f}")

    # ── 6. Generate Topic-Sentiment Correlation Report ──────────────────────────
    print("\n[6] Generating Topic-Sentiment Correlation Report ...")
    
    correlation_report = {
        "phase": "PHASE_4_SEVERITY_MAPPING",
        "optimal_k": int(optimal_k),
        "correlation_method": "topic_document_correlation",
        "metrics": {
            "total_documents": len(doc_topic_matrix),
            "total_topics": optimal_k,
            "avg_documents_per_topic": float(df_metrics['num_documents'].mean()),
            "avg_sentiment_score": float(sentiment_scores.mean()),
            "corpus_negative_pct": float((sentiment_scores > 0.5).sum() / len(sentiment_scores) * 100)
        },
        "top_severe_topics": df_metrics.head(5)[["topic_id", "severity_score", "pct_negative_docs"]].to_dict('records'),
        "outputs": {
            "severity_ranking": f"topic_severity_ranking_k{optimal_k}.csv",
            "barrier_mapping": f"topic_barrier_mapping_k{optimal_k}.csv",
            "barrier_priority": f"barrier_priority_matrix_k{optimal_k}.csv",
            "visualizations": f"severity_*.png"
        }
    }
    
    with open(OUT_DIR / "phase4_correlation_report.json", "w", encoding="utf-8") as f:
        json.dump(correlation_report, f, indent=2)
    print(f"  ✓ Correlation report saved")

    # ── 7. Generate Visualizations ───────────────────────────────────────────────
    print("\n[7] Generating Visualizations ...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Topic Severity Scores
    ax = axes[0, 0]
    sorted_metrics = df_metrics.sort_values("severity_score", ascending=True)
    colors = plt.cm.Reds(sorted_metrics['severity_score'] / sorted_metrics['severity_score'].max())
    ax.barh(sorted_metrics['topic_id'].astype(str), sorted_metrics['severity_score'], color=colors)
    ax.set_xlabel("Severity Score")
    ax.set_ylabel("Topic ID")
    ax.set_title("Topic Severity Ranking")
    ax.grid(axis="x", alpha=0.3)
    
    # Plot 2: Sentiment Score Distribution per Topic
    ax = axes[0, 1]
    topic_ids = []
    topic_sentiments = []
    for tid in range(optimal_k):
        docs_in_topic = doc_topic_matrix[:, tid] > 0
        if docs_in_topic.sum() > 0:
            topic_ids.append(tid)
            topic_sentiments.append(sentiment_scores[docs_in_topic].mean())
    
    ax.scatter(topic_ids, topic_sentiments, s=100, alpha=0.6, color="#e74c3c")
    ax.set_xlabel("Topic ID")
    ax.set_ylabel("Mean Sentiment Score")
    ax.set_title("Average Negativity by Topic")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Category Distribution
    ax = axes[1, 0]
    category_counts = df_mapping['category'].value_counts()
    category_counts.plot(kind='bar', ax=ax, color="#3498db")
    ax.set_ylabel("Number of Topics")
    ax.set_title("Topics by Barrier Category")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis="y", alpha=0.3)
    
    # Plot 4: Barrier Term Frequency
    ax = axes[1, 1]
    if len(df_barrier_priority) > 0:
        top_barriers = df_barrier_priority.head(10)
        ax.barh(top_barriers['barrier_term'], top_barriers['priority_score'], color="#2ecc71")
        ax.set_xlabel("Priority Score")
        ax.set_title("Top 10 Priority Barriers")
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No barriers detected", ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"severity_analysis_k{optimal_k}.png", dpi=150)
    plt.close()
    print(f"  ✓ Visualizations saved")

    # ── 8. Generate Executive Summary ─────────────────────────────────────────────
    print("\n[8] Generating Executive Summary ...")
    
    top_3_barriers = df_barrier_priority.head(3) if len(df_barrier_priority) > 0 else pd.DataFrame()
    
    executive_summary = f"""
# PHASE 4: SEVERITY MAPPING REPORT
## Learning Barriers Priority Analysis

**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Optimal k Value:** {optimal_k} topics  
**Total Documents:** {len(doc_topic_matrix):,}  
**Barrier Terms Audited:** {len(df_barriers) if not df_barriers.empty else 0}  

## Key Findings

### 1. Most Severe Learning Barriers
"""
    
    if len(top_3_barriers) > 0:
        for idx, (_, row) in enumerate(top_3_barriers.iterrows(), 1):
            executive_summary += f"""
#### {idx}. {row['barrier_term'].upper()}
- **Category:** {row['category']}
- **Priority Score:** {row['priority_score']:.4f}
- **Topic ID:** {int(row['topic_id'])}
- **Topic Severity:** {row['topic_severity']:.4f}
- **Barrier Importance:** {row['barrier_importance']:.4f}
"""
    
    executive_summary += f"""

### 2. Topic Distribution by Category
"""
    
    for category in BARRIER_CATEGORIES.keys():
        cat_topics = df_mapping[df_mapping['category'] == category]
        if len(cat_topics) > 0:
            executive_summary += f"\n- **{category}:** {len(cat_topics)} topics\n"
    
    executive_summary += f"""

### 3. Corpus Sentiment Metrics
- **Average Negative Sentiment:** {sentiment_scores.mean():.1%}
- **Highly Negative Documents (>0.5 prob):** {(sentiment_scores > 0.5).sum():,} documents
- **Percentage of Corpus:** {(sentiment_scores > 0.5).sum() / len(sentiment_scores) * 100:.1f}%

### 4. Recommendations for Remediation

Based on the priority analysis, we recommend focusing on:

1. **Immediate Focus (Priority >0.0200)**
   - Investigate and address the highest-priority barrier terms
   - Allocate resources to user support for these specific issues

2. **Medium-term Focus (Priority 0.0100-0.0200)**
   - Document and track frequency of secondary barriers
   - Develop preventive measures

3. **Long-term Strategy**
   - Monitor sentiment trends for all topics
   - Update LDA model periodically as barriers evolve

## Artifacts Generated
- `topic_severity_ranking_k{optimal_k}.csv` - Complete severity rankings
- `topic_barrier_mapping_k{optimal_k}.csv` - Topic-to-barrier mappings
- `barrier_priority_matrix_k{optimal_k}.csv` - Priority-weighted barrier list
- `severity_analysis_k{optimal_k}.png` - Visual summary
- `phase4_correlation_report.json` - Detailed metrics

---
*End of Phase 4 Report*
"""
    
    with open(OUT_DIR / "phase4_executive_summary.md", "w", encoding="utf-8") as f:
        f.write(executive_summary)
    print(f"  ✓ Executive summary saved")

    print("\n" + "=" * 70)
    print("  PHASE 4 COMPLETE")
    print(f"  Outputs -> {OUT_DIR}")
    print("=" * 70)
    print("""
## FINAL DELIVERABLES ##

1. Barrier Severity Ranking (CSV)
   - All topics ranked by severity
   - Sentiment correlation metrics
   - Document counts per topic

2. Barrier Priority Matrix (CSV)
   - Individual barrier terms with priority scores
   - Category classifications
   - Topic associations

3. Topic-Barrier Mapping (CSV)
   - Semantic categorization of topics
   - Barrier term associations

4. Executive Summary (MD)
   - Key findings and recommendations
   - Actionable remediation priorities

5. Visualizations (PNG)
   - Severity distributions
   - Category analysis
   - Priority rankings

---
PROJECT PHASES COMPLETED:
✓ PHASE_1: Corpus Audit & Barrier Detection
✓ PHASE_2: Dictionary & Corpus Engineering
✓ PHASE_3A: Exploratory LDA Training
✓ PHASE_3B: Final LDA Model Training
✓ PHASE_4: Severity Mapping & Prioritization

READY FOR: Domain expert review & stakeholder communication
""")
