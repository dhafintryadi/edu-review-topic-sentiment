"""
PHASE_3B: Final LDA Topic Extraction — Sekolah Rakyat
==========================================================
Input  : phase2_outputs/lda_dictionary.gensim, lda_corpus.mm
         phase3a_outputs/phase3a_summary.json (to determine optimal k)
Output : phase3b_outputs/ (final LDA model, document-topic distributions)

WORKFLOW:
1. Load Phase 3A summary to retrieve optimal k value
2. Train final LDA model using selected k
3. Extract document-topic distributions (dense matrix)
4. Save all artifacts for Phase 4 (Severity Mapping)
"""

import json
import warnings
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensim import corpora, models
from scipy import sparse

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
PHASE1_DIR = Path(__file__).parent / "phase1_outputs"
PHASE2_DIR = Path(__file__).parent / "phase2_outputs"
PHASE3A_DIR= Path(__file__).parent / "phase3a_outputs"
OUT_DIR    = Path(__file__).parent / "phase3b_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 2 Validated barrier phrases to audit final model
PROTECTED_TERMS = [
    "sering_error", "stuck_loading", "gabisa_buka", "force_close",
    "sering_lag", "ruang_guru", "mudah_paham", "bug", "crash",
    "login", "bayar", "aplikasi", "tidak", "latih_soal", "bank_soal",
    "kurang_lengkap", "bayar_mahal", "sering_keluar"
]

print("=" * 70)
print("  PHASE 3B — Final LDA Topic Extraction")
print("=" * 70)

if __name__ == "__main__":
    # ── 1. Determine Optimal k from Phase 3A ──────────────────────────────────────
    print("\n[1] Loading Phase 3A Summary to determine optimal k ...")
    
    with open(PHASE3A_DIR / "phase3a_summary.json", "r", encoding="utf-8") as f:
        phase3a_summary = json.load(f)
    
    # Select k based on composite score (user should verify this)
    results = pd.DataFrame(phase3a_summary["results"])
    
    print("\nPhase 3A Results:")
    print(results[["k", "coherence_cv", "barrier_terms_found"]].to_string(index=False))
    
    # Default: Use k with highest coherence; user can override
    optimal_k = results.loc[results["coherence_cv"].idxmax(), "k"]
    print(f"\n[AUTO-SELECTED] Optimal k = {int(optimal_k)} (highest Cv coherence)")
    print("  NOTE: User should verify this selection based on interpretability")
    print("  To override, set OPTIMAL_K = <desired_k> below")
    
    # Allow manual override
    OPTIMAL_K = 8  # <-- Phase 3A Analysis: k=8 recommended (Cv=0.4288, composite=10.0)
    print(f"\n[USING] k = {OPTIMAL_K} (Phase 3A Recommendation)")

    # ── 2. Load Data ──────────────────────────────────────────────────────────────
    print(f"\n[2] Loading Dictionary and Corpus ...")
    
    dictionary = corpora.Dictionary.load(str(PHASE2_DIR / "lda_dictionary.gensim"))
    corpus     = corpora.MmCorpus(str(PHASE2_DIR / "lda_corpus.mm"))
    print(f"  Vocabulary size: {len(dictionary):,}")
    print(f"  Corpus size    : {len(corpus):,}")

    # Load original text for document mapping
    try:
        df = pd.read_csv(PHASE1_DIR / "lda_ready_corpus.csv", low_memory=False)
        if 'content_raw' not in df.columns:
            df['content_raw'] = df.get('content_topic', '')
        
        reviews_text = df["content_raw"].fillna("").astype(str).tolist()
        doc_ids = df.get("reviewId", [None] * len(df)).tolist()
        
        if len(reviews_text) != len(corpus):
            print(f"  [WARN] DataFrame length ({len(reviews_text)}) != Corpus length ({len(corpus)})")
            reviews_text = reviews_text[:len(corpus)] + [""] * max(0, len(corpus) - len(reviews_text))
            doc_ids = doc_ids[:len(corpus)] + [None] * max(0, len(corpus) - len(doc_ids))
    except Exception as e:
        print(f"  [ERROR] Failed to load original corpus text: {e}")
        reviews_text = [""] * len(corpus)
        doc_ids = [None] * len(corpus)

    # Load token lists for later use (document-token matrix)
    with open(PHASE1_DIR / "lda_token_lists.json", "r", encoding="utf-8") as f:
        token_lists = json.load(f)

    # ── 3. Train Final LDA Model ──────────────────────────────────────────────────
    print(f"\n[3] Training Final LDA Model (k={OPTIMAL_K}) ...")
    
    PASSES      = 30  # Increased for final model
    ITERATIONS  = 150  # Increased for final model
    ALPHA       = "asymmetric"
    ETA         = "auto"
    RANDOM_STATE= 42

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=OPTIMAL_K,
        random_state=RANDOM_STATE,
        passes=PASSES,
        iterations=ITERATIONS,
        alpha=ALPHA,
        eta=ETA,
        per_word_topics=True  # Enable per-word topic tracking
    )
    
    print(f"  ✓ Final LDA Model trained successfully")
    
    # Save model
    model_path = OUT_DIR / f"lda_model_final_k{OPTIMAL_K}.gensim"
    lda_model.save(str(model_path))
    print(f"  ✓ Model saved: {model_path}")

    # ── 4. Extract Document-Topic Distribution Matrix ──────────────────────────────
    print(f"\n[4] Extracting Document-Topic Distribution Matrix ...")
    
    doc_topic_list = []
    doc_indices = []
    
    for doc_idx, bow in enumerate(corpus):
        if not bow:
            doc_topic_list.append([0.0] * OPTIMAL_K)
            doc_indices.append(doc_idx)
            continue
        
        topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        topic_dist = [0.0] * OPTIMAL_K
        for topic_id, prob in topics:
            topic_dist[topic_id] = prob
        
        doc_topic_list.append(topic_dist)
        doc_indices.append(doc_idx)
    
    doc_topic_matrix = np.array(doc_topic_list)
    print(f"  Document-Topic Matrix shape: {doc_topic_matrix.shape}")
    print(f"  Sparsity: {(doc_topic_matrix == 0).sum() / doc_topic_matrix.size:.2%}")
    
    # Save as dense pickle (easier to load later)
    with open(OUT_DIR / f"doc_topic_matrix_k{OPTIMAL_K}.pkl", "wb") as f:
        pickle.dump(doc_topic_matrix, f)
    print(f"  ✓ Document-Topic matrix saved")

    # ── 5. Extract Top Keywords per Topic ──────────────────────────────────────────
    print(f"\n[5] Extracting Top Keywords ...")
    
    topic_keywords = []
    for topic_id in range(OPTIMAL_K):
        top_words = lda_model.show_topic(topic_id, topn=40)
        for rank, (word, weight) in enumerate(top_words, 1):
            topic_keywords.append({
                "topic_id": topic_id,
                "rank": rank,
                "word": word,
                "weight": float(weight)
            })
    
    df_kw = pd.DataFrame(topic_keywords)
    df_kw.to_csv(OUT_DIR / f"topic_keywords_k{OPTIMAL_K}.csv", index=False)
    print(f"  ✓ {len(df_kw)} keyword entries saved")

    # ── 6. Extract Representative Documents per Topic ──────────────────────────────
    print(f"\n[6] Extracting Representative Documents ...")
    
    reps_data = []
    for doc_idx, topic_dist in enumerate(doc_topic_matrix):
        if sum(topic_dist) == 0:
            continue
        
        dom_topic = np.argmax(topic_dist)
        dom_prob = topic_dist[dom_topic]
        
        reps_data.append({
            "doc_index": doc_idx,
            "doc_id": doc_ids[doc_idx] if doc_idx < len(doc_ids) else None,
            "topic_id": dom_topic,
            "probability": dom_prob,
            "text": reviews_text[doc_idx] if doc_idx < len(reviews_text) else ""
        })
    
    df_reps = pd.DataFrame(reps_data)
    
    # Get top 10 per topic
    top_reps = []
    for t_id in range(OPTIMAL_K):
        topic_docs = df_reps[df_reps["topic_id"] == t_id]
        topic_docs = topic_docs.sort_values("probability", ascending=False).head(10)
        top_reps.append(topic_docs)
    
    df_top_reps = pd.concat(top_reps, ignore_index=True)
    df_top_reps.to_csv(OUT_DIR / f"representative_documents_k{OPTIMAL_K}.csv", index=False)
    print(f"  ✓ {len(df_top_reps)} representative documents saved")

    # ── 7. Barrier Phrase Audit ───────────────────────────────────────────────────
    print(f"\n[7] Auditing Barrier Phrase Distribution ...")
    
    audit_data = []
    for term in PROTECTED_TERMS:
        if term not in dictionary.token2id:
            continue
        
        term_id = dictionary.token2id[term]
        term_topics = lda_model.get_term_topics(term_id, minimum_probability=0.0)
        
        if term_topics:
            term_topics.sort(key=lambda x: x[1], reverse=True)
            top_topic_id, top_prob = term_topics[0]
            
            # Rank in topic
            topic_terms = lda_model.show_topic(top_topic_id, topn=100)
            rank = -1
            for r, (w, _) in enumerate(topic_terms, 1):
                if w == term:
                    rank = r
                    break
            
            audit_data.append({
                "barrier_term": term,
                "dominant_topic_id": top_topic_id,
                "topic_probability": float(top_prob),
                "rank_in_topic": rank if rank > 0 else ">100"
            })
    
    df_audit = pd.DataFrame(audit_data)
    if not df_audit.empty:
        df_audit = df_audit.sort_values(["dominant_topic_id", "topic_probability"], ascending=[True, False])
        df_audit.to_csv(OUT_DIR / f"barrier_phrase_audit_k{OPTIMAL_K}.csv", index=False)
        print(f"  ✓ {len(df_audit)} barrier phrases audited")

    # ── 8. Topic-Document Distribution Statistics ──────────────────────────────────
    print(f"\n[8] Computing Topic Statistics ...")
    
    topic_stats = []
    for t_id in range(OPTIMAL_K):
        topic_col = doc_topic_matrix[:, t_id]
        non_zero = topic_col[topic_col > 0]
        
        topic_stats.append({
            "topic_id": t_id,
            "num_docs_assigned": len(non_zero),
            "pct_docs": len(non_zero) / len(doc_topic_matrix) * 100,
            "mean_prob": float(non_zero.mean()) if len(non_zero) > 0 else 0.0,
            "median_prob": float(np.median(non_zero)) if len(non_zero) > 0 else 0.0,
            "max_prob": float(non_zero.max()) if len(non_zero) > 0 else 0.0
        })
    
    df_stats = pd.DataFrame(topic_stats)
    df_stats.to_csv(OUT_DIR / f"topic_statistics_k{OPTIMAL_K}.csv", index=False)
    print(f"  Topic Distribution Statistics:")
    print(df_stats[["topic_id", "num_docs_assigned", "pct_docs", "mean_prob"]].to_string(index=False))

    # ── 9. Generate Final LDA Summary ─────────────────────────────────────────────
    print(f"\n[9] Generating Phase 3B Summary ...")
    
    final_summary = {
        "phase": "PHASE_3B_FINAL_LDA_EXTRACTION",
        "optimal_k": int(OPTIMAL_K),
        "parameters": {
            "passes": PASSES,
            "iterations": ITERATIONS,
            "alpha": ALPHA,
            "eta": ETA
        },
        "artifacts": {
            "model": f"lda_model_final_k{OPTIMAL_K}.gensim",
            "doc_topic_matrix": f"doc_topic_matrix_k{OPTIMAL_K}.pkl",
            "keywords": f"topic_keywords_k{OPTIMAL_K}.csv",
            "representative_docs": f"representative_documents_k{OPTIMAL_K}.csv",
            "barrier_audit": f"barrier_phrase_audit_k{OPTIMAL_K}.csv",
            "topic_statistics": f"topic_statistics_k{OPTIMAL_K}.csv"
        },
        "corpus_metrics": {
            "vocab_size": len(dictionary),
            "num_documents": len(corpus),
            "avg_doc_topics": float(np.count_nonzero(doc_topic_matrix) / len(corpus)),
            "barrier_terms_audited": len(df_audit) if not df_audit.empty else 0
        }
    }
    
    with open(OUT_DIR / "phase3b_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)
    print(f"  ✓ Phase 3B Summary saved")

    # ── 10. Generate Visualization ───────────────────────────────────────────────
    print(f"\n[10] Generating Topic Distribution Visualization ...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Documents per Topic
    docs_per_topic = df_stats["num_docs_assigned"].values
    ax1.bar(range(OPTIMAL_K), docs_per_topic, color="#3498db")
    ax1.set_xlabel("Topic ID")
    ax1.set_ylabel("Number of Documents")
    ax1.set_title(f"Document Distribution Across Topics (k={OPTIMAL_K})")
    ax1.grid(axis="y", alpha=0.3)
    
    # Plot 2: Mean Topic Probability
    mean_probs = df_stats["mean_prob"].values
    ax2.bar(range(OPTIMAL_K), mean_probs, color="#2ecc71")
    ax2.set_xlabel("Topic ID")
    ax2.set_ylabel("Mean Topic Probability")
    ax2.set_title(f"Average Topic Strength (k={OPTIMAL_K})")
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "topic_distribution_k{}.png".format(OPTIMAL_K), dpi=150)
    plt.close()
    print(f"  ✓ Visualization saved")

    print("\n" + "=" * 70)
    print("  PHASE 3B COMPLETE")
    print(f"  Outputs -> {OUT_DIR}")
    print("=" * 70)
    print("\nNEXT: PHASE_4_SEVERITY_MAPPING")
    print("  - Load doc-topic matrix from phase3b_outputs/")
    print("  - Correlate with sentiment probability scores")
    print("  - Generate barrier severity ranking")
