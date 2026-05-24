"""
PHASE_3A: Exploratory LDA Training — Sekolah Rakyat
==========================================================
Input  : phase2_outputs/lda_dictionary.gensim, lda_corpus.mm
         phase1_outputs/lda_ready_corpus.csv
Output : phase3a_outputs/
"""

import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
PHASE1_DIR = Path(__file__).parent / "phase1_outputs"
PHASE2_DIR = Path(__file__).parent / "phase2_outputs"
OUT_DIR    = Path(__file__).parent / "phase3a_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
K_RANGES    = [4, 6, 8, 10, 12]
PASSES      = 20
ITERATIONS  = 100
ALPHA       = "asymmetric"
ETA         = "auto"
RANDOM_STATE= 42
WORKERS     = 3  # For LdaMulticore

# Phase 2 Validated phrases to track
PROTECTED_TERMS = [
    "sering_error", "stuck_loading", "gabisa_buka", "force_close",
    "sering_lag", "ruang_guru", "mudah_paham", "bug", "crash",
    "login", "bayar", "aplikasi", "tidak", "latih_soal", "bank_soal",
    "kurang_lengkap", "bayar_mahal", "sering_keluar"
]

print("=" * 65)
print("  PHASE 3A — Exploratory LDA Training")
print("=" * 65)

if __name__ == "__main__":
    # ── 1. Load Data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading artifacts and data ...")

    # Load Dictionary and Corpus
    dictionary = corpora.Dictionary.load(str(PHASE2_DIR / "lda_dictionary.gensim"))
    corpus     = corpora.MmCorpus(str(PHASE2_DIR / "lda_corpus.mm"))
    print(f"  Vocabulary size: {len(dictionary):,}")
    print(f"  Corpus size    : {len(corpus):,}")

    # Load original text for representative review matching
    try:
        df = pd.read_csv(PHASE1_DIR / "lda_ready_corpus.csv", low_memory=False)
        # We only need the review text and token counts
        if 'content_raw' not in df.columns:
            # Fallback to content_topic if raw is missing
            df['content_raw'] = df.get('content_topic', '')
            
        reviews_text = df["content_raw"].fillna("").astype(str).tolist()
        # Check length match
        if len(reviews_text) != len(corpus):
            print(f"  [WARN] DataFrame length ({len(reviews_text)}) != Corpus length ({len(corpus)})")
            reviews_text = reviews_text[:len(corpus)] + [""] * max(0, len(corpus) - len(reviews_text))
    except Exception as e:
        print(f"  [ERROR] Failed to load original corpus text: {e}")
        reviews_text = [""] * len(corpus)

    # Load token lists for Coherence Calculation (Cv requires original token streams)
    print("  Loading token lists for Coherence calculation ...")
    with open(PHASE1_DIR / "lda_token_lists.json", "r", encoding="utf-8") as f:
        token_lists = json.load(f)

    # ── 2. Training Loop ─────────────────────────────────────────────────────────
    summary_stats = []

    for k in K_RANGES:
        print(f"\n" + "-" * 50)
        print(f"  Training LdaModel | k = {k}")
        print("-" * 50)
        
        # Train model (Single Core to prevent multiprocessing sandbox blocks)
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=RANDOM_STATE,
            passes=PASSES,
            iterations=ITERATIONS,
            alpha=ALPHA,
            eta=ETA
        )
        
        # Save model
        model_path = OUT_DIR / f"lda_model_k{k}.gensim"
        lda_model.save(str(model_path))
        print(f"  -> Model saved")

        # ── 3. Compute Coherence (Cv) ─────────────────────────────────────────────
        # Not used for optimization, strictly as a secondary diagnostic
        print("  -> Computing Cv Coherence Score ...")
        coherence_model = CoherenceModel(
            model=lda_model, 
            texts=token_lists, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        cv_score = coherence_model.get_coherence()
        print(f"  -> Coherence (Cv) : {cv_score:.4f}")

        # ── 4. Extract Keywords ───────────────────────────────────────────────────
        print("  -> Extracting top 30 keywords per topic ...")
        topic_keywords = []
        for topic_id in range(k):
            top_words = lda_model.show_topic(topic_id, topn=30)
            for rank, (word, weight) in enumerate(top_words, 1):
                topic_keywords.append({
                    "topic_id": topic_id,
                    "rank": rank,
                    "word": word,
                    "weight": float(weight)
                })
        
        df_kw = pd.DataFrame(topic_keywords)
        df_kw.to_csv(OUT_DIR / f"topic_keywords_k{k}.csv", index=False)

        # ── 5. Barrier Cluster Audit ──────────────────────────────────────────────
        print("  -> Auditing Barrier Phrase distribution ...")
        audit_data = []
        for term in PROTECTED_TERMS:
            if term not in dictionary.token2id:
                continue
                
            term_id = dictionary.token2id[term]
            
            # Get topic distribution for this single word
            # get_term_topics returns list of (topic_id, probability) for topics where word prob is > minimum_probability
            term_topics = lda_model.get_term_topics(term_id, minimum_probability=0.0)
            if term_topics:
                # Sort by probability descending
                term_topics.sort(key=lambda x: x[1], reverse=True)
                top_topic_id, top_prob = term_topics[0]
                
                # Find the rank of this word in the top_topic
                topic_terms = lda_model.show_topic(top_topic_id, topn=100) # search deeper
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
            df_audit.to_csv(OUT_DIR / f"barrier_cluster_audit_k{k}.csv", index=False)

        # ── 6. Extract Representative Reviews ─────────────────────────────────────
        print("  -> Extracting representative reviews ...")
        # We will compute dominant topic for a sample of the corpus for speed
        # We sample 20,000 docs to find representatives (or full if less)
        sample_size = min(20000, len(corpus))
        np.random.seed(RANDOM_STATE)
        sample_indices = np.random.choice(len(corpus), sample_size, replace=False)
        
        reps_data = []
        for idx in sample_indices:
            bow = corpus[int(idx)]
            if not bow:
                continue
            topics = lda_model.get_document_topics(bow)
            if not topics:
                continue
                
            # Get dominant topic
            dom_topic, dom_prob = max(topics, key=lambda x: x[1])
            reps_data.append({
                "doc_index": int(idx),
                "topic_id": dom_topic,
                "probability": float(dom_prob),
                "text": reviews_text[int(idx)]
            })
            
        df_reps = pd.DataFrame(reps_data)
        
        # Get top 5 per topic
        top_reps = []
        for t_id in range(k):
            topic_docs = df_reps[df_reps["topic_id"] == t_id]
            topic_docs = topic_docs.sort_values("probability", ascending=False).head(5)
            top_reps.append(topic_docs)
            
        if top_reps:
            df_top_reps = pd.concat(top_reps, ignore_index=True)
            df_top_reps.to_csv(OUT_DIR / f"representative_reviews_k{k}.csv", index=False)
            
        # Append to summary
        summary_stats.append({
            "k": k,
            "coherence_cv": round(cv_score, 4),
            "barrier_terms_found": len(audit_data),
            "unique_topics_with_barriers": int(df_audit["dominant_topic_id"].nunique()) if not df_audit.empty else 0
        })

    # ── 7. Generate Summary JSON and Plot ─────────────────────────────────────────
    print("\n[7] Generating Phase 3A Summary ...")

    summary_dict = {
        "phase": "PHASE_3A_LDA_EXPLORATORY_TRAINING",
        "parameters": {
            "k_ranges": K_RANGES,
            "passes": PASSES,
            "iterations": ITERATIONS,
            "alpha": ALPHA,
            "eta": ETA
        },
        "results": summary_stats
    }

    with open(OUT_DIR / "phase3a_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # Plot Coherence
    ks = [s["k"] for s in summary_stats]
    cvs = [s["coherence_cv"] for s in summary_stats]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, cvs, marker="o", color="#e74c3c", linewidth=2, markersize=8)
    plt.title("Topic Coherence (Cv) vs. Number of Topics (k)\n*For Diagnostic Purposes Only*", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Topics (k)")
    plt.ylabel("Coherence Score (Cv)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "coherence_diagnostic.png", dpi=150)
    plt.close()

    print("=" * 65)
    print("  PHASE 3A COMPLETE")
    print(f"  Outputs -> {OUT_DIR}")
    print("=" * 65)
