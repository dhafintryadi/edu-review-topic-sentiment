"""
core/topic_engine.py
====================
LDA K=8 topic inference engine.
Logic extracted verbatim from Topic-Modelling/run_phase3c.py.
Locked to: assets/lda_model_final_k8.gensim (OPTIMAL MODEL).
Zero-division guards and df_metrics fix preserved exactly.
All inputs read from assets/ — zero dependency on Topic-Modelling/.
"""
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

LOGGER = logging.getLogger(__name__)

CORE_DIR      = Path(__file__).resolve().parent
REPO_ROOT     = CORE_DIR.parent
ASSETS_DIR    = REPO_ROOT / "assets"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# All model/corpus assets read from assets/
DEFAULT_LDA_MODEL  = ASSETS_DIR / "lda_model_final_k8.gensim"
DEFAULT_DICTIONARY = ASSETS_DIR / "lda_dictionary.gensim"
DEFAULT_TAXONOMY   = ASSETS_DIR / "phase3b_topic_taxonomy.json"
DEFAULT_CORPUS     = ASSETS_DIR / "lda_ready_corpus.csv"

# Output dir defaults to artifacts/ for production modularity
OUT_DIR = ARTIFACTS_DIR


# ── Priority tier classification (exact from run_phase3c.py) ──────────────────
def get_priority_tier(rank: int, total: int) -> str:
    if rank <= 2: return "CRITICAL"
    if rank <= 4: return "HIGH"
    if rank <= 6: return "MEDIUM"
    return "LOW"


# ── Public entry point ─────────────────────────────────────────────────────────
def run_topic_severity(
    lda_model_path: Path | None = None,
    dictionary_path: Path | None = None,
    taxonomy_path: Path | None = None,
    corpus_path: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run LDA topic inference + severity scoring.
    Writes phase3c_outputs/ exactly as the legacy script does.

    Returns
    -------
    df_metrics : DataFrame with per-barrier severity scores (used by severity_analyzer).
    """
    lda_model_path  = lda_model_path  or DEFAULT_LDA_MODEL
    dictionary_path = dictionary_path or DEFAULT_DICTIONARY
    taxonomy_path   = taxonomy_path   or DEFAULT_TAXONOMY
    corpus_path     = corpus_path     or DEFAULT_CORPUS
    output_dir      = output_dir      or OUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading taxonomy from %s", taxonomy_path)
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy_data = json.load(f)

    # Flatten taxonomy: topic_id → category / label
    topic_to_category: dict[int, str] = {}
    topic_to_label: dict[int, str] = {}
    for category_name, labels_list in taxonomy_data["categories"].items():
        for label in labels_list:
            t_id = int(label.split(":")[0].replace("Topic", "").strip())
            topic_to_category[t_id] = category_name
            topic_to_label[t_id] = label

    LOGGER.info("Loading dictionary from %s", dictionary_path)
    dictionary = Dictionary.load(str(dictionary_path))

    LOGGER.info("Loading OPTIMAL LDA model from %s", lda_model_path)
    lda_model = LdaModel.load(str(lda_model_path))

    LOGGER.info("Loading corpus from %s", corpus_path)
    df = pd.read_csv(corpus_path)
    df = df.drop_duplicates(subset=["reviewId"]).copy()
    LOGGER.info("Unique documents to process: %d", len(df))

    # Ensure numeric columns
    df["probability_negative"]  = pd.to_numeric(df["probability_negative"],  errors="coerce").fillna(0)
    df["calibrated_confidence"] = pd.to_numeric(df["calibrated_confidence"], errors="coerce").fillna(0)

    # ── Topic inference ────────────────────────────────────────────────────────
    LOGGER.info("Inferring topics for unique documents...")
    dominant_topics, topic_probs = [], []
    for text in df["lda_text"].astype(str):
        tokens = text.split()
        bow = dictionary.doc2bow(tokens)
        if bow:
            topics = sorted(lda_model.get_document_topics(bow), key=lambda x: x[1], reverse=True)
            dominant_topics.append(topics[0][0])
            topic_probs.append(topics[0][1])
        else:
            dominant_topics.append(-1)
            topic_probs.append(0.0)

    df["dominant_topic"]    = dominant_topics
    df["topic_probability"] = topic_probs
    df = df[df["dominant_topic"] != -1].copy()
    df["topic_label"]      = df["dominant_topic"].map(topic_to_label)
    df["barrier_category"] = df["dominant_topic"].map(topic_to_category)

    # ── Severity metrics per topic (exact formula from run_phase3c.py) ─────────
    LOGGER.info("Calculating severity metrics...")
    topic_metrics = []
    for t_id in sorted(df["dominant_topic"].unique()):
        t_df  = df[df["dominant_topic"] == t_id]
        freq  = len(t_df)
        neg_df = t_df[t_df["predicted_label_name"] == "negative"]

        # Zero-division guard (established in migration tasks)
        negative_ratio = len(neg_df) / freq if freq > 0 else 0
        mean_neg_conf  = neg_df["calibrated_confidence"].mean() if len(neg_df) > 0 else 0.0
        high_conf_neg  = len(neg_df[neg_df["calibrated_confidence"] >= 0.8])
        high_conf_ratio = high_conf_neg / freq if freq > 0 else 0

        # Severity formula: freq × neg_ratio × mean_neg_conf × (1 + high_conf_ratio)
        severity_score = freq * negative_ratio * mean_neg_conf * (1 + high_conf_ratio)

        topic_metrics.append({
            "topic_id":                 t_id,
            "topic_label":              topic_to_label.get(t_id, f"Topic {t_id}"),
            "barrier_category":         topic_to_category.get(t_id, "Unknown"),
            "frequency":                freq,
            "negative_ratio":           round(negative_ratio, 4),
            "mean_negative_confidence": round(mean_neg_conf, 4),
            "high_confidence_negatives": high_conf_neg,
            "severity_score":           round(severity_score, 2),
        })

    # df_metrics — named exactly as fixed in migration task (NameError fix)
    df_metrics = pd.DataFrame(topic_metrics)
    df_metrics = df_metrics.sort_values(by="severity_score", ascending=False)
    df_metrics.to_csv(output_dir / "topic_sentiment_matrix.csv", index=False)

    # ── Aggregate to barrier categories (TB-1 → TB-7) ─────────────────────────
    category_metrics = df_metrics.groupby("barrier_category").agg(
        total_frequency=("frequency", "sum"),
        avg_negative_ratio=("negative_ratio", "mean"),
        total_severity=("severity_score", "sum"),
        high_conf_negatives=("high_confidence_negatives", "sum"),
    ).reset_index()
    category_metrics = category_metrics.sort_values(by="total_severity", ascending=False)
    category_metrics["rank"] = range(1, len(category_metrics) + 1)
    category_metrics["priority_tier"] = category_metrics["rank"].apply(
        lambda r: get_priority_tier(r, len(category_metrics))
    )

    barrier_ranking = {
        "model": "LDA_k8",
        "severity_formula": "frequency * negative_ratio * mean_neg_confidence * (1 + high_conf_neg_ratio)",
        "rankings": category_metrics.to_dict(orient="records"),
    }
    with open(output_dir / "barrier_severity_ranking.json", "w", encoding="utf-8") as f:
        json.dump(barrier_ranking, f, indent=4)

    # ── Critical barriers CSV ──────────────────────────────────────────────────
    top_cats = category_metrics[
        category_metrics["priority_tier"].isin(["CRITICAL", "HIGH"])
    ]["barrier_category"].tolist()
    critical_barriers = df[
        (df["predicted_label_name"] == "negative") & (df["calibrated_confidence"] >= 0.90)
    ].copy()
    critical_barriers = critical_barriers[critical_barriers["barrier_category"].isin(top_cats)]
    critical_evidence = (
        critical_barriers
        .sort_values(by="calibrated_confidence", ascending=False)
        .groupby("topic_label").head(10)
    )
    critical_evidence[
        ["reviewId", "content_raw", "topic_label", "barrier_category",
         "calibrated_confidence", "probability_negative"]
    ].to_csv(output_dir / "critical_learning_barriers.csv", index=False)

    # ── Adaptive learning mapping ──────────────────────────────────────────────
    intervention_logic = {
        "TB-1 Cognitive Difficulty":          "Trigger dynamic scaffolding, simplify vocabulary, offer alternative explanation formats (e.g., visual vs text).",
        "TB-2 Engagement & Motivation Problem": "Incorporate micro-rewards, progress visualization, and shorter, chunked learning milestones.",
        "TB-3 Lack of Learning Support":      "Activate AI Mentor (Qwen/TinyLlama) for real-time Q&A, suggest relevant question banks.",
        "TB-4 System Usability Issues":       "Fallback to offline mode, optimize asset delivery, gracefully degrade UX during high latency.",
        "TB-5 Access & Resource Limitation":  "Enable low-bandwidth video modes, background downloading, and robust offline caching.",
        "TB-6 Cost / Affordability Barrier":  "Highlight freemium pathways, unlock targeted micro-scholarships based on effort.",
        "TB-7 Content Quality Mismatch":      "Route user to prerequisite diagnostic test, realign curriculum path to current ZPD.",
    }
    adaptive_mapping = {
        "framework": "Sekolah Rakyat Adaptive Learning",
        "barrier_interventions": {
            row["barrier_category"]: {
                "severity_rank":      row["rank"],
                "priority":           row["priority_tier"],
                "design_implication": intervention_logic.get(row["barrier_category"], "Generic intervention required."),
            }
            for _, row in category_metrics.iterrows()
        },
    }
    with open(output_dir / "adaptive_learning_mapping.json", "w", encoding="utf-8") as f:
        json.dump(adaptive_mapping, f, indent=4)

    # ── Severity interpretation report ────────────────────────────────────────
    interpretation_report = {
        "execution_summary": "Phase 3C Severity Scoring successfully integrated sentiment analysis with LDA taxonomy.",
        "methodology": {
            "formula": "severity_score = freq * negative_ratio * mean_neg_conf * (1 + high_conf_ratio)",
            "adjustment": "Neutral sentiments were not discarded but weighted zero for severity calculation. High confidence negatives (>0.8) applied a multiplier effect to penalize critical failures.",
        },
        "key_findings": {
            "most_severe_barrier":             category_metrics.iloc[0]["barrier_category"],
            "least_severe_barrier":            category_metrics.iloc[-1]["barrier_category"],
            "total_critical_barriers_detected": int(
                category_metrics[category_metrics["priority_tier"] == "CRITICAL"]["total_severity"].sum()
            ),
        },
    }
    with open(output_dir / "severity_interpretation_report.json", "w", encoding="utf-8") as f:
        json.dump(interpretation_report, f, indent=4)

    LOGGER.info("Topic engine complete. Outputs in: %s", output_dir)
    return df_metrics
