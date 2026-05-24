import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

# Define Paths
BASE_DIR = Path("c:/Users/ASUS/Documents/AITF-2026/PKL/Topic-Modelling")
PHASE1_OUT = BASE_DIR / "phase1_outputs"
PHASE2_OUT = BASE_DIR / "phase2_outputs"
PHASE3A_OUT = BASE_DIR / "phase3a_outputs"
PHASE3B_OUT = BASE_DIR / "phase3b_outputs"
OUT_DIR = BASE_DIR / "phase3c_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Starting Phase 3C: Severity Scoring and Sentiment Integration...")

# 1. Load Taxonomy Mapping from Phase 3B
with open(PHASE3B_OUT / "phase3b_topic_taxonomy.json", "r", encoding="utf-8") as f:
    taxonomy_data = json.load(f)

# Flatten taxonomy data to map topic ID -> category and label
topic_to_category = {}
topic_to_label = {}
for category_name, labels_list in taxonomy_data["categories"].items():
    for label in labels_list:
        t_id = int(label.split(":")[0].replace("Topic", "").strip())
        topic_to_category[t_id] = category_name
        topic_to_label[t_id] = label

# 2. Load LDA Model and Dictionary
dictionary = Dictionary.load(str(PHASE2_OUT / "lda_dictionary.gensim"))
lda_model = LdaModel.load(str(PHASE3A_OUT / "lda_model_k8.gensim"))

# 3. Load Corpus and Drop Oversampling Duplicates to retain real-world frequency
print("Loading corpus and dropping oversampling duplicates...")
df = pd.read_csv(PHASE1_OUT / "lda_ready_corpus.csv")
df = df.drop_duplicates(subset=["reviewId"]).copy()
print(f"Unique documents to process: {len(df)}")

# 4. Perform Inference
print("Inferring topics for unique documents...")
dominant_topics = []
topic_probs = []

# Ensure probability_negative is numeric
df['probability_negative'] = pd.to_numeric(df['probability_negative'], errors='coerce').fillna(0)
df['calibrated_confidence'] = pd.to_numeric(df['calibrated_confidence'], errors='coerce').fillna(0)

for idx, text in enumerate(df["lda_text"].astype(str)):
    tokens = text.split()
    bow = dictionary.doc2bow(tokens)
    if bow:
        topics = lda_model.get_document_topics(bow)
        # Sort by probability descending
        topics = sorted(topics, key=lambda x: x[1], reverse=True)
        dominant_topics.append(topics[0][0])
        topic_probs.append(topics[0][1])
    else:
        dominant_topics.append(-1)
        topic_probs.append(0.0)

df["dominant_topic"] = dominant_topics
df["topic_probability"] = topic_probs

# Filter out empty documents
df = df[df["dominant_topic"] != -1].copy()

# Map taxonomy
df["topic_label"] = df["dominant_topic"].map(topic_to_label)
df["barrier_category"] = df["dominant_topic"].map(topic_to_category)

# 5. Calculate Severity Metrics Per Topic
print("Calculating severity metrics...")
topic_metrics = []

for t_id in sorted(df["dominant_topic"].unique()):
    t_df = df[df["dominant_topic"] == t_id]
    freq = len(t_df)
    
    # Weight high-confidence negative reviews higher
    # Base negative ratio: ratio of negative predictions
    neg_df = t_df[t_df["predicted_label_name"] == "negative"]
    negative_ratio = len(neg_df) / freq if freq > 0 else 0
    
    # Confidence weighting: Mean confidence of negative reviews
    if len(neg_df) > 0:
        mean_neg_conf = neg_df["calibrated_confidence"].mean()
    else:
        mean_neg_conf = 0.0
        
    # User's extended interpretation layer: High-confidence negatives carry more weight
    high_conf_neg_count = len(neg_df[neg_df["calibrated_confidence"] >= 0.8])
    high_conf_ratio = high_conf_neg_count / freq if freq > 0 else 0
    
    # Severity Formula
    # severity_score = topic_frequency * negative_sentiment_ratio * mean_calibrated_confidence
    # We add a small bump for high confidence negative density
    severity_score = freq * negative_ratio * mean_neg_conf * (1 + high_conf_ratio)
    
    topic_metrics.append({
        "topic_id": t_id,
        "topic_label": topic_to_label.get(t_id, f"Topic {t_id}"),
        "barrier_category": topic_to_category.get(t_id, "Unknown"),
        "frequency": freq,
        "negative_ratio": round(negative_ratio, 4),
        "mean_negative_confidence": round(mean_neg_conf, 4),
        "high_confidence_negatives": high_conf_neg_count,
        "severity_score": round(severity_score, 2)
    })

df_metrics = pd.DataFrame(topic_metrics)
df_metrics = df_metrics.sort_values(by="severity_score", ascending=False)
df_metrics.to_csv(OUT_DIR / "topic_sentiment_matrix.csv", index=False)

# 6. Aggregate to Barrier Categories (TB-1 to TB-7)
category_metrics = df_metrics.groupby("barrier_category").agg(
    total_frequency=("frequency", "sum"),
    avg_negative_ratio=("negative_ratio", "mean"),
    total_severity=("severity_score", "sum"),
    high_conf_negatives=("high_confidence_negatives", "sum")
).reset_index()

category_metrics = category_metrics.sort_values(by="total_severity", ascending=False)

# Rank and classify Priority Tiers
def get_priority_tier(rank, total):
    if rank <= 2: return "CRITICAL"
    if rank <= 4: return "HIGH"
    if rank <= 6: return "MEDIUM"
    return "LOW"

category_metrics["rank"] = range(1, len(category_metrics) + 1)
category_metrics["priority_tier"] = category_metrics["rank"].apply(lambda r: get_priority_tier(r, len(category_metrics)))

barrier_ranking = {
    "model": "LDA_k8",
    "severity_formula": "frequency * negative_ratio * mean_neg_confidence * (1 + high_conf_neg_ratio)",
    "rankings": category_metrics.to_dict(orient="records")
}

with open(OUT_DIR / "barrier_severity_ranking.json", "w", encoding="utf-8") as f:
    json.dump(barrier_ranking, f, indent=4)

# 7. Extract Critical Learning Barriers (High severity, high confidence negative reviews)
critical_barriers = df[(df["predicted_label_name"] == "negative") & (df["calibrated_confidence"] >= 0.90)].copy()
# Only take samples from the top 3 CRITICAL/HIGH categories
top_categories = category_metrics[category_metrics["priority_tier"].isin(["CRITICAL", "HIGH"])]["barrier_category"].tolist()
critical_barriers = critical_barriers[critical_barriers["barrier_category"].isin(top_categories)]

# Get top 50 critical barriers as evidence
critical_evidence = critical_barriers.sort_values(by="calibrated_confidence", ascending=False).groupby("topic_label").head(10)
critical_evidence = critical_evidence[["reviewId", "content_raw", "topic_label", "barrier_category", "calibrated_confidence", "probability_negative"]]
critical_evidence.to_csv(OUT_DIR / "critical_learning_barriers.csv", index=False)

# 8. Adaptive Learning Mapping (Sekolah Rakyat Design Implications)
adaptive_mapping = {
    "framework": "Sekolah Rakyat Adaptive Learning",
    "barrier_interventions": {}
}

# Example mapping based on theory and category
intervention_logic = {
    "TB-1 Cognitive Difficulty": "Trigger dynamic scaffolding, simplify vocabulary, offer alternative explanation formats (e.g., visual vs text).",
    "TB-2 Engagement & Motivation Problem": "Incorporate micro-rewards, progress visualization, and shorter, chunked learning milestones.",
    "TB-3 Lack of Learning Support": "Activate AI Mentor (Qwen/TinyLlama) for real-time Q&A, suggest relevant question banks.",
    "TB-4 System Usability Issues": "Fallback to offline mode, optimize asset delivery, gracefully degrade UX during high latency.",
    "TB-5 Access & Resource Limitation": "Enable low-bandwidth video modes, background downloading, and robust offline caching.",
    "TB-6 Cost / Affordability Barrier": "Highlight freemium pathways, unlock targeted micro-scholarships based on effort.",
    "TB-7 Content Quality Mismatch": "Route user to prerequisite diagnostic test, realign curriculum path to current ZPD."
}

for _, row in category_metrics.iterrows():
    cat = row["barrier_category"]
    adaptive_mapping["barrier_interventions"][cat] = {
        "severity_rank": row["rank"],
        "priority": row["priority_tier"],
        "design_implication": intervention_logic.get(cat, "Generic intervention required.")
    }

with open(OUT_DIR / "adaptive_learning_mapping.json", "w", encoding="utf-8") as f:
    json.dump(adaptive_mapping, f, indent=4)

# 9. Severity Interpretation Report
interpretation_report = {
    "execution_summary": "Phase 3C Severity Scoring successfully integrated sentiment analysis with LDA taxonomy.",
    "methodology": {
        "formula": "severity_score = freq * negative_ratio * mean_neg_conf * (1 + high_conf_ratio)",
        "adjustment": "Neutral sentiments were not discarded but weighted zero for severity calculation. High confidence negatives (>0.8) applied a multiplier effect to penalize critical failures."
    },
    "key_findings": {
        "most_severe_barrier": category_metrics.iloc[0]["barrier_category"],
        "least_severe_barrier": category_metrics.iloc[-1]["barrier_category"],
        "total_critical_barriers_detected": int(category_metrics[category_metrics["priority_tier"] == "CRITICAL"]["total_severity"].sum())
    }
}

with open(OUT_DIR / "severity_interpretation_report.json", "w", encoding="utf-8") as f:
    json.dump(interpretation_report, f, indent=4)

print("Phase 3C completed successfully! Outputs generated in phase3c_outputs/")
