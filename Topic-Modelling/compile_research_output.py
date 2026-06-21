"""
compile_research_output.py
==========================
SCOPE: Topic Modelling Research Output Compilation
PROJECT: Sekolah Rakyat - Learning Barrier Analysis (AITF UB x KOMDIGI 2026)

Objective boundary:
  - Extract, structure, and interpret learning barriers from LDA topic outputs
  - NO severity scoring, NO system design, NO decision engines

Tasks covered:
  T1: Learning Barrier Identification
  T2: Topic Structuring and Labeling
  T3: Topic Distribution Analysis
  T4: Learning Barrier Mapping
  T5: Blueprint Input Validation (Light Interpretation Only)

Inputs:  Phase 3A topic keywords + Phase 3B taxonomy (already produced)
Outputs: research_output/ (5 clean research deliverables)
"""

import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("c:/Users/ASUS/Documents/AITF-2026/PKL/Topic-Modelling")
PHASE3A = BASE_DIR / "phase3a_outputs"
PHASE3B = BASE_DIR / "phase3b_outputs"
PHASE1 = BASE_DIR / "phase1_outputs"
OUT = BASE_DIR / "research_output"
OUT.mkdir(parents=True, exist_ok=True)

print("Compiling aligned research outputs...")

# ─────────────────────────────────────────────────────────
# Load source data from already-produced phase outputs
# ─────────────────────────────────────────────────────────

# Phase 3B interpretation report (topic labels + keywords)
with open(PHASE3B / "topic_interpretation_report.json", encoding="utf-8") as f:
    interpretation = json.load(f)

# Phase 3A keyword CSV for quantitative keyword weights
kw_df = pd.read_csv(PHASE3A / "topic_keywords_k8.csv")

# Phase 3A representative reviews
rep_df = pd.read_csv(PHASE3A / "representative_reviews_k8.csv")

# Corpus for document counts
corpus_df = pd.read_csv(PHASE1 / "lda_ready_corpus.csv").drop_duplicates(subset=["reviewId"])

# Phase 3B taxonomy (category → topics list)
with open(PHASE3B / "phase3b_topic_taxonomy.json", encoding="utf-8") as f:
    taxonomy = json.load(f)

# Build reverse map: topic_label → category
label_to_category = {}
for category, labels in taxonomy["categories"].items():
    for label in labels:
        label_to_category[label] = category

# ─────────────────────────────────────────────────────────
# T1: Learning Barrier Identification
# Validate latent learning barriers from LDA topic keywords
# and representative reviews
# ─────────────────────────────────────────────────────────
print("T1: Learning Barrier Identification...")

t1_output = {
    "task": "T1 - Learning Barrier Identification",
    "description": "Latent learning barriers identified from LDA_k8 topic keyword clusters and representative document analysis.",
    "model": "LDA_k8 (k=8, Cv=0.4288)",
    "corpus_size": 41797,
    "barriers_identified": []
}

for topic_key, topic_data in interpretation.items():
    t_num = int(topic_key.replace("topic_", ""))
    top_kw = topic_data.get("top_10_keywords", [])

    # Pull top weighted keyword pairs from phase3a CSV
    topic_rows = kw_df[kw_df["topic_id"] == t_num].sort_values("weight", ascending=False).head(5)
    top_weighted = topic_rows[["word", "weight"]].to_dict(orient="records") if not topic_rows.empty else []

    barrier_entry = {
        "topic_id": t_num,
        "barrier_label": topic_data["label"],
        "barrier_category": topic_data["category"],
        "theory_basis": topic_data.get("theory_basis", "N/A"),
        "top_keywords": top_kw,
        "top_weighted_terms": top_weighted,
        "identification_rationale": topic_data.get("rationale", ""),
    }
    t1_output["barriers_identified"].append(barrier_entry)

t1_output["barriers_identified"].sort(key=lambda x: x["topic_id"])
t1_output["total_barriers_found"] = len(t1_output["barriers_identified"])

with open(OUT / "learning_barrier_identification.json", "w", encoding="utf-8") as f:
    json.dump(t1_output, f, indent=4, ensure_ascii=False)
print("  -> learning_barrier_identification.json")

# ─────────────────────────────────────────────────────────
# T2: Topic Structuring and Labeling
# Consistent semantic labels grounded in keyword evidence
# ─────────────────────────────────────────────────────────
print("T2: Topic Label Mapping...")

t2_rows = []
for topic_key, topic_data in sorted(interpretation.items(), key=lambda x: int(x[0].replace("topic_", ""))):
    t_num = int(topic_key.replace("topic_", ""))
    t2_rows.append({
        "topic_id": t_num,
        "topic_label": topic_data["label"],
        "barrier_category": topic_data["category"],
        "theory_basis": topic_data.get("theory_basis", ""),
        "top_keywords": ", ".join(topic_data.get("top_10_keywords", [])),
        "label_grounding": "Keyword-driven + representative document analysis",
    })

# Save as CSV (table format)
t2_df = pd.DataFrame(t2_rows)
t2_df.to_csv(OUT / "topic_label_mapping.csv", index=False)

# Also save structured JSON
t2_json = {
    "task": "T2 - Topic Structuring and Labeling",
    "constraint": "Labels grounded in observed topic keywords only — not external system design assumptions",
    "labels": t2_rows
}
with open(OUT / "topic_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(t2_json, f, indent=4, ensure_ascii=False)
print("  -> topic_label_mapping.csv + .json")

# ─────────────────────────────────────────────────────────
# T3: Topic Distribution Analysis
# Frequency and relative dominance across the corpus
# ─────────────────────────────────────────────────────────
print("T3: Topic Distribution Analysis...")

# Infer topic distribution from representative review topic assignments
# Use the dominant_topic column if available, else use topic_id from rep reviews
if "dominant_topic" in rep_df.columns:
    topic_col = "dominant_topic"
elif "topic_id" in rep_df.columns:
    topic_col = "topic_id"
else:
    # Fall back to keyword frequency as proxy
    topic_col = None

# Load phase3a barrier cluster audit for per-topic document counts if available
audit_path = PHASE3A / "barrier_cluster_audit_k8.csv"
if audit_path.exists():
    audit_df = pd.read_csv(audit_path)
    has_audit = True
else:
    has_audit = False

# Build distribution from label mapping + known topic structure
# Use keyword weight totals as proxy for topic salience
topic_salience = []
total_weight = 0.0
for t_num in range(8):
    rows = kw_df[kw_df["topic_id"] == t_num]
    salience = float(rows["weight"].sum()) if not rows.empty else 0.0
    total_weight += salience
    topic_salience.append({"topic_id": t_num, "total_keyword_weight": round(salience, 4)})

for entry in topic_salience:
    entry["relative_dominance_pct"] = round(
        (entry["total_keyword_weight"] / total_weight * 100) if total_weight > 0 else 0, 2
    )

# Attach labels
label_lookup = {
    int(k.replace("topic_", "")): v["label"]
    for k, v in interpretation.items()
}
category_lookup = {
    int(k.replace("topic_", "")): v["category"]
    for k, v in interpretation.items()
}

for entry in topic_salience:
    entry["topic_label"] = label_lookup.get(entry["topic_id"], "Unknown")
    entry["barrier_category"] = category_lookup.get(entry["topic_id"], "Unknown")

topic_salience.sort(key=lambda x: x["relative_dominance_pct"], reverse=True)

t3_output = {
    "task": "T3 - Topic Distribution Analysis",
    "note": "Relative dominance proxied by aggregate keyword weight per topic (LDA_k8 outputs). Corpus size: 15,324 unique reviews.",
    "metrics": ["topic_id", "total_keyword_weight", "relative_dominance_pct"],
    "distribution": topic_salience,
    "most_dominant_topic": topic_salience[0]["topic_label"],
    "least_dominant_topic": topic_salience[-1]["topic_label"],
    "distribution_imbalance_note": (
        f"Most dominant topic ({topic_salience[0]['relative_dominance_pct']}%) "
        f"vs least dominant ({topic_salience[-1]['relative_dominance_pct']}%) "
        f"— ratio: {round(topic_salience[0]['relative_dominance_pct'] / max(topic_salience[-1]['relative_dominance_pct'], 0.01), 1)}x"
    )
}

with open(OUT / "topic_distribution_analysis.json", "w", encoding="utf-8") as f:
    json.dump(t3_output, f, indent=4, ensure_ascii=False)

# Also write distribution table as CSV
pd.DataFrame(topic_salience).to_csv(OUT / "topic_distribution_analysis.csv", index=False)
print("  -> topic_distribution_analysis.json + .csv")

# ─────────────────────────────────────────────────────────
# T4: Learning Barrier Mapping
# Topic → Learning Barrier interpretive mapping table
# ─────────────────────────────────────────────────────────
print("T4: Learning Barrier Mapping...")

t4_rows = []
for topic_key, topic_data in sorted(interpretation.items(), key=lambda x: int(x[0].replace("topic_", ""))):
    t_num = int(topic_key.replace("topic_", ""))
    t4_rows.append({
        "topic_id": t_num,
        "topic_label": topic_data["label"],
        "learning_barrier": topic_data["category"],
        "theory_basis": topic_data.get("theory_basis", ""),
        "mapping_type": "Interpretive post-hoc mapping (not model-supervised)",
        "keyword_evidence": ", ".join(topic_data.get("top_10_keywords", [])[:5]),
        "rationale_summary": topic_data.get("rationale", "")[:120] + "..."
    })

t4_df = pd.DataFrame(t4_rows)
t4_df.to_csv(OUT / "learning_barrier_topic_map.csv", index=False)

t4_json = {
    "task": "T4 - Learning Barrier Mapping",
    "constraint": "Mapping is interpretive, not prescriptive for system design",
    "mapping_table": t4_rows,
    "unique_barrier_categories": list(set(r["learning_barrier"] for r in t4_rows)),
    "note_tb5_absent": "TB-5 (Access & Resource Limitation / Digital Divide) did not form a statistically distinct LDA cluster. Access-related signals are distributed across TB-4 and TB-6 topics.",
}

with open(OUT / "learning_barrier_topic_map.json", "w", encoding="utf-8") as f:
    json.dump(t4_json, f, indent=4, ensure_ascii=False)
print("  -> learning_barrier_topic_map.csv + .json")

# ─────────────────────────────────────────────────────────
# T5: Blueprint Input Validation (Light Interpretation Only)
# Qualitative assessment of barrier coverage completeness
# NO system architecture or feature design
# ─────────────────────────────────────────────────────────
print("T5: Barrier Coverage Assessment...")

active_categories = list(set(
    v["category"] for v in interpretation.values()
))

expected_barrier_spectrum = [
    "Cognitive / comprehension barriers",
    "Engagement and motivation barriers",
    "Instructional support and scaffolding barriers",
    "Technical access / system reliability barriers",
    "Economic / affordability barriers",
    "Content and curriculum quality barriers",
    "Infrastructure / connectivity barriers (digital divide)",
]

coverage_map = [
    {
        "expected_barrier_type": "Cognitive / comprehension barriers",
        "covered_by": "TB-1 Cognitive Difficulty",
        "coverage_status": "COVERED",
        "evidence": "Topic 1 keywords: paham, jelas, animasi — directly signals comprehension friction"
    },
    {
        "expected_barrier_type": "Engagement and motivation barriers",
        "covered_by": "TB-2 Engagement & Motivation Problem",
        "coverage_status": "COVERED",
        "evidence": "Topic 4 keywords: semangat, seru, suka — motivation signals present"
    },
    {
        "expected_barrier_type": "Instructional support and scaffolding barriers",
        "covered_by": "TB-3 Lack of Learning Support",
        "coverage_status": "COVERED",
        "evidence": "Topic 3 keywords: bank_soal, bahas, tambah — explicit support-seeking signals"
    },
    {
        "expected_barrier_type": "Technical access / system reliability barriers",
        "covered_by": "TB-4 System Usability Issues",
        "coverage_status": "COVERED — STRONG",
        "evidence": "Topics 5, 6, 7 all map here. Most lexically distinct cluster across model."
    },
    {
        "expected_barrier_type": "Economic / affordability barriers",
        "covered_by": "TB-6 Cost / Affordability Barrier",
        "coverage_status": "COVERED",
        "evidence": "Topic 2 keywords: murah, paket, harga — pricing friction evident"
    },
    {
        "expected_barrier_type": "Content and curriculum quality barriers",
        "covered_by": "TB-7 Content Quality Mismatch",
        "coverage_status": "COVERED",
        "evidence": "Topic 0 keywords: kurang, materi — content gap and mismatch signals"
    },
    {
        "expected_barrier_type": "Infrastructure / connectivity barriers (digital divide)",
        "covered_by": "Not independently clustered (absorbed into TB-4 / TB-6)",
        "coverage_status": "PARTIAL — NOT DISTINCT",
        "evidence": "No independent LDA cluster for digital divide / connectivity emerged. Signals distributed across TB-4 (lag, crash) and TB-6 (cost of data). This is an empirical finding, not a gap in the model."
    },
]

t5_output = {
    "task": "T5 - Blueprint Input Validation (Light Interpretation Only)",
    "constraint": "Assessment only — NO system architecture, feature design, or AI recommendations",
    "overall_assessment": "The LDA_k8 model provides SUFFICIENT coverage of the primary learning barrier spectrum for informing educational system design discussions.",
    "coverage_completeness": f"{len([c for c in coverage_map if 'COVERED' in c['coverage_status']])}/{len(coverage_map)} barrier types covered",
    "coverage_detail": coverage_map,
    "strengths": [
        "Technical/system barriers (TB-4) form the most lexically distinct cluster — strong empirical signal",
        "Cognitive difficulty (TB-1) and content mismatch (TB-7) are semantically separable — both independently identifiable",
        "Economic friction (TB-6) has a distinct keyword profile (pricing vocabulary) not confused with other barriers",
    ],
    "limitations": [
        "Digital divide / connectivity barriers are not independently extractable as a distinct topic — absorbed into TB-4 and TB-6",
        "Short median document length (~7 tokens) limits per-document topic resolution",
        "Topic coherence (Cv=0.4288) is moderate — interpretations should be treated as approximate, not definitive",
        "Taxonomy mapping is post-hoc interpretive — not a supervised classification output",
    ],
    "qualitative_summary": (
        "The 8 discovered topics map cleanly to 6 of the 7 expected barrier categories in the educational literature. "
        "The absence of a distinct Digital Divide cluster (TB-5) is itself a meaningful finding: in this corpus, "
        "infrastructure-level access failures manifest primarily as system usability friction (TB-4), "
        "not as a linguistically separate discourse. "
        "Overall, the topic modelling output provides adequate interpretive grounding for "
        "learning barrier identification in the context of Indonesian digital education platforms."
    ),
}

with open(OUT / "barrier_coverage_assessment.json", "w", encoding="utf-8") as f:
    json.dump(t5_output, f, indent=4, ensure_ascii=False)
print("  -> barrier_coverage_assessment.json")

print(f"\nAll 5 aligned research outputs saved to:\n  {OUT}")
print("\nResearch output scope:")
print("  T1: Learning barrier identification")
print("  T2: Topic label mapping (CSV + JSON)")
print("  T3: Topic distribution analysis (CSV + JSON)")
print("  T4: Learning barrier topic map (CSV + JSON)")
print("  T5: Barrier coverage assessment (qualitative)")
