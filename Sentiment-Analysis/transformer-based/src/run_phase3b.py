"""
src/run_phase3b.py
Main orchestrator for Phase 3B Error Analysis and Interpretability.
"""

import os, sys, json, logging
import pandas as pd

import error_analysis
import confidence_analysis
import neutral_class_analysis
import text_pattern_analysis

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
RESULTS = os.path.join(ROOT, "results")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def extract_qualitative_examples(df):
    logger.info("Extracting representative qualitative examples...")
    
    examples = []
    
    def add_section(title, subset):
        examples.append(f"=== {title.upper()} ===")
        if subset.empty:
            examples.append("  (None found)\n")
            return
        for _, row in subset.head(5).iterrows():
            examples.append(f"Text : {row['original_text']}")
            examples.append(f"True : {row['true_label'].upper()}  |  Pred: {row['predicted_label'].upper()}  |  Conf: {row['confidence_score']:.4f}")
            examples.append("-" * 40)
        examples.append("\n")

    # High confidence errors
    subset = df[(df['prediction_correct'] == False) & (df['confidence_score'] > 0.8)]
    add_section("High Confidence Errors (Model was certain but wrong)", subset)
    
    # Ambiguous neutral examples
    subset = df[(df['true_label'] == 'neutral') & (df['predicted_label'] != 'neutral')]
    add_section("True Neutral Misclassified as Pos/Neg", subset)
    
    # Critical confusions
    subset = df[((df['true_label'] == 'negative') & (df['predicted_label'] == 'positive')) | 
                ((df['true_label'] == 'positive') & (df['predicted_label'] == 'negative'))]
    add_section("Critical Confusions (Pos <-> Neg)", subset)
    
    # Very short errors
    df['length'] = df['original_text'].apply(lambda x: len(str(x).split()))
    subset = df[(df['prediction_correct'] == False) & (df['length'] <= 3)]
    add_section("Very Short Text Errors", subset)

    with open(os.path.join(RESULTS, "qualitative_examples.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(examples))
    
    logger.info("Saved qualitative_examples.txt")

def generate_summary():
    logger.info("Compiling Phase 3B Summary...")
    
    with open(os.path.join(RESULTS, "error_analysis_summary.json")) as f:
        e_sum = json.load(f)
    with open(os.path.join(RESULTS, "confidence_analysis_summary.json")) as f:
        c_sum = json.load(f)
    with open(os.path.join(RESULTS, "neutral_analysis_summary.json")) as f:
        n_sum = json.load(f)
    with open(os.path.join(RESULTS, "text_pattern_summary.json")) as f:
        t_sum = json.load(f)
        
    summary = f"""===========================================================
PHASE 3B - ERROR ANALYSIS & INTERPRETABILITY
===========================================================

[1. ERROR SEVERITY]
Total Errors: {e_sum['total_errors']}
- Critical (Pos<->Neg) : {e_sum['critical_confusions']}
- Moderate (Neutral<->Pos/Neg) : {e_sum['moderate_confusions']}
- Minor (Pos->Neutral) : {e_sum['minor_confusions']}

[2. CONFIDENCE CALIBRATION]
Mean Confidence (Correct)   : {c_sum['mean_confidence_correct']:.4f}
Mean Confidence (Incorrect) : {c_sum['mean_confidence_incorrect']:.4f}
Confidence Gap              : {c_sum['confidence_gap']:.4f}
High-Confidence Errors      : {c_sum['num_high_confidence_errors']}

[3. NEUTRAL CLASS AMBIGUITY]
Total True Neutral  : {n_sum['total_true_neutral']}
Predicted Positive  : {n_sum['predicted_as_positive']}
Predicted Negative  : {n_sum['predicted_as_negative']}
Predicted Neutral   : {n_sum['predicted_as_neutral']}

Ambiguity Breakdown in True Neutrals:
{json.dumps(n_sum['ambiguity_breakdown'], indent=2)}

[4. TEXT PATTERNS]
Accuracy by Length:
{json.dumps(t_sum['accuracy_by_length'], indent=2)}
Accuracy (w/ Emphasis)   : {t_sum['accuracy_with_emphasis']:.4f}
Accuracy (w/ Repetition) : {t_sum['accuracy_with_repetition']:.4f}
Overall Accuracy         : {t_sum['overall_accuracy']:.4f}

===========================================================
"""
    with open(os.path.join(RESULTS, "phase3b_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    
    logger.info("Saved phase3b_summary.txt")

def main():
    logger.info("=" * 60)
    logger.info("STARTING PHASE 3B - ERROR ANALYSIS & INTERPRETABILITY")
    logger.info("=" * 60)
    
    preds_path = os.path.join(RESULTS, "test_predictions.csv")
    if not os.path.exists(preds_path):
        logger.error(f"Missing {preds_path}! Run Phase 3A evaluate_model.py first.")
        sys.exit(1)
        
    df = pd.read_csv(preds_path)
    df['prediction_correct'] = df['true_label'] == df['predicted_label']
    
    # Run modules
    error_analysis.run()
    confidence_analysis.run()
    neutral_class_analysis.run()
    text_pattern_analysis.run()
    
    # Orchestrator tasks
    extract_qualitative_examples(df)
    generate_summary()
    
    logger.info("PHASE 3B COMPLETE.")

if __name__ == "__main__":
    main()
