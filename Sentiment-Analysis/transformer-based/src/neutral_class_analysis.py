"""
src/neutral_class_analysis.py
Deep dive into neutral class ambiguity and confusion flows.
"""

import os, json, logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

RESULTS = os.path.join(ROOT, "results")
VIS_DIR = os.path.join(ROOT, "visualizations")

logger = logging.getLogger(__name__)

# Heuristic lists for tagging
SUGGESTION_WORDS = ['tolong', 'sebaiknya', 'harap', 'tambah', 'mohon', 'semoga', 'update', 'perbaiki', 'kurang']
POS_WORDS = ['bagus', 'keren', 'membantu', 'suka', 'mantap', 'baik', 'mudah']
NEG_WORDS = ['jelek', 'susah', 'error', 'buruk', 'kecewa', 'lambat', 'lemot']

def tag_ambiguity(text):
    text = str(text).lower()
    is_suggestion = any(w in text for w in SUGGESTION_WORDS)
    has_pos = any(w in text for w in POS_WORDS)
    has_neg = any(w in text for w in NEG_WORDS)
    
    if has_pos and has_neg:
        return 'mixed_sentiment'
    elif is_suggestion:
        return 'suggestion_or_request'
    elif has_pos and not has_neg:
        return 'lukewarm_positive'
    elif has_neg and not has_pos:
        return 'lukewarm_negative'
    else:
        return 'unclear_sentiment'

def run():
    logger.info("Running neutral_class_analysis.py...")
    preds_path = os.path.join(RESULTS, "test_predictions.csv")
    df = pd.read_csv(preds_path)
    
    # Isolate true neutral reviews
    neutral_df = df[df['true_label'] == 'neutral'].copy()
    
    # Tag ambiguity
    neutral_df['ambiguity_category'] = neutral_df['original_text'].apply(tag_ambiguity)
    neutral_df.to_csv(os.path.join(RESULTS, "neutral_review_analysis.csv"), index=False)
    
    # Breakdown of Neutral Confusions
    # How are neutral true labels predicted?
    neutral_pred_counts = neutral_df['predicted_label'].value_counts()
    
    plt.figure(figsize=(8, 5))
    plt.pie(neutral_pred_counts, labels=neutral_pred_counts.index, autopct='%1.1f%%', 
            colors=['#1f77b4', '#ff7f0e', '#2ca02c'], startangle=140, explode=[0.05]*len(neutral_pred_counts))
    plt.title("True Neutral Reviews: Prediction Breakdown")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "neutral_confusion_breakdown.png"), dpi=300)
    plt.close()

    # Flow from all classes
    # We will make a stacked bar for "Where did errors go?"
    confusion_matrix_data = pd.crosstab(df['true_label'], df['predicted_label'])
    
    confusion_matrix_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title('Confusion Flow (Stacked Bar)')
    plt.xlabel('True Label')
    plt.ylabel('Count')
    plt.legend(title='Predicted Label')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "confusion_flow.png"), dpi=300)
    plt.close()

    ambiguity_counts = neutral_df['ambiguity_category'].value_counts().to_dict()
    
    summary = {
        "total_true_neutral": len(neutral_df),
        "predicted_as_positive": int(neutral_pred_counts.get('positive', 0)),
        "predicted_as_negative": int(neutral_pred_counts.get('negative', 0)),
        "predicted_as_neutral": int(neutral_pred_counts.get('neutral', 0)),
        "ambiguity_breakdown": ambiguity_counts
    }
    
    with open(os.path.join(RESULTS, "neutral_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("neutral_class_analysis completed.")
    return summary

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
