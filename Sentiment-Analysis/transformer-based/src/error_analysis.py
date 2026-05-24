"""
src/error_analysis.py
Analyzes misclassification patterns and severity.
"""

import os, json, logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

RESULTS = os.path.join(ROOT, "results")
VIS_DIR = os.path.join(ROOT, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

def determine_severity(row):
    t, p = row['true_label'], row['predicted_label']
    if t == p:
        return 'correct'
    
    if (t == 'negative' and p == 'positive') or (t == 'positive' and p == 'negative'):
        return 'critical_confusion'
    elif t == 'neutral' and p in ['positive', 'negative']:
        return 'moderate_confusion'
    elif t == 'negative' and p == 'neutral':
        return 'moderate_confusion'
    elif t == 'positive' and p == 'neutral':
        return 'minor_confusion'
    else:
        return 'unclassified_error'

def run():
    logger.info("Running error_analysis.py...")
    preds_path = os.path.join(RESULTS, "test_predictions.csv")
    df = pd.read_csv(preds_path)
    
    # Basic correct/incorrect
    df['prediction_correct'] = df['true_label'] == df['predicted_label']
    
    # Severity
    df['severity'] = df.apply(determine_severity, axis=1)
    
    # Save error subset
    errors_df = df[~df['prediction_correct']].copy()
    errors_df.to_csv(os.path.join(RESULTS, "error_analysis.csv"), index=False)
    
    # 1. Per-Class F1 Chart
    labels = ['negative', 'neutral', 'positive']
    p, r, f1, _ = precision_recall_fscore_support(df['true_label'], df['predicted_label'], labels=labels)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=f1, palette="viridis")
    plt.title("F1-Score per Sentiment Class")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.0)
    for i, v in enumerate(f1):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "per_class_f1.png"), dpi=300)
    plt.close()
    
    # 2. Misclassification Patterns
    severity_counts = errors_df['severity'].value_counts()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=severity_counts.index, y=severity_counts.values, palette="Reds_r")
    plt.title("Error Severity Distribution")
    plt.ylabel("Number of Errors")
    plt.xticks(rotation=15)
    for i, v in enumerate(severity_counts.values):
        plt.text(i, v + (v*0.02), str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "misclassification_patterns.png"), dpi=300)
    plt.close()
    
    # Generate summary metrics
    summary = {
        "total_errors": len(errors_df),
        "critical_confusions": len(errors_df[errors_df['severity'] == 'critical_confusion']),
        "moderate_confusions": len(errors_df[errors_df['severity'] == 'moderate_confusion']),
        "minor_confusions": len(errors_df[errors_df['severity'] == 'minor_confusion']),
        "f1_scores": dict(zip(labels, f1))
    }
    
    with open(os.path.join(RESULTS, "error_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("error_analysis completed.")
    return summary

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
