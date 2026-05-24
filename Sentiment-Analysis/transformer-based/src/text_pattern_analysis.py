"""
src/text_pattern_analysis.py
Correlates noisy text characteristics with prediction correctness.
"""

import os, json, logging, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

RESULTS = os.path.join(ROOT, "results")
VIS_DIR = os.path.join(ROOT, "visualizations")

logger = logging.getLogger(__name__)

def get_length_bucket(length):
    if length < 5:
        return 'very_short'
    elif length < 15:
        return 'short'
    elif length < 30:
        return 'medium'
    else:
        return 'long'

def run():
    logger.info("Running text_pattern_analysis.py...")
    preds_path = os.path.join(RESULTS, "test_predictions.csv")
    df = pd.read_csv(preds_path)
    
    df['prediction_correct'] = df['true_label'] == df['predicted_label']
    df['text_length'] = df['original_text'].apply(lambda x: len(str(x).split()))
    df['length_bucket'] = df['text_length'].apply(get_length_bucket)
    
    df['contains_emphasis'] = df['original_text'].apply(lambda x: bool(re.search(r'!{2,}|\?{2,}', str(x))))
    df['contains_repetition'] = df['original_text'].apply(lambda x: bool(re.search(r'(.)\1{2,}', str(x))))
    
    # Accuracy per length bucket
    bucket_acc = df.groupby('length_bucket')['prediction_correct'].mean().reindex(['very_short', 'short', 'medium', 'long'])
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=bucket_acc.index, y=bucket_acc.values, palette="Blues_d")
    plt.title("Prediction Accuracy by Review Length")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    for i, v in enumerate(bucket_acc.values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "text_length_vs_accuracy.png"), dpi=300)
    plt.close()

    summary = {
        "accuracy_by_length": bucket_acc.to_dict(),
        "accuracy_with_emphasis": df[df['contains_emphasis']]['prediction_correct'].mean(),
        "accuracy_with_repetition": df[df['contains_repetition']]['prediction_correct'].mean(),
        "overall_accuracy": df['prediction_correct'].mean()
    }
    
    with open(os.path.join(RESULTS, "text_pattern_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("text_pattern_analysis completed.")
    return summary

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
