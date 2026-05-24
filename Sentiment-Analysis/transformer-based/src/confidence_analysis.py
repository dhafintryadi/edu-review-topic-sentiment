"""
src/confidence_analysis.py
Analyzes confidence calibration and high-confidence errors.
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

def run():
    logger.info("Running confidence_analysis.py...")
    preds_path = os.path.join(RESULTS, "test_predictions.csv")
    df = pd.read_csv(preds_path)
    
    df['prediction_correct'] = df['true_label'] == df['predicted_label']
    
    # Calibration stats
    mean_conf_correct = df[df['prediction_correct']]['confidence_score'].mean()
    mean_conf_incorrect = df[~df['prediction_correct']]['confidence_score'].mean()
    gap = mean_conf_correct - mean_conf_incorrect
    
    # High confidence errors (confidence > 0.8 but wrong)
    high_conf_errors = df[(~df['prediction_correct']) & (df['confidence_score'] > 0.8)].copy()
    high_conf_errors.to_csv(os.path.join(RESULTS, "high_confidence_errors.csv"), index=False)
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df, 
        x="confidence_score", 
        hue="prediction_correct", 
        element="step", 
        stat="density", 
        common_norm=False,
        palette={True: '#2ca02c', False: '#d62728'},
        alpha=0.6,
        bins=30
    )
    plt.axvline(mean_conf_correct, color='green', linestyle='dashed', linewidth=2, label=f'Mean Correct: {mean_conf_correct:.2f}')
    plt.axvline(mean_conf_incorrect, color='red', linestyle='dashed', linewidth=2, label=f'Mean Incorrect: {mean_conf_incorrect:.2f}')
    
    plt.title("Confidence Score Calibration (Correct vs Incorrect)")
    plt.xlabel("Confidence Score (Softmax Probability)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "confidence_distribution.png"), dpi=300)
    plt.close()
    
    summary = {
        "mean_confidence_correct": float(mean_conf_correct),
        "mean_confidence_incorrect": float(mean_conf_incorrect),
        "confidence_gap": float(gap),
        "num_high_confidence_errors": len(high_conf_errors)
    }
    
    with open(os.path.join(RESULTS, "confidence_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("confidence_analysis completed.")
    return summary

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
