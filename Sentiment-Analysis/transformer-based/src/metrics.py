"""
src/metrics.py
Training and evaluation metrics pipeline.
Computes accuracy, precision, recall, f1_macro, f1_weighted, and logs confusion matrix.
"""

import os
import json
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SANITY_DIR = os.path.join(RESULTS_DIR, "sanity_checks")
os.makedirs(SANITY_DIR, exist_ok=True)

def compute_metrics(eval_pred):
    """
    Computes standard classification metrics for HuggingFace Trainer.
    Saves the confusion matrix to a file for later visualization.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    
    # Calculate macro metrics (primary)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    
    # Calculate weighted metrics (secondary)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    
    # Save the CM for later
    cm_path = os.path.join(SANITY_DIR, "latest_confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump({"confusion_matrix": cm.tolist()}, f, indent=2)
        
    logger.info(f"Evaluation - F1 Macro: {f1_macro:.4f} | F1 Weighted: {f1_weighted:.4f} | Acc: {acc:.4f}")
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_weighted": f1_weighted,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted
    }
