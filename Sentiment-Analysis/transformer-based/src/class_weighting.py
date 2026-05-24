"""
src/class_weighting.py
Phase 2C: Imbalance Strategy Preparer
Calculates PyTorch-compatible class weights for the Weighted Cross-Entropy Loss based on the cleaned dataset.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

class ImbalanceHandler:
    def __init__(self, label_col: str = "sentiment_id"):
        self.label_col = label_col
        self.classes = np.array([0, 1, 2]) # 0: Negative, 1: Neutral, 2: Positive

    def compute_weights(self, df: pd.DataFrame) -> dict:
        """
        Computes class weights based on the distribution in the given DataFrame (usually train).
        Returns a dictionary with standard stats and the computed weights.
        """
        if self.label_col not in df.columns:
            logger.error(f"Label column '{self.label_col}' not found.")
            return {}
            
        labels = df[self.label_col].dropna().astype(int).values
        
        # Calculate actual distribution
        counts = np.bincount(labels, minlength=len(self.classes))
        total = len(labels)
        
        distribution = {
            "negative": int(counts[0]),
            "neutral": int(counts[1]),
            "positive": int(counts[2])
        }
        
        percentages = {
            "negative": round(counts[0] / total * 100, 2),
            "neutral": round(counts[1] / total * 100, 2),
            "positive": round(counts[2] / total * 100, 2)
        }
        
        # Calculate sklearn class weights (n_samples / (n_classes * np.bincount(y)))
        weights = compute_class_weight(
            class_weight='balanced',
            classes=self.classes,
            y=labels
        )
        
        # Imbalance ratio (majority class count / minority class count)
        majority_count = np.max(counts)
        minority_count = np.min(counts)
        imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
        
        stats = {
            "total_samples": total,
            "distribution_counts": distribution,
            "distribution_percentages": percentages,
            "class_weights": [round(float(w), 4) for w in weights],
            "imbalance_ratio": round(float(imbalance_ratio), 2)
        }
        
        logger.info(f"Class Weights Calculated: {stats['class_weights']} (Imbalance Ratio: {stats['imbalance_ratio']})")
        
        return stats
