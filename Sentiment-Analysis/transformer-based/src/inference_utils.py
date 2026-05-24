"""
src/inference_utils.py
Utility helpers for the final inference pipeline:
- Temperature scaling
- Batch tokenization
- Probability extraction
"""

import numpy as np
import torch
from scipy.special import softmax


TEMPERATURE = 0.697   # Optimal T from Phase 3C-3 calibration
ID2LABEL    = {0: "negative", 1: "neutral", 2: "positive"}


def apply_temperature_scaling(logits: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    """Divide logits by temperature then softmax → calibrated probabilities."""
    scaled = logits / temperature
    return softmax(scaled, axis=1)


def extract_prediction_columns(logits: np.ndarray, temperature: float = TEMPERATURE):
    """
    Given raw logits (N x 3), returns a dict of per-row prediction values.
    """
    raw_probs  = softmax(logits, axis=1)            # uncalibrated
    cal_probs  = apply_temperature_scaling(logits, temperature)

    pred_labels = np.argmax(cal_probs, axis=1)
    confidence  = np.max(cal_probs, axis=1)

    return {
        "predicted_label":       pred_labels,
        "predicted_label_name":  [ID2LABEL[i] for i in pred_labels],
        "confidence_score":      np.max(raw_probs, axis=1),     # raw confidence
        "calibrated_confidence": confidence,                    # calibrated confidence
        "probability_negative":  cal_probs[:, 0],
        "probability_neutral":   cal_probs[:, 1],
        "probability_positive":  cal_probs[:, 2],
    }
