"""
src/calibrate_model.py
Phase 3C-3: Post-hoc Temperature Scaling for confidence calibration.
No retraining — operates purely on saved model logits from the best experiment.
Computes Expected Calibration Error (ECE) before and after scaling.
Generates a reliability diagram (calibration curve).
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.optimize import minimize_scalar

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

RESULTS = os.path.join(ROOT, "results")
VIS_DIR = os.path.join(ROOT, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from training_config import TRAIN_CONFIG
from dataset_encoder import build_encoded_datasets
from metrics import compute_metrics


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) using equal-width confidence bins."""
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (max_probs >= lo) & (max_probs < hi)
        if mask.sum() == 0:
            continue
        avg_confidence = max_probs[mask].mean()
        avg_accuracy   = correct[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(avg_confidence - avg_accuracy)
    return float(ece)


def plot_reliability_diagram(probs_before, probs_after, labels, save_path, n_bins=10):
    """Reliability diagram for before and after temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, probs, title in zip(axes, [probs_before, probs_after], ["Before Calibration", "After Temperature Scaling"]):
        max_probs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        correct = (preds == labels).astype(float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_conf, bin_acc = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (max_probs >= lo) & (max_probs < hi)
            if mask.sum() == 0:
                continue
            bin_conf.append(max_probs[mask].mean())
            bin_acc.append(correct[mask].mean())

        ece = compute_ece(probs, labels, n_bins)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, color="#1f77b4", label="Model Calibration")
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved reliability diagram to {save_path}")


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by temperature and apply softmax."""
    scaled = logits / temperature
    return softmax(scaled, axis=1)


def find_optimal_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find T that minimizes NLL on validation predictions."""
    from scipy.special import log_softmax

    def nll_loss(T):
        log_probs = log_softmax(logits / T, axis=1)
        return -log_probs[np.arange(len(labels)), labels].mean()

    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def main():
    logger.info("=" * 60)
    logger.info("PHASE 3C-3 — CONFIDENCE CALIBRATION ANALYSIS")
    logger.info("=" * 60)

    # Determine best model to calibrate: prefer focal experiment, fall back to baseline
    focal_model_path = os.path.join(ROOT, "models", "focal_experiment", "best_model")
    baseline_model_path = os.path.join(ROOT, "models", "final_baseline", "best_model")

    if os.path.exists(focal_model_path):
        model_path = focal_model_path
        model_source = "focal_experiment"
    else:
        model_path = baseline_model_path
        model_source = "baseline"

    logger.info(f"Calibrating model: {model_source} from {model_path}")

    tokenizer_hf = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    data_dir = os.path.join(ROOT, "data", "cleaned")
    encoded_dict, _ = build_encoded_datasets(
        data_dir=data_dir,
        model_name="indolem/indobertweet-base-uncased",
        max_length=TRAIN_CONFIG["max_length"],
        batch_size=16,
        splits={"test": "test_clean.csv"}
    )
    test_dataset = encoded_dict["test"]

    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    predictions = trainer.predict(test_dataset)

    logits     = predictions.predictions      # raw logits
    true_labels = predictions.label_ids

    # Softmax probs before calibration
    probs_before = softmax(logits, axis=1)
    ece_before   = compute_ece(probs_before, true_labels)
    logger.info(f"ECE before calibration: {ece_before:.4f}")

    # Find optimal temperature and apply
    opt_T = find_optimal_temperature(logits, true_labels)
    logger.info(f"Optimal Temperature: {opt_T:.4f}")
    probs_after  = apply_temperature_scaling(logits, opt_T)
    ece_after    = compute_ece(probs_after, true_labels)
    logger.info(f"ECE after calibration (T={opt_T:.4f}): {ece_after:.4f}")

    # Save calibrated predictions as separate artifact
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    df_cal = pd.DataFrame({
        "true_label": [id2label[l] for l in true_labels],
        "predicted_label_before": [id2label[np.argmax(p)] for p in probs_before],
        "confidence_before": np.max(probs_before, axis=1),
        "predicted_label_after": [id2label[np.argmax(p)] for p in probs_after],
        "confidence_after": np.max(probs_after, axis=1),
    })
    df_cal.to_csv(os.path.join(RESULTS, "calibrated_predictions.csv"), index=False)
    logger.info("Saved calibrated_predictions.csv")

    # Reliability diagram
    plot_reliability_diagram(
        probs_before, probs_after, true_labels,
        save_path=os.path.join(VIS_DIR, "calibration_curve.png")
    )

    # Save metrics
    cal_metrics = {
        "model_source": model_source,
        "optimal_temperature": opt_T,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "delta_ece": ece_before - ece_after,
        "accuracy_before": float((np.argmax(probs_before, axis=1) == true_labels).mean()),
        "accuracy_after": float((np.argmax(probs_after, axis=1) == true_labels).mean()),
    }
    with open(os.path.join(RESULTS, "calibration_metrics.json"), "w") as f:
        json.dump(cal_metrics, f, indent=2)

    logger.info(f"ECE reduced by {cal_metrics['delta_ece']:.4f}")
    logger.info("PHASE 3C-3 COMPLETE.")


if __name__ == "__main__":
    main()
