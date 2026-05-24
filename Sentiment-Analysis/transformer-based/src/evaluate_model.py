"""
src/evaluate_model.py
Evaluates the best model from Phase 3A on the test set.
Exports predictions, classification report, and final metrics.
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from scipy.special import softmax
from sklearn.metrics import classification_report

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models", "final_baseline")
os.makedirs(RESULTS, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from training_config import TRAIN_CONFIG
from dataset_encoder import build_encoded_datasets
from metrics import compute_metrics

def main():
    logger.info("=" * 60)
    logger.info("PHASE 3A - MODEL EVALUATION")
    logger.info("=" * 60)
    
    best_model_dir = os.path.join(MODELS, "best_model")
    if not os.path.exists(best_model_dir):
        logger.error(f"Best model not found at {best_model_dir}. Run training first.")
        sys.exit(1)
        
    logger.info("Loading tokenizer and best model...")
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    
    # Load raw dataset for original text
    data_dir = os.path.join(ROOT, "data", "cleaned")
    encoded_dict, _ = build_encoded_datasets(
        data_dir=data_dir,
        model_name="indolem/indobertweet-base-uncased",
        max_length=TRAIN_CONFIG["max_length"],
        batch_size=16,
        splits={"test": "test_clean.csv"}
    )
    test_dataset = encoded_dict["test"]
    df_test = pd.read_csv(os.path.join(data_dir, "test_clean.csv"))
    
    logger.info(f"Evaluating on {len(test_dataset)} test samples...")
    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    
    predictions = trainer.predict(test_dataset)
    
    # Extract metrics
    metrics = predictions.metrics
    with open(os.path.join(RESULTS, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    logger.info(f"Test Accuracy:  {metrics.get('test_accuracy', 0):.4f}")
    logger.info(f"Test Macro F1:  {metrics.get('test_f1_macro', 0):.4f}")
    
    # Extract logits and labels
    logits = predictions.predictions
    true_labels = predictions.label_ids
    probs = softmax(logits, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    # Map labels back to names
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    
    # Generate classification report
    report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=["negative", "neutral", "positive"],
        digits=4
    )
    with open(os.path.join(RESULTS, "classification_report.txt"), "w") as f:
        f.write("PHASE 3A FINAL CLASSIFICATION REPORT\n")
        f.write("====================================\n\n")
        f.write(report)
        
    logger.info("Saved classification_report.txt")
    
    # Generate test_predictions.csv
    df_preds = pd.DataFrame({
        "original_text": df_test["content"] if "content" in df_test.columns else df_test["text"],
        "true_label": [id2label[label] for label in true_labels],
        "predicted_label": [id2label[label] for label in pred_labels],
        "confidence_score": confidences
    })
    
    df_preds.to_csv(os.path.join(RESULTS, "test_predictions.csv"), index=False)
    logger.info("Saved test_predictions.csv")
    logger.info("EVALUATION COMPLETE.")

if __name__ == "__main__":
    main()
