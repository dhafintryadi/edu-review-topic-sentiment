"""
src/run_phase3c2.py
Phase 3C-2: Neutral Boundary Optimization Experiment.
Uses WeightedTrainer with adjusted class weights (Neutral: 4.6889 → 2.5).
Everything else is identical to Phase 3A baseline.
Variable changed: class_weights[1] only.
"""

import os, sys, time, json, logging, datetime
import pandas as pd
import numpy as np
from scipy.special import softmax as scipy_softmax
from sklearn.metrics import classification_report
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

LOGS = os.path.join(ROOT, "logs", "training_logs")
os.makedirs(LOGS, exist_ok=True)
LOG_FILE = os.path.join(LOGS, "phase3c2.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

from training_config import TRAIN_CONFIG
from freeze_layers import apply_layer_freezing
from metrics import compute_metrics
from weighted_trainer import WeightedTrainer
from dataset_encoder import build_encoded_datasets
from neutral_boundary_optimizer import get_adjusted_weights, get_experiment_metadata

RESULTS = os.path.join(ROOT, "results")
MODELS  = os.path.join(ROOT, "models", "neutral_experiment")
BASELINE_METRICS_PATH = os.path.join(RESULTS, "final_metrics.json")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

ADJUSTED_WEIGHTS = get_adjusted_weights()
EXPERIMENT_META  = get_experiment_metadata()


def main():
    logger.info("=" * 60)
    logger.info("PHASE 3C-2 - NEUTRAL BOUNDARY OPTIMIZATION")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info(f"Variable: Neutral weight {EXPERIMENT_META['original_neutral_weight']} → {EXPERIMENT_META['adjusted_neutral_weight']}")
    logger.info("=" * 60)

    start_time = time.time()

    data_dir = os.path.join(ROOT, "data", "cleaned")
    encoded_dict, tokenizer = build_encoded_datasets(
        data_dir=data_dir,
        model_name="indolem/indobertweet-base-uncased",
        max_length=TRAIN_CONFIG["max_length"],
        batch_size=16,
        splits={
            "train": "train_clean.csv",
            "validation": "validation_clean.csv",
            "test": "test_clean.csv"
        }
    )

    train_dataset = encoded_dict["train"]
    eval_dataset  = encoded_dict["validation"]
    test_dataset  = encoded_dict["test"]

    output_dir = os.path.join(MODELS, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model and applying layer freezing...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "indolem/indobertweet-base-uncased",
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2}
    )
    apply_layer_freezing(model)

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=LOGS,
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        evaluation_strategy=TRAIN_CONFIG["evaluation_strategy"],
        save_strategy=TRAIN_CONFIG["save_strategy"],
        load_best_model_at_end=TRAIN_CONFIG["load_best_model_at_end"],
        metric_for_best_model=TRAIN_CONFIG["metric_for_best_model"],
        greater_is_better=TRAIN_CONFIG["greater_is_better"],
        dataloader_num_workers=TRAIN_CONFIG["dataloader_num_workers"],
        fp16=TRAIN_CONFIG["fp16"],
        seed=TRAIN_CONFIG["seed"],
        logging_strategy=TRAIN_CONFIG.get("logging_strategy", "steps"),
        logging_steps=TRAIN_CONFIG.get("logging_steps", 25),
        save_total_limit=TRAIN_CONFIG.get("save_total_limit", 1),
        save_safetensors=TRAIN_CONFIG.get("save_safetensors", True),
        report_to=TRAIN_CONFIG.get("report_to", "none"),
        remove_unused_columns=TRAIN_CONFIG.get("remove_unused_columns", False),
    )

    trainer = WeightedTrainer(
        class_weights=ADJUSTED_WEIGHTS,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer.underlying,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    last_checkpoint = get_last_checkpoint(output_dir)
    resume = last_checkpoint if last_checkpoint else None
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
    else:
        logger.info("Starting from scratch.")

    try:
        trainer.train(resume_from_checkpoint=resume)
        best_model_path = os.path.join(MODELS, "best_model")
        trainer.save_model(best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
    except KeyboardInterrupt:
        logger.warning("Training interrupted! Checkpoints preserved.")
        sys.exit(1)

    end_time = time.time()
    total_runtime = end_time - start_time

    pd.DataFrame(trainer.state.log_history).to_csv(
        os.path.join(RESULTS, "neutral_experiment_history.csv"), index=False
    )

    logger.info("Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    true_labels = predictions.label_ids
    probs = scipy_softmax(logits, axis=1)
    pred_labels = np.argmax(probs, axis=1)

    report_dict = classification_report(
        true_labels, pred_labels,
        target_names=["negative", "neutral", "positive"],
        output_dict=True, digits=4
    )

    exp_metrics = {
        "test_accuracy":   predictions.metrics.get("test_accuracy", 0),
        "test_macro_f1":   predictions.metrics.get("test_f1_macro", 0),
        "test_weighted_f1": predictions.metrics.get("test_f1_weighted", 0),
        "test_loss":       predictions.metrics.get("test_loss", 0),
        "neutral_precision": report_dict["neutral"]["precision"],
        "neutral_recall":    report_dict["neutral"]["recall"],
        "neutral_f1":        report_dict["neutral"]["f1-score"],
        "negative_f1":       report_dict["negative"]["f1-score"],
        "positive_f1":       report_dict["positive"]["f1-score"],
    }

    if os.path.exists(BASELINE_METRICS_PATH):
        with open(BASELINE_METRICS_PATH) as f:
            baseline = json.load(f)
        exp_metrics["delta_macro_f1"]    = exp_metrics["test_macro_f1"] - baseline.get("test_f1_macro", 0)
        exp_metrics["delta_accuracy"]    = exp_metrics["test_accuracy"] - baseline.get("test_accuracy", 0)
        exp_metrics["delta_weighted_f1"] = exp_metrics["test_weighted_f1"] - baseline.get("test_f1_weighted", 0)
        logger.info(f"Delta Macro F1 vs Baseline: {exp_metrics['delta_macro_f1']:+.4f}")

    exp_metrics["experiment_metadata"] = {
        **EXPERIMENT_META,
        "seed": TRAIN_CONFIG["seed"],
        "total_runtime_seconds": total_runtime,
        "total_runtime_formatted": str(datetime.timedelta(seconds=int(total_runtime))),
        "epochs_completed": trainer.state.epoch,
        "best_epoch": trainer.state.best_model_checkpoint,
        "early_stopping_triggered": trainer.state.epoch < TRAIN_CONFIG["num_train_epochs"]
    }

    with open(os.path.join(RESULTS, "neutral_experiment_metrics.json"), "w") as f:
        json.dump(exp_metrics, f, indent=2)

    logger.info(f"Test Accuracy : {exp_metrics['test_accuracy']:.4f}")
    logger.info(f"Test Macro F1 : {exp_metrics['test_macro_f1']:.4f}")
    logger.info(f"Neutral F1    : {exp_metrics['neutral_f1']:.4f}")
    logger.info("PHASE 3C-2 COMPLETE.")


if __name__ == "__main__":
    main()
