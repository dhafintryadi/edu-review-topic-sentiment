"""
src/trainer_factory.py
Orchestrates the creation of the Trainer instance.
Combines configuration, layer freezing, metrics, and weighted loss.
"""

import os
import logging
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback
)

from training_config import TRAIN_CONFIG, CLASS_WEIGHTS
from freeze_layers import apply_layer_freezing
from metrics import compute_metrics
from weighted_trainer import WeightedTrainer

logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

def build_trainer(
    train_dataset,
    eval_dataset,
    tokenizer,
    model_name: str = "indolem/indobertweet-base-uncased",
    output_dir: str = os.path.join(ROOT, "models", "checkpoints"),
    logging_dir: str = os.path.join(ROOT, "logs", "training_logs"),
    class_weights: list = CLASS_WEIGHTS,
    training_config: dict = TRAIN_CONFIG
) -> WeightedTrainer:
    """
    Constructs and configures the WeightedTrainer.
    """
    logger.info(f"Loading Model: {model_name} (num_labels=3)")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2}
    )
    
    # 1. Apply Layer Freezing for CPU optimization
    apply_layer_freezing(model)
    
    # 2. Configure Training Arguments
    logger.info("Setting up TrainingArguments...")
    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        
        # Batching & accumulation
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        
        # Learning Rate & Epochs
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        weight_decay=training_config["weight_decay"],
        
        # Evaluation & Saving Strategy
        evaluation_strategy=training_config["evaluation_strategy"],
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        
        # Hardware optimization
        dataloader_num_workers=training_config["dataloader_num_workers"],
        fp16=training_config["fp16"],
        
        # Others
        seed=training_config["seed"],
        logging_strategy=training_config.get("logging_strategy", "steps"),
        logging_steps=training_config.get("logging_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 1),
        warmup_ratio=training_config.get("warmup_ratio", 0.0),
        save_safetensors=training_config.get("save_safetensors", False),
        report_to=training_config.get("report_to", "all"),
        remove_unused_columns=training_config.get("remove_unused_columns", True)
    )
    
    # 3. Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=2)
    ]
    
    # 4. Instantiate WeightedTrainer
    logger.info("Instantiating WeightedTrainer...")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    return trainer
