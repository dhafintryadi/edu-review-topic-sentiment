"""
src/training_config.py
Centralized configuration and hyperparameter storage for Phase 2D and 3.
Includes explicit device and environment logging.
"""

import os
import torch
import psutil
import logging

logger = logging.getLogger(__name__)

# Basic Training Strategy Configuration
TRAIN_CONFIG = {
    "max_length": 96,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "num_train_epochs": 5, # Updated to 5 for full run
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_strategy": "steps",
    "logging_steps": 25,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1_macro",
    "greater_is_better": True,
    "dataloader_num_workers": 0, # Windows safe
    "fp16": False, # CPU safe
    "seed": 42,
    "save_safetensors": True,
    "report_to": "none",
    "remove_unused_columns": False
}

# Derived or fixed class weights from Phase 2C
CLASS_WEIGHTS = [2.1538, 4.6889, 0.4306]

def log_device_info():
    """Logs explicit hardware and environment information."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_cores = os.cpu_count()
    mem_info = psutil.virtual_memory()
    mem_total_gb = mem_info.total / (1024 ** 3)
    mem_avail_gb = mem_info.available / (1024 ** 3)
    
    logger.info("=" * 40)
    logger.info("DEVICE & ENVIRONMENT INFO")
    logger.info("=" * 40)
    logger.info(f"Torch Version : {torch.__version__}")
    logger.info(f"Target Device : {device.upper()}")
    logger.info(f"CPU Cores     : {cpu_cores}")
    logger.info(f"Total Memory  : {mem_total_gb:.2f} GB")
    logger.info(f"Avail Memory  : {mem_avail_gb:.2f} GB")
    logger.info("=" * 40)
    
    return device
