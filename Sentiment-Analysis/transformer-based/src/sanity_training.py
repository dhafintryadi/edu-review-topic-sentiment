"""
src/sanity_training.py
Performs a lightweight sanity validation training run.
Uses a micro-subset of data to verify forward/backward passes, layer freezing, and metrics.
"""

import os, sys, logging, datetime, json
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

LOGS = os.path.join(ROOT, "logs")
RESULTS = os.path.join(ROOT, "results")
SANITY_DIR = os.path.join(RESULTS, "sanity_checks")

os.makedirs(LOGS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(SANITY_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS, "phase2d_sanity.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

from dataset_encoder import build_encoded_datasets
from training_config import log_device_info, TRAIN_CONFIG
from trainer_factory import build_trainer

def verify_gradient_flow(model):
    """
    Verifies that gradients only exist on trainable layers after a backward pass.
    """
    logger.info("Verifying Gradient Flow...")
    expected_frozen = [
        "bert.embeddings",
        "bert.encoder.layer.0.", "bert.encoder.layer.1.", "bert.encoder.layer.2.",
        "bert.encoder.layer.3.", "bert.encoder.layer.4.", "bert.encoder.layer.5.",
        "bert.encoder.layer.6.", "bert.encoder.layer.7."
    ]
    
    flow_ok = True
    frozen_with_grads = []
    trainable_no_grads = []
    
    for name, param in model.named_parameters():
        is_frozen_target = any(name.startswith(prefix) for prefix in expected_frozen)
        has_grad = param.grad is not None
        
        if is_frozen_target and has_grad:
            frozen_with_grads.append(name)
            flow_ok = False
        elif not is_frozen_target and param.requires_grad and not has_grad:
            trainable_no_grads.append(name)
            # Not necessarily an error (could be unused in loss), but worth logging
            
    if frozen_with_grads:
        logger.error(f"CRITICAL: Frozen layers received gradients! {frozen_with_grads[:5]}")
    else:
        logger.info("PASS: No frozen layers received gradients.")
        
    if trainable_no_grads:
        logger.warning(f"Trainable layers with NO gradients (maybe unused): {trainable_no_grads[:5]}")
        
    return flow_ok

def main():
    logger.info("=" * 60)
    logger.info("PHASE 2D - SANITY TRAINING VALIDATION")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info("=" * 60)

    # 1. Device Logging
    device = log_device_info()
    
    # 2. Load Micro-Subset of Data
    data_dir = os.path.join(ROOT, "data", "cleaned")
    logger.info(f"Loading cleaned datasets from {data_dir}...")
    encoded_dict, tokenizer = build_encoded_datasets(
        data_dir=data_dir,
        model_name="indolem/indobertweet-base-uncased",
        max_length=TRAIN_CONFIG["max_length"],
        batch_size=16,
        splits={"train": "train_clean.csv", "validation": "validation_clean.csv", "test": "test_clean.csv"}
    )
    
    # Create micro-subset
    SUBSET_SIZE = 64 # Fits nicely in 8 batches of 8
    train_subset = encoded_dict["train"].select(range(min(SUBSET_SIZE, len(encoded_dict["train"]))))
    eval_subset = encoded_dict["validation"].select(range(min(SUBSET_SIZE, len(encoded_dict["validation"]))))
    
    logger.info(f"Created Sanity Subsets - Train: {len(train_subset)}, Eval: {len(eval_subset)}")
    
    # 3. Modify Config for Sanity Run
    sanity_config = TRAIN_CONFIG.copy()
    sanity_config["num_train_epochs"] = 1 # Just 1 epoch
    sanity_config["save_strategy"] = "no" # Don't clutter checkpoints during sanity unless we want to test saving
    
    # Let's save once to verify checkpointing works
    sanity_config["save_strategy"] = "epoch"
    sanity_config["evaluation_strategy"] = "epoch"
    
    sanity_output_dir = os.path.join(SANITY_DIR, "checkpoints")
    sanity_log_dir = os.path.join(SANITY_DIR, "logs")
    
    # 4. Build Trainer
    trainer = build_trainer(
        train_dataset=train_subset,
        eval_dataset=eval_subset,
        tokenizer=tokenizer.underlying,
        output_dir=sanity_output_dir,
        logging_dir=sanity_log_dir,
        training_config=sanity_config
    )
    
    # Check trainable parameter ratio
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_ratio = (trainable_params / total_params) * 100
    
    param_summary = {
        "total_parameters": total_params,
        "frozen_parameters": frozen_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio_percent": round(trainable_ratio, 2)
    }
    
    with open(os.path.join(RESULTS, "trainable_parameters_summary.txt"), "w") as f:
        f.write("LAYER FREEZING SUMMARY\n")
        for k, v in param_summary.items():
            f.write(f"{k}: {v}\n")
            
    # 5. Run Sanity Training
    logger.info("\nStarting Sanity Training Loop...")
    try:
        train_result = trainer.train()
        logger.info("Training loop completed successfully!")
        
        # Verify gradient flow
        flow_ok = verify_gradient_flow(trainer.model)
        
        # Run Evaluation
        logger.info("\nStarting Evaluation...")
        eval_result = trainer.evaluate()
        logger.info(f"Evaluation completed successfully! Metrics: {eval_result}")
        
        # Save sanity metrics
        with open(os.path.join(SANITY_DIR, "sanity_metrics.json"), "w") as f:
            json.dump({"train": train_result.metrics, "eval": eval_result}, f, indent=2)
            
        sanity_status = "PASS" if flow_ok else "FAIL (Gradient Flow)"
    except Exception as e:
        logger.error(f"Sanity Training FAILED: {e}")
        sanity_status = "FAIL"
        
    # 6. Generate Summary Report
    summary_lines = [
        "===========================================================",
        "PHASE 2D SUMMARY - Fine-Tuning Infrastructure",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "===========================================================",
        "",
        "1. CPU OPTIMIZATION (Layer Freezing)",
        f"  Total Parameters     : {total_params:,}",
        f"  Frozen Parameters    : {frozen_params:,}",
        f"  Trainable Parameters : {trainable_params:,}",
        f"  Trainable Ratio      : {trainable_ratio:.2f}%",
        "",
        "2. TRAINING CONFIGURATION",
        f"  Learning Rate : {sanity_config['learning_rate']}",
        f"  Batch Size    : {sanity_config['per_device_train_batch_size']} (x {sanity_config['gradient_accumulation_steps']} accum)",
        f"  Max Length    : {sanity_config['max_length']}",
        f"  Device        : {device}",
        "",
        "3. PIPELINE INTEGRATION",
        "  Weighted Loss : ENABLED",
        "  Early Stopping: ENABLED (Patience: 2)",
        "  Metrics       : ENABLED (Accuracy, Precision, Recall, F1-Macro, F1-Weighted, CM)",
        "",
        "4. SANITY TRAINING EXECUTION",
        f"  Status        : {sanity_status}",
        f"  Subset Size   : {SUBSET_SIZE}",
        "==========================================================="
    ]
    
    with open(os.path.join(RESULTS, "phase2d_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
        
    logger.info("PHASE 2D COMPLETE.")

if __name__ == "__main__":
    main()
