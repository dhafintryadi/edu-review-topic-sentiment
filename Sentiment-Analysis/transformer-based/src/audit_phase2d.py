"""
src/audit_phase2d.py
Light technical audit of Phase 2D Fine-Tuning Infrastructure.
"""

import os, sys, json, logging, datetime, copy
import torch
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

RESULTS = os.path.join(ROOT, "results")
SANITY_DIR = os.path.join(RESULTS, "sanity_checks")
LOGS = os.path.join(ROOT, "logs")

LOG_FILE = os.path.join(LOGS, "phase2d_audit.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

from training_config import TRAIN_CONFIG, CLASS_WEIGHTS
from trainer_factory import build_trainer
from dataset_encoder import build_encoded_datasets

EXPECTED_CLASS_WEIGHTS = [2.1538, 4.6889, 0.4306]
EXPECTED_FROZEN_PREFIXES = [
    "bert.embeddings",
    "bert.encoder.layer.0.", "bert.encoder.layer.1.", "bert.encoder.layer.2.", "bert.encoder.layer.3.",
    "bert.encoder.layer.4.", "bert.encoder.layer.5.", "bert.encoder.layer.6.", "bert.encoder.layer.7."
]

def main():
    logger.info("=" * 60)
    logger.info("PHASE 2D - TECHNICAL INFRASTRUCTURE AUDIT")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info("=" * 60)

    audit_metrics = {}
    red_flags = []
    
    # Pre-setup: load tiny dataset
    data_dir = os.path.join(ROOT, "data", "cleaned")
    encoded_dict, tokenizer = build_encoded_datasets(
        data_dir=data_dir,
        model_name="indolem/indobertweet-base-uncased",
        max_length=TRAIN_CONFIG["max_length"],
        batch_size=16,
        splits={"train": "train_clean.csv", "validation": "validation_clean.csv", "test": "test_clean.csv"}
    )
    
    # 2 samples are enough for the audit logic
    train_subset = encoded_dict["train"].select(range(2))
    eval_subset = encoded_dict["validation"].select(range(2))

    trainer = build_trainer(
        train_dataset=train_subset,
        eval_dataset=eval_subset,
        tokenizer=tokenizer.underlying,
        output_dir=os.path.join(SANITY_DIR, "audit_checkpoints"),
        logging_dir=os.path.join(SANITY_DIR, "audit_logs"),
        training_config=TRAIN_CONFIG
    )
    model = trainer.model

    # 1 & 2. Verify Optimizer & Layer Freezing Integrity
    logger.info("\n[1 & 2] Verifying Layer Freezing and Optimizer Parameters...")
    frozen_param_count = 0
    trainable_param_count = 0
    unexpected_trainable = []
    unexpected_frozen = []

    for name, param in model.named_parameters():
        is_frozen_target = any(name.startswith(prefix) for prefix in EXPECTED_FROZEN_PREFIXES)
        
        if is_frozen_target:
            frozen_param_count += param.numel()
            if param.requires_grad:
                unexpected_trainable.append(name)
        else:
            trainable_param_count += param.numel()
            if not param.requires_grad:
                unexpected_frozen.append(name)

    logger.info(f"Trainable Params: {trainable_param_count:,} | Frozen Params: {frozen_param_count:,}")
    if unexpected_trainable:
        red_flags.append(f"Unexpected trainable parameters found in frozen layers: {unexpected_trainable[:3]}")
    if unexpected_frozen:
        logger.warning(f"Unexpected frozen parameters found in trainable layers: {unexpected_frozen[:3]}")
    
    # Initialize Optimizer
    trainer.create_optimizer()
    optimizer_param_count = sum(p.numel() for group in trainer.optimizer.param_groups for p in group['params'])
    logger.info(f"Optimizer holds {optimizer_param_count:,} parameters.")
    
    if optimizer_param_count != trainable_param_count:
        red_flags.append(f"Optimizer parameter mismatch! Optimizer: {optimizer_param_count}, Expected Trainable: {trainable_param_count}")

    audit_metrics["layer_freezing"] = {
        "trainable_param_count": trainable_param_count,
        "frozen_param_count": frozen_param_count,
        "optimizer_param_count": optimizer_param_count,
        "optimizer_matches_trainable": optimizer_param_count == trainable_param_count
    }

    # 3 & 4. Verify Gradient Flow and Weight Updates
    logger.info("\n[3 & 4] Verifying Gradient Flow and Weight Updates...")
    # Capture initial weights for trainable layers
    initial_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.clone().detach()

    # Create dummy batch safely via Trainer's dataloader
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    batch = {k: v.to(model.device) for k, v in batch.items()}

    # Forward + Backward
    model.train()
    trainer.optimizer.zero_grad()
    loss = trainer.compute_loss(model, batch)
    loss.backward()

    # Check Gradients
    gradients_ok = True
    frozen_with_grads = []
    trainable_without_grads = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad and param.grad is not None:
            frozen_with_grads.append(name)
            gradients_ok = False
        elif param.requires_grad and param.grad is None:
            trainable_without_grads.append(name)
            
    if frozen_with_grads:
        red_flags.append(f"Frozen layers received gradients! {frozen_with_grads[:3]}")
    if trainable_without_grads:
        logger.warning(f"Trainable layers with NO gradients: {trainable_without_grads[:3]}")
        
    logger.info(f"Gradients flow OK: {gradients_ok}")

    # Optimizer Step
    trainer.optimizer.step()

    # Check Weight Updates
    updates_ok = True
    layers_not_updated = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.equal(initial_weights[name], param.data):
                layers_not_updated.append(name)
                updates_ok = False

    if not updates_ok:
        red_flags.append(f"Trainable layers did NOT update after optimizer.step(): {layers_not_updated[:3]}")
    else:
        logger.info("All trainable layers successfully updated their weights.")

    audit_metrics["gradient_and_update"] = {
        "gradients_ok": gradients_ok,
        "updates_ok": updates_ok
    }

    # 5. Verify Weighted Loss Activation
    logger.info("\n[5] Verifying Weighted Loss Activation...")
    loss_fct = trainer.loss_fct
    if not isinstance(loss_fct, torch.nn.CrossEntropyLoss):
        red_flags.append("Trainer is NOT using CrossEntropyLoss.")
    
    if loss_fct.weight is None:
        red_flags.append("CrossEntropyLoss does NOT have class weights attached.")
    else:
        weights_match = np.allclose(loss_fct.weight.detach().cpu().numpy(), EXPECTED_CLASS_WEIGHTS, atol=1e-4)
        logger.info(f"Loss function weights match expected: {weights_match}")
        if not weights_match:
            red_flags.append("CrossEntropyLoss weights do not match expected class weights.")
            
    audit_metrics["weighted_loss"] = {
        "is_cross_entropy": isinstance(loss_fct, torch.nn.CrossEntropyLoss),
        "has_weights": loss_fct.weight is not None
    }

    # 6. Verify Metric Configuration
    logger.info("\n[6] Verifying Metric Configuration...")
    metric_best = trainer.args.metric_for_best_model
    logger.info(f"metric_for_best_model: {metric_best}")
    if metric_best != "f1_macro":
        red_flags.append(f"metric_for_best_model is '{metric_best}', expected 'f1_macro'.")

    # 7. Verify Callback Registration
    logger.info("\n[7] Verifying Callback Registration...")
    callbacks = trainer.callback_handler.callbacks
    early_stop_found = False
    for cb in callbacks:
        if type(cb).__name__ == "EarlyStoppingCallback":
            early_stop_found = True
            if cb.early_stopping_patience != 2:
                red_flags.append(f"EarlyStopping patience is {cb.early_stopping_patience}, expected 2.")
            break
            
    if not early_stop_found:
        red_flags.append("EarlyStoppingCallback not found in Trainer.")
        
    audit_metrics["callbacks"] = {
        "early_stopping_found": early_stop_found
    }

    # 8 & 9. Verify Checkpoint Integrity
    logger.info("\n[8 & 9] Verifying Checkpoint Integrity...")
    checkpoint_dirs = [d for d in os.listdir(os.path.join(SANITY_DIR, "checkpoints")) if d.startswith("checkpoint")]
    if not checkpoint_dirs:
        red_flags.append("No checkpoints found in sanity_checks/checkpoints!")
        checkpoint_ok = False
    else:
        latest_ckpt = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
        ckpt_path = os.path.join(SANITY_DIR, "checkpoints", latest_ckpt)
        logger.info(f"Found latest checkpoint: {ckpt_path}")
        
        required_files = ["config.json", "model.safetensors", "tokenizer.json", "training_args.bin"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(ckpt_path, f))]
        
        if missing_files:
            red_flags.append(f"Checkpoint missing required files: {missing_files}")
            checkpoint_ok = False
        else:
            checkpoint_ok = True
            logger.info("Checkpoint integrity verified. All required files exist.")
            
    audit_metrics["checkpoint_integrity"] = checkpoint_ok

    # Final Verdict
    logger.info("\n[FINAL VERDICT]")
    if len(red_flags) == 0:
        audit_verdict = "PASS"
    elif any("CRITICAL" in flag for flag in red_flags):
        audit_verdict = "FAIL"
    else:
        audit_verdict = "PASS WITH MINOR WARNINGS"

    logger.info(f"VERDICT: {audit_verdict}")
    for flag in red_flags:
        logger.error(f"RED FLAG: {flag}")

    audit_metrics["final_verdict"] = audit_verdict
    audit_metrics["red_flags"] = red_flags

    # Save metrics
    with open(os.path.join(RESULTS, "phase2d_audit_metrics.json"), "w") as f:
        json.dump(audit_metrics, f, indent=2)

    # Generate Report Text
    report_lines = [
        "===========================================================",
        "PHASE 2D TECHNICAL AUDIT REPORT",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "===========================================================",
        "",
        "1. OPTIMIZER & LAYER FREEZING",
        f"  Trainable Parameters: {trainable_param_count:,}",
        f"  Frozen Parameters: {frozen_param_count:,}",
        f"  Optimizer Registered Parameters: {optimizer_param_count:,}",
        f"  Mismatch: {not audit_metrics['layer_freezing']['optimizer_matches_trainable']}",
        "",
        "2. GRADIENT & UPDATE FLOW",
        f"  Frozen layers received gradients: {not audit_metrics['gradient_and_update']['gradients_ok']}",
        f"  Trainable layers updated weights: {audit_metrics['gradient_and_update']['updates_ok']}",
        "",
        "3. LOSS FUNCTION",
        f"  Using CrossEntropyLoss: {audit_metrics['weighted_loss']['is_cross_entropy']}",
        f"  Weights Attached: {audit_metrics['weighted_loss']['has_weights']}",
        "",
        "4. METRICS & CALLBACKS",
        f"  metric_for_best_model: {metric_best}",
        f"  EarlyStopping Attached: {audit_metrics['callbacks']['early_stopping_found']}",
        "",
        "5. CHECKPOINT INTEGRITY",
        f"  Checkpoint Reload Ready: {checkpoint_ok}",
        "",
        "===========================================================",
        f"FINAL VERDICT: {audit_verdict}",
        "==========================================================="
    ]
    
    if red_flags:
        report_lines.append("RED FLAGS DETECTED:")
        for flag in red_flags:
            report_lines.append(f" - {flag}")

    with open(os.path.join(RESULTS, "phase2d_audit_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Audit report saved to {os.path.join(RESULTS, 'phase2d_audit_report.txt')}")
    logger.info("PHASE 2D AUDIT COMPLETE.")

if __name__ == "__main__":
    main()
