"""
src/run_phase3a.py
Executes the Phase 3A Controlled Baseline Fine-Tuning.
Includes resume support, robust logging, and runtime tracking.
"""

import os, sys, time, json, logging, datetime
import pandas as pd
from transformers.trainer_utils import get_last_checkpoint

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

LOGS = os.path.join(ROOT, "logs", "training_logs")
os.makedirs(LOGS, exist_ok=True)
LOG_FILE = os.path.join(LOGS, "phase3a.log")

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
from trainer_factory import build_trainer
from dataset_encoder import build_encoded_datasets

RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models", "final_baseline")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

def main():
    logger.info("=" * 60)
    logger.info("PHASE 3A - CONTROLLED BASELINE FINE-TUNING")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Load dataset
    data_dir = os.path.join(ROOT, "data", "cleaned")
    encoded_dict, tokenizer = build_encoded_datasets(
        data_dir=data_dir,
        model_name="indolem/indobertweet-base-uncased",
        max_length=TRAIN_CONFIG["max_length"],
        batch_size=16,
        splits={"train": "train_clean.csv", "validation": "validation_clean.csv", "test": "test_clean.csv"}
    )
    
    train_dataset = encoded_dict["train"]
    eval_dataset = encoded_dict["validation"]
    
    output_dir = os.path.join(MODELS, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    trainer = build_trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer.underlying,
        output_dir=output_dir,
        logging_dir=LOGS,
        training_config=TRAIN_CONFIG
    )
    
    # Checkpoint recovery
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is not None:
        logger.info(f"Detected existing checkpoint at {last_checkpoint}. Resuming training...")
        resume_from_checkpoint = last_checkpoint
    else:
        logger.info("No existing checkpoint found. Starting training from scratch.")
        resume_from_checkpoint = None

    # Train
    logger.info("Starting Trainer...")
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save Best Model explicitly
        best_model_path = os.path.join(MODELS, "best_model")
        trainer.save_model(best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
        
    except KeyboardInterrupt:
        logger.warning("Training manually interrupted! Model state preserved in checkpoints.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

    end_time = time.time()
    total_runtime = end_time - start_time
    
    # Extract History
    history = trainer.state.log_history
    df_history = pd.DataFrame(history)
    history_path = os.path.join(RESULTS, "training_history.csv")
    df_history.to_csv(history_path, index=False)
    logger.info(f"Training history saved to {history_path}")
    
    # Runtime tracking
    runtime_summary = {
        "total_runtime_seconds": total_runtime,
        "total_runtime_formatted": str(datetime.timedelta(seconds=int(total_runtime))),
        "epochs_completed": trainer.state.epoch,
        "global_step": trainer.state.global_step
    }
    with open(os.path.join(RESULTS, "runtime_summary.json"), "w") as f:
        json.dump(runtime_summary, f, indent=2)
        
    logger.info("PHASE 3A TRAINING COMPLETE.")

if __name__ == "__main__":
    main()
