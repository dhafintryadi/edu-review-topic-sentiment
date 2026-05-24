"""
src/visualize_phase3a.py
Generates publication-quality visualizations for Phase 3A:
- Loss curves
- Macro F1 progression
- Confusion Matrix (normalized & annotated)
"""

import os, sys, json, logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

RESULTS = os.path.join(ROOT, "results")
VIS_DIR = os.path.join(ROOT, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Styling
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})
sns.set_theme(style="whitegrid")

def plot_training_curves():
    history_path = os.path.join(RESULTS, "training_history.csv")
    if not os.path.exists(history_path):
        logger.warning(f"History file not found: {history_path}")
        return

    df = pd.read_csv(history_path)
    
    # Extract training and validation logs
    # Trainer logs 'loss' for training steps and 'eval_loss'/'eval_f1_macro' for eval steps.
    # Usually, they are on different rows but share the 'epoch' column.
    
    # Let's group by epoch. For eval metrics, we take the last value per epoch.
    df_eval = df.dropna(subset=['eval_loss']).copy()
    
    if df_eval.empty:
        logger.warning("No evaluation metrics found in history.")
        return
        
    df_train = df.dropna(subset=['loss']).copy()
    if not df_train.empty:
        # Group train loss by epoch (mean)
        train_loss = df_train.groupby('epoch')['loss'].mean().reset_index()
    else:
        # If no explicit train loss is logged per epoch, fallback
        train_loss = pd.DataFrame({'epoch': df_eval['epoch'], 'loss': np.nan})

    # Combine for easy plotting
    epochs = df_eval['epoch'].values
    val_loss = df_eval['eval_loss'].values
    val_f1 = df_eval['eval_f1_macro'].values
    
    # Best epoch logic
    best_idx = np.argmax(val_f1)
    best_epoch = epochs[best_idx]
    
    # 1. Plot Loss Curve
    plt.figure(figsize=(10, 6))
    if not train_loss.empty:
        plt.plot(train_loss['epoch'], train_loss['loss'], label='Training Loss', marker='o', linewidth=2, color='#1f77b4')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s', linewidth=2, color='#ff7f0e')
    
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'training_loss_curve.png'), dpi=300)
    plt.close()
    logger.info("Saved training_loss_curve.png")
    
    # 2. Plot Macro F1 Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1, label='Validation Macro F1', marker='D', linewidth=2, color='#2ca02c')
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    
    plt.title('Validation Macro F1 Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'macro_f1_curve.png'), dpi=300)
    plt.close()
    logger.info("Saved macro_f1_curve.png")


def plot_confusion_matrix():
    preds_path = os.path.join(RESULTS, "test_predictions.csv")
    if not os.path.exists(preds_path):
        logger.warning(f"Predictions file not found: {preds_path}")
        return

    df = pd.read_csv(preds_path)
    labels = ["negative", "neutral", "positive"]
    
    cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=labels)
    cm_norm = confusion_matrix(df['true_label'], df['predicted_label'], labels=labels, normalize='true')
    
    # Create annotations: Count (Percentage%)
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", cbar=True, 
                xticklabels=[l.capitalize() for l in labels],
                yticklabels=[l.capitalize() for l in labels],
                annot_kws={"size": 12})
    
    plt.title('Confusion Matrix (Test Set)', pad=20)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()
    logger.info("Saved confusion_matrix.png")


def main():
    logger.info("=" * 60)
    logger.info("PHASE 3A - VISUALIZATION")
    logger.info("=" * 60)
    
    plot_training_curves()
    plot_confusion_matrix()
    
    logger.info("VISUALIZATION COMPLETE.")

if __name__ == "__main__":
    main()
