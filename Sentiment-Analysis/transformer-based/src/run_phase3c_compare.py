"""
src/run_phase3c_compare.py
Phase 3C Final Comparison — Aggregates all experiment metrics against baseline.
Generates side-by-side comparison table and visualization.
Produces a final production model recommendation.
"""

import os, sys, json, logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

RESULTS = os.path.join(ROOT, "results")
VIS_DIR = os.path.join(ROOT, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_json_safe(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    logger.info("=" * 60)
    logger.info("PHASE 3C — FINAL COMPARISON REPORT")
    logger.info("=" * 60)

    baseline    = load_json_safe(os.path.join(RESULTS, "final_metrics.json"))
    focal       = load_json_safe(os.path.join(RESULTS, "focal_loss_metrics.json"))
    neutral_exp = load_json_safe(os.path.join(RESULTS, "neutral_experiment_metrics.json"))
    calibration = load_json_safe(os.path.join(RESULTS, "calibration_metrics.json"))

    rows = []
    if baseline:
        rows.append({
            "Experiment": "3A Baseline (WeightedCE)",
            "Accuracy": baseline.get("test_accuracy", 0),
            "Macro F1": baseline.get("test_f1_macro", 0),
            "Weighted F1": baseline.get("test_f1_weighted", 0),
            "Test Loss": baseline.get("test_loss", 0),
            "Neutral F1": None,
            "Delta Macro F1": 0.0,
        })
    if focal:
        rows.append({
            "Experiment": "3C-1 Focal Loss (γ=2.0)",
            "Accuracy": focal.get("test_accuracy", 0),
            "Macro F1": focal.get("test_macro_f1", 0),
            "Weighted F1": focal.get("test_weighted_f1", 0),
            "Test Loss": focal.get("test_loss", 0),
            "Neutral F1": focal.get("neutral_f1"),
            "Delta Macro F1": focal.get("delta_macro_f1", 0),
        })
    if neutral_exp:
        rows.append({
            "Experiment": "3C-2 Neutral Down-weight",
            "Accuracy": neutral_exp.get("test_accuracy", 0),
            "Macro F1": neutral_exp.get("test_macro_f1", 0),
            "Weighted F1": neutral_exp.get("test_weighted_f1", 0),
            "Test Loss": neutral_exp.get("test_loss", 0),
            "Neutral F1": neutral_exp.get("neutral_f1"),
            "Delta Macro F1": neutral_exp.get("delta_macro_f1", 0),
        })

    if not rows:
        logger.warning("No experiment metrics found. Run experiments first.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "phase3c_comparison.csv"), index=False)
    logger.info("Saved phase3c_comparison.csv")

    # Visualization — grouped bar chart
    experiments = df["Experiment"].tolist()
    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, df["Accuracy"],    width, label="Accuracy",    color="#1f77b4")
    b2 = ax.bar(x,         df["Macro F1"],    width, label="Macro F1",    color="#2ca02c")
    b3 = ax.bar(x + width, df["Weighted F1"], width, label="Weighted F1", color="#ff7f0e")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Phase 3C: Experiment Comparison vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=10, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "baseline_vs_optimized_metrics.png"), dpi=300)
    plt.close()
    logger.info("Saved baseline_vs_optimized_metrics.png")

    # Build recommendation
    best_row = df.loc[df["Macro F1"].idxmax()]
    cal_note = ""
    if calibration:
        cal_note = (
            f"\n[CALIBRATION]\n"
            f"Temperature: {calibration.get('optimal_temperature', 'N/A'):.4f}\n"
            f"ECE Before:  {calibration.get('ece_before', 0):.4f}\n"
            f"ECE After:   {calibration.get('ece_after', 0):.4f}\n"
            f"Delta ECE:   -{calibration.get('delta_ece', 0):.4f}\n"
        )

    summary = f"""===========================================================
PHASE 3C — FINAL COMPARISON SUMMARY
===========================================================

[METRIC COMPARISON]
{df[['Experiment','Accuracy','Macro F1','Weighted F1','Delta Macro F1']].to_string(index=False)}
{cal_note}
[RECOMMENDATION]
Best experiment by Macro F1: {best_row['Experiment']}
  Accuracy   : {best_row['Accuracy']:.4f}
  Macro F1   : {best_row['Macro F1']:.4f}
  Delta F1   : {best_row['Delta Macro F1']:+.4f} vs baseline

All experiments used the same seed, dataset, and layer freezing strategy.
===========================================================
"""
    with open(os.path.join(RESULTS, "phase3c_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info(f"Best model by Macro F1: {best_row['Experiment']}")
    logger.info("PHASE 3C COMPARISON COMPLETE.")


if __name__ == "__main__":
    main()
