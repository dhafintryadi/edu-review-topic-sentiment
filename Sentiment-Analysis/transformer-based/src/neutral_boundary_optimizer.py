"""
src/neutral_boundary_optimizer.py
Phase 3C-2: Neutral Class Weight Down-weighting Experiment.
Strategy: Reduce Neutral class weight from 4.6889 → 2.5.
Rationale: Phase 3B shows the model over-predicts Negative for True Neutral reviews (46.2%).
Soft mitigation preserves all training samples and dataset integrity.
"""

from training_config import CLASS_WEIGHTS

# Phase 3A Baseline weights: [negative, neutral, positive]
# [2.1538, 4.6889, 0.4306]

NEUTRAL_EXPERIMENT_WEIGHTS = [
    CLASS_WEIGHTS[0],   # negative: unchanged (2.1538)
    2.5,                # neutral: reduced from 4.6889 → 2.5 (soft down-weighting)
    CLASS_WEIGHTS[2],   # positive: unchanged (0.4306)
]

EXPERIMENT_METADATA = {
    "experiment": "3C-2 Neutral Boundary Optimization",
    "strategy": "Neutral class weight down-weighting",
    "original_neutral_weight": CLASS_WEIGHTS[1],
    "adjusted_neutral_weight": 2.5,
    "rationale": (
        "Phase 3B: 46.2% of True Neutral samples are predicted as Negative. "
        "Down-weighting Neutral from 4.6889 to 2.5 reduces overcorrection "
        "toward the minority Neutral class while preserving all training samples."
    ),
    "original_weights": CLASS_WEIGHTS,
    "adjusted_weights": NEUTRAL_EXPERIMENT_WEIGHTS,
}


def get_adjusted_weights():
    return NEUTRAL_EXPERIMENT_WEIGHTS


def get_experiment_metadata():
    return EXPERIMENT_METADATA
