"""
src/focal_trainer.py
Custom HuggingFace Trainer using Focal Loss instead of Weighted CrossEntropy.
Extends WeightedTrainer by overriding only the loss function.
All other infrastructure (layer freezing, metrics, callbacks) remains identical.
"""

import torch
import logging
from weighted_trainer import WeightedTrainer
from focal_loss import FocalLoss

logger = logging.getLogger(__name__)

FOCAL_GAMMA = 2.0  # Standard Focal Loss gamma from Lin et al. 2017


class FocalTrainer(WeightedTrainer):
    """
    Identical to WeightedTrainer except uses Focal Loss.
    Variable changed: loss_function only.
    """

    def __init__(self, class_weights=None, gamma: float = FOCAL_GAMMA, *args, **kwargs):
        super().__init__(class_weights=class_weights, *args, **kwargs)
        self.gamma = gamma

        if class_weights is not None:
            alpha = torch.tensor(class_weights, dtype=torch.float32)
        else:
            alpha = None

        self.loss_fct = FocalLoss(gamma=self.gamma, alpha=alpha, reduction="mean")
        logger.info(
            f"FocalTrainer initialized | gamma={self.gamma} | "
            f"alpha={class_weights}"
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = self.loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss
