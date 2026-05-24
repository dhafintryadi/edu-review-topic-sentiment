"""
src/focal_loss.py
Implements Focal Loss for multi-class classification.
Focal Loss (Lin et al. 2017) reduces the loss contribution of easy-to-classify
examples, focusing training on hard misclassified samples.
Compatible with HuggingFace Trainer workflow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Args:
        gamma (float): Focusing parameter. gamma=0 is equivalent to standard CE.
                       gamma=2 is the standard value from the original paper.
        alpha (Tensor or None): Per-class weight tensor (same as class_weights in CrossEntropyLoss).
        reduction (str): 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # shape: (num_classes,) or None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes) raw logits from the model.
            labels: (batch_size,) ground-truth class indices.
        Returns:
            Scalar focal loss value.
        """
        num_classes = logits.size(-1)
        
        # Compute standard cross-entropy loss (per-sample, unreduced)
        log_probs = F.log_softmax(logits, dim=-1)       # (B, C)
        probs = torch.exp(log_probs)                     # (B, C)
        
        # Gather the probability of the true class: p_t
        labels_flat = labels.view(-1)                    # (B,)
        log_pt = log_probs.gather(dim=-1, index=labels_flat.unsqueeze(1)).squeeze(1)  # (B,)
        pt = probs.gather(dim=-1, index=labels_flat.unsqueeze(1)).squeeze(1)          # (B,)
        
        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - pt) ** self.gamma          # (B,)
        
        # Apply alpha weighting per class if provided
        if self.alpha is not None:
            at = self.alpha.to(logits.device)[labels_flat]   # (B,)
            focal_weight = at * focal_weight
        
        # Final focal loss per sample: -alpha * (1-pt)^gamma * log(pt)
        loss = -focal_weight * log_pt                    # (B,)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
