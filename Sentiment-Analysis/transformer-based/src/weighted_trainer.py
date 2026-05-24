"""
src/weighted_trainer.py
Custom HuggingFace Trainer utilizing Weighted Cross-Entropy Loss to combat severe class imbalance.
"""

import torch
import logging
from torch import nn
from transformers import Trainer

logger = logging.getLogger(__name__)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
        if self.class_weights is not None:
            # Move weights to the appropriate device (CPU/GPU)
            self.weight_tensor = torch.tensor(
                self.class_weights, 
                dtype=torch.float32, 
                device=self.args.device
            )
            self.loss_fct = nn.CrossEntropyLoss(weight=self.weight_tensor)
            logger.info(f"WeightedTrainer initialized with class weights: {self.class_weights}")
        else:
            self.loss_fct = nn.CrossEntropyLoss()
            logger.warning("WeightedTrainer initialized WITHOUT class weights. Using standard CrossEntropyLoss.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Overrides the standard loss computation to apply class weights.
        Compatible with HuggingFace Trainer API >= 4.x
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Calculate weighted loss
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        # Added num_items_in_batch for compatibility with newer transformers versions if passed
        # Usually not strictly needed for basic usage, but good practice if available.
        
        return (loss, outputs) if return_outputs else loss
