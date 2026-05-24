"""
src/freeze_layers.py
Implements partial layer freezing for CPU-efficient fine-tuning.
Freezes embeddings and lower encoder layers.
"""

import logging
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

def apply_layer_freezing(model: PreTrainedModel) -> dict:
    """
    Freezes the embedding layer and encoder layers 0-7.
    Leaves the top layers (8-11) and the classifier trainable.
    """
    logger.info("Applying CPU-optimized layer freezing strategy...")
    
    # Layers to completely freeze
    frozen_prefixes = [
        "bert.embeddings",
    ] + [f"bert.encoder.layer.{i}." for i in range(8)]
    
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen_prefixes):
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Calculate statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_ratio = (trainable_params / total_params) * 100
    
    stats = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_ratio_percent": round(trainable_ratio, 2)
    }
    
    logger.info("=" * 40)
    logger.info("LAYER FREEZING SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Total Parameters     : {total_params:,}")
    logger.info(f"Frozen Parameters    : {frozen_params:,}")
    logger.info(f"Trainable Parameters : {trainable_params:,}")
    logger.info(f"Trainable Ratio      : {stats['trainable_ratio_percent']}%")
    logger.info("=" * 40)
    
    return stats
