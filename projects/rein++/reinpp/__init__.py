"""Rein++ inference and evaluation. Model registration is explicit."""
__version__ = "0.1.0"

def register_modules():
    from .models.backbones import reins, dino_v3, reins_dinov3
    from .models.heads import rein_mask2former
    from .datasets import multi_domain_dataset, multi_domain_iou
