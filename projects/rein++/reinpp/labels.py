"""Evaluation label conversions used by the original experiments.

Mapillary grouping derives from openseg-group/openseg.pytorch (MIT).
See licenses/OpenSeg-MIT.txt and docs/third-party-notices.md.
"""
import numpy as np

CITYSCAPES_LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25,
                       26, 27, 28, 31, 32, 33)
CLASS_NAMES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'trafficlight', 'trafficsign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train',
               'motorcycle', 'bicycle')


def mapillary_lookup(labels):
    """Map the supplied Vistas v1.2 metadata to the original 19-class protocol."""
    direct = {name: i for i, name in enumerate(CLASS_NAMES)}
    groups = [
        (('trafficsign', 'front', 'back'), 'trafficsign'),
        (('onrail',), 'train'),
        (('cyclist', 'rider'), 'rider'),
        (('pole', 'streetlight'), 'pole'),
        (('curb', 'pedestrianarea'), 'sidewalk'),
        (('crosswalkplain', 'parking', 'bikelane', 'servicelane', 'lanemarking'), 'road'),
    ]
    lut = np.full(256, 255, dtype=np.uint8)
    if len(labels) != 66:
        raise ValueError('Expected the 66 labels in Mapillary Vistas v1.2 metadata')
    for index, label in enumerate(labels):
        name = label['readable'].lower().replace(' ', '').replace('-', '')
        target = direct.get(name)
        if target is None:
            target = next((direct[group] for words, group in groups
                           if any(word in name for word in words)), 255)
        lut[index] = target
    return lut


def cityscapes_lookup():
    lut = np.full(256, 255, dtype=np.uint8)
    lut[list(CITYSCAPES_LABEL_IDS)] = np.arange(19, dtype=np.uint8)
    return lut


def convert_mask(mask, lookup):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('Expected a two-dimensional integer label-ID mask, not RGB colors')
    if mask.size == 0 or mask.min() < 0 or mask.max() >= len(lookup):
        raise ValueError('Label ID is outside the conversion table')
    return lookup[mask]
