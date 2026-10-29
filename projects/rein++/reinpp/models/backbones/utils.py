from torch.nn import functional as F

def get_pyramid_features(feats, ratios=(4, 2, 1, 0.5)):
    if len(feats) != len(ratios):
        raise ValueError("Expected one feature per pyramid level")
    for level, ratio in enumerate(ratios):
        if ratio != 1:
            feats[level] = F.interpolate(feats[level], scale_factor=ratio, mode="bilinear", align_corners=False)
    return feats
