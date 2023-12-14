import torch
def intersect_and_union(
    pred_label: torch.tensor, label: torch.tensor, num_classes: int, ignore_index: int
):
    """Calculate Intersection and Union.

    Args:
        pred_label (torch.tensor): Prediction segmentation map
            or predict result filename. The shape is (H, W).
        label (torch.tensor): Ground truth segmentation map
            or label filename. The shape is (H, W).
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.

    Returns:
        torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
        torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
    """

    mask = label != ignore_index
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1
    ).cpu()
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1
    ).cpu()
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1
    ).cpu()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union
