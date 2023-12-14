import numpy as np
import torch
import matplotlib.pyplot as plt

PALETTE = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]


def label2color(seg_label, is_logits=True) -> np.ndarray:
    if is_logits:
        seg_label = seg_label.argmax(dim=1)
    if isinstance(seg_label, torch.Tensor):
        seg_label = seg_label.squeeze(0).cpu().numpy()
    palette = np.array(PALETTE)
    color_seg = np.zeros((seg_label.shape[0], seg_label.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg_label == label, :] = color
    return color_seg
def show_result(im_path,seg_label,is_logits=True):
    color=label2color(seg_label,is_logits)
    im=plt.imread(im_path)
    plt.figure(1,figsize=[12,6],dpi=150)
    plt.subplot(121)    
    plt.imshow(im)
    plt.axis('off')
    plt.title('input image')
    plt.subplot(122)
    plt.imshow(im)
    plt.imshow(color)
    plt.axis('off')
    plt.title('segmentation')
    plt.show()