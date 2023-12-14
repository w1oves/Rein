import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import mmengine
import os
import os.path as osp
from .metric import intersect_and_union


def load_im(im_path,scale=(1024,512)):
    with open(im_path, "rb") as f:
        im_bytes = f.read()
    im_np = np.frombuffer(im_bytes, np.uint8)
    im = cv2.imdecode(im_np, cv2.IMREAD_COLOR)
    im = cv2.resize(im, scale, dst=None, interpolation=cv2.INTER_LINEAR)
    if not im.flags.c_contiguous:
        im = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))
    else:
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).contiguous()
    return im


def load_ann(ann_path):
    with open(ann_path, "rb") as f:
        im_bytes = f.read()
    im_np = np.frombuffer(im_bytes, np.uint8)
    im = cv2.imdecode(im_np, cv2.IMREAD_UNCHANGED)
    return im


class Cityscapes(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        val_imdir = osp.join(data_dir, "leftImg8bit", "val")
        val_anndir = osp.join(data_dir, "gtFine", "val")
        assert osp.isdir(val_imdir), f"{val_imdir} dont exist"
        imgs = list(mmengine.scandir(val_imdir, suffix=".png", recursive=True))
        self.imgs = [osp.join(val_imdir, img) for img in imgs]
        self.anns = [
            osp.join(
                val_anndir, img.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
            )
            for img in imgs
        ]
        for ann in self.anns:
            assert osp.isfile(ann), f"{ann} dont exist"
        self.i = torch.zeros(size=[19])
        self.u = torch.zeros(size=[19])

    def __getitem__(self, index):
        return load_im(self.imgs[index]), load_ann(self.anns[index])

    def __len__(self):
        assert len(self.imgs) == len(self.anns)
        return len(self.imgs)

    def evaluate(self, pred: torch.Tensor, gt):
        pred = torch.nn.functional.interpolate(
            pred, gt.shape[-2:], mode="bilinear", align_corners=False
        )
        pred = pred.argmax(dim=1).view(gt.shape[-2:])
        gt = torch.from_numpy(gt)
        i, u = intersect_and_union(pred, gt, 19, 255)
        cur_miou = torch.nanmean(i/u).item()
        self.i = self.i + i
        self.u = self.u + u
        return cur_miou, torch.nanmean(self.i / self.u).item()


if __name__ == "__main__":
    Cityscapes("cityscapes")
