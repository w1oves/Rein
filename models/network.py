import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import get_dinov2, get_dinov2_rein
from .data_preprocessor import DataPreprocessor
from .head import get_head
from utils.tools import load_backbone
def count(d):
    return sum(v.numel() for v in d.values())

class Network(nn.Module):
    def __init__(self, backbone_path, rein_path, head_path, use_rein=True) -> None:
        super().__init__()
        self.data_preprocessor = DataPreprocessor(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            size=(512, 512),
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=255,
        )
        if use_rein:
            self.backbone = get_dinov2_rein()
        else:
            self.backbone = get_dinov2()
        self.head = get_head(use_rein)
        backbone_state=load_backbone(backbone_path)
        print(f'Parameters of backbone: {count(backbone_state)/1e6:.2f}M')
        head_state=torch.load(head_path)
        print(f'Parameters of head: {count(head_state)/1e6:.2f}M')
        if use_rein:
            self.backbone.load_state_dict(backbone_state, strict=False)
            rein_state = torch.load(rein_path)
            print(f'Parameters of rein: {count(rein_state)/1e6:.2f}M')
            self.backbone.rein.load_state_dict(rein_state)
            self.backbone.rein.pre_compute()
            self.head.load_state_dict(head_state, strict=False)
            self.head.link(self.backbone.rein.link_to_querys())
        else:
            self.backbone.load_state_dict(backbone_state)
            self.head.load_state_dict(head_state, strict=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head.predict(x)
        return x

    def inference(self, inputs):
        inputs = self.data_preprocessor(inputs)
        h_stride, w_stride = 341, 341
        h_crop, w_crop = 512, 512
        batch_size, _, h_img, w_img = inputs.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, 19, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        crop_imgs = []
        pads = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                crop_imgs.append(crop_img)
                count_mat[:, :, y1:y2, x1:x2] += 1
                pads.append(
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    )
                )
        crop_seg_logits = self(torch.cat(crop_imgs, dim=0))
        crop_seg_logits = [
            F.pad(
                logit,
                pad,
            )
            for logit, pad in zip(crop_seg_logits, pads)
        ]
        preds = sum(crop_seg_logits)
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        return seg_logits
