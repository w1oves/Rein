import os
import os.path as osp
import torch
from torch import Tensor
from typing import OrderedDict
import torch.nn.functional as F


def load_backbone(path: str) -> OrderedDict[str, Tensor]:
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"{path} dont exist(absolute path: {osp.abspath(path)})"
        )
    weight = torch.load(path, map_location="cpu")
    weight["pos_embed"] = torch.cat(
        (
            weight["pos_embed"][:, :1, :],
            F.interpolate(
                weight["pos_embed"][:, 1:, :]
                .reshape(1, 37, 37, 1024)
                .permute(0, 3, 1, 2),
                size=(32, 32),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(1, 1024, 1024),
        ),
        dim=1,
    )
    weight["patch_embed.proj.weight"] = F.interpolate(
        weight["patch_embed.proj.weight"].float(),
        size=(16, 16),
        mode="bicubic",
        align_corners=False,
    )
    return weight