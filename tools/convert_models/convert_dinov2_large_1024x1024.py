import torch
import os.path as osp
from collections import OrderedDict
from torch import Tensor
import torch.nn.functional as F
import sys


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
                size=(64, 64),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(1, 4096, 1024),
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


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <path> <save_path>")
        sys.exit(1)
    path = sys.argv[1]
    save_path = sys.argv[2]
    state = load_backbone(path)
    torch.save(state, save_path)


# Check if the script is run directly (and not imported)
if __name__ == "__main__":
    main()
