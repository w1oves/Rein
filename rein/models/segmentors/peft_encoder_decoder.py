from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
import torch.nn.functional as F
from mmengine.logging import MMLogger
from collections import OrderedDict


@MODELS.register_module()
class PEFTEncoderDecoder(EncoderDecoder):
    def state_dict(self):
        state = super().state_dict()
        new_state = OrderedDict()
        for k, v in state:
            if "backbone" in k and "rein" not in k:
                continue
            new_state[k] = v
        return new_state
