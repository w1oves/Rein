
from mmseg.models.builder import BACKBONES, MODELS
from .reins import Reins
from .dino_v3 import DINOv3
from .utils import get_pyramid_features
import torch

@BACKBONES.register_module(name='ReinPPReinsDINOv3')
class ReinsDINOv3(DINOv3):

    def __init__(self, reins_config=None, **kwargs):
        super().__init__(**kwargs)
        self.reins: Reins = MODELS.build(reins_config)

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        x, rope = self.prepare_tokens_with_masks(x, masks)
        outputs = []
        for idx_layer, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=rope[0], W=rope[1])
            else:
                rope_sincos = None
            last_inp = x
            x = blk(x, rope_sincos)
            other_tokens, x = (x[:, :self.n_storage_tokens + 1], x[:, self.n_storage_tokens + 1:])
            _, last_inp = (x[:, :self.n_storage_tokens + 1], x[:, self.n_storage_tokens + 1:])
            x = self.reins.forward(last_inp, x, idx_layer, batch_first=True, has_cls_token=False)
            x = torch.cat([other_tokens, x], dim=1)
            if idx_layer in self.out_indices:
                outputs.append(x[:, self.n_storage_tokens + 1:].permute(0, 2, 1).reshape(B, -1, h // self.patch_size, w // self.patch_size).contiguous())
        return self.reins.return_auto(outputs)

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        ret['feature'] = get_pyramid_features(ret['feature'])
        return ret
