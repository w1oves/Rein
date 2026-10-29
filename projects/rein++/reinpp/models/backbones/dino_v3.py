# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import torch
import torch.nn.init
from torch import Tensor, nn
from mmengine.model import BaseModule
from mmseg.models.builder import BACKBONES
from .dinov3_layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from .utils import get_pyramid_features
logger = logging.getLogger('dinov3')
ffn_layer_dict = {'mlp': Mlp, 'swiglu': SwiGLUFFN, 'swiglu32': partial(SwiGLUFFN, align_to=32), 'swiglu64': partial(SwiGLUFFN, align_to=64), 'swiglu128': partial(SwiGLUFFN, align_to=128)}
norm_layer_dict = {'layernorm': partial(nn.LayerNorm, eps=1e-06), 'layernormbf16': partial(nn.LayerNorm, eps=1e-05), 'rmsnorm': RMSNorm}
dtype_dict = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}

@BACKBONES.register_module(name='ReinPPDINOv3')
class DINOv3(BaseModule):

    def __init__(self, *, img_size: int=224, patch_size: int=16, in_chans: int=3, pos_embed_rope_base: float=100.0, pos_embed_rope_min_period: float | None=None, pos_embed_rope_max_period: float | None=None, pos_embed_rope_normalize_coords: Literal['min', 'max', 'separate']='separate', pos_embed_rope_shift_coords: float | None=None, pos_embed_rope_jitter_coords: float | None=None, pos_embed_rope_rescale_coords: float | None=None, pos_embed_rope_dtype: str='bf16', embed_dim: int=768, depth: int=12, num_heads: int=12, ffn_ratio: float=4.0, qkv_bias: bool=True, drop_path_rate: float=0.0, layerscale_init: float | None=None, norm_layer: str='layernorm', ffn_layer: str='mlp', ffn_bias: bool=True, proj_bias: bool=True, n_storage_tokens: int=0, mask_k_bias: bool=False, untie_cls_and_patch_norms: bool=False, untie_global_and_local_cls_norm: bool=False, device: Any | None=None, out_indices: list=[5, 7, 9, 11], init_cfg: dict | None=None, **ignored_kwargs):
        super().__init__(init_cfg)
        if len(ignored_kwargs) > 0:
            logger.warning(f'Ignored kwargs: {ignored_kwargs}')
        del ignored_kwargs
        norm_layer_cls = norm_layer_dict[norm_layer]
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, flatten_embedding=False)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f'using base={pos_embed_rope_base} for rope new')
        logger.info(f'using min_period={pos_embed_rope_min_period} for rope new')
        logger.info(f'using max_period={pos_embed_rope_max_period} for rope new')
        logger.info(f'using normalize_coords={pos_embed_rope_normalize_coords} for rope new')
        logger.info(f'using shift_coords={pos_embed_rope_shift_coords} for rope new')
        logger.info(f'using rescale_coords={pos_embed_rope_rescale_coords} for rope new')
        logger.info(f'using jitter_coords={pos_embed_rope_jitter_coords} for rope new')
        logger.info(f'using dtype={pos_embed_rope_dtype} for rope new')
        self.rope_embed = RopePositionEmbedding(embed_dim=embed_dim, num_heads=num_heads, base=pos_embed_rope_base, min_period=pos_embed_rope_min_period, max_period=pos_embed_rope_max_period, normalize_coords=pos_embed_rope_normalize_coords, shift_coords=pos_embed_rope_shift_coords, jitter_coords=pos_embed_rope_jitter_coords, rescale_coords=pos_embed_rope_rescale_coords, dtype=dtype_dict[pos_embed_rope_dtype], device=device)
        logger.info(f'using {ffn_layer} layer as FFN')
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [SelfAttentionBlock(dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio_sequence[i], qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias, drop_path=drop_path_rate, norm_layer=norm_layer_cls, act_layer=nn.GELU, ffn_layer=ffn_layer_cls, init_values=layerscale_init, mask_k_bias=mask_k_bias, device=device) for i in range(depth)]
        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer_cls(embed_dim)
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(1, 0, cls_token.shape[-1], dtype=cls_token.dtype, device=cls_token.device)
        x = torch.cat([cls_token.expand(B, -1, -1), storage_tokens.expand(B, -1, -1), x], dim=1)
        return (x, (H, W))

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[List[Tensor]]:
        B, _, h, w = x_list[0].shape
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        outputs = []
        for idx_layer, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
            if idx_layer in self.out_indices:
                output = []
                for x_i, masks in zip(x, masks_list):
                    output.append(x_i[:, self.n_storage_tokens + 1:].permute(0, 2, 1).reshape(B, -1, h // self.patch_size, w // self.patch_size).contiguous())
                outputs.append(output)
        return outputs

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor]=None) -> List[Tensor] | List[List[Tensor]]:
        if isinstance(x, torch.Tensor):
            return [out[0] for out in self.forward_features_list([x], [masks])]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int=1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = ([], len(self.blocks))
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f'only {len(output)} / {len(blocks_to_take)} blocks found'
        return output

    def get_intermediate_layers(self, x: torch.Tensor, *, n: Union[int, Sequence]=1, reshape: bool=False, return_class_token: bool=False, return_extra_tokens: bool=False, norm: bool=True) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, :self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1:])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1:self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1:] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous() for out in outputs]
        if not return_class_token and (not return_extra_tokens):
            return tuple(outputs)
        elif return_class_token and (not return_extra_tokens):
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, **kwargs):
        ret = dict(feature=self.forward_features(*args, **kwargs))
        ret['feature'] = get_pyramid_features(ret['feature'])
        return ret
