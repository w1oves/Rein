
from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
from typing import OrderedDict, List

class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

@MODELS.register_module(name='ReinPPReins')
class Reins(nn.Module):

    def __init__(self, num_layers: int, embed_dims: int, patch_size: int, query_dims: int=256, token_length: int=100, use_softmax: bool=True, link_token_to_query: bool=True, scale_init: float=0.001, zero_mlp_delta_f: bool=False, delete_first: bool=True, link_mode: str='all', mlp_ratio=2, act: str='none', parallel: bool=False) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.delete_first = delete_first
        self.link_mode = link_mode
        self.act = act
        self.mlp_ratio = mlp_ratio
        self.parallel = parallel
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(torch.empty([self.num_layers, self.token_length, self.embed_dims]))
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        if self.act == 'gelu':
            self.mlp_delta_f = nn.Sequential(nn.LayerNorm(self.embed_dims), nn.Linear(self.embed_dims, int(self.embed_dims * self.mlp_ratio)), QuickGELU(), nn.Linear(int(self.embed_dims * self.mlp_ratio), self.embed_dims))
            for layer in self.mlp_delta_f:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    print(f'Init {layer} using kaiming uniform')
        elif self.act == 'relu':
            self.mlp_delta_f = nn.Sequential(nn.LayerNorm(self.embed_dims), nn.Linear(self.embed_dims, int(self.embed_dims * self.mlp_ratio)), nn.ReLU(), nn.Linear(int(self.embed_dims * self.mlp_ratio), self.embed_dims))
            for layer in self.mlp_delta_f:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                    print(f'Init {layer} using kaiming uniform')
        else:
            self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
            nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
            print(f'Init {self.mlp_delta_f} using kaiming uniform')
        val = math.sqrt(6.0 / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims))
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        if self.link_mode == 'all':
            self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        else:
            self.merge = nn.Linear(self.query_dims, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

    def return_auto(self, feats):
        outs = dict()
        outs['feature'] = feats
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            if self.link_mode == 'external':
                querys = tokens
            else:
                if self.link_mode == 'max':
                    tokens = F.max_pool1d(tokens, kernel_size=self.num_layers)
                elif self.link_mode == 'avg':
                    tokens = F.avg_pool1d(tokens, kernel_size=self.num_layers)
                elif self.link_mode == 'last':
                    tokens = tokens[:, :, -1].unsqueeze(-1)
                else:
                    tokens = torch.cat([F.max_pool1d(tokens, kernel_size=self.num_layers), F.avg_pool1d(tokens, kernel_size=self.num_layers), tokens[:, :, -1].unsqueeze(-1)], dim=-1)
                querys = self.merge(tokens.flatten(-2, -1))
            outs['query'] = querys
            outs['last'] = self.get_tokens(-1)[-1]
        return outs

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(self, inp_feats: Tensor, out_feats: Tensor, layer: int, batch_first=False, has_cls_token=True) -> Tensor:
        if batch_first:
            inp_feats = inp_feats.permute(1, 0, 2)
            out_feats = out_feats.permute(1, 0, 2)
        if has_cls_token:
            _, inp_feats = torch.tensor_split(inp_feats, [1], dim=0)
            out_cls_token, out_feats = torch.tensor_split(out_feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        if self.parallel:
            x = inp_feats
        else:
            x = out_feats
        delta_feat = self.forward_delta_feat(x, tokens, layer)
        delta_feat = delta_feat * self.scale
        feats = out_feats + delta_feat
        if has_cls_token:
            feats = torch.cat([out_cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum('nbc,mc->nbm', feats, tokens)
        if self.use_softmax:
            attn = attn * self.embed_dims ** (-0.5)
            attn = F.softmax(attn, dim=-1)
        if self.delete_first:
            delta_f = torch.einsum('nbm,mc->nbc', attn[:, :, 1:], self.mlp_token2feat(tokens[1:, :]))
        else:
            delta_f = torch.einsum('nbm,mc->nbc', attn, self.mlp_token2feat(tokens))
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f

@MODELS.register_module(name='ReinPPLoRAReins')
class LoRAReins(Reins):

    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(torch.empty([self.num_layers, self.token_length, self.lora_dim]))
        self.learnable_tokens_b = nn.Parameter(torch.empty([self.num_layers, self.lora_dim, self.embed_dims]))
        val = math.sqrt(6.0 / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + (self.embed_dims * self.lora_dim) ** 0.5))
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]

@MODELS.register_module(name='ReinPPMultiHeadReins')
class MultiHeadReins(LoRAReins):

    def __init__(self, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dims = self.embed_dims // num_heads

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        N, B, C = feats.shape
        mh_feats = feats.view(N, B, self.num_heads, self.head_dims)
        M, C = tokens.shape
        mh_tokens = tokens.view(M, self.num_heads, self.head_dims)
        attn = torch.einsum('nbhc,mhc->nbhm', mh_feats, mh_tokens)
        if self.use_softmax:
            attn = attn * self.head_dims ** (-0.5)
            attn = F.softmax(attn, dim=-1)
        if self.delete_first:
            delta_f = torch.einsum('nbhm,mhc->nbhc', attn[:, :, :, 1:], self.mlp_token2feat(tokens[1:, :]).view(M - 1, self.num_heads, self.head_dims))
        else:
            delta_f = torch.einsum('nbhm,mhc->nbhc', attn, self.mlp_token2feat(tokens).view(M, self.num_heads, self.head_dims))
        delta_f = self.mlp_delta_f(delta_f.reshape(N, B, -1) + feats)
        return delta_f

class DimTransfers(nn.Module):

    def __init__(self, stages: List[int], inp_dims: List[int], out_dims: List[int]):
        super().__init__()
        self.stages = stages
        self.out_dims = out_dims
        self.inp_dims = inp_dims
        mlps = []
        for inp_dim, out_dim in zip(self.inp_dims, self.out_dims):
            if inp_dim == out_dim:
                mlps.append(nn.Identity())
            else:
                mlps.append(nn.Linear(inp_dim, out_dim))
        self.mlps = nn.ModuleList(mlps)

    def get_inp_stage(self, idx_layer):
        for idx_stage, num_layers in enumerate(self.stages):
            if idx_layer < num_layers:
                return idx_stage
            else:
                idx_layer -= num_layers
        raise NotImplementedError

    def forward(self, feats, idx_layer):
        idx_stage = self.get_inp_stage(idx_layer)
        return self.mlps[idx_stage](feats)
