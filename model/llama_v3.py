# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import inspect

import math
from typing import List, Optional, Tuple, Union, Dict
from torch import Tensor
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, dense_to_sparse, unbatch_edge_index
from torch_geometric.nn import LayerNorm, HeteroLayerNorm, HANConv, HGTConv, GATConv  # RGATConv
# from torch_geometric.nn.pool import SAGPooling
from .GNN import RGATConv, SRGATConv, SAGPooling
from transformers import LlamaModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, \
    SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, \
    replace_return_docstrings
from transformers.models.llama.configuration_llama import LlamaConfig
from torch_geometric.loader import DataLoader

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class KgAdapterMLP(nn.Module):
    # TODO: zero-init gate ?
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class KgAdapterTripsEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.kg_adapter_hidden_size * 3, config.kg_adapter_hidden_size),
            torch.nn.LayerNorm(config.kg_adapter_hidden_size),
            torch.nn.GELU(), torch.nn.Linear(config.kg_adapter_hidden_size, config.kg_adapter_hidden_size))

    def forward(self, sg):
        bsz = len(sg.ptr) - 1
        max_trip_num = max(sg.num_edges).item()
        node_reps, _ = to_dense_batch(sg.x, sg.batch)
        edge_ids = unbatch_edge_index(sg.edge_index, sg.batch)
        pad_token = torch.zeros(sg.x.size(1) * 3).to(sg.x.device)
        # node_rep, edge_rep -> trip_rep

        batch_trip_reps = []
        batch_trip_mask = []
        for bs in range(bsz):
            node_rep = node_reps[bs]
            edge_idx = edge_ids[bs]
            h_rep = node_rep[edge_idx[0]]
            t_rep = node_rep[edge_idx[1]]
            rid = list(range(sg.num_edges[:bs].sum().item(), sg.num_edges[:bs].sum().item() + edge_idx.size(1)))
            r_rep = sg.edge_rep[rid]
            trip_reps = torch.cat([h_rep, r_rep, t_rep], dim=-1)
            batch_trip_mask.append(
                torch.cat([torch.ones(trip_reps.size(0)), torch.zeros(max_trip_num - trip_reps.size(0))]))
            trip_reps = torch.cat(
                [trip_reps, pad_token.repeat(max_trip_num - trip_reps.size(0)).view(-1, trip_reps.size(1))])
            batch_trip_reps.append(trip_reps)

        batch_trip_mask = torch.stack(batch_trip_mask).to(sg.x.device)
        batch_trip_reps = torch.stack(batch_trip_reps)
        batch_trip_reps = self.mlp(batch_trip_reps)

        return batch_trip_reps, batch_trip_mask


class Lora_layer(nn.Module):
    def __init__(self, r, lora_alpha, in_features, out_features):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = self.lora_alpha / self.r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input_tensor):
        # input_tensor = self.lora_dropout(input_tensor)
        input_tensor = torch.matmul(input_tensor, self.lora_A)
        input_tensor = torch.matmul(input_tensor, self.lora_B)
        input_tensor = input_tensor * self.scaling

        return input_tensor


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.add_lora = config.add_lora
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        if self.add_lora:
            self.lora_q = Lora_layer(r=16, lora_alpha=16 * 2, in_features=self.hidden_size,
                                     out_features=self.num_heads * self.head_dim)
            self.lora_v = Lora_layer(r=16, lora_alpha=16 * 2, in_features=self.hidden_size,
                                     out_features=self.num_heads * self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.add_lora:
            query_states = query_states + self.lora_q(hidden_states)
            value_states = value_states + self.lora_v(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class KgAdapterCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.kg_adapter_hidden_size
        self.num_heads = 4
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            q_hidden_states: torch.Tensor,
            k_hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, kv_len, hidden_size = q_hidden_states.size(0), q_hidden_states.size(1), k_hidden_states.size(
            1), q_hidden_states.size(-1)
        align_mask = None
        if isinstance(attention_mask, Tuple):
            align_mask = attention_mask[1]
            attention_mask = attention_mask[0]

        query_states = self.q_proj(q_hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(k_hidden_states).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(k_hidden_states).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if past_key_value[0].shape[2] != k_hidden_states.shape[1]:
                kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            if past_key_value[0].shape[2] != k_hidden_states.shape[1]:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            else:
                key_states = past_key_value[0]
                value_states = past_key_value[1]

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        if align_mask is not None:
            align_mask = align_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).transpose(-1, -2)
            attn_weights = attn_weights.masked_fill(align_mask == 0, torch.finfo(attn_weights.dtype).min)
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class KgAdapterRGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations, config=None):
        super().__init__()
        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations,
                              attention_mode="additive-self-attention", heads=1,
                              dim=1, concat=False,
                              edge_dim=config.kg_adapter_hidden_size if (
                                      config.use_edge_emb or config.mix_emb) else None)
        self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations,
                              attention_mode="additive-self-attention", heads=1,
                              dim=1, concat=False,
                              edge_dim=config.kg_adapter_hidden_size if (
                                      config.use_edge_emb or config.mix_emb) else None)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type, node_type=None, edge_attr=None):
        x = self.conv1(x, edge_index, edge_type, edge_attr).relu()
        x, gat_attn_weights = self.conv2(x, edge_index, edge_type, edge_attr, return_attention_weights=True)
        x = x.relu()
        x = self.lin(x)
        return x, gat_attn_weights


class KgAdapterSentRGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations, config=None):
        super().__init__()
        self.use_SRGAT = config.use_SRGAT
        self.gelu = nn.GELU()
        self.conv1 = SRGATConv(config, hidden_channels, 4, num_relations)

    def forward(self, x, edge_index, edge_type, node_type=None, edge_attr=None):
        x, gat_attn_weights = self.conv1(x, edge_index, edge_type=edge_type, node_type=node_type, edge_attr=edge_attr,
                                         return_attention_weights=True)
        x = self.gelu(x)
        # x, edge_index, edge_attr, batch, perm, score = self.pooling(x, edge_index, edge_attr)
        return x, gat_attn_weights[1]


class KgAdapterInfoMerge(nn.Module):
    def __init__(self, method, hidden_size, in_encoder=True):
        super().__init__()
        self.method = method
        if method == "gate":
            self.side_gate_params = nn.Parameter(torch.zeros(1))
        elif method == "linear":
            self.proj = nn.Linear(hidden_size * 2, hidden_size)
        # else use sum

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor = None,
    ):
        if self.method == "gate":
            gate = torch.sigmoid(self.side_gate_params)
            x = gate * x1 + (1 - gate) * x2

        elif self.method == "linear":
            if x3 is None:
                x = self.proj(torch.cat([x1, x2], dim=-1))
            else:
                x = self.proj(torch.cat([x1, x2, x3], dim=-1))
        else:
            if x3 is None:
                x = x1 + x2
            else:
                x = x1 + x2 + x3

        return x


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            sg=None,
            attention_mask: Optional[Dict] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Dict] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Optional[Dict]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        extra_outputs = {'input_hidden': hidden_states}

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask['text_self_mask'],
            position_ids=position_ids,
            past_key_value=past_key_value['sa_key_value'] if past_key_value is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        extra_outputs['sa_hidden'] = hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        extra_outputs['ffn_hidden'] = hidden_states
        hidden_states = residual + hidden_states

        outputs = {'text_hidden_states': hidden_states, 'extra_outputs': extra_outputs}

        if output_attentions:
            outputs['self_attn_weights'] = self_attn_weights

        if use_cache:
            outputs['sa_key_value'] = present_key_value

        return outputs


class LlamaWithKgAdapterEncLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.align_mask = config.align_mask
        self.enc_interact_with_LLM = config.enc_interact_with_LLM
        self.use_trips = config.use_trips
        self.exp_set = config.exp_set

        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        ################## kg_adapter_encoder #######################
        self.kg_adapter_cross_attn = KgAdapterCrossAttention(config=config)
        self.kg_adapter_ffn = KgAdapterMLP(
            hidden_size=config.kg_adapter_hidden_size,
            intermediate_size=config.kg_adapter_intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.kg_adapter_downscale = nn.Linear(config.hidden_size, config.kg_adapter_hidden_size)
        self.kg_adapter_info_merge = KgAdapterInfoMerge(config.kg_adapter_info_merge, config.hidden_size,
                                                        in_encoder=True)
        self.kg_adapter_sg_layernorm = LlamaRMSNorm(config.kg_adapter_hidden_size,
                                                    eps=config.rms_norm_eps)  # RMSNorm(config.kg_adapter_hidden_size)
        self.kg_adapter_text_layernorm = LlamaRMSNorm(config.kg_adapter_hidden_size,
                                                      eps=config.rms_norm_eps)  # RMSNorm(config.kg_adapter_hidden_size)
        self.kg_adapter_ffn_layernorm = LlamaRMSNorm(config.kg_adapter_hidden_size,
                                                     eps=config.rms_norm_eps)  # RMSNorm(config.kg_adapter_hidden_size)

        if self.enc_interact_with_LLM:
            self.kg_adapter_upscale = nn.Linear(config.kg_adapter_hidden_size, config.hidden_size)
        #############################################################

    def forward(
            self,
            hidden_states: torch.Tensor,
            sg=None,
            attention_mask: Optional[Dict] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Dict] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Optional[Dict]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        input_hidden = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask['text_self_mask'],
            position_ids=position_ids,
            past_key_value=past_key_value['sa_key_value'] if past_key_value is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_hidden = hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        ffn_hidden = hidden_states
        hidden_states = residual + hidden_states

        ############# kg-adapter code ######################
        sg_rep = update_flat_sg(sg, self.exp_set)
        # node_rep, node_mask = sg.node_rep, sg.node_mask
        merged_text_rep = self.kg_adapter_info_merge(input_hidden, attn_hidden, ffn_hidden)
        text_rep = self.kg_adapter_downscale(merged_text_rep)  # (bs,n,100)
        kg_adapter_residual = sg_rep
        sg_rep = self.kg_adapter_sg_layernorm(sg_rep)
        text_rep = self.kg_adapter_text_layernorm(text_rep)
        mask = attention_mask['trip_text_mask'] if self.use_trips else attention_mask['node_text_mask']
        mask = (mask, attention_mask['align_mask']) if self.align_mask else mask
        kg_adapter_hidden, kg_adapter_cross_attn_weights, kg_adapter_present_key_value = self.kg_adapter_cross_attn(
            q_hidden_states=sg_rep,
            k_hidden_states=text_rep,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value['ca_n2t_key_value'] if past_key_value is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        kg_adapter_hidden = kg_adapter_residual + kg_adapter_hidden

        # TODO: use same downsacle and layernorm? or two different?
        kg_adapter_residual = kg_adapter_hidden
        kg_adapter_hidden = self.kg_adapter_ffn_layernorm(kg_adapter_hidden)
        kg_adapter_hidden = self.kg_adapter_ffn(kg_adapter_hidden)
        kg_adapter_hidden = kg_adapter_residual + kg_adapter_hidden
        if self.use_trips:
            sg.trip_rep = kg_adapter_hidden
        else:
            sg.node_rep = kg_adapter_hidden
        sg = update_structure_sg(sg, self.exp_set)
        #######################################################
        if self.enc_interact_with_LLM:
            hidden_states += self.kg_adapter_upscale(kg_adapter_hidden.sum(1)).view(hidden_states.size(0), 1,
                                                                                    hidden_states.size(-1))

        outputs = {'text_hidden_states': hidden_states}

        if output_attentions:
            outputs['self_attn_weights'] = self_attn_weights
            outputs['cross_attn_weights'] = kg_adapter_cross_attn_weights

        if use_cache:
            outputs['sa_key_value'] = present_key_value
            outputs['ca_n2t_key_value'] = kg_adapter_present_key_value

        return outputs


def rebuild_trip_rep(sg):
    # rebuild trip_rep after prune
    cnt = 0
    bsz = sg.node_rep.size(0)
    cul_edge_num = [0]

    tmp = unbatch_edge_index(sg.edge_index, sg.batch)
    for bs in range(bsz):
        if bs < len(tmp):
            cnt += tmp[bs].size(1)
        else:
            cnt += 0
        cul_edge_num.append(cnt)

    trip_rep = []
    trip_num = []
    for bs in range(bsz):
        node = sg.node_rep[bs][sg.node_mask[bs]]
        edge = sg.edge_rep[cul_edge_num[bs]: cul_edge_num[bs + 1]]

        trip_rep.append(torch.cat([node, edge]))
        trip_num.append([len(node), len(edge)])

    if (sg.trip_mask == 0).any().item():
        trip_pad_rep = sg.trip_rep[sg.trip_mask == 0][0].unsqueeze(0)
    else:
        trip_pad_rep = torch.zeros(1, sg.x.size(1)).to(sg.x.device)

    max_trip_num = max([x.size(0) for x in trip_rep])
    trip_mask = torch.zeros(bsz, max_trip_num)
    node_mask = torch.zeros(bsz, max_trip_num)
    edge_mask = torch.zeros(bsz, max_trip_num)
    for bs in range(bsz):
        trip_mask[bs, :len(trip_rep[bs])] = 1
        node_mask[bs, :trip_num[bs][0]] = 1
        edge_mask[bs, trip_num[bs][0]: trip_num[bs][0] + trip_num[bs][1]] = 1
        trip_rep[bs] = torch.cat([trip_rep[bs], trip_pad_rep.repeat(max_trip_num - len(trip_rep[bs]), 1)])

    sg.trip_rep = torch.stack(trip_rep)
    sg.trips['trip_mask'] = trip_mask.bool()
    sg.trips['node_mask'] = node_mask.bool()
    sg.trips['edge_mask'] = edge_mask.bool()
    sg.trip_mask = sg.trips['trip_mask']


class LlamaWithKgAdapterDecLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dynamic_prune = config.dynamic_prune
        self.use_gnn = config.use_gnn
        self.use_trips = config.use_trips
        self.exp_set = config.exp_set
        self.output_sg = config.output_sg
        self.keep_ratio = config.keep_ratio
        self.fuse_rate = config.fuse_rate
        self.scaling_rate = config.scaling_rate  # same as the scaling in lora
        self.use_prefix = config.use_prefix
        self.no_res = config.no_res
        self.linear_scale = config.linear_scale
        self.info_merge_pos = config.info_merge_pos

        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        ################## kg_adapter_decoder #######################
        self.act_fn = ACT2FN[config.hidden_act]
        if config.use_SRGAT:
            self.kg_adapter_gat = KgAdapterSentRGAT(in_channels=config.kg_adapter_hidden_size,
                                                    hidden_channels=config.kg_adapter_hidden_size,
                                                    out_channels=config.kg_adapter_hidden_size,
                                                    num_relations=config.num_relations,
                                                    config=config
                                                    )
            self.pooling = SAGPooling(config.kg_adapter_hidden_size, ratio=config.keep_ratio, GNN=GATConv,
                                      nonlinearity=torch.tanh)
        else:
            self.kg_adapter_gat = KgAdapterRGAT(in_channels=config.kg_adapter_hidden_size,
                                                hidden_channels=config.kg_adapter_hidden_size,
                                                out_channels=config.kg_adapter_hidden_size,
                                                num_relations=config.num_relations,
                                                config=config
                                                )

        self.kg_adapter_sg_to_trip_layer = KgAdapterTripsEncoder(config=config)
        self.kg_adapter_t2n_cross_attn = KgAdapterCrossAttention(config=config)
        self.kg_adapter_n2t_cross_attn = KgAdapterCrossAttention(config=config)
        self.kg_adapter_ffn = KgAdapterMLP(
            hidden_size=config.kg_adapter_hidden_size,
            intermediate_size=config.kg_adapter_intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.kg_adapter_node_ffn = KgAdapterMLP(
            hidden_size=config.kg_adapter_hidden_size,
            intermediate_size=config.kg_adapter_intermediate_size,
            hidden_act=config.hidden_act,
        )

        self.kg_adapter_downscale = nn.Linear(config.hidden_size, config.kg_adapter_hidden_size)
        self.kg_adapter_downscale_proj = nn.Linear(config.kg_adapter_hidden_size, config.kg_adapter_hidden_size)
        self.kg_adapter_upscale = nn.Linear(config.kg_adapter_hidden_size, config.hidden_size)
        self.kg_adapter_upscale_proj = nn.Linear(config.kg_adapter_hidden_size, config.kg_adapter_hidden_size)
        self.kg_adapter_text_info_merge = KgAdapterInfoMerge(config.kg_adapter_info_merge, config.hidden_size)
        self.kg_adapter_trip_info_merge = KgAdapterInfoMerge(config.kg_adapter_info_merge, config.hidden_size)
        self.kg_adapter_output_info_merge = KgAdapterInfoMerge(config.kg_adapter_info_merge, config.hidden_size)
        self.kg_adapter_text_layernorm = LlamaRMSNorm(config.kg_adapter_hidden_size, eps=config.rms_norm_eps)
        self.kg_adapter_sg_layernorm = LayerNorm(config.kg_adapter_hidden_size, mode="node")  # pyg graph layernorm
        self.kg_adapter_node_layernorm = LlamaRMSNorm(config.kg_adapter_hidden_size, eps=config.rms_norm_eps)
        self.kg_adapter_ffn_layernorm = LlamaRMSNorm(config.kg_adapter_hidden_size, eps=config.rms_norm_eps)

        #############################################################

    def forward(
            self,
            hidden_states: torch.Tensor,
            sg=None,
            attention_mask: Optional[Dict] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Dict] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Optional[Dict]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        ############# kg-adapter code ######################

        # get text rep from LLM side
        if self.no_res:
            text_rep = hidden_states
        else:
            text_rep = self.kg_adapter_text_info_merge(hidden_states, sg.text_rep)

        if self.linear_scale:
            text_rep = self.kg_adapter_downscale(text_rep)
        else:
            text_rep = self.kg_adapter_downscale_proj(self.act_fn(self.kg_adapter_downscale(text_rep)))
        text_rep_residual = text_rep
        text_rep = self.kg_adapter_text_layernorm(text_rep)

        # update node rep by GNN
        if self.use_gnn:
            node_rep_residual = sg.x
            sg.x = self.kg_adapter_sg_layernorm(sg.x, sg.batch)
            if len(sg.edge_rep) > 0:
                sg.edge_rep = self.kg_adapter_sg_layernorm(sg.edge_rep)
            sg.x, gat_attn_weights = self.kg_adapter_gat(
                x=sg.x,
                edge_index=sg.edge_index,
                edge_type=sg.edge_type,
                node_type=sg.node_type if "node_type" in sg.keys else None,
                edge_attr=sg.edge_rep if len(sg.edge_rep) > 0 else None,  # change: sg.edge_rep.clone()
            )
            sg.x = sg.x + node_rep_residual

        if self.use_trips:
            trip_rep, trip_mask = self.kg_adapter_sg_to_trip_layer(sg)
            if self.no_res:
                trip_rep = trip_rep
            else:
                trip_rep = self.kg_adapter_trip_info_merge(trip_rep, sg.trip_rep)
            attention_mask['trip_text_mask'] = _prepare_4d_attention_mask(attention_mask['base_mask'],
                                                                          text_rep.dtype, sg.trip_rep.size(1))
            attention_mask['text_trip_mask'] = _prepare_4d_attention_mask(trip_mask, text_rep.dtype,
                                                                          text_rep.size(1))
            node_rep = trip_rep
        else:
            node_rep = update_flat_sg(sg)
        # bi-direction attention, refer to JointLK: "JointLK: Joint Reasoning with Language Models and Knowledge Graphs for Commonsense Question Answering"
        node_rep_residual = node_rep
        # infuse text info into trip rep
        node_hidden, node_cross_attn_weights, n2t_present_key_value = self.kg_adapter_n2t_cross_attn(
            q_hidden_states=node_rep,
            k_hidden_states=text_rep,
            attention_mask=attention_mask['node_text_mask'] if not self.use_trips else attention_mask['trip_text_mask'],
            position_ids=position_ids,
            past_key_value=past_key_value['ca_n2t_key_value'] if past_key_value is not None else None,
            output_attentions=True,
            use_cache=use_cache,
        )
        # FFN
        node_hidden = node_rep_residual + self.fuse_rate * node_hidden
        node_rep_residual = node_hidden
        node_hidden = self.kg_adapter_node_layernorm(node_hidden)
        node_hidden = self.kg_adapter_node_ffn(node_hidden)
        node_hidden = node_hidden + node_rep_residual
        if self.use_trips:
            sg.trip_rep = node_hidden
        else:
            sg.node_rep = node_hidden

        # infuse trip info into text rep
        text_hidden, text_cross_attn_weights, t2n_present_key_value = self.kg_adapter_t2n_cross_attn(
            q_hidden_states=text_rep,
            k_hidden_states=node_rep,
            attention_mask=attention_mask['text_node_mask'] if not self.use_trips else attention_mask['text_trip_mask'],
            position_ids=position_ids,
            past_key_value=past_key_value['ca_t2n_key_value'] if past_key_value is not None else None,
            output_attentions=True,
            use_cache=use_cache,
        )
        text_hidden = text_hidden + text_rep_residual
        text_rep_residual = text_hidden
        text_hidden = self.kg_adapter_ffn_layernorm(text_hidden)
        text_hidden = self.kg_adapter_ffn(text_hidden)
        text_hidden = text_hidden + text_rep_residual

        if self.linear_scale:
            text_hidden = self.kg_adapter_upscale(text_hidden)
        else:
            text_hidden = self.kg_adapter_upscale(self.act_fn(self.kg_adapter_upscale_proj(text_hidden)))
        sg.text_rep = text_hidden  # cache for next layer residual
        text_hidden = text_hidden * self.scaling_rate
        fused_hidden_states = self.kg_adapter_output_info_merge(hidden_states, text_hidden)

        # dynamic pruning
        score = None
        # update text rep
        # if self.use_prefix:
        #     global_node_rep = global_mean_pool(sg.x, sg.batch).view(node_rep.size(0), 1, -1)
        #     global_node_rep = self.kg_adapter_ffn_layernorm(global_node_rep)
        #     global_node_rep = self.kg_adapter_upscale(self.act_fn(self.kg_adapter_upscale_proj(global_node_rep)))
        #     global_node_rep = global_node_rep * self.scaling_rate
        #
        #     # prefix = self.kg_adapter_prefix_wte.weight.reshape(1, 10, -1).repeat(node_rep.size(0), 1, 1)
        #     # prefix = prefix + global_node_rep
        #
        # # dynamic pruning
        # # sg = update_structure_sg(sg, "nodes")  # node_rep -> sg.x
        # score = None
        # if "use_SRGAT" in self.exp_set and self.keep_ratio < 1:  # if keep ratio >= 1, there is no need to prune
        #     node_scores = node_cross_attn_weights.sum(1).sum(-1)[sg.node_mask.bool()]
        #     sg.x, sg.edge_index, sg.edge_rep, sg.edge_type, sg.node_type, sg.batch, perm, score = self.pooling(
        #         x=sg.x,
        #         score=node_scores,
        #         edge_index=sg.edge_index,
        #         edge_attr=sg.edge_rep,
        #         edge_type=sg.edge_type,
        #         node_type=sg.node_type if "node_type" in sg.keys else None,
        #         batch=sg.batch)
        #     sg.node_ids = sg.node_ids[sg.node_mask.bool()][perm]
        #     sg.node_ids, _ = to_dense_batch(sg.node_ids, sg.batch)
        #     sg.node_rep, sg.node_mask = to_dense_batch(sg.x, sg.batch)
        #
        #     attention_mask['node_text_mask'] = _prepare_4d_attention_mask(attention_mask['base_mask'],
        #                                                                   text_rep.dtype,
        #                                                                   sg.node_rep.size(1))
        #     attention_mask['text_node_mask'] = _prepare_4d_attention_mask(sg.node_mask, text_rep.dtype,
        #                                                                   text_rep.size(1))

        #######################################################

        #######################################################

        residual = hidden_states
        if self.info_merge_pos == "before":
            hidden_states = self.input_layernorm(fused_hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask['text_self_mask'],
            position_ids=position_ids,
            past_key_value=past_key_value['sa_key_value'] if past_key_value is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_hidden = hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if self.info_merge_pos == "mid":
            hidden_states = self.post_attention_layernorm(fused_hidden_states)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        ffn_hidden = hidden_states
        hidden_states = residual + hidden_states

        if self.info_merge_pos == "after":
            hidden_states = hidden_states + fused_hidden_states
        else:
            hidden_states = hidden_states

        outputs = {'text_hidden_states': hidden_states}

        if self.output_sg:
            outputs['sg_state'] = (sg.node_ids, sg.edge_index, sg.batch, score) if "use_SRGAT" in self.exp_set else (
                sg.node_ids, sg.edge_index, sg.batch)

        if output_attentions:
            outputs['self_attn_weights'] = self_attn_weights
            if not self.use_prefix:
                outputs['cross_attn_weights'] = (text_cross_attn_weights, node_cross_attn_weights)

        if use_cache:
            outputs['sa_key_value'] = present_key_value
            outputs['ca_n2t_key_value'] = n2t_present_key_value
            if not self.use_prefix:
                outputs['ca_t2n_key_value'] = t2n_present_key_value

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


def update_flat_sg(sg, exp_set="nodes"):
    # x, edge -> trip_rep
    if "use_cat_trips" in exp_set:
        trip_ids = sg.trips['trip_ids']
        node_mask = sg.trips['node_mask']
        edge_mask = sg.trips['edge_mask']
        trip_rep = sg.trip_rep.to(sg.x.dtype)

        trip_rep[node_mask] = sg.x
        trip_rep[edge_mask] = sg.edge_rep.to(sg.x.dtype)

        return trip_rep

    elif "use_trips" in exp_set:
        trip_num = sg.trips['trip_num']
        trip_rep = sg.trip_rep
        # <h>S<r>R<t>O<h>S<r>R<t>O....
        for bs in range(trip_rep.size(0)):
            max_trip_num = (trip_num[bs + 1] - trip_num[bs]) * 3
            trip_rep[bs, [x for x in range(0, max_trip_num, 3)]] = sg.x.index_select(0, sg.edge_index[0][
                                                                                        trip_num[bs]: trip_num[
                                                                                            bs + 1]])
            trip_rep[bs, [x for x in range(1, max_trip_num, 3)]] = sg.edge_rep[trip_num[bs]: trip_num[bs + 1]]
            trip_rep[bs, [x for x in range(2, max_trip_num, 3)]] = sg.x.index_select(0, sg.edge_index[1][
                                                                                        trip_num[bs]: trip_num[
                                                                                            bs + 1]])
        return trip_rep

    # x -> node_rep
    else:
        node_rep = sg.node_rep
        node_mask = sg.node_mask.bool()
        node_rep[node_mask] = sg.x

        return node_rep


def update_structure_sg(sg, exp_set="nodes"):
    # trip_rep -> x, edge_rep
    if "use_cat_trips" in exp_set:
        trip_ids = sg.trips['trip_ids']
        node_mask = sg.trips['node_mask']
        edge_mask = sg.trips['edge_mask']
        trip_rep = sg.trip_rep
        sg.x = trip_rep[node_mask]
        sg.edge_rep = trip_rep[edge_mask]
        return sg
    # trip_rep -> x, edge
    elif "use_trips" in exp_set:
        trip_rep = sg.trip_rep.view(-1, sg.trip_rep.size(-1))
        trip_ids = sg.trips['trip_ids'].view(-1)

        sg.edge_rep = trip_rep[trip_ids < 0]

        node_idx = [-1 for x in range(sg.x.size(0))]
        used = set()
        tmp = (trip_ids[trip_ids > 0] - 1).tolist()
        for tid, nid in enumerate(tmp):
            if nid not in used:
                node_idx[nid] = tid
                used.add(nid)
        sg.x = trip_rep[trip_ids > 0].index_select(0, torch.tensor(node_idx, device=trip_rep.device))

        return sg
    # node_rep -> x
    else:
        node_rep = sg.node_rep
        node_mask = sg.node_mask
        sg.x = node_rep[node_mask.bool()]
        return sg


def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    from torch.autograd import Variable
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


def _build_new_ca_attention_mask(attention_mask, input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask



LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaKgAdapterModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_node_emb = config.use_node_emb
        self.use_edge_emb = config.use_edge_emb
        self.mix_emb = config.mix_emb
        self.use_trips = config.use_trips
        self.enc_sa = config.enc_sa
        self.output_sg = config.output_sg
        self.config = config
        self.use_kg_encoder = config.use_kg_encoder
        self.linear_emb = config.linear_emb

        self.act_fn = ACT2FN[config.hidden_act]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        if self.use_node_emb:
            self.embed_nodes = nn.Embedding(config.node_num, config.kg_adapter_node_emb_size, 0)
        self.kg_adapter_nodes_proj = nn.Linear(config.kg_adapter_node_emb_size, config.kg_adapter_hidden_size)
        if self.use_edge_emb:
            self.kg_adapter_embed_edges = torch.nn.Sequential(
                torch.nn.Linear(config.num_relations + 1 + 5 * 2, config.kg_adapter_hidden_size),
                torch.nn.LayerNorm(config.kg_adapter_hidden_size),
                ACT2FN[config.hidden_act],
                torch.nn.Linear(config.kg_adapter_hidden_size, config.kg_adapter_hidden_size))
            # self.kg_adapter_embed_edges = nn.Embedding(config.num_relations + 1, config.kg_adapter_node_emb_size, -1)
        self.kg_adapter_edges_proj = nn.Linear(config.kg_adapter_hidden_size, config.kg_adapter_hidden_size)
        # if self.mix_emb:  # to load same preprocessed model
        self.kg_adapter_t2n_mix_facotr = nn.Parameter(torch.zeros(1))
        self.kg_adapter_t2e_mix_facotr = nn.Parameter(torch.zeros(1))
        self.kg_adapter_t2n_proj = nn.Linear(config.hidden_size, config.kg_adapter_node_emb_size)
        self.kg_adapter_t2e_proj = nn.Linear(config.hidden_size, config.kg_adapter_hidden_size)

        enc_stride = config.kg_adapter_enc_range[2] if len(config.kg_adapter_enc_range) == 3 else 1
        dec_stride = config.kg_adapter_dec_range[2] if len(config.kg_adapter_dec_range) == 3 else 1
        self.kg_adapter_enc_range = [x for x in
                                     range(config.kg_adapter_enc_range[0], config.kg_adapter_enc_range[1], enc_stride)]
        self.kg_adapter_dec_range = [x for x in
                                     range(config.kg_adapter_dec_range[0], config.kg_adapter_dec_range[1], dec_stride)]
        module_lst = []
        for i in range(config.num_hidden_layers):
            if i in self.kg_adapter_enc_range:
                module_lst.append(LlamaWithKgAdapterEncLayer(config))
            elif i in self.kg_adapter_dec_range:
                module_lst.append(LlamaWithKgAdapterDecLayer(config))
            else:
                module_lst.append(LlamaDecoderLayer(config))
        self.layers = nn.ModuleList(module_lst)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length=None,
                                        type="sa"):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1 and type == "sa":
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            sg=None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_sg_states = True if self.output_sg else None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0]['sa_key_value'][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed nodes
        nodes_embeds = None
        if self.use_node_emb:
            if self.linear_emb:
                nodes_embeds = self.embed_nodes(sg.node_ids).view(batch_size, -1, self.config.kg_adapter_node_emb_size)
            else:
                nodes_embeds = self.act_fn(self.embed_nodes(sg.node_ids).view(batch_size, -1, self.config.kg_adapter_node_emb_size))
        if self.mix_emb:
            if nodes_embeds is None:
                nodes_embeds = self.act_fn(self.kg_adapter_t2n_proj(self.embed_tokens(sg.nid2swid).sum(2)))
            else:
                if self.linear_emb:
                    nodes_embeds = self.act_fn(nodes_embeds + self.kg_adapter_t2n_proj(self.embed_tokens(sg.nid2swid).sum(2)))
                else:
                    t2n_gate = torch.sigmoid(self.kg_adapter_t2n_mix_facotr)
                    nodes_embeds = self.act_fn(t2n_gate * nodes_embeds + (1 - t2n_gate) * self.kg_adapter_t2n_proj(
                        self.embed_tokens(sg.nid2swid).sum(2)))

        sg.node_rep = self.kg_adapter_nodes_proj(nodes_embeds)
        _, sg.node_mask = to_dense_batch(sg.x, sg.batch)
        sg = update_structure_sg(sg)

        # embed edges
        edges_embeds = None
        if self.use_edge_emb:
            # Prepare edge feature
            edge_vec = make_one_hot(sg.edge_type, self.config.num_relations + 1)  # [E, 39]
            node_type = sg.node_type.view(-1).contiguous()  # [`total_n_nodes`, ]
            head_type = node_type[sg.edge_index[0]]  # [E,] #head=src
            tail_type = node_type[sg.edge_index[1]]  # [E,] #tail=tgt
            head_vec = make_one_hot(head_type, 5)  # [E,5]
            tail_vec = make_one_hot(tail_type, 5)  # [E,5]
            headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,10]
            edge_feature = torch.cat([edge_vec, headtail_vec], dim=1).to(nodes_embeds.dtype)
            edges_embeds = self.kg_adapter_embed_edges(edge_feature)  # [E, emb_dim]
        if self.mix_emb:
            if edges_embeds is None:
                if self.linear_emb:
                    edges_embeds = self.kg_adapter_t2e_proj(self.embed_tokens(sg.eid2swid).sum(1))
                else:
                    edges_embeds = self.act_fn(self.kg_adapter_t2e_proj(self.embed_tokens(sg.eid2swid).sum(1)))
            else:
                if self.linear_emb:
                    edges_embeds = self.act_fn(edges_embeds + self.kg_adapter_t2e_proj(self.embed_tokens(sg.eid2swid).sum(1)))
                else:
                    t2e_gate = torch.sigmoid(self.kg_adapter_t2e_mix_facotr)
                    edges_embeds = self.act_fn(t2e_gate * edges_embeds + (1 - t2e_gate) * self.kg_adapter_t2e_proj(
                        self.embed_tokens(sg.eid2swid).sum(1)))

        sg.edge_rep = self.kg_adapter_edges_proj(edges_embeds) if edges_embeds is not None else []

        if self.use_trips:
            bsz = len(sg.ptr) - 1
            max_trip_num = max(sg.num_edges).item()
            edge_ids = unbatch_edge_index(sg.edge_index, sg.batch)
            # node_rep, edge_rep -> trip_rep
            batch_trip_mask = []
            for bs in range(bsz):
                edge_idx = edge_ids[bs]
                batch_trip_mask.append(
                    torch.cat([torch.ones(edge_idx.size(1)), torch.zeros((max_trip_num - edge_idx.size(1)))], dim=0))

            sg.trip_mask = torch.stack(batch_trip_mask).to(
                sg.x.device)
            sg.trip_rep = torch.zeros((sg.trip_mask.size(0), sg.trip_mask.size(1), sg.x.size(1))).to(
                sg.x.device)

            # trip_rep = sg.trips['trip_rep']
            # trip_num = sg.trips['trip_num']
            # # <h>S<r>R<t>O<h>S<r>R<t>O....
            # for bs in range(batch_size):
            #     max_trip_num = (trip_num[bs + 1] - trip_num[bs]) * 3
            #     # trip_rep[bs, [x for x in range(0, max_trip_num, 6)]] = h
            #     trip_rep[bs, [x for x in range(0, max_trip_num, 3)]] = sg.x.index_select(0, sg.edge_index[0,
            #                                                                                 trip_num[bs]: trip_num[
            #                                                                                     bs + 1]])
            #     # trip_rep[bs, [x for x in range(2, max_trip_num, 6)]] = r
            #     trip_rep[bs, [x for x in range(1, max_trip_num, 3)]] = sg.edge_rep[trip_num[bs]: trip_num[bs + 1]]
            #     # trip_rep[bs, [x for x in range(4, max_trip_num, 6)]] = t
            #     trip_rep[bs, [x for x in range(2, max_trip_num, 3)]] = sg.x.index_select(0, sg.edge_index[1,
            #                                                                                 trip_num[bs]: trip_num[
            #                                                                                     bs + 1]])
            # sg.trip_rep = trip_rep
            # sg.trip_mask = sg.trips['trip_mask']

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        text_self_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, type="sa"
        )
        # TODO: is CA mask correct?
        node_text_mask = self._prepare_decoder_attention_mask(
            attention_mask, (sg.node_rep.size(0), sg.node_rep.size(1)), inputs_embeds, past_key_values_length, type="ca"
        )

        text_node_mask = self._prepare_decoder_attention_mask(
            sg.node_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, type="ca"
        )
        align_mask = None
        trip_text_mask = None
        text_trip_mask = None
        if self.config.align_mask:
            # align_mask = self._prepare_decoder_attention_mask(
            #     sg.align_mask, (batch_size, sg.node_rep.size(1)), inputs_embeds, past_key_values_length, type="ca"
            # )
            if past_key_values_length != 0:
                align_mask = torch.cat([torch.zeros(batch_size, seq_length_with_past - sg.align_mask.size(1),
                                                    sg.align_mask.size(-1), device=inputs_embeds.device),
                                        sg.align_mask], dim=1)
            else:
                align_mask = sg.align_mask
        if self.use_trips:
            trip_text_mask = self._prepare_decoder_attention_mask(
                attention_mask, (sg.trip_rep.size(0), sg.trip_rep.size(1)), inputs_embeds, past_key_values_length,
                type="ca"
            )
            text_trip_mask = self._prepare_decoder_attention_mask(
                sg.trip_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, type="ca"
            )
        attention_mask = {"base_mask": attention_mask,
                          "text_self_mask": text_self_mask,
                          "node_text_mask": node_text_mask,
                          "text_node_mask": text_node_mask,
                          "align_mask": align_mask,
                          "trip_text_mask": trip_text_mask,
                          "text_trip_mask": text_trip_mask,
                          "seq_length": seq_length,
                          }

        hidden_states = inputs_embeds
        sg.text_rep = hidden_states  # cache for residual

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_sg_states = () if output_sg_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    sg=sg,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs['text_hidden_states']

            if use_cache:
                cache = {'sa_key_value': layer_outputs['sa_key_value']}
                if 'ca_n2t_key_value' in layer_outputs.keys():
                    cache['ca_n2t_key_value'] = layer_outputs['ca_n2t_key_value']
                if 'ca_t2n_key_value' in layer_outputs.keys():
                    cache['ca_t2n_key_value'] = layer_outputs['ca_t2n_key_value']

                next_decoder_cache += (cache,)

            if output_sg_states and "sg_state" in layer_outputs.keys():
                all_sg_states += (layer_outputs['sg_state'],)

            # if output_attentions:
            #     all_self_attns += (layer_outputs['self_attn_weights'],)
            #     if "cross_attn_weights" in layer_outputs.keys():
            #         all_self_attns += (layer_outputs["cross_attn_weights"],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_sg_states] if
                         v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_sg_states,  # replace attentions with sg_states
        )


class LlamaKgAdapterForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaKgAdapterModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            sg=None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            sg=sg.clone().detach(),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_model_inputs(
            self,
            inputs: Optional[torch.Tensor] = None,
            bos_token_id: Optional[int] = None,
            model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """

        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        if (
                self.config.is_encoder_decoder
                and hasattr(self, "encoder")
                and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed."
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. In the presence of `inputs_embeds` for text models:
        # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
        # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
        # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
        # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
        # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                        "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                        "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                    )
                # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
                # the attention mask) can rely on the actual model input.
                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, model_kwargs=model_kwargs
                )
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, sg=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "sg": sg,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
