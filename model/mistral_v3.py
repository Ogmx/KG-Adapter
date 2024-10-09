# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Mistral model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, Parameter
from torch_geometric.utils import to_dense_batch, dense_to_sparse, unbatch_edge_index, unbatch
from torch_geometric.nn import LayerNorm, HeteroLayerNorm, HANConv, HGTConv, GATConv  # RGATConv
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from .GNN import RGATConv, SRGATConv, SAGPooling
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, \
    SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)
from transformers import MistralConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

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


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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


class MistralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.add_lora = config.add_lora
        self.use_prefix = config.use_prefix or config.use_kg_encoder

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        if self.add_lora:
            self.lora_q = Lora_layer(r=16, lora_alpha=16 * 2, in_features=self.hidden_size,
                                     out_features=self.num_heads * self.head_dim)
            self.lora_v = Lora_layer(r=16, lora_alpha=16 * 2, in_features=self.hidden_size,
                                     out_features=self.num_key_value_heads * self.head_dim)
        if self.use_prefix:
            self.kg_adapter_preifx = torch.nn.Parameter(torch.zeros(10, self.hidden_size))
            self.kg_adapter_gating_factor = torch.nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

            nn.init.kaiming_uniform_(self.kg_adapter_preifx)
            nn.init.zeros_(self.kg_adapter_gating_factor)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        node_rep = None
        if isinstance(hidden_states, Tuple) and self.use_prefix:
            node_rep = hidden_states[-1]
            hidden_states = hidden_states[0]

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.add_lora:
            query_states = query_states + self.lora_q(hidden_states)
            value_states = value_states + self.lora_v(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value_temp = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

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

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if self.use_prefix and node_rep is not None:
            if past_key_value is not None and len(past_key_value) > 2:
                prefix_key_states = past_key_value[2]
                prefix_value_states = past_key_value[3]
            else:
                prefix_hidden = self.kg_adapter_preifx.reshape(1, 10, -1).repeat(bsz, 1, 1)
                prefix_hidden = prefix_hidden + node_rep
                prefix_key_states = self.k_proj(prefix_hidden).view(bsz, 10, self.num_key_value_heads,
                                                                    self.head_dim).transpose(1, 2)
                prefix_value_states = self.v_proj(prefix_hidden).view(bsz, 10, self.num_key_value_heads,
                                                                      self.head_dim).transpose(1, 2)

                past_key_value_temp = past_key_value_temp + (prefix_key_states, prefix_value_states)

            prefix_key_states = repeat_kv(prefix_key_states, self.num_key_value_groups)
            prefix_value_states = repeat_kv(prefix_value_states, self.num_key_value_groups)
            prefix_mask = torch.ones((bsz, 1, q_len, 10), dtype=torch.bool, device=query_states.device)
            prefix_attn_weights = torch.matmul(query_states, prefix_key_states.transpose(2, 3)) / math.sqrt(
                self.head_dim)
            prefix_attn_weights = prefix_attn_weights + prefix_mask
            prefix_attn_weights = nn.functional.softmax(prefix_attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype)
            prefix_attn_output = torch.matmul(prefix_attn_weights, prefix_value_states)

            attn_output = attn_output + self.kg_adapter_gating_factor * prefix_attn_output

        past_key_value = past_key_value_temp if use_cache else None

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class KgAdapterCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MistralConfig):
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
    def __init__(self, method, hidden_size):
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


class MistralFlashAttention2(MistralAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
                _flash_supports_window_size
                and hasattr(self.config, "sliding_window") is not None
                and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            if hasattr(self.config, "sliding_window") and kv_seq_len > self.config.sliding_window:
                slicing_tokens = kv_seq_len - self.config.sliding_window

                past_key = past_key_value[0]
                past_value = past_key_value[1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key much have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                past_key_value = (past_key, past_value)

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # TODO: Mistral does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            dropout=0.0,
            softmax_scale=None,
            use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len:]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.use_prefix = config.use_kg_encoder
        self.hidden_size = config.hidden_size
        self.self_attn = (
            MistralAttention(config=config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else MistralFlashAttention2(config)
        )
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            sg=None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value=None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=(hidden_states, sg) if self.use_prefix else hidden_states,
            attention_mask=attention_mask['text_self_mask'],
            position_ids=position_ids,
            past_key_value=past_key_value['sa_key_value'] if past_key_value is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = {"text_hidden_states": hidden_states}

        if output_attentions:
            outputs['self_attn_weights'] = self_attn_weights

        if use_cache:
            outputs['sa_key_value'] = present_key_value

        return outputs


class MistralWithKgAdapterDecLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_id=None):
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

        self.self_attn = MistralAttention(config=config)
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        self.kg_adapter_text_layernorm = MistralRMSNorm(config.kg_adapter_hidden_size, eps=config.rms_norm_eps)
        self.kg_adapter_sg_layernorm = LayerNorm(config.kg_adapter_hidden_size, mode="node")  # pyg graph layernorm
        self.kg_adapter_node_layernorm = MistralRMSNorm(config.kg_adapter_hidden_size, eps=config.rms_norm_eps)
        self.kg_adapter_ffn_layernorm = MistralRMSNorm(config.kg_adapter_hidden_size, eps=config.rms_norm_eps)

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

        score = text_cross_attn_weights.sum(1).sum(1)
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


# def rebuild_trip_rep(sg):
#     # rebuild trip_rep after prune
#     cnt = 0
#     bsz = sg.node_rep.size(0)
#     cul_edge_num = [0]
#
#     tmp = unbatch_edge_index(sg.edge_index, sg.batch)
#     for bs in range(bsz):
#         if bs < len(tmp):
#             cnt += tmp[bs].size(1)
#         else:
#             cnt += 0
#         cul_edge_num.append(cnt)
#
#     trip_rep = []
#     trip_num = []
#     for bs in range(bsz):
#         node = sg.node_rep[bs][sg.node_mask[bs]]
#         edge = sg.edge_rep[cul_edge_num[bs]: cul_edge_num[bs + 1]]
#
#         trip_rep.append(torch.cat([node, edge]))
#         trip_num.append([len(node), len(edge)])
#
#     if (sg.trip_mask == 0).any().item():
#         trip_pad_rep = sg.trip_rep[sg.trip_mask == 0][0].unsqueeze(0)
#     else:
#         trip_pad_rep = torch.zeros(1, sg.x.size(1)).to(sg.x.device)
#
#     max_trip_num = max([x.size(0) for x in trip_rep])
#     trip_mask = torch.zeros(bsz, max_trip_num)
#     node_mask = torch.zeros(bsz, max_trip_num)
#     edge_mask = torch.zeros(bsz, max_trip_num)
#     for bs in range(bsz):
#         trip_mask[bs, :len(trip_rep[bs])] = 1
#         node_mask[bs, :trip_num[bs][0]] = 1
#         edge_mask[bs, trip_num[bs][0]: trip_num[bs][0] + trip_num[bs][1]] = 1
#         trip_rep[bs] = torch.cat([trip_rep[bs], trip_pad_rep.repeat(max_trip_num - len(trip_rep[bs]), 1)])
#
#     sg.trip_rep = torch.stack(trip_rep)
#     sg.trips['trip_mask'] = trip_mask.bool()
#     sg.trips['node_mask'] = node_mask.bool()
#     sg.trips['edge_mask'] = edge_mask.bool()
#     sg.trip_mask = sg.trips['trip_mask']


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
    # trip_rep -> x, edge
    if "use_cat_trips" in exp_set:
        trip_ids = sg.trips['trip_ids']
        node_mask = sg.trips['node_mask']
        edge_mask = sg.trips['edge_mask']
        trip_rep = sg.trip_rep
        sg.x = trip_rep[node_mask]
        sg.edge_rep = trip_rep[edge_mask]
        return sg

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

################################################################
MISTRAL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MistralConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralPreTrainedModel(PreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MistralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

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


MISTRAL_INPUTS_DOCSTRING = r"""
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
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralKgAdapterModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig):
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
            if i in self.kg_adapter_dec_range:
                module_lst.append(MistralWithKgAdapterDecLayer(config, layer_id=i))
            else:
                module_lst.append(MistralDecoderLayer(config))

        self.layers = nn.ModuleList(module_lst)
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
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

        if (
                attention_mask is not None
                and hasattr(self.config, "_flash_attn_2_enabled")
                and self.config._flash_attn_2_enabled
                and past_key_values is not None
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

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

        # process mask
        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            text_self_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )

        node_text_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype, sg.node_rep.size(1))
        text_node_mask = _prepare_4d_attention_mask(sg.node_mask, inputs_embeds.dtype, seq_length)
        align_mask = None
        trip_text_mask = None
        text_trip_mask = None
        sg_self_mask = None
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
            trip_text_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype, sg.trip_rep.size(1))
            text_trip_mask = _prepare_4d_attention_mask(sg.trip_mask, inputs_embeds.dtype, seq_length)
        if self.enc_sa or self.use_kg_encoder:
            if self.use_trips:
                sg_self_mask = _prepare_4d_attention_mask(sg.trip_mask, inputs_embeds.dtype, sg.trip_mask.size(1))
            else:
                sg_self_mask = _prepare_4d_attention_mask(sg.node_mask, inputs_embeds.dtype, sg.node_mask.size(1))

        attention_mask = {"base_mask": attention_mask,
                          "text_self_mask": text_self_mask,
                          "node_text_mask": node_text_mask,
                          "text_node_mask": text_node_mask,
                          "align_mask": align_mask,
                          "trip_text_mask": trip_text_mask,
                          "text_trip_mask": text_trip_mask,
                          "sg_self_mask": sg_self_mask,
                          "seq_length": seq_length,
                          }

        hidden_states = inputs_embeds
        sg.text_rep = hidden_states  # cache for residual

        if self.use_kg_encoder:
            graph_hidden = self.kg_adapter_kg_encoder(hidden_states, sg, attention_mask)

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
                    sg=sg if not self.use_kg_encoder else graph_hidden,
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
                if 'sa_sg_key_value' in layer_outputs.keys():
                    cache['sa_sg_key_value'] = layer_outputs['sa_sg_key_value']
                if 'prefix_sa_key_value' in layer_outputs.keys():
                    cache['prefix_sa_key_value'] = layer_outputs['prefix_sa_key_value']

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


class MistralKgAdapterForCausalLM(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralKgAdapterModel(config)
        self.vocab_size = config.vocab_size
        self.train_lm_head = config.train_lm_head
        if self.train_lm_head:
            self.kg_adapter_lm_lora = Lora_layer(r=16, lora_alpha=16 * 2, in_features=config.hidden_size,
                                     out_features=config.vocab_size)

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

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
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
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
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
        if self.train_lm_head:
            logits = logits + self.kg_adapter_lm_lora(hidden_states)
        logits = logits.float()

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

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, sg=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values:
            # past_length = past_key_values[0][0].shape[2]
            past_length = past_key_values[0]['sa_key_value'][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

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
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
