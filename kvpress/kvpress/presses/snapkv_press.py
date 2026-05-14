# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states


@dataclass
class SnapKVPress(ScorerPress):
    """
    SnapKV: Attention-based KV cache compression using recent token patterns.

    Uses attention patterns of the most recent tokens to estimate importance
    of previous key-value pairs.

    Based on SnapKV (https://arxiv.org/abs/2404.14469).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    window_size : int, default=64
        Number of recent tokens to use for computing attention-based importance scores.
    kernel_size : int, default=5
        Size of the pooling kernel applied to attention weights for smoothing.
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """

        bsz, _, k_len, _ = keys.shape
        num_heads = module.config.num_attention_heads
        # Handle different attribute names for head dimension across architectures
        # GPTNeoX uses head_size, Llama/Mistral use head_dim
        head_dim = getattr(module, 'head_dim', None) or getattr(module, 'head_size', keys.shape[-1])
        # Handle models without GQA (like GPTNeoX) - they have same number of KV heads as query heads
        num_kv_heads = getattr(module.config, 'num_key_value_heads', num_heads)
        num_key_value_groups = num_heads // num_kv_heads

        # Get last window_size queries
        query_states = get_prerope_query_states(module, hidden_states[:, -window_size:])

        # Apply RoPE - handle partial rotary embedding for GPTNeoX
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]

        # Check if using partial rotary embedding (GPTNeoX)
        rotary_ndims = getattr(module, 'rotary_ndims', cos.shape[-1])
        if rotary_ndims < head_dim:
            # Apply RoPE only to the first rotary_ndims dimensions
            query_rot, query_pass = query_states[..., :rotary_ndims], query_states[..., rotary_ndims:]
            query_rot = (query_rot * cos.unsqueeze(1)) + (rotate_half(query_rot) * sin.unsqueeze(1))
            query_states = torch.cat([query_rot, query_pass], dim=-1)
        else:
            # Full rotary embedding
            query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=k_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, k_len, _ = keys.shape
        # Handle models without GQA (like GPTNeoX)
        num_heads = module.config.num_attention_heads
        num_kv_heads = getattr(module.config, 'num_key_value_heads', num_heads)
        num_key_value_groups = num_heads // num_kv_heads

        assert (
            hidden_states.shape[1] > self.window_size
        ), f"Query length {hidden_states.shape[1]} should be greater than the window size {self.window_size}"

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, k_len - self.window_size)
        scores = scores.mean(2)

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item() + 1)

        return scores
