# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from transformers import Cache, QuantizedCache
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.phi3.modeling_phi3 import Phi3Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention


def get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Extracts the query states from a given attention module and hidden states tensor.

    This function supports multiple attention module types: Phi3Attention, Qwen3Attention, Gemma3Attention,
    GPTNeoXAttention, and Llama-like modules. It handles the appropriate projection and reshaping to obtain the query states
    in the expected format.

    Parameters
    ----------
    module : nn.Module
        The attention module from which to extract query states.
    hidden_states : torch.Tensor
        The input hidden states of shape (batch_size, seq_len, hidden_dim).

    Returns
    -------
    query_states : torch.Tensor
        The extracted query states of shape (batch_size, num_heads, seq_len, head_dim).
    """
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    # Handle different attribute names for head dimension across architectures
    head_dim = getattr(module, 'head_dim', None) or getattr(module, 'head_size', hidden_states.shape[-1] // num_heads)

    if hasattr(module, 'qkv_proj'):
        # Phi3-style combined QKV projection
        qkv = module.qkv_proj(hidden_states)
        query_states = qkv[..., : num_heads * head_dim]
    elif hasattr(module, "query_key_value"):
        # GPTNeoX-style combined QKV projection
        qkv = module.query_key_value(hidden_states)
        # GPTNeoX has num_heads * head_dim * 3 output
        query_states = qkv[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        # Llama-like separate Q projection
        query_states = module.q_proj(hidden_states)
    else:
        raise NotImplementedError(f"Press not yet implemented for {module.__class__}.")

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    # Support for Qwen3 and Gemma3 QK norm
    from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
    from transformers.models.phi3.modeling_phi3 import Phi3Attention
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    if isinstance(module, (Qwen3Attention, Gemma3Attention)):
        query_states = module.q_norm(query_states)

    return query_states


def get_prerope_key_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Extracts the key states from a given attention module and hidden states tensor.

    This function supports multiple attention module types: Phi3Attention, Qwen3Attention, Gemma3Attention,
    GPTNeoXAttention, and Llama-like modules. It handles the appropriate projection and reshaping to obtain the key states
    in the expected format.

    Parameters
    ----------
    module : nn.Module
        The attention module from which to extract key states.
    hidden_states : torch.Tensor
        The input hidden states of shape (batch_size, seq_len, hidden_dim).

    Returns
    -------
    key_states : torch.Tensor
        The extracted key states of shape (batch_size, num_heads, seq_len, head_dim).
    """
    bsz, k_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    # Handle different attribute names for head dimension across architectures
    head_dim = getattr(module, 'head_dim', None) or getattr(module, 'head_size', hidden_states.shape[-1] // num_heads)
    # Handle models without GQA
    num_kv_heads = getattr(module.config, 'num_key_value_heads', num_heads)

    from transformers.models.phi3.modeling_phi3 import Phi3Attention
    if hasattr(module, 'qkv_proj'):
        # Phi3-style combined QKV projection
        qkv = module.qkv_proj(hidden_states)
        query_pos = num_heads * head_dim
        key_states = qkv[..., query_pos : query_pos + num_kv_heads * head_dim]
    elif hasattr(module, "query_key_value"):
        # GPTNeoX-style combined QKV projection
        qkv = module.query_key_value(hidden_states)
        query_pos = num_heads * head_dim
        key_states = qkv[..., query_pos : query_pos + num_kv_heads * head_dim]
    elif hasattr(module, "k_proj"):
        # Llama-like separate K projection
        key_states = module.k_proj(hidden_states)
    else:
        raise NotImplementedError(f"Press not yet implemented for {module.__class__}.")

    key_states = key_states.view(bsz, k_len, -1, head_dim).transpose(1, 2)

    # Support for Qwen3 and Gemma3 QK norm
    from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    if isinstance(module, (Qwen3Attention, Gemma3Attention)):
        key_states = module.k_norm(key_states)
    return key_states


def dequantize_layer(cache_layer) -> tuple[torch.Tensor, torch.Tensor]:
    keys = cache_layer._dequantize(cache_layer._quantized_keys)
    values = cache_layer._dequantize(cache_layer._quantized_values)
    return keys, values


def extract_keys_and_values(cache: Cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts the keys and values from a given cache layer,
    handling both quantized and unquantized caches.
    """
    if isinstance(cache, QuantizedCache):
        keys, values = dequantize_layer(cache.layers[layer_idx])
    else:
        keys = cache.layers[layer_idx].keys
        values = cache.layers[layer_idx].values
    return keys, values
