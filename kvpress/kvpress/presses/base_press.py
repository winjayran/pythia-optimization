# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch
from torch import nn
from transformers import (
    Gemma3ForConditionalGeneration,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    PreTrainedModel,
    QuantizedCache,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
)

from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = (
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Gemma3ForConditionalGeneration,
    GPTNeoXForCausalLM,
)


@dataclass
class BasePress:
    """
    Base class for all KV cache compression methods.

    This class provides the foundation for implementing various key-value cache compression
    techniques. Subclasses must implement the `compress` method to define their specific
    compression logic.

    The compression is applied only during pre-filling (not during generation).
    """

    def post_init_from_model(self, model: PreTrainedModel):
        """
        Optional method to initialize press parameters from the model
        """
        pass

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core logic of the compression method.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where compression is applied.
        hidden_states : torch.Tensor
            Hidden states of the current layer with shape (batch_size, seq_len, hidden_dim).
            These represent the input to the attention layer.
        keys : torch.Tensor
            Key tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are keys ready for compression.
        values : torch.Tensor
            Value tensors from the KV cache with shape (batch_size, num_kv_heads, seq_len, head_dim).
            These are values ready for compression.
        attentions : torch.Tensor
            Attention weights from the layer with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if attention weights are not computed or needed.
        kwargs : dict
            Additional keyword arguments from the forward pass.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the compressed keys and values tensors. The returned tensors
            should have reduced sequence length dimension compared to the input tensors.
        """

        raise NotImplementedError("compress method must be implemented in subclass")

    def forward_hook(self, module: nn.Module, args: tuple, kwargs: dict, output: tuple):
        """
        Default forward hook called after the forward pass of an attention layer.

        This hook automatically applies compression during the pre-filling phase by:
        1. Checking if we're still in pre-filling (not generation) phase
        2. Extracting keys and values from the cache (handling quantization)
        3. Calling the compress method to reduce the cache size
        4. Updating the cache with compressed keys and values

        The hook ensures compression is only applied during pre-filling and correctly
        handles both quantized and unquantized caches.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        args : tuple
            Positional arguments to the forward pass (hidden_states, attention_mask, etc.).
            For GPTNeoX: (hidden_states, attention_mask, layer_past, cache_position, position_embeddings)
            For other models: typically empty or (hidden_states,)
        kwargs : dict
            Keyword arguments passed to the attention layer's forward method.
            For standard models: contains hidden_states, past_key_values, cache_position, etc.
            For GPTNeoX: contains additional kwargs like FlashAttentionKwargs
        output : tuple
            Output from the attention layer's forward pass. Contains:
            - [0]: Hidden states output
            - [1]: Attention weights (may be None)

        Returns
        -------
        tuple
            The potentially modified output from the forward pass. This
            is the same as the input output, but the underlying cache has been compressed in-place.
        """
        # Handle different architectures: GPTNeoX passes hidden_states as positional arg
        # while other models pass it in kwargs
        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif args and len(args) > 0:
            hidden_states = args[0]
        else:
            return output

        # Get cache from kwargs (standard models use past_key_values, GPTNeoX uses layer_past)
        if "past_key_values" in kwargs:
            cache = kwargs["past_key_values"]
        elif "layer_past" in kwargs:
            cache = kwargs["layer_past"]  # GPTNeoX passes cache as layer_past in kwargs
        elif len(args) > 2 and args[2] is not None:
            cache = args[2]  # fallback to positional arg
        else:
            return output

        # Get cache_position for checking prefill vs decode
        if "cache_position" in kwargs:
            cache_position = kwargs["cache_position"]
        elif len(args) > 3 and args[3] is not None:
            cache_position = args[3]  # cache_position for GPTNeoX
        else:
            cache_position = None

        cache_layer = cache.layers[module.layer_idx]
        q_len = hidden_states.shape[1]

        # Don't compress after pre-filling
        if cache_position is not None and cache_position[-1] > q_len:
            return output

        keys, values = extract_keys_and_values(cache, module.layer_idx)

        keys, values = self.compress(module, hidden_states, keys, values, output[1] if len(output) > 1 else None, kwargs)

        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.values = torch.zeros(0, dtype=values.dtype, device=values.device)  # type: ignore[index]
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.

        This method registers forward hooks on all attention layers of the model to enable
        automatic KV cache compression during the pre-filling phase. The hooks are automatically
        removed when exiting the context manager.

        Apply this context manager during the pre-filling phase to compress the context.

        Parameters
        ----------
        model : PreTrainedModel
            The transformer model to apply compression to.

        Examples
        --------
        >>> from kvpress import KnormPress
        >>> press = KnormPress(compression_ratio=0.5)
        >>> with press(model):
        ...     # Forward pass with compression applied
        ...     outputs = model(input_ids, past_key_values=cache)
        """
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")

        if isinstance(model, Gemma3ForConditionalGeneration):
            logger.warning_once("Compression in Gemma3 is only applied to layer without sliding window attention")

        self.post_init_from_model(model)
        hooks = []
        try:
            # Handle GPTNeoX architecture (e.g., Pythia models)
            if isinstance(model, GPTNeoXForCausalLM):
                language_model = model.gpt_neox
                for layer in language_model.layers:
                    # GPTNeoX uses 'attention' instead of 'self_attn'
                    layer.attention.rotary_emb = language_model.rotary_emb
                    hooks.append(layer.attention.register_forward_hook(self.forward_hook, with_kwargs=True))
            else:
                # Handle standard architecture (Llama, Mistral, etc.)
                language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
                for layer in language_model.layers:
                    if isinstance(model, Gemma3ForConditionalGeneration) and layer.self_attn.is_sliding:
                        # Skip layers with sliding window attention, only for Gemma3
                        continue
                    layer.self_attn.rotary_emb = language_model.rotary_emb
                    hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()
