# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import torch

from ..utils import is_torch_npu_available


_is_torch_npu_available = is_torch_npu_available()


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


def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor, value: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda or Ascend NPU
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    #   - key head_dim == value head_dim <= 256 (otherwise it will fall back to the math kernel)
    # 2.xpu
    #   - torch version >= 2.8
    try:
        from ..utils.import_utils import is_torch_greater_or_equal
        _is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
    except ImportError:
        _is_torch_greater_or_equal_than_2_5 = True
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None and key.shape[-1] == value.shape[-1] <= 256


def sdpa_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    dropout_p=0.0,
    is_causal=None,
    rope_rotary_cos_sin=None,
    past_key_value=None, # Add this
    **kwargs,
) -> torch.Tensor:
    """
    Standard forward for SDPA attention.
    """
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if is_causal is None:
        is_causal = False
        
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    if attention_mask is not None:
        attention_mask = attention_mask.contiguous()

    # Handle Grouped-Query Attention (GQA)
    # Skip expansion during tracing as attention_plugin handles it natively
    if not torch.jit.is_tracing() and query.shape[1] != key.shape[1]:
        num_q_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        if num_q_heads % num_kv_heads == 0:
            n_rep = num_q_heads // num_kv_heads
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)
        else:
            raise ValueError(f"Query heads ({num_q_heads}) must be divisible by KV heads ({num_kv_heads}) for GQA")

    if torch.jit.is_tracing():
        # Call custom trt::attention_plugin operator during tracing
        from tensorrt_edgellm.llm_models.layers.attention_plugin import attention_plugin
        
        # Retrieval logic updated for transformers v5
        num_q_heads = getattr(module.config, "num_attention_heads", 0)
        num_kv_heads = getattr(module.config, "num_key_value_heads", num_q_heads)
        head_size = getattr(module, "head_dim", 0)
        sliding_window_size = getattr(module, "sliding_window", -1)
        if sliding_window_size is None: sliding_window_size = -1
        
        # We need context_lengths and kvcache_start_index from kwargs
        context_lengths = kwargs.get("context_lengths")
        kvcache_start_index = kwargs.get("kvcache_start_index")
        
        # If they are missing, create dummy ones for tracing
        if context_lengths is None:
            context_lengths = torch.tensor([query.shape[2]], dtype=torch.int32, device=query.device)
        if kvcache_start_index is None:
            kvcache_start_index = torch.tensor([0], dtype=torch.int32, device=query.device)

        # Transpose and reshape q, k, v to [B, S, H*D] for the plugin
        q_plugin = query.transpose(1, 2).reshape(query.shape[0], query.shape[2], -1).contiguous()
        k_plugin = key.transpose(1, 2).reshape(key.shape[0], key.shape[2], -1).contiguous()
        v_plugin = value.transpose(1, 2).reshape(value.shape[0], value.shape[2], -1).contiguous()
        
        attn_output, present_key_value = attention_plugin(
            q_plugin,
            k_plugin,
            v_plugin,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
            num_q_heads,
            num_kv_heads,
            False, # enable_tree_attention (handled by separate branch if needed)
            head_size,
            False, # enable_fp8_kv_cache
            sliding_window_size
        )
        return attn_output, present_key_value
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
