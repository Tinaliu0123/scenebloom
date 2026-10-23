# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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
#
# Modified by the SceneBloom authors: package-relative imports and comments
# translated to English.

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_npu_available, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle

from ..tuner.freefuse_lora_layer import FreeFuseLinear


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_projections(attn: "FreeFuseFluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value

def _get_add_projections(attn: "FreeFuseFluxAttention", encoder_hidden_states=None):
    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn: "FreeFuseFluxAttention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "FreeFuseFluxAttention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class FluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "FreeFuseFluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxIPAdapterAttnProcessor(torch.nn.Module):
    """Flux Attention processor for IP-Adapter."""

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self, hidden_size: int, cross_attention_dim: int, num_tokens=(4,), scale=1.0, device=None, dtype=None
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [
                nn.Linear(cross_attention_dim, hidden_size, bias=True, device=device, dtype=dtype)
                for _ in range(len(num_tokens))
            ]
        )
        self.to_v_ip = nn.ModuleList(
            [
                nn.Linear(cross_attention_dim, hidden_size, bias=True, device=device, dtype=dtype)
                for _ in range(len(num_tokens))
            ]
        )

    def __call__(
        self,
        attn: "FreeFuseFluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[List[torch.Tensor]] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        ip_query = query

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # IP-adapter
            ip_attn_output = torch.zeros_like(hidden_states)

            for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
                ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
            ):
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, attn.head_dim)
                ip_value = ip_value.view(batch_size, -1, attn.heads, attn.head_dim)

                current_ip_hidden_states = dispatch_attention_fn(
                    ip_query,
                    ip_key,
                    ip_value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )
                current_ip_hidden_states = current_ip_hidden_states.reshape(batch_size, -1, attn.heads * attn.head_dim)
                current_ip_hidden_states = current_ip_hidden_states.to(ip_query.dtype)
                ip_attn_output += scale * current_ip_hidden_states

            return hidden_states, encoder_hidden_states, ip_attn_output
        else:
            return hidden_states


class FreeFuseFluxAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = FluxAttnProcessor
    _available_processors = [
        FluxAttnProcessor,
        FluxIPAdapterAttnProcessor,
    ]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = torch.nn.ModuleList([])
            self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters]
        # if len(unused_kwargs) > 0:
        #     logger.warning(
        #         f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
        #     )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        self.attn = FreeFuseFluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


@maybe_allow_in_graph
class FreeFuseFluxTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FreeFuseFluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        concept_embeds_and_rope_map = kwargs.pop('concept_embeds_and_rope_map', None)
        concept_embeds_gate = {}
        for lora_name, concept_embeds_and_rope in concept_embeds_and_rope_map.items():
            norm_concept_encoder_hidden_states, concept_gate_msa, concept_shift_mlp, concept_scale_mlp, concept_gate_mlp = self.norm1_context(
                concept_embeds_and_rope['embeds'], emb=temb
            )
            concept_embeds_gate[lora_name] = (concept_embeds_and_rope['embeds'], concept_gate_msa, concept_shift_mlp, concept_scale_mlp, concept_gate_mlp)
            concept_embeds_and_rope['embeds'] = norm_concept_encoder_hidden_states


        joint_attention_kwargs = joint_attention_kwargs or {}


        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
            concept_embeds_and_rope_map=concept_embeds_and_rope_map,
        )

        concept_sim_maps = None

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3 and isinstance(attention_outputs[2], torch.Tensor):
            attn_output, context_attn_output, ip_attn_output = attention_outputs
        elif len(attention_outputs) == 4 and isinstance(attention_outputs[2], dict):
            attn_output, context_attn_output, concept_embeds_and_rope_map, concept_sim_maps = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3 and isinstance(attention_outputs[2], torch.Tensor):
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        
        for lora_name, concept_embeds_and_rope in concept_embeds_and_rope_map.items():
            concept_hidden_states, concept_gate_msa, concept_shift_mlp, concept_scale_mlp, concept_gate_mlp = concept_embeds_gate[lora_name]

            concept_attn_output = concept_gate_msa.unsqueeze(1) * concept_embeds_and_rope['embeds']
            concept_hidden_states = concept_hidden_states + concept_attn_output

            norm_concept_hidden_states = self.norm2_context(concept_hidden_states)
            norm_concept_hidden_states = norm_concept_hidden_states * (1 + concept_scale_mlp[:, None]) + concept_shift_mlp[:, None]

            prev_activated = None
            for module in self.ff_context.modules():
                if isinstance(module, FreeFuseLinear):
                    prev_activated = module.active_adapters
                    module.disable_masks()
                    module.set_adapter([lora_name])
            concept_ff_output = self.ff_context(norm_concept_hidden_states)
            concept_hidden_states = concept_hidden_states + concept_gate_mlp.unsqueeze(1) * concept_ff_output
            if concept_hidden_states.dtype == torch.float16:
                concept_hidden_states = concept_hidden_states.clip(-65504, 65504)
            
            if prev_activated is not None:
                for module in self.ff_context.modules():
                    if isinstance(module, FreeFuseLinear):
                        module.set_adapter(prev_activated)
                        module.enable_masks()

            concept_embeds_and_rope['embeds'] = concept_hidden_states

        return encoder_hidden_states, hidden_states, concept_embeds_and_rope_map, concept_sim_maps


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class FluxTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "img_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
            "txt_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FreeFuseFluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
    
    def flatten_mask(self, freefuse_mask: Dict[str, torch.Tensor]):
        # flatten the mask to 2D
        freefuse_mask_flattend = {}
        for name, mask in freefuse_mask.items():
            freefuse_mask_flattend[name] = mask.reshape(mask.shape[0], -1, 1)
        return freefuse_mask_flattend

    def flatten_mask_add_text(self, freefuse_mask: Dict[str, torch.Tensor], text_length=512):
        # flatten the mask to 2D and add the text part
        freefuse_mask_flattened = {}
        for name, mask in freefuse_mask.items():
            freefuse_mask_flattened[name] = torch.cat((torch.ones(mask.shape[0], text_length, 1).to(mask.device), mask.reshape(mask.shape[0], -1, 1)), dim=1)
        return freefuse_mask_flattened

    def _convert_to_01_mask(self, freefuse_mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert freefuse_mask to a 0-1 mask, using the same max_indices logic as FluxAttnProcessor2_0.
        Not handled: the eos map needs special treatment (its mask should be 1 so that every
        LoRA can act there). The current pipeline does not call this method directly.

        Args:
            freefuse_mask: the original freefuse_mask dict
            
        Returns:
            the converted 0-1 mask dict
        """
        if not freefuse_mask:
            return freefuse_mask
        
        # collect and stack the masks of all LoRAs
        lora_names = list(freefuse_mask.keys())
        lora_target_maps = []
        
        for lora_name in lora_names:
            mask = freefuse_mask[lora_name]
            # flatten the mask to B x H*W
            flattened_mask = mask.reshape(mask.shape[0], -1)
            lora_target_maps.append(flattened_mask)
        
        # stack all LoRA masks: B x Loras x H*W
        lora_target_map = torch.stack(lora_target_maps, dim=1)
        
        # index of the LoRA with the maximum at each position: B x H*W
        max_indices = lora_target_map.argmax(dim=1)
        
        # build a 0-1 mask for each LoRA
        converted_mask = {}
        for i, lora_name in enumerate(lora_names):
            original_shape = freefuse_mask[lora_name].shape
            # create the 0-1 mask
            mask_01 = torch.zeros_like(lora_target_map[:, 0], dtype=torch.float32)
            mask_01[max_indices == i] = 1.0
            
            # restore the original shape
            mask_01 = mask_01.reshape(original_shape)
            converted_mask[lora_name] = mask_01
        
        return converted_mask

    def set_freefuse_masks(self, freefuse_mask: Dict[str, torch.Tensor], set_to_lora=True, derive_01_mask=False, text_length=512):
        self.freefuse_mask=freefuse_mask
        
        # with derive_01_mask on, convert freefuse_mask to a 0-1 mask
        if derive_01_mask and freefuse_mask:
            freefuse_mask = self._convert_to_01_mask(freefuse_mask)
        
        # the mask is not applied before ensemble/derive01: until then it is not binary
        # but takes values in (0, 1), mostly close to 0
        if set_to_lora:
            for name, module in self.named_modules():
                if isinstance(module, FreeFuseLinear):
                    if 'single_transformer_blocks' in name:
                        if any(s in name for s in ['to_q', 'to_k', 'to_v', 'proj_mlp', 'proj_out']):
                            module.set_freefuse_masks(self.flatten_mask_add_text(freefuse_mask, text_length))
                    elif 'transformer_blocks' in name:
                        if any(s in name for s in ['to_q', 'to_k', 'to_v', 'to_out', 'ff']) and 'ff_context' not in name:
                            module.set_freefuse_masks(self.flatten_mask(freefuse_mask))

    def set_freefuse_token_pos_maps(self, freefuse_token_pos_maps: Dict[str, List[int]]):
        # for the context part only: no LoRA may affect another LoRA's trigger tokens
        self.freefuse_token_pos_maps = freefuse_token_pos_maps
        for name, module in self.named_modules():
            if isinstance(module, FreeFuseLinear):
                if 'single_transformer_blocks' in name:
                    if any(s in name for s in ['to_q', 'to_k', 'to_v', 'proj_mlp', 'proj_out']):
                        module.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
                elif 'transformer_blocks' in name:
                    if any(s in name for s in ['add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out', 'ff_context']):
                        module.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
    
    def reset_freefuse(self):
        self.reset_attn_map()
        # drop any existing FreeFuse mask
        if hasattr(self, "freefuse_mask"):
            del self.freefuse_mask
        
        if hasattr(self, "freefuse_token_pos_maps"):
            del self.freefuse_token_pos_maps
        
        if hasattr(self, "cross_attn_map"):
            del self.cross_attn_map
            self.count = 0

        for _, module in self.named_modules():
            if isinstance(module, FreeFuseLinear):
                if hasattr(module, 'freefuse_mask'):
                    del module.freefuse_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        concept_embeds_map: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        concept_embeds_and_rope_map = {}
        if concept_embeds_map is not None:
            for lora_name, embeds in concept_embeds_map.items():
                concept_embeds_and_rope_map[lora_name] = {'embeds': self.context_embedder(embeds)}

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.pos_embed(ids)
        
        for lora_name, embeds_and_rope in concept_embeds_and_rope_map.items():
            concept_embeds_and_rope_map[lora_name]['rope'] = self.pos_embed(torch.cat((torch.zeros(embeds_and_rope['embeds'].shape[1], 3, device=embeds_and_rope['embeds'].device), img_ids)))

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # Extract attention bias blocks configuration for block-specific attention mask application
        attention_bias_blocks = None
        base_attention_mask = None
        if joint_attention_kwargs is not None:
            attention_bias_blocks = joint_attention_kwargs.get('attention_bias_blocks', None)
            base_attention_mask = joint_attention_kwargs.get('attention_mask', None)

        concept_sim_maps = None
        for index_block, block in enumerate(self.transformer_blocks):
            # Prepare block-specific joint_attention_kwargs with filtered attention_mask
            block_joint_attention_kwargs = joint_attention_kwargs
            if base_attention_mask is not None and attention_bias_blocks is not None:
                block_name = f"transformer_blocks.{index_block}"
                if block_name not in attention_bias_blocks:
                    # This block should not use attention bias, remove it from kwargs
                    block_joint_attention_kwargs = {k: v for k, v in joint_attention_kwargs.items() if k != 'attention_mask'}
            
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states, concept_embeds_and_rope_map, cur_concept_sim_maps = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    block_joint_attention_kwargs,
                    concept_embeds_and_rope_map,
                )

            else:
                encoder_hidden_states, hidden_states, concept_embeds_and_rope_map, cur_concept_sim_maps = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_joint_attention_kwargs,
                    concept_embeds_and_rope_map=concept_embeds_and_rope_map,
                )
            
            if cur_concept_sim_maps is not None:
                concept_sim_maps = cur_concept_sim_maps

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            # Prepare block-specific joint_attention_kwargs with filtered attention_mask
            block_joint_attention_kwargs = joint_attention_kwargs
            if base_attention_mask is not None and attention_bias_blocks is not None:
                block_name = f"single_transformer_blocks.{index_block}"
                if block_name not in attention_bias_blocks:
                    # This block should not use attention bias, remove it from kwargs
                    block_joint_attention_kwargs = {k: v for k, v in joint_attention_kwargs.items() if k != 'attention_mask'}
            
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    block_joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output,concept_sim_maps

        print("Warning: 默认应该调用returen dict=False，否则不会返回concept_sim_maps")
        return Transformer2DModelOutput(sample=output)
