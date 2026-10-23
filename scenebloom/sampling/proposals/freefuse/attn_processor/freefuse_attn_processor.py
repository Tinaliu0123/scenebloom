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
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

from ..models.freefuse_transformer_flux import FreeFuseFluxAttention, _get_projections, _get_fused_projections, _get_qkv_projections, _get_add_projections
from peft.tuners.lora import LoraLayer


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class FreeFuseFluxAttnProcessor:
    """
    Attention processor that combines cross-attention for top-k token selection
    with concept attention for final similarity map calculation.
    
    This processor:
    1. Performs standard attention computation
    2. Uses cross-attention (Q-K dot product with RoPE) to select top-k image tokens
    3. Uses concept attention (hidden states inner product) on selected tokens for final sim map
    4. Computes concept similarity maps for LoRA mask generation
    
    This combines the best of both worlds:
    - Cross-attention provides better spatial localization for token selection
    - Concept attention provides more accurate semantic similarity for final maps
    """
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")
        self.cal_concept_sim_map = False

    def __call__(
        self,
        attn: "FreeFuseFluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None,
        concept_token_texts: Optional[Dict[str, List[str]]] = None,
        top_k_ratio: float = 0.1,
        eos_token_index: Optional[int] = None,
        background_token_positions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass that uses cross-attention for top-k token selection and 
        concept attention for final similarity map calculation.
        
        This processor combines the best of both worlds:
        - Cross-attention (Q-K dot product with RoPE) for selecting top-k image tokens
        - Concept attention (hidden states inner product) for computing final similarity maps
        
        Args:
            freefuse_token_pos_maps: Dict mapping lora_name -> [[positions for prompt 1], ...]
                                     Each position list contains token indices in encoder_hidden_states
            concept_token_texts: Optional dict for visualization, mapping lora_name -> token text list
            top_k_ratio: Ratio of top image tokens to use for concept attention enhancement (default: 0.025 = 2.5%)
            eos_token_index: Optional index of the first EOS token in the prompt, used for 
                            extracting background similarity map. Mutually exclusive with 
                            background_token_positions.
            background_token_positions: Optional list of token indices for user-defined background
                                        concept. These tokens are used to compute background sim_map.
                                        Mutually exclusive with eos_token_index.
        """
        # Standard QKV projections
        img_query, img_key, img_value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        img_query = attn.norm_q(img_query)
        img_key = attn.norm_k(img_key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            txt_img_query = torch.cat([encoder_query, img_query], dim=1)
            txt_img_key = torch.cat([encoder_key, img_key], dim=1)
            txt_img_value = torch.cat([encoder_value, img_value], dim=1)
          
        if image_rotary_emb is not None:
            txt_img_query = apply_rotary_emb(txt_img_query, image_rotary_emb, sequence_dim=1)
            txt_img_key = apply_rotary_emb(txt_img_key, image_rotary_emb, sequence_dim=1)

        attention_output = dispatch_attention_fn(
            txt_img_query,
            txt_img_key,
            txt_img_value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attention_output = attention_output.flatten(2, 3)
        attention_output = attention_output.to(img_query.dtype)

        if encoder_hidden_states is not None:
            # Split attention output into encoder (text) and image parts
            encoder_hidden_states_out, hidden_states_out = attention_output.split_with_sizes(
                [encoder_hidden_states.shape[1], attention_output.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )

            concept_sim_maps = None
            all_concept_keys = None
            all_concept_hidden_states = None
            txt_len = encoder_hidden_states.shape[1]
            img_len = hidden_states.shape[1]
            
            if freefuse_token_pos_maps is not None:
                all_concept_keys = {}
                all_concept_hidden_states = {}
                # Split txt_img_key and txt_img_query back to get components with RoPE applied
                # txt_img_key: (B, txt_len + img_len, heads, head_dim)
                encoder_key_rope = txt_img_key[:, :txt_len, :, :]  # (B, txt_len, heads, head_dim)
                img_key_rope = txt_img_key[:, txt_len:, :, :]  # (B, img_len, heads, head_dim)
                # img_query with RoPE applied
                img_query_rope = txt_img_query[:, txt_len:, :, :]  # (B, img_len, heads, head_dim)
                
                for lora_name, positions_list in freefuse_token_pos_maps.items():
                    # positions_list[0] contains positions for the first (and usually only) prompt
                    pos = positions_list[0]  # List[int]
                    if len(pos) > 0:
                        # Extract concept keys from encoder_key_rope at specified positions (for cross-attn top-k selection)
                        pos_tensor = torch.tensor(pos, device=encoder_key_rope.device)
                        concept_key = encoder_key_rope[:, pos_tensor, :, :]  # (B, concept_len, heads, head_dim)
                        all_concept_keys[lora_name] = concept_key
                        
                        # Extract concept embeddings from encoder_hidden_states_out (for concept attention)
                        concept_embeds = encoder_hidden_states_out[:, pos_tensor, :]  # (B, concept_len, hidden_dim)
                        all_concept_hidden_states[lora_name] = concept_embeds

            # Compute similarity maps using cross-attn for top-k selection + concept attention for final sim map
            if self.cal_concept_sim_map and all_concept_keys is not None and all_concept_hidden_states is not None:
                concept_sim_maps = {}
                batch_size = img_query_rope.shape[0]
                head_dim = img_query_rope.shape[-1]
                scale = 1.0 / 1000  # Same scale as DirectExtractCrossAttnSelfAttnEnhancedFluxAttnProcessor
                
                all_cross_attn_scores = {}
                for lora_name in all_concept_keys.keys():
                    concept_key = all_concept_keys[lora_name]
                    
                    # Compute cross-attention to select top-k image tokens
                    cross_attn_weights = torch.einsum('bihd,bjhd->bhij', img_query_rope, concept_key) * scale
                    cross_attn_weights = F.softmax(cross_attn_weights, dim=2)
                    cross_attn_scores = cross_attn_weights.mean(dim=1).mean(dim=-1)
                    all_cross_attn_scores[lora_name] = cross_attn_scores
                    
                for lora_name in all_concept_keys.keys():
                    cross_attn_scores = all_cross_attn_scores[lora_name] * len(all_concept_keys.keys())
                    # subtract the cross_attn_weights of every other LoRA
                    for other_lora_name in all_concept_keys.keys():
                        if other_lora_name != lora_name:
                            cross_attn_scores -= all_cross_attn_scores[other_lora_name]
                    k = max(1, int(img_len * top_k_ratio))
                    _, top_k_indices = torch.topk(cross_attn_scores, k, dim=-1)
                    
                    
                    # Save top-k indices for external visualization access
                    if not hasattr(self, 'top_k_indices_map'):
                        self.top_k_indices_map = {}
                    self.top_k_indices_map[lora_name] = top_k_indices.detach().cpu()
                    
                    # === Step 2: Use concept attention on top-k tokens for final sim map ===
                    # Extract core image tokens from hidden_states_out using top-k indices
                    # hidden_states_out: (B, img_len, hidden_dim)
                    top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, hidden_states_out.shape[-1])
                    core_image_tokens = torch.gather(hidden_states_out, dim=1, index=top_k_indices_expanded)  # (B, k, hidden_dim)
                    
                    # Compute self-modal similarity: core image tokens vs all image tokens
                    # This uses concept attention (hidden states inner product) similar to 
                    # DirectExtractConceptAttnSelfConceptEnhancedFluxAttnProcessor
                    # core_image_tokens: (B, k, hidden_dim)
                    # hidden_states_out: (B, img_len, hidden_dim)
                    self_modal_sim = core_image_tokens @ hidden_states_out.transpose(-1, -2)  # (B, k, img_len)
                    
                    # Average over core tokens to get final sim map
                    self_modal_sim_avg = self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, img_len, 1)
                    
                    # Apply softmax to normalize
                    concept_sim_map = F.softmax(self_modal_sim_avg / torch.tensor(4000, device=self_modal_sim_avg.device), dim=1)
                    concept_sim_maps[lora_name] = concept_sim_map
                
                # === Background sim_map extraction using cross-attn + concept attention ===
                # Mutual exclusion check: can only specify one of eos_token_index or background_token_positions
                if eos_token_index is not None and background_token_positions is not None and len(background_token_positions) > 0:
                    raise ValueError(
                        "Cannot specify both 'eos_token_index' and 'background_token_positions'. "
                        "Please use only one method for background detection."
                    )
                
                # Method 1: User-defined background tokens (priority)
                if background_token_positions is not None and len(background_token_positions) > 0:
                    # Extract background keys from encoder_key_rope at specified positions (for cross-attn top-k selection)
                    bg_pos_tensor = torch.tensor(background_token_positions, device=encoder_key_rope.device)
                    bg_key = encoder_key_rope[:, bg_pos_tensor, :, :]  # (B, num_bg_tokens, heads, head_dim)
                    
                    # Extract background embeddings from encoder_hidden_states_out (for concept attention)
                    bg_embeds = encoder_hidden_states_out[:, bg_pos_tensor, :]  # (B, num_bg_tokens, hidden_dim)
                    
                    # Step 1: Use cross-attention to select top-k image tokens for background
                    bg_cross_attn_weights = torch.einsum('bihd,bjhd->bhij', img_query_rope, bg_key) * scale
                    bg_cross_attn_weights = F.softmax(bg_cross_attn_weights, dim=2)  # (B, heads, img_len, num_bg_tokens)
                    
                    # Mean over heads and background tokens to get per-image-token scores
                    bg_cross_attn_scores = bg_cross_attn_weights.mean(dim=1).mean(dim=-1)  # (B, img_len)
                    
                    # Select top-k image tokens based on cross-attention scores
                    bg_k = max(1, int(img_len * top_k_ratio))
                    _, bg_top_k_indices = torch.topk(bg_cross_attn_scores, bg_k, dim=-1)  # (B, k)
                    
                    # Step 2: Use concept attention on top-k tokens for final sim map
                    bg_top_k_indices_expanded = bg_top_k_indices.unsqueeze(-1).expand(-1, -1, hidden_states_out.shape[-1])
                    bg_core_image_tokens = torch.gather(hidden_states_out, dim=1, index=bg_top_k_indices_expanded)  # (B, k, hidden_dim)
                    
                    # Compute self-modal similarity: core image tokens vs all image tokens
                    bg_self_modal_sim = bg_core_image_tokens @ hidden_states_out.transpose(-1, -2)  # (B, k, img_len)
                    
                    # Average over core tokens to get final sim map
                    bg_self_modal_sim_avg = bg_self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, img_len, 1)
                    
                    # Apply softmax to normalize
                    bg_sim_map = F.softmax(bg_self_modal_sim_avg / torch.tensor(4000, device=bg_self_modal_sim_avg.device), dim=1)
                    
                    # Add to concept_sim_maps with special key
                    concept_sim_maps['__bg__'] = bg_sim_map
                
                # Method 2: EOS token (fallback)
                elif eos_token_index is not None:
                    # Extract EOS key from encoder_key_rope (for cross-attn top-k selection)
                    eos_key = encoder_key_rope[:, eos_token_index:eos_token_index+1, :, :]  # (B, 1, heads, head_dim)
                    
                    # Extract EOS embedding from encoder_hidden_states_out (for concept attention)
                    eos_embed = encoder_hidden_states_out[:, eos_token_index:eos_token_index+1, :]  # (B, 1, hidden_dim)
                    
                    # Step 1: Use cross-attention to select top-k image tokens for EOS/background
                    eos_cross_attn_weights = torch.einsum('bihd,bjhd->bhij', img_query_rope, eos_key) * scale
                    eos_cross_attn_weights = F.softmax(eos_cross_attn_weights, dim=2)  # (B, heads, img_len, 1)
                    
                    # Mean over heads (EOS only has 1 token)
                    eos_cross_attn_scores = eos_cross_attn_weights.mean(dim=1).squeeze(-1)  # (B, img_len)
                    
                    # Select top-k image tokens based on cross-attention scores
                    eos_k = max(1, int(img_len * top_k_ratio))
                    _, eos_top_k_indices = torch.topk(eos_cross_attn_scores, eos_k, dim=-1)  # (B, k)
                    
                    # Step 2: Use concept attention on top-k tokens for final sim map
                    eos_top_k_indices_expanded = eos_top_k_indices.unsqueeze(-1).expand(-1, -1, hidden_states_out.shape[-1])
                    eos_core_image_tokens = torch.gather(hidden_states_out, dim=1, index=eos_top_k_indices_expanded)  # (B, k, hidden_dim)
                    
                    # Compute self-modal similarity: core image tokens vs all image tokens
                    eos_self_modal_sim = eos_core_image_tokens @ hidden_states_out.transpose(-1, -2)  # (B, k, img_len)
                    
                    # Average over core tokens to get final sim map
                    eos_self_modal_sim_avg = eos_self_modal_sim.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, img_len, 1)
                    
                    # Apply softmax to normalize
                    eos_sim_map = F.softmax(eos_self_modal_sim_avg / torch.tensor(4000, device=eos_self_modal_sim_avg.device), dim=1)
                    
                    # Add to concept_sim_maps with special key
                    concept_sim_maps['__eos__'] = eos_sim_map

            # Apply output projections
            hidden_states_out = attn.to_out[0](hidden_states_out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

            # Return 4 values for compatibility with FreeFuseFluxTransformerBlock
            # The third value (concept_embeds_and_rope_map) is empty since we extract from encoder_hidden_states
            return hidden_states_out, encoder_hidden_states_out, {}, concept_sim_maps
        else:
            print("Warning: DirectExtractCrossAttnSelfConceptEnhancedFluxAttnProcessor is only used in double stream block, but now encoder_hidden_states is None")
            return attention_output