"""
@author: alexatartaglini

Multimodal LLaMA weight conversion.

Module with a function for converting the weights of a Multimodal LLaMA model.
"""

from typing import cast

import einops
import torch

from transformer_lens.HookedVLMConfig import HookedVLMConfig


def convert_mllama_weights(mllama, cfg: HookedVLMConfig):
    language_model = mllama.language_model
    vision_model = mllama.vision_model

    state_dict = {}

    # Language model
    state_dict["language_model.embed.W_E"] = language_model.model.embed_tokens.weight

    # Some models with the Llama architecture use Grouped Query Attention, and so for these we need to modify
    # the state dict keys for the K/V attention weight/biases, prepending "_" to the key names.
    using_gqa = cfg.n_key_value_heads is not None
    gqa_uscore = "_" if using_gqa else ""
    # need a cast since MyPy isn't smart enough to realize that using_gqa implies n_key_value_heads is not None
    n_kv_heads = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads_language)

    # llama has no biases anywhere and deals with everything else roughly like
    # GPTNeoX with different names

    assert cfg.d_mlp_language is not None  # keep mypy happy

    for l in range(cfg.n_layers_language):
        state_dict[f"language_model.blocks.{l}.ln1.w"] = language_model.model.layers[l].input_layernorm.weight

        if l in cfg.cross_attn_layers:  # TODO: check if GQA, if so add "_" to the key names
            W_Q = language_model.model.layers[l].cross_attn.q_proj.weight
            W_K = language_model.model.layers[l].cross_attn.k_proj.weight
            W_V = language_model.model.layers[l].cross_attn.v_proj.weight
            
            if not cfg.load_in_4bit:
                W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads_language)
                W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
                W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)

            state_dict[f"language_model.blocks.{l}.cross_attn.W_Q"] = W_Q
            state_dict[f"language_model.blocks.{l}.cross_attn.{gqa_uscore}W_K"] = W_K
            state_dict[f"language_model.blocks.{l}.cross_attn.{gqa_uscore}W_V"] = W_V

            state_dict[f"language_model.blocks.{l}.cross_attn.b_Q"] = torch.zeros(
                cfg.n_heads_language, cfg.d_head_language, dtype=cfg.dtype, device=cfg.device
            )
            state_dict[f"language_model.blocks.{l}.cross_attn.{gqa_uscore}b_K"] = torch.zeros(
                n_kv_heads, cfg.d_head_language, dtype=cfg.dtype, device=cfg.device
            )
            state_dict[f"language_model.blocks.{l}.cross_attn.{gqa_uscore}b_V"] = torch.zeros(
                n_kv_heads, cfg.d_head_language, dtype=cfg.dtype, device=cfg.device
            )

            state_dict[f"language_model.blocks.{l}.cross_attn.W_O"] = language_model.model.layers[l].cross_attn.o_proj.weight

            state_dict[f"language_model.blocks.{l}.cross_attn.b_O"] = torch.zeros(
                cfg.d_model_language, dtype=cfg.dtype, device=cfg.device
            )

            state_dict[f"language_model.blocks.{l}.cross_attn.W_Q_norm.w"] = language_model.model.layers[l].cross_attn.q_norm.weight
            state_dict[f"language_model.blocks.{l}.cross_attn.W_K_norm.w"] = language_model.model.layers[l].cross_attn.k_norm.weight
            state_dict[f"language_model.blocks.{l}.cross_attn.attn_gate"] = language_model.model.layers[l].cross_attn_attn_gate
            state_dict[f"language_model.blocks.{l}.cross_attn.mlp_gate"] = language_model.model.layers[l].cross_attn_mlp_gate
        else:
            W_Q = language_model.model.layers[l].self_attn.q_proj.weight
            W_K = language_model.model.layers[l].self_attn.k_proj.weight
            W_V = language_model.model.layers[l].self_attn.v_proj.weight

            # in case of quantization,
            # parameters should stay as bitsandbytes.nn.modules.Params4bit
            if not cfg.load_in_4bit:
                W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads_language)
                W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
                W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)

            state_dict[f"language_model.blocks.{l}.attn.W_Q"] = W_Q
            state_dict[f"language_model.blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
            state_dict[f"language_model.blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

            state_dict[f"language_model.blocks.{l}.attn.b_Q"] = torch.zeros(
                cfg.n_heads_language, cfg.d_head_language, dtype=cfg.dtype, device=cfg.device
            )
            state_dict[f"language_model.blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
                n_kv_heads,
                cfg.d_head_language,
                dtype=cfg.dtype,
                device=cfg.device,
            )
            state_dict[f"language_model.blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
                n_kv_heads,
                cfg.d_head_language,
                dtype=cfg.dtype,
                device=cfg.device,
            )

            W_O = language_model.model.layers[l].self_attn.o_proj.weight

            if not cfg.load_in_4bit:
                W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads_language)

            state_dict[f"language_model.blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

            state_dict[f"language_model.blocks.{l}.attn.b_O"] = torch.zeros(
                cfg.d_model_language, dtype=cfg.dtype, device=cfg.device
            )

        state_dict[f"language_model.blocks.{l}.ln2.w"] = language_model.model.layers[l].post_attention_layernorm.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            state_dict[f"language_model.blocks.{l}.mlp.W_in"] = language_model.model.layers[l].mlp.up_proj.weight.T
            state_dict[f"language_model.blocks.{l}.mlp.W_gate"] = language_model.model.layers[l].mlp.gate_proj.weight.T
            state_dict[f"language_model.blocks.{l}.mlp.W_out"] = language_model.model.layers[l].mlp.down_proj.weight.T
        else:
            state_dict[f"language_model.blocks.{l}.mlp.W_in"] = language_model.model.layers[l].mlp.up_proj.weight
            state_dict[f"language_model.blocks.{l}.mlp.W_gate"] = language_model.model.layers[l].mlp.gate_proj.weight
            state_dict[f"language_model.blocks.{l}.mlp.W_out"] = language_model.model.layers[l].mlp.down_proj.weight

        state_dict[f"language_model.blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp_language, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"language_model.blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model_language, dtype=cfg.dtype, device=cfg.device
        )

    state_dict["language_model.ln_final.w"] = language_model.model.norm.weight

    state_dict["language_model.unembed.W_U"] = language_model.lm_head.weight.T
    state_dict["language_model.unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=cfg.device)

    # Vision model
    state_dict["vision_model.embed.W_patch"] = vision_model.patch_embedding.weight
    state_dict["vision_model.cls_token"] = vision_model.class_embedding
    state_dict["vision_model.gated_pos_embed.W_gate"] = vision_model.gated_positional_embedding.gate
    state_dict["vision_model.gated_pos_embed.w"] = vision_model.gated_positional_embedding.embedding
    state_dict["vision_model.gated_pos_embed.tile_embed.w"] = vision_model.gated_positional_embedding.tile_embedding.weight
    state_dict["vision_model.pre_tile_pos_embed.W_gate"] = vision_model.pre_tile_positional_embedding.gate
    state_dict["vision_model.pre_tile_pos_embed.w"] = vision_model.pre_tile_positional_embedding.embedding.weight
    state_dict["vision_model.post_tile_pos_embed.W_gate"] = vision_model.post_tile_positional_embedding.gate
    state_dict["vision_model.post_tile_pos_embed.w"] = vision_model.post_tile_positional_embedding.embedding.weight
    state_dict["vision_model.ln_pre.w"] = vision_model.layernorm_pre.weight
    state_dict["vision_model.ln_pre.b"] = vision_model.layernorm_pre.bias
    state_dict["vision_model.ln_post.w"] = vision_model.layernorm_post.weight
    state_dict["vision_model.ln_post.b"] = vision_model.layernorm_post.bias

    for l in range(cfg.n_layers_vision):
        state_dict[f"vision_model.transformer.blocks.{l}.ln1.w"] = vision_model.transformer.layers[l].input_layernorm.weight
        state_dict[f"vision_model.transformer.blocks.{l}.ln1.b"] = vision_model.transformer.layers[l].input_layernorm.bias

        W_Q = vision_model.transformer.layers[l].self_attn.q_proj.weight
        W_K = vision_model.transformer.layers[l].self_attn.k_proj.weight
        W_V = vision_model.transformer.layers[l].self_attn.v_proj.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads_vision)  # TODO: check if this is correct
            W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_heads_vision)  # TODO: check if this is correct
            W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_heads_vision)  # TODO: check if this is correct

        state_dict[f"vision_model.transformer.blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"vision_model.transformer.blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"vision_model.transformer.blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        state_dict[f"vision_model.transformer.blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads_vision, cfg.d_head_vision, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"vision_model.transformer.blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            cfg.n_heads_vision,
            cfg.d_head_vision,
            dtype=cfg.dtype,
            device=cfg.device,
        )
        state_dict[f"vision_model.transformer.blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            cfg.n_heads_vision,
            cfg.d_head_vision,
            dtype=cfg.dtype,
            device=cfg.device,
        )

        W_O = vision_model.transformer.layers[l].self_attn.o_proj.weight

        if not cfg.load_in_4bit:
            W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads_vision)  # TODO: check if this is correct

        state_dict[f"vision_model.transformer.blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

        state_dict[f"vision_model.transformer.blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model_vision, dtype=cfg.dtype, device=cfg.device
        )

        state_dict[f"vision_model.transformer.blocks.{l}.ln2.w"] = vision_model.transformer.layers[l].post_attention_layernorm.weight
        state_dict[f"vision_model.transformer.blocks.{l}.ln2.b"] = vision_model.transformer.layers[l].post_attention_layernorm.bias

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            state_dict[f"vision_model.transformer.blocks.{l}.mlp.W_in"] = vision_model.transformer.layers[l].mlp.fc1.weight.T
            state_dict[f"vision_model.transformer.blocks.{l}.mlp.b_in"] = vision_model.transformer.layers[l].mlp.fc1.bias
            state_dict[f"vision_model.transformer.blocks.{l}.mlp.W_out"] = vision_model.transformer.layers[l].mlp.fc2.weight.T
            state_dict[f"vision_model.transformer.blocks.{l}.mlp.b_out"] = vision_model.transformer.layers[l].mlp.fc2.bias
        else:
            state_dict[f"vision_model.transformer.blocks.{l}.mlp.W_in"] = vision_model.transformer.layers[l].mlp.fc1.weight
            state_dict[f"vision_model.transformer.blocks.{l}.mlp.b_in"] = vision_model.transformer.layers[l].mlp.fc1.bias
            state_dict[f"vision_model.transformer.blocks.{l}.mlp.W_out"] = vision_model.transformer.layers[l].mlp.fc2.weight
            state_dict[f"vision_model.transformer.blocks.{l}.mlp.b_out"] = vision_model.transformer.layers[l].mlp.fc2.bias

    # Global transformer
    for l in range(cfg.n_layers_global):
        state_dict[f"vision_model.global.blocks.{l}.ln1.w"] = vision_model.global_transformer.layers[l].input_layernorm.weight
        state_dict[f"vision_model.global.blocks.{l}.ln1.b"] = vision_model.global_transformer.layers[l].input_layernorm.bias

        W_Q = vision_model.global_transformer.layers[l].self_attn.q_proj.weight
        W_K = vision_model.global_transformer.layers[l].self_attn.k_proj.weight
        W_V = vision_model.global_transformer.layers[l].self_attn.v_proj.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads_vision)  # TODO: check if this is correct
            W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_heads_vision)  # TODO: check if this is correct
            W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_heads_vision)  # TODO: check if this is correct

        state_dict[f"vision_model.global.blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"vision_model.global.blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
        state_dict[f"vision_model.global.blocks.{l}.attn.{gqa_uscore}W_V"] = W_V

        state_dict[f"vision_model.global.blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads_vision, cfg.d_head_vision, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"vision_model.global.blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(
            cfg.n_heads_vision,
            cfg.d_head_vision,
            dtype=cfg.dtype,
            device=cfg.device,
        )
        state_dict[f"vision_model.global.blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(
            cfg.n_heads_vision,
            cfg.d_head_vision,
            dtype=cfg.dtype,
            device=cfg.device,
        )

        W_O = vision_model.global_transformer.layers[l].self_attn.o_proj.weight

        if not cfg.load_in_4bit:
            W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads_vision)  # TODO: check if this is correct

        state_dict[f"vision_model.global.blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

        state_dict[f"vision_model.global.blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model_vision, dtype=cfg.dtype, device=cfg.device
        )

        state_dict[f"vision_model.global.blocks.{l}.ln2.w"] = vision_model.global_transformer.layers[l].post_attention_layernorm.weight
        state_dict[f"vision_model.global.blocks.{l}.ln2.b"] = vision_model.global_transformer.layers[l].post_attention_layernorm.bias

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            state_dict[f"vision_model.global.blocks.{l}.mlp.W_in"] = vision_model.global_transformer.layers[l].mlp.fc1.weight.T
            state_dict[f"vision_model.global.blocks.{l}.mlp.b_in"] = vision_model.global_transformer.layers[l].mlp.fc1.bias
            state_dict[f"vision_model.global.blocks.{l}.mlp.W_out"] = vision_model.global_transformer.layers[l].mlp.fc2.weight.T
            state_dict[f"vision_model.global.blocks.{l}.mlp.b_out"] = vision_model.global_transformer.layers[l].mlp.fc2.bias
        else:
            state_dict[f"vision_model.global.blocks.{l}.mlp.W_in"] = vision_model.global_transformer.layers[l].mlp.fc1.weight
            state_dict[f"vision_model.global.blocks.{l}.mlp.b_in"] = vision_model.global_transformer.layers[l].mlp.fc1.bias
            state_dict[f"vision_model.global.blocks.{l}.mlp.W_out"] = vision_model.global_transformer.layers[l].mlp.fc2.weight
            state_dict[f"vision_model.global.blocks.{l}.mlp.b_out"] = vision_model.global_transformer.layers[l].mlp.fc2.bias

    # Connector
    # TODO: if not cfg.load_in_4bit, then we need to rearrange the weights and biases
    state_dict["connector.w"] = mllama.multi_modal_projector.weight
    state_dict["connector.b"] = mllama.multi_modal_projector.bias

    return state_dict
