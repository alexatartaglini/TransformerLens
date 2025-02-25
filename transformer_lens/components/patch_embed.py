"""Hooked Transformer Embed Component.

This module contains all the component :class:`Embed`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components import LayerNorm
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.HookedVLMConfig import HookedVLMConfig


# Embed & Unembed  # TODO: implement
class PatchEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig, HookedVLMConfig], additional_tokens: int = 0):
        super().__init__()
        if isinstance(cfg, HookedTransformerConfig) or isinstance(cfg, Dict):
            self.cfg = HookedTransformerConfig.unwrap(cfg)
        else:
            self.cfg = HookedVLMConfig.unwrap(cfg)
        self.W_E: Float[torch.Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(self.cfg.d_vocab + additional_tokens, self.cfg.d_model, dtype=self.cfg.dtype)
        )
        # Some models (e.g. Bloom) need post embedding layer norm
        if self.cfg.post_embedding_ln:
            self.ln = LayerNorm(self.cfg)

    def forward(
        self, tokens: Int[torch.Tensor, "batch pos"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        if self.cfg.post_embedding_ln:
            return self.ln(self.W_E[tokens, :])
        return self.W_E[tokens, :]
