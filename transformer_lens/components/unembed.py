"""Hooked Transformer Unembed Component.

This module contains all the component :class:`Unembed`.
"""

from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.HookedVLMConfig import HookedVLMConfig
from transformer_lens.utilities.addmm import batch_addmm


class Unembed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig, HookedVLMConfig]):
        super().__init__()
        if isinstance(cfg, HookedTransformerConfig) or isinstance(cfg, Dict):
            self.cfg = HookedTransformerConfig.unwrap(cfg)
        else:
            self.cfg = HookedVLMConfig.unwrap(cfg)
        # Note that there's a separate variable for d_vocab_out and d_vocab (the input vocab size). For language tasks these are always the same, but for algorithmic tasks we may want them to be different.
        self.W_U: Float[torch.Tensor, "d_model d_vocab_out"] = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_vocab_out, dtype=self.cfg.dtype)
        )
        self.b_U: Float[torch.Tensor, "d_vocab_out"] = nn.Parameter(
            torch.zeros(self.cfg.d_vocab_out, dtype=self.cfg.dtype)
        )

    def forward(
        self, residual: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_vocab_out"]:
        return batch_addmm(self.b_U, self.W_U, residual)
