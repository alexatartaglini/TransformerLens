"""
@author: alexatartaglini

Hooked VLM Cross Attention Component.

This module contains all the component :class:`CrossAttention`.
"""
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers.utils import is_bitsandbytes_available

from transformer_lens.components import AbstractAttention, RMSNorm
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.HookedVLMConfig import HookedVLMConfig

if is_bitsandbytes_available():
    from bitsandbytes.nn.modules import Params4bit


class CrossAttention(AbstractAttention):
    def __init__(
        self,
        cfg: Union[Dict, HookedTransformerConfig, HookedVLMConfig],
        attn_type: str = "global",
        layer_id: Optional[int] = None,
    ):
        super().__init__(cfg, attn_type, layer_id)
        if isinstance(cfg, HookedTransformerConfig) or isinstance(cfg, Dict):
            self.cfg = HookedTransformerConfig.unwrap(cfg)
        else:
            self.cfg = HookedVLMConfig.unwrap(cfg)

        super().__init__(cfg, attn_type, layer_id)
        self.repeat_kv_heads = cfg.n_heads // cfg.n_key_value_heads
        self.W_O = nn.Parameter(
            torch.empty(
                self.cfg.d_model,
                self.cfg.d_model,
                dtype=cfg.dtype,
            )
        )
        self._W_K = nn.Parameter(
            torch.empty(
                cfg.n_key_value_heads,
                self.cfg.d_model,
                self.cfg.d_head,
                dtype=cfg.dtype,
            )
        )
        self._W_V = nn.Parameter(
            torch.empty(
                cfg.n_key_value_heads,
                self.cfg.d_model,
                self.cfg.d_head,
                dtype=cfg.dtype,
            )
        )
        self._b_K = nn.Parameter(
            torch.zeros(cfg.n_key_value_heads, self.cfg.d_head, dtype=cfg.dtype)
        )
        self._b_V = nn.Parameter(
            torch.zeros(cfg.n_key_value_heads, self.cfg.d_head, dtype=cfg.dtype)
        )
        self.b_O = nn.Parameter(
            torch.zeros(self.cfg.d_model, dtype=cfg.dtype)
        )

        self.W_Q_norm = RMSNorm(cfg, length=self.cfg.d_head)
        self.W_K_norm = RMSNorm(cfg, length=self.cfg.d_head)

        self.attn_gate = nn.Parameter(torch.ones(1, dtype=cfg.dtype))
        self.mlp_gate = nn.Parameter(torch.ones(1, dtype=cfg.dtype))
