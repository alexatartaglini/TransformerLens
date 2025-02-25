"""Hooked Transformer RMS Norm Pre Component.

This module contains all the component :class:`RMSNormPre`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.HookedVLMConfig import HookedVLMConfig


class RMSNormPre(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig, HookedVLMConfig]):
        """RMSNormPre - LayerNormPre without the centering and bias (RMS = Root Mean Square)"""
        super().__init__()
        if isinstance(cfg, HookedTransformerConfig) or isinstance(cfg, Dict):
            self.cfg = HookedTransformerConfig.unwrap(cfg)
        else:
            self.cfg = HookedVLMConfig.unwrap(cfg)
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        return self.hook_normalized(x / scale).to(self.cfg.dtype)  # [batch, pos, length]
