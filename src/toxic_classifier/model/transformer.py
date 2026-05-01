"""Pre-LayerNorm Transformer encoder, hand-rolled with stock PyTorch layers.

Pre-LN means each residual sublayer is `x + Sublayer(LN(x))`, in contrast to
post-LN's `LN(x + Sublayer(x))`. Pre-LN is significantly easier to train
stably without aggressive learning-rate warmup, especially at small scale
(see Xiong et al., "On Layer Normalization in the Transformer Architecture",
ICML 2020). For a from-scratch 4-layer model on a single GPU, that
robustness matters more than the marginal final-quality edge sometimes
reported for post-LN.
"""
from __future__ import annotations

import torch
from torch import nn


class PreLNEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                       # (B, L, D)
        key_padding_mask: torch.Tensor | None, # (B, L), True == pad (ignored)
    ) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)
        x = x + self.ff(self.ln2(x))
        return x


class PreLNEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [PreLNEncoderBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.layers:
            x = block(x, key_padding_mask)
        return self.final_ln(x)
