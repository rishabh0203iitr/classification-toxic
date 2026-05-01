"""Top-level ToxicClassifier: token + positional embeddings → encoder → pool → head."""
from __future__ import annotations

import math

import torch
from torch import nn

from .transformer import PreLNEncoder


class ToxicClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        pool: str = "cls",
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if pool not in {"cls", "mean"}:
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}")
        self.pool = pool
        self.pad_id = pad_id
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.encoder = PreLNEncoder(d_model, n_heads, n_layers, dim_ff, dropout)
        self.head = nn.Linear(d_model, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        ids: torch.Tensor,                         # (B, L)
        key_padding_mask: torch.Tensor | None = None,  # (B, L), True == pad
    ) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(ids) * math.sqrt(self.d_model) + self.pos_emb(pos)
        x = self.emb_drop(x)
        x = self.encoder(x, key_padding_mask=key_padding_mask)

        if self.pool == "cls":
            pooled = x[:, 0, :]
        else:  # mean over non-pad positions
            if key_padding_mask is None:
                pooled = x.mean(dim=1)
            else:
                mask = (~key_padding_mask).float().unsqueeze(-1)  # (B, L, 1)
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.head(pooled).squeeze(-1)  # (B,)
        return logits

    @torch.no_grad()
    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not only_trainable))
