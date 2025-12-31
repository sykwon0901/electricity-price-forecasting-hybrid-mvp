"""Gated Transformer hurdle model (volume-only) for window-based forecasting.

This implements a lightweight "REAL" gated residual Transformer encoder, with a two-head hurdle output:

  Head 1 (classification): active_logits over horizon steps
  Head 2 (regression):     log1p(volume) over horizon steps

The intended usage is:
  - Keep price targets (High/Low/Close) from persistence baseline
  - Replace Volume only with the hurdle prediction

Forward signature:
  active_logits, logvol_hat = model(x_seq, id_code)

where:
  x_seq  : (B, WIN, n_features)
  id_code: (B,) int64 codes
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (non-trainable)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        t = x.size(1)
        return x + self.pe[:, :t, :]


class REALGatedEncoderLayer(nn.Module):
    """A lightweight gated residual Transformer encoder layer.

    Two gated residual blocks:
      x <- LN( x + sigmoid(g(x)) * Attn(x) )
      x <- LN( x + sigmoid(h(x)) * FF(x) )
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.ff1 = nn.Linear(d_model, dim_ff)
        self.ff2 = nn.Linear(dim_ff, d_model)
        self.act = nn.GELU()
        self.ln2 = nn.LayerNorm(d_model)

        self.gate_attn = nn.Linear(d_model, d_model)
        self.gate_ff = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        gate1 = torch.sigmoid(self.gate_attn(x))
        x = self.ln1(x + self.dropout(gate1 * attn_out))

        ff = self.ff2(self.dropout(self.act(self.ff1(x))))
        gate2 = torch.sigmoid(self.gate_ff(x))
        x = self.ln2(x + self.dropout(gate2 * ff))
        return x


class GatedTransformerHurdle(nn.Module):
    """Gated Transformer encoder with two-head hurdle output."""

    def __init__(
        self,
        *,
        n_features: int,
        n_ids: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.10,
        id_emb_dim: int = 8,
        max_len: int = 256,
    ):
        super().__init__()
        self.in_proj = nn.Linear(int(n_features), int(d_model))
        self.pos = SinusoidalPositionalEncoding(d_model=int(d_model), max_len=int(max_len))

        self.layers = nn.ModuleList(
            [
                REALGatedEncoderLayer(d_model=int(d_model), nhead=int(nhead), dim_ff=int(dim_ff), dropout=float(dropout))
                for _ in range(int(n_layers))
            ]
        )

        self.id_emb = nn.Embedding(int(n_ids), int(id_emb_dim))

        self.head = nn.Sequential(
            nn.Linear(int(d_model) + int(id_emb_dim), int(d_model)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

        self.active_out = nn.Linear(int(d_model), int(horizon))  # logits
        self.logvol_out = nn.Linear(int(d_model), int(horizon))  # regression (log1p volume)

    def forward(self, x_seq: torch.Tensor, id_code: torch.Tensor):
        # x_seq: (B, WIN, F), id_code: (B,)
        x = self.in_proj(x_seq)  # (B, WIN, d_model)
        x = self.pos(x)

        for layer in self.layers:
            x = layer(x)

        h = x[:, -1, :]  # last token summary
        emb = self.id_emb(id_code)

        z = torch.cat([h, emb], dim=-1)
        z = self.head(z)

        active_logits = self.active_out(z)  # (B, H)
        logvol_hat = self.logvol_out(z)     # (B, H)
        return active_logits, logvol_hat
