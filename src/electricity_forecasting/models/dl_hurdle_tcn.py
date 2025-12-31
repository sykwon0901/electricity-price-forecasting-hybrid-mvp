"""Hurdle-TCN (two-head) model for volume-only forecasting.

Architecture
------------
Input:  X shape (B, WIN, F)
Output:
  - active_logits: (B, HORIZON)  -> sigmoid => P(active)
  - logvol_hat:    (B, HORIZON)  -> predicts log1p(volume)

Notes
-----
* The gate (active head) and the regressor (volume head) are trained jointly.
* In evaluation, prices (high/low/close) are taken from the persistence baseline;
  only the volume channel is replaced by the model prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """A causal residual temporal block for TCN.

    The implementation uses left padding via Conv1d(padding=pad) and then crops
    the right side to keep output length equal to input length.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2 for a meaningful TCN")

        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: (B, C, T)

        Returns:
            y: (B, C_out, T)
        """
        T = x.size(-1)

        y = self.conv1(x)
        y = y[..., :T]  # crop to causal length
        y = F.relu(y)
        y = self.dropout(y)

        y = self.conv2(y)
        y = y[..., :T]
        y = F.relu(y)
        y = self.dropout(y)

        res = x if self.downsample is None else self.downsample(x)
        res = res[..., :T]

        return F.relu(y + res)


class TCNEncoder(nn.Module):
    """Stacked TemporalBlocks."""

    def __init__(
        self,
        n_features: int,
        channels: Sequence[int] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_ch = int(n_features)
        for i, ch in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=int(ch),
                    kernel_size=int(kernel_size),
                    dilation=2**i,
                    dropout=float(dropout),
                )
            )
            in_ch = int(ch)
        self.net = nn.Sequential(*layers)
        self.out_channels = int(in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HurdleTCN(nn.Module):
    """Two-head Hurdle-TCN for volume.

    Input:  (B, WIN, F)
    Output: (B, HORIZON) logits + (B, HORIZON) log1p(volume)
    """

    def __init__(
        self,
        n_features: int,
        horizon: int,
        channels: Sequence[int] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.encoder = TCNEncoder(
            n_features=int(n_features),
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        d = self.encoder.out_channels
        self.head_active = nn.Linear(d, self.horizon)
        self.head_logvol = nn.Linear(d, self.horizon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            x: (B, WIN, F)

        Returns:
            active_logits: (B, HORIZON)
            logvol_hat: (B, HORIZON)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, WIN, F). Got {tuple(x.shape)}")
        # (B, WIN, F) -> (B, F, WIN)
        x_c = x.transpose(1, 2)
        h = self.encoder(x_c)  # (B, C, WIN)
        h_last = h[:, :, -1]   # (B, C)

        active_logits = self.head_active(h_last)
        logvol_hat = self.head_logvol(h_last)
        return active_logits, logvol_hat
