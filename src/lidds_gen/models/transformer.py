"""Transformer-encoder window classifier with masked mean pooling (Sec. 4.3)."""
from __future__ import annotations

import math

import torch
from torch import nn

from ..data.vocab import PAD_ID


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int):
        super().__init__()
        pos = torch.arange(max_length).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.2,
        max_length: int = 200,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = SinusoidalPositionalEncoding(d_model, max_length)
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        # Nested tensors change numerics slightly between train and eval; keep them off.
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) padded ids; lengths: (B,) true lengths. Returns logits (B,)."""
        pad_mask = tokens.eq(PAD_ID)                       # True = ignore (mask M in Eq. 2)
        h = self.encoder(self.pos(self.embedding(tokens)), src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)
