from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NGramEncoderConfig:
    num_features: int = 147_457
    embedding_dim: int = 32
    padding_idx: int = 0
    dropout: float = 0.05


class NGramFeatureEncoder(nn.Module):
    """Mean-pool hashed n-gram embeddings into one lexical vector per Qwen token."""

    def __init__(self, config: NGramEncoderConfig = NGramEncoderConfig()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.num_features,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[config.padding_idx].zero_()
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        if feature_ids.ndim != 3:
            raise ValueError("NGRAM-ENCODER-SHAPE-INVALID: expected [B,L,K]")
        if feature_ids.dtype not in (torch.int64, torch.long):
            feature_ids = feature_ids.long()
        if torch.any(feature_ids < 0) or torch.any(feature_ids >= self.config.num_features):
            raise ValueError("NGRAM-ENCODER-ID-OUT-OF-RANGE")
        mask = feature_ids.ne(self.config.padding_idx)
        x = self.embedding(feature_ids)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        counts = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
        pooled = x.sum(dim=-2) / counts
        has_features = mask.any(dim=-1, keepdim=True)
        out = self.norm(pooled)
        out = out * has_features.to(dtype=out.dtype)
        return self.dropout(out)

    @property
    def output_dim(self) -> int:
        return self.config.embedding_dim
