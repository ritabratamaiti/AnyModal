"""
Projector base class and registry.

Projectors map encoded non-text representations into the language model's embedding space.
All projectors take (batch, seq_len, input_dim) and produce (batch, output_seq_len, output_dim).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectorType(str, Enum):
    """Supported projector architectures."""

    LINEAR = "linear"
    MLP = "mlp"
    QFORMER = "qformer"
    PERCEIVER = "perceiver"
    C_ABSTRACTOR = "c_abstractor"


class BaseProjector(nn.Module, ABC):
    """Abstract base class for all projectors.

    A projector takes encoded representations from a non-text modality
    and maps them into the language model's token embedding space.

    Input:  (batch, input_seq_len, input_dim)
    Output: (batch, output_seq_len, output_dim)

    For simple projectors (Linear, MLP), output_seq_len == input_seq_len.
    For resampling projectors (Q-Former, Perceiver), output_seq_len is configurable.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input embeddings to LLM embedding space.

        Args:
            x: Tensor of shape (batch, seq_len, input_dim).

        Returns:
            Tensor of shape (batch, output_seq_len, output_dim).
        """
        ...


class LinearProjector(BaseProjector):
    """Single linear layer projection.

    Simplest possible projector — a learned linear map.
    Used in early LLaVA (v1) and other lightweight approaches.
    Preserves sequence length.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPProjector(BaseProjector):
    """Multi-layer perceptron projection with configurable depth.

    Two-layer MLP with GELU activation (LLaVA v1.5 style).
    Optionally deeper with layer norm. Preserves sequence length.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__(input_dim, output_dim)
        hidden_dim = hidden_dim or output_dim

        layers: list[nn.Module] = []
        in_features = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Cross-attention building blocks shared by Q-Former and Perceiver
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Multi-head cross-attention layer.

    Queries attend to key-value pairs from another sequence (e.g., vision tokens).
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, q_len, query_dim)
            key_value: (batch, kv_len, kv_dim)
            attention_mask: Optional (batch, 1, q_len, kv_len) or broadcastable.
        """
        B, Q, _ = query.shape
        K = key_value.shape[1]

        q = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Q, -1)
        return self.out_proj(attn_output)


class CrossAttentionBlock(nn.Module):
    """Cross-attention + FFN block with pre-norm residual connections."""

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.cross_attn = CrossAttention(query_dim, kv_dim, num_heads, head_dim, dropout)

        self.norm_ff = nn.LayerNorm(query_dim)
        ff_dim = query_dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(query_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        query = query + self.cross_attn(self.norm_q(query), self.norm_kv(key_value))
        query = query + self.ff(self.norm_ff(query))
        return query


# ---------------------------------------------------------------------------
# Q-Former projector (BLIP-2 style)
# ---------------------------------------------------------------------------

class QFormerProjector(BaseProjector):
    """Q-Former style projector (BLIP-2).

    Uses a set of learned query tokens that attend to the input sequence
    via cross-attention layers. Produces a fixed-length output sequence
    regardless of input length.

    This resamples the input: output_seq_len = num_queries (default 32).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_queries: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        query_dim = output_dim

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, query_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    query_dim=query_dim,
                    kv_dim=input_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, x)

        return self.norm(queries)


# ---------------------------------------------------------------------------
# Perceiver Resampler (Flamingo style)
# ---------------------------------------------------------------------------

class PerceiverProjector(BaseProjector):
    """Perceiver Resampler (Flamingo / Otter style).

    Similar concept to Q-Former but with a perceiver architecture:
    learned latent queries cross-attend to input, with additional
    self-attention among the latent queries for richer interactions.

    Resamples input to num_latents tokens.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.num_latents = num_latents
        latent_dim = output_dim

        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, latent_dim) * 0.02)

        self.cross_attn_layers = nn.ModuleList()
        self.self_attn_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionBlock(
                    query_dim=latent_dim,
                    kv_dim=input_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                )
            )
            # Self-attention among latents
            self.self_attn_layers.append(
                CrossAttentionBlock(
                    query_dim=latent_dim,
                    kv_dim=latent_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                )
            )

        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        latents = self.latent_queries.expand(B, -1, -1)

        for cross_attn, self_attn in zip(self.cross_attn_layers, self.self_attn_layers):
            latents = cross_attn(latents, x)
            latents = self_attn(latents, latents)

        return self.norm(latents)


# ---------------------------------------------------------------------------
# C-Abstractor (Honeybee style)
# ---------------------------------------------------------------------------

class ResidualConvBlock(nn.Module):
    """Residual block with two convolutions and layer norm."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, dim)  # equivalent to LayerNorm for conv
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class CAbstractorProjector(BaseProjector):
    """Convolutional Abstractor (Honeybee style).

    Reshapes the token sequence into a 2D spatial grid, applies
    convolutional layers with adaptive pooling to reduce spatial
    resolution, then flattens back to a sequence.

    Good for vision encoders where tokens have spatial structure
    (e.g., ViT patch tokens form a grid). Reduces sequence length
    via spatial downsampling.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_conv_blocks: int = 2,
        output_tokens: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__(input_dim, output_dim)
        hidden_dim = hidden_dim or output_dim
        self.output_tokens = output_tokens

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv_blocks = nn.ModuleList(
            [ResidualConvBlock(hidden_dim) for _ in range(num_conv_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        x = self.input_proj(x)

        # Reshape to 2D spatial grid
        h = w = int(math.isqrt(N))
        if h * w < N:
            # Not a perfect square — pad to nearest
            h = w = int(math.ceil(math.sqrt(N)))
            pad_len = h * w - N
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.view(B, h, w, -1).permute(0, 3, 1, 2)  # (B, C, H, W)

        for block in self.conv_blocks:
            x = block(x)

        # Adaptive pooling to reduce spatial resolution if requested
        if self.output_tokens is not None:
            out_h = out_w = int(math.ceil(math.sqrt(self.output_tokens)))
            x = F.adaptive_avg_pool2d(x, (out_h, out_w))

        # Flatten back to sequence
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, seq, hidden)

        return self.norm(self.output_proj(x))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROJECTOR_REGISTRY: dict[ProjectorType, type[BaseProjector]] = {
    ProjectorType.LINEAR: LinearProjector,
    ProjectorType.MLP: MLPProjector,
    ProjectorType.QFORMER: QFormerProjector,
    ProjectorType.PERCEIVER: PerceiverProjector,
    ProjectorType.C_ABSTRACTOR: CAbstractorProjector,
}


def create_projector(
    projector_type: str | ProjectorType,
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> BaseProjector:
    """Factory function to create a projector by type.

    Args:
        projector_type: One of 'linear', 'mlp', 'qformer', 'perceiver', 'c_abstractor'.
        input_dim: Dimension of input embeddings (from encoder).
        output_dim: Dimension of output embeddings (LLM hidden size).
        **kwargs: Additional arguments forwarded to the projector constructor.

    Returns:
        Instantiated projector module.

    Example::

        projector = create_projector("mlp", input_dim=768, output_dim=2048, num_layers=3)
        projector = create_projector("qformer", input_dim=1152, output_dim=2048, num_queries=64)
    """
    ptype = ProjectorType(projector_type)
    cls = _PROJECTOR_REGISTRY[ptype]
    return cls(input_dim=input_dim, output_dim=output_dim, **kwargs)
