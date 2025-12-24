# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class AlignModalities(nn.Module):
    """
    Aligns features from different modalities (e.g., video to audio).

    Uses a 1D convolution to project features and an optional gating mechanism
    to control the blending with anchor features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        with_gate: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.normalize = normalize

        # 1x1 convolution for projection
        self.conv_weight = mx.zeros((out_channels, 1, in_channels))
        self.conv_bias = mx.zeros((out_channels,))

        if normalize:
            self.layer_norm = nn.LayerNorm(out_channels)

        self.gate = mx.array([0.0]) if with_gate else None

    def __call__(self, anchor: mx.array, tgt: Optional[mx.array] = None) -> mx.array:
        """
        Align target features to anchor features.

        Args:
            anchor: Input anchor tensor (B, T, C)
            tgt: Optional target features to align (B, in_channels, T)

        Returns:
            Aligned features (B, T, out_channels)
        """
        if tgt is None:
            return anchor

        # Apply 1x1 convolution: (B, in_channels, T) -> (B, out_channels, T)
        # Transpose tgt to (B, T, in_channels) for conv1d
        tgt_t = mx.transpose(tgt, (0, 2, 1))
        post_conv = mx.conv1d(tgt_t, self.conv_weight)
        post_conv = post_conv + self.conv_bias

        # post_conv is now (B, T, out_channels)
        if self.normalize:
            post_conv = self.layer_norm(post_conv)

        if self.gate is None:
            return post_conv
        else:
            return anchor + mx.tanh(self.gate) * post_conv


class EmbedAnchors(nn.Module):
    """
    Embeds temporal anchors for audio alignment.

    Anchors are tokens that mark specific time spans in the audio
    (e.g., "+", "-" for positive/negative examples).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        out_dim: int,
    ):
        super().__init__()
        # +1 for padding index
        self.embed = nn.Embedding(num_embeddings + 1, embedding_dim)
        self.gate = mx.array([0.0])
        self.proj = nn.Linear(embedding_dim, out_dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        anchor_ids: Optional[mx.array] = None,
        anchor_alignment: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Apply anchor embeddings to input features.

        Args:
            x: Input features (B, T, C)
            anchor_ids: Anchor token IDs (B, num_anchors)
            anchor_alignment: Mapping from timesteps to anchor indices (B, T)

        Returns:
            Features with anchor embeddings added (B, T, C)
        """
        if anchor_ids is None or anchor_alignment is None:
            return x

        # Gather anchor IDs based on alignment
        # anchor_alignment maps each timestep to an anchor index
        batch_size, seq_len = anchor_alignment.shape

        # Use mx.take_along_axis for efficient gathering
        # anchor_alignment: (B, T) with values indexing into anchor_ids
        # anchor_ids: (B, num_anchors)
        # We need to gather anchor_ids[b, anchor_alignment[b, t]] for each b, t

        # Expand anchor_alignment for take_along_axis
        gathered_ids = mx.take_along_axis(anchor_ids, anchor_alignment, axis=1)

        # Embed and project
        embs = self.embed(gathered_ids)
        proj = self.proj(embs)

        return x + mx.tanh(self.gate) * proj
