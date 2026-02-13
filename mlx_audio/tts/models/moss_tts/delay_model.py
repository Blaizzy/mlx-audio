"""Delay-variant architecture core for MOSS-TTS."""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from .backbone import MossTTSBackbone
from .config import ModelConfig


class MossTTSDelayModel(nn.Module):
    """Shared Delay-model core; generation policy lives in `model.py`."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.is_local_variant:
            raise ValueError(
                "MossTTSDelayModel requires a Delay config (no local_num_layers)."
            )

        self.config = config
        self.channels = config.channels
        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_ext = [
            nn.Embedding(config.audio_vocab_size + 1, config.hidden_size)
            for _ in range(config.n_vq)
        ]
        self.backbone = MossTTSBackbone(config.language_config)
        self.lm_heads = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False),
            *[
                nn.Linear(config.hidden_size, config.audio_vocab_size + 1, bias=False)
                for _ in range(config.n_vq)
            ],
        ]

    def make_cache(self) -> List[KVCache]:
        return self.backbone.make_cache()

    def _prepare_multi_modal_embeddings(
        self,
        input_ids: mx.array,
        n_vq_for_inference: Optional[int] = None,
    ) -> mx.array:
        if input_ids.ndim != 3:
            raise ValueError(
                f"Expected input_ids shape (batch, time, channels), got {input_ids.shape}"
            )
        batch_size, sequence_length, channels = input_ids.shape
        expected_channels = self.channels
        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels, got {channels} channels"
            )

        n_vq_for_inference = (
            self.config.n_vq
            if n_vq_for_inference is None
            else int(min(n_vq_for_inference, self.config.n_vq))
        )

        embeddings = self.text_embedding(input_ids[:, :, 0])
        for channel_idx in range(n_vq_for_inference):
            embeddings = embeddings + self.emb_ext[channel_idx](input_ids[:, :, channel_idx + 1])
        return embeddings.astype(mx.float32)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[Optional[KVCache]]] = None,
        n_vq_for_inference: Optional[int] = None,
    ) -> mx.array:
        embeddings = self._prepare_multi_modal_embeddings(
            input_ids,
            n_vq_for_inference=n_vq_for_inference,
        )
        return self.backbone(embeddings, cache=cache)

    def compute_next_logits(
        self,
        global_hidden_state: mx.array,
        n_vq_for_inference: Optional[int] = None,
    ) -> List[mx.array]:
        n_vq_for_inference = (
            self.config.n_vq
            if n_vq_for_inference is None
            else int(min(n_vq_for_inference, self.config.n_vq))
        )
        logits: List[mx.array] = []
        channels_for_step = 1 + n_vq_for_inference
        for channel_idx in range(channels_for_step):
            head_logits = self.lm_heads[channel_idx](global_hidden_state)
            if channel_idx > 0 and 0 <= self.config.audio_pad_code < head_logits.shape[-1]:
                token_ids = mx.arange(head_logits.shape[-1], dtype=mx.int32)
                pad_mask = token_ids == int(self.config.audio_pad_code)
                head_logits = mx.where(pad_mask[None, :], -mx.inf, head_logits)
            logits.append(head_logits)
        return logits


__all__ = ["MossTTSDelayModel"]
