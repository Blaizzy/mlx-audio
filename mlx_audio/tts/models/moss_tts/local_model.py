"""Local-variant architecture for MOSS-TTS (global backbone + local transformer)."""

from __future__ import annotations

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from .backbone import MossTTSBackbone
from .config import ModelConfig
from .local_transformer import MossTTSLocalTransformer
from .sampling import ChannelSamplingConfig, sample_channel_token


class MossTTSMLP(nn.Module):
    """SwiGLU projector used around local/global feature boundaries."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MossTTSLocalModel(nn.Module):
    """Shared local-model core; generation policy lives in `model.py`."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not config.is_local_variant:
            raise ValueError(
                "MossTTSLocalModel requires local_num_layers in config. "
                "Delay variant belongs in Phase 3."
            )

        self.config = config
        self.channels = config.channels

        self.embedding_list = [
            nn.Embedding(config.vocab_size, config.hidden_size),
            *[
                nn.Embedding(config.audio_vocab_size + 1, config.hidden_size)
                for _ in range(config.n_vq)
            ],
        ]

        self.backbone = MossTTSBackbone(config.language_config)
        self.local_transformer = MossTTSLocalTransformer(config.local_transformer_config())

        local_hidden_size = self.local_transformer.config.hidden_size
        self.speech_embedding_to_local_mlp = MossTTSMLP(
            input_size=config.hidden_size,
            hidden_size=config.additional_mlp_ffn_hidden_size,
            output_size=local_hidden_size,
        )
        self.local_to_speech_embedding_mlps = [
            MossTTSMLP(
                input_size=local_hidden_size,
                hidden_size=config.additional_mlp_ffn_hidden_size,
                output_size=config.hidden_size,
            )
            for _ in range(self.channels)
        ]

        self.layer_norm_before_lm_heads = [
            nn.RMSNorm(config.hidden_size, eps=config.language_config.rms_norm_eps)
            for _ in range(self.channels)
        ]
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

        fused = mx.zeros(
            (batch_size, sequence_length, self.config.hidden_size),
            dtype=self.embedding_list[0].weight.dtype,
        )
        for channel_idx in range(1 + n_vq_for_inference):
            fused = fused + self.embedding_list[channel_idx](input_ids[:, :, channel_idx])
        return fused

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

    def sample_next_channels(
        self,
        global_hidden_state: mx.array,
        input_history: mx.array,
        channel_sampling: List[ChannelSamplingConfig],
        n_vq_for_inference: Optional[int] = None,
    ) -> mx.array:
        """
        Decode one Local timestep:
        global hidden -> local transformer -> (1 + n_vq) next-channel tokens.
        """

        n_vq_for_inference = (
            self.config.n_vq
            if n_vq_for_inference is None
            else int(min(n_vq_for_inference, self.config.n_vq))
        )
        channels_for_step = 1 + n_vq_for_inference

        batch_size = global_hidden_state.shape[0]
        local_hidden_size = self.local_transformer.config.hidden_size
        local_inputs = mx.zeros(
            (batch_size, 0, local_hidden_size), dtype=global_hidden_state.dtype
        )
        current_input = self.speech_embedding_to_local_mlp(global_hidden_state)

        sampled_channels: List[mx.array] = []
        for channel_idx in range(channels_for_step):
            local_inputs = mx.concatenate(
                [local_inputs, current_input[:, None, :]], axis=1
            )
            local_outputs = self.local_transformer(local_inputs)
            hidden_state = local_outputs[:, -1, :]
            hidden_state = self.local_to_speech_embedding_mlps[channel_idx](hidden_state)
            hidden_state = self.layer_norm_before_lm_heads[channel_idx](hidden_state)

            logits = self.lm_heads[channel_idx](hidden_state)
            if channel_idx > 0 and 0 <= self.config.audio_pad_code < logits.shape[-1]:
                channel_ids = mx.arange(logits.shape[-1], dtype=mx.int32)
                pad_mask = channel_ids == int(self.config.audio_pad_code)
                logits = mx.where(pad_mask[None, :], -mx.inf, logits)

            previous_tokens = input_history[:, :, channel_idx]
            next_token = sample_channel_token(
                logits,
                channel_sampling[channel_idx],
                previous_tokens=previous_tokens,
            )
            sampled_channels.append(next_token)

            current_input = self.embedding_list[channel_idx](next_token.astype(mx.int32))
            current_input = self.speech_embedding_to_local_mlp(current_input)

        next_tokens = mx.stack(sampled_channels, axis=-1).astype(mx.int32)
        if channels_for_step < self.channels:
            pad = mx.full(
                (batch_size, self.channels - channels_for_step),
                self.config.audio_pad_code,
                dtype=mx.int32,
            )
            next_tokens = mx.concatenate([next_tokens, pad], axis=-1)
        return next_tokens


__all__ = ["MossTTSLocalModel", "MossTTSMLP"]
