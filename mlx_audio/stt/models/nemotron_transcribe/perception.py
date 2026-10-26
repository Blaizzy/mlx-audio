# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# Adapted from NVIDIA NeMo Speech's parallel_expert_encoder.py (Apache-2.0).
"""Context-windowed ASR and Sortformer experts with sinusoidal speaker fusion."""

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.vad.models.nemotron_diarization import Model as DiarizationModel
from mlx_audio.vad.models.nemotron_diarization.nemotron_diarization import (
    Encoder,
    MelFeatures,
)


def normalize_features(features):
    """NeMo's per-feature sample standard deviation, over valid frames only."""
    reference = features[..., :1]
    mean = reference + (features - reference).mean(axis=-1, keepdims=True)
    centered = features - mean
    variance = (centered * centered).sum(axis=-1, keepdims=True) / max(
        1, features.shape[-1] - 1
    )
    return centered / (mx.sqrt(variance) + 1e-5)


class ParallelExpertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.asr_encoder = Encoder(config.asr_config)
        self.diarization_model = DiarizationModel(config.diarization_config)
        self.asr_norm = nn.LayerNorm(config.asr_config.d_model)
        self.diar_norm = nn.LayerNorm(config.diarization_config.num_speakers)
        self._config = config.pe_encoder_config
        self._factor = config.asr_config.subsampling_factor
        position = mx.arange(config.diarization_config.num_speakers)[:, None]
        width = config.asr_config.d_model
        frequencies = mx.exp(mx.arange(0, width, 2) * -(math.log(10000) / width))
        angles = position * frequencies
        self._diar_kernel = mx.stack([mx.sin(angles), mx.cos(angles)], axis=-1).reshape(
            -1, width
        )

    def fuse(self, states, probabilities):
        cfg = self._config
        probabilities = probabilities.astype(states.dtype)
        if cfg["speaker_feature_mode"] == "thresholded":
            probabilities = (probabilities > cfg["speaker_activity_threshold"]).astype(
                states.dtype
            )
        targets = self.diar_norm(probabilities)
        infusion = targets @ self._diar_kernel.astype(targets.dtype)
        return self.asr_norm(states) + cfg.get("spk_kernel_scale", 1.0) * infusion

    def __call__(self, features):
        """One utterance of unnormalized (1, mel, valid_frames) features."""
        cfg, factor = self._config, self._factor
        size = cfg.get("online_inference_length", 500) * factor
        left = max(0, cfg.get("chunk_left_context", 50)) * factor
        right = max(0, cfg.get("chunk_right_context", 50)) * factor
        normalized = normalize_features(features)
        dtype = self.asr_norm.weight.dtype
        normalized = normalized.astype(dtype)
        state = self.diarization_model.init_streaming_state()
        outputs = []
        total = features.shape[-1]
        for start in range(0, total, size):
            end = min(start + size, total)
            context_start, context_end = max(0, start - left), min(total, end + right)
            length = context_end - context_start
            x, lengths = self.asr_encoder.pre_encode(
                normalized[:, :, context_start:context_end], mx.array([length])
            )
            encoded = self.asr_encoder(x, lengths)
            first = (start - context_start) // factor
            count = (end + factor - 1) // factor - (start + factor - 1) // factor
            encoded = encoded[:, first : first + count]
            # Include the padded final encoder frame, as NeMo's high-resolution
            # head pools all eight predictions before trimming to the ASR grid.
            central = count * factor
            probabilities = (
                self.diarization_model.streaming_step(
                    features[:, :, context_start:context_end],
                    state,
                    central_frames=central,
                    feature_length=length,
                    left_context_frames=start - context_start,
                )
                .reshape(1, count, factor, -1)
                .mean(axis=2)
            )
            output = self.fuse(encoded, probabilities)
            mx.eval(output)
            outputs.append(output)
        return mx.concatenate(outputs, axis=1)


class Perception(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.preprocessor = MelFeatures(config.processor_config)
        self.encoder = ParallelExpertEncoder(config)
        self.proj = nn.Linear(
            config.asr_config.d_model, config.llm_config["hidden_size"]
        )

    def __call__(self, audio):
        return self.proj(self.encoder(self.preprocessor(audio)))
