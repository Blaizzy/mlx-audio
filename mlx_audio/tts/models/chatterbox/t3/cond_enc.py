# Ported from https://github.com/resemble-ai/chatterbox

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..config import T3Config
from .perceiver import Perceiver


@dataclass
class T3Cond:
    """
    Container for T3 conditioning information.

    Attributes:
        speaker_emb: Speaker embedding from voice encoder (B, speaker_dim)
        clap_emb: Optional CLAP embedding for semantic conditioning
        cond_prompt_speech_tokens: Optional speech token prompt (B, T)
        cond_prompt_speech_emb: Optional embedded speech prompt (B, T, D)
        emotion_adv: Emotion exaggeration factor, typically 0.3-0.7 (scalar or (B, 1))
    """

    speaker_emb: mx.array
    clap_emb: Optional[mx.array] = None
    cond_prompt_speech_tokens: Optional[mx.array] = None
    cond_prompt_speech_emb: Optional[mx.array] = None
    emotion_adv: Optional[mx.array] = None

    def __post_init__(self):
        """Set default emotion_adv if not provided."""
        if self.emotion_adv is None:
            self.emotion_adv = mx.array(0.5)


class T3CondEnc(nn.Module):
    """
    Conditioning encoder for T3 model.
    Handles speaker embeddings, emotion control, and prompt speech tokens.
    """

    def __init__(self, hp: T3Config):
        super().__init__()
        self.hp = hp

        # Speaker embedding projection
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(f"encoder_type '{hp.encoder_type}' not supported")

        # Emotion control
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

        # Perceiver resampler for prompt speech tokens
        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = Perceiver()

    def __call__(self, cond: T3Cond) -> mx.array:
        """
        Process conditioning inputs into a single conditioning tensor.

        Args:
            cond: T3Cond dataclass with conditioning information

        Returns:
            Conditioning embeddings (B, cond_len, D)
        """
        # Validate
        has_tokens = cond.cond_prompt_speech_tokens is not None
        has_emb = cond.cond_prompt_speech_emb is not None
        assert (
            has_tokens == has_emb
        ), "cond_prompt_speech_tokens and cond_prompt_speech_emb must both be provided or both be None"

        # Speaker embedding projection (B, speaker_dim) -> (B, 1, D)
        B = cond.speaker_emb.shape[0]
        cond_spkr = self.spkr_enc(
            mx.reshape(cond.speaker_emb, (B, self.hp.speaker_embed_size))
        )
        cond_spkr = mx.expand_dims(cond_spkr, 1)  # (B, 1, D)

        # Empty placeholder for concatenation
        empty = cond_spkr[:, :0, :]  # (B, 0, D)

        # CLAP embedding (not implemented yet)
        if cond.clap_emb is not None:
            raise NotImplementedError("clap_emb not yet implemented")
        cond_clap = empty  # (B, 0, D)

        # Conditional prompt speech embeddings
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty  # (B, 0, D)
        elif self.hp.use_perceiver_resampler:
            # Resample to fixed length using Perceiver
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # Emotion exaggeration
        cond_emotion_adv = empty  # (B, 0, D)
        if self.hp.emotion_adv:
            assert (
                cond.emotion_adv is not None
            ), "emotion_adv must be provided when hp.emotion_adv is True"
            # Reshape to (B, 1, 1)
            emotion_val = cond.emotion_adv
            if emotion_val.ndim == 0:
                emotion_val = mx.reshape(emotion_val, (1, 1, 1))
            elif emotion_val.ndim == 1:
                emotion_val = mx.reshape(emotion_val, (-1, 1, 1))
            elif emotion_val.ndim == 2:
                emotion_val = mx.expand_dims(emotion_val, -1)

            cond_emotion_adv = self.emotion_adv_fc(emotion_val)

        # Concatenate all conditioning signals
        cond_embeds = mx.concatenate(
            [
                cond_spkr,
                cond_clap,
                cond_prompt_speech_emb,
                cond_emotion_adv,
            ],
            axis=1,
        )

        return cond_embeds
