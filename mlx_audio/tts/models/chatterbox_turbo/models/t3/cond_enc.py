# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .t3_config import T3Config


@dataclass
class T3Cond:
    """Dataclass container for conditioning info."""

    speaker_emb: mx.array
    clap_emb: Optional[mx.array] = None
    cond_prompt_speech_tokens: Optional[mx.array] = None
    cond_prompt_speech_emb: Optional[mx.array] = None
    emotion_adv: Optional[mx.array] = None


class T3CondEnc(nn.Module):
    """
    Handle all non-text conditioning, like speaker embeddings / prompts, CLAP, emotion, etc.
    """

    def __init__(self, hp: T3Config):
        super().__init__()
        self.hp = hp

        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(str(hp.encoder_type))

        # Emotion adv (not used in Turbo)
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

        # Perceiver resampler (not used in Turbo)
        self.perceiver = None

    def __call__(self, cond: T3Cond) -> mx.array:
        # Validate
        assert (cond.cond_prompt_speech_tokens is None) == (
            cond.cond_prompt_speech_emb is None
        ), "no embeddings for cond_prompt_speech_tokens"

        # Speaker embedding projection
        speaker_emb = cond.speaker_emb.reshape(-1, self.hp.speaker_embed_size)
        cond_spkr = self.spkr_enc(speaker_emb)[:, None, :]  # (B, 1, dim)

        B = cond_spkr.shape[0]
        dim = cond_spkr.shape[-1]

        # Empty tensor for unused conditions
        empty = mx.zeros((B, 0, dim))

        # CLAP (not implemented)
        assert cond.clap_emb is None, "clap_embed not implemented"
        cond_clap = empty

        # Cond prompt
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty
        elif self.hp.use_perceiver_resampler and self.perceiver is not None:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # Emotion Adv (not used in Turbo)
        cond_emotion_adv = empty
        if (
            self.hp.emotion_adv
            and cond.emotion_adv is not None
            and self.emotion_adv_fc is not None
        ):
            emotion_val = cond.emotion_adv.reshape(-1, 1, 1)
            cond_emotion_adv = self.emotion_adv_fc(emotion_val)

        # Concat and return
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
