from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .perceiver import Perceiver


class T3Config:
    start_text_token = 255
    stop_text_token = 0
    text_tokens_dict_size = 704
    max_text_tokens = 2048

    start_speech_token = 6561
    stop_speech_token = 6562
    speech_tokens_dict_size = 8194
    max_speech_tokens = 4096

    llama_config_name = "Llama_520M"
    input_pos_emb = "learned"
    speech_cond_prompt_len = 150

    # For T3CondEnc
    encoder_type = "voice_encoder"
    speaker_embed_size = 256
    use_perceiver_resampler = True
    emotion_adv = True

    @property
    def n_channels(self):
        return 1024  # hidden size


@dataclass
class T3Cond:
    """
    Dataclass container for most / all conditioning info.
    TODO: serialization methods aren't used, keeping them around for convenience
    """

    speaker_emb: Tensor
    clap_emb: Optional[Tensor] = None
    cond_prompt_speech_tokens: Optional[Tensor] = None
    cond_prompt_speech_emb: Optional[Tensor] = None
    emotion_adv: Optional[Tensor] = 0.5

    def save(self, fpath):
        torch.save(self.__dict__, fpath)

    @staticmethod
    def load(path):
        kwargs = mx.load(path, weights_only=True)
        return T3Cond(**kwargs)


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

        # emotion adv
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)

        # perceiver resampler
        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = Perceiver()

    def __call__(self, cond: T3Cond):
        # Validate
        assert (cond.cond_prompt_speech_tokens is None) == (
            cond.cond_prompt_speech_emb is None
        ), "no embeddings for cond_prompt_speech_tokens"

        # Speaker embedding projection
        cond_spkr = self.spkr_enc(
            cond.speaker_emb.view(-1, self.hp.speaker_embed_size)
        )[
            :, None
        ]  # (B, 1, dim)
        empty = mx.zeros_like(cond_spkr[:, :0])  # (B, 0, dim)

        # TODO CLAP
        assert cond.clap_emb is None, "clap_embed not implemented"
        cond_clap = empty  # (B, 0, dim)

        # Cond prompt
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty  # (B, 0, dim)
        elif self.hp.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)

        # Emotion Adv: must provide a value if this model uses emotion conditioning
        cond_emotion_adv = empty  # (B, 0, dim)
        if self.hp.emotion_adv:
            assert cond.emotion_adv is not None
            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv.view(-1, 1, 1))

        # Concat and return
        cond_embeds = mx.concatenate(
            (
                cond_spkr,
                cond_clap,
                cond_prompt_speech_emb,
                cond_emotion_adv,
            ),
            axis=1,
        )
        return cond_embeds
