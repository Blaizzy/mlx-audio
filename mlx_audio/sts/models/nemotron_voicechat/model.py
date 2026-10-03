from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.nemotron_voicechat import NemotronVoiceChatCodec
from mlx_audio.lm.models.cache import ArraysCache, KVCache
from mlx_audio.lm.models.nemotron_h import NemotronHModel
from mlx_audio.stt.models.nemotron_asr.conformer import Conformer
from mlx_audio.stt.models.nemotron_asr.rnnt import JointNetwork, PredictNetwork

from .config import ModelConfig, NemotronVoiceChatConfig
from .tts import EARTTSModel

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)


@dataclass
class VoiceChatModelOutput:
    hidden_states: mx.array
    text_logits: mx.array
    function_logits: mx.array
    cache: object | None = None


def sanitize_weights(
    weights: Mapping[str, mx.array], codec: NemotronVoiceChatCodec
) -> dict[str, mx.array]:
    converted: dict[str, mx.array] = {}
    lstm_biases: dict[str, list[mx.array]] = {}
    codec_prefix = "tts_model.audio_codec."
    for key, value in weights.items():
        if key == "tts_model._control_codes":
            converted["tts_model.control_codes"] = value
            continue
        if key == "tts_model.tts_model.audio_prompt_projection_W":
            continue
        if key.startswith("stt_model.rnnt_decoder.") and ".dec_rnn.lstm." in key:
            base, suffix = key.rsplit(".dec_rnn.lstm.", 1)
            stem = f"{base}.dec_rnn.lstm"
            if suffix.startswith("weight_ih_l"):
                layer = suffix[len("weight_ih_l") :]
                converted[f"{stem}.{layer}.Wx"] = value
            elif suffix.startswith("weight_hh_l"):
                layer = suffix[len("weight_hh_l") :]
                converted[f"{stem}.{layer}.Wh"] = value
            elif suffix.startswith(("bias_ih_l", "bias_hh_l")):
                layer = suffix.rsplit("_l", 1)[1]
                lstm_biases.setdefault(f"{stem}.{layer}.bias", []).append(value)
            continue
        if key.startswith(codec_prefix):
            continue
        if key.startswith("stt_model.perception."):
            if value.ndim == 4:
                value = value.transpose(0, 2, 3, 1)
            elif value.ndim == 3 and key.endswith(".weight"):
                value = value.transpose(0, 2, 1)
        elif (
            key.startswith("stt_model.llm.")
            and value.ndim == 3
            and key.endswith(".weight")
        ):
            value = value.transpose(0, 2, 1)
        converted[key] = value

    for key, values in lstm_biases.items():
        converted[key] = sum(values)

    converted.update(
        {
            f"{codec_prefix}{key}": value
            for key, value in codec.sanitize(weights, prefix=codec_prefix).items()
        }
    )
    return converted


class _FeaturizerBuffers(nn.Module):
    def __init__(self, config: NemotronVoiceChatConfig):
        super().__init__()
        self.fb = mx.zeros(
            (1, config.preprocessor.features, config.preprocessor.n_fft // 2 + 1)
        )
        self.window = mx.zeros((config.preprocessor.win_length,))


class _PreprocessorBuffers(nn.Module):
    def __init__(self, config: NemotronVoiceChatConfig):
        super().__init__()
        self.featurizer = _FeaturizerBuffers(config)


class AudioPerception(nn.Module):
    def __init__(self, config: NemotronVoiceChatConfig):
        super().__init__()
        self.config = config
        self.preprocessor = _PreprocessorBuffers(config)
        self.encoder = Conformer(config.encoder)
        self.proj = nn.Linear(config.encoder.d_model, config.output_dim)

    def __call__(
        self, mel: mx.array, lengths: mx.array | None = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        encoded, lengths = self.encoder(
            mel,
            lengths=lengths,
            att_context_size=self.config.encoder.att_context_size[0],
        )
        return self.proj(encoded), lengths, encoded


class VoiceChatSTT(nn.Module):
    def __init__(self, config: NemotronVoiceChatConfig):
        super().__init__()
        self.config = config
        self.perception = AudioPerception(config)
        self.llm = NemotronHModel(config.llm)
        self.llm.pop("embeddings")
        self.embed_tokens = nn.Embedding(config.llm.vocab_size, config.llm.hidden_size)
        self.lm_head = nn.Linear(
            config.llm.hidden_size, config.llm.vocab_size, bias=False
        )
        if config.use_function_head:
            self.function_head = nn.Linear(
                config.llm.hidden_size, config.llm.vocab_size, bias=False
            )
        self.rnnt_decoder = PredictNetwork(config.decoder)
        self.rnnt_joint = JointNetwork(config.joint)

    def __call__(self, inputs_embeds: mx.array, *, cache=None) -> VoiceChatModelOutput:
        hidden = self.llm(input_embeddings=inputs_embeds, cache=cache)
        text_logits = self.lm_head(hidden)
        function_logits = (
            self.function_head(hidden)
            if self.config.use_function_head
            else mx.zeros_like(text_logits)
        )
        return VoiceChatModelOutput(
            hidden_states=hidden,
            text_logits=text_logits,
            function_logits=function_logits,
            cache=cache,
        )

    def make_cache(self):
        caches = []
        for layer in self.llm.layers:
            if layer.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif layer.block_type == "*":
                caches.append(KVCache())
        return caches


class VoiceChatTTS(nn.Module):
    def __init__(self, config: NemotronVoiceChatConfig):
        super().__init__()
        self.config = config
        self.codec_silence_tokens = mx.zeros(
            (config.codec.num_quantizers,), dtype=mx.int64
        )
        self.control_codes = mx.zeros((3,), dtype=mx.int32)
        self.audio_prompt_latents = {
            config.speaker_name: mx.zeros(
                (1, config.audio_prompt_frames, config.tts.hidden_size),
                dtype=mx.float32,
            )
        }
        self.tts_model = EARTTSModel(config.tts)
        self.audio_codec = NemotronVoiceChatCodec(config.codec)

    def set_tokenizer(self, tokenizer) -> None:
        self.tts_model.embed_subword.set_tokenizer(tokenizer)

    def initialize(
        self, *, batch_size: int, pad_id: int, eos_id: int
    ) -> tuple[mx.array, list]:
        prompt_latent = self.audio_prompt_latents[self.config.speaker_name]
        prompt_latent = mx.broadcast_to(
            prompt_latent, (batch_size,) + prompt_latent.shape[1:]
        )
        prompt_length = prompt_latent.shape[1]
        codes = mx.broadcast_to(
            self.codec_silence_tokens[None, None],
            (batch_size, prompt_length, self.config.codec.num_quantizers),
        ).astype(mx.int32)
        codes[:, 0] = self.config.codec.codebook_size
        codes[:, -1] = self.config.codec.codebook_size
        subword_ids = mx.full((batch_size, prompt_length), pad_id, dtype=mx.int32)
        subword_ids[:, -1] = eos_id
        subword_mask = mx.zeros((batch_size, prompt_length), dtype=mx.bool_)
        subword_mask[:, -2:] = True
        audio_mask = mx.zeros((batch_size, prompt_length), dtype=mx.bool_)
        audio_mask[:, -1] = True
        return self.tts_model.warmup(
            codes,
            audio_mask,
            subword_ids,
            subword_mask,
            prompt_latent,
            guidance_enabled=True,
        )


class Model(nn.Module):
    def __init__(self, config: ModelConfig | NemotronVoiceChatConfig):
        super().__init__()
        if isinstance(config, ModelConfig):
            config = config.config
        self.config = config
        self.model_type = config.model_type
        self.stt_model = VoiceChatSTT(config)
        self.tts_model = VoiceChatTTS(config)
        self.tokenizer = None

    @staticmethod
    def post_load_hook(model: "Model", model_path: Path) -> "Model":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model.config.pretrained_llm)
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "<SPECIAL_12>"
        model.tokenizer = tokenizer
        model.tts_model.set_tokenizer(tokenizer)
        return model

    def _require_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("VoiceChat tokenizer has not been initialized")
        return self.tokenizer

    def __call__(self, inputs_embeds: mx.array, cache=None, **kwargs):
        return self.stt_model(inputs_embeds, cache=cache)

    def create_session(self):
        from .session import VoiceChatSession

        return VoiceChatSession(self, self._require_tokenizer())

    def create_duplex_session(self, **kwargs):
        return self.create_session().create_streaming_session(**kwargs)

    def generate(
        self,
        audio: str | Path | mx.array,
        *,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        verbose: bool = False,
        **kwargs,
    ):
        result = self.create_session().generate(
            audio,
            system_prompt=system_prompt,
            **kwargs,
        )
        if verbose:
            print(result.text)
        return result

    @property
    def layers(self):
        return self.stt_model.llm.layers

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        if self.config.prepared_weights:
            return dict(weights)
        return sanitize_weights(weights, self.tts_model.audio_codec)

    @property
    def cast_predicate(self):
        def predicate(key: str) -> bool:
            return "A_log" not in key and not key.endswith(
                ("special_flags", "is_continuation", "pad_tensor")
            )

        return predicate
