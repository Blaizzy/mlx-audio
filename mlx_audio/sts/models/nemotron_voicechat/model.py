from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.nemotron_voicechat import NemotronVoiceChatCodec
from mlx_audio.lm.models.nemotron_h import NemotronHModel
from mlx_audio.lm.sample_utils import make_logits_processors, make_sampler
from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram
from mlx_audio.stt.models.nemotron_asr.conformer import Conformer
from mlx_audio.utils import load_audio

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
class VoiceChatOutput:
    text: str
    audio: mx.array
    text_tokens: mx.array
    audio_tokens: mx.array
    sample_rate: int


def sanitize_weights(
    weights: Mapping[str, mx.array], codec: NemotronVoiceChatCodec
) -> dict[str, mx.array]:
    converted: dict[str, mx.array] = {}
    codec_prefix = "tts_model.audio_codec."
    for key, value in weights.items():
        if key in {
            "tts_model._control_codes",
            "tts_model.tts_model.audio_prompt_projection_W",
        }:
            continue
        # NVIDIA's shipped offline configuration disables the RNNT branch.
        if key.startswith("stt_model.rnnt_"):
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

    def __call__(self, waveform: mx.array) -> tuple[mx.array, mx.array]:
        mel = log_mel_spectrogram(waveform, self.config.preprocessor)
        encoded, lengths = self.encoder(
            mel, att_context_size=self.config.encoder.att_context_size[0]
        )
        return self.proj(encoded), lengths


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

    def initialize(
        self,
        waveform: mx.array,
        prompt_tokens: mx.array | None,
        *,
        pad_id: int,
    ) -> dict:
        audio_embeddings, audio_lengths = self.perception(waveform)
        audio_length = int(audio_lengths[0])
        audio_embeddings = audio_embeddings[:, :audio_length]
        prompt_length = 0
        if prompt_tokens is not None and prompt_tokens.shape[-1] > 0:
            prompt_embeddings = self.embed_tokens(prompt_tokens)
            prompt_length = prompt_tokens.shape[-1]
            audio_embeddings = mx.concatenate(
                [prompt_embeddings, audio_embeddings], axis=1
            )

        total_length = audio_embeddings.shape[1]
        return {
            "audio_embeddings": audio_embeddings,
            "text_tokens": mx.full(
                (audio_embeddings.shape[0], total_length), pad_id, dtype=mx.int32
            ),
            "function_tokens": mx.full(
                (audio_embeddings.shape[0], total_length), pad_id, dtype=mx.int32
            ),
            "fused_embeddings": mx.zeros_like(audio_embeddings),
            "prompt_length": prompt_length,
        }

    def step(
        self,
        index: int,
        state: dict,
        *,
        temperature: float,
        top_p: float,
        repetition_penalty: float | None,
        presence_penalty: float | None,
    ) -> mx.array:
        previous_text = state["text_tokens"][:, max(index - 1, 0)]
        previous_function = (
            state["function_tokens"][:, max(index - 1, 0)]
            if self.config.use_function_head
            else None
        )

        fused = (
            self.config.audio_channel_weight
            * state["audio_embeddings"][:, index : index + 1]
            + self.config.text_channel_weight
            * self.embed_tokens(previous_text)[:, None]
        )
        if previous_function is not None:
            fused = (
                fused
                + self.config.function_channel_weight
                * self.embed_tokens(previous_function)[:, None]
            )

        state["fused_embeddings"][:, index : index + 1] = fused
        if index < state["prompt_length"]:
            return state["text_tokens"][:, index : index + 1]

        hidden = self.llm(input_embeddings=state["fused_embeddings"][:, : index + 1])
        logits = self.lm_head(hidden)[:, -1]
        history = state["text_tokens"][:, :index]
        processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )
        for processor in processors:
            logits = processor(history, logits)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        state["text_tokens"][:, index] = sampler(logits)
        if self.config.use_function_head:
            state["function_tokens"][:, index] = mx.argmax(
                self.function_head(hidden)[:, -1], axis=-1
            )
        return state["text_tokens"][:, index : index + 1]


class VoiceChatTTS(nn.Module):
    def __init__(self, config: NemotronVoiceChatConfig):
        super().__init__()
        self.config = config
        self.codec_silence_tokens = mx.zeros(
            (config.codec.num_quantizers,), dtype=mx.int64
        )
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

    def _prompt_tokens(self, system_prompt: str | None) -> mx.array | None:
        if not system_prompt:
            return None
        tokenizer = self._require_tokenizer()
        token_ids = [tokenizer.bos_token_id]
        token_ids.extend(tokenizer.encode(system_prompt, add_special_tokens=False))
        token_ids.append(tokenizer.eos_token_id)
        return mx.array([token_ids], dtype=mx.int32)

    def generate(
        self,
        audio: str | Path | mx.array,
        *,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float | None = 1.0,
        presence_penalty: float | None = 0.0,
        verbose: bool = False,
        **kwargs,
    ) -> VoiceChatOutput:
        tokenizer = self._require_tokenizer()
        if isinstance(audio, Path):
            audio = str(audio)
        waveform = load_audio(audio, sample_rate=self.config.source_sample_rate)
        if waveform.ndim != 1:
            waveform = waveform.squeeze()
        prompt_tokens = self._prompt_tokens(system_prompt)
        state = self.stt_model.initialize(
            waveform,
            prompt_tokens,
            pad_id=tokenizer.pad_token_id,
        )
        total_length = state["text_tokens"].shape[1]
        generated_codes = mx.zeros(
            (1, total_length, self.config.codec.num_quantizers),
            dtype=mx.int32,
        )
        previous_codes, tts_cache = self.tts_model.initialize(
            batch_size=1,
            pad_id=tokenizer.pad_token_id,
            eos_id=tokenizer.eos_token_id,
        )

        for index in range(total_length):
            current_text = self.stt_model.step(
                index,
                state,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
            )
            if index > 0:
                previous_codes, tts_cache = self.tts_model.tts_model.generate_step(
                    current_text,
                    previous_codes,
                    tts_cache,
                    text_eos_id=tokenizer.eos_token_id,
                    silence_codes=self.tts_model.codec_silence_tokens[None, None],
                    guidance_enabled=True,
                )
                generated_codes[:, index] = previous_codes[:, 0]
            mx.eval(current_text, previous_codes)

        prompt_length = state["prompt_length"]
        text_tokens = state["text_tokens"][:, prompt_length:]
        audio_tokens = generated_codes[:, prompt_length:]
        special_ids = {
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        }
        silence_id = tokenizer.convert_tokens_to_ids("<SPECIAL_11>")
        if silence_id is not None:
            special_ids.add(silence_id)
        decoded_ids = [
            int(token)
            for token in text_tokens[0].tolist()
            if int(token) not in special_ids
        ]
        text = tokenizer.decode(decoded_ids, skip_special_tokens=False).strip()
        decoded_audio = self.tts_model.audio_codec.decode(
            audio_tokens.transpose(0, 2, 1)
        )[0, 0]
        mx.eval(decoded_audio)
        if verbose:
            print(text)
        return VoiceChatOutput(
            text=text,
            audio=decoded_audio,
            text_tokens=text_tokens,
            audio_tokens=audio_tokens,
            sample_rate=self.config.target_sample_rate,
        )

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        if self.config.prepared_weights:
            return dict(weights)
        return sanitize_weights(weights, self.tts_model.audio_codec)
