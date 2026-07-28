"""Nemotron 3.5 ASR (streaming, 0.6B) for MLX.

A cache-aware streaming FastConformer-RNNT with language-ID prompt conditioning
(NeMo ``EncDecRNNTBPEModelWithPrompt``). Run offline, the chunked-limited attention
mask reproduces the training-time look-ahead, so a single full-utterance pass gives
the same result the streaming model would.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.nemo.alignment import (
    AlignedResult,
    AlignedToken,
    sentences_to_result,
    tokens_to_sentences,
)
from mlx_audio.stt.utils import load_audio
from mlx_audio.utils import from_dict

from . import tokenizer as tok
from .audio import iter_log_mel_spectrogram, log_mel_spectrogram
from .config import (
    ConformerArgs,
    JointArgs,
    NemotronASRConfig,
    PredictArgs,
    PreprocessArgs,
    PromptArgs,
)
from .conformer import Conformer
from .rnnt import JointNetwork, PredictNetwork


class ModelConfig:
    """Wrapper so the shared loader can build the config via ``from_dict``."""

    def __init__(self, config: NemotronASRConfig):
        self.config = config

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        cfg = NemotronASRConfig(
            preprocessor=from_dict(PreprocessArgs, config.get("preprocessor", {})),
            encoder=from_dict(ConformerArgs, config.get("encoder", {})),
            prompt=from_dict(PromptArgs, config.get("prompt", {})),
            decoder=from_dict(PredictArgs, config.get("decoder", {})),
            joint=from_dict(JointArgs, config.get("joint", {})),
            vocabulary=config.get("vocabulary", []),
            model_type=config.get("model_type", "nemotron_asr"),
            target=config.get("target", NemotronASRConfig.target),
            default_language=config.get("default_language", "auto"),
            default_att_context_size=config.get("default_att_context_size", [56, 13]),
            max_symbols=config.get("max_symbols", 10),
        )
        return cls(cfg)


class Model(nn.Module):
    def __init__(self, config: Union[ModelConfig, NemotronASRConfig]):
        super().__init__()
        if isinstance(config, ModelConfig):
            config = config.config
        self.config = config

        self.preprocessor_config = config.preprocessor
        self.encoder_config = config.encoder
        self.vocabulary = config.vocabulary
        self.prompt_dictionary = config.prompt.prompt_dictionary
        self.num_prompts = config.prompt.num_prompts
        self.blank_id = config.decoder.vocab_size  # == num_classes
        self.max_symbols = config.max_symbols
        self.default_language = config.default_language
        self.default_att_context_size = config.default_att_context_size

        self.encoder = Conformer(config.encoder)
        # prompt_kernel: Sequential(Linear, ReLU, Linear) — list keeps keys 0/2.
        self.prompt_kernel = [
            nn.Linear(
                config.encoder.d_model + config.prompt.num_prompts,
                config.prompt.prompt_hidden,
            ),
            nn.ReLU(),
            nn.Linear(config.prompt.prompt_hidden, config.encoder.d_model),
        ]
        self.decoder = PredictNetwork(config.decoder)
        self.joint = JointNetwork(config.joint)

    def _prepare_audio(
        self, audio: Union[str, Path, mx.array], dtype: mx.Dtype
    ) -> mx.array:
        if isinstance(audio, (str, Path)):
            return load_audio(audio, self.preprocessor_config.sample_rate, dtype=dtype)
        return audio.astype(dtype) if audio.dtype != dtype else audio

    def _mel_chunk_frames(self, chunk_duration: float) -> int:
        if chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive")
        return max(
            int(
                chunk_duration
                * self.preprocessor_config.sample_rate
                / self.preprocessor_config.hop_length
            ),
            1,
        )

    # ------------------------------------------------------------------ prompt
    def _resolve_prompt_index(self, language: Optional[str]) -> int:
        lang = language or self.default_language
        if lang in self.prompt_dictionary:
            return self.prompt_dictionary[lang]
        if self.default_language in self.prompt_dictionary:
            return self.prompt_dictionary[self.default_language]
        return 0

    def apply_prompt(self, encoded: mx.array, language: Optional[str]) -> mx.array:
        """Concatenate the one-hot language prompt and project back to d_model."""
        idx = self._resolve_prompt_index(language)
        b, t, _ = encoded.shape
        one_hot = mx.zeros((b, t, self.num_prompts), dtype=encoded.dtype)
        one_hot[:, :, idx] = 1.0
        x = mx.concatenate([encoded, one_hot], axis=-1)
        for layer in self.prompt_kernel:
            x = layer(x)
        return x

    # ------------------------------------------------------------------ decode
    def decode(
        self,
        mel: mx.array,
        language: Optional[str] = None,
        att_context_size: Optional[list] = None,
    ) -> AlignedResult:
        """Greedy RNN-T decode of a single mel spectrogram (1, T, F) or (T, F)."""
        if mel.ndim == 2:
            mel = mx.expand_dims(mel, 0)

        if (
            mel.shape[0] == 1
            and self.encoder_config.att_context_style == "chunked_limited"
        ):
            from .streaming import stream_encode

            result = None
            for result in self._decode_prompted_chunks(
                stream_encode(
                    self,
                    mel,
                    language or self.default_language,
                    att_context_size=att_context_size,
                )
            ):
                pass
            return result or sentences_to_result([])

        encoded, lengths = self.encoder(
            mel, att_context_size=att_context_size or self.default_att_context_size
        )
        encoded = self.apply_prompt(encoded, language)
        mx.eval(encoded, lengths)

        features = encoded
        max_length = int(lengths[0])

        frame_sec = (
            self.encoder_config.subsampling_factor
            * self.preprocessor_config.hop_length
            / self.preprocessor_config.sample_rate
        )

        last_token = self.blank_id
        decoder_hidden = None
        hypothesis: list[AlignedToken] = []
        time = 0
        new_symbols = 0

        while time < max_length:
            feature = features[:, time : time + 1]
            current_token = (
                mx.array([[last_token]], dtype=mx.int32)
                if last_token != self.blank_id
                else None
            )
            decoder_output, (h, c) = self.decoder(current_token, decoder_hidden)
            decoder_output = decoder_output.astype(feature.dtype)
            proposed_hidden = (h.astype(feature.dtype), c.astype(feature.dtype))

            joint_output = self.joint(feature, decoder_output)
            pred_token = int(mx.argmax(joint_output))

            if pred_token != self.blank_id:
                last_token = pred_token
                decoder_hidden = proposed_hidden
                if not tok.is_special_token(last_token, self.vocabulary):
                    hypothesis.append(
                        AlignedToken(
                            last_token,
                            start=time * frame_sec,
                            duration=frame_sec,
                            text=tok.decode([last_token], self.vocabulary),
                        )
                    )
                new_symbols += 1
                if self.max_symbols is not None and new_symbols >= self.max_symbols:
                    time += 1
                    new_symbols = 0
            else:
                time += 1
                new_symbols = 0

        return sentences_to_result(tokens_to_sentences(hypothesis))

    # ---------------------------------------------------------------- generate
    def generate(
        self,
        audio: Union[str, Path, mx.array],
        *,
        language: Optional[str] = None,
        att_context_size: Optional[list] = None,
        chunk_duration: Optional[float] = 30.0,
        dtype: mx.Dtype = mx.float32,
        verbose: bool = False,
        **kwargs,
    ) -> AlignedResult:
        """Transcribe an audio file or waveform. ``language`` is a prompt key
        (e.g. ``"en-US"``, ``"auto"``); defaults to the model's default."""
        kwargs.pop("generation_stream", None)
        kwargs.pop("max_tokens", None)

        audio_data = self._prepare_audio(audio, dtype)

        if chunk_duration is None:
            mel = log_mel_spectrogram(audio_data, self.preprocessor_config)
            result = self.decode(
                mel, language=language, att_context_size=att_context_size
            )
            mx.clear_cache()
        else:
            result = None
            for result in self._stream_generate_audio_data(
                audio_data,
                language=language,
                chunk_duration=chunk_duration,
                att_context_size=att_context_size,
            ):
                pass
            result = result or sentences_to_result([])

        if verbose:
            print(result.text)
        return result

    # ---------------------------------------------------------- stream_generate
    def stream_generate(
        self,
        audio: Union[str, Path, mx.array],
        *,
        language: Optional[str] = None,
        chunk_frames: Optional[int] = None,
        chunk_duration: float = 30.0,
        att_context_size: Optional[list] = None,
        dtype: mx.Dtype = mx.float32,
        **kwargs,
    ):
        """Cache-aware streaming transcription.

        Yields a cumulative ``AlignedResult`` per chunk as audio is processed, using
        per-layer attention/conv caches and incremental subsampling (O(n), no
        recompute). Token-identical to :meth:`generate` at the native chunk size.
        """
        audio_data = self._prepare_audio(audio, dtype)
        yield from self._stream_generate_audio_data(
            audio_data,
            language=language,
            chunk_frames=chunk_frames,
            chunk_duration=chunk_duration,
            att_context_size=att_context_size,
        )

    def _stream_generate_audio_data(
        self,
        audio_data: mx.array,
        *,
        language: Optional[str] = None,
        chunk_frames: Optional[int] = None,
        chunk_duration: float = 30.0,
        att_context_size: Optional[list] = None,
    ):
        from .streaming import stream_encode_chunks

        mel_chunks = iter_log_mel_spectrogram(
            audio_data,
            self.preprocessor_config,
            chunk_frames=self._mel_chunk_frames(chunk_duration),
        )
        prompted_chunks = stream_encode_chunks(
            self,
            mel_chunks,
            language or self.default_language,
            chunk_frames=chunk_frames,
            att_context_size=att_context_size,
        )
        try:
            yield from self._decode_prompted_chunks(prompted_chunks)
        finally:
            mx.clear_cache()

    def _decode_prompted_chunks(self, prompted_chunks):
        frame_sec = (
            self.encoder_config.subsampling_factor
            * self.preprocessor_config.hop_length
            / self.preprocessor_config.sample_rate
        )
        last_token = self.blank_id
        decoder_hidden = None
        hypothesis: list[AlignedToken] = []
        global_time = 0

        for prompted in prompted_chunks:
            chunk_len = prompted.shape[1]
            time = 0
            new_symbols = 0
            while time < chunk_len:
                feature = prompted[:, time : time + 1]
                current_token = (
                    mx.array([[last_token]], dtype=mx.int32)
                    if last_token != self.blank_id
                    else None
                )
                decoder_output, (h, c) = self.decoder(current_token, decoder_hidden)
                decoder_output = decoder_output.astype(feature.dtype)
                proposed_hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
                joint_output = self.joint(feature, decoder_output)
                pred_token = int(mx.argmax(joint_output))
                if pred_token != self.blank_id:
                    last_token = pred_token
                    decoder_hidden = proposed_hidden
                    if not tok.is_special_token(last_token, self.vocabulary):
                        hypothesis.append(
                            AlignedToken(
                                last_token,
                                start=(global_time + time) * frame_sec,
                                duration=frame_sec,
                                text=tok.decode([last_token], self.vocabulary),
                            )
                        )
                    new_symbols += 1
                    if self.max_symbols is not None and new_symbols >= self.max_symbols:
                        time += 1
                        new_symbols = 0
                else:
                    time += 1
                    new_symbols = 0
            global_time += chunk_len
            yield sentences_to_result(tokens_to_sentences(hypothesis))
            mx.clear_cache()
