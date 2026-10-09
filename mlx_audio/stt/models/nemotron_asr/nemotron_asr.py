"""Nemotron 3.5 ASR (streaming, 0.6B) for MLX.

A cache-aware streaming FastConformer-RNNT with language-ID prompt conditioning
(NeMo ``EncDecRNNTBPEModelWithPrompt``). Run offline, the chunked-limited attention
mask reproduces the training-time look-ahead, so a single full-utterance pass gives
the same result the streaming model would.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.stt.models.nemo.alignment import (
    AlignedResult,
    sentences_to_result,
    tokens_to_sentences,
)
from mlx_audio.stt.streaming import StreamingSession
from mlx_audio.stt.utils import load_audio
from mlx_audio.utils import from_dict

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
from .rnnt import GreedyDecoderState, JointNetwork, PredictNetwork


class ModelConfig:
    """Wrapper so the shared loader can build the config via ``from_dict``."""

    def __init__(self, config: NemotronASRConfig):
        self.config = config

    @classmethod
    def from_dict(cls, config: dict) -> "ModelConfig":
        if config.get("model_type") == "nemotron_asr_streaming":
            return cls(cls._from_hf(config))
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

    @staticmethod
    def _from_hf(config: dict) -> NemotronASRConfig:
        encoder = config["encoder_config"]
        path = Path(config["model_path"])
        tokenizer = json.loads((path / "tokenizer.json").read_text())
        vocabulary = tokenizer["model"]["vocab"]
        blank_id = config["blank_token_id"]
        if set(vocabulary.values()) != set(range(blank_id)):
            raise ValueError(
                "Nemotron streaming vocabulary must be contiguous before blank"
            )
        vocabulary = [
            piece for piece, _ in sorted(vocabulary.items(), key=lambda x: x[1])
        ]
        lookaheads = json.loads((path / "processor_config.json").read_text())[
            "supported_num_lookahead_tokens"
        ]
        left_context = encoder["sliding_window"] - 1
        supported = [[left_context, value] for value in lookaheads]
        return NemotronASRConfig(
            preprocessor=PreprocessArgs(
                sample_rate=16000,
                features=encoder["num_mel_bins"],
                n_fft=512,
                window_size=0.025,
                window_stride=0.01,
                preemph=0.97,
                normalize="NA",
                pad_mode="constant",
            ),
            encoder=ConformerArgs(
                feat_in=encoder["num_mel_bins"],
                n_layers=encoder["num_hidden_layers"],
                d_model=encoder["hidden_size"],
                n_heads=encoder["num_attention_heads"],
                ff_expansion_factor=encoder["intermediate_size"]
                // encoder["hidden_size"],
                subsampling_factor=encoder["subsampling_factor"],
                subsampling_conv_channels=encoder["subsampling_conv_channels"],
                conv_kernel_size=encoder["conv_kernel_size"],
                causal_downsampling=True,
                conv_context_size="causal",
                conv_norm_type="layer_norm",
                self_attention_model="rel_pos",
                att_context_style="chunked_limited",
                att_context_size=supported,
                pos_emb_max_len=encoder["max_position_embeddings"],
                use_bias=encoder.get("attention_bias", False),
                xscaling=encoder.get("scale_input", False),
            ),
            prompt=PromptArgs(num_prompts=0, prompt_hidden=0),
            decoder=PredictArgs(
                pred_hidden=config["decoder_hidden_size"],
                pred_rnn_layers=config["num_decoder_layers"],
                vocab_size=blank_id,
                blank_as_pad=True,
            ),
            joint=JointArgs(
                joint_hidden=config["decoder_hidden_size"],
                activation=config["hidden_act"],
                encoder_hidden=encoder["hidden_size"],
                pred_hidden=config["decoder_hidden_size"],
                num_classes=blank_id,
            ),
            vocabulary=vocabulary,
            model_type="nemotron_asr_streaming",
            target=config["architectures"][0],
            default_language="en",
            default_att_context_size=[
                left_context,
                encoder["default_num_lookahead_tokens"],
            ],
            max_symbols=config["max_symbols_per_step"],
        )


class Model(nn.Module):
    def __init__(self, config: Union[ModelConfig, NemotronASRConfig]):
        super().__init__()
        if isinstance(config, ModelConfig):
            config = config.config
        self.config = config
        self.model_type = config.model_type

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
        self.prompt_kernel = None
        if config.prompt.num_prompts:
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

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        if self.model_type != "nemotron_asr_streaming":
            return weights

        converted = {}
        ignored = {
            f"encoder.layers.{i}.conv.norm.num_batches_tracked"
            for i in range(len(self.encoder.layers))
        }
        replacements = (
            ("encoder.subsampling.conv_in.", "encoder.pre_encode.conv.0."),
            (
                "encoder.subsampling.layers.0.depthwise_conv.",
                "encoder.pre_encode.conv.2.",
            ),
            (
                "encoder.subsampling.layers.0.pointwise_conv.",
                "encoder.pre_encode.conv.3.",
            ),
            (
                "encoder.subsampling.layers.1.depthwise_conv.",
                "encoder.pre_encode.conv.5.",
            ),
            (
                "encoder.subsampling.layers.1.pointwise_conv.",
                "encoder.pre_encode.conv.6.",
            ),
            ("encoder.subsampling.linear.", "encoder.pre_encode.out."),
            (".conv.norm.", ".conv.batch_norm."),
            (".self_attn.q_proj.", ".self_attn.linear_q."),
            (".self_attn.k_proj.", ".self_attn.linear_k."),
            (".self_attn.v_proj.", ".self_attn.linear_v."),
            (".self_attn.o_proj.", ".self_attn.linear_out."),
            (".self_attn.relative_k_proj.", ".self_attn.linear_pos."),
            (".self_attn.bias_u", ".self_attn.pos_bias_u"),
            (".self_attn.bias_v", ".self_attn.pos_bias_v"),
            ("decoder.embedding.", "decoder.prediction.embed."),
            ("decoder.decoder_projector.", "joint.pred."),
            ("encoder_projector.", "joint.enc."),
            ("joint.head.", "joint.joint_net.2."),
        )
        for name, value in weights.items():
            if name in ignored or re.fullmatch(r"decoder.lstm.bias_(ih|hh)_l\d+", name):
                continue
            target = name
            for old, new in replacements:
                target = target.replace(old, new)
            match = re.fullmatch(r"decoder\.lstm\.weight_(ih|hh)_l(\d+)", name)
            if match:
                weight = "Wx" if match[1] == "ih" else "Wh"
                target = f"decoder.prediction.dec_rnn.lstm.{match[2]}.{weight}"
            if value.ndim == 3:
                value = value.transpose(0, 2, 1)
            elif value.ndim == 4:
                value = value.transpose(0, 2, 3, 1)
            if target in converted:
                raise ValueError(f"Duplicate Nemotron streaming weight: {target}")
            converted[target] = value

        for layer in range(self.decoder.prediction["dec_rnn"].num_layers):
            ih = f"decoder.lstm.bias_ih_l{layer}"
            hh = f"decoder.lstm.bias_hh_l{layer}"
            if ih not in weights or hh not in weights:
                raise ValueError(f"Missing Nemotron LSTM biases for layer {layer}")
            converted[f"decoder.prediction.dec_rnn.lstm.{layer}.bias"] = (
                weights[ih] + weights[hh]
            )

        expected = dict(tree_flatten(self.parameters()))
        missing = expected.keys() - converted.keys()
        extra = converted.keys() - expected.keys()
        if missing or extra:
            raise ValueError(
                "Nemotron streaming weight mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for name, value in converted.items():
            if value.shape != expected[name].shape:
                raise ValueError(
                    f"Nemotron streaming weight shape mismatch for {name}: "
                    f"{value.shape} != {expected[name].shape}"
                )
        return converted

    def create_streaming_session(
        self,
        *,
        temperature=0.0,
        language=None,
        transcription_delay_ms=None,
    ) -> StreamingSession:
        """Create an independent live-input greedy transcription session."""
        from .session import NemotronStreamingSession

        return NemotronStreamingSession(
            self,
            temperature=temperature,
            language=language,
            transcription_delay_ms=transcription_delay_ms,
        )

    def _prepare_audio(
        self, audio: Union[str, Path, mx.array], dtype: mx.Dtype
    ) -> mx.array:
        if isinstance(audio, (str, Path)):
            return load_audio(audio, self.preprocessor_config.sample_rate, dtype=dtype)
        return mx.array(audio, dtype=dtype)

    def create_speaker_streaming_session(self, diarization_model, **kwargs):
        """Create a speaker-masked PCM session with independent ASR caches.

        Configure the Nemotron diarization preset before creating the session.
        ``feed(pcm)`` returns speaker-tagged token deltas; flush with
        ``feed([], final=True)``. See :class:`SpeakerStreamingSession` for options.
        """
        from .speaker_streaming import SpeakerStreamingSession

        return SpeakerStreamingSession(self, diarization_model, **kwargs)

    def stream_generate_speakers(self, audio, diarization_model, **kwargs):
        """Yield speaker-tagged token deltas from a file, waveform, or PCM iterable.

        Arrays and iterable chunks must be mono at the model sample rate.
        Options include ``language``, ``threshold``, ``att_context_size``,
        ``cache_gating`` and ``cache_gating_buffer_size`` (default 2 ASR chunks).
        """
        import numpy as np

        session = self.create_speaker_streaming_session(diarization_model, **kwargs)
        if isinstance(audio, (str, Path, mx.array, np.ndarray)):
            waveform = self._prepare_audio(audio, mx.float32)
            step = session.chunk_mel * self.preprocessor_config.hop_length
            audio = (waveform[i : i + step] for i in range(0, len(waveform), step))
        for samples in audio:
            yield from session.feed(samples)
        yield from session.feed([], final=True)

    def generate_speakers(self, audio, diarization_model, **kwargs):
        """Return ``{speaker_id: AlignedResult}`` using speaker-masked ASR streams.

        Speaker IDs are session-local arrival-order labels. Masking cannot
        separate simultaneous voices; token timestamps remain emission times.
        """
        tokens = {}
        for delta in self.stream_generate_speakers(audio, diarization_model, **kwargs):
            tokens.setdefault(delta.speaker, []).extend(delta.tokens)
        return {
            speaker: sentences_to_result(tokens_to_sentences(hypothesis))
            for speaker, hypothesis in tokens.items()
        }

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
        if self.prompt_kernel is None:
            return encoded
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

        hypothesis = GreedyDecoderState(self).decode(encoded[:, : int(lengths[0])], 0)
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
        decoder = GreedyDecoderState(self)
        hypothesis = []
        global_time = 0
        for prompted in prompted_chunks:
            hypothesis.extend(decoder.decode(prompted, global_time))
            global_time += prompted.shape[1]
            yield sentences_to_result(tokens_to_sentences(hypothesis))
            mx.clear_cache()
