# Copyright © 2023 Apple Inc.

import base64
import gzip
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tqdm
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult
from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .timing import add_word_timestamps
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer


class HFTokenizerWrapper:
    """
    Wrapper around HuggingFace WhisperTokenizer that provides a compatible interface
    with the tiktoken-based Tokenizer class.
    """

    def __init__(
        self,
        hf_tokenizer,
        multilingual: bool = True,
        language: str = None,
        task: str = "transcribe",
    ):
        self.hf_tokenizer = hf_tokenizer
        self.multilingual = multilingual
        self.language = language or "en"
        self.task = task or "transcribe"

        # Normalize language code
        if self.language in TO_LANGUAGE_CODE:
            self.language = TO_LANGUAGE_CODE[self.language]

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        return self.hf_tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens, skip_special_tokens: bool = False) -> str:
        """Decode token ids to text."""
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return self.hf_tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @property
    def eot(self) -> int:
        """End of transcript token."""
        return self.hf_tokenizer.eos_token_id

    @property
    def timestamp_begin(self) -> int:
        """First timestamp token id."""
        # Whisper timestamp tokens start at <|0.00|>
        return self.hf_tokenizer.convert_tokens_to_ids("<|0.00|>")

    @property
    def sot(self) -> int:
        """Start of transcript token."""
        return self.hf_tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    @property
    def no_timestamps(self) -> int:
        """No timestamps token."""
        return self.hf_tokenizer.convert_tokens_to_ids("<|notimestamps|>")

    @property
    def no_speech(self) -> int:
        """No speech token."""
        return self.hf_tokenizer.convert_tokens_to_ids("<|nospeech|>")

    @property
    def transcribe(self) -> int:
        """Transcribe task token."""
        return self.hf_tokenizer.convert_tokens_to_ids("<|transcribe|>")

    @property
    def translate(self) -> int:
        """Translate task token."""
        return self.hf_tokenizer.convert_tokens_to_ids("<|translate|>")

    @property
    def sot_prev(self) -> int:
        """Start of previous token."""
        return self.hf_tokenizer.convert_tokens_to_ids("<|startofprev|>")

    @property
    def language_token(self) -> int:
        """Language token for the configured language."""
        return self.hf_tokenizer.convert_tokens_to_ids(f"<|{self.language}|>")

    @property
    def sot_sequence(self) -> Tuple[int, ...]:
        """Start of transcript sequence including language and task tokens."""
        if self.multilingual:
            return (self.sot, self.language_token, self.task_token)
        return (self.sot,)

    @property
    def task_token(self) -> int:
        """Task token (transcribe or translate)."""
        if self.task == "translate":
            return self.translate
        return self.transcribe


def _format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )


@dataclass
class STTOutput:
    text: str
    segments: List[dict] = None
    language: str = None


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

    @classmethod
    def from_dict(cls, config: dict) -> "ModelDimensions":
        """Create ModelDimensions from a config dict.

        Handles both MLX format (n_mels, n_audio_ctx, etc.) and
        HuggingFace transformers format (d_model, encoder_layers, etc.).
        """
        config = config.copy()

        # Check if this is HuggingFace format (has d_model or encoder_layers)
        if "d_model" in config or "encoder_layers" in config:
            # Map HuggingFace config to MLX format
            return cls(
                n_mels=config.get("num_mel_bins", 128),
                n_audio_ctx=config.get("max_source_positions", 1500),
                n_audio_state=config.get("d_model", 1280),
                n_audio_head=config.get("encoder_attention_heads", 20),
                n_audio_layer=config.get("encoder_layers", 32),
                n_vocab=config.get("vocab_size", 51866),
                n_text_ctx=config.get("max_target_positions", 448),
                n_text_state=config.get("d_model", 1280),
                n_text_head=config.get("decoder_attention_heads", 20),
                n_text_layer=config.get("decoder_layers", 32),
            )

        # MLX format - filter to known fields
        config.pop("model_type", None)
        config.pop("quantization", None)
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in known_fields}
        return cls(**filtered)


# Alias for compatibility with load_model
ModelConfig = ModelDimensions


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
    ):
        q = self.query(x)

        if xa is None:
            k = self.key(x)
            v = self.value(x)
            if kv_cache is not None:
                k = mx.concatenate([kv_cache[0], k], axis=1)
                v = mx.concatenate([kv_cache[1], v], axis=1)
        elif kv_cache is None:
            k = self.key(xa)
            v = self.value(xa)
        else:
            k, v = kv_cache

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), (k, v), qk

    def qkv_attention(self, q, k, v, mask=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = mx.softmax(qk, axis=-1, precise=True)
        out = (w @ v).transpose(0, 2, 1, 3)
        out = out.reshape(n_batch, n_ctx, n_state)
        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(self, x, xa=None, mask=None, kv_cache=None):
        kv, cross_kv = kv_cache if kv_cache else (None, None)
        y, kv, _ = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv)
        x += y
        cross_qk = None
        if self.cross_attn:
            y, cross_kv, cross_qk = self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=cross_kv
            )
            x += y
        x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))
        return x, (kv, cross_kv), cross_qk


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self._positional_embedding = sinusoids(n_ctx, n_state).astype(dtype)

        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = nn.LayerNorm(n_state)

    def __call__(self, x):
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        assert x.shape[1:] == self._positional_embedding.shape, "incorrect audio shape"
        x = x + self._positional_embedding

        for block in self.blocks:
            x, _, _ = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = mx.zeros((n_ctx, n_state))

        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True)
            for _ in range(n_layer)
        ]
        self.ln = nn.LayerNorm(n_state)
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(n_ctx).astype(
            dtype
        )

    def __call__(self, x, xa, kv_cache=None):
        """
        x : mx.array, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : mx.array, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        cross_qk = [None] * len(self.blocks)
        for e, block in enumerate(self.blocks):
            x, kv_cache[e], cross_qk[e] = block(
                x, xa, mask=self._mask, kv_cache=kv_cache[e]
            )

        x = self.ln(x)
        return self.token_embedding.as_linear(x), kv_cache, cross_qk


class Model(nn.Module):
    def __init__(self, dims: ModelDimensions, dtype: mx.Dtype = mx.float16):
        super().__init__()
        self.dims = dims
        self.dtype = dtype
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dtype,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dtype,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = np.zeros(
            (self.dims.n_text_layer, self.dims.n_text_head), dtype=bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self._alignment_heads = mx.array(np.asarray(all_heads.nonzero()).T)

    def set_alignment_heads(self, dump: Union[bytes, np.ndarray]):
        if isinstance(dump, np.ndarray):
            self._alignment_heads = mx.array(dump)
        elif isinstance(dump, bytes):
            array = np.frombuffer(
                gzip.decompress(base64.b85decode(dump)), dtype=bool
            ).copy()
            mask = array.reshape(self.dims.n_text_layer, self.dims.n_text_head)
            self._alignment_heads = mx.array(np.asarray(mask.nonzero()).T)
        else:
            raise ValueError(
                f"Invalid type for `dump`: {type(dump)}. Expected a np.ndarray or base85-encoded bytes containing"
                " alignment_head information"
            )

    def sanitize(self, weights: dict) -> dict:
        """Sanitize weights for MLX compatibility.

        Handles:
        - Removing 'model.' prefix (HuggingFace format)
        - Remapping HuggingFace key names to MLX key names
        - Transposing Conv1d weights from (out, in, kernel) to (out, kernel, in)
        - Converting to model dtype
        """
        # Key remapping from HuggingFace to MLX format
        # Order matters: more specific patterns must come before generic ones
        key_map = [
            ("encoder.embed_positions.weight", None),  # Skip, computed in MLX
            ("decoder.embed_positions.weight", "decoder.positional_embedding"),
            ("encoder.layer_norm.", "encoder.ln_post."),
            ("decoder.layer_norm.", "decoder.ln."),
            ("encoder.layers.", "encoder.blocks."),
            ("decoder.layers.", "decoder.blocks."),
            (".self_attn_layer_norm.", ".attn_ln."),
            (".final_layer_norm.", ".mlp_ln."),
            (".encoder_attn_layer_norm.", ".cross_attn_ln."),
            (".fc1.", ".mlp1."),
            (".fc2.", ".mlp2."),
            # Attention mappings - specific before generic
            (".self_attn.q_proj.", ".attn.query."),
            (".self_attn.k_proj.", ".attn.key."),
            (".self_attn.v_proj.", ".attn.value."),
            (".self_attn.out_proj.", ".attn.out."),
            (".encoder_attn.q_proj.", ".cross_attn.query."),
            (".encoder_attn.k_proj.", ".cross_attn.key."),
            (".encoder_attn.v_proj.", ".cross_attn.value."),
            (".encoder_attn.out_proj.", ".cross_attn.out."),
            ("decoder.embed_tokens.", "decoder.token_embedding."),
        ]

        # Check if this is HuggingFace format (has 'model.' prefix)
        is_hf_format = any(k.startswith("model.") for k in weights.keys())

        sanitized = {}
        for k, v in weights.items():
            if is_hf_format:
                # Remove 'model.' prefix if present (HuggingFace format)
                if k.startswith("model."):
                    k = k[6:]

                # Apply key remapping
                skip = False
                for old, new in key_map:
                    if old in k:
                        if new is None:
                            skip = True
                            break
                        k = k.replace(old, new)

                if skip:
                    continue

                # Transpose Conv1d weights: HF uses (out, in, kernel), MLX uses (out, kernel, in)
                if "conv1.weight" in k or "conv2.weight" in k:
                    if v.ndim == 3:
                        v = v.transpose(0, 2, 1)

            # Convert to model dtype
            if v.dtype != self.dtype and v.dtype != mx.uint32:
                v = v.astype(self.dtype)

            sanitized[k] = v
        return sanitized

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)[0]

    def forward_with_cross_qk(self, mel, tokens):
        logits, _, cross_qk = self.decoder(tokens, self.encoder(mel))
        return logits, cross_qk

    def __call__(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))[0]

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    detect_language = detect_language_function
    decode = decode_function

    @classmethod
    def from_pretrained(
        cls,
        path_or_hf_repo: str = "mlx-community/whisper-tiny-asr-fp16",
        dtype: mx.Dtype = mx.float16,
    ) -> "Whisper":
        """
        Load a pretrained Whisper model.

        .. deprecated::
            Use `mlx_audio.stt.load()` instead. This method will be removed in a future version.
        """
        import glob

        warnings.warn(
            "Model.from_pretrained() is deprecated. Use mlx_audio.stt.load() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        model_path = Path(path_or_hf_repo)
        if not model_path.exists():
            model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

        with open(str(model_path / "config.json"), "r") as f:
            config = json.loads(f.read())
            config.pop("model_type", None)
            quantization = config.pop("quantization", None)

        model_args = ModelDimensions.from_dict(config)

        wf = glob.glob(str(model_path / "*.safetensors"))
        weights = {}
        for wf in wf:
            weights.update(mx.load(wf))

        model = Model(model_args, dtype)

        if quantization is not None:
            class_predicate = (
                lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                and f"{p}.scales" in weights
            )
            nn.quantize(model, **quantization, class_predicate=class_predicate)

        weights = tree_unflatten(list(weights.items()))
        model.update(weights)
        mx.eval(model.parameters())
        return model

    @staticmethod
    def post_load_hook(model: "Model", model_path: Path) -> "Model":
        """
        Post-load hook called by load_model to initialize the processor.

        Args:
            model: The loaded model instance
            model_path: Path to the model directory

        Returns:
            The model with processor initialized
        """
        try:
            from transformers import WhisperProcessor

            processor = WhisperProcessor.from_pretrained(str(model_path))
            model._processor = processor
        except Exception as e:
            # Fallback: processor not available, will use tiktoken-based tokenizer
            model._processor = None
            warnings.warn(
                f"Could not load WhisperProcessor: {e}. Using tiktoken tokenizer."
            )

        return model

    def get_tokenizer(self, language: str = None, task: str = "transcribe"):
        """
        Get a tokenizer for the current model configuration.

        If a HuggingFace processor was loaded, wraps it to provide a compatible interface.
        Otherwise falls back to the tiktoken-based tokenizer.

        Args:
            language: Language code (e.g., "en", "ja"). If None, uses "en" for multilingual.
            task: Either "transcribe" or "translate".

        Returns:
            A tokenizer instance with encode/decode methods and special token properties.
        """
        if hasattr(self, "_processor") and self._processor is not None:
            return HFTokenizerWrapper(
                self._processor.tokenizer,
                multilingual=self.is_multilingual,
                language=language,
                task=task,
            )
        else:
            # Fallback to tiktoken-based tokenizer
            return get_tokenizer(
                self.is_multilingual,
                num_languages=self.num_languages,
                language=language,
                task=task,
            )

    def _prepare_audio(
        self, audio: Union[str, np.ndarray, mx.array], padding: int = N_SAMPLES
    ) -> Tuple[mx.array, int]:
        """Prepare audio for transcription.

        Args:
            audio: Path to audio file or audio waveform array.
            padding: Padding samples (default N_SAMPLES for 30s window).

        Returns:
            Tuple of (mel spectrogram, content_frames).
        """
        if isinstance(audio, str):
            from mlx_audio.stt.utils import load_audio

            audio = load_audio(audio)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        mel = log_mel_spectrogram(audio, n_mels=self.dims.n_mels, padding=padding)
        content_frames = mel.shape[-2] - N_FRAMES if padding else mel.shape[-2]

        return mel, content_frames

    def _detect_language(self, mel: mx.array, language: Optional[str] = None) -> str:
        """Detect or validate language for transcription.

        Args:
            mel: Mel spectrogram.
            language: Override language code. If None, auto-detect.

        Returns:
            Language code (e.g., "en", "ja").
        """
        if language is not None:
            return language

        if not self.is_multilingual:
            return "en"

        mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(self.dtype)
        _, probs = self.detect_language(mel_segment)
        return max(probs, key=probs.get)

    def generate(
        self,
        audio: Union[str, np.ndarray, mx.array],
        *,
        verbose: Optional[bool] = None,
        generation_stream: bool = False,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None,
        **decode_options,
    ):
        """
        Transcribe an audio file using Whisper

        Parameters
        ----------
        audio: Union[str, np.ndarray, mx.array]
            The path to the audio file to open, or the audio waveform

        verbose: bool
            Whether to display the text being decoded to the console. If True, displays all the details,
            If False, displays minimal details. If None, does not display anything

        temperature: Union[float, Tuple[float, ...]]
            Temperature for sampling. It can be a tuple of temperatures, which will be successively used
            upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

        compression_ratio_threshold: float
            If the gzip compression ratio is above this value, treat as failed

        logprob_threshold: float
            If the average log probability over sampled tokens is below this value, treat as failed

        no_speech_threshold: float
            If the no_speech probability is higher than this value AND the average log probability
            over sampled tokens is below `logprob_threshold`, consider the segment as silent

        condition_on_previous_text: bool
            if True, the previous output of the model is provided as a prompt for the next window;
            disabling may make the text inconsistent across windows, but the model becomes less prone to
            getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

        word_timestamps: bool
            Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
            and include the timestamps for each word in each segment.

        prepend_punctuations: str
            If word_timestamps is True, merge these punctuation symbols with the next word

        append_punctuations: str
            If word_timestamps is True, merge these punctuation symbols with the previous word

        initial_prompt: Optional[str]
            Optional text to provide as a prompt for the first window. This can be used to provide, or
            "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
            to make it more likely to predict those word correctly.

        decode_options: dict
            Keyword arguments to construct `DecodingOptions` instances

        clip_timestamps: Union[str, List[float]]
            Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.
            The last end timestamp defaults to the end of the file.

        hallucination_silence_threshold: Optional[float]
            When word_timestamps is True, skip silent periods longer than this threshold (in seconds)
            when a possible hallucination is detected

        Returns
        -------
        A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
        the spoken language ("language"), which is detected when `decode_options["language"]` is None.
        """

        decode_options.pop("max_tokens", None)
        decode_options.pop("generation_stream", None)

        # Use shared audio preparation
        mel, content_frames = self._prepare_audio(audio)
        content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)

        if verbose:
            system_encoding = sys.getdefaultencoding()
            if system_encoding != "utf-8":
                make_safe = lambda x: x.encode(
                    system_encoding, errors="replace"
                ).decode(system_encoding)
            else:
                make_safe = lambda x: x

        # Use shared language detection
        language = self._detect_language(mel, language=decode_options.get("language"))
        if decode_options.get("language") is None:
            if verbose:
                print(f"Detected language: {LANGUAGES[language].title()}")
        decode_options["language"] = language
        task: str = decode_options.get("task", "transcribe")
        tokenizer = self.get_tokenizer(language=language, task=task)

        if isinstance(clip_timestamps, str):
            clip_timestamps = [
                float(ts)
                for ts in (clip_timestamps.split(",") if clip_timestamps else [])
            ]
        seek_points: List[int] = [
            round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps
        ]
        if len(seek_points) == 0:
            seek_points.append(0)
        if len(seek_points) % 2 == 1:
            seek_points.append(content_frames)
        else:
            seek_points[-1] = min(content_frames, seek_points[-1])
        seek_clips: List[Tuple[int, int]] = list(
            zip(seek_points[::2], seek_points[1::2])
        )

        punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

        if word_timestamps and task == "translate":
            warnings.warn("Word-level timestamps on translations may not be reliable.")

        def decode_with_fallback(segment: mx.array) -> DecodingResult:
            temperatures = (
                [temperature] if isinstance(temperature, (int, float)) else temperature
            )
            decode_result = None

            for t in temperatures:
                kwargs = {**decode_options}
                if t > 0:
                    # disable beam_size and patience when t > 0
                    kwargs.pop("beam_size", None)
                    kwargs.pop("patience", None)
                else:
                    # disable best_of when t == 0
                    kwargs.pop("best_of", None)

                options = DecodingOptions(**kwargs, temperature=t)
                decode_result = self.decode(segment, options)

                needs_fallback = False
                if (
                    compression_ratio_threshold is not None
                    and decode_result.compression_ratio > compression_ratio_threshold
                ):
                    needs_fallback = True  # too repetitive
                if (
                    logprob_threshold is not None
                    and decode_result.avg_logprob < logprob_threshold
                ):
                    needs_fallback = True  # average log probability is too low
                if (
                    no_speech_threshold is not None
                    and decode_result.no_speech_prob > no_speech_threshold
                ):
                    needs_fallback = False  # silence
                if not needs_fallback:
                    break

            return decode_result

        clip_idx = 0
        seek = seek_clips[clip_idx][0]
        input_stride = (
            N_FRAMES // self.dims.n_audio_ctx
        )  # mel frames per output token: 2
        time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
        )  # time per output token: 0.02 (seconds)
        all_tokens = []
        all_segments = []
        prompt_reset_since = 0

        if initial_prompt is not None:
            initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
            all_tokens.extend(initial_prompt_tokens)
        else:
            initial_prompt_tokens = []

        def new_segment(
            *, start: float, end: float, tokens: mx.array, result: DecodingResult
        ):
            tokens = tokens.tolist()
            text_tokens = [token for token in tokens if token < tokenizer.eot]
            return {
                "seek": seek,
                "start": start,
                "end": end,
                "text": tokenizer.decode(text_tokens),
                "tokens": tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }

        # show the progress bar when verbose is False (if True, transcribed text will be printed)
        with tqdm.tqdm(
            total=content_frames, unit="frames", disable=verbose is not False
        ) as pbar:
            last_speech_timestamp = 0.0
            for seek_clip_start, seek_clip_end in seek_clips:
                while seek < seek_clip_end:
                    time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                    window_end_time = float(
                        (seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE
                    )
                    segment_size = min(
                        N_FRAMES, content_frames - seek, seek_clip_end - seek
                    )
                    mel_segment = mel[seek : seek + segment_size]
                    segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
                    mel_segment = pad_or_trim(mel_segment, N_FRAMES, axis=-2).astype(
                        self.dtype
                    )

                    decode_options["prompt"] = all_tokens[prompt_reset_since:]
                    result: DecodingResult = decode_with_fallback(mel_segment)

                    tokens = np.array(result.tokens)

                    if no_speech_threshold is not None:
                        # no voice activity check
                        should_skip = result.no_speech_prob > no_speech_threshold
                        if (
                            logprob_threshold is not None
                            and result.avg_logprob > logprob_threshold
                        ):
                            # don't skip if the logprob is high enough, despite the no_speech_prob
                            should_skip = False

                        if should_skip:
                            seek += segment_size  # fast-forward to the next segment boundary
                            continue

                    previous_seek = seek
                    current_segments = []

                    # anomalous words are very long/short/improbable
                    def word_anomaly_score(word: dict) -> float:
                        probability = word.get("probability", 0.0)
                        duration = word["end"] - word["start"]
                        score = 0.0
                        if probability < 0.15:
                            score += 1.0
                        if duration < 0.133:
                            score += (0.133 - duration) * 15
                        if duration > 2.0:
                            score += duration - 2.0
                        return score

                    def is_segment_anomaly(segment: Optional[dict]) -> bool:
                        if segment is None or not segment["words"]:
                            return False
                        words = [
                            w for w in segment["words"] if w["word"] not in punctuation
                        ]
                        words = words[:8]
                        score = sum(word_anomaly_score(w) for w in words)
                        return score >= 3 or score + 0.01 >= len(words)

                    def next_words_segment(segments: List[dict]) -> Optional[dict]:
                        return next((s for s in segments if s["words"]), None)

                    timestamp_tokens = tokens >= tokenizer.timestamp_begin
                    single_timestamp_ending = timestamp_tokens[-2:].tolist() == [
                        False,
                        True,
                    ]

                    consecutive = np.where(
                        np.logical_and(timestamp_tokens[:-1], timestamp_tokens[1:])
                    )[0]
                    consecutive += 1
                    if len(consecutive) > 0:
                        # if the output contains two consecutive timestamp tokens
                        slices = consecutive.tolist()
                        if single_timestamp_ending:
                            slices.append(len(tokens))

                        last_slice = 0
                        for current_slice in slices:
                            sliced_tokens = tokens[last_slice:current_slice]
                            start_timestamp_pos = (
                                sliced_tokens[0].item() - tokenizer.timestamp_begin
                            )
                            end_timestamp_pos = (
                                sliced_tokens[-1].item() - tokenizer.timestamp_begin
                            )
                            current_segments.append(
                                new_segment(
                                    start=time_offset
                                    + start_timestamp_pos * time_precision,
                                    end=time_offset
                                    + end_timestamp_pos * time_precision,
                                    tokens=sliced_tokens,
                                    result=result,
                                )
                            )
                            last_slice = current_slice

                        if single_timestamp_ending:
                            # single timestamp at the end means no speech after the last timestamp.
                            seek += segment_size
                        else:
                            # otherwise, ignore the unfinished segment and seek to the last timestamp
                            last_timestamp_pos = (
                                tokens[last_slice - 1].item()
                                - tokenizer.timestamp_begin
                            )
                            seek += last_timestamp_pos * input_stride
                    else:
                        duration = segment_duration
                        timestamps = tokens[timestamp_tokens.nonzero()[0]]
                        if (
                            len(timestamps) > 0
                            and timestamps[-1].item() != tokenizer.timestamp_begin
                        ):
                            # no consecutive timestamps but it has a timestamp; use the last one.
                            last_timestamp_pos = (
                                timestamps[-1].item() - tokenizer.timestamp_begin
                            )
                            duration = last_timestamp_pos * time_precision

                        current_segments.append(
                            new_segment(
                                start=time_offset,
                                end=time_offset + duration,
                                tokens=tokens,
                                result=result,
                            )
                        )
                        seek += segment_size

                    if word_timestamps:
                        add_word_timestamps(
                            segments=current_segments,
                            model=self,
                            tokenizer=tokenizer,
                            mel=mel_segment,
                            num_frames=segment_size,
                            prepend_punctuations=prepend_punctuations,
                            append_punctuations=append_punctuations,
                            last_speech_timestamp=last_speech_timestamp,
                        )

                        if not single_timestamp_ending:
                            last_word_end = _get_end(current_segments)
                            if (
                                last_word_end is not None
                                and last_word_end > time_offset
                            ):
                                seek = round(last_word_end * FRAMES_PER_SECOND)

                        # skip silence before possible hallucinations
                        if hallucination_silence_threshold is not None:
                            threshold = hallucination_silence_threshold
                            if not single_timestamp_ending:
                                last_word_end = _get_end(current_segments)
                                if (
                                    last_word_end is not None
                                    and last_word_end > time_offset
                                ):
                                    remaining_duration = window_end_time - last_word_end
                                    if remaining_duration > threshold:
                                        seek = round(last_word_end * FRAMES_PER_SECOND)
                                    else:
                                        seek = previous_seek + segment_size

                            # if first segment might be a hallucination, skip leading silence
                            first_segment = next_words_segment(current_segments)
                            if first_segment is not None and is_segment_anomaly(
                                first_segment
                            ):
                                gap = first_segment["start"] - time_offset
                                if gap > threshold:
                                    seek = previous_seek + round(
                                        gap * FRAMES_PER_SECOND
                                    )
                                    continue

                            # skip silence before any possible hallucination that is surrounded
                            # by silence or more hallucinations
                            hal_last_end = last_speech_timestamp
                            for si in range(len(current_segments)):
                                segment = current_segments[si]
                                if not segment["words"]:
                                    continue
                                if is_segment_anomaly(segment):
                                    next_segment = next_words_segment(
                                        current_segments[si + 1 :]
                                    )
                                    if next_segment is not None:
                                        hal_next_start = next_segment["words"][0][
                                            "start"
                                        ]
                                    else:
                                        hal_next_start = time_offset + segment_duration
                                    silence_before = (
                                        segment["start"] - hal_last_end > threshold
                                        or segment["start"] < threshold
                                        or segment["start"] - time_offset < 2.0
                                    )
                                    silence_after = (
                                        hal_next_start - segment["end"] > threshold
                                        or is_segment_anomaly(next_segment)
                                        or window_end_time - segment["end"] < 2.0
                                    )
                                    if silence_before and silence_after:
                                        seek = round(
                                            max(time_offset + 1, segment["start"])
                                            * FRAMES_PER_SECOND
                                        )
                                        if (
                                            content_duration - segment["end"]
                                            < threshold
                                        ):
                                            seek = content_frames
                                        current_segments[si:] = []
                                        break
                                hal_last_end = segment["end"]

                        last_word_end = _get_end(current_segments)
                        if last_word_end is not None:
                            last_speech_timestamp = last_word_end

                    if verbose:
                        for segment in current_segments:
                            start, end, text = (
                                segment["start"],
                                segment["end"],
                                segment["text"],
                            )
                            line = f"[{_format_timestamp(start)} --> {_format_timestamp(end)}] {text}"
                            print(make_safe(line))

                    # if a segment is instantaneous or does not contain text, clear it
                    for i, segment in enumerate(current_segments):
                        if (
                            segment["start"] == segment["end"]
                            or segment["text"].strip() == ""
                        ):
                            segment["text"] = ""
                            segment["tokens"] = []
                            segment["words"] = []

                    all_segments.extend(
                        [
                            {"id": i, **segment}
                            for i, segment in enumerate(
                                current_segments, start=len(all_segments)
                            )
                        ]
                    )
                    all_tokens.extend(
                        [
                            token
                            for segment in current_segments
                            for token in segment["tokens"]
                        ]
                    )

                    if not condition_on_previous_text or result.temperature > 0.5:
                        # do not feed the prompt tokens if a high temperature was used
                        prompt_reset_since = len(all_tokens)

                    # update progress bar
                    pbar.update(min(content_frames, seek) - previous_seek)

        # Clear cache after each segment to avoid memory leaks
        mx.clear_cache()

        if verbose:
            print("\n\033[94mSegments:\033[0m")
            if hasattr(all_segments, "segments"):
                print(all_segments)
            elif hasattr(all_segments, "tokens"):
                print(all_segments.tokens)
            else:
                print(all_segments)

        return STTOutput(
            text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
            segments=all_segments,
            language=language,
        )

    def generate_streaming(
        self,
        audio,
        *,
        chunk_duration: float = 1.0,
        language: str = None,
        task: str = "transcribe",
        frame_threshold: int = 25,
    ):
        """Generate transcription with streaming output using AlignAtt.

        This method yields partial transcriptions as audio is processed,
        achieving ~1 second latency instead of waiting for full 30-second windows.

        Args:
            audio: Path to audio file or audio waveform array.
            chunk_duration: Duration of each chunk in seconds (default 1.0).
            language: Language code (e.g., "en"). Auto-detected if None.
            task: "transcribe" or "translate" (to English).
            frame_threshold: AlignAtt threshold (lower = faster, higher = more accurate).

        Yields:
            StreamingResult objects with partial/final transcriptions.

        Example:
            >>> for result in model.generate_streaming("audio.wav"):
            ...     print(f"[{'FINAL' if result.is_final else 'partial'}] {result.text}")
        """
        from mlx_audio.stt.models.whisper.streaming import (
            StreamingConfig,
            StreamingDecoder,
        )
        from mlx_audio.stt.utils import load_audio

        # Load audio if path provided
        if isinstance(audio, str):
            audio = load_audio(audio)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        # Auto-detect language if not specified
        if language is None:
            # Get mel for first chunk to detect language
            first_chunk = audio[: int(chunk_duration * SAMPLE_RATE)]
            mel_detect = log_mel_spectrogram(
                np.array(first_chunk), n_mels=self.dims.n_mels
            )
            language = self._detect_language(mel_detect)

        # Configure streaming
        config = StreamingConfig(frame_threshold=frame_threshold)
        decoder = StreamingDecoder(self, config, language=language, task=task)

        # Process audio in chunks
        chunk_samples = int(chunk_duration * SAMPLE_RATE)
        total_samples = len(audio)
        audio_duration = total_samples / SAMPLE_RATE

        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]
            is_last = end >= total_samples

            mel = log_mel_spectrogram(np.array(chunk), n_mels=self.dims.n_mels)
            result = decoder.decode_chunk(mel, is_last=is_last)

            # Add progress info
            result.progress = end / total_samples
            result.audio_position = end / SAMPLE_RATE
            result.audio_duration = audio_duration

            if result.text.strip() or is_last:
                yield result

            if is_last:
                break
