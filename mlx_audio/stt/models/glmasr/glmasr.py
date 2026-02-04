"""GLM-ASR model for speech-to-text transcription using MLX."""

import glob
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

from mlx_audio.stt.generate import wired_limit
from mlx_audio.stt.utils import get_model_path

from ..base import STTOutput
from .config import LlamaConfig, ModelConfig, WhisperConfig


def split_audio_into_chunks(
    wav: np.ndarray,
    sr: int,
    chunk_duration: float = 30.0,
    min_chunk_duration: float = 1.0,
    search_expand_sec: float = 2.0,
    min_window_ms: float = 100.0,
) -> List[Tuple[np.ndarray, float]]:
    """Split long audio into chunks at low-energy boundaries.

    Args:
        wav: Audio waveform (1D numpy array).
        sr: Sample rate.
        chunk_duration: Maximum chunk duration in seconds (default: 30).
        min_chunk_duration: Minimum chunk duration in seconds (default: 1.0).
        search_expand_sec: Window to search for silence around cut point.
        min_window_ms: Minimum window size for energy calculation.

    Returns:
        List of (chunk_waveform, offset_seconds) tuples.
    """
    # Ensure mono
    if wav.ndim > 1:
        wav = wav.mean(axis=-1) if wav.shape[-1] <= 2 else wav.mean(axis=0)

    total_samples = len(wav)
    total_sec = total_samples / sr

    # If short enough, return as-is
    if total_sec <= chunk_duration:
        # Pad if too short
        if total_sec < min_chunk_duration:
            min_samples = int(min_chunk_duration * sr)
            wav = np.pad(wav, (0, min_samples - len(wav)))
        return [(wav, 0.0)]

    chunks = []
    start_sample = 0
    max_chunk_samples = int(chunk_duration * sr)
    search_samples = int(search_expand_sec * sr)
    min_window_samples = int(min_window_ms * sr / 1000)

    while start_sample < total_samples:
        end_sample = min(start_sample + max_chunk_samples, total_samples)

        # If this is the last chunk, take the rest
        if end_sample >= total_samples:
            chunk = wav[start_sample:total_samples]
            offset_sec = start_sample / sr
            # Pad if too short
            if len(chunk) < min_chunk_duration * sr:
                min_samples = int(min_chunk_duration * sr)
                chunk = np.pad(chunk, (0, min_samples - len(chunk)))
            chunks.append((chunk, offset_sec))
            break

        # Search for low-energy point around the cut
        search_start = max(start_sample, end_sample - search_samples)
        search_end = min(total_samples, end_sample + search_samples)
        search_region = wav[search_start:search_end]

        # Calculate energy using sliding window
        if len(search_region) > min_window_samples:
            energy = np.convolve(
                search_region**2,
                np.ones(min_window_samples) / min_window_samples,
                mode="valid",
            )
            # Find minimum energy point
            min_idx = np.argmin(energy) + min_window_samples // 2
            cut_sample = search_start + min_idx
        else:
            cut_sample = end_sample

        # Ensure we make progress
        cut_sample = max(cut_sample, start_sample + sr)  # At least 1 second

        chunk = wav[start_sample:cut_sample]
        offset_sec = start_sample / sr

        # Pad if too short
        if len(chunk) < min_chunk_duration * sr:
            min_samples = int(min_chunk_duration * sr)
            chunk = np.pad(chunk, (0, min_samples - len(chunk)))

        chunks.append((chunk, offset_sec))
        start_sample = cut_sample

    return chunks


@dataclass
class StreamingResult:
    """Result object for streaming transcription.

    Attributes:
        text: Decoded text for this emission.
        is_final: True if this is a final (committed) result, False if partial.
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        language: Language of the transcription.
        prompt_tokens: Total prompt tokens (only set on final result).
        generation_tokens: Total generation tokens (only set on final result).
    """

    text: str
    is_final: bool
    start_time: float
    end_time: float
    language: str = "en"
    prompt_tokens: int = 0
    generation_tokens: int = 0


class WhisperAttention(nn.Module):
    """Whisper attention layer with optional Rotary Position Embeddings."""

    def __init__(self, config: WhisperConfig, use_rope: bool = False):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.use_rope = use_rope

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        if use_rope:
            self.rope = nn.RoPE(self.head_dim // 2, traditional=config.rope_traditional)

    def __call__(
        self,
        hidden_states: mx.array,
    ) -> mx.array:
        bsz, tgt_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if self.use_rope:
            query_states = self.rope(query_states)
            key_states = self.rope(key_states)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=self.scaling
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, tgt_len, self.embed_dim
        )

        return self.out_proj(attn_output)


class WhisperEncoderLayer(nn.Module):
    """Whisper encoder layer with optional RoPE support."""

    def __init__(self, config: WhisperConfig, use_rope: bool = False):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(config, use_rope=use_rope)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        hidden_states: mx.array,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nn.Module):
    """Whisper encoder with optional rotary position embeddings."""

    def __init__(self, config: WhisperConfig, use_rope: bool = False):
        super().__init__()
        self.config = config
        self.use_rope = use_rope
        embed_dim = config.d_model

        self.conv1 = nn.Conv1d(config.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        # Always create for weight loading compatibility (only used when not using RoPE)
        self.embed_positions = nn.Embedding(config.max_source_positions, embed_dim)

        self.layers = [
            WhisperEncoderLayer(config, use_rope=use_rope)
            for _ in range(config.encoder_layers)
        ]

    def __call__(self, input_features: mx.array) -> mx.array:
        """Encode audio features."""
        hidden_states = nn.gelu(self.conv1(input_features))
        hidden_states = nn.gelu(self.conv2(hidden_states))

        # Add position embeddings if not using RoPE
        if not self.use_rope:
            seq_len = hidden_states.shape[1]
            embed_pos = self.embed_positions.weight[:seq_len]
            hidden_states = hidden_states + embed_pos

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class AdaptingMLP(nn.Module):
    """MLP adapter for audio-to-LM projection."""

    def __init__(self, input_dim: int, intermediate_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim, bias=True)
        self.fc2 = nn.Linear(intermediate_dim, output_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class AudioEncoder(nn.Module):
    """Audio encoder with Whisper backbone and MLP adapter.

    This matches the HuggingFace weight structure exactly:
    - audio_encoder.whisper.*
    - audio_encoder.layer_norm.*
    - audio_encoder.proj.*
    - audio_encoder.adapting.*
    - audio_encoder.audio_bos_eos_token.*
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        whisper_config = config.whisper_config
        lm_hidden_size = config.lm_config.hidden_size

        # Whisper encoder
        self.whisper = WhisperEncoder(whisper_config, use_rope=config.use_rope)

        # Layer norm after whisper encoder
        self.layer_norm = nn.LayerNorm(whisper_config.d_model)

        # Projection from whisper dim to intermediate
        # HF model: proj goes from 1280 -> 2048
        self.proj = nn.Linear(whisper_config.d_model, lm_hidden_size, bias=True)

        # MLP adapter: matches HF structure with layers 0 and 2
        # Layer 0: 5120 -> 4096 (merge_factor * whisper_dim -> intermediate)
        # Layer 2: 4096 -> 2048 (intermediate -> lm_hidden)
        merged_dim = whisper_config.d_model * config.merge_factor
        intermediate_dim = lm_hidden_size * 2  # 4096 for this model

        # Use a custom module to match HF weight naming (adapting.0.*, adapting.2.*)
        self.adapting = AdaptingMLP(merged_dim, intermediate_dim, lm_hidden_size)

        # Begin/End of audio token embeddings
        self.audio_bos_eos_token = nn.Embedding(2, lm_hidden_size)

    def __call__(self, input_features: mx.array) -> Tuple[mx.array, int]:
        """Encode audio features and project to LM space."""
        # Whisper encoding
        audio_features = self.whisper(input_features)

        # Layer norm
        audio_features = self.layer_norm(audio_features)

        # Merge audio features by merge_factor
        batch_size, seq_len, _ = audio_features.shape
        merge_factor = self.config.merge_factor

        new_seq_len = (seq_len - merge_factor) // merge_factor + 1
        max_len = self.config.max_whisper_length // merge_factor
        new_seq_len = min(new_seq_len, max_len)

        merged_features = []
        for i in range(new_seq_len):
            start_idx = i * merge_factor
            end_idx = start_idx + merge_factor
            chunk = audio_features[:, start_idx:end_idx, :]
            chunk = chunk.reshape(batch_size, -1)
            merged_features.append(chunk)

        merged_audio = mx.stack(merged_features, axis=1)

        # Project through MLP adapter
        audio_embeds = self.adapting(merged_audio)

        return audio_embeds, new_seq_len

    def get_boa_eoa_tokens(self) -> Tuple[mx.array, mx.array]:
        """Get begin-of-audio and end-of-audio token embeddings."""
        boa = self.audio_bos_eos_token(mx.array([0]))
        eoa = self.audio_bos_eos_token(mx.array([1]))
        return boa, eoa


class LanguageModel(nn.Module):
    """Language model wrapper using mlx_lm's LlamaModel."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        from mlx_lm.models.llama import LlamaModel

        self.model = LlamaModel(config)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.config.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def embed_tokens(self):
        return self.model.embed_tokens


class Model(nn.Module):
    """GLM-ASR model combining Whisper encoder with LLaMA decoder.

    Weight structure matches HuggingFace format:
    - audio_encoder.* : Audio encoder with Whisper + MLP adapter
    - model.* / language_model.model.* : LLaMA decoder
    - lm_head.* / language_model.lm_head.* : Language modeling head
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.lm_config.vocab_size

        # Audio encoder (matches HF naming: audio_encoder.*)
        self.audio_encoder = AudioEncoder(config)

        # Language model with LlamaModel backbone
        self.language_model = LanguageModel(config.lm_config)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embeddings from the language model."""
        return self.language_model.embed_tokens

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def _merge_audio_text_embeddings(
        self,
        input_ids: mx.array,
        audio_embeds: Optional[mx.array] = None,
        audio_offsets: Optional[List[List[int]]] = None,
        audio_length: Optional[List[List[int]]] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """Merge pre-computed audio embeddings into text embeddings."""
        text_embeds = self.get_input_embeddings()(input_ids)

        # Skip if no audio or cache already populated
        if audio_embeds is None or (cache is not None and cache[0].offset > 0):
            return text_embeds

        batch_size = text_embeds.shape[0]

        for b in range(batch_size):
            if audio_offsets is not None and len(audio_offsets) > b:
                offsets = audio_offsets[b]
                lengths = audio_length[b] if audio_length else [audio_embeds.shape[1]]

                audio_idx = 0
                for offset, length in zip(offsets, lengths):
                    if audio_idx < audio_embeds.shape[0]:
                        audio_chunk = audio_embeds[audio_idx, :length]
                        end_pos = min(offset + length, text_embeds.shape[1])
                        actual_length = end_pos - offset
                        text_embeds[b, offset:end_pos] = audio_chunk[:actual_length]
                        audio_idx += 1

        return text_embeds

    def __call__(
        self,
        input_ids: mx.array,
        audios: Optional[mx.array] = None,
        audio_embeds: Optional[mx.array] = None,
        audio_offsets: Optional[List[List[int]]] = None,
        audio_length: Optional[List[List[int]]] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """Forward pass."""
        # Compute audio embeddings if raw audio provided and no pre-computed embeds
        if audios is not None and audio_embeds is None:
            audio_embeds, _ = self.audio_encoder(audios)

        input_embeds = self._merge_audio_text_embeddings(
            input_ids=input_ids,
            audio_embeds=audio_embeds,
            audio_offsets=audio_offsets,
            audio_length=audio_length,
            cache=cache,
        )

        logits = self.language_model(input_embeddings=input_embeds, cache=cache)

        return logits

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights for loading."""
        sanitized = {}
        for k, v in weights.items():
            new_key = k

            # Remap adapting layer names: 0 -> fc1, 2 -> fc2
            if "audio_encoder.adapting.0." in k:
                new_key = k.replace(
                    "audio_encoder.adapting.0.", "audio_encoder.adapting.fc1."
                )
            elif "audio_encoder.adapting.2." in k:
                new_key = k.replace(
                    "audio_encoder.adapting.2.", "audio_encoder.adapting.fc2."
                )

            # Remap model.* -> language_model.model.* for LanguageModel wrapper
            if new_key.startswith("model."):
                new_key = "language_model." + new_key

            # Remap lm_head.* -> language_model.lm_head.*
            if new_key.startswith("lm_head."):
                new_key = "language_model." + new_key

            # Handle conv weight transposition
            if "conv" in new_key and "weight" in new_key:
                if v.ndim == 3 and v.shape[-1] < v.shape[-2]:
                    sanitized[new_key] = v.transpose(0, 2, 1)
                else:
                    sanitized[new_key] = v
            else:
                sanitized[new_key] = v
        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """
        Hook called after model weights are loaded.
        Used to initialize the tokenizer which is required for text input.
        """
        from transformers import AutoTokenizer

        if not hasattr(model, "_tokenizer") or model._tokenizer is None:
            model._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        **kwargs,
    ) -> "Model":
        """
        Load model from pretrained weights.

        .. deprecated::
            Use `mlx_audio.stt.load()` instead. This method will be removed in a future version.
        """
        warnings.warn(
            "Model.from_pretrained() is deprecated. Use mlx_audio.stt.load() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from mlx_audio.stt.utils import load

        return load(model_path)

    def _preprocess_audio(self, audio) -> mx.array:
        """Preprocess audio to mel spectrogram.

        Args:
            audio: Audio path (str), waveform (np.ndarray/mx.array), or mel spectrogram

        Returns:
            Mel spectrogram of shape (batch, seq_len, n_mels)
        """
        from mlx_audio.stt.utils import load_audio
        from mlx_audio.utils import hanning, mel_filters, stft

        # Audio hyperparameters for GLM-ASR (128 mel bins)
        N_FFT = 400
        HOP_LENGTH = 160
        N_MELS = self.config.whisper_config.num_mel_bins  # 128

        # Load audio if path
        if isinstance(audio, str):
            audio = load_audio(audio, sr=self.sample_rate)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        # If already 3D (batch, seq, mels), assume it's mel spectrogram
        if audio.ndim == 3:
            return audio

        # Compute mel spectrogram
        window = hanning(N_FFT)
        freqs = stft(audio, window=window, n_fft=N_FFT, hop_length=HOP_LENGTH)
        magnitudes = freqs[:-1, :].abs().square()

        filters = mel_filters(
            self.sample_rate, N_FFT, N_MELS, norm="slaney", mel_scale=None
        )
        mel_spec = magnitudes @ filters.T

        log_spec = mx.maximum(mel_spec, 1e-10).log10()
        log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # Add batch dimension: (seq_len, n_mels) -> (1, seq_len, n_mels)
        return log_spec[None]

    def stream_generate(
        self,
        input_ids: Optional[mx.array] = None,
        *,
        audio_embeds: Optional[mx.array] = None,
        audio_offsets: Optional[List[List[int]]] = None,
        audio_length: Optional[List[List[int]]] = None,
        max_tokens: int = 128,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        generation_stream: Optional[mx.Stream] = None,
        verbose: bool = False,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        """Stream generate tokens from input.

        Args:
            input_ids: Input token IDs
            audio_embeds: Pre-computed audio embeddings
            audio_offsets: Positions to insert audio embeddings
            audio_length: Lengths of audio embeddings
            max_tokens: Maximum tokens to generate
            sampler: Sampler function for token selection
            generation_stream: Optional mx.Stream for async generation

        Yields:
            Tuple of (token, logprobs)
        """
        from mlx_lm.generate import generate_step

        input_embeddings = self._merge_audio_text_embeddings(
            input_ids=input_ids,
            audio_embeds=audio_embeds,
            audio_offsets=audio_offsets,
            audio_length=audio_length,
        )[
            0
        ]  # Remove batch dimension for generate_step

        streams = [generation_stream] if generation_stream is not None else None
        with wired_limit(self, streams):
            prompt = input_ids[0] if input_ids.ndim > 1 else input_ids
            for token, logprobs in tqdm(
                generate_step(
                    prompt=prompt,
                    input_embeddings=input_embeddings,
                    model=self.language_model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ),
                total=max_tokens,
                disable=not verbose,
                desc="Streaming",
            ):
                if token in self.config.lm_config.eos_token_id:
                    break

                yield token, logprobs

    def _generate_single_chunk(
        self,
        audio_chunk: np.ndarray,
        *,
        max_tokens: int = 128,
        sampler: Optional[Callable] = None,
        generation_stream: Optional[mx.Stream] = None,
        verbose: bool = False,
    ) -> Tuple[str, int, int]:
        """Generate transcription for a single audio chunk.

        Returns:
            Tuple of (text, prompt_tokens, generation_tokens).
        """
        # Preprocess chunk to mel spectrogram
        mel = self._preprocess_audio(audio_chunk)

        # Encode audio
        audio_embeds, audio_len = self.audio_encoder(mel)
        mx.eval(audio_embeds)

        prompt_text = "<|user|>\n<|begin_of_audio|>"
        tokens = self._tokenizer.encode(prompt_text)

        audio_placeholder_tokens = [0] * audio_len
        tokens.extend(audio_placeholder_tokens)

        end_prompt = (
            "<|end_of_audio|>\nPlease transcribe this audio into text<|assistant|>\n"
        )
        tokens.extend(self._tokenizer.encode(end_prompt))

        input_ids = mx.array([tokens])

        audio_start = len(self._tokenizer.encode("<|user|>\n<|begin_of_audio|>"))
        audio_offsets = [[audio_start]]
        audio_length = [[audio_len]]

        generated_tokens = []

        for token, _ in self.stream_generate(
            input_ids=input_ids,
            audio_embeds=audio_embeds,
            audio_offsets=audio_offsets,
            audio_length=audio_length,
            max_tokens=max_tokens,
            sampler=sampler,
            generation_stream=generation_stream,
            verbose=verbose,
        ):
            generated_tokens.append(token)

        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text, input_ids.shape[1], len(generated_tokens)

    def generate(
        self,
        audio,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        generation_stream: Optional[mx.Stream] = None,
        verbose: bool = False,
        stream: bool = False,
        chunk_duration: float = 30.0,
        min_chunk_duration: float = 1.0,
        **kwargs,
    ) -> Union[STTOutput, Generator[StreamingResult, None, None]]:
        """Generate transcription from audio.

        Automatically chunks long audio and processes sequentially.

        Args:
            audio: Audio path (str), waveform (mx.array), or mel spectrogram
            max_tokens: Maximum tokens to generate per chunk
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            min_p: Minimum probability threshold
            min_tokens_to_keep: Minimum tokens to keep in sampling
            generation_stream: Optional mx.Stream for async generation
            verbose: Print tokens during generation
            stream: If True, return a generator that yields StreamingResult objects
            chunk_duration: Maximum chunk duration in seconds (default: 30)
            min_chunk_duration: Minimum chunk duration in seconds (default: 1.0)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            STTOutput with transcription text, or Generator[StreamingResult] if stream=True
        """
        from mlx_audio.stt.utils import load_audio

        if stream:
            return self.stream_transcribe(
                audio,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                min_tokens_to_keep=min_tokens_to_keep,
                chunk_duration=chunk_duration,
                min_chunk_duration=min_chunk_duration,
                verbose=verbose,
            )

        from mlx_lm.sample_utils import make_sampler

        start_time = time.time()

        if isinstance(audio, str):
            audio = load_audio(audio, sr=self.sample_rate)
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        chunks = split_audio_into_chunks(
            audio,
            sr=self.sample_rate,
            chunk_duration=chunk_duration,
            min_chunk_duration=min_chunk_duration,
        )

        sampler = make_sampler(
            temperature,
            top_p,
            min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k,
        )

        # Process chunks
        all_texts = []
        segments = []
        total_prompt_tokens = 0
        total_generation_tokens = 0
        remaining_tokens = max_tokens

        chunk_iter = tqdm(
            chunks, desc="Processing chunks", disable=not verbose or len(chunks) == 1
        )
        for chunk_audio, offset_sec in chunk_iter:
            # Stop if we've exhausted the token budget
            if remaining_tokens <= 0:
                break

            actual_chunk_duration = len(chunk_audio) / self.sample_rate

            text, prompt_toks, gen_toks = self._generate_single_chunk(
                chunk_audio,
                max_tokens=remaining_tokens,
                sampler=sampler,
                generation_stream=generation_stream,
                verbose=verbose and len(chunks) == 1,
            )
            all_texts.append(text)
            total_prompt_tokens += prompt_toks
            total_generation_tokens += gen_toks
            remaining_tokens -= gen_toks

            # Create segment for this chunk
            segments.append(
                {
                    "text": text,
                    "start": offset_sec,
                    "end": offset_sec + actual_chunk_duration,
                }
            )

            # Clear cache between chunks
            mx.clear_cache()

        end_time = time.time()

        if verbose:
            print()

        # Combine transcriptions
        full_text = " ".join(all_texts)

        return STTOutput(
            text=full_text.strip(),
            segments=segments,
            prompt_tokens=total_prompt_tokens,
            generation_tokens=total_generation_tokens,
            total_tokens=total_prompt_tokens + total_generation_tokens,
            total_time=end_time - start_time,
            prompt_tps=(
                total_prompt_tokens / (end_time - start_time)
                if end_time > start_time
                else 0
            ),
            generation_tps=(
                total_generation_tokens / (end_time - start_time)
                if end_time > start_time
                else 0
            ),
        )

    def _generate_chunk_stream(
        self,
        audio_chunk: np.ndarray,
        offset_sec: float,
        chunk_duration: float,
        *,
        remaining_tokens: int = 128,
        sampler: Optional[Callable] = None,
        verbose: bool = False,
        is_last_chunk: bool = False,
        total_prompt_tokens: int = 0,
        total_generation_tokens: int = 0,
    ) -> Generator[StreamingResult, None, None]:
        """Generate streaming results for a single audio chunk.

        Args:
            remaining_tokens: How many tokens can still be generated (global budget).

        Yields:
            StreamingResult objects for this chunk. The chunk-final result
            includes token counts (prompt_tokens, generation_tokens).
        """
        # Preprocess chunk to mel spectrogram
        mel = self._preprocess_audio(audio_chunk)

        # Encode audio
        audio_embeds, audio_len = self.audio_encoder(mel)
        mx.eval(audio_embeds)

        prompt_text = "<|user|>\n<|begin_of_audio|>"
        tokens = self._tokenizer.encode(prompt_text)

        audio_placeholder_tokens = [0] * audio_len
        tokens.extend(audio_placeholder_tokens)

        end_prompt = (
            "<|end_of_audio|>\nPlease transcribe this audio into text<|assistant|>\n"
        )
        tokens.extend(self._tokenizer.encode(end_prompt))

        input_ids = mx.array([tokens])
        chunk_prompt_tokens = input_ids.shape[1]

        audio_start = len(self._tokenizer.encode("<|user|>\n<|begin_of_audio|>"))
        audio_offsets = [[audio_start]]
        audio_length = [[audio_len]]

        token_count = 0

        for token, _ in self.stream_generate(
            input_ids=input_ids,
            audio_embeds=audio_embeds,
            audio_offsets=audio_offsets,
            audio_length=audio_length,
            max_tokens=remaining_tokens,
            sampler=sampler,
            verbose=verbose,
        ):
            text = self._tokenizer.decode([int(token)])

            # Estimate timing based on token position within chunk
            prev_progress = token_count / max(remaining_tokens, 1)
            token_count += 1
            curr_progress = min(token_count / max(remaining_tokens, 1), 1.0)

            estimated_start = offset_sec + (chunk_duration * prev_progress)
            estimated_end = offset_sec + (chunk_duration * curr_progress)

            yield StreamingResult(
                text=text,
                is_final=False,
                start_time=estimated_start,
                end_time=estimated_end,
                language="en",
            )

        # Update totals with this chunk's counts
        total_prompt_tokens += chunk_prompt_tokens
        total_generation_tokens += token_count

        # Yield chunk-final result with token counts
        yield StreamingResult(
            text="",
            is_final=is_last_chunk,
            start_time=offset_sec,
            end_time=offset_sec + chunk_duration,
            language="en",
            prompt_tokens=total_prompt_tokens,
            generation_tokens=total_generation_tokens,
        )

        # Clear cache after chunk
        mx.clear_cache()

    def stream_transcribe(
        self,
        audio,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        chunk_duration: float = 30.0,
        min_chunk_duration: float = 1.0,
        verbose: bool = False,
    ) -> Generator[StreamingResult, None, None]:
        """Stream transcription token-by-token from audio.

        Automatically chunks long audio and streams tokens from each chunk sequentially.

        Args:
            audio: Audio path (str), waveform (mx.array/np.ndarray), or mel spectrogram
            max_tokens: Maximum tokens to generate per chunk
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            min_p: Minimum probability threshold
            min_tokens_to_keep: Minimum tokens to keep in sampling
            chunk_duration: Maximum chunk duration in seconds (default: 30)
            min_chunk_duration: Minimum chunk duration in seconds (default: 1.0)
            verbose: Print progress during generation

        Yields:
            StreamingResult objects with text, timing, and status information.
        """
        from mlx_lm.sample_utils import make_sampler

        from mlx_audio.stt.utils import load_audio

        # Load audio
        if isinstance(audio, str):
            audio = load_audio(audio, sr=self.sample_rate)
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Split into chunks
        chunks = split_audio_into_chunks(
            audio,
            sr=self.sample_rate,
            chunk_duration=chunk_duration,
            min_chunk_duration=min_chunk_duration,
        )

        sampler = make_sampler(
            temperature,
            top_p,
            min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k,
        )

        # Track token counts across chunks
        total_prompt_tokens = 0
        total_generation_tokens = 0
        remaining_tokens = max_tokens

        # Process each chunk
        chunk_iter = tqdm(
            enumerate(chunks),
            total=len(chunks),
            desc="Processing chunks",
            disable=not verbose or len(chunks) == 1,
        )

        for chunk_idx, (chunk_audio, offset_sec) in chunk_iter:

            actual_chunk_duration = len(chunk_audio) / self.sample_rate
            is_last_chunk = chunk_idx == len(chunks) - 1
            prev_gen_tokens = total_generation_tokens

            for result in self._generate_chunk_stream(
                chunk_audio,
                offset_sec,
                actual_chunk_duration,
                remaining_tokens=remaining_tokens,
                sampler=sampler,
                verbose=verbose and len(chunks) == 1,
                is_last_chunk=is_last_chunk,
                total_prompt_tokens=total_prompt_tokens,
                total_generation_tokens=total_generation_tokens,
            ):
                # Track cumulative token counts from chunk-final results
                if result.prompt_tokens > 0 or result.generation_tokens > 0:
                    total_prompt_tokens = result.prompt_tokens
                    total_generation_tokens = result.generation_tokens

                yield result

            chunk_gen_tokens = total_generation_tokens - prev_gen_tokens
            remaining_tokens -= chunk_gen_tokens

            if remaining_tokens <= 0:
                # Yield final result
                yield StreamingResult(
                    text="",
                    is_final=True,
                    start_time=offset_sec,
                    end_time=offset_sec,
                    language="en",
                    prompt_tokens=total_prompt_tokens,
                    generation_tokens=total_generation_tokens,
                )
                break
