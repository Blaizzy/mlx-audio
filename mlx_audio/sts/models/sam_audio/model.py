# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_reduce
from tqdm import tqdm

from mlx_audio.codec.models.dacvae import DACVAE

from .align import EmbedAnchors
from .config import SAMAudioConfig
from .processor import Batch, SAMAudioProcessor
from .text_encoder import T5TextEncoder
from .transformer import DiT


@contextlib.contextmanager
def wired_limit(model: nn.Module):
    """
    Context manager to set optimal wired memory limit during inference.

    This helps prevent memory pressure by setting the wired limit to the
    maximum recommended working set size for the Metal device.
    """
    if not mx.metal.is_available():
        yield
        return

    # Calculate model size
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]

    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Model requires {model_mb} MB which is close to the "
            f"maximum recommended size of {max_rec_mb} MB. Processing may be slow."
        )

    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield
    finally:
        mx.synchronize()
        mx.set_wired_limit(old_limit)


def _fallback(value, default):
    return default if value is None else value


# Default ODE solver options
DFLT_ODE_OPT = {"method": "midpoint", "step_size": 2 / 32}


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int, theta: float = 10000):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        inv_freq = mx.exp(
            -math.log(theta) * mx.arange(half_dim, dtype=mx.float32) / half_dim
        )
        self._inv_freq = inv_freq

    def __call__(self, x: mx.array, pos: Optional[mx.array] = None) -> mx.array:
        if pos is None:
            seq_len = x.shape[1]
            pos = mx.arange(seq_len, dtype=mx.float32)

        # Compute sinusoidal embeddings
        emb = pos[:, None] * self._inv_freq[None, :]
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
        return emb


@dataclass
class SeparationResult:
    """Result of audio separation (supports both batch and streaming modes).

    In batch mode (separate/separate_long):
        - target/residual: List[mx.array] with all separated audio
        - noise: mx.array with initial noise
        - peak_memory: float with peak memory usage
        - chunk_idx/is_last: None

    In streaming mode (separate_streaming generator):
        - target/residual: mx.array with the current chunk
        - noise: mx.array on last chunk only, None otherwise
        - peak_memory: float on last chunk only, None otherwise
        - chunk_idx: int index of this chunk
        - is_last: bool True if this is the final chunk
    """

    target: Union[List[mx.array], mx.array]  # Separated target audio(s) or chunk
    residual: Union[List[mx.array], mx.array]  # Residual/background audio(s) or chunk
    noise: Optional[mx.array] = None  # Initial noise used for generation
    peak_memory: Optional[float] = None  # Peak memory usage in GB

    # Streaming-specific fields (None for batch mode)
    chunk_idx: Optional[int] = None  # Index of this chunk
    is_last: Optional[bool] = None  # True if this is the final chunk


class SAMAudio(nn.Module):
    """
    SAM-Audio: Segment Anything Model for Audio.

    A foundation model for audio source separation using text or temporal prompts.
    Uses ODE-based diffusion for high-quality audio separation.
    """

    def __init__(self, config: SAMAudioConfig):
        super().__init__()
        self.config = config

        # Audio codec (DACVAE)
        self.audio_codec = DACVAE(config.audio_codec)

        # Text encoder (T5)
        self.text_encoder = T5TextEncoder(config.text_encoder)

        # Diffusion transformer
        self.transformer = DiT(config.transformer)

        # Input projection
        self.proj = nn.Linear(config.in_channels, config.transformer.dim)

        # Anchor embeddings for temporal prompts
        self.embed_anchors = EmbedAnchors(
            config.num_anchors,
            config.anchor_embedding_dim,
            config.transformer.dim,
        )

        # Memory projection for text features
        self.memory_proj = nn.Linear(config.text_encoder.dim, config.transformer.dim)

        # Timestep embedding
        self.timestep_emb = SinusoidalEmbedding(config.transformer.dim)

        self.dtype = self.proj.weight.dtype

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self.audio_codec.sample_rate

    def _prepare_inputs(
        self,
        audios: Union[mx.array, List[str]],
        descriptions: List[str] = None,
        anchors: Optional[List[List[Tuple[str, float, float]]]] = None,
    ) -> Batch:
        """
        Prepare audio and anchor inputs, handling both mx.array and file paths.

        Args:
            audios: Either an mx.array (B, 1, T) or list of audio file paths
            descriptions: Text descriptions (needed for batch size)
            anchors: Optional temporal anchors [[("+", start, end), ...], ...]
                     Note: anchors are only processed when audios is a list of file paths.
                     When audios is an mx.array, pass anchor_ids and anchor_alignment directly.

        Returns:
            Batch with audio tensor and optional anchor data
        """
        if isinstance(audios, mx.array):
            return Batch(audios=audios, descriptions=descriptions)

        if isinstance(audios, list) and len(audios) > 0 and isinstance(audios[0], str):
            batch = self.processor(
                descriptions=descriptions,
                audios=audios,
                anchors=anchors,
            )
            return batch

        raise TypeError(f"audios must be mx.array or List[str], got {type(audios)}")

    def post_load_hook(self, model_path: Path) -> "SAMAudio":
        """
        Post-load hook called by load_model to initialize tokenizer and conditionals.
        """
        # Initialize processor for anchor handling
        if not hasattr(self, "processor"):
            self.processor = SAMAudioProcessor.from_pretrained(model_path)
        return self

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitize PyTorch weights for MLX loading.

        Handles:
        - Removing keys not needed (text_encoder, span_predictor, etc.)
        - Combining LSTM biases (bias_ih + bias_hh -> bias)
        - Converting weight names to MLX conventions
        - Transposing weights for Conv/Linear layers

        Note: text_encoder weights are NOT in the SAM-Audio checkpoint.
              T5 is loaded separately from HuggingFace.
        """
        import re

        # Keys to remove
        keys_to_remove = {
            k
            for k in weights
            if k.startswith(
                (
                    "text_encoder.",
                    "span_predictor.",
                    "visual_ranker.",
                    "text_ranker.",
                    "vision_encoder.",
                    "align_masked_video.",
                )
            )
            or "wm_rates" in k
        }

        # Combine LSTM biases
        lstm_biases = {}
        for key in list(weights.keys()):
            match = re.search(r"(.+\.lstm)\.bias_(ih|hh)_l(\d+)$", key)
            if match:
                base, bias_type, layer_idx = match.groups()
                lstm_biases.setdefault((base, layer_idx), {})[bias_type] = weights[key]
                keys_to_remove.add(key)

        # Build sanitized weights
        sanitized = {}

        # Add combined LSTM biases
        for (base, idx), biases in lstm_biases.items():
            if "ih" in biases and "hh" in biases:
                new_key = _convert_weight_name(f"{base}.combined_bias_l{idx}")
                sanitized[new_key] = biases["ih"] + biases["hh"]

        # Process remaining weights (no transpose here - done during load with target shape info)
        for key, value in weights.items():
            if key in keys_to_remove:
                continue
            sanitized[_convert_weight_name(key)] = value

        return sanitized

    def align_inputs(
        self,
        noisy_audio: mx.array,
        audio_features: mx.array,
        anchor_ids: Optional[mx.array] = None,
        anchor_alignment: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Align and project inputs for the transformer.

        Args:
            noisy_audio: Noisy audio being denoised (B, T, C)
            audio_features: Clean audio features (B, T, C)
            anchor_ids: Anchor token IDs (B, num_anchors)
            anchor_alignment: Timestep to anchor mapping (B, T)

        Returns:
            Aligned and projected features (B, T, dim)
        """
        # Concatenate noisy audio, zeros, and audio features
        x = mx.concatenate(
            [
                noisy_audio,
                mx.zeros_like(audio_features),
                audio_features,
            ],
            axis=2,
        )

        # Project to transformer dimension
        projected = self.proj(x)

        # Apply anchor embeddings if provided
        aligned = self.embed_anchors(projected, anchor_ids, anchor_alignment)

        return aligned

    def __call__(
        self,
        noisy_audio: mx.array,
        audio_features: mx.array,
        text_features: mx.array,
        time: mx.array,
        text_mask: Optional[mx.array] = None,
        anchor_ids: Optional[mx.array] = None,
        anchor_alignment: Optional[mx.array] = None,
        audio_pad_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass - one step of the ODE.

        Args:
            noisy_audio: Noisy audio tensor (B, T, C)
            audio_features: Clean audio features (B, T, C)
            text_features: Encoded text features (B, T_text, C_text)
            time: Timestep tensor (B,)
            text_mask: Padding mask for text (B, T_text)
            anchor_ids: Anchor token IDs (B, num_anchors)
            anchor_alignment: Timestep to anchor mapping (B, T)
            audio_pad_mask: Padding mask for audio (B, T)

        Returns:
            Predicted velocity field (B, T, C)
        """
        # Align inputs
        aligned_inputs = self.align_inputs(
            noisy_audio,
            audio_features,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
        )

        # Timestep embedding
        timestep_emb = self.timestep_emb(time, pos=time)
        timestep_emb = mx.expand_dims(timestep_emb, 1)

        # Prepare memory (text features + timestep)
        if text_features is not None:
            memory = self.memory_proj(text_features) + timestep_emb
        else:
            memory = timestep_emb

        # Pass through transformer
        return self.transformer(
            aligned_inputs,
            time,
            padding_mask=audio_pad_mask,
            memory=memory,
            memory_padding_mask=text_mask,
        )

    def _get_audio_features(self, audios: mx.array) -> mx.array:
        """Encode audio and prepare features."""
        audio_features = self.audio_codec(audios)
        # Transpose: (B, C, T) -> (B, T, C)
        audio_features = mx.transpose(audio_features, (0, 2, 1))
        # Duplicate for target/residual prediction
        return mx.concatenate([audio_features, audio_features], axis=2)

    def _ode_step_euler(
        self,
        t: float,
        dt: float,
        noisy_audio: mx.array,
        audio_features: mx.array,
        text_features: mx.array,
        text_mask: Optional[mx.array],
        anchor_ids: Optional[mx.array],
        anchor_alignment: Optional[mx.array],
        audio_pad_mask: Optional[mx.array],
    ) -> mx.array:
        """Euler ODE solver step (faster, 1 forward pass per step)."""
        batch_size = noisy_audio.shape[0]

        time_t = mx.full((batch_size,), t, dtype=mx.float32)
        v_t = self(
            noisy_audio=noisy_audio,
            audio_features=audio_features,
            text_features=text_features,
            time=time_t,
            text_mask=text_mask,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
            audio_pad_mask=audio_pad_mask,
        )

        return noisy_audio + dt * v_t

    def _ode_step_midpoint(
        self,
        t: float,
        dt: float,
        noisy_audio: mx.array,
        audio_features: mx.array,
        text_features: mx.array,
        text_mask: Optional[mx.array],
        anchor_ids: Optional[mx.array],
        anchor_alignment: Optional[mx.array],
        audio_pad_mask: Optional[mx.array],
    ) -> mx.array:
        """Midpoint ODE solver step (higher quality, 2 forward passes per step)."""
        batch_size = noisy_audio.shape[0]

        # Evaluate at t
        time_t = mx.full((batch_size,), t, dtype=mx.float32)
        v_t = self(
            noisy_audio=noisy_audio,
            audio_features=audio_features,
            text_features=text_features,
            time=time_t,
            text_mask=text_mask,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
            audio_pad_mask=audio_pad_mask,
        )

        # Midpoint
        midpoint = noisy_audio + 0.5 * dt * v_t
        time_mid = mx.full((batch_size,), t + 0.5 * dt, dtype=mx.float32)
        v_mid = self(
            noisy_audio=midpoint,
            audio_features=audio_features,
            text_features=text_features,
            time=time_mid,
            text_mask=text_mask,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
            audio_pad_mask=audio_pad_mask,
        )

        # Update
        return noisy_audio + dt * v_mid

    def separate(
        self,
        audios: Union[mx.array, List[str]],
        descriptions: List[str],
        sizes: Optional[mx.array] = None,
        anchors: Optional[List[List[Tuple[str, float, float]]]] = None,
        anchor_ids: Optional[mx.array] = None,
        anchor_alignment: Optional[mx.array] = None,
        audio_pad_mask: Optional[mx.array] = None,
        noise: Optional[mx.array] = None,
        ode_opt: Dict[str, Any] = None,
        ode_decode_chunk_size: Optional[
            int
        ] = None,  # Set it to 50 for better performance
        _text_features: Optional[mx.array] = None,
        _text_mask: Optional[mx.array] = None,
    ) -> SeparationResult:
        """
        Separate audio sources using text prompts.

        Args:
            audios: Input audio tensor (B, 1, length) or list of audio file paths
            descriptions: Text descriptions of target sounds
            sizes: Sequence lengths (B,) - if None, computed from audio_features
            anchors: Temporal anchors [[("+", start, end), ...], ...] - only used with file paths
            anchor_ids: Anchor token IDs for temporal prompts (use with mx.array input)
            anchor_alignment: Timestep to anchor mapping (use with mx.array input)
            audio_pad_mask: Padding mask for audio
            noise: Initial noise (optional)
            ode_opt: ODE solver options
            ode_decode_chunk_size: Decode in chunks of N frames (reduces peak memory)
            _text_features: Pre-computed text features (internal use)
            _text_mask: Pre-computed text mask (internal use)

        Returns:
            SeparationResult with target and residual audio
        """
        # Prepare inputs (handle file paths and anchors)
        batch = self._prepare_inputs(
            audios=audios,
            descriptions=descriptions,
            anchors=anchors,
        )
        audios = _fallback(batch.audios, audios)
        descriptions = _fallback(batch.descriptions, descriptions)
        sizes = _fallback(batch.sizes, sizes)
        anchor_ids = _fallback(batch.anchor_ids, anchor_ids)
        anchor_alignment = _fallback(batch.anchor_alignment, anchor_alignment)

        with wired_limit(self):
            if ode_opt is None:
                ode_opt = DFLT_ODE_OPT

            step_size = ode_opt.get("step_size")

            if step_size <= 0 or step_size >= 1:
                raise ValueError(
                    f"Step size {step_size} must be between 0 and 1 (exclusive). For instance, use step_size (2 / 32) = 0.0625 for 16 steps. Read more in the sam_audio [README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/sts/models/sam_audio/README.md)."
                )

            # Encode audio
            audio_features = self._get_audio_features(audios)
            mx.eval(audio_features)

            batch_size, seq_len, _ = audio_features.shape
            if sizes is None:
                sizes = mx.full((batch_size,), seq_len, dtype=mx.int32)

            # Use cached text features or encode
            if _text_features is not None and _text_mask is not None:
                text_features = _text_features
                text_mask = _text_mask
            else:
                text_features, text_mask = self.text_encoder(descriptions)
                mx.eval(text_features, text_mask)

            mx.clear_cache()

            channels = audio_features.shape[2] // 2  # Stacked features

            if noise is None:
                noise = mx.random.normal(audio_features.shape, dtype=self.dtype)

            # ODE integration from t=0 to t=1
            step_size = ode_opt.get("step_size", 2 / 32)
            method = ode_opt.get("method", "midpoint")
            num_steps = int(1.0 / step_size)

            # Select ODE step function (euler is faster but lower quality)
            ode_step_fn = (
                self._ode_step_euler if method == "euler" else self._ode_step_midpoint
            )

            noisy_audio = noise
            for i in range(num_steps):
                t = i * step_size
                noisy_audio = ode_step_fn(
                    t=t,
                    dt=step_size,
                    noisy_audio=noisy_audio,
                    audio_features=audio_features,
                    text_features=text_features,
                    text_mask=text_mask,
                    anchor_ids=anchor_ids,
                    anchor_alignment=anchor_alignment,
                    audio_pad_mask=audio_pad_mask,
                )
                mx.eval(noisy_audio)

                if (i + 1) % 4 == 0:
                    mx.clear_cache()

            mx.clear_cache()

            generated_features = mx.transpose(noisy_audio, (0, 2, 1))

            # Split into target and residual
            # generated_features has shape (B, 2*C, T)
            target_features = generated_features[:, :channels, :]
            residual_features = generated_features[:, channels:, :]

            # Decode to waveforms
            target_wavs = self.audio_codec.decode(
                target_features, chunk_size=ode_decode_chunk_size
            )
            mx.eval(target_wavs)
            mx.clear_cache()

            residual_wavs = self.audio_codec.decode(
                residual_features, chunk_size=ode_decode_chunk_size
            )
            mx.eval(residual_wavs)
            mx.clear_cache()

            # Unbatch based on sizes
            wav_sizes = [
                self.audio_codec.feature_idx_to_wav_idx(s) for s in sizes.tolist()
            ]

            target_list = []
            residual_list = []
            for b in range(batch_size):
                size = wav_sizes[b]
                target_list.append(target_wavs[b, :size, :])
                residual_list.append(residual_wavs[b, :size, :])

            return SeparationResult(
                target=target_list,
                residual=residual_list,
                noise=noise,
                peak_memory=mx.get_peak_memory() / 1e9,
            )

    def separate_long(
        self,
        audios: Union[mx.array, List[str]],
        descriptions: List[str],
        chunk_seconds: float = 10.0,
        overlap_seconds: float = 3.0,
        anchor_ids: Optional[mx.array] = None,
        anchor_alignment: Optional[mx.array] = None,
        ode_opt: Dict[str, Any] = None,
        ode_decode_chunk_size: Optional[int] = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> SeparationResult:
        """
        Separate long audio files using chunked processing to reduce memory usage.

        Args:
            audios: Input audio tensor (B, 1, length) or list of audio file paths
                    (currently only B=1 supported)
            descriptions: Text descriptions of target sounds
            chunk_seconds: Length of each chunk in seconds (default 10s, good balance)
            overlap_seconds: Overlap between chunks for smooth crossfade (default 3s, ~30%)
            anchor_ids: Anchor token IDs for temporal prompts
            anchor_alignment: Timestep to anchor mapping
            ode_opt: ODE solver options
            ode_decode_chunk_size: Decode in chunks of N frames (reduces peak memory)
            seed: Random seed for reproducible noise generation
            verbose: Print progress information

        Returns:
            SeparationResult with target and residual audio
        """
        # Prepare inputs (handle file paths)
        batch = self._prepare_inputs(
            audios=audios, descriptions=descriptions, anchors=None
        )
        audios = _fallback(batch.audios, audios)
        descriptions = _fallback(batch.descriptions, descriptions)

        if audios.shape[0] != 1:
            raise ValueError("separate_long currently only supports batch_size=1")

        sr = self.sample_rate
        chunk_samples = int(chunk_seconds * sr)
        overlap_samples = int(overlap_seconds * sr)
        hop_samples = chunk_samples - overlap_samples

        total_samples = audios.shape[2]
        total_duration = total_samples / sr

        # If audio is short enough, use regular separate
        if total_samples <= chunk_samples:
            if verbose:
                print(f"Audio is {total_duration:.1f}s, processing in single pass...")
            sizes = mx.array([total_samples // self.audio_codec.config.hop_length])
            noise = mx.random.normal(
                (1, total_samples // self.audio_codec.config.hop_length, 256),
                key=mx.random.key(seed),
            )
            return self.separate(
                audios,
                descriptions,
                sizes,
                noise=noise,
                ode_opt=ode_opt,
                ode_decode_chunk_size=ode_decode_chunk_size,
                anchor_ids=anchor_ids,
                anchor_alignment=anchor_alignment,
            )

        # Process in chunks
        target_chunks = []
        residual_chunks = []
        noise_chunks = []

        num_chunks = math.ceil((total_samples - overlap_samples) / hop_samples)

        # Pre-encode text features once (major speedup!)
        if verbose:
            print("Encoding text prompt...")
        text_features, text_mask = self.text_encoder(descriptions)
        mx.eval(text_features, text_mask)

        if verbose:
            print(
                f"Processing {total_duration:.1f}s audio in {num_chunks} chunks ({chunk_seconds}s each)..."
            )

        for i in tqdm(range(num_chunks), desc="Processing chunks"):
            start = i * hop_samples
            end = min(start + chunk_samples, total_samples)

            # Extract chunk
            chunk = audios[:, :, start:end]

            # Set random seed for reproducible noise generation
            mx.random.seed(seed + i)

            # Process chunk - let separate() generate noise with correct shape
            # sizes will be computed internally from audio features
            result = self.separate(
                chunk,
                descriptions,
                sizes=None,
                ode_opt=ode_opt,
                ode_decode_chunk_size=ode_decode_chunk_size,
                _text_features=text_features,
                _text_mask=text_mask,
            )

            target_chunk = result.target[0]
            residual_chunk = result.residual[0]

            # Evaluate to ensure computation is complete before manipulation
            mx.eval(target_chunk, residual_chunk)
            mx.clear_cache()

            # Apply crossfade for overlapping regions
            if i > 0 and overlap_samples > 0:
                # Create smooth crossfade weights (use cosine for smoother transition)
                t = mx.linspace(0, 1, overlap_samples)[:, None]
                # Cosine crossfade for smoother blending
                fade_in = 0.5 * (1 - mx.cos(math.pi * t))
                fade_out = 1 - fade_in

                # Crossfade with previous chunk's tail
                prev_target_tail = target_chunks[-1][-overlap_samples:]
                prev_residual_tail = residual_chunks[-1][-overlap_samples:]

                curr_target_head = target_chunk[:overlap_samples]
                curr_residual_head = residual_chunk[:overlap_samples]

                # Blend overlapping region
                blended_target = (
                    prev_target_tail * fade_out + curr_target_head * fade_in
                )
                blended_residual = (
                    prev_residual_tail * fade_out + curr_residual_head * fade_in
                )

                # Trim previous chunk and add blended region
                target_chunks[-1] = target_chunks[-1][:-overlap_samples]
                residual_chunks[-1] = residual_chunks[-1][:-overlap_samples]

                target_chunks.append(blended_target)
                residual_chunks.append(blended_residual)

                # Add rest of current chunk (after overlap)
                target_chunks.append(target_chunk[overlap_samples:])
                residual_chunks.append(residual_chunk[overlap_samples:])
            else:
                target_chunks.append(target_chunk)
                residual_chunks.append(residual_chunk)

            # Store noise for this chunk
            if result.noise is not None:
                noise_chunks.append(result.noise)

            # Clear cache after each chunk
            mx.clear_cache()

        if verbose:
            print("Concatenating chunks...")

        # Concatenate all chunks
        full_target = mx.concatenate(target_chunks, axis=0)
        full_residual = mx.concatenate(residual_chunks, axis=0)

        # Concatenate noise if available (for reproducibility info)
        full_noise = None
        if noise_chunks:
            full_noise = mx.concatenate(noise_chunks, axis=1)

        mx.eval(full_target, full_residual)

        return SeparationResult(
            target=[full_target],
            residual=[full_residual],
            noise=full_noise,
            peak_memory=mx.get_peak_memory() / 1e9,
        )

    def separate_streaming(
        self,
        audios: Union[mx.array, List[str]],
        descriptions: List[str],
        target_callback: Optional[Callable[[mx.array, int, bool], None]] = None,
        residual_callback: Optional[Callable[[mx.array, int, bool], None]] = None,
        chunk_seconds: float = 10.0,
        overlap_seconds: float = 3.0,
        anchor_ids: Optional[mx.array] = None,
        anchor_alignment: Optional[mx.array] = None,
        ode_opt: Dict[str, Any] = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> Union[Generator[SeparationResult, None, None], int]:
        """
        Stream audio separation - get audio ASAP.

        Processes audio in chunks and streams each chunk's output immediately.
        Time to first audio: ~10-15 seconds (one chunk's ODE) instead of waiting
        for the entire audio to be processed.

        Can be used in two modes:
        1. Generator mode (no callbacks): yields SeparationResult objects
        2. Callback mode: calls callbacks for each chunk, returns total samples

        Args:
            audios: Input audio tensor (B, 1, length) or list of audio file paths
                    (currently only B=1 supported)
            descriptions: Text descriptions of target sounds
            target_callback: Optional callback for target audio chunks:
                            callback(audio_chunk, chunk_index, is_last) -> None
            residual_callback: Optional callback for residual chunks (same signature)
            chunk_seconds: Length of each audio chunk in seconds (default 10s)
            overlap_seconds: Overlap between chunks for smooth crossfade (default 3s)
            anchor_ids: Anchor token IDs for temporal prompts
            anchor_alignment: Timestep to anchor mapping
            ode_opt: ODE solver options
            seed: Random seed for reproducible noise generation
            verbose: Print progress information

        Returns:
            Generator mode: Generator yielding StreamingChunk objects
                - chunk.target: target audio (samples, 1)
                - chunk.residual: residual audio (samples, 1)
                - chunk.chunk_idx: chunk index
                - chunk.is_last: True if final chunk
                - chunk.peak_memory: peak memory in GB (only on last chunk)
                - chunk.noise: accumulated noise for reproducibility (only on last chunk)
            Callback mode: Total number of samples processed (int)

        Example (Generator mode):
            ```python
            from mlx_audio.audio_io import write as audio_write
            import numpy as np

            target_chunks = []
            residual_chunks = []

            for chunk in model.separate_streaming(
                audios, descriptions,
                chunk_seconds=10.0,
                verbose=True,
            ):
                target_chunks.append(np.array(chunk.target[:, 0]))
                residual_chunks.append(np.array(chunk.residual[:, 0]))

                if chunk.is_last:
                    print(f"Peak memory: {chunk.peak_memory:.2f} GB")

            audio_write('target.wav', np.concatenate(target_chunks), 48000)
            audio_write('residual.wav', np.concatenate(residual_chunks), 48000)
            ```

        Example (Callback mode):
            ```python
            def write_target(chunk, idx, is_last):
                t_f.write(np.array(chunk[:, 0]))
                t_f.flush()

            samples = model.separate_streaming(
                audios, descriptions,
                target_callback=write_target,
                chunk_seconds=10.0,
            )
            ```
        """
        # Prepare inputs (handle file paths)
        batch = self._prepare_inputs(
            audios=audios, descriptions=descriptions, anchors=None
        )
        audios = _fallback(batch.audios, audios)
        descriptions = _fallback(batch.descriptions, descriptions)

        if audios.shape[0] != 1:
            raise ValueError("separate_streaming currently only supports batch_size=1")

        # Create the generator
        gen = self._separate_streaming_generator(
            audios=audios,
            descriptions=descriptions,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
            ode_opt=ode_opt,
            seed=seed,
            verbose=verbose,
        )

        # Generator mode: return the generator directly
        if target_callback is None:
            return gen

        # Callback mode: iterate and call callbacks
        total_written = 0
        for chunk in gen:
            target_callback(chunk.target, chunk.chunk_idx, chunk.is_last)
            total_written += chunk.target.shape[0]

            if residual_callback is not None:
                residual_callback(chunk.residual, chunk.chunk_idx, chunk.is_last)

        return total_written

    def _separate_streaming_generator(
        self,
        audios: mx.array,
        descriptions: List[str],
        chunk_seconds: float,
        overlap_seconds: float,
        anchor_ids: Optional[mx.array],
        anchor_alignment: Optional[mx.array],
        ode_opt: Dict[str, Any],
        seed: int,
        verbose: bool,
    ) -> Generator[SeparationResult, None, None]:
        """Internal generator for streaming separation."""
        sr = self.sample_rate
        chunk_samples = int(chunk_seconds * sr)
        overlap_samples = int(overlap_seconds * sr)
        hop_samples = chunk_samples - overlap_samples

        total_samples = audios.shape[2]
        total_duration = total_samples / sr
        num_chunks = math.ceil((total_samples - overlap_samples) / hop_samples)

        # Pre-encode text features once
        if verbose:
            print("Encoding text prompt...")
        text_features, text_mask = self.text_encoder(descriptions)
        mx.eval(text_features, text_mask)

        if verbose:
            print(
                f"Processing {total_duration:.1f}s audio in {num_chunks} chunks ({chunk_seconds}s each)..."
            )

        # Track previous chunk tails for crossfade
        prev_target_tail = None
        prev_residual_tail = None
        chunk_idx = 0

        # Track noise from each audio chunk for reproducibility
        noise_chunks = []

        for i in tqdm(range(num_chunks), desc="Processing chunks", disable=not verbose):
            start = i * hop_samples
            end = min(start + chunk_samples, total_samples)
            is_last_audio_chunk = i == num_chunks - 1

            # Extract chunk
            chunk = audios[:, :, start:end]

            # Set random seed for reproducible noise generation
            mx.random.seed(seed + i)

            # Process chunk
            result = self.separate(
                chunk,
                descriptions,
                sizes=None,
                anchor_ids=anchor_ids,
                anchor_alignment=anchor_alignment,
                ode_opt=ode_opt,
                _text_features=text_features,
                _text_mask=text_mask,
            )

            target_chunk = result.target[0]  # (samples, 1)
            residual_chunk = result.residual[0]
            mx.eval(target_chunk, residual_chunk)

            # Track noise for reproducibility
            if result.noise is not None:
                noise_chunks.append(result.noise)

            # Handle crossfade with previous chunk
            if i > 0 and overlap_samples > 0 and prev_target_tail is not None:
                # Create smooth crossfade weights
                t = mx.linspace(0, 1, overlap_samples)[:, None]
                fade_in = 0.5 * (1 - mx.cos(math.pi * t))
                fade_out = 1 - fade_in

                # Blend overlapping region
                curr_target_head = target_chunk[:overlap_samples]
                curr_residual_head = residual_chunk[:overlap_samples]

                blended_target = (
                    prev_target_tail * fade_out + curr_target_head * fade_in
                )
                blended_residual = (
                    prev_residual_tail * fade_out + curr_residual_head * fade_in
                )
                mx.eval(blended_target, blended_residual)

                # Yield blended region
                yield SeparationResult(
                    target=blended_target,
                    residual=blended_residual,
                    chunk_idx=chunk_idx,
                    is_last=False,
                )
                chunk_idx += 1

                # Yield middle part (after overlap, before tail)
                if is_last_audio_chunk:
                    # Last chunk: yield everything after overlap with stats
                    middle_target = target_chunk[overlap_samples:]
                    middle_residual = residual_chunk[overlap_samples:]
                    mx.eval(middle_target, middle_residual)

                    # Concatenate noise for reproducibility
                    full_noise = (
                        mx.concatenate(noise_chunks, axis=1) if noise_chunks else None
                    )

                    yield SeparationResult(
                        target=middle_target,
                        residual=middle_residual,
                        chunk_idx=chunk_idx,
                        is_last=True,
                        peak_memory=mx.get_peak_memory() / 1e9,
                        noise=full_noise,
                    )
                else:
                    # Not last: yield middle, save tail for next crossfade
                    middle_target = target_chunk[overlap_samples:-overlap_samples]
                    middle_residual = residual_chunk[overlap_samples:-overlap_samples]
                    mx.eval(middle_target, middle_residual)

                    yield SeparationResult(
                        target=middle_target,
                        residual=middle_residual,
                        chunk_idx=chunk_idx,
                        is_last=False,
                    )
                    chunk_idx += 1

                    # Save tail for next crossfade
                    prev_target_tail = target_chunk[-overlap_samples:]
                    prev_residual_tail = residual_chunk[-overlap_samples:]
                    mx.eval(prev_target_tail, prev_residual_tail)
            else:
                # First chunk or no overlap
                if is_last_audio_chunk or overlap_samples == 0:
                    # Yield entire chunk (possibly with stats if last)
                    if is_last_audio_chunk:
                        full_noise = (
                            mx.concatenate(noise_chunks, axis=1)
                            if noise_chunks
                            else None
                        )
                        yield SeparationResult(
                            target=target_chunk,
                            residual=residual_chunk,
                            chunk_idx=chunk_idx,
                            is_last=True,
                            peak_memory=mx.get_peak_memory() / 1e9,
                            noise=full_noise,
                        )
                    else:
                        yield SeparationResult(
                            target=target_chunk,
                            residual=residual_chunk,
                            chunk_idx=chunk_idx,
                            is_last=False,
                        )
                else:
                    # First chunk with overlap: yield all but tail
                    write_target = target_chunk[:-overlap_samples]
                    write_residual = residual_chunk[:-overlap_samples]
                    mx.eval(write_target, write_residual)

                    yield SeparationResult(
                        target=write_target,
                        residual=write_residual,
                        chunk_idx=chunk_idx,
                        is_last=False,
                    )
                    chunk_idx += 1

                    # Save tail for crossfade
                    prev_target_tail = target_chunk[-overlap_samples:]
                    prev_residual_tail = residual_chunk[-overlap_samples:]
                    mx.eval(prev_target_tail, prev_residual_tail)

            mx.clear_cache()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "SAMAudio":
        """
        Load a pretrained SAM-Audio model.

        Args:
            model_name_or_path: HuggingFace model ID or local path

        Returns:
            Loaded SAMAudio model

        Note:
            The SAM-Audio models on HuggingFace are gated and require approval.
            Visit https://huggingface.co/facebook/sam-audio-large to request access.
        """
        import glob
        import warnings

        # Download or locate model
        if Path(model_name_or_path).exists():
            model_path = Path(model_name_or_path)
        else:
            try:
                model_path = Path(
                    snapshot_download(
                        repo_id=model_name_or_path,
                        allow_patterns=["*.safetensors", "*.json", "*.pt"],
                    )
                )
            except Exception as e:
                warnings.warn(
                    f"Could not download model from {model_name_or_path}: {e}\n"
                    "SAM-Audio models are gated on HuggingFace. "
                    "Please request access at https://huggingface.co/facebook/sam-audio-large\n"
                    "Creating model with default config instead."
                )
                # Return model with default config
                return cls(SAMAudioConfig())

        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = SAMAudioConfig.from_dict(config_dict)
        else:
            warnings.warn(f"Config not found at {config_path}, using default config")
            config = SAMAudioConfig()

        # Create model
        model = cls(config)

        # Initialize processor
        model = model.post_load_hook(model_path)

        # Load weights
        weights_path = glob.glob(str(model_path / "*.safetensors"))
        pt_weights_path = model_path / "checkpoint.pt"

        if weights_path:
            try:
                weights = {}
                for weights_file in weights_path:
                    weights.update(mx.load(str(weights_file)))
                model = _load_weights(model, weights)
            except Exception as e:
                warnings.warn(f"Could not load weights: {e}")
        elif pt_weights_path.exists():
            try:
                # Load PyTorch checkpoint and convert to MLX
                import torch

                pt_weights = torch.load(
                    str(pt_weights_path), map_location="cpu", weights_only=True
                )
                # Convert to MLX arrays
                weights = {k: mx.array(v.numpy()) for k, v in pt_weights.items()}
                model = _load_weights(model, weights)
            except Exception as e:
                warnings.warn(f"Could not load weights: {e}")
        else:
            warnings.warn(
                f"Weights not found at {model_path}. " "Model will have random weights."
            )

        return model


def _load_weights(model: SAMAudio, weights: dict, strict: bool = False) -> SAMAudio:
    """Load PyTorch weights into MLX model."""
    import mlx.nn as nn

    # Sanitize weights (remove unwanted keys, combine LSTM biases, rename)
    sanitized = model.sanitize(weights)

    # Get MLX parameter names and shapes
    mlx_params = dict(nn.utils.tree_flatten(model.parameters()))

    # Match weights to parameters, transposing as needed
    new_weights = []
    loaded_keys = set()

    for key, value in sanitized.items():
        if key not in mlx_params:
            continue

        target_shape = mlx_params[key].shape
        v, t = value.shape, target_shape

        if v != t:
            # 2D: Linear layers
            if len(v) == 2 and v == t[::-1]:
                value = mx.transpose(value)
            # 3D: Conv layers
            elif len(v) == 3 and len(t) == 3:
                if (v[0], v[1], v[2]) == (t[0], t[2], t[1]):  # Conv1d
                    value = mx.transpose(value, (0, 2, 1))
                elif (v[0], v[1], v[2]) == (t[2], t[0], t[1]):  # ConvTranspose1d
                    value = mx.transpose(value, (1, 2, 0))
                elif v[1:] == (1, 1) and v[0] == t[2]:  # Weight norm (N,1,1)->(1,1,N)
                    value = mx.transpose(value, (1, 2, 0))
                elif v != t:
                    continue
            elif v != t:
                continue

        new_weights.append((key, value))
        loaded_keys.add(key)

    # Warn about missing params (exclude wm_model since watermarking is disabled)
    missing = {k for k in mlx_params.keys() - loaded_keys if "wm_model" not in k}
    if 0 < len(missing) < 50:
        import warnings

        warnings.warn(
            f"Missing {len(missing)} parameters: {', '.join(sorted(missing)[:10])}..."
        )

    model.load_weights(new_weights, strict=strict)
    mx.eval(model.parameters())
    model.eval()
    return model


def _convert_weight_name(name: str) -> str:
    """Convert PyTorch weight name to MLX convention."""
    import re

    result = name

    # === AUDIO CODEC ENCODER MAPPING ===
    # encoder.block.0.* -> encoder.conv_in.*
    if result.startswith("audio_codec.encoder.block.0."):
        result = result.replace(
            "audio_codec.encoder.block.0.", "audio_codec.encoder.conv_in."
        )

    # encoder.block.{1-4}.block.{0-2}.* -> encoder.blocks.{0-3}.res{1-3}.*
    # encoder.block.{1-4}.block.3.* -> encoder.blocks.{0-3}.snake.*
    # encoder.block.{1-4}.block.4.* -> encoder.blocks.{0-3}.conv.*
    for enc_idx in range(1, 5):
        blk_idx = enc_idx - 1
        # ResidualUnits (block.0, block.1, block.2 -> res1, res2, res3)
        for res_idx in range(3):
            res_name = f"res{res_idx + 1}"
            old_prefix = f"audio_codec.encoder.block.{enc_idx}.block.{res_idx}."
            new_prefix = f"audio_codec.encoder.blocks.{blk_idx}.{res_name}."
            if result.startswith(old_prefix):
                result = result.replace(old_prefix, new_prefix)
                # Map ResidualUnit internals: block.0 -> snake1, block.1 -> conv1, block.2 -> snake2, block.3 -> conv2
                result = _map_residual_unit(result, new_prefix)
                break
        # Snake (block.3 -> snake)
        old_prefix = f"audio_codec.encoder.block.{enc_idx}.block.3."
        new_prefix = f"audio_codec.encoder.blocks.{blk_idx}.snake."
        if result.startswith(old_prefix):
            result = result.replace(old_prefix, new_prefix)
        # Downsampling conv (block.4 -> conv)
        old_prefix = f"audio_codec.encoder.block.{enc_idx}.block.4."
        new_prefix = f"audio_codec.encoder.blocks.{blk_idx}.conv."
        if result.startswith(old_prefix):
            result = result.replace(old_prefix, new_prefix)

    # encoder.block.5.* -> encoder.snake_out.*
    if result.startswith("audio_codec.encoder.block.5."):
        result = result.replace(
            "audio_codec.encoder.block.5.", "audio_codec.encoder.snake_out."
        )

    # encoder.block.6.* -> encoder.conv_out.*
    if result.startswith("audio_codec.encoder.block.6."):
        result = result.replace(
            "audio_codec.encoder.block.6.", "audio_codec.encoder.conv_out."
        )

    # === AUDIO CODEC DECODER MAPPING ===
    # decoder.model.0.* -> decoder.conv_in.*
    if result.startswith("audio_codec.decoder.model.0."):
        result = result.replace(
            "audio_codec.decoder.model.0.", "audio_codec.decoder.conv_in."
        )

    # decoder.model.{1-4}.block.{X}.* -> decoder.blocks.{0-3}.block_{X}.*
    # Structure:
    #   block.0 -> block_0 (Snake1d)
    #   block.1 -> block_1 (WNConvTranspose1d main upsample)
    #   block.3 -> block_3 (WNConvTranspose1d watermark upsample)
    #   block.4 -> block_4 (ResidualUnit Snake dilation=1)
    #   block.5 -> block_5 (ResidualUnit Snake dilation=3)
    #   block.6 -> block_6 (ResidualUnit ELU)
    #   block.7 -> block_7 (ResidualUnit ELU)
    #   block.8 -> block_8 (ResidualUnit Snake dilation=9)
    #   block.11 -> block_11 (WNConv1d watermark downsample)
    for dec_idx in range(1, 5):
        blk_idx = dec_idx - 1
        # Map each block.{X} to block_{X}
        for block_num in [0, 1, 3, 4, 5, 6, 7, 8, 11]:
            old_prefix = f"audio_codec.decoder.model.{dec_idx}.block.{block_num}."
            new_prefix = f"audio_codec.decoder.blocks.{blk_idx}.block_{block_num}."
            if result.startswith(old_prefix):
                result = result.replace(old_prefix, new_prefix)
                # ResidualUnits have internal block structure
                if block_num in [4, 5, 6, 7, 8]:
                    result = _map_decoder_residual_unit(result, new_prefix)
                break

    # decoder final layers (snake_out and conv_out are part of wm_model.encoder_block.pre)
    # wm_model.encoder_block.pre.0 -> snake_out (Snake1d)
    # wm_model.encoder_block.pre.1 -> conv_out (NormConv1d outputs 1 channel)
    if result.startswith("audio_codec.decoder.wm_model.encoder_block.pre.0."):
        result = result.replace(
            "audio_codec.decoder.wm_model.encoder_block.pre.0.",
            "audio_codec.decoder.snake_out.",
        )
    if result.startswith("audio_codec.decoder.wm_model.encoder_block.pre.1."):
        result = result.replace(
            "audio_codec.decoder.wm_model.encoder_block.pre.1.",
            "audio_codec.decoder.conv_out.",
        )

    # wm_model block mappings: pre.N -> pre_N, post.N -> post_N
    for block in ["encoder_block", "decoder_block"]:
        for prefix in ["pre", "post"]:
            for idx in range(4):
                old = f".{block}.{prefix}.{idx}."
                new = f".{block}.{prefix}_{idx}."
                if old in result:
                    result = result.replace(old, new)

    # LSTM weight mapping: weight_ih_lN -> layers.N.Wx, weight_hh_lN -> layers.N.Wh, combined_bias_lN -> layers.N.bias
    lstm_patterns = [
        (r"\.lstm\.weight_ih_l(\d+)$", ".lstm.layers.{}.Wx"),
        (r"\.lstm\.weight_hh_l(\d+)$", ".lstm.layers.{}.Wh"),
        (r"\.lstm\.combined_bias_l(\d+)$", ".lstm.layers.{}.bias"),
    ]
    for pattern, replacement in lstm_patterns:
        match = re.search(pattern, result)
        if match:
            result = re.sub(pattern, replacement.format(match.group(1)), result)
            break

    # === QUANTIZER MAPPING ===
    # quantizer.in_proj.* -> quantizer_in_proj.*
    if result.startswith("audio_codec.quantizer.in_proj."):
        result = result.replace(
            "audio_codec.quantizer.in_proj.", "audio_codec.quantizer_in_proj."
        )
    # quantizer.out_proj.* -> quantizer_out_proj.*
    if result.startswith("audio_codec.quantizer.out_proj."):
        result = result.replace(
            "audio_codec.quantizer.out_proj.", "audio_codec.quantizer_out_proj."
        )

    return result


def _map_residual_unit(name: str, prefix: str) -> str:
    """Map ResidualUnit internal structure from PyTorch to MLX (encoder style)."""
    # PyTorch ResidualUnit: block.0 (Snake), block.1 (Conv), block.2 (Snake), block.3 (Conv)
    # MLX ResidualUnit: act1, conv1, act2, conv2
    suffix = name[len(prefix) :]

    if suffix.startswith("block.0."):
        return prefix + "act1." + suffix[8:]
    elif suffix.startswith("block.1."):
        return prefix + "conv1." + suffix[8:]
    elif suffix.startswith("block.2."):
        return prefix + "act2." + suffix[8:]
    elif suffix.startswith("block.3."):
        return prefix + "conv2." + suffix[8:]

    return name


def _map_decoder_residual_unit(name: str, prefix: str) -> str:
    """Map decoder ResidualUnit internal structure from PyTorch to MLX.

    Decoder ResidualUnits have slightly different structure:
    - Snake-based (block_4, block_5, block_8): block.0 (Snake), block.1 (Conv), block.2 (Snake), block.3 (Conv)
    - ELU-based (block_6, block_7): block.1 (Conv), block.3 (Conv) - no Snake layers

    MLX ResidualUnit: act1, conv1, act2, conv2
    """
    suffix = name[len(prefix) :]

    # block.0.alpha -> act1.alpha (Snake activation)
    if suffix.startswith("block.0."):
        return prefix + "act1." + suffix[8:]
    # block.1.* -> conv1.* (First conv)
    elif suffix.startswith("block.1."):
        return prefix + "conv1." + suffix[8:]
    # block.2.alpha -> act2.alpha (Snake activation)
    elif suffix.startswith("block.2."):
        return prefix + "act2." + suffix[8:]
    # block.3.* -> conv2.* (Second conv)
    elif suffix.startswith("block.3."):
        return prefix + "conv2." + suffix[8:]

    return name
