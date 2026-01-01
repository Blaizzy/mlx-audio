# Original: ClearerVoice-Studio (github.com/modelscope/ClearerVoice-Studio)
# Copyright (c) Speech Lab, Alibaba Group
# Licensed under Apache License 2.0
# MLX port by Dmitry Starkov

"""
Processor for MossFormer2 SE speech enhancement.

Handles loading, processing, and saving audio for speech enhancement.
"""

import time
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from mlx.utils import tree_unflatten

# Reuse audio utilities from sam_audio
from ..sam_audio.processor import load_audio
from .config import MossFormer2SEConfig
from .deltas import compute_deltas
from .fbank import compute_fbank
from .mossformer2_se_wrapper import MossFormer2SE
from .stft import ISTFTCache, create_window, stft

# Constants
MAX_WAV_VALUE = 32768.0
DEFAULT_REPO = "starkdmi/MossFormer2_SE_48K_MLX"


class MossFormer2SEProcessor:
    """Processor for MossFormer2 SE speech enhancement.

    Handles model loading, audio processing, and enhancement.

    Example:
        >>> processor = MossFormer2SEProcessor.from_pretrained("starkdmi/MossFormer2_SE_48K_MLX")
        >>> enhanced = processor.enhance("noisy.wav")
        >>> save_audio(enhanced, "clean.wav")
    """

    def __init__(
        self,
        model,
        config: MossFormer2SEConfig,
    ):
        """Initialize processor.

        Args:
            model: Loaded MossFormer2 model (TestNet)
            config: Model configuration
        """
        self.model = model
        self.config = config
        self._istft_cache = ISTFTCache()
        self._window = None
        self._warmed_up = False

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_REPO,
        precision: str = "fp32",
    ) -> "MossFormer2SEProcessor":
        """Load processor from pretrained model.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            precision: One of "fp32", "fp16", "int8", "int6", "int4"

        Returns:
            Configured processor
        """
        print(f"Loading MossFormer2 SE 48K ({precision})...")

        # Enable fast LayerNorm
        nn.LayerNorm.__call__ = lambda self, x: mx.fast.layer_norm(
            x, self.weight, self.bias, self.eps
        )

        config = MossFormer2SEConfig()
        model = MossFormer2SE(config)

        # Check if quantized
        quantized_precisions = {"int4": 4, "int6": 6, "int8": 8}
        is_quantized = precision in quantized_precisions

        if is_quantized:
            bits = quantized_precisions[precision]
            model_file = f"model_{precision}.safetensors"

            if Path(model_name_or_path).exists():
                weights_path = str(Path(model_name_or_path) / model_file)
            else:
                weights_path = hf_hub_download(
                    repo_id=model_name_or_path, filename=model_file
                )

            nn.quantize(model, group_size=64, bits=bits)
        else:
            model_file = f"model_{precision}.safetensors"

            if Path(model_name_or_path).exists():
                weights_path = str(Path(model_name_or_path) / model_file)
            else:
                weights_path = hf_hub_download(
                    repo_id=model_name_or_path, filename=model_file
                )

        weights = mx.load(weights_path)
        model.update(tree_unflatten(list(weights.items())))

        total_params = sum(v.size for v in weights.values() if hasattr(v, "size"))
        print(f"Model loaded: {total_params:,} parameters")

        return cls(model=model.model, config=config)

    def warmup(self, chunked: bool = False) -> None:
        """Warm up model for optimal performance.

        Args:
            chunked: If True, warmup for chunked mode
        """
        print("Warming up model...")
        warmup_start = time.time()

        if chunked:
            window = create_window(
                self.config.win_type, self.config.win_len, periodic=False
            )
            chunk_audio = mx.random.uniform(-0.1, 0.1, shape=(48000,)).astype(
                mx.float32
            )
            _ = self._process_chunk(chunk_audio, window, 48000)
        else:
            warmup_audio = mx.random.uniform(-0.1, 0.1, shape=(1, 24000)).astype(
                mx.float32
            )
            _ = self._decode_one_audio(warmup_audio)

        self._warmed_up = True
        warmup_time = time.time() - warmup_start
        print(f"Warmup complete: {warmup_time:.2f}s\n")

    def enhance(
        self,
        audio_input: Union[str, np.ndarray, mx.array],
        chunked: Optional[bool] = None,
    ) -> np.ndarray:
        """Enhance audio.

        Args:
            audio_input: Audio file path or audio array
            chunked: Force chunked processing. If None, auto-selects:
                     - < 60s: Full mode (faster, best quality)
                     - >= 60s: Chunked mode (lower RAM)

        Returns:
            Enhanced audio as numpy array
        """
        # Load audio if path
        if isinstance(audio_input, str):
            audio_np, sr = load_audio(audio_input, self.config.sample_rate)
        elif isinstance(audio_input, mx.array):
            audio_np = np.array(audio_input)
            sr = self.config.sample_rate
        else:
            audio_np = audio_input
            sr = self.config.sample_rate

        # Ensure correct shape
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        elif audio_np.ndim == 2 and audio_np.shape[0] > audio_np.shape[1]:
            audio_np = audio_np.T
            audio_np = audio_np[0:1]

        duration = audio_np.shape[1] / self.config.sample_rate

        # Auto-select mode
        use_chunked = (
            chunked
            if chunked is not None
            else (duration >= self.config.auto_chunk_threshold)
        )

        # Process
        if use_chunked:
            enhanced = self._decode_chunked(audio_np)
        else:
            enhanced = self._decode_one_audio(audio_np)

        return enhanced

    def _decode_one_audio(self, inputs: np.ndarray) -> np.ndarray:
        """Full audio processing."""
        if hasattr(inputs, "numpy"):
            inputs_np = inputs.numpy()
        else:
            inputs_np = inputs

        if inputs_np.ndim == 2:
            inputs_np = inputs_np[0, :]

        input_len = inputs_np.shape[0]
        original_len = input_len
        inputs_np = inputs_np * MAX_WAV_VALUE

        window = create_window(
            self.config.win_type, self.config.win_len, periodic=False
        )

        # Check if segmented processing is needed
        if input_len > self.config.sample_rate * self.config.one_time_decode_length:
            print(
                f"  Using segmented processing for {input_len / self.config.sample_rate:.1f}s audio"
            )

            window_size = int(self.config.sample_rate * self.config.decode_window)
            stride = int(window_size * 0.75)
            t = inputs_np.shape[0]

            # Pad input
            if t < window_size:
                inputs_np = np.concatenate([inputs_np, np.zeros(window_size - t)], 0)
            elif t < window_size + stride:
                padding = window_size + stride - t
                inputs_np = np.concatenate([inputs_np, np.zeros(padding)], 0)
            else:
                if (t - window_size) % stride != 0:
                    padding = t - (t - window_size) // stride * stride
                    inputs_np = np.concatenate([inputs_np, np.zeros(padding)], 0)

            audio = mx.array(inputs_np)
            t = audio.shape[0]
            output_segments = []
            output_ranges = []
            give_up_length = (window_size - stride) // 2
            current_idx = 0

            while current_idx + window_size <= t:
                audio_segment = audio[current_idx : current_idx + window_size]
                output_segment = self._process_chunk(audio_segment, window, window_size)
                mx.eval(output_segment)

                if current_idx == 0:
                    output_segments.append(output_segment[:-give_up_length])
                    output_ranges.append(
                        (current_idx, current_idx + window_size - give_up_length)
                    )
                else:
                    output_segments.append(
                        output_segment[give_up_length:-give_up_length]
                    )
                    output_ranges.append(
                        (
                            current_idx + give_up_length,
                            current_idx + window_size - give_up_length,
                        )
                    )

                current_idx += stride

            outputs_np = np.zeros(t)
            for segment, (start, end) in zip(output_segments, output_ranges):
                segment_np = np.array(segment)
                outputs_np[start:end] = segment_np

            outputs_np = outputs_np[:original_len]

        else:
            # Process entire audio
            audio = mx.array(inputs_np)
            output = self._process_chunk(audio, window, len(audio))
            mx.eval(output)
            outputs_np = np.array(output)

        return outputs_np / MAX_WAV_VALUE

    def _decode_chunked(self, inputs: np.ndarray) -> np.ndarray:
        """Chunked audio processing with discard-edges reassembly."""
        if hasattr(inputs, "numpy"):
            inputs_np = inputs.numpy()
        else:
            inputs_np = inputs

        if inputs_np.ndim == 2:
            inputs_np = inputs_np[0, :]

        original_len = inputs_np.shape[0]
        inputs_np = inputs_np * MAX_WAV_VALUE

        window = create_window(
            self.config.win_type, self.config.win_len, periodic=False
        )

        chunk_samples = int(self.config.sample_rate * self.config.chunk_seconds)
        overlap_samples = int(chunk_samples * self.config.chunk_overlap)
        stride = chunk_samples - overlap_samples
        give_up = overlap_samples // 2

        if original_len <= chunk_samples:
            audio = mx.array(inputs_np)
            result = self._process_chunk(audio, window, original_len)
            return np.array(result) / MAX_WAV_VALUE

        num_full_chunks = (original_len - chunk_samples) // stride + 1
        remaining = original_len - (num_full_chunks - 1) * stride - chunk_samples
        has_partial = remaining > 0

        print(
            f"  Chunked: {num_full_chunks} x {self.config.chunk_seconds}s"
            + (f" + partial" if has_partial else "")
        )

        chunks = []
        chunk_starts = []
        current_idx = 0

        while current_idx + chunk_samples <= original_len:
            audio_segment = mx.array(
                inputs_np[current_idx : current_idx + chunk_samples]
            )
            chunk_result = self._process_chunk(audio_segment, window, chunk_samples)
            chunks.append(np.array(chunk_result))
            chunk_starts.append(current_idx)
            current_idx += stride

        if current_idx < original_len:
            remaining = original_len - current_idx
            audio_segment = mx.array(inputs_np[current_idx:])
            chunk_result = self._process_chunk(audio_segment, window, remaining)
            chunks.append(np.array(chunk_result))
            chunk_starts.append(current_idx)

        # Reassemble
        output = np.zeros(original_len)
        num_chunks = len(chunks)

        for idx, (chunk, start_idx) in enumerate(zip(chunks, chunk_starts)):
            chunk_len = len(chunk)
            is_first = idx == 0
            is_last = idx == num_chunks - 1

            if is_last and chunk_len < chunk_samples:
                keep_start = give_up if not is_first else 0
                keep_end = chunk_len
            else:
                keep_start = 0 if is_first else give_up
                keep_end = chunk_len - give_up

            output_start = start_idx + keep_start
            output_end = min(start_idx + keep_end, original_len)
            chunk_slice = chunk[keep_start : keep_start + (output_end - output_start)]
            output[output_start:output_end] = chunk_slice

        return output / MAX_WAV_VALUE

    def _process_chunk(
        self, audio_segment: mx.array, window: mx.array, chunk_length: int
    ) -> mx.array:
        """Process a single audio chunk."""
        # Feature extraction
        fbanks = compute_fbank(audio_segment, self.config)
        fbank_transposed = mx.transpose(fbanks, [1, 0])
        fbank_delta = compute_deltas(fbank_transposed, win_length=5)
        fbank_delta_delta = compute_deltas(fbank_delta, win_length=5)
        fbank_delta = mx.transpose(fbank_delta, [1, 0])
        fbank_delta_delta = mx.transpose(fbank_delta_delta, [1, 0])
        fbanks = mx.concatenate([fbanks, fbank_delta, fbank_delta_delta], axis=1)
        fbanks = mx.expand_dims(fbanks, axis=0)

        # Model inference
        Out_List = self.model(fbanks)
        pred_mask = Out_List[-1][0]

        # STFT
        real_part, imag_part = stft(
            audio_segment.reshape(1, -1),
            self.config.fft_len,
            self.config.win_inc,
            self.config.win_len,
            window,
            center=False,
        )

        # Apply mask
        pred_mask = mx.transpose(pred_mask, [1, 0])
        pred_mask = mx.expand_dims(pred_mask, axis=-1)
        spectrum_real = real_part[0] * pred_mask[:, :, 0]
        spectrum_imag = imag_part[0] * pred_mask[:, :, 0]

        # iSTFT
        output_segment = self._istft_cache.istft(
            spectrum_real.reshape(1, *spectrum_real.shape),
            spectrum_imag.reshape(1, *spectrum_imag.shape),
            self.config.fft_len,
            self.config.win_inc,
            self.config.win_len,
            window,
            center=False,
            audio_length=chunk_length,
        )
        mx.eval(output_segment)

        return output_segment[0]
