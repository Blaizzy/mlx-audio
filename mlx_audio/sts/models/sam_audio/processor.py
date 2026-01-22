# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

from .config import DACVAEConfig

# Type alias for anchor format: (token, start_time, end_time)
Anchor = Tuple[str, float, float]


def load_audio(audio_path: str, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample if needed.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    import os

    # Check if file exists first
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = None
    sr = None

    # Try miniaudio first
    try:
        from mlx_audio.audio_io import read as audio_read

        audio, sr = audio_read(audio_path)
    except ImportError:
        pass
    except Exception as e:
        # miniaudio failed, try librosa
        pass

    # Fallback to librosa
    if audio is None:
        try:
            import librosa

            audio, sr = librosa.load(audio_path, sr=None)
        except ImportError:
            raise ImportError(
                "Please install miniaudio or librosa for audio loading: "
                "pip install miniaudio librosa"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {audio_path}: {e}")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    # Resample if needed
    if sr != target_sr:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            try:
                import scipy.signal

                num_samples = int(len(audio) * target_sr / sr)
                audio = scipy.signal.resample(audio, num_samples)
            except ImportError:
                raise ImportError(
                    "Please install librosa or scipy for resampling: "
                    "pip install librosa scipy"
                )

    return audio.astype(np.float32), target_sr


def batch_audio(
    audios: List[Union[str, np.ndarray, mx.array]],
    audio_sampling_rate: int = 48_000,
) -> Tuple[mx.array, mx.array]:
    """
    Batch multiple audio samples with padding.

    Args:
        audios: List of audio file paths or arrays
        audio_sampling_rate: Target sample rate

    Returns:
        Tuple of (batched_audio, sizes)
        - batched_audio: (batch, 1, max_length)
        - sizes: (batch,) original lengths
    """
    wavs = []
    for audio in audios:
        if isinstance(audio, str):
            wav, _ = load_audio(audio, audio_sampling_rate)
            wav = mx.array(wav)
        elif isinstance(audio, np.ndarray):
            wav = mx.array(audio)
        else:
            wav = audio

        # Ensure 1D
        if wav.ndim > 1:
            wav = wav.mean(axis=-1) if wav.shape[-1] <= 2 else wav.mean(axis=0)

        wavs.append(wav)

    # Get sizes and pad
    sizes = mx.array([wav.shape[0] for wav in wavs])
    max_len = int(sizes.max().item())

    # Pad all wavs to max length
    batched = []
    for wav in wavs:
        if wav.shape[0] < max_len:
            pad_amount = max_len - wav.shape[0]
            wav = mx.pad(wav, [(0, pad_amount)])
        batched.append(wav)

    # Stack and add channel dimension: (batch, length) -> (batch, 1, length)
    batched = mx.stack(batched, axis=0)
    batched = mx.expand_dims(batched, 1)

    return batched, sizes


def mask_from_sizes(sizes: mx.array) -> mx.array:
    """
    Create a boolean mask from sequence sizes.

    Args:
        sizes: Tensor of sequence lengths (batch,)

    Returns:
        Boolean mask (batch, max_len) where True = valid position
    """
    max_len = int(sizes.max().item())
    batch_size = sizes.shape[0]

    # Create range tensor
    positions = mx.arange(max_len)[None, :]  # (1, max_len)
    sizes_expanded = sizes[:, None]  # (batch, 1)

    # Mask: True where position < size
    return positions < sizes_expanded


@dataclass
class Batch:
    """
    Batched data for SAM-Audio processing.

    Attributes:
        audios: Batched audio tensor (batch, 1, length)
        sizes: Feature sequence sizes (batch,)
        wav_sizes: Waveform sizes (batch,)
        descriptions: Text descriptions
        anchor_ids: Anchor token IDs (batch, num_anchors)
        anchor_alignment: Timestep to anchor mapping (batch, max_len)
        audio_pad_mask: Padding mask (batch, max_len)
    """

    audios: mx.array
    sizes: Optional[mx.array] = None
    wav_sizes: Optional[mx.array] = None
    descriptions: Optional[List[str]] = None
    anchor_ids: Optional[mx.array] = None
    anchor_alignment: Optional[mx.array] = None
    audio_pad_mask: Optional[mx.array] = None

    def __post_init__(self):
        assert self.audios.shape[0] == len(self.descriptions)


class SAMAudioProcessor:
    """
    Processor for SAM-Audio inputs.

    Handles audio loading, batching, and anchor processing for
    text and temporal prompts.
    """

    ANCHOR_DICT = {"<null>": 0, "+": 1, "-": 2, "<pad>": 3}

    def __init__(
        self,
        audio_hop_length: int,
        audio_sampling_rate: int = 48_000,
    ):
        self.audio_hop_length = audio_hop_length
        self.audio_sampling_rate = audio_sampling_rate

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "SAMAudioProcessor":
        """
        Load processor from pretrained model.

        Args:
            model_name_or_path: HuggingFace model ID or local path

        Returns:
            Configured processor
        """
        # Download or locate model
        if Path(model_name_or_path).exists():
            model_path = Path(model_name_or_path)
        else:
            model_path = Path(
                snapshot_download(
                    repo_id=model_name_or_path,
                    allow_patterns=["*.json"],
                )
            )

        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        # Extract audio codec config
        audio_config = config_dict.get("audio_codec", {})
        codec_config = DACVAEConfig(**audio_config)

        return cls(
            audio_hop_length=codec_config.hop_length,
            audio_sampling_rate=codec_config.sample_rate,
        )

    def wav_to_feature_idx(self, wav_idx: Union[int, mx.array]) -> Union[int, mx.array]:
        """Convert waveform index to feature index."""
        if isinstance(wav_idx, mx.array):
            return mx.ceil(wav_idx / self.audio_hop_length).astype(mx.int32)
        return math.ceil(wav_idx / self.audio_hop_length)

    def feature_to_wav_idx(
        self, feature_idx: Union[int, mx.array]
    ) -> Union[int, mx.array]:
        """Convert feature index to waveform index."""
        if isinstance(feature_idx, mx.array):
            return (feature_idx * self.audio_hop_length).astype(mx.int32)
        return feature_idx * self.audio_hop_length

    def process_anchors(
        self,
        anchors: Optional[List[List[Anchor]]],
        audio_pad_mask: mx.array,
        batch_size: int,
    ) -> Tuple[mx.array, mx.array]:
        """
        Process temporal anchors into IDs and alignments.

        Args:
            anchors: List of anchor lists for each sample
            audio_pad_mask: Padding mask (batch, seq_len)
            batch_size: Batch size

        Returns:
            Tuple of (anchor_ids, anchor_alignment)
        """
        seq_len = audio_pad_mask.shape[1]

        if anchors is None:
            # Default: null anchors - create directly with correct values
            # Column 0: <null>, Column 1: <pad>
            null_col = mx.full(
                (batch_size, 1), self.ANCHOR_DICT["<null>"], dtype=mx.int32
            )
            pad_col = mx.full(
                (batch_size, 1), self.ANCHOR_DICT["<pad>"], dtype=mx.int32
            )
            anchor_ids = mx.concatenate([null_col, pad_col], axis=1)

            anchor_alignment = mx.zeros((batch_size, seq_len), dtype=mx.int32)
            # Point padded positions to pad token
            # Where mask is False (padded), set alignment to 1 (pad token index)
            anchor_alignment = mx.where(
                audio_pad_mask,
                anchor_alignment,
                mx.ones_like(anchor_alignment),
            )
        else:
            # Process provided anchors using numpy for easier manipulation
            anchor_alignment_np = np.zeros((batch_size, seq_len), dtype=np.int32)
            # Set padded positions to 1 (pad token index)
            mask_np = np.array(audio_pad_mask)
            anchor_alignment_np[~mask_np] = 1

            all_ids = []
            for i, anchor_list in enumerate(anchors):
                current = [self.ANCHOR_DICT["<null>"], self.ANCHOR_DICT["<pad>"]]

                for token, start_time, end_time in anchor_list:
                    start_idx = self.wav_to_feature_idx(
                        int(start_time * self.audio_sampling_rate)
                    )
                    end_idx = self.wav_to_feature_idx(
                        int(end_time * self.audio_sampling_rate)
                    )

                    # Update alignment for this span
                    anchor_idx = len(current)
                    anchor_alignment_np[i, start_idx : min(end_idx, seq_len)] = (
                        anchor_idx
                    )

                    current.append(self.ANCHOR_DICT.get(token, 0))

                all_ids.append(current)

            # Convert back to MLX
            anchor_alignment = mx.array(anchor_alignment_np)

            # Pad anchor IDs to same length
            max_anchors = max(len(ids) for ids in all_ids)
            padded_ids = []
            for ids in all_ids:
                ids_padded = ids + [self.ANCHOR_DICT["<pad>"]] * (
                    max_anchors - len(ids)
                )
                padded_ids.append(ids_padded)

            anchor_ids = mx.array(padded_ids, dtype=mx.int32)

        return anchor_ids, anchor_alignment

    def __call__(
        self,
        descriptions: List[str],
        audios: List[Union[str, np.ndarray, mx.array]],
        anchors: Optional[List[List[Anchor]]] = None,
    ) -> Batch:
        """
        Process inputs for SAM-Audio.

        Args:
            descriptions: Text descriptions of target sounds
            audios: Audio file paths or arrays
            anchors: Optional temporal anchors for each sample
                Format: [[("+", start_time, end_time), ...], ...]

        Returns:
            Batch object ready for model input

        Example:
            >>> processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
            >>> batch = processor(
            ...     descriptions=["A man speaking"],
            ...     audios=["audio.wav"],
            ... )
            >>> # With temporal anchors:
            >>> batch = processor(
            ...     descriptions=["Speech"],
            ...     audios=["audio.wav"],
            ...     anchors=[[("+", 1.5, 3.0)]],  # Target speech from 1.5s to 3.0s
            ... )
        """
        assert len(descriptions) == len(audios)
        if anchors is not None:
            assert len(descriptions) == len(anchors)

        # Batch audio
        audios_batched, wav_sizes = batch_audio(audios, self.audio_sampling_rate)

        # Convert to feature sizes
        sizes = self.wav_to_feature_idx(wav_sizes)

        # Create padding mask
        audio_pad_mask = mask_from_sizes(sizes)

        # Process anchors
        anchor_ids, anchor_alignment = self.process_anchors(
            anchors, audio_pad_mask, len(descriptions)
        )

        return Batch(
            audios=audios_batched,
            sizes=sizes,
            wav_sizes=wav_sizes,
            descriptions=descriptions,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
            audio_pad_mask=audio_pad_mask,
        )


def save_audio(
    audio: Union[mx.array, np.ndarray],
    path: str,
    sample_rate: int = 48000,
):
    """
    Save audio to file.

    Args:
        audio: Audio array (length,) or (length, 1)
        path: Output file path
        sample_rate: Sample rate
    """
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    if audio.ndim > 1:
        audio = audio.squeeze()

    try:
        from mlx_audio.audio_io import write as audio_write

        audio_write(path, audio, sample_rate)
    except ImportError:
        try:
            from scipy.io import wavfile

            # Normalize to int16 range
            audio_int = (audio * 32767).astype(np.int16)
            wavfile.write(path, sample_rate, audio_int)
        except ImportError:
            raise ImportError(
                "Please install miniaudio or scipy for audio saving: "
                "pip install miniaudio scipy"
            )
