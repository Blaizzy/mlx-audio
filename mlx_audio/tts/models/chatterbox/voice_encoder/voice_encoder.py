# Ported from https://github.com/resemble-ai/chatterbox

import math
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VoiceEncConfig
from .melspec import melspectrogram


def get_num_wins(
    n_frames: int,
    step: int,
    min_coverage: float,
    hp: VoiceEncConfig,
):
    """Calculate number of windows and target length for partial utterance splitting."""
    assert n_frames > 0
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def get_frame_step(
    overlap: float,
    rate: Optional[float],
    hp: VoiceEncConfig,
):
    """Compute how many frames separate two partial utterances."""
    assert 0 <= overlap < 1
    if rate is None:
        frame_step = int(round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(round((hp.sample_rate / rate) / hp.ve_partial_frames))
    assert 0 < frame_step <= hp.ve_partial_frames
    return frame_step


def sanitize_lstm_weights(
    key: str, value: mx.array, num_layers: int = 3
) -> Dict[str, mx.array]:
    """
    Convert PyTorch LSTM weight keys to MLX LSTM weight keys.

    PyTorch LSTM weight format:
        lstm.weight_ih_l{layer} -> input-hidden weights for layer
        lstm.weight_hh_l{layer} -> hidden-hidden weights for layer
        lstm.bias_ih_l{layer} -> input-hidden bias for layer
        lstm.bias_hh_l{layer} -> hidden-hidden bias for layer

    MLX LSTM weight format (for list of LSTM layers):
        lstm.{layer}.Wx -> input weights
        lstm.{layer}.Wh -> hidden weights
        lstm.{layer}.bias -> combined bias (ih + hh)
    """
    result = {}

    # Extract layer number from key like "lstm.weight_ih_l0"
    import re

    match = re.search(r"lstm\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)", key)
    if not match:
        result[key] = value
        return result

    weight_type = match.group(1)
    layer_idx = int(match.group(2))

    if weight_type == "weight_ih":
        result[f"lstm.layers.{layer_idx}.Wx"] = value
    elif weight_type == "weight_hh":
        result[f"lstm.layers.{layer_idx}.Wh"] = value
    elif weight_type == "bias_ih":
        # MLX combines biases, so we need special handling
        result[f"lstm.layers.{layer_idx}.bias_ih"] = value
    elif weight_type == "bias_hh":
        result[f"lstm.layers.{layer_idx}.bias_hh"] = value

    return result


class StackedLSTM(nn.Module):
    """Multi-layer LSTM that matches PyTorch's nn.LSTM(num_layers=N)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create stacked LSTM layers
        self.layers = [
            nn.LSTM(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ]

    def __call__(self, x: mx.array, hidden=None):
        """
        Args:
            x: Input tensor (seq_len, batch, input_size) or (batch, seq_len, input_size)
            hidden: Tuple of (h_0, c_0), each of shape (num_layers, batch, hidden_size)

        Returns:
            output: Output tensor
            (h_n, c_n): Final hidden and cell states
        """
        if hidden is None:
            h_list = [None] * self.num_layers
            c_list = [None] * self.num_layers
        else:
            h_0, c_0 = hidden
            h_list = [h_0[i] for i in range(self.num_layers)]
            c_list = [c_0[i] for i in range(self.num_layers)]

        output = x
        new_h = []
        new_c = []

        for i, layer in enumerate(self.layers):
            all_h, all_c = layer(output, hidden=h_list[i], cell=c_list[i])
            output = all_h
            # Extract final timestep: all_h has shape (batch, seq_len, hidden_size)
            # Use [:, -1, :] to get (batch, hidden_size) for the last timestep
            new_h.append(all_h[:, -1, :] if all_h.ndim == 3 else all_h)
            new_c.append(all_c[:, -1, :] if all_c.ndim == 3 else all_c)

        h_n = mx.stack(new_h, axis=0)
        c_n = mx.stack(new_c, axis=0)

        return output, (h_n, c_n)


class VoiceEncoder(nn.Module):
    """LSTM-based voice encoder for speaker embeddings."""

    def __init__(self, hp: VoiceEncConfig = None):
        super().__init__()
        self.hp = hp or VoiceEncConfig()

        # Network definition: 3-layer stacked LSTM
        self.lstm = StackedLSTM(self.hp.num_mels, self.hp.ve_hidden_size, num_layers=3)
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)

        # Cosine similarity scaling (fixed initial parameter values)
        self.similarity_weight = mx.array([10.0])
        self.similarity_bias = mx.array([-5.0])

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitize PyTorch weights for MLX.

        Handles:
        - LSTM weight renaming (weight_ih_l0 -> layers.0.Wx, etc.)
        - Bias combining (PyTorch has separate ih/hh biases)

        This method is idempotent - key renaming only affects PyTorch-format keys,
        MLX-format keys pass through unchanged on subsequent runs.
        """
        new_weights = {}
        bias_ih = {}
        bias_hh = {}

        for key, value in weights.items():
            # Handle LSTM weights
            if "lstm." in key and any(
                x in key for x in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]
            ):
                import re

                match = re.search(
                    r"lstm\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)", key
                )
                if match:
                    weight_type = match.group(1)
                    layer_idx = int(match.group(2))

                    if weight_type == "weight_ih":
                        new_weights[f"lstm.layers.{layer_idx}.Wx"] = value
                    elif weight_type == "weight_hh":
                        new_weights[f"lstm.layers.{layer_idx}.Wh"] = value
                    elif weight_type == "bias_ih":
                        bias_ih[layer_idx] = value
                    elif weight_type == "bias_hh":
                        bias_hh[layer_idx] = value
            else:
                new_weights[key] = value

        # Combine ih and hh biases (MLX LSTM uses combined bias)
        for layer_idx in bias_ih:
            if layer_idx in bias_hh:
                combined_bias = bias_ih[layer_idx] + bias_hh[layer_idx]
                new_weights[f"lstm.layers.{layer_idx}.bias"] = combined_bias

        return new_weights

    def __call__(self, mels: mx.array) -> mx.array:
        """
        Computes the embeddings of a batch of partial utterances.

        Args:
            mels: Batch of unscaled mel spectrograms (B, T, M)
                  where T is hp.ve_partial_frames

        Returns:
            Embeddings as (B, E) where E is hp.speaker_embed_size.
            Embeddings are L2-normed.
        """
        if self.hp.normalized_mels:
            min_val = mx.min(mels)
            max_val = mx.max(mels)
            if float(min_val) < 0 or float(max_val) > 1:
                raise Exception(f"Mels outside [0, 1]. Min={min_val}, Max={max_val}")

        # Pass full sequence through LSTM layers (vectorized, no per-timestep loop)
        _, (h_n, _) = self.lstm(mels)

        # Get final hidden state from last layer
        final_hidden = h_n[-1]  # (B, H)

        # Project
        raw_embeds = self.proj(final_hidden)

        # Apply ReLU if configured
        if self.hp.ve_final_relu:
            raw_embeds = nn.relu(raw_embeds)

        # L2 normalize
        embeds = raw_embeds / mx.linalg.norm(raw_embeds, axis=1, keepdims=True)

        return embeds

    def inference(
        self,
        mels: mx.array,
        mel_lens: List[int],
        overlap: float = 0.5,
        rate: Optional[float] = None,
        min_coverage: float = 0.8,
        batch_size: Optional[int] = None,
    ) -> mx.array:
        """
        Computes embeddings of a batch of full utterances.

        Args:
            mels: (B, T, M) unscaled mels
            mel_lens: List of mel lengths for each batch item
            overlap: Overlap between partial windows
            rate: Rate for frame step calculation
            min_coverage: Minimum coverage for partial windows
            batch_size: Batch size for processing partials

        Returns:
            (B, E) embeddings
        """
        # Compute where to split the utterances into partials
        frame_step = get_frame_step(overlap, rate, self.hp)
        n_partials_list = []
        target_lens = []
        for l in mel_lens:
            n_p, t_l = get_num_wins(l, frame_step, min_coverage, self.hp)
            n_partials_list.append(n_p)
            target_lens.append(t_l)

        # Possibly pad the mels to reach the target lengths
        len_diff = max(target_lens) - mels.shape[1]
        if len_diff > 0:
            pad = mx.zeros((mels.shape[0], len_diff, self.hp.num_mels))
            mels = mx.concatenate([mels, pad], axis=1)

        # Group all partials together (vectorized extraction)
        # For each mel, extract all partials at once using mx.take
        partial_list = []
        for mel, n_partial in zip(mels, n_partials_list):
            if n_partial > 0:
                # Create indices for all partials at once: (n_partial, ve_partial_frames)
                partial_starts = mx.arange(n_partial) * frame_step  # (n_partial,)
                frame_offsets = mx.arange(
                    self.hp.ve_partial_frames
                )  # (ve_partial_frames,)
                # Broadcast to get all indices: (n_partial, ve_partial_frames)
                indices = partial_starts[:, None] + frame_offsets[None, :]

                # Extract all partials at once for each mel dimension
                # mel is (T, M), we need (n_partial, ve_partial_frames, M)
                mel_partials = mx.take(mel, indices.flatten(), axis=0).reshape(
                    n_partial, self.hp.ve_partial_frames, mel.shape[1]
                )
                partial_list.append(mel_partials)

        partials = mx.concatenate(partial_list, axis=0)  # (total_partials, T, M)

        # Forward the partials (in batches if needed)
        if batch_size is None or batch_size >= len(partials):
            partial_embeds = self(partials)
        else:
            embed_chunks = []
            for i in range(0, len(partials), batch_size):
                chunk = partials[i : i + batch_size]
                embed_chunks.append(self(chunk))
            partial_embeds = mx.concatenate(embed_chunks, axis=0)

        # Reduce the partial embeds into full embeds
        # Compute slice indices without numpy
        slices = [0]
        for n in n_partials_list:
            slices.append(slices[-1] + n)
        raw_embeds = []
        for start, end in zip(slices[:-1], slices[1:]):
            raw_embeds.append(mx.mean(partial_embeds[start:end], axis=0))
        raw_embeds = mx.stack(raw_embeds)

        # L2-normalize the final embeds
        embeds = raw_embeds / mx.linalg.norm(raw_embeds, axis=1, keepdims=True)

        return embeds

    @staticmethod
    def utt_to_spk_embed(utt_embeds: mx.array) -> mx.array:
        """
        Takes L2-normalized utterance embeddings, computes mean and L2-normalizes
        to get a speaker embedding.
        """
        assert len(utt_embeds.shape) == 2
        utt_embeds = mx.mean(utt_embeds, axis=0)
        return utt_embeds / mx.linalg.norm(utt_embeds)

    @staticmethod
    def voice_similarity(embeds_x: mx.array, embeds_y: mx.array) -> float:
        """Cosine similarity for L2-normalized embeddings."""
        if len(embeds_x.shape) != 1:
            embeds_x = VoiceEncoder.utt_to_spk_embed(embeds_x)
        if len(embeds_y.shape) != 1:
            embeds_y = VoiceEncoder.utt_to_spk_embed(embeds_y)
        return float(embeds_x @ embeds_y)

    def embeds_from_mels(
        self,
        mels: List[mx.array],
        mel_lens: Optional[List[int]] = None,
        as_spk: bool = False,
        batch_size: int = 32,
        **kwargs,
    ) -> mx.array:
        """
        Convenience function for deriving utterance or speaker embeddings from mel spectrograms.

        Args:
            mels: List of (Ti, M) arrays or stacked (B, T, M) array
            mel_lens: Individual mel lengths if mels is stacked
            as_spk: Whether to return speaker embedding (single) or utterance embeddings (per-utt)
            batch_size: Batch size for processing
            **kwargs: Additional args for inference()

        Returns:
            (B, E) embeddings if as_spk is False, else (E,) speaker embedding
        """
        # Handle list of mels
        if isinstance(mels, list):
            mel_lens = [mel.shape[0] for mel in mels]
            max_len = max(mel_lens)
            # Pad and stack
            padded = []
            for mel in mels:
                if mel.shape[0] < max_len:
                    pad = mx.zeros((max_len - mel.shape[0], mel.shape[1]))
                    mel = mx.concatenate([mel, pad], axis=0)
                padded.append(mel)
            mels = mx.stack(padded)

        # Embed them
        utt_embeds = self.inference(mels, mel_lens, batch_size=batch_size, **kwargs)

        return self.utt_to_spk_embed(utt_embeds) if as_spk else utt_embeds

    def embeds_from_wavs(
        self,
        wavs: List[mx.array],
        sample_rate: int,
        as_spk: bool = False,
        batch_size: int = 32,
        trim_top_db: Optional[float] = 20,
        **kwargs,
    ) -> mx.array:
        """
        Wrapper around embeds_from_mels that first converts waveforms to mel spectrograms.

        Args:
            wavs: List of waveform arrays
            sample_rate: Sample rate of the waveforms
            as_spk: Whether to return speaker embedding
            batch_size: Batch size for processing
            trim_top_db: Trim silence below this threshold (None to disable)
            **kwargs: Additional args for inference()

        Returns:
            Embeddings
        """
        import scipy.signal

        # Resample if needed
        if sample_rate != self.hp.sample_rate:
            import numpy as np

            resampled_wavs = []
            for wav in wavs:
                # Convert to numpy for resampling
                wav_np = np.array(wav)
                # Calculate resampling ratio
                gcd = math.gcd(sample_rate, self.hp.sample_rate)
                up = self.hp.sample_rate // gcd
                down = sample_rate // gcd
                wav_resampled = scipy.signal.resample_poly(wav_np, up, down)
                resampled_wavs.append(mx.array(wav_resampled))
            wavs = resampled_wavs

        # Trim silence if requested
        if trim_top_db is not None:
            import numpy as np

            trimmed_wavs = []
            for wav in wavs:
                wav_np = np.array(wav)
                # Simple energy-based trimming
                # Calculate frame energy
                frame_length = 2048
                hop_length = 512
                # Compute RMS energy per frame
                n_frames = 1 + (len(wav_np) - frame_length) // hop_length
                if n_frames > 0:
                    rms = np.array(
                        [
                            math.sqrt(
                                np.mean(
                                    wav_np[
                                        i * hop_length : i * hop_length + frame_length
                                    ]
                                    ** 2
                                )
                            )
                            for i in range(n_frames)
                        ]
                    )
                    # Convert to dB
                    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
                    threshold = np.max(rms_db) - trim_top_db
                    # Find non-silent frames
                    non_silent = np.where(rms_db >= threshold)[0]
                    if len(non_silent) > 0:
                        start_frame = non_silent[0]
                        end_frame = non_silent[-1] + 1
                        start_sample = start_frame * hop_length
                        end_sample = min(
                            end_frame * hop_length + frame_length, len(wav_np)
                        )
                        wav_np = wav_np[start_sample:end_sample]
                trimmed_wavs.append(mx.array(wav_np))
            wavs = trimmed_wavs

        # Set default rate if not provided
        if "rate" not in kwargs:
            kwargs["rate"] = 1.3  # Resemble's default value

        # Convert waveforms to mel spectrograms
        mels = []
        for wav in wavs:
            mel = melspectrogram(wav, self.hp)
            # Transpose from (M, T) to (T, M)
            mel = mx.transpose(mel)
            mels.append(mel)

        return self.embeds_from_mels(
            mels, as_spk=as_spk, batch_size=batch_size, **kwargs
        )
