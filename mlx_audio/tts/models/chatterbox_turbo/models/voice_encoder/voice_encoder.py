# Copyright (c) 2025 Resemble AI
# MIT License
# MLX port of Voice Encoder (LSTM-based speaker encoder)
# Adapted from https://github.com/CorentinJ/Real-Time-Voice-Cloning

from typing import List, Optional, Union

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from numpy.lib.stride_tricks import as_strided

from .config import VoiceEncConfig
from .melspec import melspectrogram


def pack(arrays, seq_len: int = None, pad_value=0):
    """
    Given a list of arrays, packs them into a single array by padding.
    """
    if seq_len is None:
        seq_len = max(len(array) for array in arrays)
    else:
        assert seq_len >= max(len(array) for array in arrays)

    # Convert lists to np.array
    if isinstance(arrays[0], list):
        arrays = [np.array(array) for array in arrays]

    # Fill the packed tensor with the array data
    packed_shape = (len(arrays), seq_len, *arrays[0].shape[1:])
    packed_array = np.full(packed_shape, pad_value, dtype=np.float32)

    for i, array in enumerate(arrays):
        packed_array[i, : len(array)] = array

    return mx.array(packed_array)


def get_num_wins(
    n_frames: int,
    step: int,
    min_coverage: float,
    hp: VoiceEncConfig,
):
    assert n_frames > 0
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def get_frame_step(
    overlap: float,
    rate: float,
    hp: VoiceEncConfig,
):
    assert 0 <= overlap < 1
    if rate is None:
        frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
    assert 0 < frame_step <= hp.ve_partial_frames
    return frame_step


def stride_as_partials(
    mel: np.ndarray,
    hp: VoiceEncConfig,
    overlap=0.5,
    rate: float = None,
    min_coverage=0.8,
):
    """
    Takes unscaled mels in (T, M) format and creates overlapping partials.
    """
    assert 0 < min_coverage <= 1
    frame_step = get_frame_step(overlap, rate, hp)

    # Compute how many partials can fit in the mel
    n_partials, target_len = get_num_wins(len(mel), frame_step, min_coverage, hp)

    # Trim or pad the mel spectrogram to match the number of partials
    if target_len > len(mel):
        mel = np.concatenate((mel, np.full((target_len - len(mel), hp.num_mels), 0)))
    elif target_len < len(mel):
        mel = mel[:target_len]

    # Ensure the numpy array data is float32 and contiguous in memory
    mel = mel.astype(np.float32, order="C")

    # Re-arrange the array in memory to be of shape (N, P, M) with partials overlapping
    shape = (n_partials, hp.ve_partial_frames, hp.num_mels)
    strides = (mel.strides[0] * frame_step, mel.strides[0], mel.strides[1])
    partials = as_strided(mel, shape, strides)
    return partials


class VoiceEncoder(nn.Module):
    """
    LSTM-based voice encoder for speaker embeddings.
    """

    def __init__(self, hp: VoiceEncConfig = None):
        super().__init__()
        if hp is None:
            hp = VoiceEncConfig()
        self.hp = hp

        # LSTM layers (3 layers stacked manually since MLX LSTM doesn't support num_layers)
        self.lstm1 = nn.LSTM(
            input_size=hp.num_mels,
            hidden_size=hp.ve_hidden_size,
        )
        self.lstm2 = nn.LSTM(
            input_size=hp.ve_hidden_size,
            hidden_size=hp.ve_hidden_size,
        )
        self.lstm3 = nn.LSTM(
            input_size=hp.ve_hidden_size,
            hidden_size=hp.ve_hidden_size,
        )

        # Projection layer
        self.proj = nn.Linear(hp.ve_hidden_size, hp.speaker_embed_size)

        # Cosine similarity scaling (not used during inference)
        self.similarity_weight = mx.array([10.0])
        self.similarity_bias = mx.array([-5.0])

    def __call__(self, mels: mx.array) -> mx.array:
        """
        Computes the embeddings of a batch of partial utterances.

        Args:
            mels: a batch of unscaled mel spectrograms as array of shape (B, T, M)
                  where T is hp.ve_partial_frames

        Returns:
            embeddings: array of shape (B, E) where E is hp.speaker_embed_size.
                       Embeddings are L2-normed and thus lay in the range [-1, 1].
        """
        if self.hp.normalized_mels:
            min_val = mels.min()
            max_val = mels.max()
            if min_val < 0 or max_val > 1:
                raise Exception(f"Mels outside [0, 1]. Min={min_val}, Max={max_val}")

        # Pass through stacked LSTM layers
        # MLX LSTM returns (output, (hidden_state, cell_state))
        # We only need the output for the next layer
        x, _ = self.lstm1(mels)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        # Get the final hidden state (last timestep)
        # x shape is (B, T, hidden_size), we want the last timestep
        final_hidden = x[:, -1, :]

        # Project to speaker embedding size
        raw_embeds = self.proj(final_hidden)

        if self.hp.ve_final_relu:
            raw_embeds = mx.maximum(raw_embeds, 0)

        # L2 normalize the embeddings
        norm = mx.linalg.norm(raw_embeds, axis=1, keepdims=True)
        return raw_embeds / (norm + 1e-8)

    def inference(
        self,
        mels: mx.array,
        mel_lens: List[int],
        overlap: float = 0.5,
        rate: float = None,
        min_coverage: float = 0.8,
        batch_size: int = None,
    ) -> mx.array:
        """
        Computes the embeddings of a batch of full utterances.

        Args:
            mels: (B, T, M) unscaled mels
            mel_lens: list of lengths for each mel

        Returns:
            (B, E) embeddings
        """
        # Compute where to split the utterances into partials
        frame_step = get_frame_step(overlap, rate, self.hp)
        n_partials_list = []
        target_lens = []
        for length in mel_lens:
            n, t = get_num_wins(length, frame_step, min_coverage, self.hp)
            n_partials_list.append(n)
            target_lens.append(t)

        # Possibly pad the mels to reach the target lengths
        mels_np = np.array(mels)
        len_diff = max(target_lens) - mels_np.shape[1]
        if len_diff > 0:
            pad = np.zeros(
                (mels_np.shape[0], len_diff, self.hp.num_mels), dtype=np.float32
            )
            mels_np = np.concatenate([mels_np, pad], axis=1)

        # Group all partials together
        partials = []
        for mel, n_partial in zip(mels_np, n_partials_list):
            for i in range(n_partial):
                start = i * frame_step
                end = start + self.hp.ve_partial_frames
                partials.append(mel[start:end])

        partials = mx.array(np.stack(partials))

        # Forward the partials
        batch_size = batch_size or len(partials)
        n_chunks = int(np.ceil(len(partials) / batch_size))
        partial_embeds = []
        for i in range(n_chunks):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(partials))
            batch = partials[start:end]
            embeds = self(batch)
            partial_embeds.append(embeds)
        partial_embeds = mx.concatenate(partial_embeds, axis=0)

        # Reduce the partial embeds into full embeds
        slices = np.concatenate(([0], np.cumsum(n_partials_list)))
        raw_embeds = []
        for start, end in zip(slices[:-1], slices[1:]):
            # Convert numpy integers to Python integers for MLX slicing
            mean_embed = mx.mean(partial_embeds[int(start) : int(end)], axis=0)
            raw_embeds.append(mean_embed)
        raw_embeds = mx.stack(raw_embeds)

        # L2 normalize
        norm = mx.linalg.norm(raw_embeds, axis=1, keepdims=True)
        embeds = raw_embeds / (norm + 1e-8)

        return embeds

    @staticmethod
    def utt_to_spk_embed(utt_embeds: np.ndarray) -> np.ndarray:
        """
        Takes an array of L2-normalized utterance embeddings, computes the mean
        embedding and L2-normalizes it to get a speaker embedding.
        """
        assert utt_embeds.ndim == 2
        utt_embeds = np.mean(utt_embeds, axis=0)
        return utt_embeds / np.linalg.norm(utt_embeds, 2)

    def embeds_from_mels(
        self,
        mels: Union[mx.array, List[np.ndarray]],
        mel_lens: Optional[List[int]] = None,
        as_spk: bool = False,
        batch_size: int = 32,
        **kwargs,
    ) -> np.ndarray:
        """
        Convenience function for deriving utterance or speaker embeddings from mel spectrograms.
        """
        # Load mels in memory and pack them
        if isinstance(mels, list):
            mels = [np.asarray(mel) for mel in mels]
            assert all(
                m.shape[1] == mels[0].shape[1] for m in mels
            ), "Mels aren't in (B, T, M) format"
            mel_lens = [mel.shape[0] for mel in mels]
            mels = pack(mels)

        # Embed them
        utt_embeds = self.inference(mels, mel_lens, batch_size=batch_size, **kwargs)
        utt_embeds = np.array(utt_embeds)

        return self.utt_to_spk_embed(utt_embeds) if as_spk else utt_embeds

    def embeds_from_wavs(
        self,
        wavs: List[np.ndarray],
        sample_rate: int,
        as_spk: bool = False,
        batch_size: int = 32,
        trim_top_db: Optional[float] = 20,
        **kwargs,
    ) -> np.ndarray:
        """
        Wrapper around embeds_from_mels that takes raw waveforms.
        """
        if sample_rate != self.hp.sample_rate:
            wavs = [
                librosa.resample(
                    wav,
                    orig_sr=sample_rate,
                    target_sr=self.hp.sample_rate,
                    res_type="kaiser_fast",
                )
                for wav in wavs
            ]

        if trim_top_db:
            wavs = [librosa.effects.trim(wav, top_db=trim_top_db)[0] for wav in wavs]

        if "rate" not in kwargs:
            kwargs["rate"] = 1.3  # Resemble's default value

        mels = [melspectrogram(w, self.hp).T for w in wavs]

        return self.embeds_from_mels(
            mels, as_spk=as_spk, batch_size=batch_size, **kwargs
        )
