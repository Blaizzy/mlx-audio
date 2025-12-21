# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import logging
from typing import Dict, Optional, Tuple

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .decoder import ConditionalDecoder
from .encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .hifigan import F0Predictor, HiFTGenerator
from .mel import mel_spectrogram
from .xvector import CAMPPlus

logger = logging.getLogger(__name__)

# Constants
S3GEN_SR = 24000  # Output sample rate
S3_SR = 16000  # Input tokenizer sample rate
S3GEN_SIL = 4299  # Silence token
SPEECH_VOCAB_SIZE = 6561


def drop_invalid_tokens(x: mx.array) -> mx.array:
    """Remove tokens outside valid vocabulary."""
    return x[x < SPEECH_VOCAB_SIZE]


class S3Token2Mel(nn.Module):
    """
    S3Gen's CFM decoder: maps S3 speech tokens to mel-spectrograms.
    """

    def __init__(self, meanflow: bool = False):
        super().__init__()
        self.meanflow = meanflow

        # Token embedding
        self.input_embedding = nn.Embedding(SPEECH_VOCAB_SIZE, 512)

        # Speaker encoder (CAMPPlus for x-vector extraction)
        self.speaker_encoder = CAMPPlus(
            feat_dim=80,
            embedding_size=192,
            growth_rate=32,
            bn_size=4,
            init_channels=128,
        )

        # Speaker embedding projection
        self.spk_embed_affine_layer = nn.Linear(192, 80)

        # Encoder
        self.encoder = UpsampleConformerEncoder(
            input_size=512,
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
        )
        self.encoder_proj = nn.Linear(512, 80)

        # Decoder (CFM estimator)
        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            meanflow=meanflow,
        )

        # Flow matching
        self.decoder = CausalConditionalCFM(
            in_channels=240,
            spk_emb_dim=80,
            sigma_min=1e-6,
            t_scheduler="cosine",
            inference_cfg_rate=0.7,
            estimator=estimator,
        )

        self.token_mel_ratio = 2
        self.pre_lookahead_len = 3

    def embed_ref(
        self,
        ref_wav: mx.array,
        ref_sr: int,
        ref_speech_tokens: Optional[mx.array] = None,
        ref_speech_token_lens: Optional[mx.array] = None,
        device: str = "auto",
    ) -> Dict[str, mx.array]:
        """
        Embed reference audio for speaker conditioning.

        Args:
            ref_wav: Reference waveform
            ref_sr: Sample rate
            ref_speech_tokens: Pre-computed speech tokens (optional)
            ref_speech_token_lens: Token lengths (optional)

        Returns:
            Dictionary with prompt tokens, features, and embedding
        """
        if isinstance(ref_wav, np.ndarray):
            ref_wav = mx.array(ref_wav)

        if ref_wav.ndim == 1:
            ref_wav = ref_wav[None, :]

        # Resample to 24kHz for mel extraction
        ref_wav_np = np.array(ref_wav[0])
        if ref_sr != S3GEN_SR:
            ref_wav_24k = librosa.resample(
                ref_wav_np, orig_sr=ref_sr, target_sr=S3GEN_SR
            )
        else:
            ref_wav_24k = ref_wav_np

        # Extract mel features
        ref_mels = mel_spectrogram(ref_wav_24k)
        ref_mels = ref_mels.transpose(0, 2, 1)  # (B, T, 80)

        # Use provided tokens or create placeholder
        if ref_speech_tokens is None:
            ref_speech_tokens = mx.zeros((1, ref_mels.shape[1] // 2), dtype=mx.int32)
            ref_speech_token_lens = mx.array([ref_speech_tokens.shape[1]])
        else:
            # Align tokens and mel lengths (mel = 2 * tokens)
            actual_token_len = ref_speech_tokens.shape[1]
            expected_token_len = ref_mels.shape[1] // 2

            if actual_token_len != expected_token_len:
                if actual_token_len < expected_token_len:
                    # Tokens shorter - truncate mel to match
                    expected_mel_len = 2 * actual_token_len
                    ref_mels = ref_mels[:, :expected_mel_len, :]
                else:
                    # Tokens longer - truncate tokens to match mel
                    ref_speech_tokens = ref_speech_tokens[:, :expected_token_len]
                    actual_token_len = expected_token_len

            ref_speech_token_lens = mx.array([actual_token_len])

        # Resample to 16kHz for speaker encoder
        if ref_sr != S3_SR:
            ref_wav_16k = librosa.resample(ref_wav_np, orig_sr=ref_sr, target_sr=S3_SR)
        else:
            ref_wav_16k = ref_wav_np

        # Extract speaker embedding using CAMPPlus
        ref_x_vector = self.speaker_encoder.inference([mx.array(ref_wav_16k)])
        mx.eval(ref_x_vector)

        return {
            "prompt_token": ref_speech_tokens,
            "prompt_token_len": ref_speech_token_lens,
            "prompt_feat": ref_mels,
            "prompt_feat_len": mx.array([ref_mels.shape[1]]),
            "embedding": ref_x_vector,
        }

    def __call__(
        self,
        speech_tokens: mx.array,
        ref_dict: Dict[str, mx.array],
        n_cfm_timesteps: Optional[int] = None,
        finalize: bool = True,
    ) -> mx.array:
        """
        Generate mel-spectrogram from speech tokens.

        Args:
            speech_tokens: Speech token IDs (B, T)
            ref_dict: Reference embedding dictionary
            n_cfm_timesteps: Number of CFM steps
            finalize: Whether this is the final chunk

        Returns:
            Mel-spectrogram (B, 80, T_mel)
        """
        B = speech_tokens.shape[0]

        # Get reference data
        prompt_token = ref_dict["prompt_token"]
        prompt_token_len = ref_dict["prompt_token_len"]
        prompt_feat = ref_dict["prompt_feat"]
        embedding = ref_dict["embedding"]

        # Broadcast reference data if needed
        if prompt_token.shape[0] != B:
            prompt_token = mx.broadcast_to(prompt_token, (B,) + prompt_token.shape[1:])
        if embedding.shape[0] != B:
            embedding = mx.broadcast_to(embedding, (B,) + embedding.shape[1:])
        if prompt_feat.shape[0] != B:
            prompt_feat = mx.broadcast_to(prompt_feat, (B,) + prompt_feat.shape[1:])

        # Speaker embedding projection
        embedding = embedding / (
            mx.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8
        )
        embedding = self.spk_embed_affine_layer(embedding)

        # Concatenate prompt and input tokens
        token_len = mx.array([speech_tokens.shape[1]] * B)
        token = mx.concatenate([prompt_token, speech_tokens], axis=1)
        token_len = prompt_token_len + token_len

        # Create mask
        max_len = token.shape[1]
        mask = mx.arange(max_len)[None, :] < token_len[:, None]
        mask = mask[:, :, None].astype(mx.float32)

        # Embed tokens
        token_emb = self.input_embedding(token.astype(mx.int32)) * mask

        # Encode
        h, h_masks = self.encoder(token_emb, token_len)

        # Handle non-finalized chunks
        if not finalize:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]

        h_lengths = mx.sum(h_masks[:, 0, :].astype(mx.int32), axis=-1)
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1
        h = self.encoder_proj(h)

        # Prepare conditioning
        zeros_padding = mx.zeros((B, mel_len2, 80))
        conds = mx.concatenate([prompt_feat, zeros_padding], axis=1)
        conds = conds.transpose(0, 2, 1)  # (B, 80, T)

        # Mask for decoder
        mask = mx.arange(h.shape[1])[None, :] < h_lengths[:, None]
        mask = mask[:, None, :].astype(mx.float32)

        # Default timesteps
        if n_cfm_timesteps is None:
            n_cfm_timesteps = 2 if self.meanflow else 10

        # Generate noise for meanflow
        noised_mels = None
        if self.meanflow:
            noised_mels = mx.random.normal((B, 80, speech_tokens.shape[1] * 2))

        # Flow matching
        feat, _ = self.decoder(
            mu=h.transpose(0, 2, 1),
            mask=mask,
            n_timesteps=n_cfm_timesteps,
            spks=embedding,
            cond=conds,
            noised_mels=noised_mels,
            meanflow=self.meanflow,
        )

        # Remove prompt portion
        feat = feat[:, :, mel_len1:]

        return feat


class S3Token2Wav(S3Token2Mel):
    """
    Full S3Gen: speech tokens to waveform.
    Combines token-to-mel (CFM) and mel-to-wav (HiFiGAN).
    """

    def __init__(self, meanflow: bool = False):
        super().__init__(meanflow)

        # HiFiGAN vocoder
        f0_predictor = F0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # Trim fade for artifact reduction
        n_trim = S3GEN_SR // 50  # 20ms
        trim_fade = np.zeros(2 * n_trim)
        trim_fade[n_trim:] = (np.cos(np.linspace(np.pi, 0, n_trim)) + 1) / 2
        self.trim_fade = mx.array(trim_fade.astype(np.float32))

    def _forward_with_noise(
        self,
        speech_tokens: mx.array,
        ref_dict: Dict[str, mx.array],
        n_cfm_timesteps: Optional[int] = None,
        finalize: bool = True,
        noised_mels: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Generate mel-spectrogram from speech tokens with provided noise.

        This is like __call__ but accepts pre-computed noise for streaming
        consistency across chunks.

        Args:
            speech_tokens: Speech token IDs (B, T)
            ref_dict: Reference embedding dictionary
            n_cfm_timesteps: Number of CFM steps
            finalize: Whether this is the final chunk
            noised_mels: Pre-computed noise for flow matching (for consistency)

        Returns:
            Mel-spectrogram (B, 80, T_mel)
        """
        B = speech_tokens.shape[0]

        # Get reference data
        prompt_token = ref_dict["prompt_token"]
        prompt_token_len = ref_dict["prompt_token_len"]
        prompt_feat = ref_dict["prompt_feat"]
        embedding = ref_dict["embedding"]

        # Broadcast reference data if needed
        if prompt_token.shape[0] != B:
            prompt_token = mx.broadcast_to(prompt_token, (B,) + prompt_token.shape[1:])
        if embedding.shape[0] != B:
            embedding = mx.broadcast_to(embedding, (B,) + embedding.shape[1:])
        if prompt_feat.shape[0] != B:
            prompt_feat = mx.broadcast_to(prompt_feat, (B,) + prompt_feat.shape[1:])

        # Speaker embedding projection
        embedding = embedding / (
            mx.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8
        )
        embedding = self.spk_embed_affine_layer(embedding)

        # Concatenate prompt and input tokens
        token_len = mx.array([speech_tokens.shape[1]] * B)
        token = mx.concatenate([prompt_token, speech_tokens], axis=1)
        token_len = prompt_token_len + token_len

        # Create mask
        max_len = token.shape[1]
        mask = mx.arange(max_len)[None, :] < token_len[:, None]
        mask = mask[:, :, None].astype(mx.float32)

        # Embed tokens
        token_emb = self.input_embedding(token.astype(mx.int32)) * mask

        # Encode
        h, h_masks = self.encoder(token_emb, token_len)

        # Handle non-finalized chunks
        if not finalize:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]

        h_lengths = mx.sum(h_masks[:, 0, :].astype(mx.int32), axis=-1)
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1
        h = self.encoder_proj(h)

        # Prepare conditioning
        zeros_padding = mx.zeros((B, mel_len2, 80))
        conds = mx.concatenate([prompt_feat, zeros_padding], axis=1)
        conds = conds.transpose(0, 2, 1)  # (B, 80, T)

        # Mask for decoder
        mask = mx.arange(h.shape[1])[None, :] < h_lengths[:, None]
        mask = mask[:, None, :].astype(mx.float32)

        # Default timesteps
        if n_cfm_timesteps is None:
            n_cfm_timesteps = 2 if self.meanflow else 10

        # Use provided noise or generate new (for non-streaming calls)
        if noised_mels is None and self.meanflow:
            noised_mels = mx.random.normal((B, 80, speech_tokens.shape[1] * 2))

        # Flow matching
        feat, _ = self.decoder(
            mu=h.transpose(0, 2, 1),
            mask=mask,
            n_timesteps=n_cfm_timesteps,
            spks=embedding,
            cond=conds,
            noised_mels=noised_mels,
            meanflow=self.meanflow,
        )

        # Remove prompt portion
        feat = feat[:, :, mel_len1:]

        return feat

    def inference(
        self,
        speech_tokens: mx.array,
        ref_dict: Optional[Dict[str, mx.array]] = None,
        ref_wav: Optional[mx.array] = None,
        ref_sr: Optional[int] = None,
        n_cfm_timesteps: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Full inference: speech tokens to waveform.

        Args:
            speech_tokens: Speech token IDs (B, T)
            ref_dict: Pre-computed reference embeddings
            ref_wav: Reference waveform (if ref_dict not provided)
            ref_sr: Reference sample rate
            n_cfm_timesteps: Number of CFM steps

        Returns:
            audio: Generated waveform (B, T_audio)
            source: Source signal for debugging
        """
        # Get reference dict
        if ref_dict is None:
            if ref_wav is None:
                raise ValueError("Must provide either ref_dict or ref_wav")
            ref_dict = self.embed_ref(ref_wav, ref_sr)

        # Default timesteps for meanflow
        if n_cfm_timesteps is None:
            n_cfm_timesteps = 2 if self.meanflow else 10

        # Generate mel
        output_mels = self(
            speech_tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,
        )

        # Vocoder
        output_mels = output_mels.transpose(0, 2, 1)  # (B, T, 80) for HiFiGAN
        output_wavs, output_sources = self.mel2wav.inference(output_mels, None)

        # Apply trim fade to reduce artifacts
        fade_len = len(self.trim_fade)
        if output_wavs.shape[1] >= fade_len:
            faded_start = output_wavs[:, :fade_len] * self.trim_fade
            output_wavs = mx.concatenate(
                [faded_start, output_wavs[:, fade_len:]], axis=1
            )

        return output_wavs, output_sources

    def inference_stream(
        self,
        speech_tokens: mx.array,
        ref_dict: Dict[str, mx.array],
        n_cfm_timesteps: Optional[int] = None,
        prev_audio_samples: int = 0,
        is_final: bool = False,
        cached_noise: Optional[mx.array] = None,
    ) -> Tuple[mx.array, int, Optional[mx.array]]:
        """
        Streaming inference: convert speech tokens to waveform for streaming.

        This method processes accumulated tokens and returns the new audio
        samples that weren't returned in previous chunks. It caches and extends
        the random noise to ensure audio consistency across chunks.

        Args:
            speech_tokens: All accumulated speech token IDs (B, T)
            ref_dict: Pre-computed reference embeddings
            n_cfm_timesteps: Number of CFM steps
            prev_audio_samples: Number of audio samples already returned
            is_final: Whether this is the final chunk
            cached_noise: Previously cached noise to reuse for consistency

        Returns:
            new_audio: New audio samples (B, T_new)
            total_samples: Total number of samples generated so far
            cached_noise: Extended noise cache for next chunk (None if is_final)
        """
        # Default timesteps for meanflow
        if n_cfm_timesteps is None:
            n_cfm_timesteps = 2 if self.meanflow else 10

        B = speech_tokens.shape[0]
        required_noise_len = speech_tokens.shape[1] * 2  # mel = 2 * tokens

        # Handle noise caching for consistent audio across chunks
        if self.meanflow:
            if cached_noise is None:
                # First chunk: generate new noise
                noised_mels = mx.random.normal((B, 80, required_noise_len))
            else:
                # Subsequent chunks: extend cached noise if needed
                current_noise_len = cached_noise.shape[2]
                if current_noise_len >= required_noise_len:
                    # Reuse existing noise (truncate if longer)
                    noised_mels = cached_noise[:, :, :required_noise_len]
                else:
                    # Extend noise: keep old noise, add new noise for new portion
                    new_noise_len = required_noise_len - current_noise_len
                    new_noise = mx.random.normal((B, 80, new_noise_len))
                    noised_mels = mx.concatenate([cached_noise, new_noise], axis=2)
        else:
            noised_mels = None

        # Generate mel from all accumulated tokens with consistent noise
        output_mels = self._forward_with_noise(
            speech_tokens,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=is_final,
            noised_mels=noised_mels,
        )

        # Vocoder
        output_mels = output_mels.transpose(0, 2, 1)  # (B, T, 80) for HiFiGAN
        output_wavs, _ = self.mel2wav.inference(output_mels, None)

        # Apply trim fade only on first chunk
        if prev_audio_samples == 0:
            fade_len = len(self.trim_fade)
            if output_wavs.shape[1] >= fade_len:
                faded_start = output_wavs[:, :fade_len] * self.trim_fade
                output_wavs = mx.concatenate(
                    [faded_start, output_wavs[:, fade_len:]], axis=1
                )

        total_samples = output_wavs.shape[1]

        # Return only new samples (samples after what we've already returned)
        if prev_audio_samples > 0 and prev_audio_samples < total_samples:
            new_audio = output_wavs[:, prev_audio_samples:]
        elif prev_audio_samples == 0:
            new_audio = output_wavs
        else:
            # No new samples
            new_audio = output_wavs[:, :0]  # Empty with correct shape

        # Return cached noise for next chunk (unless final)
        next_cached_noise = noised_mels if not is_final else None

        return new_audio, total_samples, next_cached_noise

    def sanitize(self, weights: dict) -> dict:
        """
        Sanitize PyTorch weights for MLX compatibility.

        Handles:
        - Conv weight transposition (PyTorch OIHW/OIK -> MLX OHWI/OKI)
        - BatchNorm running stats
        - CAMPPlus speaker encoder weights
        """
        new_weights = {}

        for key, value in weights.items():
            new_key = key

            # Skip num_batches_tracked
            if "num_batches_tracked" in key:
                continue

            # Handle speaker_encoder (CAMPPlus) weights
            if key.startswith("speaker_encoder."):
                # Use CAMPPlus sanitize
                spk_key = key.replace("speaker_encoder.", "")
                spk_weights = self.speaker_encoder.sanitize({spk_key: value})
                for sk, sv in spk_weights.items():
                    new_weights[f"speaker_encoder.{sk}"] = sv
                continue

            # Handle Conv weight transposition
            # Skip if weights are already in MLX format (from pre-converted models)
            if "weight" in key and value.ndim >= 3:
                if value.ndim == 4:
                    # Conv2d: (O, I, H, W) -> (O, H, W, I)
                    # PyTorch format: shape[2] == shape[3] (square kernel like 3x3, 1x1)
                    if value.shape[2] == value.shape[3]:
                        # PyTorch format (O, I, H, W) with H==W: transpose
                        value = mx.array(np.array(value).transpose(0, 2, 3, 1))
                    # else: already in MLX format, skip transpose
                elif value.ndim == 3:
                    # Conv1d: (O, I, K) -> (O, K, I)
                    # For pre-converted MLX models, Conv1d weights are already transposed.
                    # PyTorch format has kernel at end: (O, I, K) where K is typically small (1-7)
                    # Only transpose if last dim is a typical small kernel size (1-7)
                    if value.shape[2] <= 7 and value.shape[1] > value.shape[2]:
                        # PyTorch format with small kernel at end: transpose
                        value = mx.array(np.array(value).transpose(0, 2, 1))
                    # else: already in MLX format or ambiguous, skip transpose

            new_weights[new_key] = value

        return new_weights


# Alias for compatibility
S3Gen = S3Token2Wav
