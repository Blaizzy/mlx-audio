# Copyright (c) 2025 Resemble AI
# MIT License
# MLX port of S3Gen (Speech Token to Waveform)

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
        device: str = "auto",
    ) -> Dict[str, mx.array]:
        """
        Embed reference audio for speaker conditioning.

        Args:
            ref_wav: Reference waveform
            ref_sr: Sample rate

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

        # Placeholder for speaker embedding (would come from CAMPPlus in full implementation)
        # For now, use zero embedding
        ref_x_vector = mx.zeros((1, 192))

        # Placeholder for speech tokens (would come from S3Tokenizer)
        # In full implementation, this would tokenize the reference
        ref_speech_tokens = mx.zeros((1, ref_mels.shape[1] // 2), dtype=mx.int32)
        ref_speech_token_lens = mx.array([ref_speech_tokens.shape[1]])

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


# Alias for compatibility
S3Gen = S3Token2Wav
