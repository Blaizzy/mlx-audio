# Ported from https://github.com/resemble-ai/chatterbox

import math
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from scipy import signal

from .decoder import ConditionalDecoder
from .f0_predictor import ConvRNNF0Predictor
from .flow import CausalMaskedDiffWithXvec
from .flow_matching import CFM_PARAMS, CausalConditionalCFM
from .hifigan import HiFTGenerator
from .mel import mel_spectrogram
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .xvector import CAMPPlus


def resample_audio(audio: mx.array, orig_sr: int, target_sr: int) -> mx.array:
    """Resample audio using scipy (numpy required for scipy)."""
    if orig_sr == target_sr:
        return audio
    import numpy as np

    audio_np = np.array(audio)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio_np, up, down, padtype="edge")
    return mx.array(resampled.astype(np.float32))


# Constants
S3GEN_SR = 24000
S3_SR = 16000
SPEECH_VOCAB_SIZE = 6561


class S3Token2Mel(nn.Module):
    """
    S3Gen CFM decoder maps S3 speech tokens to mel-spectrograms.

    This is the flow matching component that converts speech tokens to mel spectrograms
    using a Conformer encoder and flow matching decoder with speaker conditioning.
    """

    def __init__(self):
        super().__init__()

        # Speaker encoder
        self.speaker_encoder = CAMPPlus()

        # Conformer encoder with upsampling
        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )

        # Flow matching decoder
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
            act_fn="gelu",
        )

        cfm_params = CFM_PARAMS
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=cfm_params,
            estimator=estimator,
        )

        # Integration wrapper
        self.flow = CausalMaskedDiffWithXvec(encoder=encoder, decoder=decoder)

    def embed_ref(
        self,
        ref_wav: mx.array,
        ref_sr: int,
        ref_speech_tokens: mx.array,
        ref_speech_token_lens: mx.array,
    ):
        """
        Embed reference audio for speaker conditioning.

        Args:
            ref_wav: Reference waveform (1, T) at ref_sr
            ref_sr: Reference sample rate
            ref_speech_tokens: Reference speech tokens (1, T_tok)
            ref_speech_token_lens: Reference token lengths (1,)

        Returns:
            Dictionary with prompt_token, prompt_token_len, prompt_feat, embedding
        """
        if len(ref_wav.shape) == 1:
            ref_wav = mx.expand_dims(ref_wav, 0)

        # Resample to 24kHz for mel extraction if needed
        if ref_sr == S3GEN_SR:
            ref_wav_24 = ref_wav
        else:
            ref_wav_24 = resample_audio(ref_wav.squeeze(0), ref_sr, S3GEN_SR)
            ref_wav_24 = mx.expand_dims(ref_wav_24, 0)

        # Extract mel features at 24kHz
        ref_mels_24 = mel_spectrogram(
            ref_wav_24,
            n_fft=1920,
            num_mels=80,
            sampling_rate=S3GEN_SR,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000,
            center=False,
        )
        ref_mels_24 = mx.transpose(ref_mels_24, [0, 2, 1])  # (B, T, D)

        # Speaker embedding (expects 16kHz audio)
        if ref_sr == S3_SR:
            ref_wav_16 = ref_wav
        else:
            ref_wav_16 = resample_audio(ref_wav.squeeze(0), ref_sr, S3_SR)
            ref_wav_16 = mx.expand_dims(ref_wav_16, 0)

        ref_x_vector = self.speaker_encoder.inference(ref_wav_16)

        # Make sure mel_len = 2 * stoken_len
        # The relationship should be: mel_frames = 2 * num_tokens
        # If they don't match, we need to align them.
        # Original PyTorch truncates TOKENS to match MEL: tokens = mel // 2
        # But if tokens are already shorter than mel // 2, we need to truncate MEL instead
        actual_token_len = ref_speech_tokens.shape[1]
        expected_token_len = ref_mels_24.shape[1] // 2

        if actual_token_len != expected_token_len:
            if actual_token_len < expected_token_len:
                # Tokens are shorter - truncate mel to match (mel = 2 * tokens)
                expected_mel_len = 2 * actual_token_len
                ref_mels_24 = ref_mels_24[:, :expected_mel_len, :]
            else:
                # Tokens are longer - truncate tokens to match mel
                ref_speech_tokens = ref_speech_tokens[:, :expected_token_len]
                actual_token_len = expected_token_len

        # Ensure token_len matches actual shape
        ref_speech_token_lens = mx.array([actual_token_len])

        return dict(
            prompt_token=ref_speech_tokens,
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=mx.array([ref_mels_24.shape[1]]),
            embedding=ref_x_vector,
        )

    def __call__(
        self,
        speech_tokens: mx.array,
        ref_dict: dict,
        finalize: bool = False,
    ):
        """
        Generate mel-spectrograms from S3 speech tokens.

        Args:
            speech_tokens: S3 speech tokens (1, T)
            ref_dict: Reference embeddings from embed_ref()
            finalize: Whether streaming is finished (if False, last 3 tokens ignored)

        Returns:
            output_mels: Generated mel-spectrograms (1, D, T')
        """
        if len(speech_tokens.shape) == 1:
            speech_tokens = mx.expand_dims(speech_tokens, 0)

        speech_token_lens = mx.array([speech_tokens.shape[1]])

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            **ref_dict,
        )

        return output_mels


class S3Token2Wav(S3Token2Mel):
    """
    Full S3Gen decoder: token-to-mel (CFM) + mel-to-waveform (HiFiGAN).

    This combines the flow matching decoder with the HiFi-GAN vocoder to
    generate high-quality waveforms from speech tokens.
    """

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitize PyTorch weights for MLX.

        Handles:
        - Speaker encoder routing through CAMPPlus.sanitize
        - Conv1d/Conv2d weight transposition (PyTorch: out, in, kernel -> MLX: out, kernel, in)
        - ConvTranspose1d weight transposition
        - Decoder block naming (down_blocks.0.0 -> down_blocks_0.resnet)
        - Transformer attention naming (attn1.to_q -> attn.query_proj)
        - FFN naming (ff.net.0.proj -> ff.layers.0)
        - Weight normalization handling (parametrizations.weight.original0/1)

        This method is idempotent - it checks shapes before transposing to support
        both PyTorch-format and pre-converted MLX-format weights.
        """
        import re

        from mlx.utils import tree_flatten

        new_weights = {}

        # Get expected shapes from model for idempotent transposition
        curr_weights = dict(tree_flatten(self.parameters()))

        # Separate speaker_encoder weights and route through CAMPPlus.sanitize
        speaker_weights = {}
        other_weights = {}
        for key, value in weights.items():
            if key.startswith("speaker_encoder."):
                speaker_weights[key[len("speaker_encoder.") :]] = value
            else:
                other_weights[key] = value

        # Sanitize speaker encoder weights
        if speaker_weights:
            sanitized_speaker = self.speaker_encoder.sanitize(speaker_weights)
            for k, v in sanitized_speaker.items():
                new_weights[f"speaker_encoder.{k}"] = v

        # First pass: collect weight normalization pairs (g and v)
        # Weight norm formula: w = g * v / ||v||
        wn_pairs = {}  # base_key -> {"g": tensor, "v": tensor}
        non_wn_weights = {}

        for key, value in other_weights.items():
            if "parametrizations.weight.original0" in key:
                base_key = key.replace(".parametrizations.weight.original0", ".weight")
                if base_key not in wn_pairs:
                    wn_pairs[base_key] = {}
                wn_pairs[base_key]["g"] = value
            elif "parametrizations.weight.original1" in key:
                base_key = key.replace(".parametrizations.weight.original1", ".weight")
                if base_key not in wn_pairs:
                    wn_pairs[base_key] = {}
                wn_pairs[base_key]["v"] = value
            else:
                non_wn_weights[key] = value

        # Merge weight-normalized weights
        for base_key, pair in wn_pairs.items():
            if "g" in pair and "v" in pair:
                g = pair["g"]  # magnitude, shape (out_ch, 1, 1)
                v = pair[
                    "v"
                ]  # weight, shape (out_ch, in_ch, kernel) or (out_ch, kernel, in_ch) for ConvT
                # Compute ||v|| over all dimensions except the first (output channels)
                # For 3D weights: norm over dims 1 and 2
                v_norm = mx.sqrt(
                    mx.sum(v * v, axis=tuple(range(1, v.ndim)), keepdims=True)
                )
                # Normalized weight: w = g * v / ||v||
                w = g * v / (v_norm + 1e-12)
                non_wn_weights[base_key] = w
            elif "v" in pair:
                # Only v available, use directly (shouldn't happen)
                non_wn_weights[base_key] = pair["v"]

        # Process remaining weights
        for key, value in non_wn_weights.items():
            new_key = key

            # Skip num_batches_tracked for BatchNorm
            if "num_batches_tracked" in key:
                continue

            # === Decoder block naming ===
            # down_blocks.X.0 -> down_blocks_X.resnet (first tuple element is resnet)
            new_key = re.sub(
                r"down_blocks\.(\d+)\.0\.", r"down_blocks_\1.resnet.", new_key
            )
            # down_blocks.X.1.Y -> down_blocks_X.transformer_Y (second tuple element is transformer list)
            new_key = re.sub(
                r"down_blocks\.(\d+)\.1\.(\d+)\.",
                r"down_blocks_\1.transformer_\2.",
                new_key,
            )
            # down_blocks.X.2 -> down_blocks_X.downsample (third tuple element is downsample)
            new_key = re.sub(
                r"down_blocks\.(\d+)\.2\.", r"down_blocks_\1.downsample.", new_key
            )

            # mid_blocks.X.0 -> mid_blocks_X.resnet
            new_key = re.sub(
                r"mid_blocks\.(\d+)\.0\.", r"mid_blocks_\1.resnet.", new_key
            )
            # mid_blocks.X.1.Y -> mid_blocks_X.transformer_Y
            new_key = re.sub(
                r"mid_blocks\.(\d+)\.1\.(\d+)\.",
                r"mid_blocks_\1.transformer_\2.",
                new_key,
            )

            # up_blocks.X.0 -> up_blocks_X.resnet
            new_key = re.sub(r"up_blocks\.(\d+)\.0\.", r"up_blocks_\1.resnet.", new_key)
            # up_blocks.X.1.Y -> up_blocks_X.transformer_Y
            new_key = re.sub(
                r"up_blocks\.(\d+)\.1\.(\d+)\.",
                r"up_blocks_\1.transformer_\2.",
                new_key,
            )
            # up_blocks.X.2 -> up_blocks_X.upsample
            new_key = re.sub(
                r"up_blocks\.(\d+)\.2\.", r"up_blocks_\1.upsample.", new_key
            )

            # === ResNet block naming (CausalBlock1D structure) ===
            # .block1.block.0. -> .block1.conv.conv. (CausalConv1d)
            new_key = re.sub(r"\.block1\.block\.0\.", r".block1.conv.conv.", new_key)
            # .block1.block.2. -> .block1.norm. (LayerNorm)
            new_key = re.sub(r"\.block1\.block\.2\.", r".block1.norm.", new_key)
            new_key = re.sub(r"\.block2\.block\.0\.", r".block2.conv.conv.", new_key)
            new_key = re.sub(r"\.block2\.block\.2\.", r".block2.norm.", new_key)
            # .mlp.1. -> .mlp_linear. (Linear in MLP)
            new_key = re.sub(r"\.mlp\.1\.", r".mlp_linear.", new_key)

            # === Transformer attention naming ===
            # PyTorch: attn1.to_q -> MLX: attn.query_proj (sanitized format)
            new_key = new_key.replace(".attn1.to_q.", ".attn.query_proj.")
            new_key = new_key.replace(".attn1.to_k.", ".attn.key_proj.")
            new_key = new_key.replace(".attn1.to_v.", ".attn.value_proj.")
            new_key = new_key.replace(".attn1.to_out.0.", ".attn.out_proj.")

            # === Transformer FFN naming ===
            new_key = new_key.replace(".ff.net.0.proj.", ".ff.layers.0.")
            new_key = new_key.replace(".ff.net.2.", ".ff.layers.1.")

            # === Downsample/Upsample naming ===
            # Downsample1D and Upsample1D wrap conv in self.conv
            new_key = re.sub(
                r"\.downsample\.(weight|bias)$", r".downsample.conv.\1", new_key
            )
            new_key = re.sub(
                r"\.upsample\.(weight|bias)$", r".upsample.conv.\1", new_key
            )

            # === Final block naming ===
            new_key = new_key.replace(
                ".final_block.block.0.", ".final_block.conv.conv."
            )
            new_key = new_key.replace(".final_block.block.2.", ".final_block.norm.")

            # === Encoder naming ===
            # .embed.out.0. -> .embed.linear. (first linear)
            new_key = re.sub(r"\.embed\.out\.0\.", r".embed.linear.", new_key)
            # .embed.out.1. -> .embed.norm. (layer norm)
            new_key = re.sub(r"\.embed\.out\.1\.", r".embed.norm.", new_key)
            # Same for up_embed
            new_key = re.sub(r"\.up_embed\.out\.0\.", r".up_embed.linear.", new_key)
            new_key = re.sub(r"\.up_embed\.out\.1\.", r".up_embed.norm.", new_key)

            # === HiFi-GAN F0 predictor naming (idempotent) ===
            # PyTorch Sequential indices: 0, 2, 4, 6, 8 -> MLX list indices: 0, 1, 2, 3, 4
            # Only apply if we detect PyTorch-style indices (8 or 6 present in weights)
            # This makes it idempotent - already-converted weights won't be touched
            # Use a function to map indices in a single step to avoid cascading replacements
            has_pytorch_indices = any(
                ".condnet.6." in k or ".condnet.8." in k for k in weights.keys()
            )
            if has_pytorch_indices:

                def remap_condnet_idx(match):
                    idx_map = {"0": "0", "2": "1", "4": "2", "6": "3", "8": "4"}
                    return f".condnet.{idx_map[match.group(1)]}."

                new_key = re.sub(r"\.condnet\.([02468])\.", remap_condnet_idx, new_key)
            # condnet.0 stays condnet.0

            # === Conv weight transposition (idempotent) ===
            # Only transpose if shape doesn't match expected MLX format
            if "weight" in new_key and value.ndim == 3:
                if (
                    new_key in curr_weights
                    and value.shape != curr_weights[new_key].shape
                ):
                    # Check if this is ConvTranspose (ups) or regular Conv
                    if ".ups." in new_key:
                        # ConvTranspose1d: PyTorch (in, out, kernel) -> MLX (out, kernel, in)
                        value = mx.transpose(value, (1, 2, 0))
                    else:
                        # Conv1d: PyTorch (out, in, kernel) -> MLX (out, kernel, in)
                        value = mx.swapaxes(value, 1, 2)
            elif "weight" in new_key and value.ndim == 4:
                # Conv2d: PyTorch (out, in, H, W) -> MLX (out, H, W, in)
                if (
                    new_key in curr_weights
                    and value.shape != curr_weights[new_key].shape
                ):
                    value = mx.transpose(value, (0, 2, 3, 1))

            new_weights[new_key] = value

        # Filter out keys that don't exist in the model
        def should_keep(k):
            if k in curr_weights:
                return True
            # Keep quantization-related keys
            if k.endswith(".scales") or k.endswith(".biases"):
                return True
            return False

        filtered_weights = {k: v for k, v in new_weights.items() if should_keep(k)}

        return filtered_weights

    def __init__(self):
        super().__init__()

        # F0 predictor for vocoder
        f0_predictor = ConvRNNF0Predictor()

        # HiFi-GAN vocoder
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # Fade-in window to reduce artifacts (20ms)
        n_trim = S3GEN_SR // 50
        trim_fade = mx.zeros(2 * n_trim)
        trim_fade_cos = (mx.cos(mx.linspace(mx.pi, 0, n_trim)) + 1) / 2
        trim_fade = mx.concatenate([trim_fade[:n_trim], trim_fade_cos])
        self.trim_fade = trim_fade

    def __call__(
        self,
        speech_tokens: mx.array,
        ref_dict: dict,
        finalize: bool = False,
    ):
        """
        Generate waveforms from S3 speech tokens.

        Args:
            speech_tokens: S3 speech tokens (1, T)
            ref_dict: Reference embeddings from embed_ref()
            finalize: Whether streaming is finished

        Returns:
            output_wavs: Generated waveforms (1, T_wav)
        """
        # Generate mel-spectrograms
        output_mels = super().__call__(
            speech_tokens, ref_dict=ref_dict, finalize=finalize
        )

        # Generate waveform from mels
        hift_cache_source = mx.zeros((1, 1, 0))
        output_wavs, _ = self.mel2wav.inference(
            speech_feat=output_mels, cache_source=hift_cache_source
        )

        # Fade in to reduce spillover artifacts
        fade_len = len(self.trim_fade)
        if output_wavs.shape[1] >= fade_len:
            output_wavs[:, :fade_len] *= self.trim_fade

        return output_wavs

    def flow_inference(
        self,
        speech_tokens: mx.array,
        ref_dict: dict,
        finalize: bool = False,
    ):
        """Run only the flow matching (token-to-mel) inference."""
        return super().__call__(speech_tokens, ref_dict=ref_dict, finalize=finalize)

    def hift_inference(
        self, speech_feat: mx.array, cache_source: Optional[mx.array] = None
    ):
        """Run only the HiFi-GAN (mel-to-wav) inference."""
        if cache_source is None:
            cache_source = mx.zeros((1, 1, 0))
        return self.mel2wav.inference(
            speech_feat=speech_feat, cache_source=cache_source
        )

    def inference(
        self,
        speech_tokens: mx.array,
        ref_dict: dict,
        cache_source: Optional[mx.array] = None,
        finalize: bool = True,
    ):
        """
        Full inference pipeline with separate flow and vocoder steps.

        Args:
            speech_tokens: S3 speech tokens (1, T)
            ref_dict: Reference embeddings from embed_ref()
            cache_source: Source cache for HiFi-GAN (for streaming)
            finalize: Whether streaming is finished

        Returns:
            output_wavs: Generated waveforms (1, T_wav)
            output_sources: Source features from HiFi-GAN
        """
        output_mels = self.flow_inference(
            speech_tokens, ref_dict=ref_dict, finalize=finalize
        )
        output_wavs, output_sources = self.hift_inference(output_mels, cache_source)

        # Fade in to reduce spillover artifacts
        fade_len = len(self.trim_fade)
        if output_wavs.shape[1] >= fade_len:
            output_wavs[:, :fade_len] *= self.trim_fade

        return output_wavs, output_sources
