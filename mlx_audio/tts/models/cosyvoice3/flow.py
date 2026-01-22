"""
CosyVoice3 Flow module for mel-spectrogram generation.

Based on: https://github.com/FunAudioLLM/CosyVoice
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .dit import DiT


class PreLookaheadLayer(nn.Module):
    """Pre-lookahead layer for conditioning."""

    def __init__(
        self, in_channels: int = 80, channels: int = 1024, pre_lookahead_len: int = 3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len

        self.conv1 = nn.Conv1d(
            in_channels,
            channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, in_channels, kernel_size=3, stride=1, padding=0
        )

    def __call__(
        self, inputs: mx.array, context: Optional[mx.array] = None
    ) -> mx.array:
        """
        Args:
            inputs: (B, T, D)
            context: Optional context for inference (B, pre_lookahead_len, D)

        Returns:
            Output with residual connection (B, T, D)
        """
        # MLX Conv1d expects input as (B, T, D) - no transpose needed!
        outputs = inputs

        # Lookahead padding - pad on the time axis (axis 1)
        if context is None or context.size == 0:
            # Pad with zeros
            outputs = mx.pad(outputs, [(0, 0), (0, self.pre_lookahead_len), (0, 0)])
        else:
            # Use provided context
            outputs = mx.concatenate([outputs, context], axis=1)
            remaining_pad = self.pre_lookahead_len - context.shape[1]
            if remaining_pad > 0:
                outputs = mx.pad(outputs, [(0, 0), (0, remaining_pad), (0, 0)])

        # First conv with LeakyReLU (PyTorch default negative_slope=0.01)
        outputs = nn.leaky_relu(self.conv1(outputs), negative_slope=0.01)

        # Causal padding for second conv - pad on time axis
        outputs = mx.pad(outputs, [(0, 0), (2, 0), (0, 0)])  # kernel_size - 1
        outputs = self.conv2(outputs)

        # Residual connection
        return outputs + inputs


class CausalConditionalCFM(nn.Module):
    """
    Causal Conditional Flow Matching module.

    Implements flow matching for mel-spectrogram generation with
    classifier-free guidance support.
    """

    def __init__(
        self,
        in_channels: int = 240,
        n_spks: int = 1,
        spk_emb_dim: int = 80,
        sigma_min: float = 1e-6,
        solver: str = "euler",
        t_scheduler: str = "cosine",
        training_cfg_rate: float = 0.2,
        inference_cfg_rate: float = 0.7,
        estimator: Optional[DiT] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = sigma_min
        self.solver = solver
        self.t_scheduler = t_scheduler
        self.training_cfg_rate = training_cfg_rate
        self.inference_cfg_rate = inference_cfg_rate

        # The DiT estimator
        self.estimator = estimator

        # Fixed random noise for reproducibility (matching PyTorch)
        # PyTorch: set_all_random_seed(0); self.rand_noise = torch.randn([1, 80, 50 * 300])
        # We need to use the exact same random noise as PyTorch
        # This is generated once with torch.manual_seed(0); torch.randn([1, 80, 50*300])
        self._rand_noise = None  # Will be initialized lazily or loaded

    def _init_rand_noise(self):
        """Initialize random noise matching PyTorch's rand_noise."""
        import numpy as np
        import os

        # Try to load from cached file first
        cache_path = "/tmp/pt_rand_noise_full.npy"
        if os.path.exists(cache_path):
            rand_noise_np = np.load(cache_path)
            self._rand_noise = mx.array(rand_noise_np)
            return

        # Otherwise, generate using torch if available (for exact match)
        try:
            import torch

            torch.manual_seed(0)
            rand_noise = torch.randn([1, 80, 50 * 300])
            self._rand_noise = mx.array(rand_noise.numpy())
            # Save for future use
            np.save(cache_path, rand_noise.numpy())
        except ImportError:
            # Fallback to MLX random (won't match PyTorch exactly)
            mx.random.seed(0)
            self._rand_noise = mx.random.normal((1, 80, 50 * 300))

    def __call__(
        self,
        mu: mx.array,
        mask: mx.array,
        n_timesteps: int,
        spks: mx.array,
        cond: mx.array,
        streaming: bool = False,
        temperature: float = 1.0,
    ) -> mx.array:
        """
        Generate mel-spectrogram using flow matching.

        Args:
            mu: Mean from token embedding (B, mel_dim, T)
            mask: Mask for valid positions (B, T)
            n_timesteps: Number of ODE steps
            spks: Speaker embedding (B, spk_dim)
            cond: Condition mel (B, mel_dim, T)
            streaming: Whether to use streaming mode
            temperature: Sampling temperature

        Returns:
            Generated mel-spectrogram (B, mel_dim, T)
        """
        B, D, T = mu.shape

        # Initialize random noise lazily if not already done
        if self._rand_noise is None:
            self._init_rand_noise()

        # Use fixed random noise (matching PyTorch CausalConditionalCFM)
        # This ensures reproducible outputs
        z = self._rand_noise[:, :, :T]
        if B > 1:
            z = mx.broadcast_to(z, (B, D, T))
        z = z * temperature

        # Time schedule
        if self.t_scheduler == "cosine":
            t_span = 1 - mx.cos(mx.linspace(0, math.pi / 2, n_timesteps + 1))
        else:
            t_span = mx.linspace(0, 1, n_timesteps + 1)

        return self.solve_euler(z, t_span, mu, mask, spks, cond, streaming)

    def solve_euler(
        self,
        x: mx.array,
        t_span: mx.array,
        mu: mx.array,
        mask: mx.array,
        spks: mx.array,
        cond: mx.array,
        streaming: bool = False,
    ) -> mx.array:
        """
        Solve ODE using Euler method.

        Args:
            x: Initial noise (B, mel_dim, T)
            t_span: Time steps
            mu: Mean (B, mel_dim, T)
            mask: Mask (B, T)
            spks: Speaker embedding (B, spk_dim)
            cond: Condition (B, mel_dim, T)
            streaming: Whether to use streaming

        Returns:
            Generated mel-spectrogram (B, mel_dim, T)
        """
        n_steps = len(t_span) - 1

        for i in range(n_steps):
            t = t_span[i]
            dt = t_span[i + 1] - t_span[i]

            # Get velocity prediction with CFG
            dphi_dt = self.forward_estimator_cfg(x, mask, mu, t, spks, cond, streaming)

            # Euler step
            x = x + dt * dphi_dt
            mx.eval(x)

        return x

    def forward_estimator_cfg(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: mx.array,
        cond: mx.array,
        streaming: bool = False,
    ) -> mx.array:
        """
        Forward pass with classifier-free guidance.

        Args:
            x: Current state (B, mel_dim, T)
            mask: Mask (B, T)
            mu: Mean (B, mel_dim, T)
            t: Current timestep
            spks: Speaker embedding (B, spk_dim)
            cond: Condition (B, mel_dim, T)
            streaming: Whether to use streaming

        Returns:
            Velocity prediction with CFG (B, mel_dim, T)
        """
        cfg_rate = self.inference_cfg_rate

        if cfg_rate > 0:
            # Duplicate inputs for CFG
            B = x.shape[0]

            x_double = mx.concatenate([x, x], axis=0)
            mask_double = mx.concatenate([mask, mask], axis=0)
            mu_double = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)

            if t.ndim == 0:
                t_double = t
            else:
                t_double = mx.concatenate([t, t], axis=0)

            spks_double = mx.concatenate([spks, mx.zeros_like(spks)], axis=0)
            cond_double = mx.concatenate([cond, mx.zeros_like(cond)], axis=0)

            # Forward through estimator
            out = self.estimator(
                x_double,
                mask_double,
                mu_double,
                t_double,
                spks_double,
                cond_double,
                streaming,
            )

            # Split and apply CFG
            # PyTorch formula: (1 + cfg_rate) * cond - cfg_rate * uncond
            # = cond + cfg_rate * (cond - uncond)
            out_cond, out_uncond = mx.split(out, 2, axis=0)
            out = (1.0 + cfg_rate) * out_cond - cfg_rate * out_uncond

            return out
        else:
            return self.estimator(x, mask, mu, t, spks, cond, streaming)


class CausalMaskedDiffWithDiT(nn.Module):
    """
    Causal Masked Diffusion with DiT for speech synthesis.

    This is the main flow module that combines token embedding,
    speaker embedding, and the flow matching decoder.
    """

    def __init__(
        self,
        input_size: int = 80,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        pre_lookahead_layer: Optional[PreLookaheadLayer] = None,
        decoder: Optional[CausalConditionalCFM] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.spk_embed_dim = spk_embed_dim
        self.output_type = output_type
        self.vocab_size = vocab_size
        self.input_frame_rate = input_frame_rate
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

        # Token embedding
        self.input_embedding = nn.Embedding(vocab_size, input_size)

        # Speaker embedding projection
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)

        # Pre-lookahead layer
        if pre_lookahead_layer is None:
            pre_lookahead_layer = PreLookaheadLayer(input_size, 1024, pre_lookahead_len)
        self.pre_lookahead_layer = pre_lookahead_layer

        # Flow matching decoder
        self.decoder = decoder

    def __call__(
        self,
        token: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: mx.array,
        prompt_feat: mx.array,
        prompt_feat_len: mx.array,
        embedding: mx.array,
        n_timesteps: int = 10,
        streaming: bool = False,
        temperature: float = 1.0,
    ) -> mx.array:
        """
        Generate mel-spectrogram from speech tokens.

        Args:
            token: Speech tokens (B, T)
            token_len: Token lengths (B,)
            prompt_token: Prompt tokens (B, T_prompt)
            prompt_token_len: Prompt token lengths (B,)
            prompt_feat: Prompt mel features (B, T_prompt * ratio, mel_dim) - PyTorch format
            prompt_feat_len: Prompt feature lengths (B,)
            embedding: Speaker embedding (B, spk_embed_dim)
            n_timesteps: Number of ODE steps
            streaming: Whether to use streaming
            temperature: Sampling temperature

        Returns:
            Generated mel-spectrogram (B, mel_dim, T * ratio)
        """
        # Normalize speaker embedding (matching PyTorch F.normalize)
        embedding_norm = mx.sqrt(mx.sum(embedding ** 2, axis=1, keepdims=True) + 1e-12)
        embedding = embedding / embedding_norm

        # Speaker embedding projection
        spks = self.spk_embed_affine_layer(embedding)  # (B, output_size)

        # Combine prompt and target tokens
        if prompt_token.size > 0:
            all_tokens = mx.concatenate([prompt_token, token], axis=1)
            total_token_len = prompt_token_len + token_len
        else:
            all_tokens = token
            total_token_len = token_len

        # Clamp tokens to valid range (matching PyTorch torch.clamp(token, min=0))
        all_tokens = mx.maximum(all_tokens, 0)

        # Create mask for token embedding (B, T, 1)
        B = all_tokens.shape[0]
        T = all_tokens.shape[1]
        token_mask = (mx.arange(T)[None, :] < total_token_len[:, None]).astype(mx.float32)
        token_mask = token_mask[:, :, None]  # (B, T, 1)

        # Token embedding with mask
        token_embed = self.input_embedding(all_tokens) * token_mask  # (B, T, D)

        # Apply pre-lookahead
        token_embed = self.pre_lookahead_layer(token_embed)

        # Upsample by token_mel_ratio using repeat_interleave equivalent
        # (matching PyTorch h.repeat_interleave(self.token_mel_ratio, dim=1))
        T_out = T * self.token_mel_ratio
        mu = self._upsample_repeat(token_embed, self.token_mel_ratio)  # (B, T_out, D)
        mu = mu.transpose(0, 2, 1)  # (B, D, T_out)

        # Create mask for mel generation
        mel_len = total_token_len * self.token_mel_ratio
        max_len = T_out
        mask = mx.arange(max_len)[None, :] < mel_len[:, None]  # (B, T_out)

        # Condition mel (use prompt_feat padded to full length)
        # prompt_feat is expected as (B, T, mel_dim) format (matching PyTorch)
        if prompt_feat.ndim == 3 and prompt_feat.size > 0:
            # prompt_feat is (B, T_prompt, mel_dim)
            B_feat, T_feat, D_feat = prompt_feat.shape

            # Copy prompt features
            T_copy = min(T_feat, T_out)
            # Create with numpy for easier indexing
            import numpy as np

            conds_np = np.zeros((B, T_out, self.output_size), dtype=np.float32)
            conds_np[:, :T_copy, :] = np.array(prompt_feat[:, :T_copy, :])
            conds_btd = mx.array(conds_np)
            # Transpose to (B, mel_dim, T) for decoder
            cond = conds_btd.transpose(0, 2, 1)
        else:
            cond = mx.zeros((B, self.output_size, T_out))

        # Generate through flow matching
        mel = self.decoder(mu, mask, n_timesteps, spks, cond, streaming, temperature)

        return mel

    def _upsample(self, x: mx.array, target_len: int) -> mx.array:
        """Upsample sequence using linear interpolation."""
        B, T, D = x.shape
        if T == target_len:
            return x

        # Create interpolation indices
        indices = mx.linspace(0, T - 1, target_len)
        idx_floor = mx.floor(indices).astype(mx.int32)
        idx_ceil = mx.minimum(idx_floor + 1, T - 1)
        weights = indices - idx_floor.astype(mx.float32)

        # Interpolate
        x_floor = x[:, idx_floor, :]
        x_ceil = x[:, idx_ceil, :]
        output = (
            x_floor * (1 - weights[None, :, None]) + x_ceil * weights[None, :, None]
        )

        return output

    def _upsample_repeat(self, x: mx.array, ratio: int) -> mx.array:
        """Upsample by repeating each element (matching PyTorch repeat_interleave)."""
        B, T, D = x.shape
        # Repeat each element along the time axis
        # Shape: (B, T, D) -> (B, T, 1, D) -> (B, T, ratio, D) -> (B, T*ratio, D)
        x = x[:, :, None, :]  # (B, T, 1, D)
        x = mx.broadcast_to(x, (B, T, ratio, D))  # (B, T, ratio, D)
        x = x.reshape(B, T * ratio, D)  # (B, T*ratio, D)
        return x

    def inference(
        self,
        token: mx.array,
        token_len: mx.array,
        prompt_token: mx.array,
        prompt_token_len: mx.array,
        prompt_feat: mx.array,
        prompt_feat_len: mx.array,
        embedding: mx.array,
        n_timesteps: int = 10,
        streaming: bool = False,
        temperature: float = 1.0,
    ) -> mx.array:
        """
        Inference wrapper for mel generation.

        Returns only the generated part (without prompt), matching PyTorch:
        feat = feat[:, :, mel_len1:]
        where mel_len1 = prompt_feat.shape[1]
        """
        mel = self(
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            n_timesteps,
            streaming,
            temperature,
        )

        # Extract only the generated part (matching PyTorch exactly)
        # PyTorch uses: mel_len1 = prompt_feat.shape[1]
        # This is the actual prompt mel length, not calculated from token length
        if prompt_feat.size > 0 and prompt_feat.ndim == 3:
            mel_len1 = prompt_feat.shape[1]  # (B, T, D) -> T is the mel length
        else:
            mel_len1 = 0

        # Return only the generated part: feat[:, :, mel_len1:]
        return mel[:, :, mel_len1:]
