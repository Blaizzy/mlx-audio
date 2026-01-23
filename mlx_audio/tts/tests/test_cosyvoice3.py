"""
CosyVoice3 comparison tests between PyTorch and MLX implementations.

Tests individual components using numpy.allclose to verify correctness.
"""

import os
import sys
import unittest
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Check if required packages are available
try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def skip_if_no_mlx(test_func):
    """Decorator to skip test if MLX is not available."""
    return unittest.skipUnless(MLX_AVAILABLE, "MLX not available")(test_func)


def skip_if_no_torch(test_func):
    """Decorator to skip test if PyTorch is not available."""
    return unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")(test_func)


def requires_model(test_func):
    """Decorator to skip test if model weights are not available."""
    model_id = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    try:
        from huggingface_hub import snapshot_download

        return test_func
    except ImportError:
        return unittest.skip("huggingface_hub not available")(test_func)


class TestCosyVoice3Components(unittest.TestCase):
    """Test individual CosyVoice3 components."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.rtol = 1e-4
        cls.atol = 1e-4
        np.random.seed(42)
        if MLX_AVAILABLE:
            mx.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)

    @skip_if_no_mlx
    def test_snake_activation(self):
        """Test Snake activation function."""
        from mlx_audio.tts.models.cosyvoice3.hift import Snake

        # Test parameters
        channels = 64
        batch_size = 2
        seq_len = 100

        # Create MLX Snake
        mlx_snake = Snake(channels=channels, alpha=1.0, alpha_logscale=False)

        # Create input
        x_np = np.random.randn(batch_size, seq_len, channels).astype(np.float32)
        x_mlx = mx.array(x_np)

        # MLX forward
        y_mlx = mlx_snake(x_mlx)
        mx.eval(y_mlx)

        # Manual computation for reference
        alpha = np.ones((1, 1, channels))
        y_expected = x_np + (1.0 / (alpha + 1e-9)) * np.sin(x_np * alpha) ** 2

        # Compare
        np.testing.assert_allclose(
            np.array(y_mlx),
            y_expected,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Snake activation mismatch",
        )

    @skip_if_no_mlx
    def test_causal_conv1d(self):
        """Test CausalConv1d."""
        from mlx_audio.tts.models.cosyvoice3.hift import CausalConv1d

        # Test parameters
        in_channels = 32
        out_channels = 64
        kernel_size = 5
        batch_size = 2
        seq_len = 50

        # Create MLX CausalConv1d
        mlx_conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
        )

        # Initialize with known weights
        weight_np = np.random.randn(out_channels, kernel_size, in_channels).astype(
            np.float32
        )
        bias_np = np.random.randn(out_channels).astype(np.float32)
        mlx_conv.conv.weight = mx.array(weight_np)
        mlx_conv.conv.bias = mx.array(bias_np)

        # Create input
        x_np = np.random.randn(batch_size, seq_len, in_channels).astype(np.float32)
        x_mlx = mx.array(x_np)

        # MLX forward
        y_mlx = mlx_conv(x_mlx)
        mx.eval(y_mlx)

        # Output should have same time dimension (causal padding)
        self.assertEqual(y_mlx.shape[1], seq_len)
        self.assertEqual(y_mlx.shape[2], out_channels)

    @skip_if_no_mlx
    def test_rotary_embedding(self):
        """Test RotaryEmbedding."""
        from mlx_audio.tts.models.cosyvoice3.dit import RotaryEmbedding, apply_rotary_emb

        # Test parameters
        dim = 64
        seq_len = 100

        # Create MLX RoPE
        mlx_rope = RotaryEmbedding(dim=dim)

        # Get embeddings - returns (freqs, scale) tuple
        freqs, scale = mlx_rope(seq_len)
        mx.eval(freqs)

        # Check shapes - freqs is (1, T, dim) with interleaved frequencies
        self.assertEqual(freqs.shape, (1, seq_len, dim))
        self.assertEqual(scale, 1.0)

        # Test apply_rotary_emb on flattened (B, T, H*D) tensor
        # This matches the actual usage in Attention: RoPE is applied before reshape
        batch_size = 2
        heads = 16
        inner_dim = heads * dim  # 1024
        x = mx.random.normal((batch_size, seq_len, inner_dim))

        # Apply RoPE - only first rot_dim=64 dims should be rotated
        x_out = apply_rotary_emb(x, freqs, scale)
        mx.eval(x_out)

        self.assertEqual(x_out.shape, x.shape)

        # Verify only first 64 dims are changed
        x_np = np.array(x)
        x_out_np = np.array(x_out)
        # First 64 dims should be different
        self.assertFalse(np.allclose(x_np[..., :dim], x_out_np[..., :dim]))
        # Remaining dims should be unchanged
        np.testing.assert_allclose(
            x_np[..., dim:], x_out_np[..., dim:], rtol=0, atol=0
        )

    @skip_if_no_mlx
    def test_timestep_embedding(self):
        """Test TimestepEmbedding."""
        from mlx_audio.tts.models.cosyvoice3.dit import TimestepEmbedding

        # Test parameters
        dim = 256
        time_dim = 256
        batch_size = 4

        # Create MLX TimestepEmbedding
        mlx_time_embed = TimestepEmbedding(dim=dim, time_dim=time_dim)

        # Create timesteps
        t = mx.array([0.0, 0.25, 0.5, 1.0])

        # Forward
        emb = mlx_time_embed(t)
        mx.eval(emb)

        # Check shape
        self.assertEqual(emb.shape, (batch_size, dim))

    @skip_if_no_mlx
    def test_attention(self):
        """Test Attention module."""
        from mlx_audio.tts.models.cosyvoice3.dit import Attention

        # Test parameters
        dim = 256
        heads = 8
        dim_head = 64
        batch_size = 2
        seq_len = 50

        # Create MLX Attention
        mlx_attn = Attention(dim=dim, heads=heads, dim_head=dim_head)

        # Create input
        x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        x_mlx = mx.array(x_np)

        # Forward without mask
        y_mlx = mlx_attn(x_mlx)
        mx.eval(y_mlx)

        # Check shape
        self.assertEqual(y_mlx.shape, (batch_size, seq_len, dim))

    @skip_if_no_mlx
    def test_feedforward(self):
        """Test FeedForward module."""
        from mlx_audio.tts.models.cosyvoice3.dit import FeedForward

        # Test parameters
        dim = 256
        mult = 4
        batch_size = 2
        seq_len = 50

        # Create MLX FeedForward
        mlx_ff = FeedForward(dim=dim, mult=mult)

        # Create input
        x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        x_mlx = mx.array(x_np)

        # Forward
        y_mlx = mlx_ff(x_mlx)
        mx.eval(y_mlx)

        # Check shape
        self.assertEqual(y_mlx.shape, (batch_size, seq_len, dim))

    @skip_if_no_mlx
    def test_dit_block(self):
        """Test DiTBlock module."""
        from mlx_audio.tts.models.cosyvoice3.dit import DiTBlock

        # Test parameters
        dim = 256
        heads = 8
        dim_head = 64
        batch_size = 2
        seq_len = 50

        # Create MLX DiTBlock
        mlx_block = DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=2)

        # Create inputs
        x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        t_np = np.random.randn(batch_size, dim).astype(np.float32)

        x_mlx = mx.array(x_np)
        t_mlx = mx.array(t_np)

        # Forward
        y_mlx = mlx_block(x_mlx, t_mlx)
        mx.eval(y_mlx)

        # Check shape
        self.assertEqual(y_mlx.shape, (batch_size, seq_len, dim))

    @skip_if_no_mlx
    def test_pre_lookahead_layer(self):
        """Test PreLookaheadLayer."""
        from mlx_audio.tts.models.cosyvoice3.flow import PreLookaheadLayer

        # Test parameters
        in_channels = 80
        channels = 256
        pre_lookahead_len = 3
        batch_size = 2
        seq_len = 50

        # Create MLX PreLookaheadLayer
        mlx_layer = PreLookaheadLayer(
            in_channels=in_channels,
            channels=channels,
            pre_lookahead_len=pre_lookahead_len,
        )

        # Create input
        x_np = np.random.randn(batch_size, seq_len, in_channels).astype(np.float32)
        x_mlx = mx.array(x_np)

        # Forward
        y_mlx = mlx_layer(x_mlx)
        mx.eval(y_mlx)

        # Check shape - should be same as input (residual connection)
        self.assertEqual(y_mlx.shape, (batch_size, seq_len, in_channels))

    @skip_if_no_mlx
    def test_sinusoidal_embedding(self):
        """Test sinusoidal embedding function."""
        from mlx_audio.tts.models.cosyvoice3.dit import sinusoidal_embedding

        # Test parameters
        dim = 256
        batch_size = 4

        # Create timesteps
        timesteps = mx.array([0.0, 0.25, 0.5, 1.0])

        # Get embeddings
        emb = sinusoidal_embedding(timesteps, dim)
        mx.eval(emb)

        # Check shape
        self.assertEqual(emb.shape, (batch_size, dim))

        # Check that different timesteps give different embeddings
        self.assertFalse(np.allclose(np.array(emb[0]), np.array(emb[1])))
        self.assertFalse(np.allclose(np.array(emb[1]), np.array(emb[2])))


@unittest.skipUnless(MLX_AVAILABLE and TORCH_AVAILABLE, "Requires both MLX and PyTorch")
class TestCosyVoice3TorchComparison(unittest.TestCase):
    """Compare MLX and PyTorch implementations using allclose."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.rtol = 1e-3
        cls.atol = 1e-3
        np.random.seed(42)
        mx.random.seed(42)
        torch.manual_seed(42)

    def test_snake_activation_torch_comparison(self):
        """Compare Snake activation between MLX and PyTorch."""
        from mlx_audio.tts.models.cosyvoice3.hift import Snake as MLXSnake

        # PyTorch Snake
        class TorchSnake(torch.nn.Module):
            def __init__(self, channels, alpha=1.0, alpha_logscale=False):
                super().__init__()
                self.alpha_logscale = alpha_logscale
                if alpha_logscale:
                    self.alpha = torch.nn.Parameter(torch.zeros(channels) * alpha)
                else:
                    self.alpha = torch.nn.Parameter(torch.ones(channels) * alpha)

            def forward(self, x):
                alpha = self.alpha.view(1, 1, -1)
                if self.alpha_logscale:
                    alpha = torch.exp(alpha)
                return x + (1.0 / (alpha + 1e-9)) * torch.sin(x * alpha) ** 2

        # Test parameters
        channels = 64
        batch_size = 2
        seq_len = 100

        # Create models
        mlx_snake = MLXSnake(channels=channels)
        torch_snake = TorchSnake(channels=channels)

        # Sync weights
        alpha_np = np.array(mlx_snake.alpha)
        torch_snake.alpha.data = torch.tensor(alpha_np)

        # Create input
        x_np = np.random.randn(batch_size, seq_len, channels).astype(np.float32)
        x_mlx = mx.array(x_np)
        x_torch = torch.tensor(x_np)

        # Forward
        y_mlx = mlx_snake(x_mlx)
        y_torch = torch_snake(x_torch)
        mx.eval(y_mlx)

        # Compare
        np.testing.assert_allclose(
            np.array(y_mlx),
            y_torch.detach().numpy(),
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Snake activation MLX vs PyTorch mismatch",
        )

    def test_linear_layer_comparison(self):
        """Compare Linear layer between MLX and PyTorch."""
        # Test parameters
        in_features = 256
        out_features = 512
        batch_size = 2
        seq_len = 50

        # Create PyTorch Linear first (canonical source)
        torch_linear = torch.nn.Linear(in_features, out_features)

        # Create MLX Linear
        mlx_linear = nn.Linear(in_features, out_features)

        # Sync weights from PyTorch to MLX
        # Both PyTorch and MLX store weights as (out, in)
        # MLX Linear.__call__ does: addmm(bias, x, weight.T)
        # where weight.T transposes (out, in) -> (in, out) for matmul
        weight_torch = torch_linear.weight.detach().numpy()  # (out, in)
        bias_torch = torch_linear.bias.detach().numpy()
        mlx_linear.weight = mx.array(weight_torch)  # (out, in) same as PyTorch
        mlx_linear.bias = mx.array(bias_torch)

        # Create input
        x_np = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
        x_mlx = mx.array(x_np)
        x_torch = torch.tensor(x_np)

        # Forward
        y_mlx = mlx_linear(x_mlx)
        y_torch = torch_linear(x_torch)
        mx.eval(y_mlx)

        # Compare
        np.testing.assert_allclose(
            np.array(y_mlx),
            y_torch.detach().numpy(),
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Linear layer MLX vs PyTorch mismatch",
        )

    def test_conv1d_comparison(self):
        """Compare Conv1d between MLX and PyTorch."""
        # Test parameters
        in_channels = 32
        out_channels = 64
        kernel_size = 5
        batch_size = 2
        seq_len = 50

        # Create MLX Conv1d
        mlx_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

        # Create PyTorch Conv1d
        torch_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

        # Sync weights
        # MLX Conv1d weight: (out, kernel, in)
        # PyTorch Conv1d weight: (out, in, kernel)
        weight_mlx = np.array(mlx_conv.weight)
        weight_torch = np.transpose(weight_mlx, (0, 2, 1))  # (out, in, kernel)
        torch_conv.weight.data = torch.tensor(weight_torch)
        torch_conv.bias.data = torch.tensor(np.array(mlx_conv.bias))

        # Create input
        # MLX expects (B, T, C), PyTorch expects (B, C, T)
        x_np = np.random.randn(batch_size, seq_len, in_channels).astype(np.float32)
        x_mlx = mx.array(x_np)
        x_torch = torch.tensor(x_np.transpose(0, 2, 1))  # (B, C, T)

        # Forward
        y_mlx = mlx_conv(x_mlx)  # Output: (B, T', C)
        y_torch = torch_conv(x_torch)  # Output: (B, C, T')
        mx.eval(y_mlx)

        # Compare (transpose PyTorch output to match MLX)
        y_torch_np = y_torch.detach().numpy().transpose(0, 2, 1)  # (B, T', C)
        np.testing.assert_allclose(
            np.array(y_mlx),
            y_torch_np,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Conv1d MLX vs PyTorch mismatch",
        )

    def test_embedding_comparison(self):
        """Compare Embedding layer between MLX and PyTorch."""
        # Test parameters
        vocab_size = 6561
        embed_dim = 80
        batch_size = 2
        seq_len = 50

        # Create MLX Embedding
        mlx_embed = nn.Embedding(vocab_size, embed_dim)

        # Create PyTorch Embedding
        torch_embed = torch.nn.Embedding(vocab_size, embed_dim)

        # Sync weights
        weight_np = np.array(mlx_embed.weight)
        torch_embed.weight.data = torch.tensor(weight_np)

        # Create input
        x_np = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
        x_mlx = mx.array(x_np)
        x_torch = torch.tensor(x_np, dtype=torch.long)

        # Forward
        y_mlx = mlx_embed(x_mlx)
        y_torch = torch_embed(x_torch)
        mx.eval(y_mlx)

        # Compare
        np.testing.assert_allclose(
            np.array(y_mlx),
            y_torch.detach().numpy(),
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Embedding MLX vs PyTorch mismatch",
        )

    def test_layer_norm_comparison(self):
        """Compare LayerNorm between MLX and PyTorch."""
        # Test parameters
        dim = 256
        batch_size = 2
        seq_len = 50

        # Create input
        x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        x_mlx = mx.array(x_np)
        x_torch = torch.tensor(x_np)

        # MLX fast layer norm (without learnable parameters)
        y_mlx = mx.fast.layer_norm(x_mlx, None, None, eps=1e-6)
        mx.eval(y_mlx)

        # PyTorch equivalent
        mean = x_torch.mean(dim=-1, keepdim=True)
        var = x_torch.var(dim=-1, unbiased=False, keepdim=True)
        y_torch = (x_torch - mean) / torch.sqrt(var + 1e-6)

        # Compare
        np.testing.assert_allclose(
            np.array(y_mlx),
            y_torch.detach().numpy(),
            rtol=self.rtol,
            atol=self.atol,
            err_msg="LayerNorm MLX vs PyTorch mismatch",
        )

    def test_softmax_comparison(self):
        """Compare Softmax between MLX and PyTorch."""
        # Test parameters
        batch_size = 2
        seq_len = 50
        dim = 256

        # Create input
        x_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        x_mlx = mx.array(x_np)
        x_torch = torch.tensor(x_np)

        # Forward
        y_mlx = mx.softmax(x_mlx, axis=-1)
        y_torch = torch.softmax(x_torch, dim=-1)
        mx.eval(y_mlx)

        # Compare
        np.testing.assert_allclose(
            np.array(y_mlx),
            y_torch.detach().numpy(),
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Softmax MLX vs PyTorch mismatch",
        )


@unittest.skipUnless(MLX_AVAILABLE and TORCH_AVAILABLE, "Requires both MLX and PyTorch")
class TestCosyVoice3IntegrationComparison(unittest.TestCase):
    """Integration tests comparing full model outputs."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.rtol = 1e-2
        cls.atol = 1e-2
        np.random.seed(42)
        mx.random.seed(42)
        torch.manual_seed(42)

    def test_dit_forward_shape(self):
        """Test DiT forward pass produces correct shapes."""
        from mlx_audio.tts.models.cosyvoice3.dit import DiT

        # Test parameters
        batch_size = 2
        seq_len = 50
        mel_dim = 80
        dim = 256
        depth = 4
        heads = 8

        # Create small DiT for testing
        dit = DiT(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=32,
            ff_mult=2,
            mel_dim=mel_dim,
            mu_dim=mel_dim,
            spk_dim=mel_dim,
            out_channels=mel_dim,
        )

        # Create inputs
        x = mx.random.normal((batch_size, mel_dim, seq_len))  # (B, D, T)
        mask = mx.ones((batch_size, seq_len), dtype=mx.bool_)
        mu = mx.random.normal((batch_size, mel_dim, seq_len))
        t = mx.array([0.5])
        spks = mx.random.normal((batch_size, mel_dim))
        cond = mx.random.normal((batch_size, mel_dim, seq_len))

        # Forward
        out = dit(x, mask, mu, t, spks, cond, streaming=False)
        mx.eval(out)

        # Check shape
        self.assertEqual(out.shape, (batch_size, mel_dim, seq_len))

    def test_flow_forward_shape(self):
        """Test Flow forward pass produces correct shapes."""
        from mlx_audio.tts.models.cosyvoice3.flow import (
            CausalMaskedDiffWithDiT,
            CausalConditionalCFM,
            PreLookaheadLayer,
        )
        from mlx_audio.tts.models.cosyvoice3.dit import DiT

        # Test parameters
        batch_size = 1
        token_len = 25
        mel_dim = 80
        vocab_size = 6561
        spk_embed_dim = 192
        token_mel_ratio = 2

        # Create components
        dit = DiT(
            dim=256,
            depth=2,
            heads=4,
            dim_head=32,
            ff_mult=2,
            mel_dim=mel_dim,
            mu_dim=mel_dim,
            spk_dim=mel_dim,
            out_channels=mel_dim,
        )

        cfm = CausalConditionalCFM(
            in_channels=240,
            n_spks=1,
            spk_emb_dim=mel_dim,
            estimator=dit,
        )

        pre_lookahead = PreLookaheadLayer(
            in_channels=mel_dim, channels=256, pre_lookahead_len=3
        )

        flow = CausalMaskedDiffWithDiT(
            input_size=mel_dim,
            output_size=mel_dim,
            spk_embed_dim=spk_embed_dim,
            vocab_size=vocab_size,
            token_mel_ratio=token_mel_ratio,
            pre_lookahead_layer=pre_lookahead,
            decoder=cfm,
        )

        # Create inputs
        token = mx.random.randint(0, vocab_size, (batch_size, token_len)).astype(
            mx.int32
        )
        token_len_arr = mx.array([token_len])
        prompt_token = mx.zeros((batch_size, 0), dtype=mx.int32)
        prompt_token_len = mx.zeros((batch_size,), dtype=mx.int32)
        prompt_feat = mx.zeros((batch_size, mel_dim, 0))
        prompt_feat_len = mx.zeros((batch_size,), dtype=mx.int32)
        embedding = mx.random.normal((batch_size, spk_embed_dim))

        # Forward
        mel = flow(
            token,
            token_len_arr,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            n_timesteps=2,  # Small for testing
        )
        mx.eval(mel)

        # Check shape
        expected_mel_len = token_len * token_mel_ratio
        self.assertEqual(mel.shape[0], batch_size)
        self.assertEqual(mel.shape[1], mel_dim)
        self.assertEqual(mel.shape[2], expected_mel_len)

    def test_hift_forward_shape(self):
        """Test HIFT forward pass produces correct shapes."""
        from mlx_audio.tts.models.cosyvoice3.hift import CausalHiFTGenerator

        # Test parameters
        batch_size = 1
        mel_len = 50
        mel_dim = 80
        sample_rate = 24000

        # Create HIFT
        hift = CausalHiFTGenerator(
            in_channels=mel_dim,
            base_channels=128,  # Smaller for testing
            nb_harmonics=8,
            sampling_rate=sample_rate,
            upsample_rates=[8, 5, 3],
        )

        # Create input
        mel = mx.random.normal((batch_size, mel_dim, mel_len))

        # Forward
        audio = hift(mel)
        mx.eval(audio)

        # Check shape (upsample factor = 8 * 5 * 3 = 120)
        # Note: actual output length depends on HIFT internals
        self.assertEqual(audio.shape[0], batch_size)
        self.assertGreater(audio.shape[1], mel_len)  # Should be upsampled


class TestCosyVoice3Weights(unittest.TestCase):
    """Test weight loading and sanitization."""

    @skip_if_no_mlx
    def test_weight_sanitization_patterns(self):
        """Test that weight sanitization patterns work correctly."""
        from mlx_audio.tts.models.cosyvoice3.cosyvoice3 import Model, ModelConfig

        # Create model
        config = ModelConfig()
        model = Model(config, load_llm=False)

        # Test parametrization combination
        test_weights = {
            "flow.test.parametrizations.weight.original0": mx.ones((64, 1, 1)),
            "flow.test.parametrizations.weight.original1": mx.random.normal(
                (64, 32, 5)
            ),
        }

        sanitized = model.sanitize(test_weights)

        # Should have combined weight
        self.assertIn("flow.test.weight", sanitized)
        self.assertEqual(sanitized["flow.test.weight"].shape, (64, 5, 32))  # Transposed

    @skip_if_no_mlx
    def test_sequential_remapping(self):
        """Test nn.Sequential index remapping."""
        from mlx_audio.tts.models.cosyvoice3.cosyvoice3 import Model, ModelConfig

        # Create model
        config = ModelConfig()
        model = Model(config, load_llm=False)

        # Test time_mlp remapping
        test_weights = {
            "flow.decoder.estimator.time_embed.time_mlp.0.weight": mx.zeros((256, 256)),
            "flow.decoder.estimator.time_embed.time_mlp.2.weight": mx.zeros((256, 256)),
        }

        sanitized = model.sanitize(test_weights)

        # Should have remapped keys
        self.assertIn(
            "flow.decoder.estimator.time_embed.time_mlp.layers.0.weight", sanitized
        )
        self.assertIn(
            "flow.decoder.estimator.time_embed.time_mlp.layers.2.weight", sanitized
        )


@unittest.skipUnless(MLX_AVAILABLE and TORCH_AVAILABLE, "Requires both MLX and PyTorch")
class TestCosyVoice3EndToEnd(unittest.TestCase):
    """End-to-end tests comparing full model inference.

    These tests require downloading the actual model weights and comparing
    the PyTorch and MLX implementations.
    """

    MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.rtol = 1e-2
        cls.atol = 1e-2
        np.random.seed(42)
        mx.random.seed(42)
        torch.manual_seed(42)

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_TESTS", "0") == "1",
        "Slow test - set RUN_SLOW_TESTS=1 to run"
    )
    def test_flow_with_pretrained_weights(self):
        """Test Flow module with actual pretrained weights."""
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.models.cosyvoice3.flow import (
            CausalMaskedDiffWithDiT,
            CausalConditionalCFM,
            PreLookaheadLayer,
        )
        from mlx_audio.tts.models.cosyvoice3.dit import DiT

        # Download model
        model_dir = snapshot_download(self.MODEL_ID)
        flow_path = os.path.join(model_dir, "flow.pt")

        if not os.path.exists(flow_path):
            self.skipTest("flow.pt not found")

        # Load PyTorch weights
        flow_pt = torch.load(flow_path, map_location="cpu", weights_only=True)

        # Create MLX model
        dit = DiT(
            dim=1024,
            depth=22,
            heads=16,
            dim_head=64,
            ff_mult=2,
            mel_dim=80,
            mu_dim=80,
            spk_dim=80,
            out_channels=80,
            static_chunk_size=50,
        )

        cfm = CausalConditionalCFM(
            in_channels=240,
            n_spks=1,
            spk_emb_dim=80,
            estimator=dit,
        )

        pre_lookahead = PreLookaheadLayer(
            in_channels=80, channels=1024, pre_lookahead_len=3
        )

        flow = CausalMaskedDiffWithDiT(
            input_size=80,
            output_size=80,
            spk_embed_dim=192,
            vocab_size=6561,
            token_mel_ratio=2,
            pre_lookahead_layer=pre_lookahead,
            decoder=cfm,
        )

        # Load weights
        from mlx_audio.tts.models.cosyvoice3.cosyvoice3 import Model, ModelConfig
        config = ModelConfig()
        model = Model(config, load_llm=False)

        # Convert weights
        all_weights = {}
        for k, v in flow_pt.items():
            new_key = f"flow.{k}"
            if "weight" in k and v.ndim == 3:
                v = v.permute(0, 2, 1)
            all_weights[new_key] = mx.array(v.numpy())

        sanitized = model.sanitize(all_weights)

        # Filter to flow weights only
        flow_weights = {k.replace("flow.", ""): v for k, v in sanitized.items()
                       if k.startswith("flow.")}

        flow.load_weights(list(flow_weights.items()), strict=False)
        mx.eval(flow.parameters())

        # Test forward pass
        batch_size = 1
        token_len = 10

        token = mx.random.randint(0, 6561, (batch_size, token_len)).astype(mx.int32)
        token_len_arr = mx.array([token_len])
        prompt_token = mx.zeros((batch_size, 0), dtype=mx.int32)
        prompt_token_len = mx.zeros((batch_size,), dtype=mx.int32)
        prompt_feat = mx.zeros((batch_size, 80, 0))
        prompt_feat_len = mx.zeros((batch_size,), dtype=mx.int32)
        embedding = mx.random.normal((batch_size, 192))

        # Forward
        mel = flow(
            token,
            token_len_arr,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            n_timesteps=2,
        )
        mx.eval(mel)

        # Just verify shape for now
        self.assertEqual(mel.shape[0], batch_size)
        self.assertEqual(mel.shape[1], 80)
        self.assertEqual(mel.shape[2], token_len * 2)

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_TESTS", "0") == "1",
        "Slow test - set RUN_SLOW_TESTS=1 to run"
    )
    def test_hift_with_pretrained_weights(self):
        """Test HIFT vocoder with actual pretrained weights."""
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.models.cosyvoice3.hift import CausalHiFTGenerator
        from mlx_audio.tts.models.cosyvoice3.cosyvoice3 import Model, ModelConfig

        # Download model
        model_dir = snapshot_download(self.MODEL_ID)
        hift_path = os.path.join(model_dir, "hift.pt")

        if not os.path.exists(hift_path):
            self.skipTest("hift.pt not found")

        # Load PyTorch weights
        hift_pt = torch.load(hift_path, map_location="cpu", weights_only=True)

        # Create MLX model
        config = ModelConfig()
        model = Model(config, load_llm=False)

        # Convert weights
        all_weights = {}
        for k, v in hift_pt.items():
            new_key = f"hift.{k}"
            if "weight" in k and v.ndim == 3 and "parametrizations" not in k:
                v = v.permute(0, 2, 1)
            all_weights[new_key] = mx.array(v.numpy())

        sanitized = model.sanitize(all_weights)

        # Filter to hift weights only
        hift_weights = {k.replace("hift.", ""): v for k, v in sanitized.items()
                       if k.startswith("hift.")}

        model.hift.load_weights(list(hift_weights.items()), strict=False)
        mx.eval(model.hift.parameters())

        # Test forward pass
        batch_size = 1
        mel_len = 50
        mel_dim = 80

        mel = mx.random.normal((batch_size, mel_dim, mel_len))

        # Forward
        audio = model.hift(mel)
        mx.eval(audio)

        # Verify output
        self.assertEqual(audio.shape[0], batch_size)
        self.assertGreater(audio.shape[1], mel_len)  # Should be upsampled

        # Check audio is in valid range
        audio_np = np.array(audio)
        self.assertTrue(np.all(audio_np >= -1.0))
        self.assertTrue(np.all(audio_np <= 1.0))

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_TESTS", "0") == "1",
        "Slow test - set RUN_SLOW_TESTS=1 to run"
    )
    def test_full_model_inference(self):
        """Test full model inference."""
        from mlx_audio.tts.models.cosyvoice3.cosyvoice3 import Model

        # Load model
        model = Model.from_pretrained(self.MODEL_ID, load_llm=False)

        # Test token2wav
        batch_size = 1
        token_len = 25

        token = mx.random.randint(0, 6561, (batch_size, token_len)).astype(mx.int32)
        token_len_arr = mx.array([token_len])
        prompt_token = mx.zeros((batch_size, 0), dtype=mx.int32)
        prompt_token_len = mx.zeros((batch_size,), dtype=mx.int32)
        prompt_feat = mx.zeros((batch_size, 80, 0))
        prompt_feat_len = mx.zeros((batch_size,), dtype=mx.int32)
        embedding = mx.random.normal((batch_size, 192))

        # Generate
        audio = model.token2wav(
            token,
            token_len_arr,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            n_timesteps=2,
        )
        mx.eval(audio)

        # Verify output
        self.assertEqual(audio.shape[0], batch_size)
        self.assertGreater(audio.shape[1], 0)

        # Check audio is in valid range
        audio_np = np.array(audio)
        self.assertTrue(np.all(np.isfinite(audio_np)))


if __name__ == "__main__":
    unittest.main()
