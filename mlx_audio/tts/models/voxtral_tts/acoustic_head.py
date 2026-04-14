"""Voxtral TTS Flow-Matching Acoustic Transformer.

Generates acoustic codes from LLM hidden states using 8-step Euler flow matching
with classifier-free guidance (alpha=1.2).
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .common import FeedForward


@dataclass
class AcousticTransformerArgs:
    input_dim: int = 3072
    dim: int = 3072
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    use_biases: bool = False
    rope_theta: float = 10000.0
    sigma: float = 1e-5
    sigma_max: float = 1.0
    norm_eps: float = 1e-5

    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36

    n_denoising_steps: int = 8
    cfg_alpha: float = 1.2


class BidirectionalAttention(nn.Module):
    """Multi-head attention without positional encoding."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        use_biases: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=use_biases)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=use_biases)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=use_biases)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=use_biases)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = (
            self.wk(x)
            .reshape(B, T, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.wv(x)
            .reshape(B, T, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        if self.n_kv_heads < self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, repeat, axis=1)
            v = mx.repeat(v, repeat, axis=1)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(out)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, args: AcousticTransformerArgs):
        super().__init__()
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = BidirectionalAttention(
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            use_biases=args.use_biases,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            bias=args.use_biases,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding matching vllm-omni convention: (cos, sin) order."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        half = dim // 2
        # Store inv_freq as buffer (matches vllm-omni/ExecuTorch)
        self.inv_freq = mx.exp(
            -math.log(theta) * mx.arange(half).astype(mx.float32) / half
        )

    def __call__(self, t: mx.array) -> mx.array:
        # t: (B,) or (B, 1)
        if t.ndim == 1:
            t = t[:, None]
        t = t.astype(mx.float32)
        emb = t * self.inv_freq  # (B, half)
        return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


class FlowMatchingAudioTransformer(nn.Module):
    """Generates acoustic codes via flow matching with classifier-free guidance."""

    def __init__(self, args: AcousticTransformerArgs):
        super().__init__()
        self.args = args

        self.input_projection = nn.Linear(
            args.n_acoustic_codebook, args.dim, bias=False
        )
        self.llm_projection = nn.Linear(args.input_dim, args.dim, bias=False)
        self.time_projection = nn.Linear(args.dim, args.dim, bias=False)

        self._time_embedding = TimeEmbedding(args.dim)
        self.layers = [AcousticTransformerBlock(args) for _ in range(args.n_layers)]

        # Semantic output uses padding: (8192//128 + 1)*128 = 8320
        semantic_padded = (args.semantic_codebook_size // 128 + 1) * 128
        self.semantic_codebook_output = nn.Linear(args.dim, semantic_padded, bias=False)
        self.acoustic_codebook_output = nn.Linear(
            args.dim, args.n_acoustic_codebook, bias=False
        )

        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def _run_transformer(self, x: mx.array) -> mx.array:
        """Run input through all transformer layers + norm."""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def _predict_velocity(
        self,
        x_t: mx.array,
        t: mx.array,
        llm_output: mx.array,
    ) -> mx.array:
        """Predict velocity field for one Euler step."""
        time_emb = self.time_projection(self._time_embedding(t))
        llm_emb = self.llm_projection(llm_output)
        acoustic_emb = self.input_projection(x_t)

        x = mx.stack([acoustic_emb, time_emb, llm_emb], axis=1)  # (B, 3, dim)
        x = self._run_transformer(x)
        return self.acoustic_codebook_output(x[:, 0, :])

    def _predict_semantic(self, llm_output: mx.array) -> mx.array:
        """Predict semantic codebook index directly from LLM hidden state.

        Unlike acoustic codes (which use flow matching), semantic codes are
        predicted via direct argmax on the LLM hidden state — no transformer pass.
        """
        logits = self.semantic_codebook_output(llm_output).astype(
            mx.float32
        )  # (B, padded)
        # Mask padding positions
        logits = logits.at[:, self.args.semantic_codebook_size + 2 :].add(-1e9)
        # Mask empty_audio token (index 0)
        logits = logits.at[:, 0].add(-1e9)
        return mx.argmax(logits, axis=-1)

    def decode_one_frame(self, llm_output: mx.array) -> mx.array:
        """Generate one frame: semantic code (argmax) + acoustic codes (flow matching).

        Returns codes with special token offsets applied:
        - Semantic code: raw index (0..8191) + 2 special tokens = range [2, 8193]
        - Acoustic codes: FSQ index (0..20) + 2 special tokens = range [2, 22]
        """
        args = self.args
        B = llm_output.shape[0]
        N_SPECIAL = 2  # empty_audio, end_audio

        semantic_codes = self._predict_semantic(llm_output)  # (B,) in [0, 8319]

        # Euler integration with linspace(0, 1, n_steps) timesteps
        # Loop runs n_steps-1 iterations (steps 0..6 for 8 timesteps)
        x_t = mx.random.normal((B, args.n_acoustic_codebook)) * args.sigma_max
        n_steps = args.n_denoising_steps
        timesteps = [i / (n_steps - 1) for i in range(n_steps)]

        llm_uncond = mx.zeros_like(llm_output)
        for step in range(n_steps - 1):
            t_val = timesteps[step]
            dt = timesteps[step + 1] - t_val
            t = mx.full((B,), t_val)

            # Batch conditional and unconditional velocity predictions
            # into a single forward pass (B=2) through the acoustic transformer
            x_t_batch = mx.concatenate([x_t, x_t], axis=0)
            t_batch = mx.concatenate([t, t], axis=0)
            llm_batch = mx.concatenate([llm_output, llm_uncond], axis=0)
            v_both = self._predict_velocity(x_t_batch, t_batch, llm_batch)
            v_cond, v_uncond = v_both[:B], v_both[B:]

            v = args.cfg_alpha * v_cond + (1.0 - args.cfg_alpha) * v_uncond
            x_t = x_t + v * dt

        # Quantize to FSQ indices [0, codebook_size-1], then add special token offset
        x_t = mx.clip(x_t, -1.0, 1.0)
        acoustic_codes = (
            mx.clip(
                mx.round((x_t + 1.0) * (args.acoustic_codebook_size - 1) / 2.0),
                0,
                args.acoustic_codebook_size - 1,
            ).astype(mx.int32)
            + N_SPECIAL
        )  # offset past special tokens

        # Semantic codes already have offset from argmax over padded logits
        # (indices 0,1 are masked, so min value is 2)
        return mx.concatenate([semantic_codes[:, None], acoustic_codes], axis=-1)

    def forward_batch(self, llm_hidden_states: mx.array) -> mx.array:
        """Generate audio codes for a sequence of frames."""
        B, T, D = llm_hidden_states.shape
        all_codes = []
        for t in range(T):
            codes = self.decode_one_frame(llm_hidden_states[:, t, :])
            all_codes.append(codes[:, None, :])
        return mx.concatenate(all_codes, axis=1)
