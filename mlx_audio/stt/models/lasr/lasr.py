
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_audio.stt.models.base import STTOutput
from .config import LasrEncoderConfig, ModelConfig

def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)

def _apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

class LasrEncoderRotaryEmbedding(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.config = config
        self.dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.base = config.rope_theta

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        # x shape: [batch, seq_len, num_heads, head_dim] or [seq_len, head_dim] depending on usage
        # We need seq_len
        seq_len = x.shape[1]
        
        # Create position indices
        indices = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        
        # Compute angles
        # indices: [seq_len], inv_freq: [dim/2]
        # output: [seq_len, dim/2]
        args = indices[:, None] * inv_freq[None, :]
        
        # Repeat for cos/sin to match dim
        # [seq_len, dim]
        args = mx.concatenate([args, args], axis=-1)
        
        cos = mx.cos(args)
        sin = mx.sin(args)
        
        # Reshape to broadcast: [1, seq_len, 1, dim] to match [batch, seq_len, num_heads, head_dim]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        
        return cos, sin

class LasrEncoderSubsampling(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.num_mel_bins, config.hidden_size)
        self.conv_0 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
        )
        self.conv_1 = nn.Conv1d(
            config.hidden_size,
            config.subsampling_conv_channels,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
        )
        self.dense_1 = nn.Linear(config.subsampling_conv_channels, config.hidden_size)
        # Assuming ReLU as activation based on transformers implementation
        self.act_fn = nn.ReLU() 

    def __call__(self, input_features: mx.array) -> mx.array:
        hidden_states = self.act_fn(self.dense_0(input_features))
        # No transpose needed as MLX Conv1d expects (N, L, C) if we confirm MLX conventions
        # Wait, MLX Conv1d expects input (N, L, C) and weights (out_channels, kernel_size, in_channels)
        # Transformers Conv1d expects (N, C, L). The loaded weights will be (out, in, kernel)
        
        # The transformers implementation does:
        # dense_0(input_features) -> (B, L, H)
        # transpose(1, 2) -> (B, H, L)
        # conv_0 -> (B, H, L)
        # conv_1 -> (B, C, L)
        # transpose(1, 2) -> (B, L, C)
        # dense_1 -> (B, L, H)
        
        # In MLX, we can keep (B, L, C) if we handle weights correctly or transpose if needed.
        # But MLX Conv1d expects (N, L, Input_Channels).
        # So we don't need to transpose to (N, C, L) like PyTorch.
        # We just need to make sure kernel size and striding works on length dimension.
        
        hidden_states = self.act_fn(self.conv_0(hidden_states))
        hidden_states = self.act_fn(self.conv_1(hidden_states))
        return self.dense_1(hidden_states)

class LasrEncoderAttention(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        
        # Handle GQA/MQA if configured, but default config suggests standard MHA
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.scaling = self.head_dim**-0.5
        
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim)
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim)
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Ensure cos/sin broadcast correctly
            # cos shape [1, L, 1, D]
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)
            
        q = q.transpose(0, 2, 1, 3) # [B, n_heads, L, D]
        k = k.transpose(0, 2, 1, 3) # [B, n_kv_heads, L, D]
        v = v.transpose(0, 2, 1, 3) # [B, n_kv_heads, L, D]

        if self.num_key_value_groups > 1:
            k = mx.repeat(k, self.num_key_value_groups, axis=1)
            v = mx.repeat(v, self.num_key_value_groups, axis=1)
            
        # Attention
        w = (q @ k.transpose(0, 1, 3, 2)) * self.scaling
        if mask is not None:
            # mask expected shape broadcastable to [B, n_heads, L, L]
            w = w + mask
            
        w = mx.softmax(w, axis=-1)
        o = w @ v
        
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o)

class LasrEncoderConvolutionModule(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        channels = config.hidden_size
        kernel_size = config.conv_kernel_size
        
         # Activation
        self.activation = nn.SiLU() if config.hidden_act == "silu" else nn.ReLU() # Simplification
        
        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=config.convolution_bias
        )
        
        # Depthwise conv in MLX: separate implementation or use Conv1d with grouping if supported perfectly.
        # Standard MLX Conv1d does not support groups!=1 efficiently in all versions/backends yet similar to torch?
        # But checking MLX docs, it does support groups.
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=0, # Manual padding in __call__
            groups=channels,
            bias=config.convolution_bias,
        )
        self.kernel_size = kernel_size
        
        self.norm = nn.BatchNorm(config.hidden_size, momentum=config.batch_norm_momentum)
        
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=config.convolution_bias
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # Input (B, L, C)
        
        
        # Pointwise 1
        hidden_states = self.pointwise_conv1(hidden_states)
        
        # GLU: split last dim
        act_size = hidden_states.shape[-1] // 2
        hidden_states = hidden_states[..., :act_size] * mx.sigmoid(hidden_states[..., act_size:])
        
        # Depthwise
        # Manual asymmetric padding for "same" convolution
        # Left: (K-1)//2, Right: K-1 - Left
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        
        # MLX pad expects list of (low, high) for each dim
        # Input (N, L, C). We pad dim 1.
        hidden_states = mx.pad(hidden_states, ((0, 0), (pad_left, pad_right), (0, 0)))
        
        hidden_states = self.depthwise_conv(hidden_states) 
        
        hidden_states = self.norm(hidden_states)
        
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        
        return hidden_states

class LasrEncoderFeedForward(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.activation = nn.SiLU() if config.hidden_act == "silu" else nn.ReLU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.activation(self.linear1(hidden_states))
        hidden_states = self.linear2(hidden_states)
        return hidden_states

class LasrEncoderBlock(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.feed_forward1 = LasrEncoderFeedForward(config)
        self.self_attn = LasrEncoderAttention(config)
        self.conv = LasrEncoderConvolutionModule(config)
        self.feed_forward2 = LasrEncoderFeedForward(config)
        
        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_self_att = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_conv = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.feed_forward_residual_weights = config.feed_forward_residual_weights
        self.conv_residual_weights = config.conv_residual_weights

    def __call__(
        self, 
        hidden_states: mx.array,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        # FF1
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = (
            self.feed_forward_residual_weights[0] * residual + self.feed_forward_residual_weights[1] * hidden_states
        )
        
        # Self Attn
        normalized_hidden_states = self.norm_self_att(hidden_states)
        attn_output = self.self_attn(
            normalized_hidden_states,
            position_embeddings=position_embeddings,
            mask=mask
        )
        hidden_states = hidden_states + attn_output
        
        # Conv
        conv_output = self.conv(self.norm_conv(hidden_states))
        hidden_states = (
             self.conv_residual_weights[0] * hidden_states + self.conv_residual_weights[1] * conv_output
        )
        
        # FF2
        residual = hidden_states
        hidden_states = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = (
            self.feed_forward_residual_weights[0] * residual + self.feed_forward_residual_weights[1] * hidden_states
        )
        
        return self.norm_out(hidden_states)

class LasrEncoder(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.config = config
        self.subsampler = LasrEncoderSubsampling(config)
        self.rotary_emb = LasrEncoderRotaryEmbedding(config)
        self.layers = [LasrEncoderBlock(config) for _ in range(config.num_hidden_layers)]
        self.out_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_features: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        hidden_states = self.subsampler(input_features)
        
        # Positional Embeddings
        cos, sin = self.rotary_emb(hidden_states)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=(cos, sin), mask=mask)
            
        return self.out_norm(hidden_states)

class LasrForCTC(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = LasrEncoder(config.encoder_config)
        self.ctc_head = nn.Linear(config.encoder_config.hidden_size, config.vocab_size)

    def __call__(self, input_features: mx.array) -> mx.array:
        hidden_states = self.encoder(input_features)
        logits = self.ctc_head(hidden_states)
        return logits
    
    def decode(self, input_features: mx.array) -> STTOutput:
        logits = self(input_features)
        logprobs = nn.log_softmax(logits, axis=-1)
        
        # Greedy decode
        tokens = mx.argmax(logprobs, axis=-1)
        
        # Decode tokens to text (Requires tokenizer)
        # This is strictly model implementation; decoding happens outside or via utility
        # Returning raw tokens or implementing minimal decoding if we have tokenizer access
        
        return STTOutput(text="", tokens=tokens) # Placeholder
