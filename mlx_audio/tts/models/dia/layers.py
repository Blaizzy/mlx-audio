from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from einops.array_api import repeat

from .config import DiaConfig


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _str_to_dtype(dtype_str: str):
    # Allow None for default behavior
    if dtype_str is None or dtype_str.lower() == "none":
        return None
    if dtype_str == "float32":
        return mx.float32
    elif dtype_str == "float16":
        return mx.float16
    elif dtype_str == "bfloat16":
        return mx.bfloat16
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")


class DenseGeneral(nn.Module):
    def __init__(
        self,
        in_shapes: Tuple[int, ...],
        out_features: Tuple[int, ...],
        axis: Tuple[int, ...] = (-1,),
        dtype: Optional[mx.Dtype] = None,
        weight_dtype: Optional[mx.Dtype] = None,
    ):
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis
        self.dtype = dtype
        self.kernel_shape = self.in_shapes + self.out_features

        weight_type = weight_dtype if weight_dtype is not None else dtype
        self.weight = mx.zeros(self.kernel_shape, dtype=weight_type)

    def __call__(self, inputs: mx.array) -> mx.array:
        norm_axis = _normalize_axes(self.axis, inputs.ndim)
        kernel_contract_axes = tuple(range(len(norm_axis)))

        output = mx.tensordot(
            inputs,
            self.weight,
            axes=(norm_axis, kernel_contract_axes),
        )

        if self.dtype is not None and output.dtype != self.dtype:
            output = output.astype(self.dtype)

        return output


def get_activation_fn(activation_string: str) -> nn.Module:
    if activation_string == "gelu":
        return nn.GELU()
    elif activation_string == "relu":
        return nn.ReLU()
    elif activation_string == "silu" or activation_string == "swish":
        return nn.SiLU()
    elif activation_string == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation_string}")


class MlpBlock(nn.Module):
    def __init__(
        self,
        config: DiaConfig,
        embed_dim: int,
        intermediate_dim: int,
        dropout_rate: float,
        activations: List[str] = ["silu", "linear"],
        use_pre_norm: bool = False,
    ):
        super().__init__()
        self.use_pre_norm = use_pre_norm
        num_activations = len(activations)

        compute_dtype = _str_to_dtype(config.training.dtype)
        weight_dtype = _str_to_dtype(config.model.weight_dtype)
        self.dtype = compute_dtype

        if use_pre_norm:
            self.pre_norm = nn.RMSNorm(
                embed_dim,
                eps=config.model.normalization_layer_epsilon,
            )

        self.wi_fused = DenseGeneral(
            in_shapes=(embed_dim,),
            out_features=(
                num_activations,
                intermediate_dim,
            ),
            axis=(-1,),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )

        self.activation_fn_0 = get_activation_fn(activations[0])  # silu
        self.activation_fn_1 = get_activation_fn(activations[1])  # linear

        self.dropout = nn.Dropout(dropout_rate)

        self.wo = DenseGeneral(
            in_shapes=(intermediate_dim,),
            out_features=(embed_dim,),
            axis=(-1,),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )

    def __call__(self, x: mx.array, deterministic: bool = False) -> mx.array:
        if self.use_pre_norm and hasattr(self, "pre_norm"):
            x = self.pre_norm(x)

        fused_x = self.wi_fused(x)

        gate_input = fused_x[..., 0, :]
        up_input = fused_x[..., 1, :]

        gate = self.activation_fn_0(gate_input)
        up = self.activation_fn_1(up_input)
        hidden = mx.multiply(gate, up)

        if self.dtype is not None and self.dtype != hidden.dtype:
            hidden = hidden.astype(self.dtype)

        if not deterministic:
            hidden = self.dropout(hidden)

        output = self.wo(hidden)
        return output


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.dtype = dtype
        half_embedding_dim = embedding_dims // 2
        fraction = (2.0 * mx.arange(half_embedding_dim)) / embedding_dims

        self._timescale = (
            self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        )

    def __call__(self, inputs: mx.array, position: mx.array):
        """Applies RoPE."""
        position = mx.expand_dims(mx.expand_dims(position, -1), -1)

        sinusoid_inp = position / self._timescale

        sin = mx.sin(sinusoid_inp).astype(inputs.dtype)
        cos = mx.cos(sinusoid_inp).astype(inputs.dtype)

        first_half = inputs[..., : self.embedding_dims // 2]
        second_half = inputs[..., self.embedding_dims // 2 :]

        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin

        return mx.concatenate([first_part, second_part], axis=-1)


class KVCache:
    def __init__(self, num_heads, max_len, head_dim, k=None, v=None):
        self.k = mx.zeros((2, num_heads, max_len, head_dim)) if k is None else k
        self.v = mx.zeros((2, num_heads, max_len, head_dim)) if v is None else v
        self.current_idx = 0
        self.max_len = max_len

    def update_and_fetch(self, k, v):
        assert self.current_idx < self.max_len
        self.k[:, :, self.current_idx : self.current_idx + 1, :] = k
        self.v[:, :, self.current_idx : self.current_idx + 1, :] = v
        self.current_idx += 1
        return self.k[:, :, : self.current_idx, :], self.v[:, :, : self.current_idx, :]

    def prefill_kv(self, k, v):
        prefill_len = k.shape[2]
        assert prefill_len <= self.max_len
        self.k[:, :, :prefill_len, :] = k
        self.v[:, :, :prefill_len, :] = v
        self.current_idx = prefill_len


class Attention(nn.Module):
    def __init__(
        self,
        config: DiaConfig,
        q_embed_dim: int,
        kv_embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout_rate: float,
        is_cross_attn: bool = False,
        out_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_cross_attn = is_cross_attn
        self.dropout_rate = dropout_rate

        compute_dtype = _str_to_dtype(config.training.dtype)
        weight_dtype = _str_to_dtype(config.model.weight_dtype)

        self.output_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.projected_query_dim = num_query_heads * head_dim

        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.num_gqa_groups = num_query_heads // num_kv_heads

        # --- Projection Layers using DenseGeneral ---
        self.q_proj = DenseGeneral(
            in_shapes=(q_embed_dim,),
            out_features=(num_query_heads, head_dim),
            axis=(-1,),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )
        self.k_proj = DenseGeneral(
            in_shapes=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )
        self.v_proj = DenseGeneral(
            in_shapes=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )
        self.o_proj = DenseGeneral(
            in_shapes=(num_query_heads, head_dim),
            out_features=(self.output_dim,),
            axis=(-2, -1),
            dtype=compute_dtype,
            weight_dtype=weight_dtype,
        )

        # --- Rotary Embedding ---
        self.rotary_emb = RotaryEmbedding(
            embedding_dims=self.head_dim,
            min_timescale=config.model.rope_min_timescale,
            max_timescale=config.model.rope_max_timescale,
            dtype=compute_dtype,
        )

    def __call__(
        self,
        Xq: mx.array,  # (B, T, D) T = 1 in AR generation
        Xkv: mx.array,  # (B, S, E) S = 1 in AR generation
        q_positions: mx.array,  # (B, T)
        kv_positions: Optional[mx.array] = None,  # (B, S)
        deterministic: bool = True,
        attn_mask: Optional[
            mx.array
        ] = None,  # None in Decoder Self Attention, Valid mask in Others
        cache: Optional[KVCache] = None,  # None in Encoder, KVCache in Decoder
        prefill: bool = False,  # True only when prefilling KV Cache
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Performs attention calculation with optional KV caching.

        Args:
            Xq: Query tensor (B, T, D). T=1 during single-step decoding.
            Xkv: Key/Value source tensor (B, S, E). S=1 during single-step decoding for self-attn.
            q_positions: Positions for queries (B, T).
            kv_positions: Positions for keys/values (B, S). If None, uses q_positions.
            deterministic: If True, disable dropout.
            attn_mask: Attention mask.
            cache: KVCache.
            prefill: If True, use prefill mode.

        Returns:
            A tuple containing:
            - output: The attention output tensor (B, T, output_dim).
            - present_kv: The K/V state to be cached for the next step ((B, N, S_new, H), (B, N, S_new, H)).
              For self-attn, S_new = S_past + S. For cross-attn, S_new = S_kv.
        """
        if kv_positions is None:
            kv_positions = q_positions
        original_dtype = Xq.dtype

        Xq_BxTxNxH = self.q_proj(Xq)
        Xq_BxTxNxH = self.rotary_emb(Xq_BxTxNxH, position=q_positions)
        Xq_BxNxTxH = mx.transpose(Xq_BxTxNxH, (0, 2, 1, 3))

        # Input values into attention calculation
        attn_k = None
        attn_v = None

        # Decoder Cross Attention
        if self.is_cross_attn:
            # Directly use cache (no need to check index)
            attn_k, attn_v = cache.k, cache.v
            if (
                attn_k.shape[1] != self.num_query_heads
                or attn_v.shape[1] != self.num_query_heads
            ):
                raise ValueError(
                    f"Cross-attention cache head dimension ({attn_k.shape[1]}) "
                    f"does not match num_query_heads ({self.num_query_heads}). "
                    "Cache should be pre-repeated for GQA."
                )
        # Self Attention
        else:
            Xk_BxSxKxH = self.k_proj(Xkv)  # (B, S, K, H)
            Xv_BxSxKxH = self.v_proj(Xkv)  # (B, S, K, H)
            Xk_BxSxKxH = self.rotary_emb(
                Xk_BxSxKxH, position=kv_positions
            )  # (B, S, K, H)

            Xk_BxKxSxH = mx.transpose(Xk_BxSxKxH, (0, 2, 1, 3))  # (B, K, S, H)
            Xv_BxKxSxH = mx.transpose(Xv_BxSxKxH, (0, 2, 1, 3))  # (B, K, S, H)
            # S=1 for Decode Step

            if self.num_gqa_groups > 1:
                Xk_BxNxSxH = repeat(
                    Xk_BxKxSxH, "b k s h -> b (k g) s h", g=self.num_gqa_groups
                )
                Xv_BxNxSxH = repeat(
                    Xv_BxKxSxH, "b k s h -> b (k g) s h", g=self.num_gqa_groups
                )
            else:
                Xk_BxNxSxH = Xk_BxKxSxH
                Xv_BxNxSxH = Xv_BxKxSxH

            # Encoder Self Attention
            if cache is None:
                attn_k = Xk_BxNxSxH
                attn_v = Xv_BxNxSxH
            # Decoder Self Attention
            else:
                # In prefill mode, we fill in cache until prefill length
                if prefill:
                    attn_k, attn_v = Xk_BxNxSxH, Xv_BxNxSxH
                    cache.prefill_kv(attn_k, attn_v)
                # In decode step, we add current K/V to cache step by step
                else:
                    attn_k, attn_v = cache.update_and_fetch(Xk_BxNxSxH, Xv_BxNxSxH)

        # Attention Calculation
        attn_scores = mx.matmul(Xq_BxNxTxH, attn_k.swapaxes(2, 3))

        # Apply Scaling
        scale_factor = 1.0
        attn_scores = attn_scores * scale_factor

        # Apply Attention Mask
        if attn_mask is not None:
            # Add large negative value where mask is False/0
            attn_scores = mx.where(
                attn_mask, attn_scores, -1e9
            )  # Using -1e9 for numerical stability

        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_output = mx.matmul(attn_weights, attn_v)

        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))  # (B, T, N, H)
        output = self.o_proj(attn_output)

        if output.dtype != original_dtype:
            output = output.astype(original_dtype)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder
        embed_dim = enc_config.n_embd

        self.pre_sa_norm = nn.RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
        )

        self.self_attention = Attention(
            config=config,
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            num_query_heads=enc_config.n_head,
            num_kv_heads=enc_config.n_head,
            head_dim=enc_config.head_dim,
            dropout_rate=model_config.dropout,
            is_cross_attn=False,
            out_embed_dim=embed_dim,
        )

        self.post_sa_norm = nn.RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
        )

        self.mlp = MlpBlock(
            config=config,
            embed_dim=embed_dim,
            intermediate_dim=enc_config.n_hidden,
            activations=enc_config.mlp_activations,
            dropout_rate=model_config.dropout,
            use_pre_norm=enc_config.use_pre_norm,
        )

        self.dropout = nn.Dropout(model_config.dropout)

    def __call__(
        self,
        x: mx.array,
        src_positions: Optional[mx.array] = None,
        deterministic: bool = True,
        attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = x
        x_norm = self.pre_sa_norm(x)

        sa_out = self.self_attention(
            Xq=x_norm,
            Xkv=x_norm,
            q_positions=src_positions,
            kv_positions=src_positions,
            deterministic=deterministic,
            attn_mask=attn_mask,
        )
        x = residual + sa_out

        residual = x
        x_norm = self.post_sa_norm(x)
        mlp_out = self.mlp(x_norm, deterministic=deterministic)
        x = residual + mlp_out

        if not deterministic:
            x = self.dropout(x)

        return x


class Encoder(nn.Module):
    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder

        self.embedding = nn.Embedding(
            model_config.src_vocab_size,
            enc_config.n_embd,
        )
        self.dropout = nn.Dropout(model_config.dropout)
        self.layers = [EncoderLayer(config=config) for _ in range(enc_config.n_layer)]
        self.norm = nn.RMSNorm(
            enc_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
        )

    def __call__(
        self,
        x_ids: mx.array,
        src_positions: Optional[mx.array] = None,
        deterministic: bool = True,
        attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        x = self.embedding(x_ids)

        if not deterministic:
            x = self.dropout(x)

        for layer_index, layer in enumerate(self.layers):
            x = layer(
                x,
                src_positions=src_positions,
                deterministic=deterministic,
                attn_mask=attn_mask,
            )

        x = self.norm(x)

        if not deterministic:
            x = self.dropout(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        enc_config = config.model.encoder
        dec_embed_dim = dec_config.n_embd
        enc_embed_dim = enc_config.n_embd

        # Norms
        self.pre_sa_norm = nn.RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
        )
        self.pre_ca_norm = nn.RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
        )
        self.pre_mlp_norm = nn.RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
        )

        # Self-Attention (GQA) with Causal Masking
        self.self_attention = Attention(
            config=config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=dec_embed_dim,
            num_query_heads=dec_config.gqa_query_heads,
            num_kv_heads=dec_config.kv_heads,
            head_dim=dec_config.gqa_head_dim,
            dropout_rate=model_config.dropout,
            is_cross_attn=False,
            out_embed_dim=dec_embed_dim,
        )

        # Cross-Attention (MHA)
        self.cross_attention = Attention(
            config=config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=enc_embed_dim,  # Note kv_embed_dim
            num_query_heads=dec_config.cross_query_heads,
            num_kv_heads=dec_config.cross_query_heads,
            head_dim=dec_config.cross_head_dim,
            dropout_rate=model_config.dropout,
            is_cross_attn=True,
            out_embed_dim=dec_embed_dim,
        )

        # MLP
        self.mlp = MlpBlock(
            config=config,
            embed_dim=dec_embed_dim,
            intermediate_dim=dec_config.n_hidden,
            activations=dec_config.mlp_activations,
            dropout_rate=model_config.dropout,
            use_pre_norm=dec_config.use_pre_norm,
        )

    def __call__(
        self,
        x: mx.array,
        encoder_out: mx.array,
        tgt_positions: mx.array,
        src_positions: Optional[mx.array],
        deterministic: bool,
        self_attn_mask: mx.array,
        cross_attn_mask: mx.array,
        self_attn_cache: KVCache,
        cross_attn_cache: KVCache,
        prefill: bool = False,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # 1. Self-Attention
        residual = x
        x_norm = self.pre_sa_norm(x)

        sa_out = self.self_attention(
            Xq=x_norm,  # (2, 1, D)
            Xkv=x_norm,  # (2, 1, D)
            q_positions=tgt_positions,  # (2, 1)
            kv_positions=tgt_positions,  # (2, 1)
            deterministic=deterministic,
            attn_mask=self_attn_mask,  # (2, 1, 1, S_max)
            cache=self_attn_cache,
            prefill=prefill,
        )
        x = residual + sa_out

        # 2. Cross-Attention
        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out = self.cross_attention(
            Xq=x_norm,
            Xkv=encoder_out,
            q_positions=tgt_positions,
            kv_positions=src_positions,
            deterministic=deterministic,
            attn_mask=cross_attn_mask,
            cache=cross_attn_cache,
        )
        x = residual + ca_out

        # 3. MLP
        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm, deterministic=deterministic)
        x = residual + mlp_out

        return x


class Decoder(nn.Module):
    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        train_config = config.training
        data_config = config.data
        weight_dtype = _str_to_dtype(config.model.weight_dtype)
        self.num_channels = data_config.channels
        self.num_layers = dec_config.n_layer

        self.embeddings = [
            nn.Embedding(model_config.tgt_vocab_size, dec_config.n_embd)
            for _ in range(self.num_channels)
        ]
        self.dropout = nn.Dropout(model_config.dropout)
        self.layers = [DecoderLayer(config=config) for _ in range(self.num_layers)]
        self.norm = nn.RMSNorm(
            dec_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
        )

        # Final Logits Projection using DenseGeneral
        self.logits_dense = DenseGeneral(
            in_shapes=(dec_config.n_embd,),
            out_features=(self.num_channels, model_config.tgt_vocab_size),
            axis=(-1,),
            dtype=mx.float32,
            weight_dtype=weight_dtype,
        )
        self.logits_in_fp32 = train_config.logits_dot_in_fp32

    def precompute_cross_attention_kv(
        self,
        max_len: int,
        encoder_out: mx.array,  # (B, S, E)
        src_positions: Optional[mx.array],  # (B, S)
    ) -> List[KVCache]:
        """
        Computes the Key and Value tensors for cross-attention for each layer from the encoder output.
        """
        per_layer_kv_cache: List[KVCache] = []

        for layer in self.layers:
            cross_attn_module = layer.cross_attention
            k_proj = cross_attn_module.k_proj(encoder_out)
            v_proj = cross_attn_module.v_proj(encoder_out)

            k_proj = cross_attn_module.rotary_emb(k_proj, position=src_positions)
            k = mx.transpose(k_proj, (0, 2, 1, 3))  # equivalent to transpose(1, 2)
            v = mx.transpose(v_proj, (0, 2, 1, 3))  # equivalent to transpose(1, 2)

            # Create KVCache without device parameter
            per_layer_kv_cache.append(
                KVCache(
                    cross_attn_module.num_kv_heads,
                    max_len,
                    cross_attn_module.head_dim,
                    k=k,
                    v=v,
                )
            )

        return per_layer_kv_cache

    def decode_step(
        self,
        tgt_ids_Bx1xC: mx.array,  # [B, 1, C]
        tgt_pos_Bx1: mx.array,  # [B, 1]
        encoder_out: mx.array,  # [B, S, E]
        self_attn_mask: Any,  # None
        cross_attn_mask: mx.array,  # [B, 1, 1, S]
        self_attention_cache: List[KVCache],
        cross_attention_cache: List[KVCache],
    ) -> mx.array:
        """
        Performs a single decoding step, managing KV caches layer by layer.

        Returns:
            A tuple containing:
            - logits_Bx1xCV: The final output logits for the current step (B, 1, C*V), cast to float32.
            - new_cache: The updated KV cache for the next decoding step.
        """
        assert (
            self_attn_mask is None
        ), "Self-attention mask should be None, kept for pattern"

        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_Bx1xC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            self_cache = self_attention_cache[i]
            cross_cache = cross_attention_cache[i]
            x = layer(
                x,  # (2, 1, D)
                encoder_out,  # (2, S, E)
                src_positions=None,  # CA KV is already computed
                tgt_positions=tgt_pos_Bx1,  # (2, 1)
                deterministic=True,
                self_attn_mask=None,
                cross_attn_mask=cross_attn_mask,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
            )

        x = self.norm(x)
        logits_Bx1xCxV = self.logits_dense(x)

        # Convert to float32 if needed
        if logits_Bx1xCxV.dtype != mx.float32:
            logits_Bx1xCxV = logits_Bx1xCxV.astype(mx.float32)

        return logits_Bx1xCxV

    def __call__(
        self,
        tgt_ids_BxTxC: mx.array,
        encoder_out: mx.array,
        tgt_positions: mx.array,
        src_positions: mx.array,
        deterministic: bool,
        self_attn_mask: mx.array,
        cross_attn_mask: mx.array,
        self_attention_cache: List[KVCache],
        cross_attention_cache: List[KVCache],
    ) -> mx.array:
        """
        Forward pass for the Decoder stack, managing KV caches.

        Args:
            tgt_ids_BxTxC: Target token IDs (B, T, C).
            encoder_out: Output from the encoder (B, S, E).
            tgt_positions: Positions for target sequence (B, T).
            src_positions: Positions for source sequence (B, S).
            deterministic: Disable dropout if True.
            self_attn_mask: Mask for self-attention.
            cross_attn_mask: Mask for cross-attention.
            self_attention_cache: List containing the self-attention KV cache for each layer.
            cross_attention_cache: List containing the cross-attention KV cache for each layer.

        Returns:
            logits: The final output logits (B, T, C * V), cast to float32.
        """
        _, _, num_channels_in = tgt_ids_BxTxC.shape
        assert num_channels_in == self.num_channels, "Input channels mismatch"

        # Embeddings
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_BxTxC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        # Apply dropout if not deterministic
        if not deterministic:
            x = self.dropout(x)

        # Process through each decoder layer
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                encoder_out,
                tgt_positions=tgt_positions,
                src_positions=src_positions,
                deterministic=deterministic,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                self_attn_cache=self_attention_cache[i],
                cross_attn_cache=cross_attention_cache[i],
                prefill=True,
            )

        # Final Norm
        x = self.norm(x)
        logits_BxTxCxV = self.logits_dense(x)

        # Convert to float32 if needed
        if logits_BxTxCxV.dtype != mx.float32:
            logits_BxTxCxV = logits_BxTxCxV.astype(mx.float32)

        return logits_BxTxCxV


class DiaModel(nn.Module):
    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def __call__(
        self,
        src_BxS: mx.array,
        tgt_BxTxC: mx.array,
        src_positions: Optional[mx.array] = None,
        tgt_positions: Optional[mx.array] = None,
        enc_self_attn_mask: Optional[mx.array] = None,
        dec_self_attn_mask: Optional[mx.array] = None,
        dec_cross_attn_mask: Optional[mx.array] = None,
        enable_dropout: bool = True,
    ):
        deterministic = not enable_dropout

        # --- Encoder Pass ---
        encoder_out = self.encoder(
            x_ids=src_BxS,
            src_positions=src_positions,
            deterministic=deterministic,
            attn_mask=enc_self_attn_mask,
        )

        # --- Decoder Pass ---
        max_len = self.config.model.max_sequence_length

        self_attention_cache = []

        for layer in self.decoder.layers:
            self_attn_module = layer.self_attention
            self_attention_cache.append(
                KVCache(
                    self_attn_module.num_query_heads, max_len, self_attn_module.head_dim
                )
            )

        logits = self.decoder(
            tgt_ids_BxTxC=tgt_BxTxC,
            encoder_out=encoder_out,
            tgt_positions=tgt_positions,
            src_positions=src_positions,
            deterministic=deterministic,
            self_attn_mask=dec_self_attn_mask,
            cross_attn_mask=dec_cross_attn_mask,
            self_attention_cache=self_attention_cache,
            cross_attention_cache=None,
        )

        return logits
