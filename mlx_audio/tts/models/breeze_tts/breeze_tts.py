"""MLX inference implementation for Breeze TTS 2.

Breeze uses an autoregressive Qwen3-style backbone for codebook zero and a
small depth decoder for the remaining Qwen3-TTS codec codebooks.  The text
encoder and codec bundled by Breeze are loaded from the original checkpoint;
no PyTorch or upstream Breeze runtime is required at inference time.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.lm.models import llama, qwen3
from mlx_audio.lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.rope_utils import initialize_rope
from mlx_audio.lm.sample_utils import make_sampler
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.utils import load_audio

from .config import ModelConfig


class _AudioEmbedding(nn.Module):
    """Sum wrapper-level audio codebook embeddings.

    The nested Qwen config describes the text backbone and therefore has a
    different vocabulary from Breeze's audio wrapper.  Keep the wrapper
    vocabulary/codebook count explicit here.  ``audio_embeds_projector`` is
    present only for checkpoints whose embedding width differs from the
    backbone width, matching ``BreezeBackboneModelEmbeddings`` upstream.
    """

    def __init__(
        self,
        num_codebooks: int,
        vocab_size: int,
        hidden_size: int,
        audio_embed_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_codebooks = int(num_codebooks)
        self.vocab_size = int(vocab_size)
        self.audio_embed_size = int(audio_embed_size or hidden_size)
        self.embed_audio_tokens = nn.Embedding(
            self.num_codebooks * self.vocab_size, self.audio_embed_size
        )
        if self.audio_embed_size != hidden_size:
            self.audio_embeds_projector = nn.Linear(
                self.audio_embed_size, hidden_size, bias=False
            )
        else:
            self.audio_embeds_projector = None

    def __call__(self, codebooks: mx.array) -> mx.array:
        if codebooks.ndim != 3 or codebooks.shape[-1] != self.num_codebooks:
            raise ValueError(
                "Breeze audio codebooks must have shape "
                f"[batch, time, {self.num_codebooks}], got {codebooks.shape}."
            )
        offsets = (
            mx.arange(self.num_codebooks, dtype=codebooks.dtype)[None, None, :]
            * self.vocab_size
        )
        hidden = self.embed_audio_tokens(codebooks + offsets)
        if self.audio_embeds_projector is not None:
            hidden = self.audio_embeds_projector(hidden)
        return mx.sum(hidden, axis=-2)


class _Backbone(nn.Module):
    """Qwen3 transformer whose names match ``backbone_model.*`` weights."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        wrapper_num_codebooks = config.wrapper_num_codebooks
        wrapper_vocab_size = config.wrapper_vocab_size
        hidden_size = config.backbone_hidden_size
        args = qwen3.ModelArgs(
            model_type="qwen3",
            hidden_size=hidden_size,
            num_hidden_layers=config.backbone_value(
                "num_hidden_layers", config.num_hidden_layers
            ),
            intermediate_size=config.backbone_value(
                "intermediate_size", config.intermediate_size
            ),
            num_attention_heads=config.backbone_value(
                "num_attention_heads", config.num_attention_heads
            ),
            num_key_value_heads=config.backbone_value(
                "num_key_value_heads", config.num_key_value_heads
            ),
            head_dim=config.backbone_value("head_dim", config.head_dim),
            rms_norm_eps=config.backbone_value("rms_norm_eps", config.rms_norm_eps),
            # This is metadata for the Qwen args.  The actual embedding below
            # deliberately uses the wrapper audio vocabulary.
            vocab_size=config.backbone_value("vocab_size", wrapper_vocab_size),
            max_position_embeddings=config.backbone_value(
                "max_position_embeddings", config.max_position_embeddings
            ),
            rope_theta=config.backbone_value("rope_theta", config.rope_theta),
            rope_scaling=config.backbone_value("rope_scaling", config.rope_scaling),
            tie_word_embeddings=config.backbone_value("tie_word_embeddings", False),
        )
        self.args = args
        self.embed_tokens = _AudioEmbedding(
            wrapper_num_codebooks,
            wrapper_vocab_size,
            args.hidden_size,
            config.wrapper_audio_embed_size,
        )
        self.layers = [
            qwen3.TransformerBlock(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        *,
        input_embeddings: Optional[mx.array] = None,
        cache: Optional[list[KVCache]] = None,
    ) -> mx.array:
        if (input_ids is None) == (input_embeddings is None):
            raise ValueError("Pass exactly one of input_ids or input_embeddings.")
        hidden = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(input_ids)
        )
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(hidden, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(hidden, mask, layer_cache)
        return self.norm(hidden)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]


class _TextEmbedding(nn.Module):
    """T5Gemma2 embedding with Breeze's separate end-of-input embedding."""

    def __init__(self, vocab_size: int, hidden_size: int, eoi_token_index: int):
        super().__init__()
        self.weight = mx.zeros((vocab_size, hidden_size))
        self.eoi_embedding = mx.zeros((hidden_size,))
        self.eoi_token_index = eoi_token_index

    def __call__(self, input_ids: mx.array) -> mx.array:
        embeds = self.weight[input_ids] * mx.array(self.weight.shape[-1] ** 0.5)
        return mx.where(
            (input_ids == self.eoi_token_index)[..., None], self.eoi_embedding, embeds
        )


class _T5Gemma2RMSNorm(nn.Module):
    """T5Gemma2's zero-initialized, offset RMS norm."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


def _t5gemma2_attention_mask(
    length: int,
    layer_type: str,
    sliding_window: Optional[int],
    attention_mask: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Build the official non-causal T5Gemma2 attention pattern.

    Full-attention layers are bidirectional.  Sliding layers use the
    asymmetric-looking but symmetric-around-the-centre window specified by
    T5Gemma2: ``(window + 1) // 2`` tokens left and ``window // 2 + 1`` right
    (both include the query position).
    """
    if length < 0:
        raise ValueError(f"Attention length must be non-negative, got {length}")
    if layer_type not in {"full_attention", "sliding_attention"}:
        raise ValueError(f"Unsupported T5Gemma2 attention layer: {layer_type}")
    if layer_type == "sliding_attention" and (
        sliding_window is None or sliding_window <= 0
    ):
        raise ValueError(
            "T5Gemma2 sliding_attention requires a positive sliding_window"
        )

    valid_keys = None
    if attention_mask is not None:
        valid_keys = mx.array(attention_mask).astype(mx.bool_)
        if valid_keys.ndim == 1:
            if valid_keys.shape[0] != length:
                raise ValueError(
                    "T5Gemma2 attention_mask length does not match input: "
                    f"{valid_keys.shape[0]} != {length}."
                )
            valid_keys = valid_keys[None, :]
        elif valid_keys.ndim == 2:
            if valid_keys.shape[1] != length:
                raise ValueError(
                    "T5Gemma2 attention_mask length does not match input: "
                    f"{valid_keys.shape[1]} != {length}."
                )
        else:
            raise ValueError(
                "T5Gemma2 attention_mask must have shape [length] or "
                f"[batch, length], got {valid_keys.shape}."
            )

    if layer_type == "full_attention":
        # ``None`` is intentional: MLX interprets this as unconstrained,
        # bidirectional attention.  A padding mask is broadcast over query
        # rows while masking invalid key/value positions, matching the
        # upstream T5Gemma2 eager implementation.
        if valid_keys is None:
            return None
        return valid_keys[:, None, None, :]

    query = mx.arange(length)[:, None]
    key = mx.arange(length)[None, :]
    distance = query - key
    left = (sliding_window + 1) // 2
    right = sliding_window // 2 + 1
    local = ((distance >= 0) & (distance < left)) | (
        (distance < 0) & (-distance < right)
    )
    if valid_keys is None:
        return local
    return local[None, None, :, :] & valid_keys[:, None, None, :]


class _TextAttention(nn.Module):
    """Bidirectional attention matching the T5Gemma2 text encoder."""

    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.n_heads = int(config["num_attention_heads"])
        self.n_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.scale = config.get("query_pre_attn_scalar", 256) ** -0.5
        layer_types = (
            config.get("layer_types")
            or ["sliding_attention"] * config["num_hidden_layers"]
        )
        if layer_idx >= len(layer_types):
            raise ValueError(
                "T5Gemma2 layer_types has fewer entries than num_hidden_layers"
            )
        self.layer_type = layer_types[layer_idx]
        self.sliding_window = (
            config.get("sliding_window", 512)
            if self.layer_type == "sliding_attention"
            else None
        )
        dim = int(config["hidden_size"])
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        eps = config["rms_norm_eps"]
        self.q_norm = _T5Gemma2RMSNorm(self.head_dim, eps)
        self.k_norm = _T5Gemma2RMSNorm(self.head_dim, eps)
        rope_parameters = config.get("rope_parameters") or {}
        rope = rope_parameters.get(self.layer_type) or {}
        self.rope = initialize_rope(
            self.head_dim,
            base=rope.get(
                "rope_theta",
                10_000.0 if self.layer_type == "sliding_attention" else 1_000_000.0,
            ),
            traditional=False,
            scaling_config=rope if rope.get("rope_type") == "linear" else None,
            max_position_embeddings=config.get("max_position_embeddings", 32768),
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = (
            self.v_proj(x)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        output = scaled_dot_product_attention(
            self.rope(queries), self.rope(keys), values, None, self.scale, mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class _TextMLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dim, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class _TextLayer(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.self_attn = _TextAttention(config, layer_idx)
        eps, dim = config["rms_norm_eps"], config["hidden_size"]
        self.pre_self_attn_layernorm = _T5Gemma2RMSNorm(dim, eps)
        self.post_self_attn_layernorm = _T5Gemma2RMSNorm(dim, eps)
        self.mlp = _TextMLP(config)
        self.pre_feedforward_layernorm = _T5Gemma2RMSNorm(dim, eps)
        self.post_feedforward_layernorm = _T5Gemma2RMSNorm(dim, eps)
        self.layer_type = self.self_attn.layer_type

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        hidden = self.self_attn(self.pre_self_attn_layernorm(x), mask)
        hidden = x + self.post_self_attn_layernorm(hidden)
        return hidden + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(hidden))
        )


class _TextEncoder(nn.Module):
    """MLX implementation of Breeze's bidirectional T5Gemma2 text encoder."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        config = self._normalize_config(config)
        self.config = config
        self.embed_tokens = _TextEmbedding(
            config["vocab_size"],
            config["hidden_size"],
            config.get("eoi_token_index", 256000),
        )
        self.layers = [
            _TextLayer(config, i) for i in range(config["num_hidden_layers"])
        ]
        self.norm = _T5Gemma2RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    @staticmethod
    def _normalize_config(config: Optional[dict]) -> dict:
        """Fill only the T5Gemma2 defaults needed by direct MLX callers."""
        config = dict(ModelConfig._as_mapping(config))
        defaults = {
            "vocab_size": 262208,
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "query_pre_attn_scalar": 256,
            "max_position_embeddings": 131072,
            "sliding_window": 4096,
            "eoi_token_index": 256000,
            "dropout_rate": 0.0,
            "attention_bias": False,
        }
        for key, value in defaults.items():
            config.setdefault(key, value)
        if not config.get("layer_types"):
            pattern = config.pop(
                "_sliding_window_pattern",
                config.pop("sliding_window_pattern", 6),
            )
            if not isinstance(pattern, int) or pattern <= 0:
                raise ValueError(
                    "T5Gemma2 sliding_window_pattern must be a positive integer"
                )
            config["layer_types"] = [
                "sliding_attention" if (idx + 1) % pattern else "full_attention"
                for idx in range(config["num_hidden_layers"])
            ]
        # Keep the upstream defaults for the two attention-specific RoPE
        # bases, while allowing an explicitly supplied mapping to override
        # either layer type.
        rope_parameters = {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
            },
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1_000_000.0,
            },
        }
        provided_rope_parameters = config.get("rope_parameters") or {}
        for key, value in provided_rope_parameters.items():
            rope_parameters[key] = dict(value or {})
        # Older T5Gemma2 exports use the generic ``rope_scaling`` field for
        # the full-attention stream.  Keep accepting that form while giving
        # the per-layer ``rope_parameters`` mapping precedence when present.
        if config.get("rope_scaling") and (
            "full_attention" not in provided_rope_parameters
        ):
            rope_parameters["full_attention"].update(config["rope_scaling"])
        config["rope_parameters"] = rope_parameters
        return config

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden = self.embed_tokens(input_ids)
        masks = {}
        for layer in self.layers:
            if layer.layer_type in masks:
                continue
            masks[layer.layer_type] = _t5gemma2_attention_mask(
                hidden.shape[1],
                layer.layer_type,
                layer.self_attn.sliding_window,
                attention_mask,
            )
        for layer in self.layers:
            hidden = layer(hidden, masks[layer.layer_type])
        return self.norm(hidden)


class _DepthModel(nn.Module):
    """Breeze's codebook-autoregressive depth decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        depth = ModelConfig._as_mapping(config.depth_decoder_config)
        self.vocab_size = int(depth.get("vocab_size", config.wrapper_vocab_size))
        self.num_codebooks = int(
            depth.get("num_codebooks", config.wrapper_num_codebooks)
        )
        self.audio_embed_size = int(
            depth.get("audio_embed_size", config.wrapper_audio_embed_size)
        )
        self.backbone_hidden_size = int(
            depth.get("backbone_hidden_size", config.backbone_hidden_size)
        )
        self.hidden_size = int(depth.get("hidden_size", 1024))
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=self.hidden_size,
            num_hidden_layers=depth.get("num_hidden_layers", 12),
            intermediate_size=depth.get("intermediate_size", 8192),
            num_attention_heads=depth.get("num_attention_heads", 8),
            num_key_value_heads=depth.get("num_key_value_heads", 2),
            head_dim=depth.get("head_dim", 128),
            rms_norm_eps=depth.get("rms_norm_eps", 1e-5),
            vocab_size=self.num_codebooks * self.vocab_size,
            max_position_embeddings=depth.get("max_position_embeddings", 33),
            rope_theta=depth.get("rope_theta", 500000.0),
            rope_scaling=depth.get("rope_scaling"),
            tie_word_embeddings=False,
            rope_traditional=False,
        )
        self.embed_tokens = nn.Embedding(args.vocab_size, self.audio_embed_size)
        if self.backbone_hidden_size != self.audio_embed_size:
            self.backbone_hidden_state_projector = nn.Linear(
                self.backbone_hidden_size, self.audio_embed_size, bias=False
            )
        else:
            self.backbone_hidden_state_projector = None
        self.inputs_embeds_projector = nn.Linear(
            self.audio_embed_size, self.hidden_size, bias=False
        )
        self.layers = [
            llama.TransformerBlock(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(self.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self, token_ids: mx.array, backbone_hidden_state: mx.array
    ) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError(
                f"Depth decoder token_ids must have shape [batch, time], got {token_ids.shape}."
            )
        if backbone_hidden_state.ndim != 2:
            raise ValueError(
                "Depth decoder backbone_hidden_state must have shape "
                f"[batch, hidden], got {backbone_hidden_state.shape}."
            )
        if backbone_hidden_state.shape[0] != token_ids.shape[0]:
            raise ValueError(
                "Depth decoder token and hidden-state batch sizes differ: "
                f"{token_ids.shape[0]} != {backbone_hidden_state.shape[0]}."
            )
        # token_ids starts with a placeholder. Position zero is replaced by the
        # backbone state, while later positions carry codebook-specific offsets.
        positions = mx.maximum(mx.arange(token_ids.shape[1]) - 1, 0)
        embeds = self.embed_tokens(token_ids + positions[None, :] * self.vocab_size)
        if self.backbone_hidden_state_projector is not None:
            backbone_hidden_state = self.backbone_hidden_state_projector(
                backbone_hidden_state
            )
        embeds = mx.concatenate(
            [backbone_hidden_state[:, None, :], embeds[:, 1:, :]], axis=1
        )
        hidden = self.inputs_embeds_projector(embeds)
        mask = create_attention_mask(hidden, None)
        for layer in self.layers:
            hidden = layer(hidden, mask, None)
        return self.norm(hidden)


class _DepthDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = _DepthModel(config)
        self.codebooks_head = _CodebooksHead(
            self.model.num_codebooks - 1,
            self.model.hidden_size,
            self.model.vocab_size,
        )

    def next_logits(
        self, token_ids: mx.array, backbone_hidden_state: mx.array
    ) -> mx.array:
        # The most recent input token is codebook N-1, and predicts codebook N.
        head_idx = token_ids.shape[1] - 2
        if not 0 <= head_idx < self.codebooks_head.weight.shape[0]:
            raise ValueError(
                "Depth decoder input must contain the placeholder and at least "
                f"one codebook token; got sequence length {token_ids.shape[1]}."
            )
        hidden = self.model(token_ids, backbone_hidden_state)[:, -1, :]
        return hidden @ self.codebooks_head.weight[head_idx]


class _CodebooksHead(nn.Module):
    """Position-specific output projections for depth-decoded codebooks."""

    def __init__(self, num_heads: int, hidden_size: int, vocab_size: int):
        super().__init__()
        self.weight = mx.zeros((num_heads, hidden_size, vocab_size))


class Model(nn.Module):
    """Breeze TTS 2 text-to-speech model."""

    preserve_ref_audio_path = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_codebooks = config.wrapper_num_codebooks
        self.vocab_size = config.wrapper_vocab_size
        self.lm_head = nn.Linear(
            config.backbone_hidden_size, self.vocab_size + 1, bias=False
        )
        self.embed_text_tokens = nn.Embedding(
            config.text_vocab_size, config.backbone_hidden_size
        )
        self.backbone_model = _Backbone(config)
        self.depth_decoder = _DepthDecoder(config)
        text_config = _TextEncoder._normalize_config(config.text_encoder_config)
        self.text_encoder = _TextEncoder(text_config)
        self.text_encoder_proj = nn.Linear(
            text_config["hidden_size"],
            config.backbone_hidden_size,
            bias=False,
        )
        self.tokenizer = None
        self.audio_tokenizer = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def model_type(self) -> str:
        return "breeze"

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Keep only main-model tensors; the bundled codec is loaded separately."""
        sanitized = {}
        for key, value in weights.items():
            if (
                key.startswith("codec_model.")
                or key.endswith(".initialized")
                or "rotary_emb.inv_freq" in key
            ):
                continue
            sanitized[key] = value
        # The official checkpoint ties these embeddings and stores only the
        # depth-decoder tensor in safetensors. MLX modules retain independent
        # parameters, so materialize the matching backbone entry at load time.
        tied = "depth_decoder.model.embed_tokens.weight"
        backbone_tied = "backbone_model.embed_tokens.embed_audio_tokens.weight"
        if tied in sanitized:
            # Always derive the backbone copy from the canonical depth tensor.
            # This handles exports that contain a stale duplicate while
            # preserving exact value/shape parity for strict MLX loading.
            sanitized[backbone_tied] = sanitized[tied]
        return sanitized

    @staticmethod
    def _validate_codec_weight_keys(expected: set[str], provided: set[str]) -> None:
        """Reject codec snapshots except for generated quantizer flags.

        Qwen3-TTS computes the 32 ``codebook.initialized`` flags at runtime;
        they are intentionally not serialized.  Every real parameter must be
        present exactly once so a changed codec architecture cannot be loaded
        with a misleading ``strict=False`` fallback.
        """
        allowed_missing = {
            name
            for name in expected
            if name.startswith("encoder_model.quantizer.")
            and name.endswith(".codebook.initialized")
        }
        missing = expected - provided - allowed_missing
        unexpected = provided - expected
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if unexpected:
                details.append(f"unexpected: {', '.join(sorted(unexpected))}")
            raise ValueError(
                "Incompatible Breeze audio tokenizer weights ("
                + "; ".join(details)
                + ")"
            )

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from transformers import PreTrainedTokenizerFast

        # AutoTokenizer attempts to instantiate the unknown upstream ``breeze``
        # config before loading the otherwise standard fast tokenizer. Loading
        # tokenizer.json directly avoids that PyTorch-runtime-only registration.
        tokenizer_config = json.loads(
            (model_path / "tokenizer_config.json").read_text()
        )
        for key in (
            "tokenizer_class",
            "model_max_length",
            "clean_up_tokenization_spaces",
            "added_tokens_decoder",
        ):
            tokenizer_config.pop(key, None)
        model.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(model_path / "tokenizer.json"), **tokenizer_config
        )
        codec_path = model_path / "audio_tokenizer"
        if not codec_path.is_dir():
            raise FileNotFoundError(
                f"Breeze checkpoint has no audio_tokenizer at {codec_path}"
            )
        model.audio_tokenizer = cls._load_audio_tokenizer(codec_path)
        return model

    @staticmethod
    def _load_audio_tokenizer(codec_path: Path):
        from mlx_audio.tts.models.qwen3_tts.config import (
            Qwen3TTSTokenizerConfig,
            Qwen3TTSTokenizerDecoderConfig,
            Qwen3TTSTokenizerEncoderConfig,
            filter_dict_for_dataclass,
        )
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizer,
        )

        config_data = json.loads((codec_path / "config.json").read_text())
        decoder = Qwen3TTSTokenizerDecoderConfig(
            **filter_dict_for_dataclass(
                Qwen3TTSTokenizerDecoderConfig, config_data["decoder_config"]
            )
        )
        encoder = Qwen3TTSTokenizerEncoderConfig(
            **filter_dict_for_dataclass(
                Qwen3TTSTokenizerEncoderConfig, config_data["encoder_config"]
            )
        )
        codec_config = Qwen3TTSTokenizerConfig(
            decoder_config=decoder, encoder_config=encoder
        )
        for key, value in config_data.items():
            if key not in {"decoder_config", "encoder_config"} and hasattr(
                codec_config, key
            ):
                setattr(codec_config, key, value)
        codec = Qwen3TTSSpeechTokenizer(codec_config)
        weights = {}
        for weight_file in codec_path.glob("*.safetensors"):
            weights.update(mx.load(str(weight_file)))
        sanitized = Qwen3TTSSpeechTokenizer.sanitize(weights)
        Model._validate_codec_weight_keys(
            {name for name, _ in tree_flatten(codec.parameters())}, set(sanitized)
        )
        # The 32 quantizer ``initialized`` flags are generated locally by
        # ``update_in_place`` below.  All serialized keys were validated above.
        # ``initialized`` is an MLX-only runtime flag, so the model loader must
        # permit precisely those known missing keys.  The complete set of
        # real keys was validated above; this is intentionally not a blanket
        # strict=False load.
        codec.load_weights(list(sanitized.items()), strict=False)
        if codec.encoder_model is not None:
            quantizer = codec.encoder_model.quantizer
            for layer in quantizer.rvq_first.vq.layers + quantizer.rvq_rest.vq.layers:
                layer.codebook.update_in_place()
        mx.eval(codec.parameters())
        codec.eval()
        return codec

    def _speaker(self, voice: Optional[str]) -> str:
        # mlx-audio's generic default is a Kokoro voice; Breeze's documented
        # default speaker tag is S0.
        speaker = "S0" if voice in (None, "", "af_heart") else voice
        return speaker if speaker.startswith("[") else f"[{speaker}]"

    def _text_ids(self, text: str) -> mx.array:
        if self.tokenizer is None:
            raise RuntimeError("Breeze tokenizer was not loaded")
        return mx.array(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def _encode_reference(self, ref_audio: Union[str, Path, mx.array]) -> mx.array:
        if self.audio_tokenizer is None:
            raise RuntimeError("Breeze audio tokenizer was not loaded")
        if isinstance(ref_audio, (str, Path)):
            audio = load_audio(str(ref_audio), sample_rate=self.sample_rate)
        else:
            audio = ref_audio
        audio = mx.array(audio)
        if audio.ndim == 1:
            audio = audio[None, None, :]
        elif audio.ndim == 2:
            audio = audio[:, None, :]
        elif audio.ndim != 3:
            raise ValueError(
                "ref_audio must be a path, [samples], [batch, samples], or [batch, channels, samples]."
            )
        if audio.shape[0] != 1:
            raise ValueError(
                "Breeze supports exactly one reference audio item per generation; "
                f"received batch size {audio.shape[0]}."
            )
        return mx.transpose(self.audio_tokenizer.encode(audio), (0, 2, 1))

    def _prompt_embeddings(
        self,
        text: str,
        *,
        voice: Optional[str],
        instruct: Optional[str],
        ref_audio: Optional[Union[str, Path, mx.array]],
        ref_text: Optional[str],
    ) -> mx.array:
        if isinstance(ref_audio, (list, tuple)):
            if len(ref_audio) != 1:
                raise ValueError(
                    "Breeze supports exactly one reference audio item per generation."
                )
            ref_audio = ref_audio[0]
        if isinstance(ref_text, (list, tuple)):
            if len(ref_text) != 1:
                raise ValueError(
                    "Breeze supports exactly one reference transcript per generation."
                )
            ref_text = ref_text[0]
        if ref_audio is not None and not ref_text:
            raise ValueError("Breeze voice cloning requires ref_text with ref_audio.")
        speaker = self._speaker(voice)
        segments: list[tuple[str, mx.array]] = []
        if ref_audio is not None:
            segments.append(("text", self._text_ids(f"{speaker}{ref_text}")))
            segments.append(("audio", self._encode_reference(ref_audio)))
        target = f"{speaker}{text}"
        if instruct:
            target = f"{speaker}<ins_bos>{instruct}<ins_eos>{text}"
        segments.append(("text", self._text_ids(target)))

        embeddings: list[mx.array] = []
        for kind, values in segments:
            if kind == "text":
                hidden = self.text_encoder(values[None, :])
                embeddings.append(self.text_encoder_proj(hidden))
            else:
                embeddings.append(self.backbone_model.embed_tokens(values))
                eos_codes = mx.full(
                    (1, 1, self.num_codebooks), self.config.codebook_eos_token_id
                )
                embeddings.append(self.backbone_model.embed_tokens(eos_codes))
        return mx.concatenate(embeddings, axis=1)

    def _sample(
        self,
        logits: mx.array,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        allow_eos: bool = False,
    ) -> int:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")

        valid = self.vocab_size + 1 if allow_eos else self.vocab_size
        if logits.shape[-1] < valid:
            raise ValueError(
                "Breeze logits do not contain the expected codebook vocabulary: "
                f"{logits.shape[-1]} < {valid}."
            )

        # Keep EOS as the one extra class used by the backbone while reserving
        # padding/control ids in every regular codebook distribution.  Clamp
        # top-k to this distribution: tiny deterministic fixtures often use a
        # vocab smaller than the user-facing default of 50.
        logits = self._mask_reserved_codec_logits(logits)[..., :valid]
        effective_top_k = min(top_k, valid) if top_k else 0
        if effective_top_k == valid:
            effective_top_k = 0
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=effective_top_k)
        token = sampler(nn.log_softmax(logits, axis=-1))
        return int(token.item())

    def _mask_reserved_codec_logits(self, logits: mx.array) -> mx.array:
        """Mask ids outside the codec codebook in a logits tensor.

        Breeze's wrapper vocabulary contains codec entries followed by
        padding/control ids (and one separate backbone EOS class).  The same
        mask is used for the first codebook and for every depth-decoder step;
        callers may still retain the EOS column by slicing after this helper.
        """
        if logits.shape[-1] < self.vocab_size:
            raise ValueError(
                "Breeze logits do not contain the wrapper vocabulary: "
                f"{logits.shape[-1]} < {self.vocab_size}."
            )
        start = self.config.codec_vocab_size
        if start >= self.vocab_size:
            return logits
        return logits.at[..., start : self.vocab_size].add(-mx.inf)

    @staticmethod
    def _apply_repetition_penalty(
        logits: mx.array, generated_tokens: list[int], penalty: float
    ) -> mx.array:
        """Apply the standard sign-aware repetition penalty to prior tokens."""
        if penalty == 1.0 or not generated_tokens:
            return logits
        token_ids = sorted(
            {token for token in generated_tokens if 0 <= token < logits.shape[-1]}
        )
        if not token_ids:
            return logits
        indices = mx.array(token_ids, dtype=mx.int32)
        selected = mx.take(logits, indices, axis=-1)
        penalized = mx.where(selected < 0, selected * penalty, selected / penalty)
        return mx.put_along_axis(logits, indices[None, :], penalized, axis=-1)

    def _depth_tokens(
        self,
        first_codebook: int,
        conditional_hidden: mx.array,
        *,
        unconditional_hidden: Optional[mx.array],
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[int]:
        tokens = [0, first_codebook]
        for _ in range(self.num_codebooks - 1):
            token_ids = mx.array(tokens, dtype=mx.int32)[None, :]
            logits = self.depth_decoder.next_logits(token_ids, conditional_hidden)
            if unconditional_hidden is not None:
                unconditional_logits = self.depth_decoder.next_logits(
                    token_ids, unconditional_hidden
                )
                logits = unconditional_logits + cfg_scale * (
                    logits - unconditional_logits
                )
            # Apply the reserved-id mask before handing logits to the sampler.
            # Keeping it here (as well as in ``_sample``) means custom samplers
            # and deterministic test doubles observe the same official flow.
            logits = self._mask_reserved_codec_logits(logits)
            tokens.append(
                self._sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            )
        return tokens[1:]

    @staticmethod
    def _audio_vector(audio: Any) -> mx.array:
        """Normalize codec output to the one-dimensional public waveform."""
        if hasattr(audio, "audio_values"):
            audio = audio.audio_values
        while isinstance(audio, (list, tuple)):
            audio = audio[0]
        audio = mx.array(audio)
        while audio.ndim > 1:
            audio = audio[0]
        return audio

    def _decode_codes(self, codes: mx.array) -> mx.array:
        """Decode one sequence and honor the tokenizer's valid length."""
        decoded = self.audio_tokenizer.decode(codes)
        lengths = None
        if isinstance(decoded, (list, tuple)):
            if not decoded:
                return mx.zeros((0,), dtype=mx.float32)
            audio_data = decoded[0]
            if len(decoded) > 1:
                lengths = decoded[1]
        else:
            audio_data = decoded
        audio = self._audio_vector(audio_data)
        if lengths is not None:
            lengths = mx.array(lengths).reshape(-1)
            if lengths.shape[0]:
                valid_samples = max(0, int(lengths[0].item()))
                audio = audio[:valid_samples]
        return audio

    def _empty_audio(self) -> mx.array:
        """Return the official silent fallback when EOS precedes all frames."""
        # The upstream implementation decodes a one-frame dummy code rather
        # than raising.  Keep that behavior when a codec is available, while
        # retaining a zero-length CPU-safe fallback for light test doubles.
        try:
            dummy = mx.ones((1, 1, self.num_codebooks), dtype=mx.int32)
            return self._decode_codes(dummy)
        except Exception:  # pragma: no cover - only used by partial fakes
            return mx.zeros((0,), dtype=mx.float32)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio: Optional[Union[str, Path, mx.array]] = None,
        ref_text: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        max_tokens: int = 750,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **_: object,
    ) -> Generator[GenerationResult, None, None]:
        """Generate Breeze audio for voice design, cloning, or direction."""
        if seed is not None:
            mx.random.seed(seed)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive.")
        if cfg_scale is not None and not math.isfinite(cfg_scale):
            raise ValueError("cfg_scale must be finite.")
        if streaming_interval <= 0:
            raise ValueError("streaming_interval must be positive.")

        started = time.perf_counter()
        cond = self._prompt_embeddings(
            text, voice=voice, instruct=instruct, ref_audio=ref_audio, ref_text=ref_text
        )
        use_cfg = bool(instruct) and cfg_scale not in (None, 1.0)
        scale = 1.0 if cfg_scale is None else cfg_scale
        if use_cfg:
            uncond = self._prompt_embeddings(
                text, voice=voice, instruct=None, ref_audio=ref_audio, ref_text=ref_text
            )

        cond_cache = self.backbone_model.make_cache()
        cond_hidden = self.backbone_model(input_embeddings=cond, cache=cond_cache)[
            :, -1, :
        ]
        if use_cfg:
            uncond_cache = self.backbone_model.make_cache()
            uncond_hidden = self.backbone_model(
                input_embeddings=uncond, cache=uncond_cache
            )[:, -1, :]

        frames: list[list[int]] = []
        pending_frames: list[list[int]] = []
        decode_rate = getattr(self.audio_tokenizer, "decode_upsample_rate", None)
        if decode_rate is None:
            decoder = getattr(self.audio_tokenizer, "decoder", None)
            decode_rate = getattr(decoder, "decode_upsample_rate", None)
        if decode_rate is None or decode_rate <= 0:
            raise ValueError("Breeze audio tokenizer has no valid decode rate")
        chunk_frames = max(1, int(streaming_interval * self.sample_rate / decode_rate))
        if stream:
            self.audio_tokenizer.decoder.reset_streaming_state()

        def stream_result(audio: mx.array, token_count: int, final: bool):
            mx.eval(audio)
            samples = audio.shape[0]
            elapsed = time.perf_counter() - started
            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0
            samples_per_sec = samples / elapsed if elapsed > 0 else 0.0
            return GenerationResult(
                audio=audio,
                samples=samples,
                sample_rate=self.sample_rate,
                segment_idx=0,
                token_count=token_count,
                audio_duration=f"00:00:{samples / self.sample_rate:06.3f}",
                real_time_factor=elapsed / max(samples / self.sample_rate, 1e-6),
                prompt={
                    "tokens": token_count,
                    "tokens-per-sec": tokens_per_sec,
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": samples_per_sec,
                },
                processing_time_seconds=elapsed,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
                is_streaming_chunk=True,
                is_final_chunk=final,
            )

        for _ in range(max_tokens):
            cond_logits = self.lm_head(cond_hidden)
            if use_cfg:
                uncond_logits = self.lm_head(uncond_hidden)
                logits = uncond_logits + scale * (cond_logits - uncond_logits)
            else:
                logits = cond_logits
            logits = self._apply_repetition_penalty(
                logits, [frame[0] for frame in frames], repetition_penalty
            )
            logits = self._mask_reserved_codec_logits(logits)
            first = self._sample(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                allow_eos=True,
            )
            if first == self.vocab_size:
                break
            frame = self._depth_tokens(
                first,
                cond_hidden,
                unconditional_hidden=uncond_hidden if use_cfg else None,
                cfg_scale=scale,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            frames.append(frame)
            pending_frames.append(frame)
            if stream and len(pending_frames) >= chunk_frames:
                chunk = pending_frames[:chunk_frames]
                del pending_frames[:chunk_frames]
                pending_codes = mx.array(chunk, dtype=mx.int32)[None, :, :]
                stream_audio = self.audio_tokenizer.decoder.streaming_step(
                    mx.transpose(pending_codes, (0, 2, 1))
                )
                yield stream_result(
                    self._audio_vector(stream_audio), len(chunk), final=False
                )
            codebooks = mx.array(frame, dtype=mx.int32)[None, None, :]
            cond_hidden = self.backbone_model(input_ids=codebooks, cache=cond_cache)[
                :, -1, :
            ]
            if use_cfg:
                uncond_hidden = self.backbone_model(
                    input_ids=codebooks, cache=uncond_cache
                )[:, -1, :]

        if not frames:
            empty_audio = self._empty_audio()
            if stream:
                self.audio_tokenizer.decoder.reset_streaming_state()
            mx.eval(empty_audio)
            elapsed = time.perf_counter() - started
            samples = empty_audio.shape[0]
            samples_per_sec = samples / elapsed if elapsed > 0 else 0.0
            yield GenerationResult(
                audio=empty_audio,
                samples=samples,
                sample_rate=self.sample_rate,
                segment_idx=0,
                token_count=0,
                audio_duration=f"00:00:{samples / self.sample_rate:06.3f}",
                real_time_factor=0.0,
                prompt={"tokens": 0, "tokens-per-sec": 0.0},
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": samples_per_sec,
                },
                processing_time_seconds=elapsed,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
                is_streaming_chunk=stream,
                is_final_chunk=True,
            )
            return
        if stream:
            if pending_frames:
                pending_codes = mx.array(pending_frames, dtype=mx.int32)[None, :, :]
                stream_audio = self.audio_tokenizer.decoder.streaming_step(
                    mx.transpose(pending_codes, (0, 2, 1))
                )
                self.audio_tokenizer.decoder.reset_streaming_state()
                yield stream_result(
                    self._audio_vector(stream_audio), len(pending_frames), final=True
                )
            else:
                self.audio_tokenizer.decoder.reset_streaming_state()
            return
        codes = mx.array(frames, dtype=mx.int32)[None, :, :]
        audio = self._decode_codes(codes)
        samples = audio.shape[0]
        mx.eval(audio)
        elapsed = time.perf_counter() - started
        token_count = len(frames)
        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0
        samples_per_sec = samples / elapsed if elapsed > 0 else 0.0
        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=token_count,
            audio_duration=f"00:00:{samples / self.sample_rate:06.3f}",
            real_time_factor=elapsed / max(samples / self.sample_rate, 1e-6),
            prompt={
                "tokens": token_count,
                "tokens-per-sec": tokens_per_sec,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": samples_per_sec,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_final_chunk=True,
        )
