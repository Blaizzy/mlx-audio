# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
# LFM2.5-Audio: Main model implementation

import json
import logging
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.lfm2 import Lfm2Model

from ....base import check_array_shape
from .config import DepthformerConfig, LFM2AudioConfig
from .conformer import MLP, ConformerEncoder
from .transformer import Depthformer

logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class LFMModality(IntEnum):
    """Modality types for LFM2 Audio.

    Note: Values 1, 2, 3 match PyTorch implementation (0 is reserved/unused).
    """

    TEXT = 1
    AUDIO_IN = 2
    AUDIO_OUT = 3


# Special token IDs used by LFM2.5-Audio
AUDIO_START_TOKEN = 128  # <|audio_start|> - triggers transition to audio mode
IM_END_TOKEN = 7  # <|im_end|> - end of turn
TEXT_END_TOKEN = 130  # <|text_end|> - marks text generation complete
AUDIO_EOS_TOKEN = 2048  # End-of-sequence for audio codebooks


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    audio_temperature: float = 1.0
    audio_top_k: int = 4


class AudioEmbeddingWithNorm(nn.Module):

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding_norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

    def embed(self, x: mx.array) -> mx.array:
        """Embed tokens with normalization."""
        return self.embedding_norm(self.embedding(x))

    def embed_raw(self, x: mx.array) -> mx.array:
        """Embed tokens WITHOUT normalization (used for conditioning)."""
        return self.embedding(x)

    def logits(self, x: mx.array) -> mx.array:
        """Project to vocabulary logits."""
        return self.to_logits(x)


class AudioEmbedding(nn.Module):
    """Audio token embeddings for multiple codebooks (input).

    Uses a shared embedding space with codebook offsets, following the reference
    implementation: audio_embedding(next_token + codebook_offsets).sum(0)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_codebooks: int = 8,
        tie: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size  # Per-codebook vocab size (2049)
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.tie = tie

        # Shared embedding space: total_vocab = vocab_size * num_codebooks
        # Each codebook uses offset: codebook_i uses indices [i*vocab_size, (i+1)*vocab_size)
        total_vocab = vocab_size * num_codebooks
        self.embedding = nn.Embedding(total_vocab, dim)

        # Precompute codebook offsets: [0, 2049, 4098, ...]
        self._codebook_offsets = mx.array(
            [i * vocab_size for i in range(num_codebooks)]
        )

        self.embedding_norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, total_vocab, bias=False)

    def __call__(self, codes: mx.array) -> mx.array:
        """
        Embed audio codes using shared embedding with codebook offsets.

        Args:
            codes: (B, num_codebooks) or (num_codebooks,) audio codes

        Returns:
            Summed embeddings (B, dim) or (dim,)
        """
        if codes.ndim == 1:
            codes = codes[None, :]

        B, K = codes.shape

        # Apply codebook offsets: each codebook's tokens are offset to use separate region
        # codes[:, i] + i * vocab_size
        offset_codes = codes + self._codebook_offsets[:K]

        # (B, K) -> (B, K, dim) -> sum over K -> (B, dim)
        embedded = self.embedding(offset_codes).sum(axis=1)

        if codes.ndim == 1:
            return embedded.squeeze(0)

        return embedded


class AudioEmbeddingWithNorm(nn.Module):

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding_norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

    def embed(self, x: mx.array) -> mx.array:
        """Embed tokens with normalization."""
        return self.embedding_norm(self.embedding(x))

    def embed_raw(self, x: mx.array) -> mx.array:
        """Embed tokens WITHOUT normalization (used for conditioning)."""
        return self.embedding(x)

    def logits(self, x: mx.array) -> mx.array:
        """Project to vocabulary logits."""
        return self.to_logits(x)


class AudioHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        depthformer_config: DepthformerConfig,
        num_codebooks: int = 8,
        vocab_size: int = 2049,
        codebook_weight: str = "log",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.codebook_weight = codebook_weight
        self.depthformer_dim = depthformer_config.dim

        self.depthformer = Depthformer(
            layers=depthformer_config.layers,
            dim=depthformer_config.dim,
            num_heads=depthformer_config.num_heads,
            num_kv_heads=depthformer_config.num_kv_heads,
            tie=depthformer_config.tie,
        )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[List[Any]] = None,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[List[Any]]]:
        """
        Predict audio hidden states.

        Args:
            x: Hidden states from LFM (B, L, D)
            cache: Optional cache for streaming
            use_cache: Whether to return cache

        Returns:
            Hidden states per codebook (B, L, num_codebooks, dim), optional cache
        """
        B, L, D = x.shape

        x = x.reshape(B, L, self.num_codebooks, self.depthformer_dim)  # (B, L, 8, 1024)
        x = x.transpose(0, 2, 1, 3)  # (B, 8, L, 1024)
        x = x.reshape(B * self.num_codebooks, L, self.depthformer_dim)  # (B*8, L, 1024)

        x, new_cache = self.depthformer(x, cache, use_cache)  # (B*8, L, 1024)

        x = x.reshape(B, self.num_codebooks, L, self.depthformer_dim)  # (B, 8, L, 1024)
        x = x.transpose(0, 2, 1, 3)  # (B, L, 8, 1024)

        return x, new_cache


class LFM2AudioModel(nn.Module):

    def __init__(self, config: LFM2AudioConfig):
        super().__init__()
        self.config = config

        self.audio_encoder = ConformerEncoder(config.encoder)

        self.audio_adapter = MLP(
            in_channels=config.encoder.d_model,
            out_channels=config.lfm.hidden_size,
            hidden_dims=config.adapter_hidden_dims,
            use_layer_norm=config.adapter_use_layer_norm,
            dropout=config.adapter_dropout,
        )

        self.lfm = Lfm2Model(config.lfm)

        self.audio_embedding = AudioEmbedding(
            config.audio_vocab_size,
            config.lfm.hidden_size,
            config.codebooks,
            config.tie_audio_embeddings,
        )

        self.depth_embeddings = [
            AudioEmbeddingWithNorm(config.audio_vocab_size, config.depthformer.dim)
            for _ in range(config.codebooks)
        ]

        self.depth_linear = nn.Linear(
            config.lfm.hidden_size, config.codebooks * config.depthformer.dim
        )

        self.audio_head = AudioHead(
            config.lfm.hidden_size,
            config.depthformer,
            config.codebooks,
            config.audio_vocab_size,
            config.codebook_weight,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
    ) -> "LFM2AudioModel":
        # Download or get local path
        if Path(model_name_or_path).exists():
            model_path = Path(model_name_or_path)
        else:
            model_path = Path(
                snapshot_download(
                    model_name_or_path,
                    allow_patterns=["*.json", "*.safetensors", "*.bin"],
                )
            )

        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        config = LFM2AudioConfig.from_dict(config_dict)

        quantization = config_dict.get("quantization", None)

        # Create model
        model = cls(config)

        weight_files = [
            wf for wf in model_path.glob("*.safetensors") if "tokenizer" not in wf.name
        ]
        weights = {}
        if len(weight_files) > 0:

            for wf in weight_files:
                weights.update(mx.load(str(wf)))
        else:
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        # Sanitize and load weights

        weights = model.sanitize(weights)

        for key, value in weights.items():
            if value.dtype == mx.float32:
                if "conv" in key or "norm" in key:
                    continue
                else:
                    weights[key] = value.astype(mx.float16)

        if quantization:
            from ....convert import build_quant_predicate

            final_predicate = build_quant_predicate(model)
            nn.quantize(
                model,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                mode=config_dict.get("quantization_mode", "affine"),
                class_predicate=final_predicate,
            )

        model.load_weights(list(weights.items()), strict=True)

        mx.eval(model.parameters())
        model.eval()

        return model

    def model_quant_predicate(self, p, m):
        # Quantize if not norm or conv
        if "norm" in p or "conv" in p:
            return False
        else:
            return True

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        import re

        sanitized = {}

        # Skip buffers and Mimi codec weights
        skip_keys = [
            "audio_loss_weights",
            "codebook_offsets",
            "downsample.",
            "upsample.",
            ".num_batches_tracked",  # BatchNorm counter (not needed for inference)
            "pos_enc.pe",  # Positional encoding buffer (precomputed)
            ".freqs",  # RoPE frequencies (precomputed)
        ]

        for key, value in weights.items():
            # Skip certain keys
            if any(skip in key for skip in skip_keys):
                continue

            new_key = key

            # =========== Conformer Encoder ===========
            if key.startswith("conformer."):
                new_key = key.replace("conformer.", "audio_encoder.")

                # Layer norms
                new_key = new_key.replace(".norm_feed_forward1.", ".ff1_norm.")
                new_key = new_key.replace(".norm_feed_forward2.", ".ff2_norm.")
                new_key = new_key.replace(".norm_self_att.", ".attn_norm.")
                new_key = new_key.replace(".norm_conv.", ".conv_norm.")
                new_key = new_key.replace(".norm_out.", ".final_norm.")

                # Feed forward
                new_key = new_key.replace(".feed_forward1.", ".ff1.")
                new_key = new_key.replace(".feed_forward2.", ".ff2.")

                # Self attention
                new_key = new_key.replace(".self_attn.linear_q.", ".attn.q_proj.")
                new_key = new_key.replace(".self_attn.linear_k.", ".attn.k_proj.")
                new_key = new_key.replace(".self_attn.linear_v.", ".attn.v_proj.")
                new_key = new_key.replace(".self_attn.linear_out.", ".attn.out_proj.")
                new_key = new_key.replace(".self_attn.linear_pos.", ".attn.pos_proj.")
                new_key = new_key.replace(".self_attn.pos_bias_u", ".attn.pos_bias_u")
                new_key = new_key.replace(".self_attn.pos_bias_v", ".attn.pos_bias_v")

                # Conv module
                new_key = new_key.replace(".conv.batch_norm.", ".conv.norm.")

            # =========== Audio Adapter (MLP) ===========
            elif key.startswith("audio_adapter.model."):
                new_key = key.replace("audio_adapter.model.", "audio_adapter.layers.")

            # =========== LFM Backbone ===========
            elif key.startswith("lfm."):
                new_key = new_key.replace(".feed_forward.linear1.", ".feed_forward.w1.")
                new_key = new_key.replace(".feed_forward.linear2.", ".feed_forward.w2.")
                new_key = new_key.replace(".feed_forward.linear3.", ".feed_forward.w3.")

            # =========== Depthformer ===========
            elif key.startswith("depthformer."):

                match = re.match(r"depthformer\.layers\.(\d+)\.(.*)", key)
                if match:
                    layer_idx = int(match.group(1))
                    rest = match.group(2)

                    # operator.qkv_proj -> split QKV
                    if rest == "operator.qkv_proj.weight":
                        new_key = (
                            f"audio_head.depthformer.blocks.{layer_idx}.attn.qkv_weight"
                        )
                    elif rest == "operator.out_proj.weight":
                        new_key = f"audio_head.depthformer.blocks.{layer_idx}.attn.o_proj.weight"
                    elif rest == "operator.bounded_attention.q_layernorm.weight":
                        new_key = f"audio_head.depthformer.blocks.{layer_idx}.attn.q_norm.weight"
                    elif rest == "operator.bounded_attention.k_layernorm.weight":
                        new_key = f"audio_head.depthformer.blocks.{layer_idx}.attn.k_norm.weight"
                    elif rest.startswith("operator_norm."):
                        new_key = f"audio_head.depthformer.blocks.{layer_idx}.attn_norm.{rest.split('.', 1)[1]}"
                    elif rest.startswith("feed_forward."):
                        new_key = f"audio_head.depthformer.blocks.{layer_idx}.ffn.{rest.split('.', 1)[1]}"
                    elif rest.startswith("ffn_norm."):
                        new_key = f"audio_head.depthformer.blocks.{layer_idx}.ffn_norm.{rest.split('.', 1)[1]}"
                    else:
                        new_key = f"audio_head.depthformer.blocks.{layer_idx}.{rest}"

            sanitized[new_key] = value

        # =========== Post-process: Split combined QKV weights for depthformer ===========
        keys_to_remove = []
        keys_to_add = {}

        for key, value in sanitized.items():
            if ".attn.qkv_weight" in key:
                # Depthformer uses GQA: 32 Q heads, 8 KV heads, head_dim=32
                # Combined QKV: (1536, 1024) = Q(1024) + K(256) + V(256)
                q_dim = 1024  # 32 heads * 32 head_dim
                kv_dim = 256  # 8 heads * 32 head_dim

                q_weight = value[:q_dim, :]
                k_weight = value[q_dim : q_dim + kv_dim, :]
                v_weight = value[q_dim + kv_dim :, :]

                base_key = key.replace(".qkv_weight", "")
                keys_to_add[f"{base_key}.q_proj.weight"] = q_weight
                keys_to_add[f"{base_key}.k_proj.weight"] = k_weight
                keys_to_add[f"{base_key}.v_proj.weight"] = v_weight
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del sanitized[key]
        sanitized.update(keys_to_add)

        # =========== Post-process: Transpose Conv weights ===========
        for key, value in list(sanitized.items()):

            if "pointwise_conv" in key and "weight" in key and value.ndim == 3:
                sanitized[key] = value if value.ndim == 2 else value.squeeze(-1)
            elif ("depthwise_conv" in key or ".conv.weight" in key) and value.ndim == 3:
                sanitized[key] = (
                    value if check_array_shape(value) else value.transpose(0, 2, 1)
                )
            elif "pre_encode.conv" in key and value.ndim == 4:
                sanitized[key] = (
                    value if check_array_shape(value) else value.transpose(0, 2, 3, 1)
                )  # NCHW -> NHWC

        return sanitized

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def make_cache(self) -> List[Any]:
        return [
            KVCache() if layer.is_attention_layer else ArraysCache(size=1)
            for layer in self.lfm.layers
        ]

    def _embed_text(self, input_ids: mx.array) -> mx.array:
        return self.lfm.embed_tokens(input_ids)

    def _embed_audio_in(self, audio_codes: mx.array) -> mx.array:

        return self.audio_embedding(audio_codes)

    def _embed_audio_out(self, audio_codes: mx.array) -> mx.array:
        return self.audio_embedding(audio_codes)

    def _encode_audio(
        self,
        mel_features: mx.array,
        lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode audio features through conformer and adapter.

        Args:
            mel_features: Mel spectrogram features (B, T, D)
            lengths: Original lengths

        Returns:
            Encoded audio embeddings (B, T', D) and lengths
        """
        # Conformer encoding
        encoded, lengths = self.audio_encoder(mel_features, lengths)

        # MLP adapter to LFM dimension
        adapted = self.audio_adapter(encoded)

        return adapted, lengths

    def _prefill(
        self,
        text_tokens: Optional[mx.array] = None,
        audio_features: Optional[mx.array] = None,
        audio_codes: Optional[mx.array] = None,
        modalities: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> Tuple[mx.array, List[Any]]:
        """
        Prefill with mixed modality inputs.

        Args:
            text_tokens: Text token IDs (B, T_text)
            audio_features: Mel features for encoding (B, T_mel, D_mel)
            audio_codes: Pre-encoded audio codes (B, T_audio, num_codebooks)
            modalities: Modality flags (B, T_total) indicating type of each position
                        0=TEXT, 1=AUDIO_IN, 2=AUDIO_OUT
            cache: Optional KV cache (will be created if None)

        Returns:
            Hidden states and cache
        """
        if modalities is not None:
            input_embeddings = self._build_interleaved_embeddings(
                text_tokens, audio_features, audio_codes, modalities
            )
        else:
            embeddings = []

            if text_tokens is not None:
                text_emb = self._embed_text(text_tokens)
                embeddings.append(text_emb)

            if audio_features is not None:
                audio_emb, _ = self._encode_audio(audio_features)
                embeddings.append(audio_emb)

            if audio_codes is not None:
                B, T, _ = audio_codes.shape
                audio_out_emb = mx.zeros((B, T, self.config.lfm.hidden_size))
                for t in range(T):
                    audio_out_emb = audio_out_emb.at[:, t, :].add(
                        self._embed_audio_out(audio_codes[:, t, :])
                    )
                embeddings.append(audio_out_emb)

            if len(embeddings) > 1:
                input_embeddings = mx.concatenate(embeddings, axis=1)
            else:
                input_embeddings = embeddings[0]

        if cache is None:
            cache = self.make_cache()

        hidden_states = self.lfm(
            inputs=None,
            cache=cache,
            input_embeddings=input_embeddings,
        )

        return hidden_states, cache

    def _build_interleaved_embeddings(
        self,
        text_tokens: Optional[mx.array],
        audio_features: Optional[mx.array],
        audio_codes: Optional[mx.array],
        modalities: mx.array,
    ) -> mx.array:
        """
        Build embeddings interleaved according to modality flags.

        Args:
            text_tokens: Text token IDs (B, T_text)
            audio_features: Mel features (B, T_mel, D_mel)
            audio_codes: Audio codes (B, T_audio, num_codebooks)
            modalities: Modality flags (B, T_total)

        Returns:
            Interleaved embeddings (B, T_total, D)
        """
        B = modalities.shape[0]
        T_total = modalities.shape[1]
        D = self.config.lfm.hidden_size

        mods_flat = modalities[0].tolist()
        unique_mods = set(mods_flat)

        if unique_mods == {LFMModality.TEXT} and text_tokens is not None:
            return self._embed_text(text_tokens)

        if unique_mods == {LFMModality.AUDIO_IN} and audio_features is not None:
            return self._encode_audio(audio_features)[0]

        text_emb = None
        if text_tokens is not None:
            text_emb = self._embed_text(text_tokens)  # (B, T_text, D)

        audio_embedding = None
        if audio_features is not None:
            audio_embedding, _ = self._encode_audio(
                audio_features
            )  # (B, T_audio_in, D)

        audio_out_emb = None
        if audio_codes is not None:
            B_audio, T_audio, _ = audio_codes.shape
            audio_out_list = []
            for t in range(T_audio):
                audio_out_list.append(self._embed_audio_out(audio_codes[:, t, :]))
            audio_out_emb = mx.stack(audio_out_list, axis=1)  # (B, T_audio, D)

        text_positions = []
        audio_in_positions = []
        audio_out_positions = []

        for pos, mod in enumerate(mods_flat):
            if mod == LFMModality.TEXT:
                text_positions.append(pos)
            elif mod == LFMModality.AUDIO_IN:
                audio_in_positions.append(pos)
            elif mod == LFMModality.AUDIO_OUT:
                audio_out_positions.append(pos)

        embeddings = mx.zeros((B, T_total, D))

        if text_emb is not None and text_positions:
            n_text = min(len(text_positions), text_emb.shape[1])
            for i in range(n_text):
                pos = text_positions[i]
                embeddings = embeddings.at[:, pos : pos + 1, :].add(
                    text_emb[:, i : i + 1, :]
                )

        if audio_embedding is not None and audio_in_positions:
            n_audio_in = min(len(audio_in_positions), audio_embedding.shape[1])
            for i in range(n_audio_in):
                pos = audio_in_positions[i]
                embeddings = embeddings.at[:, pos : pos + 1, :].add(
                    audio_embedding[:, i : i + 1, :]
                )

        if audio_out_emb is not None and audio_out_positions:
            n_audio_out = min(len(audio_out_positions), audio_out_emb.shape[1])
            for i in range(n_audio_out):
                pos = audio_out_positions[i]
                embeddings = embeddings.at[:, pos : pos + 1, :].add(
                    audio_out_emb[:, i : i + 1, :]
                )

        return embeddings

    def _sample_text_token(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> mx.array:
        """Sample a text token from logits."""
        if temperature == 0:
            return mx.argmax(logits, axis=-1)

        logits = logits / temperature

        # Top-k filtering using argsort
        if top_k > 0 and top_k < logits.shape[-1]:
            # Get indices that would sort the logits (descending)
            sorted_indices = mx.argsort(-logits, axis=-1)
            # Get the k-th largest value as threshold
            kth_indices = sorted_indices[..., top_k - 1 : top_k]
            kth_values = mx.take_along_axis(logits, kth_indices, axis=-1)
            # Mask out values below threshold
            logits = mx.where(logits >= kth_values, logits, float("-inf"))

        # mx.random.categorical expects logits (it applies softmax internally)
        return mx.random.categorical(logits)

    def _sample_audio_frame(
        self,
        hidden_state: mx.array,
        audio_cache: Optional[List[Any]] = None,
        temperature: float = 1.0,
        top_k: int = 4,
    ) -> Tuple[mx.array, List[Any]]:
        """
        Sample audio tokens for all codebooks with sequential conditioning.

        Each codebook is conditioned on the previous codebook's sampled token,
        following the reference implementation's iterative approach.

        Args:
            hidden_state: LFM hidden state (B, 1, D)
            audio_cache: Depthformer cache for streaming
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Audio codes (B, num_codebooks) and updated cache
        """
        B = hidden_state.shape[0]

        # Project to depthformer inputs: (B, 1, D) -> (B, 1, 8*1024)
        depthformer_in = self.depth_linear(hidden_state)  # (B, 1, 8192)

        # Reshape to per-codebook inputs: (B, 1, 8, 1024)
        depthformer_in = depthformer_in.reshape(
            B, 1, self.config.codebooks, self.audio_head.depthformer_dim
        )

        # Initialize previous token embedding as zeros
        depthformer_token = mx.zeros((B, self.audio_head.depthformer_dim))

        # Initialize cache if not provided
        if audio_cache is None:
            audio_cache = [None] * self.audio_head.depthformer.layers_count

        codes = []
        greedy = temperature is None or temperature <= 0 or top_k == 1

        for i in range(self.config.codebooks):
            # Get input for this codebook and add previous token embedding
            cur_input = depthformer_in[:, :, i, :]  # (B, 1, 1024)
            cur_input = cur_input + depthformer_token[:, None, :]  # Add conditioning

            # Run through depthformer with caching
            depthformer_out, audio_cache = self.audio_head.depthformer(
                cur_input, cache=audio_cache, use_cache=True
            )

            # Get logits for this codebook
            logits = self.depth_embeddings[i].logits(
                depthformer_out[:, -1, :]
            )  # (B, vocab)

            # Sample token
            if greedy:
                code = mx.argmax(logits, axis=-1, keepdims=True)
            else:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0 and top_k < logits.shape[-1]:
                    sorted_indices = mx.argsort(-logits, axis=-1)
                    kth_indices = sorted_indices[:, top_k - 1 : top_k]
                    kth_values = mx.take_along_axis(logits, kth_indices, axis=-1)
                    logits = mx.where(logits >= kth_values, logits, float("-inf"))

                # mx.random.categorical expects logits (it applies softmax internally)
                code = mx.random.categorical(logits)[:, None]

            codes.append(code.squeeze(-1))

            # Get raw embedding for next codebook conditioning (no norm - matches PyTorch)
            depthformer_token = self.depth_embeddings[i].embed_raw(
                code.squeeze(-1)
            )  # (B, 1024)

        return mx.stack(codes, axis=-1), audio_cache

    def generate_interleaved(
        self,
        text_tokens: Optional[mx.array] = None,
        audio_features: Optional[mx.array] = None,
        audio_codes: Optional[mx.array] = None,
        modalities: Optional[mx.array] = None,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        audio_temperature: float = 1.0,
        audio_top_k: int = 4,
        interleaved_n_text: Optional[int] = None,
        interleaved_n_audio: Optional[int] = None,
    ) -> Generator[mx.array, None, None]:
        """
        Generate tokens in interleaved text/audio mode.

        Alternates between generating text and audio tokens in fixed patterns.

        Args:
            text_tokens: Input text tokens
            audio_features: Input audio mel features
            audio_codes: Previous audio codes (for continuation)
            max_new_tokens: Maximum tokens to generate
            temperature: Text sampling temperature
            top_k: Text top-k sampling
            audio_temperature: Audio sampling temperature
            audio_top_k: Audio top-k sampling
            interleaved_n_text: Number of text tokens per group
            interleaved_n_audio: Number of audio frames per group

        Yields:
            Generated tokens (single element for text, 8 elements for audio)
        """
        n_text = interleaved_n_text or self.config.interleaved_n_text
        n_audio = interleaved_n_audio or self.config.interleaved_n_audio

        # Prefill with modality-aware embedding
        hidden_states, cache = self._prefill(
            text_tokens=text_tokens,
            audio_features=audio_features,
            audio_codes=audio_codes,
            modalities=modalities,
        )

        # Get last hidden state
        last_hidden = hidden_states[:, -1:, :]

        generated = 0
        modality_left = n_text  # Start with n_text tokens to generate
        text_done = False
        current_modality = LFMModality.TEXT

        while generated < max_new_tokens:
            if current_modality == LFMModality.TEXT:
                # Generate text token
                text_logits = self.lfm.embed_tokens.as_linear(last_hidden)[:, -1, :]
                text_token = self._sample_text_token(text_logits, temperature, top_k)
                token_id = text_token.item()

                # Check for im_end - stop generation
                if token_id == IM_END_TOKEN:
                    break

                yield text_token, LFMModality.TEXT

                # Check for text_end token - marks text generation complete
                if token_id == TEXT_END_TOKEN:
                    text_done = True

                # Embed and continue
                next_emb = self._embed_text(text_token[:, None])
                last_hidden = self.lfm(
                    inputs=None,
                    cache=cache,
                    input_embeddings=next_emb,
                )

                modality_left -= 1
                generated += 1

                # Switch to audio after n_text tokens
                if modality_left <= 0 or text_done:
                    modality_left = n_audio
                    current_modality = LFMModality.AUDIO_OUT

            else:  # AUDIO_OUT mode
                # Generate audio frame with sequential codebook conditioning
                audio_frame, _ = self._sample_audio_frame(
                    last_hidden,
                    audio_cache=None,  # Fresh cache for each frame
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                )

                # Check for audio EOS
                if audio_frame[0, 0].item() == AUDIO_EOS_TOKEN:
                    # Set all codebooks to EOS
                    audio_frame = mx.full(
                        audio_frame.shape, AUDIO_EOS_TOKEN, dtype=audio_frame.dtype
                    )
                    yield audio_frame.squeeze(0), LFMModality.AUDIO_OUT
                    generated += 1
                    # If text is done, break after final audio EOS
                    if text_done:
                        break
                    # Otherwise switch back to text mode
                    modality_left = n_text
                    current_modality = LFMModality.TEXT
                    continue

                yield audio_frame.squeeze(0), LFMModality.AUDIO_OUT

                # Embed and continue
                next_emb = self._embed_audio_out(audio_frame)[:, None, :]
                last_hidden = self.lfm(
                    inputs=None,
                    cache=cache,
                    input_embeddings=next_emb,
                )

                modality_left -= 1
                generated += 1

                # Switch back to text after n_audio tokens only if text not done
                if modality_left <= 0 and not text_done:
                    modality_left = n_text
                    current_modality = LFMModality.TEXT

    def generate_sequential(
        self,
        text_tokens: Optional[mx.array] = None,
        audio_features: Optional[mx.array] = None,
        audio_codes: Optional[mx.array] = None,
        modalities: Optional[mx.array] = None,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        audio_temperature: float = 1.0,
        audio_top_k: int = 4,
    ) -> Generator[Tuple[mx.array, LFMModality], None, None]:
        """
        Generate tokens in sequential mode.

        The model autonomously decides when to switch between text and audio.

        Args:
            text_tokens: Input text tokens
            audio_features: Input audio mel features
            audio_codes: Previous audio codes
            max_new_tokens: Maximum tokens to generate
            temperature: Text sampling temperature
            top_k: Text top-k sampling
            audio_temperature: Audio sampling temperature
            audio_top_k: Audio top-k sampling

        Yields:
            Tuples of (token, modality)
        """
        # Prefill with modality-aware embedding
        hidden_states, cache = self._prefill(
            text_tokens=text_tokens,
            audio_features=audio_features,
            audio_codes=audio_codes,
            modalities=modalities,
        )

        last_hidden = hidden_states[:, -1:, :]

        # Detect initial modality from input - if last token was AUDIO_START, start in audio mode
        if text_tokens is not None and text_tokens[0, -1].item() == AUDIO_START_TOKEN:
            current_modality = LFMModality.AUDIO_OUT
        else:
            current_modality = LFMModality.TEXT

        generated = 0

        while generated < max_new_tokens:
            if current_modality == LFMModality.TEXT:
                # Generate text token
                text_logits = self.lfm.embed_tokens.as_linear(last_hidden)[:, -1, :]
                text_token = self._sample_text_token(text_logits, temperature, top_k)
                token_id = text_token.item()

                # Check for end of turn
                if token_id == IM_END_TOKEN:
                    yield text_token, LFMModality.TEXT
                    break

                # Check for audio start - switch to audio mode
                if token_id == AUDIO_START_TOKEN:
                    current_modality = LFMModality.AUDIO_OUT
                    # Embed audio_start token and continue
                    next_emb = self._embed_text(text_token[:, None])
                    last_hidden = self.lfm(
                        inputs=None,
                        cache=cache,
                        input_embeddings=next_emb,
                    )
                    continue

                yield text_token, LFMModality.TEXT

                # Embed and continue
                next_emb = self._embed_text(text_token[:, None])
                last_hidden = self.lfm(
                    inputs=None,
                    cache=cache,
                    input_embeddings=next_emb,
                )

            else:  # AUDIO_OUT mode
                # Generate audio frame with sequential codebook conditioning
                audio_frame, _ = self._sample_audio_frame(
                    last_hidden,
                    audio_cache=None,  # Fresh cache for each frame
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                )

                # Check for audio EOS - switch back to text mode
                if audio_frame[0, 0].item() == AUDIO_EOS_TOKEN:
                    # Set all codebooks to EOS
                    audio_frame = mx.full(
                        audio_frame.shape, AUDIO_EOS_TOKEN, dtype=audio_frame.dtype
                    )
                    current_modality = LFMModality.TEXT

                yield audio_frame.squeeze(0), LFMModality.AUDIO_OUT

                # Embed and continue
                next_emb = self._embed_audio_out(audio_frame)[:, None, :]
                last_hidden = self.lfm(
                    inputs=None,
                    cache=cache,
                    input_embeddings=next_emb,
                )

            generated += 1

    def __call__(
        self,
        text_tokens: Optional[mx.array] = None,
        audio_features: Optional[mx.array] = None,
        audio_codes: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        """
        Forward pass for training.

        Args:
            text_tokens: Text token IDs (B, T_text)
            audio_features: Mel features (B, T_mel, D_mel)
            audio_codes: Audio codes (B, T_audio, num_codebooks)

        Returns:
            Text logits and list of audio logits per codebook
        """
        hidden_states, _ = self._prefill(
            text_tokens=text_tokens,
            audio_features=audio_features,
            audio_codes=audio_codes,
        )

        # Text head
        text_logits = self.lfm.embed_tokens.as_linear(hidden_states)
        hidden_states = self.depth_linear(hidden_states)

        # Audio head - get hidden states per codebook
        audio_hidden, _ = self.audio_head(hidden_states)  # (B, L, 8, 1024)

        # Apply logits projection for each codebook using depth_embeddings
        audio_logits = [
            self.depth_embeddings[i].logits(audio_hidden[:, :, i, :])
            for i in range(self.config.codebooks)
        ]

        return text_logits, audio_logits

    def generate_from_chat_state(
        self,
        chat_state: Any,  # ChatState from processor
        mode: str = "interleaved",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_k: int = 4,
    ) -> Generator[Tuple[mx.array, LFMModality], None, None]:
        """
        Generate from a ChatState with proper modality handling.

        Args:
            chat_state: ChatState object from processor
            mode: "interleaved" or "sequential"
            max_new_tokens: Maximum tokens to generate
            temperature: Text sampling temperature
            top_k: Text top-k sampling
            audio_temperature: Audio sampling temperature
            audio_top_k: Audio top-k sampling

        Yields:
            Tuples of (token, modality)
        """
        # Extract data from ChatState
        text_tokens = chat_state.get_text_tokens()
        audio_features = chat_state.get_audio_features()
        modalities = chat_state.get_modalities()

        if mode == "interleaved":
            yield from self.generate_interleaved(
                text_tokens=text_tokens,
                audio_features=audio_features,
                modalities=modalities,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                audio_temperature=audio_temperature,
                audio_top_k=audio_top_k,
            )
        else:
            yield from self.generate_sequential(
                text_tokens=text_tokens,
                audio_features=audio_features,
                modalities=modalities,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                audio_temperature=audio_temperature,
                audio_top_k=audio_top_k,
            )
