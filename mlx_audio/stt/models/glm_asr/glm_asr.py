"""GLM-ASR model for speech-to-text transcription using MLX."""

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.utils import get_model_path

from .config import LlamaConfig, ModelConfig, WhisperConfig


@dataclass
class STTOutput:
    """Output from speech-to-text generation."""

    text: str
    segments: List[dict] = None
    language: str = None


class RotaryEmbedding:
    """Rotary Position Embedding implementation."""

    def __init__(self, dim: int, rope_ratio: float = 1.0):
        self.dim = dim
        self.rope_ratio = rope_ratio

    def get_emb(self, max_seq_len: int, dtype: mx.Dtype, base: int = 10000) -> mx.array:
        """Get rotary embeddings for the given sequence length."""
        base = base * self.rope_ratio
        n_elem = self.dim

        theta = 1.0 / (base ** (mx.arange(0, n_elem, 2, dtype=mx.float32) / n_elem))
        seq_idx = mx.arange(max_seq_len, dtype=mx.float32)
        idx_theta = mx.outer(seq_idx, theta)

        cos_emb = mx.cos(idx_theta)
        sin_emb = mx.sin(idx_theta)
        cache = mx.stack([cos_emb, sin_emb], axis=-1)

        if dtype in (mx.float16, mx.bfloat16):
            cache = cache.astype(dtype)

        return cache


def apply_rotary_pos_emb(x: mx.array, rope_cache: mx.array) -> mx.array:
    """Apply rotary positional embeddings to input tensor."""
    b, np_heads, sq, _ = x.shape
    rot_dim = rope_cache.shape[-2] * 2

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    xshaped = x_rot.reshape(b, np_heads, sq, rot_dim // 2, 2)
    rope_cache = rope_cache[:, :sq].reshape(1, 1, sq, -1, 2)

    x_out = mx.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        axis=-1,
    )
    x_out = x_out.reshape(b, np_heads, sq, rot_dim)

    return mx.concatenate([x_out, x_pass], axis=-1)


class WhisperAttention(nn.Module):
    """Whisper attention layer with optional Rotary Position Embeddings."""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        bsz, tgt_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if rotary_pos_emb is not None:
            query_states = apply_rotary_pos_emb(query_states, rotary_pos_emb)
            key_states = apply_rotary_pos_emb(key_states, rotary_pos_emb)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=self.scaling
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, tgt_len, self.embed_dim
        )

        return self.out_proj(attn_output)


class WhisperEncoderLayer(nn.Module):
    """Whisper encoder layer with optional RoPE support."""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_pos_emb=rotary_pos_emb)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nn.Module):
    """Whisper encoder with optional rotary position embeddings."""

    def __init__(self, config: WhisperConfig, use_rope: bool = False):
        super().__init__()
        self.config = config
        self.use_rope = use_rope
        embed_dim = config.d_model

        self.conv1 = nn.Conv1d(config.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        if use_rope:
            self.rotary_embedding = RotaryEmbedding(
                config.d_model // config.encoder_attention_heads // 2
            )
            self.embed_positions = nn.Embedding(config.max_source_positions, embed_dim)
        else:
            self.embed_positions = nn.Embedding(config.max_source_positions, embed_dim)
            self.rotary_embedding = None

        self.layers = [
            WhisperEncoderLayer(config) for _ in range(config.encoder_layers)
        ]

    def __call__(self, input_features: mx.array) -> mx.array:
        """Encode audio features."""
        hidden_states = nn.gelu(self.conv1(input_features))
        hidden_states = nn.gelu(self.conv2(hidden_states))

        seq_len = hidden_states.shape[1]

        if self.use_rope and self.rotary_embedding is not None:
            rotary_embs = self.rotary_embedding.get_emb(seq_len, hidden_states.dtype)
            rotary_embs = rotary_embs[None]
        else:
            rotary_embs = None
            embed_pos = self.embed_positions.weight[:seq_len]
            hidden_states = hidden_states + embed_pos

        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_pos_emb=rotary_embs)

        return hidden_states


class AdaptingMLP(nn.Module):
    """MLP adapter for audio-to-LM projection."""

    def __init__(self, input_dim: int, intermediate_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim, bias=True)
        self.fc2 = nn.Linear(intermediate_dim, output_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class AudioEncoder(nn.Module):
    """Audio encoder with Whisper backbone and MLP adapter.

    This matches the HuggingFace weight structure exactly:
    - audio_encoder.whisper.*
    - audio_encoder.layer_norm.*
    - audio_encoder.proj.*
    - audio_encoder.adapting.*
    - audio_encoder.audio_bos_eos_token.*
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        whisper_config = config.whisper_config
        lm_hidden_size = config.lm_config.hidden_size

        # Whisper encoder
        self.whisper = WhisperEncoder(whisper_config, use_rope=config.use_rope)

        # Layer norm after whisper encoder
        self.layer_norm = nn.LayerNorm(whisper_config.d_model)

        # Projection from whisper dim to intermediate
        # HF model: proj goes from 1280 -> 2048
        self.proj = nn.Linear(whisper_config.d_model, lm_hidden_size, bias=True)

        # MLP adapter: matches HF structure with layers 0 and 2
        # Layer 0: 5120 -> 4096 (merge_factor * whisper_dim -> intermediate)
        # Layer 2: 4096 -> 2048 (intermediate -> lm_hidden)
        merged_dim = whisper_config.d_model * config.merge_factor
        intermediate_dim = lm_hidden_size * 2  # 4096 for this model

        # Use a custom module to match HF weight naming (adapting.0.*, adapting.2.*)
        self.adapting = AdaptingMLP(merged_dim, intermediate_dim, lm_hidden_size)

        # Begin/End of audio token embeddings
        self.audio_bos_eos_token = nn.Embedding(2, lm_hidden_size)

    def __call__(self, input_features: mx.array) -> Tuple[mx.array, int]:
        """Encode audio features and project to LM space."""
        # Whisper encoding
        audio_features = self.whisper(input_features)

        # Layer norm
        audio_features = self.layer_norm(audio_features)

        # Merge audio features by merge_factor
        batch_size, seq_len, _ = audio_features.shape
        merge_factor = self.config.merge_factor

        new_seq_len = (seq_len - merge_factor) // merge_factor + 1
        max_len = self.config.max_whisper_length // merge_factor
        new_seq_len = min(new_seq_len, max_len)

        merged_features = []
        for i in range(new_seq_len):
            start_idx = i * merge_factor
            end_idx = start_idx + merge_factor
            chunk = audio_features[:, start_idx:end_idx, :]
            chunk = chunk.reshape(batch_size, -1)
            merged_features.append(chunk)

        merged_audio = mx.stack(merged_features, axis=1)

        # Project through MLP adapter
        audio_embeds = self.adapting(merged_audio)

        return audio_embeds, new_seq_len

    def get_boa_eoa_tokens(self) -> Tuple[mx.array, mx.array]:
        """Get begin-of-audio and end-of-audio token embeddings."""
        boa = self.audio_bos_eos_token(mx.array([0]))
        eoa = self.audio_bos_eos_token(mx.array([1]))
        return boa, eoa


class LlamaAttention(nn.Module):
    """Multi-head attention with grouped query attention support."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class LlamaMLP(nn.Module):
    """LLaMA MLP with SiLU activation."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """LLaMA decoder layer."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class LlamaModel(nn.Module):
    """LLaMA language model backbone."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        if input_embeddings is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeddings

        if mask is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(
                hidden_states.shape[1]
            )
            mask = mask.astype(hidden_states.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, mask=mask, cache=cache[i])

        return self.norm(hidden_states)


class Model(nn.Module):
    """GLM-ASR model combining Whisper encoder with LLaMA decoder.

    Weight structure matches HuggingFace format:
    - audio_encoder.* : Audio encoder with Whisper + MLP adapter
    - model.* : LLaMA decoder
    - lm_head.* : Language modeling head
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.lm_config.vocab_size

        # Audio encoder (matches HF naming: audio_encoder.*)
        self.audio_encoder = AudioEncoder(config)

        # LLaMA model (matches HF naming: model.*)
        self.model = LlamaModel(config.lm_config)

        # LM head (matches HF naming: lm_head.*)
        if not config.lm_config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.lm_config.hidden_size, self.vocab_size, bias=False
            )

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the input embeddings from the language model."""
        return self.model.embed_tokens

    def _merge_audio_text_embeddings(
        self,
        input_ids: mx.array,
        audios: Optional[mx.array] = None,
        audio_offsets: Optional[List[List[int]]] = None,
        audio_length: Optional[List[List[int]]] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """Merge audio embeddings into text embeddings at specified positions."""
        text_embeds = self.get_input_embeddings()(input_ids)

        if audios is None or (cache is not None and cache[0] is not None):
            return text_embeds

        audio_embeds, _ = self.audio_encoder(audios)

        batch_size = text_embeds.shape[0]

        for b in range(batch_size):
            if audio_offsets is not None and len(audio_offsets) > b:
                offsets = audio_offsets[b]
                lengths = audio_length[b] if audio_length else [audio_embeds.shape[1]]

                audio_idx = 0
                for offset, length in zip(offsets, lengths):
                    if audio_idx < audio_embeds.shape[0]:
                        audio_chunk = audio_embeds[audio_idx, :length]
                        end_pos = min(offset + length, text_embeds.shape[1])
                        actual_length = end_pos - offset
                        text_embeds[b, offset:end_pos] = audio_chunk[:actual_length]
                        audio_idx += 1

        return text_embeds

    def __call__(
        self,
        input_ids: mx.array,
        audios: Optional[mx.array] = None,
        audio_offsets: Optional[List[List[int]]] = None,
        audio_length: Optional[List[List[int]]] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        """Forward pass."""
        input_embeds = self._merge_audio_text_embeddings(
            input_ids=input_ids,
            audios=audios,
            audio_offsets=audio_offsets,
            audio_length=audio_length,
            cache=cache,
        )

        hidden_states = self.model(input_embeddings=input_embeds, cache=cache)

        if self.config.lm_config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)

        return logits

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights for loading."""
        sanitized = {}
        for k, v in weights.items():
            new_key = k

            # Remap adapting layer names: 0 -> fc1, 2 -> fc2
            if "audio_encoder.adapting.0." in k:
                new_key = k.replace(
                    "audio_encoder.adapting.0.", "audio_encoder.adapting.fc1."
                )
            elif "audio_encoder.adapting.2." in k:
                new_key = k.replace(
                    "audio_encoder.adapting.2.", "audio_encoder.adapting.fc2."
                )

            # Handle conv weight transposition
            if "conv" in new_key and "weight" in new_key:
                if v.ndim == 3 and v.shape[-1] < v.shape[-2]:
                    sanitized[new_key] = v.transpose(0, 2, 1)
                else:
                    sanitized[new_key] = v
            else:
                sanitized[new_key] = v
        return sanitized

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[ModelConfig] = None,
        **kwargs,
    ) -> "Model":
        """Load model from pretrained weights."""
        from transformers import AutoTokenizer

        revision = kwargs.get("revision", None)
        force_download = kwargs.get("force_download", False)
        model_path_resolved = get_model_path(
            model_path, revision=revision, force_download=force_download
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if config is None:
            config_path = Path(model_path_resolved) / "config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config = ModelConfig.from_dict(config_dict)

        model = cls(config)
        model._tokenizer = tokenizer

        weights = {}
        weight_files = glob.glob(str(Path(model_path_resolved) / "model*.safetensors"))
        if not weight_files:
            weight_files = glob.glob(str(Path(model_path_resolved) / "*.safetensors"))

        for file in weight_files:
            weights.update(mx.load(file))

        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))

        mx.eval(model.parameters())
        return model

    def _preprocess_audio(self, audio) -> mx.array:
        """Preprocess audio to mel spectrogram.

        Args:
            audio: Audio path (str), waveform (np.ndarray/mx.array), or mel spectrogram

        Returns:
            Mel spectrogram of shape (batch, seq_len, n_mels)
        """
        from mlx_audio.stt.utils import load_audio
        from mlx_audio.utils import hanning, mel_filters, stft

        # Audio hyperparameters for GLM-ASR (128 mel bins)
        SAMPLE_RATE = 16000
        N_FFT = 400
        HOP_LENGTH = 160
        N_MELS = self.config.whisper_config.num_mel_bins  # 128

        # Load audio if path
        if isinstance(audio, str):
            audio = load_audio(audio, sr=SAMPLE_RATE)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        # If already 3D (batch, seq, mels), assume it's mel spectrogram
        if audio.ndim == 3:
            return audio

        # Compute mel spectrogram
        window = hanning(N_FFT)
        freqs = stft(audio, window=window, n_fft=N_FFT, hop_length=HOP_LENGTH)
        magnitudes = freqs[:-1, :].abs().square()

        filters = mel_filters(SAMPLE_RATE, N_FFT, N_MELS, norm="slaney", mel_scale=None)
        mel_spec = magnitudes @ filters.T

        log_spec = mx.maximum(mel_spec, 1e-10).log10()
        log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # Add batch dimension: (seq_len, n_mels) -> (1, seq_len, n_mels)
        return log_spec[None]

    def generate(
        self,
        audio,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        verbose: bool = False,
    ) -> STTOutput:
        """Generate transcription from audio.

        Args:
            audio: Audio path (str), waveform (mx.array), or mel spectrogram
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            verbose: Print tokens during generation

        Returns:
            STTOutput with transcription text
        """
        # Preprocess audio to mel spectrogram
        mel = self._preprocess_audio(audio)

        prompt_text = "<|user|>\n<|begin_of_audio|>"
        tokens = self._tokenizer.encode(prompt_text)

        # Get audio length after encoding
        _, audio_len = self.audio_encoder(mel)

        audio_placeholder_tokens = [0] * audio_len
        tokens.extend(audio_placeholder_tokens)

        end_prompt = (
            "<|end_of_audio|>\nPlease transcribe this audio into text<|assistant|>\n"
        )
        tokens.extend(self._tokenizer.encode(end_prompt))

        input_ids = mx.array([tokens])

        audio_start = len(self._tokenizer.encode("<|user|>\n<|begin_of_audio|>"))
        audio_offsets = [[audio_start]]
        audio_length = [[audio_len]]

        input_embeds = self._merge_audio_text_embeddings(
            input_ids=input_ids,
            audios=mel,
            audio_offsets=audio_offsets,
            audio_length=audio_length,
        )

        generated_tokens = []

        # Build full sequence with embeddings
        all_tokens = tokens.copy()

        for step in range(max_tokens):
            # Create input_ids for current sequence
            current_ids = mx.array([all_tokens])

            # Forward pass - we need to rebuild embeddings each time for now
            # since we're doing simple generation without KV cache
            if step == 0:
                # First pass: use the merged embeddings
                hidden_states = self.model(input_embeddings=input_embeds)
            else:
                # Subsequent passes: use token embeddings for full sequence
                full_embeds = self.model.embed_tokens(current_ids)
                # Replace audio placeholder positions with audio embeddings
                audio_embeds, _ = self.audio_encoder(mel)
                for offset, length in zip(audio_offsets[0], audio_length[0]):
                    end_pos = min(offset + length, full_embeds.shape[1])
                    actual_length = end_pos - offset
                    full_embeds[0, offset:end_pos] = audio_embeds[0, :actual_length]
                hidden_states = self.model(input_embeddings=full_embeds)

            if self.config.lm_config.tie_word_embeddings:
                logits = self.model.embed_tokens.as_linear(hidden_states)
            else:
                logits = self.lm_head(hidden_states)

            # Get next token from last position
            if temperature == 0:
                next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
            else:
                probs = mx.softmax(logits[:, -1, :] / temperature, axis=-1)
                next_token = mx.random.categorical(probs).item()

            # Check for EOS
            if next_token in self.config.lm_config.eos_token_id:
                break

            generated_tokens.append(next_token)
            all_tokens.append(next_token)

            if verbose:
                print(self._tokenizer.decode([next_token]), end="", flush=True)

            mx.eval(logits)

        if verbose:
            print()

        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        mx.clear_cache()

        return STTOutput(text=text.strip())
