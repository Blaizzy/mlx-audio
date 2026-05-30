"""Voxtral-4B-TTS for MLX."""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult

from .acoustic_head import AcousticTransformerArgs, FlowMatchingAudioTransformer
from .audio_tokenizer import AudioTokenizerArgs, VoxtralTTSAudioTokenizer
from .common import pad_to_multiple
from .text_preprocess import sanitize_tts_input_text_for_demo


def is_tekken(path: str | Path) -> bool:
    """Return True when ``path`` points to a tekken tokenizer JSON file."""
    if isinstance(path, str):
        path = Path(path)
    return path.is_file() and path.suffix == ".json" and "tekken" in path.name


# Voice name -> index mapping
VOICE_MAP = {
    "casual_female": 0,
    "casual_male": 1,
    "cheerful_female": 2,
    "neutral_female": 3,
    "neutral_male": 4,
    "pt_male": 5,
    "pt_female": 6,
    "nl_male": 7,
    "nl_female": 8,
    "it_male": 9,
    "it_female": 10,
    "fr_male": 11,
    "fr_female": 12,
    "es_male": 13,
    "es_female": 14,
    "de_male": 15,
    "de_female": 16,
    "ar_male": 17,
    "hi_male": 18,
    "hi_female": 19,
}


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "voxtral_tts"
    # LM backbone args (Mistral-like)
    dim: int = 3072
    n_layers: int = 26
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 131072
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5
    tied_embeddings: bool = True
    use_biases: bool = False
    max_position_embeddings: int = 128000

    # Sample rate
    sample_rate: int = 24000

    # Audio model args (nested in params.json under multimodal.audio_model_args)
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36

    # Audio encoding
    frame_rate: float = 12.5
    num_codebooks: int = 37

    # Acoustic transformer args
    acoustic_dim: int = 3072
    acoustic_n_layers: int = 3
    acoustic_head_dim: int = 128
    acoustic_hidden_dim: int = 9216
    acoustic_n_heads: int = 32
    acoustic_n_kv_heads: int = 8
    acoustic_rope_theta: float = 10000.0
    acoustic_sigma: float = 1e-5
    acoustic_sigma_max: float = 1.0

    # Audio tokenizer args
    tokenizer_dim: int = 1024
    tokenizer_hidden_dim: int = 4096
    tokenizer_n_heads: int = 8
    tokenizer_n_kv_heads: int = 8
    tokenizer_head_dim: int = 128
    tokenizer_patch_size: int = 240
    tokenizer_patch_proj_kernel_size: int = 7
    tokenizer_semantic_dim: int = 256
    tokenizer_acoustic_dim: int = 36
    tokenizer_norm_eps: float = 0.01
    tokenizer_decoder_transformer_lengths_str: str = "2,2,2,2"
    tokenizer_decoder_convs_kernels_str: str = "3,4,4,4"
    tokenizer_decoder_convs_strides_str: str = "1,2,2,2"

    # Special tokens
    bos_token_id: int = 1
    audio_token_id: int = 24
    begin_audio_token_id: int = 25

    @classmethod
    def from_dict(cls, params: dict) -> "ModelConfig":
        """Create config from either config.json or params.json format."""
        # If already in our format, use BaseModelArgs
        if "model_type" in params and params.get("model_type") == "voxtral_tts":
            if "multimodal" not in params:
                return super().from_dict(params)

        # Parse from Mistral's params.json format
        mm = params.get("multimodal", {})
        audio_args = mm.get("audio_model_args", {})
        encoding_args = audio_args.get("audio_encoding_args", {})
        acoustic_args = audio_args.get("acoustic_transformer_args", {})
        tokenizer_args = mm.get("audio_tokenizer_args", {})

        return cls(
            model_type="voxtral_tts",
            dim=params.get("dim", 3072),
            n_layers=params.get("n_layers", 26),
            head_dim=params.get("head_dim", 128),
            hidden_dim=params.get("hidden_dim", 9216),
            n_heads=params.get("n_heads", 32),
            n_kv_heads=params.get("n_kv_heads", 8),
            vocab_size=params.get("vocab_size", 131072),
            rope_theta=params.get("rope_theta", 1000000.0),
            norm_eps=params.get("norm_eps", 1e-5),
            tied_embeddings=params.get("tied_embeddings", True),
            use_biases=params.get("use_biases", False),
            max_position_embeddings=params.get("max_position_embeddings", 128000),
            sample_rate=encoding_args.get("sampling_rate", 24000),
            semantic_codebook_size=audio_args.get("semantic_codebook_size", 8192),
            acoustic_codebook_size=audio_args.get("acoustic_codebook_size", 21),
            n_acoustic_codebook=audio_args.get("n_acoustic_codebook", 36),
            frame_rate=encoding_args.get("frame_rate", 12.5),
            num_codebooks=encoding_args.get("num_codebooks", 37),
            acoustic_dim=acoustic_args.get("dim", 3072),
            acoustic_n_layers=acoustic_args.get("n_layers", 3),
            acoustic_head_dim=acoustic_args.get("head_dim", 128),
            acoustic_hidden_dim=acoustic_args.get("hidden_dim", 9216),
            acoustic_n_heads=acoustic_args.get("n_heads", 32),
            acoustic_n_kv_heads=acoustic_args.get("n_kv_heads", 8),
            acoustic_rope_theta=acoustic_args.get("rope_theta", 10000.0),
            acoustic_sigma=acoustic_args.get("sigma", 1e-5),
            acoustic_sigma_max=acoustic_args.get("sigma_max", 1.0),
            tokenizer_dim=tokenizer_args.get("dim", 1024),
            tokenizer_hidden_dim=tokenizer_args.get("hidden_dim", 4096),
            tokenizer_n_heads=tokenizer_args.get("n_heads", 8),
            tokenizer_n_kv_heads=tokenizer_args.get("n_kv_heads", 8),
            tokenizer_head_dim=tokenizer_args.get("head_dim", 128),
            tokenizer_patch_size=tokenizer_args.get("pretransform_patch_size", 240),
            tokenizer_patch_proj_kernel_size=tokenizer_args.get(
                "patch_proj_kernel_size", 7
            ),
            tokenizer_semantic_dim=tokenizer_args.get("semantic_dim", 256),
            tokenizer_acoustic_dim=tokenizer_args.get("acoustic_dim", 36),
            tokenizer_norm_eps=tokenizer_args.get("norm_eps", 0.01),
            tokenizer_decoder_transformer_lengths_str=tokenizer_args.get(
                "decoder_transformer_lengths_str", "2,2,2,2"
            ),
            tokenizer_decoder_convs_kernels_str=tokenizer_args.get(
                "decoder_convs_kernels_str", "3,4,4,4"
            ),
            tokenizer_decoder_convs_strides_str=tokenizer_args.get(
                "decoder_convs_strides_str", "1,2,2,2"
            ),
            bos_token_id=mm.get("bos_token_id", 1),
            audio_token_id=audio_args.get("audio_token_id", 24),
            begin_audio_token_id=audio_args.get("begin_audio_token_id", 25),
        )

    def get_acoustic_args(self) -> AcousticTransformerArgs:
        return AcousticTransformerArgs(
            input_dim=self.dim,
            dim=self.acoustic_dim,
            n_layers=self.acoustic_n_layers,
            head_dim=self.acoustic_head_dim,
            hidden_dim=self.acoustic_hidden_dim,
            n_heads=self.acoustic_n_heads,
            n_kv_heads=self.acoustic_n_kv_heads,
            use_biases=self.use_biases,
            rope_theta=self.acoustic_rope_theta,
            sigma=self.acoustic_sigma,
            sigma_max=self.acoustic_sigma_max,
            norm_eps=self.norm_eps,
            semantic_codebook_size=self.semantic_codebook_size,
            acoustic_codebook_size=self.acoustic_codebook_size,
            n_acoustic_codebook=self.n_acoustic_codebook,
        )

    def get_tokenizer_args(self) -> AudioTokenizerArgs:
        return AudioTokenizerArgs(
            sampling_rate=self.sample_rate,
            pretransform_patch_size=self.tokenizer_patch_size,
            patch_proj_kernel_size=self.tokenizer_patch_proj_kernel_size,
            semantic_codebook_size=self.semantic_codebook_size,
            semantic_dim=self.tokenizer_semantic_dim,
            acoustic_codebook_size=self.acoustic_codebook_size,
            acoustic_dim=self.tokenizer_acoustic_dim,
            dim=self.tokenizer_dim,
            hidden_dim=self.tokenizer_hidden_dim,
            n_heads=self.tokenizer_n_heads,
            n_kv_heads=self.tokenizer_n_kv_heads,
            head_dim=self.tokenizer_head_dim,
            norm_eps=self.tokenizer_norm_eps,
            decoder_transformer_lengths_str=self.tokenizer_decoder_transformer_lengths_str,
            decoder_convs_kernels_str=self.tokenizer_decoder_convs_kernels_str,
            decoder_convs_strides_str=self.tokenizer_decoder_convs_strides_str,
        )


# ============================================================================
# LM Backbone (Mistral architecture via mlx-lm)
# ============================================================================


class MistralBackbone(nn.Module):
    """Wrapper around mlx-lm's LlamaModel configured as Mistral."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        from mlx_lm.models.llama import Model as LlamaFullModel
        from mlx_lm.models.llama import ModelArgs

        lm_args = ModelArgs(
            model_type="llama",
            hidden_size=config.dim,
            num_hidden_layers=config.n_layers,
            intermediate_size=config.hidden_dim,
            num_attention_heads=config.n_heads,
            num_key_value_heads=config.n_kv_heads,
            rms_norm_eps=config.norm_eps,
            vocab_size=config.vocab_size,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_traditional=True,  # Voxtral uses interleaved RoPE (not NeoX)
            tie_word_embeddings=config.tied_embeddings,
            attention_bias=config.use_biases,
            mlp_bias=config.use_biases,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.model = LlamaFullModel(lm_args)

    def __call__(
        self, input_ids: mx.array, cache=None, input_embeddings=None
    ) -> mx.array:
        return self.model(input_ids, cache=cache, input_embeddings=input_embeddings)

    @property
    def embed_tokens(self):
        return self.model.model.embed_tokens


# ============================================================================
# Main Model
# ============================================================================


class Model(nn.Module):
    """Voxtral-4B-TTS model for text-to-speech generation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # LM backbone
        self.language_model = MistralBackbone(config)

        # Audio codebook embeddings: semantic (padded to 8320) + acoustic (padded to 768) = 9088
        semantic_padded = (config.semantic_codebook_size // 128 + 1) * 128
        acoustic_padded = pad_to_multiple(
            config.acoustic_codebook_size * config.n_acoustic_codebook, 128
        )
        audio_embedding_size = semantic_padded + acoustic_padded
        self.audio_codebook_embeddings = {
            "embeddings": nn.Embedding(audio_embedding_size, config.dim)
        }

        # Acoustic transformer (flow matching)
        self.acoustic_transformer = FlowMatchingAudioTransformer(
            config.get_acoustic_args()
        )

        # Audio tokenizer (decoder only - we only need decode, not encode)
        self.audio_tokenizer = VoxtralTTSAudioTokenizer(config.get_tokenizer_args())

        # Tokenizer and voice embeddings
        self.tokenizer = None
        self._voice_embeddings = {}
        self._voice_embedding_files = {}
        self._voice_num_audio_tokens = {}
        self._text_to_audio_token_id = None
        self._audio_to_text_token_id = None

    @property
    def sample_rate(self):
        return self.config.sample_rate

    @property
    def model_type(self):
        return self.config.model_type

    def model_quant_predicate(self, p, m):
        """Skip quantization for audio tokenizer and embeddings."""
        skip_prefixes = ("audio_tokenizer", "audio_codebook_embeddings")
        return not any(p.startswith(prefix) for prefix in skip_prefixes)

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path):
        """Load tokenizer and voice embeddings after weights are loaded."""
        # Load tekken tokenizer
        tekken_path = model_path / "tekken.json"
        if is_tekken(tekken_path):
            try:
                tekken_data = json.loads(tekken_path.read_text())
                model._load_tekken_metadata(tekken_data)
            except Exception as e:
                print(f"Warning: Could not parse tekken metadata: {e}")

            try:
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

                model.tokenizer = MistralTokenizer.from_file(str(tekken_path))
                print(f"Loaded Mistral tokenizer from {tekken_path}")
            except ImportError as e:
                raise RuntimeError(
                    "Voxtral TTS tekken tokenizers require mistral-common[audio]. "
                    "Install the `tts` extra or add `mistral-common[audio]` to "
                    "your environment."
                ) from e
        else:
            try:
                from transformers import AutoTokenizer

                model.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                print(f"Loaded HF tokenizer from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")

        model._sync_prompt_token_ids_from_tokenizer()

        # Register voice embeddings for lazy loading.
        voice_dir = model_path / "voice_embedding"
        if voice_dir.exists():
            for voice_file in voice_dir.glob("*.safetensors"):
                model._voice_embedding_files[voice_file.stem] = voice_file

        return model

    def _get_voice_embedding(self, voice: str) -> mx.array | None:
        """Load and cache a single voice embedding on first use."""
        voice_emb = self._voice_embeddings.get(voice)
        if voice_emb is not None:
            return voice_emb

        voice_file = self._voice_embedding_files.get(voice)
        if voice_file is None:
            return None

        try:
            data = mx.load(str(voice_file))
            voice_emb = data.get("embedding", next(iter(data.values())))
            self._voice_embeddings[voice] = voice_emb
            return voice_emb
        except Exception as e:
            print(f"  Warning: Could not load voice {voice}: {e}")
            return None

    def _load_tekken_metadata(self, tekken_data: dict) -> None:
        """Load prompt-related metadata from tekken.json when mistral_common is absent."""
        special_tokens = tekken_data.get("special_tokens", [])
        special_token_ids = {
            item["token_str"]: item["rank"]
            for item in special_tokens
            if "token_str" in item and "rank" in item
        }
        self._text_to_audio_token_id = special_token_ids.get("[NEXT_AUDIO_TEXT]")
        self._audio_to_text_token_id = special_token_ids.get("[REPEAT_AUDIO_TEXT]")

        audio_cfg = tekken_data.get("audio", {})
        voice_num_audio_tokens = audio_cfg.get("voice_num_audio_tokens", {})
        if isinstance(voice_num_audio_tokens, dict):
            self._voice_num_audio_tokens = {
                str(voice): int(num_tokens)
                for voice, num_tokens in voice_num_audio_tokens.items()
            }

    def _sync_prompt_token_ids_from_tokenizer(self) -> None:
        """Prefer token IDs exposed by the loaded tokenizer when available."""
        if self.tokenizer is None:
            return

        try:
            instruct_tokenizer = getattr(self.tokenizer, "instruct_tokenizer", None)
            if instruct_tokenizer is not None:
                audio_encoder = getattr(instruct_tokenizer, "audio_encoder", None)
                if audio_encoder is not None:
                    self._text_to_audio_token_id = audio_encoder.text_to_audio_token
                    self._audio_to_text_token_id = audio_encoder.audio_to_text_token
                    voice_cfg = getattr(
                        audio_encoder.audio_config, "voice_num_audio_tokens", None
                    )
                    if voice_cfg:
                        self._voice_num_audio_tokens = {
                            str(voice): int(num_tokens)
                            for voice, num_tokens in voice_cfg.items()
                        }
                    return
        except Exception:
            pass

        audio_cfg = getattr(self.tokenizer, "audio", None)
        if isinstance(audio_cfg, dict):
            voice_cfg = audio_cfg.get("voice_num_audio_tokens")
            if isinstance(voice_cfg, dict) and voice_cfg:
                self._voice_num_audio_tokens = {
                    str(voice): int(num_tokens)
                    for voice, num_tokens in voice_cfg.items()
                }

        direct_token_lookup = getattr(self.tokenizer, "token_to_id", None)
        if callable(direct_token_lookup):
            text_to_audio = direct_token_lookup("[NEXT_AUDIO_TEXT]")
            audio_to_text = direct_token_lookup("[REPEAT_AUDIO_TEXT]")
            if text_to_audio is not None:
                self._text_to_audio_token_id = int(text_to_audio)
            if audio_to_text is not None:
                self._audio_to_text_token_id = int(audio_to_text)
            if (
                self._text_to_audio_token_id is not None
                and self._audio_to_text_token_id is not None
            ):
                return

        backend = getattr(self.tokenizer, "_tokenizer", None)
        if backend is None:
            return

        text_to_audio = backend.token_to_id("[NEXT_AUDIO_TEXT]")
        audio_to_text = backend.token_to_id("[REPEAT_AUDIO_TEXT]")
        if text_to_audio is not None:
            self._text_to_audio_token_id = text_to_audio
        if audio_to_text is not None:
            self._audio_to_text_token_id = audio_to_text

    def sanitize(self, weights: dict) -> dict:
        """Remap weight names from consolidated.safetensors to our model structure."""
        new_weights = {}

        for key, value in weights.items():
            new_key = None

            # --- Acoustic transformer weights ---
            if key.startswith("acoustic_transformer."):
                suffix = key[len("acoustic_transformer.") :]
                new_key = f"acoustic_transformer.{suffix}"

            # --- Audio tokenizer weights ---
            elif key.startswith("audio_tokenizer."):
                suffix = key[len("audio_tokenizer.") :]
                new_key = f"audio_tokenizer.{suffix}"

            # --- Audio embeddings ---
            elif key.startswith("mm_audio_embeddings.audio_codebook_embeddings."):
                suffix = key[len("mm_audio_embeddings.audio_codebook_embeddings.") :]
                new_key = f"audio_codebook_embeddings.{suffix}"

            elif key == "mm_audio_embeddings.tok_embeddings.weight":
                # This is the shared text embedding - goes to LM backbone
                new_key = "language_model.model.model.embed_tokens.weight"

            # --- LM backbone weights ---
            elif key == "tok_embeddings.weight":
                new_key = "language_model.model.model.embed_tokens.weight"

            elif key == "norm.weight":
                new_key = "language_model.model.model.norm.weight"

            elif key == "output.weight":
                # Output projection (tied with embeddings in this model)
                # Skip if tied embeddings
                if self.config.tied_embeddings:
                    continue
                new_key = "language_model.model.lm_head.weight"

            elif key.startswith("layers."):
                # Transformer layer weights: layers.N.xxx -> language_model.model.model.layers.N.xxx
                match = re.match(r"layers\.(\d+)\.(.*)", key)
                if match:
                    layer_idx = match.group(1)
                    suffix = match.group(2)
                    new_key = self._remap_layer_key(layer_idx, suffix)

            # --- Already-sanitized weights (from MLX-converted models) ---
            # Handle old format: language_model.model.X -> language_model.model.model.X
            elif key.startswith("language_model.model.") and not key.startswith(
                "language_model.model.model."
            ):
                suffix = key[len("language_model.model.") :]
                new_key = f"language_model.model.model.{suffix}"

            else:
                new_key = key

            if new_key is not None:
                new_weights[new_key] = value

        return new_weights

    def _remap_layer_key(self, layer_idx: str, suffix: str) -> str:
        """Remap a single transformer layer's weight key."""
        prefix = f"language_model.model.model.layers.{layer_idx}"

        # Attention weights
        if suffix == "attention.wq.weight":
            return f"{prefix}.self_attn.q_proj.weight"
        elif suffix == "attention.wk.weight":
            return f"{prefix}.self_attn.k_proj.weight"
        elif suffix == "attention.wv.weight":
            return f"{prefix}.self_attn.v_proj.weight"
        elif suffix == "attention.wo.weight":
            return f"{prefix}.self_attn.o_proj.weight"

        # FFN weights
        elif suffix == "feed_forward.w1.weight":
            return f"{prefix}.mlp.gate_proj.weight"
        elif suffix == "feed_forward.w2.weight":
            return f"{prefix}.mlp.down_proj.weight"
        elif suffix == "feed_forward.w3.weight":
            return f"{prefix}.mlp.up_proj.weight"

        # Norms
        elif suffix == "attention_norm.weight":
            return f"{prefix}.input_layernorm.weight"
        elif suffix == "ffn_norm.weight":
            return f"{prefix}.post_attention_layernorm.weight"

        else:
            return f"{prefix}.{suffix}"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration in seconds as 00:MM:SS.mmm."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"00:{mins:02d}:{secs:02d}.{ms:03d}"

    def generate(
        self,
        text: str,
        voice: str = "casual_male",
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        verbose: bool = False,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> GenerationResult:
        """Generate speech from text.

        Args:
            text: Input text to synthesize.
            voice: Voice preset name (e.g. "casual_male", "fr_female").
            temperature: Sampling temperature for the LM.
            top_k: Top-k sampling.
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum number of audio tokens to generate.
            verbose: Show progress bar.
            stream: Enable streaming output. When True, intermediate audio
                chunks are yielded during generation for lower latency.
            streaming_interval: Approximate seconds of audio per streaming
                chunk. Each frame is 80 ms, so the interval is converted to
                a frame count (``max(1, int(streaming_interval / 0.08))``).

        Yields:
            GenerationResult with audio waveform. When *stream* is True,
            intermediate results have ``is_streaming_chunk=True`` and the
            last result additionally has ``is_final_chunk=True``.
        """
        from mlx_lm.models.cache import make_prompt_cache

        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded. Ensure post_load_hook was called."
            )

        time_start = time.time()

        input_ids = self._encode_text(text, voice)
        input_ids_mx = mx.array(input_ids)[None, :]  # (1, seq_len)
        input_embeddings = self._build_input_embeddings(input_ids_mx, voice)

        # LM backbone returns hidden states; full model returns logits
        lm_backbone = self.language_model.model.model

        # Prefill: run prompt through LM with KV cache
        cache = make_prompt_cache(self.language_model.model)
        hidden = lm_backbone(
            input_ids_mx, cache=cache, input_embeddings=input_embeddings
        )

        # First decode step uses AUDIO token (24) embedding as input
        # This matches the C reference: the first LLM decode produces the hidden state
        # for the first audio frame
        audio_tok_emb = self.language_model.embed_tokens(
            mx.array([[self.config.audio_token_id]])
        )  # (1, 1, dim)
        hidden = lm_backbone(
            mx.array([[self.config.audio_token_id]]),
            cache=cache,
            input_embeddings=audio_tok_emb,
        )

        all_codes = []
        yielded_frames = 0
        chunk_idx = 0
        # Convert streaming_interval (seconds) to frames (1 frame = 80 ms)
        frames_per_chunk = max(1, int(streaming_interval / 0.08))
        # Context frames for overlap-add decoding.  The codec decoder uses
        # sliding-window attention with windows up to 16 (at the final stage).
        # Including context frames from the previous chunk ensures smooth
        # transitions without boundary artifacts.
        context_frames = 16
        # Each codec frame produces 1920 samples (8x upsample × 240 patch)
        samples_per_frame = 1920

        for i in tqdm(range(max_tokens), disable=not verbose):
            h = hidden[:, -1, :]  # (1, dim)

            codes = self.acoustic_transformer.decode_one_frame(h)  # (1, 37)

            # End-of-audio: semantic code == END_AUDIO special token (index 1)
            semantic_code = codes[0, 0].item()
            if semantic_code <= 1:  # 0=empty_audio, 1=end_audio
                break

            all_codes.append(codes[:, None, :])  # (1, 1, 37)

            # Embed audio codes back as LLM input for next step.
            # Each codebook has its own offset in the global embedding table:
            #   Codebook 0 (semantic): offset=0, size=semantic_codebook_size+2
            #   Codebook 1..36 (acoustic): offset=prev+prev_size, size=acoustic_codebook_size+2
            global_codes = self._codes_to_global_indices(codes)  # (1, 37)
            code_embeddings = self.audio_codebook_embeddings["embeddings"](global_codes)
            next_embedding = code_embeddings.sum(axis=1, keepdims=True)  # (1, 1, dim)

            # Feed back through LM for next step using KV cache
            dummy_input = mx.array([[self.config.audio_token_id]])
            hidden = lm_backbone(
                dummy_input, cache=cache, input_embeddings=next_embedding
            )

            if i % 50 == 0:
                mx.clear_cache()

            # Streaming: yield chunk when buffer is full
            if stream and len(all_codes) - yielded_frames >= frames_per_chunk:
                # Include context frames from earlier in the sequence so the
                # codec decoder's sliding-window attention has proper context,
                # avoiding boundary artifacts between chunks.
                ctx_start = max(0, yielded_frames - context_frames)
                chunk_codes = mx.concatenate(all_codes[ctx_start:], axis=1)
                full_waveform = self.audio_tokenizer.decode(chunk_codes)
                full_waveform = full_waveform.squeeze(0)

                # Trim the context portion — keep only new audio
                ctx_used = yielded_frames - ctx_start
                trim_samples = ctx_used * samples_per_frame
                chunk_waveform = full_waveform[trim_samples:]

                chunk_samples = chunk_waveform.shape[0]
                chunk_duration = chunk_samples / self.config.sample_rate
                chunk_time = time.time() - time_start
                chunk_token_count = len(all_codes) - yielded_frames

                yield GenerationResult(
                    audio=chunk_waveform,
                    sample_rate=self.config.sample_rate,
                    samples=chunk_samples,
                    segment_idx=chunk_idx,
                    token_count=chunk_token_count,
                    audio_samples={
                        "samples": chunk_samples,
                        "samples-per-sec": self.config.sample_rate,
                    },
                    audio_duration=self._format_duration(chunk_duration),
                    real_time_factor=(
                        chunk_duration / chunk_time if chunk_time > 0 else 0
                    ),
                    prompt={
                        "tokens": chunk_token_count,
                        "tokens-per-sec": (
                            round(chunk_token_count / chunk_time, 2)
                            if chunk_time > 0
                            else 0
                        ),
                    },
                    processing_time_seconds=chunk_time,
                    peak_memory_usage=mx.get_peak_memory() / 1e9,
                    is_streaming_chunk=True,
                    is_final_chunk=False,
                )
                yielded_frames = len(all_codes)
                chunk_idx += 1
                time_start = time.time()

        if not all_codes:
            raise RuntimeError("No audio frames generated")

        # Final chunk: decode remaining frames (or all frames if not streaming)
        remaining = len(all_codes) - yielded_frames
        if stream and yielded_frames > 0 and remaining > 0:
            # Decode remainder with context for smooth transition
            ctx_start = max(0, yielded_frames - context_frames)
            final_codes = mx.concatenate(all_codes[ctx_start:], axis=1)
            full_waveform = self.audio_tokenizer.decode(final_codes).squeeze(0)
            ctx_used = yielded_frames - ctx_start
            trim_samples = ctx_used * samples_per_frame
            waveform = full_waveform[trim_samples:]
        elif stream and yielded_frames > 0 and remaining == 0:
            # Everything already yielded — emit a zero-length final marker
            waveform = mx.zeros((0,))
        else:
            # Non-streaming (or no intermediate chunks were yielded):
            # decode everything at once — identical to the original path
            audio_codes = mx.concatenate(all_codes, axis=1)
            waveform = self.audio_tokenizer.decode(audio_codes).squeeze(0)

        time_end = time.time()

        audio_samples = waveform.shape[0]
        audio_duration = audio_samples / self.config.sample_rate

        processing_time = time_end - time_start

        yield GenerationResult(
            audio=waveform,
            sample_rate=self.config.sample_rate,
            samples=audio_samples,
            segment_idx=chunk_idx if stream else 0,
            token_count=remaining if stream and yielded_frames > 0 else len(all_codes),
            audio_samples={
                "samples": audio_samples,
                "samples-per-sec": self.config.sample_rate,
            },
            audio_duration=self._format_duration(audio_duration),
            real_time_factor=(
                audio_duration / processing_time if processing_time > 0 else 0
            ),
            prompt={
                "tokens": (
                    remaining if stream and yielded_frames > 0 else len(all_codes)
                ),
                "tokens-per-sec": (
                    round(
                        (remaining if stream and yielded_frames > 0 else len(all_codes))
                        / processing_time,
                        2,
                    )
                    if processing_time > 0
                    else 0
                ),
            },
            processing_time_seconds=processing_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=stream,
            is_final_chunk=stream,
        )
        mx.clear_cache()

    def _encode_text_tokens(self, text: str) -> list[int]:
        """Tokenize speech text without adding BOS/EOS markers."""
        encode = getattr(self.tokenizer, "encode", None)
        if encode is None:
            raise RuntimeError("Tokenizer does not expose an encode method.")

        try:
            return list(encode(text, add_special_tokens=False))
        except TypeError:
            try:
                return list(encode(text, bos=False, eos=False))
            except TypeError:
                return list(encode(text))

    def _encode_text(self, text: str, voice: str) -> list:
        """Encode text with voice prompt into token IDs."""
        text = sanitize_tts_input_text_for_demo(text)

        if self._text_to_audio_token_id is None or self._audio_to_text_token_id is None:
            raise RuntimeError(
                "Missing Voxtral speech prompt token IDs. "
                "Expected [NEXT_AUDIO_TEXT] and [REPEAT_AUDIO_TEXT] from tekken.json."
            )

        if not hasattr(self.tokenizer, "encode_speech_request"):
            raise RuntimeError(
                "Voxtral TTS requires mistral-common[audio] so the Mistral "
                "speech tokenizer can build the prompt correctly. Install the "
                "`tts` extra or add `mistral-common[audio]` to your environment."
            )

        from mistral_common.protocol.speech.request import SpeechRequest

        req = SpeechRequest(input=text, voice=voice)
        result = self.tokenizer.encode_speech_request(req)
        return result.tokens

    def _codes_to_global_indices(self, codes: mx.array) -> mx.array:
        """Convert per-codebook codes to global embedding table indices.

        The embedding table is laid out as:
          [semantic_cb (8192+2 entries)] [acoustic_cb_0 (21+2)] [acoustic_cb_1 (21+2)] ...
        Each code already includes the +2 special token offset from decode_one_frame.
        """
        cfg = self.config
        n_special = 2
        semantic_size = cfg.semantic_codebook_size + n_special  # 8194
        acoustic_size = cfg.acoustic_codebook_size + n_special  # 23

        # Build offset array: [0, 8194, 8194+23, 8194+46, ...]
        offsets = [0]  # semantic codebook starts at 0
        for i in range(cfg.n_acoustic_codebook):
            offsets.append(semantic_size + i * acoustic_size)
        offsets = mx.array(offsets, dtype=codes.dtype).reshape(
            (1,) * (codes.ndim - 1) + (len(offsets),)
        )
        return codes + offsets

    def _build_input_embeddings(self, input_ids: mx.array, voice: str) -> mx.array:
        """Build input embeddings with voice conditioning at AUDIO token positions.

        Voice embeddings replace the audio token embeddings (memcpy in the C reference).
        """
        embeddings = self.language_model.embed_tokens(input_ids)  # (1, T, dim)

        audio_mask = input_ids[0] == self.config.audio_token_id  # (T,)
        voice_emb = self._get_voice_embedding(voice)
        if voice_emb is None:
            return embeddings

        # Map each audio position to a voice embedding index
        indices = mx.cumsum(audio_mask.astype(mx.int32)) - 1
        indices = mx.clip(indices, 0, voice_emb.shape[0] - 1)

        # Replace audio token embeddings with voice embeddings
        voice_expanded = voice_emb[indices]  # (T, dim)
        mask_3d = audio_mask[:, None].astype(embeddings.dtype)  # (T, 1)
        embeddings = embeddings.at[0].add(
            mask_3d * (voice_expanded.astype(embeddings.dtype) - embeddings[0])
        )

        return embeddings
