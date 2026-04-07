# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import json
import os
import re
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoTokenizer

from mlx_audio.tts.models.base import GenerationResult

from .acoustic_tokenizer import (
    AcousticTokenizer,
    Block1D,
    CausalConv1d,
    ConvRMSNorm,
    HeadConv,
    StemConv,
    TokenizerStreamingCache,
)
from .config import ModelConfig, Qwen2DecoderConfig
from .diffusion_head import DiffusionHead
from .language_model import (
    BinaryClassifier,
    MlxLmQwen2Model,
    Qwen2Model,
    SpeechConnector,
)
from .scheduler import DPMSolverMultistepScheduler

# Constants from original implementation
TTS_TEXT_WINDOW_SIZE = 5
TTS_SPEECH_WINDOW_SIZE = 6
NON_STREAMING_SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)
def _tensor_stats_mlx(tensor: Optional[mx.array]):
    if tensor is None:
        return None
    shape = [int(dim) for dim in tensor.shape]
    numel = int(tensor.size)
    stats = {
        "shape": shape,
        "dtype": str(tensor.dtype),
        "numel": numel,
    }
    if numel == 0:
        return stats
    data = np.asarray(tensor.astype(mx.float32))
    flat = data.reshape(-1)
    stats.update(
        {
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "l2": float(np.linalg.norm(flat)),
        }
    )
    return stats


def _write_trace_line(handle, payload) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _control_token_kind(
    token_id: int,
    *,
    speech_start_id: Optional[int],
    speech_end_id: Optional[int],
    speech_diffusion_id: Optional[int],
    eos_id: Optional[int],
    bos_id: Optional[int] = None,
) -> str:
    if speech_start_id is not None and token_id == speech_start_id:
        return "speech_start"
    if speech_end_id is not None and token_id == speech_end_id:
        return "speech_end"
    if speech_diffusion_id is not None and token_id == speech_diffusion_id:
        return "speech_diffusion"
    if eos_id is not None and token_id == eos_id:
        return "eos"
    if bos_id is not None and token_id == bos_id:
        return "bos"
    return "other"


def _control_score_dict(
    *,
    candidate_ids: List[int],
    candidate_logits: mx.array,
    speech_start_id: Optional[int],
    speech_end_id: Optional[int],
    speech_diffusion_id: Optional[int],
    eos_id: Optional[int],
    bos_id: Optional[int] = None,
) -> dict:
    id_to_score = {
        int(token_id): float(score)
        for token_id, score in zip(candidate_ids, candidate_logits.tolist())
    }
    token_map = {
        "speech_start": speech_start_id,
        "speech_end": speech_end_id,
        "speech_diffusion": speech_diffusion_id,
        "eos": eos_id,
    }
    if bos_id is not None:
        token_map["bos"] = bos_id
    return {
        name: (None if token_id is None else id_to_score.get(int(token_id)))
        for name, token_id in token_map.items()
    }


def _load_diffusion_noise_sequence_mlx(path: str, latent_dim: int):
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if raw and isinstance(raw[0], (int, float)):
        raw = [raw]

    sequence = []
    for idx, item in enumerate(raw):
        noise = mx.array(item, dtype=mx.float32)
        if noise.ndim == 1:
            noise = mx.expand_dims(noise, axis=0)
        if noise.ndim != 2 or int(noise.shape[1]) != int(latent_dim):
            raise ValueError(
                f"Invalid diffusion noise entry at index {idx}: "
                f"expected shape [batch, {latent_dim}] or [{latent_dim}], "
                f"got {tuple(int(dim) for dim in noise.shape)}"
            )
        sequence.append(noise)

    return sequence


class EncoderOutput:
    """Minimal encoder output wrapper to match `.mean` access pattern."""

    def __init__(self, mean: mx.array):
        self.mean = mean


class DownsampleLayer(nn.Module):
    """Downsample layer for semantic tokenizer encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[TokenizerStreamingCache] = None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        return self.conv(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )


class SemanticTokenizerEncoder(nn.Module):
    """Semantic tokenizer encoder used in non-streaming VibeVoice inference."""

    def __init__(self, config):
        super().__init__()

        self.dimension = config.vae_dim
        self.channels = config.channels
        self.n_filters = config.encoder_n_filters
        # Match upstream TokenizerEncoder: downsample uses reversed ratios.
        self.ratios = list(reversed(config.encoder_ratios))
        self.depths = (
            [int(d) for d in config.encoder_depths.split("-")]
            if isinstance(config.encoder_depths, str)
            else list(config.encoder_depths)
        )
        self.n_stages = len(self.depths)

        self.downsample_layers = []
        self.downsample_layers.append(
            [
                StemConv(
                    in_channels=self.channels,
                    out_channels=self.n_filters,
                    kernel_size=7,
                    bias=config.conv_bias,
                )
            ]
        )

        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2**i)
            out_ch = self.n_filters * (2 ** (i + 1))
            ratio = self.ratios[i]
            self.downsample_layers.append(
                [
                    DownsampleLayer(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        bias=config.conv_bias,
                    )
                ]
            )

        self.stages = []
        for i in range(self.n_stages):
            in_ch = self.n_filters * (2**i)
            stage_blocks = []
            for _ in range(self.depths[i]):
                stage_blocks.append(
                    Block1D(
                        dim=in_ch,
                        layernorm=config.layernorm,
                        eps=config.layernorm_eps,
                        causal=config.causal,
                        bias=config.conv_bias,
                        layer_scale_init_value=config.layer_scale_init_value,
                    )
                )
            self.stages.append(stage_blocks)

        self.norm = (
            nn.Identity()
            if config.disable_last_norm
            else ConvRMSNorm(in_ch, eps=config.layernorm_eps)
        )
        self.head = HeadConv(
            in_channels=in_ch,
            out_channels=self.dimension,
            kernel_size=7,
            bias=config.conv_bias,
        )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[TokenizerStreamingCache] = None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> mx.array:
        # x: (B, 1, T)
        for i in range(self.n_stages):
            x = self.downsample_layers[i][0](
                x,
                cache=cache,
                sample_indices=sample_indices,
                use_cache=use_cache,
                debug=debug,
            )
            for block in self.stages[i]:
                x = block(
                    x,
                    cache=cache,
                    sample_indices=sample_indices,
                    use_cache=use_cache,
                    debug=debug,
                )

        x = self.norm(x)
        x = self.head(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )  # (B, D, T')
        return mx.transpose(x, (0, 2, 1))  # (B, T', D)


class SemanticTokenizer(nn.Module):
    """Minimal semantic tokenizer (encoder-only) for non-streaming VibeVoice."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = SemanticTokenizerEncoder(config)

    def encode(
        self,
        audio: mx.array,
        cache: Optional[TokenizerStreamingCache] = None,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> EncoderOutput:
        if audio.ndim == 1:
            audio = audio[None, None, :]
        elif audio.ndim == 2:
            audio = audio[:, None, :]
        latents = self.encoder(
            audio,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        return EncoderOutput(mean=latents)


class Model(nn.Module):
    """VibeVoice TTS model.

    Supports two upstream variants:
    - `vibevoice_streaming` (split LM with voice-cache conditioning)
    - `vibevoice` (non-streaming 7B architecture)

    The runtime generation path currently targets the streaming variant.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        decoder_config = config.decoder_config
        self.is_non_streaming = config.model_type == "vibevoice"

        if self.is_non_streaming:
            # Non-streaming 7B variant uses a single language model backbone.
            use_mlx_lm_qwen2 = os.getenv("MLX_AUDIO_VIBEVOICE_USE_MLX_LM_QWEN2", "0") == "1"
            if use_mlx_lm_qwen2:
                self.language_model = MlxLmQwen2Model(decoder_config, use_norm=True)
            else:
                self.language_model = Qwen2Model(decoder_config, use_norm=True)
            self.lm_head = nn.Linear(
                decoder_config.hidden_size,
                decoder_config.vocab_size,
                bias=False,
            )

            self.acoustic_tokenizer = AcousticTokenizer(config.acoustic_tokenizer_config)
            self.semantic_tokenizer = SemanticTokenizer(config.semantic_tokenizer_config)
            self.acoustic_connector = SpeechConnector(
                input_dim=config.acoustic_vae_dim,
                output_dim=decoder_config.hidden_size,
            )
            self.semantic_connector = SpeechConnector(
                input_dim=config.semantic_vae_dim,
                output_dim=decoder_config.hidden_size,
            )
            self.prediction_head = DiffusionHead(config.diffusion_head_config)
        else:
            # Streaming architecture uses a split LM (text LM + TTS LM).
            tts_layers = config.tts_backbone_num_hidden_layers
            lm_layers = decoder_config.num_hidden_layers - tts_layers

            lm_config = Qwen2DecoderConfig(
                model_type=decoder_config.model_type,
                hidden_size=decoder_config.hidden_size,
                intermediate_size=decoder_config.intermediate_size,
                num_attention_heads=decoder_config.num_attention_heads,
                num_key_value_heads=decoder_config.num_key_value_heads,
                num_hidden_layers=lm_layers,
                rms_norm_eps=decoder_config.rms_norm_eps,
                vocab_size=decoder_config.vocab_size,
                max_position_embeddings=decoder_config.max_position_embeddings,
                rope_theta=decoder_config.rope_theta,
                head_dim=decoder_config.head_dim,
            )

            tts_lm_config = Qwen2DecoderConfig(
                model_type=decoder_config.model_type,
                hidden_size=decoder_config.hidden_size,
                intermediate_size=decoder_config.intermediate_size,
                num_attention_heads=decoder_config.num_attention_heads,
                num_key_value_heads=decoder_config.num_key_value_heads,
                num_hidden_layers=tts_layers,
                rms_norm_eps=decoder_config.rms_norm_eps,
                vocab_size=decoder_config.vocab_size,  # Reuse embeddings
                max_position_embeddings=decoder_config.max_position_embeddings,
                rope_theta=decoder_config.rope_theta,
                head_dim=decoder_config.head_dim,
            )

            # Base LM doesn't have final norm (it continues into tts_language_model)
            self.language_model = Qwen2Model(lm_config, use_norm=False)
            self.tts_language_model = Qwen2Model(tts_lm_config, use_norm=True)
            self.tts_input_types = nn.Embedding(2, decoder_config.hidden_size)
            self.acoustic_tokenizer = AcousticTokenizer(config.acoustic_tokenizer_config)
            self.acoustic_connector = SpeechConnector(
                input_dim=config.acoustic_vae_dim,
                output_dim=decoder_config.hidden_size,
            )
            self.prediction_head = DiffusionHead(config.diffusion_head_config)
            self.tts_eos_classifier = BinaryClassifier(decoder_config.hidden_size)

        # Noise scheduler
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type,
        )

        # Scaling factors (will be loaded from weights)
        self.speech_scaling_factor = mx.array(1.0)
        self.speech_bias_factor = mx.array(0.0)

        # Inference settings
        self.ddpm_inference_steps = (
            config.diffusion_head_config.ddpm_num_inference_steps
        )
        self._cached_ddpm_steps: Optional[int] = None
        self._cached_timesteps: Optional[List[int]] = None

        # Tokenizer placeholder
        self.tokenizer = None
        self.speech_start_id: Optional[int] = None
        self.speech_end_id: Optional[int] = None
        self.speech_diffusion_id: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.bos_id: Optional[int] = None

        # Optional voice cache state (loaded via load_voice_cache)
        self._voice_path: Optional[str] = None
        self._voice_lm_hidden: Optional[mx.array] = None
        self._voice_tts_hidden: Optional[mx.array] = None
        self._voice_neg_tts_hidden: Optional[mx.array] = None
        self._voice_lm_cache: Optional[list] = None
        self._voice_tts_cache: Optional[list] = None
        self._voice_neg_tts_cache: Optional[list] = None
        self._voice_neg_lm_cache: Optional[list] = None

    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self.config.sample_rate

    def quantized_component_prefixes(self) -> List[str]:
        """Return the component prefixes that should be quantized for this model."""
        quant_cfg = getattr(self.config, "quantization", None) or {}
        configured = quant_cfg.get("quantized_components", None)
        if configured:
            return [str(prefix) for prefix in configured]

        if self.is_non_streaming:
            prefixes = ["language_model."]
            if os.getenv("MLX_AUDIO_VIBEVOICE_QUANTIZE_PREDICTION_HEAD", "0") == "1":
                prefixes.append("prediction_head.")
            if os.getenv("MLX_AUDIO_VIBEVOICE_QUANTIZE_LM_HEAD", "0") == "1":
                prefixes.append("lm_head.")
            return prefixes

        return ["language_model.", "tts_language_model."]

    def model_quant_predicate(self, path: str, module) -> bool:
        """Selectively quantize only the LM backbone by default.

        This matches the general mlx-audio pattern for speech models:
        keep acoustic/semantic tokenizers, connectors, output heads, and
        other speech-sensitive components in higher precision, while
        quantizing the large transformer backbone where the main memory and
        latency savings come from.

        Optional experimental overrides:
        - ``MLX_AUDIO_VIBEVOICE_QUANTIZE_PREDICTION_HEAD=1`` also quantizes
          ``prediction_head``.
        - ``MLX_AUDIO_VIBEVOICE_QUANTIZE_LM_HEAD=1`` also quantizes
          ``lm_head`` on the non-streaming path.
        """
        return any(
            path.startswith(prefix) for prefix in self.quantized_component_prefixes()
        )

    def _resolve_non_streaming_token_ids(self) -> None:
        """Resolve required special token IDs for non-streaming generation."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded.")

        def _id(token: str) -> int:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id < 0:
                raise ValueError(
                    f"Tokenizer does not contain required token `{token}` for VibeVoice."
                )
            return int(token_id)

        self.speech_start_id = _id("<|vision_start|>")
        self.speech_end_id = _id("<|vision_end|>")
        self.speech_diffusion_id = _id("<|vision_pad|>")
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        self.bos_id = int(bos_id) if bos_id is not None and bos_id >= 0 else None
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None or eos_id < 0:
            eos_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if eos_id is None or eos_id < 0:
            raise ValueError("Tokenizer has no valid EOS token id for VibeVoice.")
        self.eos_id = int(eos_id)

    def _speech_tok_compress_ratio(self) -> int:
        ratio = 1
        for r in self.config.acoustic_tokenizer_config.encoder_ratios:
            ratio *= int(r)
        return max(1, ratio)

    def _parse_non_streaming_script(self, text: str) -> List[Tuple[int, str]]:
        """Parse a non-streaming VibeVoice script into normalized speaker turns.

        Supported forms:
        - Plain text -> single speaker 0
        - Speaker-labelled script:
          Speaker 1: Hello
          Speaker 2: Hi
        - Multi-line turns where unlabeled lines continue the previous speaker
        """
        script = text.strip()
        if not script:
            raise ValueError("Text is empty.")

        lines = [line.strip() for line in script.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Text is empty.")

        speaker_pattern = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
        saw_speaker_labels = False
        raw_entries: List[Tuple[int, str]] = []
        current_speaker: Optional[int] = None
        current_parts: List[str] = []

        def _flush() -> None:
            nonlocal current_speaker, current_parts
            if current_speaker is None:
                return
            merged = " ".join(part for part in current_parts if part).strip()
            if merged:
                raw_entries.append((current_speaker, f" {merged}"))
            current_speaker = None
            current_parts = []

        for line in lines:
            match = speaker_pattern.match(line)
            if match:
                saw_speaker_labels = True
                _flush()
                current_speaker = int(match.group(1))
                initial_text = match.group(2).strip()
                current_parts = [initial_text] if initial_text else []
                continue

            if saw_speaker_labels:
                if current_speaker is None:
                    current_speaker = 0
                current_parts.append(line)
            else:
                current_parts.append(line)

        if saw_speaker_labels:
            _flush()
            if not raw_entries:
                raise ValueError("No valid speaker lines found in script.")
        else:
            merged = " ".join(part for part in current_parts if part).strip()
            if not merged:
                raise ValueError("Text is empty.")
            return [(0, f" {merged}")]

        speaker_ids = [speaker_id for speaker_id, _ in raw_entries]
        if speaker_ids and min(speaker_ids) > 0:
            unique_ids = sorted(set(speaker_ids))
            expected_ids = list(range(1, len(unique_ids) + 1))
            if unique_ids == expected_ids:
                raw_entries = [(speaker_id - 1, speaker_text) for speaker_id, speaker_text in raw_entries]
            else:
                normalized_map = {speaker_id: idx for idx, speaker_id in enumerate(unique_ids)}
                raw_entries = [
                    (normalized_map[speaker_id], speaker_text)
                    for speaker_id, speaker_text in raw_entries
                ]

        return raw_entries

    def _get_non_streaming_speaker_order(
        self, parsed_script: List[Tuple[int, str]]
    ) -> List[int]:
        """Return unique speaker ids in first-appearance order."""
        ordered_speakers: List[int] = []
        seen = set()
        for speaker_id, _ in parsed_script:
            if speaker_id in seen:
                continue
            seen.add(speaker_id)
            ordered_speakers.append(int(speaker_id))
        return ordered_speakers

    def _normalize_non_streaming_ref_audios(
        self,
        ref_audio: Union[str, Path, mx.array, List[Union[str, Path, mx.array]], Tuple[Union[str, Path, mx.array], ...]],
        expected_speakers: int,
    ) -> List[Union[str, Path, mx.array]]:
        """Normalize reference audio input to a per-speaker list."""
        if isinstance(ref_audio, (list, tuple)) and not isinstance(ref_audio, mx.array):
            ref_audios = list(ref_audio)
        else:
            ref_audios = [ref_audio]

        if expected_speakers <= 0:
            if ref_audios:
                raise ValueError("Reference audio was provided but no speakers were parsed.")
            return []

        if len(ref_audios) != expected_speakers:
            raise ValueError(
                "Number of reference audio samples must match the number of speakers "
                f"({expected_speakers}), got {len(ref_audios)}."
            )

        return ref_audios

    def _coerce_non_streaming_ref_prompt_steps(
        self,
        ref_prompt_steps: Optional[Union[int, List[int], Tuple[int, ...]]],
        expected_speakers: int,
    ) -> Optional[List[int]]:
        """Coerce reference prompt lengths to a per-speaker list."""
        if ref_prompt_steps is None:
            return None
        if isinstance(ref_prompt_steps, int):
            steps = [int(ref_prompt_steps)]
        else:
            steps = [int(step) for step in ref_prompt_steps]

        if len(steps) != expected_speakers:
            raise ValueError(
                "Reference prompt step counts must match the number of speakers "
                f"({expected_speakers}), got {len(steps)}."
            )

        return [max(1, step) for step in steps]

    def _shift_negative_ids(
        self, negative_input_ids: List[int], start_idx: int
    ) -> List[int]:
        """Shift negative ids to mimic upstream non-diffusion cache correction."""
        ids = list(negative_input_ids)
        seq_len = len(ids)
        if seq_len <= 1:
            return ids
        s = max(0, min(int(start_idx), seq_len - 2))
        if s + 1 < seq_len:
            ids[s + 1 :] = ids[s:-1]
        return ids

    def _shift_negative_cache(
        self,
        neg_cache: Optional[List[Tuple[mx.array, mx.array]]],
        start_idx: int,
    ) -> Optional[List[Tuple[mx.array, mx.array]]]:
        """Shift negative KV cache along sequence dimension (MLX: [B, S, H, D])."""
        if neg_cache is None:
            return None

        shifted_cache: List[Tuple[mx.array, mx.array]] = []
        for k_cache, v_cache in neg_cache:
            if k_cache.ndim != 4 or v_cache.ndim != 4:
                shifted_cache.append((k_cache, v_cache))
                continue

            seq_len = int(k_cache.shape[1])
            if seq_len <= 1:
                shifted_cache.append((k_cache, v_cache))
                continue

            s = max(0, min(int(start_idx), seq_len - 2))
            if s + 1 < seq_len:
                k_cache = mx.concatenate(
                    [k_cache[:, : s + 1, :, :], k_cache[:, s:-1, :, :]], axis=1
                )
                v_cache = mx.concatenate(
                    [v_cache[:, : s + 1, :, :], v_cache[:, s:-1, :, :]], axis=1
                )

            shifted_cache.append((k_cache, v_cache))

        return shifted_cache

    def _reset_negative_cache_on_speech_start(
        self,
        neg_cache,
    ):
        """Reset negative cache at speech start: copy first state to last position."""
        if neg_cache is None:
            return None

        # Efficient KVCache path (mlx_lm cache objects).
        if len(neg_cache) > 0 and hasattr(neg_cache[0], "offset"):
            for layer_cache in neg_cache:
                if layer_cache is None or layer_cache.offset <= 0:
                    continue
                keys = getattr(layer_cache, "keys", None)
                values = getattr(layer_cache, "values", None)
                if keys is None or values is None:
                    continue
                last = int(layer_cache.offset) - 1
                if last >= 0:
                    keys[..., last : last + 1, :] = keys[..., :1, :]
                    values[..., last : last + 1, :] = values[..., :1, :]
            return neg_cache

        # Legacy tuple-cache path.
        reset_cache = []
        for k_cache, v_cache in neg_cache:
            if k_cache.ndim == 4 and int(k_cache.shape[1]) > 0:
                if int(k_cache.shape[1]) > 1:
                    k_cache = mx.concatenate(
                        [k_cache[:, :-1, :, :], k_cache[:, :1, :, :]], axis=1
                    )
                else:
                    k_cache = k_cache[:, :1, :, :]
            if v_cache.ndim == 4 and int(v_cache.shape[1]) > 0:
                if int(v_cache.shape[1]) > 1:
                    v_cache = mx.concatenate(
                        [v_cache[:, :-1, :, :], v_cache[:, :1, :, :]], axis=1
                    )
                else:
                    v_cache = v_cache[:, :1, :, :]
            reset_cache.append((k_cache, v_cache))

        return reset_cache

    def _merge_language_caches(
        self,
        cache_a,
        cache_b,
    ):
        """Merge two per-layer cache lists into a batched cache list.

        This relies on mlx_lm cache `merge()` (KVCache -> BatchKVCache).
        Returns None when caches are not mergeable.
        """
        if (
            cache_a is None
            or cache_b is None
            or len(cache_a) == 0
            or len(cache_b) == 0
            or len(cache_a) != len(cache_b)
        ):
            return None

        merged = []
        for layer_a, layer_b in zip(cache_a, cache_b):
            if layer_a is None or layer_b is None:
                return None
            merge_fn = getattr(layer_a, "merge", None)
            if merge_fn is None:
                return None
            merged_layer = merge_fn([layer_a, layer_b])
            merged.append(merged_layer)
        return merged

    def _split_language_caches(self, merged_cache):
        """Split batched cache list back into two per-layer cache lists."""
        if merged_cache is None:
            return None, None
        first = []
        second = []
        for layer_cache in merged_cache:
            extract_fn = getattr(layer_cache, "extract", None)
            if extract_fn is None:
                return None, None
            first.append(extract_fn(0))
            second.append(extract_fn(1))
        return first, second

    def _dual_language_step(
        self,
        pos_embed: mx.array,
        pos_cache,
        neg_embed: mx.array,
        neg_cache,
    ):
        """Advance positive and negative LM branches.

        Tries batched cache merge for one forward pass; falls back to two passes.
        Returns: (pos_hidden, pos_cache, neg_hidden, neg_cache, used_dual_batch)
        """
        if os.getenv("MLX_AUDIO_VIBEVOICE_DISABLE_DUAL_BATCH", "0") != "1":
            try:
                merged_cache = self._merge_language_caches(pos_cache, neg_cache)
                if merged_cache is not None:
                    dual_in = mx.concatenate([pos_embed, neg_embed], axis=0)
                    dual_hidden, dual_cache = self.language_model(
                        inputs_embeds=dual_in,
                        cache=merged_cache,
                    )
                    new_pos_cache, new_neg_cache = self._split_language_caches(dual_cache)
                    if new_pos_cache is not None and new_neg_cache is not None:
                        return (
                            dual_hidden[:1],
                            new_pos_cache,
                            dual_hidden[1:2],
                            new_neg_cache,
                            True,
                        )
            except Exception:
                # Fall back to safe independent passes.
                pass

        # Safe fallback: independent passes.
        pos_hidden, pos_cache = self.language_model(inputs_embeds=pos_embed, cache=pos_cache)
        neg_hidden, neg_cache = self.language_model(inputs_embeds=neg_embed, cache=neg_cache)
        return pos_hidden, pos_cache, neg_hidden, neg_cache, False

    def _build_non_streaming_prompt_tokens(
        self,
        parsed_script: List[Tuple[int, str]],
        ref_prompt_steps: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
    ) -> Tuple[List[int], List[bool]]:
        """Build non-streaming prompt tokens and speech input mask."""
        if (
            self.speech_start_id is None
            or self.speech_end_id is None
            or self.speech_diffusion_id is None
        ):
            raise ValueError("Special token ids are not initialized.")
        if not parsed_script:
            raise ValueError("Parsed script is empty.")

        prompt_tokens = []
        speech_input_mask = []

        def _append(tokens: List[int], is_speech: bool = False) -> None:
            prompt_tokens.extend(tokens)
            speech_input_mask.extend([is_speech] * len(tokens))

        speaker_order = self._get_non_streaming_speaker_order(parsed_script)
        coerced_ref_steps = self._coerce_non_streaming_ref_prompt_steps(
            ref_prompt_steps,
            len(speaker_order),
        )

        system_tokens = self.tokenizer.encode(NON_STREAMING_SYSTEM_PROMPT)
        _append(system_tokens, is_speech=False)

        if coerced_ref_steps is not None:
            _append(self.tokenizer.encode(" Voice input:\n", add_special_tokens=False))
            for speaker_id, vae_tok_len in zip(speaker_order, coerced_ref_steps):
                _append(
                    self.tokenizer.encode(
                        f" Speaker {speaker_id}:", add_special_tokens=False
                    )
                )
                _append([self.speech_start_id], is_speech=False)
                _append([self.speech_diffusion_id] * int(vae_tok_len), is_speech=True)
                _append([self.speech_end_id], is_speech=False)
                _append(
                    self.tokenizer.encode("\n", add_special_tokens=False),
                    is_speech=False,
                )

        _append(self.tokenizer.encode(" Text input:\n", add_special_tokens=False))
        for speaker_id, speaker_text in parsed_script:
            _append(
                self.tokenizer.encode(
                    f" Speaker {speaker_id}:{speaker_text}\n",
                    add_special_tokens=False,
                )
            )
        _append(self.tokenizer.encode(" Speech output:\n", add_special_tokens=False))
        _append([self.speech_start_id], is_speech=False)
        return prompt_tokens, speech_input_mask

    def _prepare_single_non_streaming_ref_embedding(
        self, ref_audio: Union[str, Path, mx.array]
    ) -> mx.array:
        """Encode one reference audio sample into prefill speech embeddings."""
        if isinstance(ref_audio, (str, Path)):
            ref_path = str(ref_audio)
            try:
                import librosa

                ref_audio, _ = librosa.load(
                    ref_path,
                    sr=self.sample_rate,
                    mono=True,
                )
                ref_audio = mx.array(ref_audio, dtype=mx.float32)
            except Exception:
                from mlx_audio.utils import load_audio

                ref_audio = load_audio(ref_path, sample_rate=self.sample_rate)
        elif not isinstance(ref_audio, mx.array):
            ref_audio = mx.array(ref_audio, dtype=mx.float32)

        if ref_audio.ndim == 2:
            # Collapse channels to mono if needed.
            ref_audio = mx.mean(ref_audio, axis=0)
        if ref_audio.ndim != 1:
            ref_audio = ref_audio.reshape((-1,))
        ref_audio = ref_audio.astype(mx.float32)
        # Match upstream processor behavior: normalize reference audio to stable level.
        # This reduces conditioning variance and can lower hiss/noise in cloned voice.
        target = mx.array(10 ** (-25.0 / 20.0), dtype=mx.float32)
        eps = mx.array(1e-6, dtype=mx.float32)
        rms = mx.sqrt(mx.mean(ref_audio * ref_audio) + eps)
        scale = target / mx.maximum(rms, eps)
        ref_audio = ref_audio * scale
        max_abs = mx.max(mx.abs(ref_audio)) + eps
        ref_audio = ref_audio / mx.maximum(max_abs, mx.array(1.0, dtype=mx.float32))
        ref_audio_3d = ref_audio[None, None, :]

        acoustic_output = self.acoustic_tokenizer.encode(ref_audio_3d)
        acoustic_latents, _ = acoustic_output.sample(
            dist_type=self.acoustic_tokenizer.std_dist_type
        )
        acoustic_features = (
            acoustic_latents + self.speech_bias_factor
        ) * self.speech_scaling_factor
        # Match upstream non-streaming prefill exactly: reference voice slots use
        # acoustic-conditioned embeddings only. Semantic feedback is introduced
        # later during autoregressive speech generation, not during voice prompt
        # prefill.
        acoustic_embed = self.acoustic_connector(acoustic_features)
        return acoustic_embed[0]

    def _prepare_non_streaming_ref_embeddings(
        self,
        ref_audio: Union[
            str,
            Path,
            mx.array,
            List[Union[str, Path, mx.array]],
            Tuple[Union[str, Path, mx.array], ...],
        ],
    ) -> List[mx.array]:
        """Encode one or more reference audio samples into prefill speech embeddings."""
        if isinstance(ref_audio, (list, tuple)) and not isinstance(ref_audio, mx.array):
            ref_audios = list(ref_audio)
        else:
            ref_audios = [ref_audio]
        return [
            self._prepare_single_non_streaming_ref_embedding(ref_item)
            for ref_item in ref_audios
        ]

    def _inject_prefill_speech_embeddings(
        self,
        input_embeds: mx.array,
        speech_input_mask: List[bool],
        speech_embeds: Union[mx.array, List[mx.array], Tuple[mx.array, ...]],
    ) -> mx.array:
        """Replace prompt diffusion placeholder embeddings with reference speech embeds."""
        speech_positions = [i for i, is_speech in enumerate(speech_input_mask) if is_speech]
        if not speech_positions:
            return input_embeds

        if isinstance(speech_embeds, (list, tuple)):
            flattened_parts = [
                part
                for part in speech_embeds
                if part is not None and int(part.shape[0]) > 0
            ]
            if not flattened_parts:
                return input_embeds
            if len(flattened_parts) == 1:
                speech_embeds = flattened_parts[0]
            else:
                speech_embeds = mx.concatenate(flattened_parts, axis=0)

        max_fill = min(len(speech_positions), int(speech_embeds.shape[0]))
        if max_fill <= 0:
            return input_embeds

        speech_pos = mx.array(speech_positions[:max_fill], dtype=mx.int32)
        speech_vals = speech_embeds[:max_fill].astype(input_embeds.dtype)
        current_vals = input_embeds[0, speech_pos, :]
        delta = speech_vals - current_vals
        return input_embeds.at[0, speech_pos, :].add(delta)

    def _select_non_streaming_next_token(
        self,
        logits: mx.array,
        allow_speech_start: bool = False,
        allow_termination: bool = True,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        token_counts: Optional[dict] = None,
        return_details: bool = False,
    ) -> Union[int, Tuple[int, dict]]:
        """Constrain generation to VibeVoice control tokens and select next token."""
        if (
            self.speech_start_id is None
            or self.speech_end_id is None
            or self.speech_diffusion_id is None
            or self.eos_id is None
        ):
            raise ValueError("Special token ids are not initialized.")

        if allow_termination:
            candidates = [self.speech_end_id, self.speech_diffusion_id, self.eos_id]
        else:
            candidates = [self.speech_diffusion_id]
        if allow_speech_start:
            candidates.insert(0, self.speech_start_id)
        if self.bos_id is not None and self.bos_id not in candidates:
            candidates.append(self.bos_id)
        candidate_ids = mx.array(candidates, dtype=mx.int32)
        candidate_logits_mx = logits[0, candidate_ids]
        candidate_scores = _control_score_dict(
            candidate_ids=candidates,
            candidate_logits=candidate_logits_mx.astype(mx.float32),
            speech_start_id=self.speech_start_id,
            speech_end_id=self.speech_end_id,
            speech_diffusion_id=self.speech_diffusion_id,
            eos_id=self.eos_id,
            bos_id=self.bos_id,
        )

        # Fast path for deterministic decoding without repetition penalty.
        # Avoids per-step NumPy conversion/sync on the hot path.
        if (
            not do_sample
            and (repetition_penalty is None or float(repetition_penalty) <= 1.0)
        ):
            best_idx = int(mx.argmax(candidate_logits_mx).item())
            result = candidates[best_idx]
            if return_details:
                return result, candidate_scores
            return result

        candidate_logits = candidate_logits_mx

        if (
            repetition_penalty is not None
            and repetition_penalty > 1.0
            and token_counts is not None
        ):
            counts = mx.array(
                [float(token_counts.get(token_id, 0)) for token_id in candidates],
                dtype=mx.float32,
            )
            penalty_base = mx.array(float(repetition_penalty), dtype=mx.float32)
            penalties = mx.power(penalty_base, counts)
            positive_mask = candidate_logits >= 0
            candidate_logits = mx.where(
                positive_mask,
                candidate_logits / penalties,
                candidate_logits * penalties,
            )

        if not do_sample:
            best_idx = int(mx.argmax(candidate_logits).item())
            result = candidates[best_idx]
            if return_details:
                return result, candidate_scores
            return result

        temp = max(float(temperature), 1e-5)
        sample_logits = candidate_logits.astype(mx.float32) / temp
        num_candidates = len(candidates)

        # Work in descending-logit order for top-k / top-p filtering.
        sorted_idx = mx.argsort(-sample_logits, axis=-1)
        sorted_logits = sample_logits[sorted_idx]
        neg_inf = mx.array(-1e9, dtype=sorted_logits.dtype)
        positions = mx.arange(num_candidates, dtype=mx.int32)

        k = (
            num_candidates
            if top_k is None
            else max(1, min(int(top_k), num_candidates))
        )
        if k < num_candidates:
            keep_k = positions < int(k)
            sorted_logits = mx.where(keep_k, sorted_logits, neg_inf)

        if top_p is not None and 0.0 < float(top_p) < 1.0:
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
            keep_p = cumsum_probs <= float(top_p)
            # Keep at least one token to avoid invalid all-masked distribution.
            keep_p = mx.logical_or(keep_p, positions == 0)
            sorted_logits = mx.where(keep_p, sorted_logits, neg_inf)

        # mx.random.categorical samples index over last dim from logits.
        sampled_pos = int(mx.random.categorical(sorted_logits[None, :])[0].item())
        sampled_local_idx = int(sorted_idx[sampled_pos].item())
        result = candidates[sampled_local_idx]
        if return_details:
            return result, candidate_scores
        return result

    def _split_non_streaming_text(
        self, text: str, max_chars_per_chunk: int = 1000
    ) -> List[str]:
        """Split long text using upstream batched-demo heuristics (2-4 chunks)."""
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= max_chars_per_chunk:
            return [normalized]

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[\.\!\?\;\:\…])\s+", normalized)
            if s.strip()
        ]
        if not sentences:
            return [normalized]

        total_chars = sum(len(s) for s in sentences)
        if total_chars <= 1000:
            num_batches = 2
        elif total_chars <= 2500:
            num_batches = 3
        else:
            num_batches = 4

        target_per_batch = max(1, total_chars // num_batches)
        chunks: List[str] = []
        cur_parts: List[str] = []
        cur_chars = 0

        for idx, sent in enumerate(sentences):
            cur_parts.append(sent)
            cur_chars += len(sent)
            sentences_left = len(sentences) - (idx + 1)
            must_cut = (cur_chars >= target_per_batch and len(chunks) < num_batches - 1) or (
                (sentences_left + len(chunks) + 1) <= (num_batches - len(chunks))
            )
            if must_cut:
                chunks.append(" ".join(cur_parts).strip())
                cur_parts = []
                cur_chars = 0

        if cur_parts:
            chunks.append(" ".join(cur_parts).strip())

        return [c for c in chunks if c] or [normalized]

    def load_voice(self, voice: str) -> None:
        """Load a VibeVoice voice-cache (.safetensors) for conditioning.


        Expected keys (per layer):
          - lm_hidden
          - lm_key_{i}, lm_value_{i}
          - tts_lm_hidden
          - tts_lm_key_{i}, tts_lm_value_{i}
          - neg_tts_lm_hidden
          - neg_lm_key_{i}, neg_lm_value_{i}   (optional for our inference)
          - neg_tts_lm_key_{i}, neg_tts_lm_value_{i}
        """
        if self.is_non_streaming:
            raise NotImplementedError(
                "Voice-cache loading is only available for vibevoice_streaming models."
            )

        voice_path = Path(self.config.model_path) / f"voices/{voice}.safetensors"

        if not voice_path.exists():
            raise FileNotFoundError(f"Voice cache not found: {voice_path}")

        tensors = mx.load(str(voice_path))

        lm_layers = self.language_model.config.num_hidden_layers
        tts_layers = self.tts_language_model.config.num_hidden_layers

        def _mx(name: str) -> mx.array:
            if name not in tensors:
                raise KeyError(f"Voice cache missing key: {name}")
            return mx.array(tensors[name])

        def _load_kv(prefix: str, i: int) -> Tuple[mx.array, mx.array]:
            k = _mx(f"{prefix}_key_{i}")
            v = _mx(f"{prefix}_value_{i}")
            # Swift caches are stored as (B, kv_heads, seq, head_dim).
            # Our attention cache expects (B, seq, kv_heads, head_dim).
            if k.ndim == 4:
                k = mx.transpose(k, (0, 2, 1, 3))
            if v.ndim == 4:
                v = mx.transpose(v, (0, 2, 1, 3))
            return (k, v)

        # Store caches/hidden states for generation
        self._voice_path = str(voice_path)
        self._voice_lm_hidden = _mx("lm_hidden")
        self._voice_tts_hidden = _mx("tts_lm_hidden")
        self._voice_neg_tts_hidden = _mx("neg_tts_lm_hidden")

        self._voice_lm_cache = [_load_kv("lm", i) for i in range(lm_layers)]
        self._voice_tts_cache = [_load_kv("tts_lm", i) for i in range(tts_layers)]
        self._voice_neg_tts_cache = [
            _load_kv("neg_tts_lm", i) for i in range(tts_layers)
        ]

        if all(
            f"neg_lm_key_{i}" in tensors and f"neg_lm_value_{i}" in tensors
            for i in range(lm_layers)
        ):
            self._voice_neg_lm_cache = [_load_kv("neg_lm", i) for i in range(lm_layers)]
        else:
            self._voice_neg_lm_cache = None

    def get_input_embeddings(self) -> nn.Embedding:
        """Get the token embedding layer."""
        return self.language_model.embed_tokens

    @staticmethod
    def _cast_module_dtype(module: nn.Module, dtype: mx.Dtype) -> None:
        """Recursively cast a module's parameters to the requested dtype."""
        from mlx.utils import tree_map

        module.update(tree_map(lambda p: p.astype(dtype), module.parameters()))

    def sanitize(self, weights: dict) -> dict:
        """Sanitize weights for loading from HuggingFace format."""
        import re

        from mlx.utils import tree_flatten

        new_weights = {}
        curr_shapes = {k: v.shape for k, v in tree_flatten(self.parameters())}

        def transform_key(key: str) -> str:
            """Transform HuggingFace key to MLX key format."""
            # Remove "model." prefix
            if key.startswith("model."):
                key = key[6:]

            # Prediction head transformations
            # t_embedder.mlp.0 -> t_embedder.mlp.layers.0
            key = re.sub(
                r"\.t_embedder\.mlp\.(\d+)\.", r".t_embedder.mlp.layers.\1.", key
            )
            # Handle weight at end: t_embedder.mlp.0.weight -> t_embedder.mlp.layers.0.weight
            key = re.sub(
                r"\.t_embedder\.mlp\.(\d+)\.weight$",
                r".t_embedder.mlp.layers.\1.weight",
                key,
            )

            # adaLN_modulation.1 -> adaLN_modulation.layers.1
            key = re.sub(
                r"\.adaLN_modulation\.(\d+)\.", r".adaLN_modulation.layers.\1.", key
            )
            key = re.sub(
                r"\.adaLN_modulation\.(\d+)\.weight$",
                r".adaLN_modulation.layers.\1.weight",
                key,
            )

            return key

        for k, v in weights.items():
            new_key = transform_key(k)

            # Handle scaling factors specially
            if "speech_scaling_factor" in new_key:
                new_weights["speech_scaling_factor"] = v
                continue
            if "speech_bias_factor" in new_key:
                new_weights["speech_bias_factor"] = v
                continue

            # Skip rotary embedding inv_freq (computed at init)
            if "rotary_emb.inv_freq" in new_key:
                continue

            # Check if key exists in model
            if new_key not in curr_shapes:
                # Preserve quantization metadata -- model isn't quantized yet at sanitize time
                if new_key.endswith((".scales", ".biases")):
                    new_weights[new_key] = v
                continue

            target_shape = curr_shapes[new_key]

            # Handle shape mismatches
            if v.shape == target_shape:
                new_weights[new_key] = v
            elif len(v.shape) == 2 and v.T.shape == target_shape:
                # Transpose Linear weights (PyTorch vs MLX layout)
                new_weights[new_key] = v.T
            elif len(v.shape) == 3:
                # Check if it's a transposed conv weight
                is_convtr = "convtr" in new_key

                if is_convtr:
                    # ConvTranspose1d: PyTorch (C_in, C_out, K) -> MLX (C_out, K, C_in)
                    # Need to transpose (0, 1, 2) -> (1, 2, 0)
                    if (
                        v.shape[1] == target_shape[0]
                        and v.shape[2] == target_shape[1]
                        and v.shape[0] == target_shape[2]
                    ):
                        new_weights[new_key] = mx.transpose(v, (1, 2, 0))
                    else:
                        new_weights[new_key] = v
                else:
                    # Conv1d weights: PyTorch (C_out, C_in, K) -> MLX (C_out, K, C_in)
                    if (
                        v.shape[0] == target_shape[0]
                        and v.shape[1] == target_shape[2]
                        and v.shape[2] == target_shape[1]
                    ):
                        # Swap last two dimensions
                        new_weights[new_key] = mx.transpose(v, (0, 2, 1))
                    else:
                        new_weights[new_key] = v
            else:
                # Keep as is - might be a different kind of mismatch
                new_weights[new_key] = v

        return new_weights

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Hook called after model weights are loaded."""
        import json

        if model.tokenizer is None:
            # Try to load tokenizer from preprocessor_config
            preprocessor_config_path = model_path / "preprocessor_config.json"
            tokenizer_name = "Qwen/Qwen2.5-0.5B"  # Default

            if preprocessor_config_path.exists():
                with open(preprocessor_config_path, encoding="utf-8") as f:
                    preprocessor_config = json.load(f)
                    tokenizer_name = preprocessor_config.get(
                        "language_model_pretrained_name", tokenizer_name
                    )

            model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if (
                getattr(model.tokenizer, "pad_token_id", None) is None
                and getattr(model.tokenizer, "eos_token", None) is not None
            ):
                model.tokenizer.pad_token = model.tokenizer.eos_token

        if model.is_non_streaming:
            model._resolve_non_streaming_token_ids()
            if os.getenv("MLX_AUDIO_VIBEVOICE_USE_MLX_LM_QWEN2", "0") == "1":
                print("Using mlx_lm qwen2 backbone for non-streaming LM.")
            if os.getenv("MLX_AUDIO_VIBEVOICE_LM_FP32", "0") == "1":
                model._cast_module_dtype(model.language_model, mx.float32)
                model._cast_module_dtype(model.lm_head, mx.float32)
            # Keep an eager reference for safe runtime fallback.
            model._prediction_head_eager = model.prediction_head
            # Optional hot-path compilation for repeated diffusion head calls.
            # Enabled by default; can be disabled with MLX_AUDIO_VIBEVOICE_COMPILE=0.
            # Compile mode:
            # - default (recommended): mx.compile(...)
            # - shapeless: mx.compile(..., shapeless=True)
            compile_mode = os.getenv("MLX_AUDIO_VIBEVOICE_COMPILE_MODE", "default")
            if os.getenv("MLX_AUDIO_VIBEVOICE_COMPILE", "1") != "0":
                try:
                    if compile_mode == "shapeless":
                        model.prediction_head = mx.compile(
                            model.prediction_head, shapeless=True
                        )
                    else:
                        model.prediction_head = mx.compile(model.prediction_head)
                except Exception:
                    # Keep functional behavior if compilation is unavailable.
                    model.prediction_head = model._prediction_head_eager
        return model

    def sample_speech_tokens(
        self,
        condition: mx.array,
        neg_condition: mx.array,
        cfg_scale: float = 3.0,
        ddpm_steps: Optional[int] = None,
        diffusion_fp32: bool = False,
        cfg_active_ratio: float = 1.0,
        noise_override: Optional[mx.array] = None,
        timing_info: Optional[dict[str, float]] = None,
    ) -> mx.array:
        """Sample speech latents using diffusion with classifier-free guidance.

        Args:
            condition: Positive conditioning, shape (B, hidden_size)
            neg_condition: Negative conditioning, shape (B, hidden_size)
            cfg_scale: Classifier-free guidance scale

        Returns:
            Sampled speech latents, shape (B, acoustic_vae_dim)
        """
        # Precision policy:
        # - diffusion_fp32=True: quality-priority path
        # - diffusion_fp32=False: speed-priority path (uses model dtype, e.g. bfloat16)
        diffusion_dtype = mx.float32 if diffusion_fp32 else condition.dtype
        condition = condition.astype(diffusion_dtype)
        neg_condition = neg_condition.astype(diffusion_dtype)

        # Reset scheduler for new generation
        self.noise_scheduler.reset()
        num_steps = int(ddpm_steps or self.ddpm_inference_steps)
        if self._cached_ddpm_steps != num_steps or self._cached_timesteps is None:
            self.noise_scheduler.set_timesteps(num_steps)
            self._cached_ddpm_steps = num_steps
            self._cached_timesteps = [int(t) for t in self.noise_scheduler.timesteps.tolist()]
        timesteps_list = self._cached_timesteps

        # Initialize noise
        batch_size = condition.shape[0]
        latent_dim = self.config.acoustic_vae_dim
        if noise_override is None:
            speech = mx.random.normal((batch_size, latent_dim), dtype=diffusion_dtype)
        else:
            speech = noise_override.astype(diffusion_dtype)
            if speech.ndim == 1:
                speech = mx.expand_dims(speech, axis=0)
            expected_shape = (batch_size, latent_dim)
            actual_shape = tuple(int(dim) for dim in speech.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"diffusion noise override must have shape {expected_shape}, "
                    f"got {actual_shape}"
                )

        prev_x0 = None

        use_negative_cfg = float(cfg_scale) > 1.0 + 1e-6
        # `cfg_active_ratio` is intentionally ignored for stability: always run
        # full CFG when enabled.
        del cfg_active_ratio
        active_cfg_steps = len(timesteps_list) if use_negative_cfg else 0
        timing_sync = os.getenv("MLX_AUDIO_VIBEVOICE_TIMING", "0") == "1"
        detailed_timing = timing_sync and os.getenv(
            "MLX_AUDIO_VIBEVOICE_DETAILED_TIMING", "0"
        ) == "1"
        layer_sync_timing = detailed_timing and os.getenv(
            "MLX_AUDIO_VIBEVOICE_LAYER_SYNC_TIMING", "0"
        ) == "1"
        if detailed_timing and layer_sync_timing:
            timing_info["__sync__"] = True
        projected_condition = self.prediction_head.project_condition(condition)
        projected_neg_condition = (
            self.prediction_head.project_condition(neg_condition)
            if use_negative_cfg
            else None
        )
        timestep_condition_cache = {}

        for step_idx, t_val in enumerate(timesteps_list):
            t_float = float(t_val)
            apply_cfg = use_negative_cfg and step_idx < active_cfg_steps
            if apply_cfg:
                # Batched positive+negative prediction for CFG.
                condition_combined = mx.concatenate(
                    [projected_condition, projected_neg_condition], axis=0
                )
                cond_batch = int(condition_combined.shape[0])
                cache_key = ("cfg", int(t_val), cond_batch)
                timestep_condition = timestep_condition_cache.get(cache_key)
                if timestep_condition is None:
                    timestep_single = mx.array([t_float], dtype=diffusion_dtype)
                    timestep_embed_single = self.prediction_head.embed_timesteps(
                        timestep_single
                    )
                    timestep_embed = mx.broadcast_to(
                        timestep_embed_single, (cond_batch, timestep_embed_single.shape[-1])
                    )
                    timestep_condition = condition_combined + timestep_embed
                    timestep_condition_cache[cache_key] = timestep_condition
                combined_speech = mx.concatenate([speech, speech], axis=0)
                pred_t0 = None
                if detailed_timing:
                    mx.eval(combined_speech, timestep_condition)
                    pred_t0 = time.perf_counter()
                try:
                    eps = self.prediction_head.forward_with_condition(
                        combined_speech,
                        condition_with_timestep=timestep_condition,
                        timing_info=timing_info if detailed_timing else None,
                    )
                except ValueError as exc:
                    # Some MLX builds can fail at runtime for compiled graphs with
                    # dynamic shape propagation (for example Split inference). Fall
                    # back to eager to preserve correctness and avoid hard failure.
                    eager_head = getattr(self, "_prediction_head_eager", None)
                    if eager_head is None or self.prediction_head is eager_head:
                        raise
                    self.prediction_head = eager_head
                    eps = self.prediction_head.forward_with_condition(
                        combined_speech,
                        condition_with_timestep=timestep_condition,
                        timing_info=timing_info if detailed_timing else None,
                    )
                    if os.getenv("MLX_AUDIO_VIBEVOICE_TIMING", "0") == "1":
                        print(
                            f"[INFO] Disabled compiled prediction_head after runtime error: {exc}"
                        )
                if detailed_timing:
                    mx.eval(eps)
                    timing_info["diffusion_pred_head"] = (
                        timing_info.get("diffusion_pred_head", 0.0)
                        + (time.perf_counter() - pred_t0)
                    )

                cond_eps = eps[:batch_size]
                uncond_eps = eps[batch_size:]
                guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            else:
                # Fast path: no negative branch (cfg_scale<=1 or sparse-CFG tail steps).
                cache_key = ("pos", int(t_val), int(projected_condition.shape[0]))
                timestep_condition = timestep_condition_cache.get(cache_key)
                if timestep_condition is None:
                    timestep_single = mx.array([t_float], dtype=diffusion_dtype)
                    timestep_embed_single = self.prediction_head.embed_timesteps(
                        timestep_single
                    )
                    timestep_embed = mx.broadcast_to(
                        timestep_embed_single,
                        (
                            int(projected_condition.shape[0]),
                            timestep_embed_single.shape[-1],
                        ),
                    )
                    timestep_condition = projected_condition + timestep_embed
                    timestep_condition_cache[cache_key] = timestep_condition
                pred_t0 = None
                if detailed_timing:
                    mx.eval(speech, timestep_condition)
                    pred_t0 = time.perf_counter()
                try:
                    guided_eps = self.prediction_head.forward_with_condition(
                        speech,
                        condition_with_timestep=timestep_condition,
                        timing_info=timing_info if detailed_timing else None,
                    )
                except ValueError as exc:
                    eager_head = getattr(self, "_prediction_head_eager", None)
                    if eager_head is None or self.prediction_head is eager_head:
                        raise
                    self.prediction_head = eager_head
                    guided_eps = self.prediction_head.forward_with_condition(
                        speech,
                        condition_with_timestep=timestep_condition,
                        timing_info=timing_info if detailed_timing else None,
                    )
                    if os.getenv("MLX_AUDIO_VIBEVOICE_TIMING", "0") == "1":
                        print(
                            f"[INFO] Disabled compiled prediction_head after runtime error: {exc}"
                        )
                if detailed_timing:
                    mx.eval(guided_eps)
                    timing_info["diffusion_pred_head"] = (
                        timing_info.get("diffusion_pred_head", 0.0)
                        + (time.perf_counter() - pred_t0)
                    )

            # Scheduler step with multi-order support on the guided branch only.
            sched_t0 = None
            if detailed_timing:
                mx.eval(guided_eps, speech)
                sched_t0 = time.perf_counter()
            output = self.noise_scheduler.step(
                guided_eps,
                t_val,
                speech,
                prev_x0=prev_x0,
            )
            if detailed_timing:
                mx.eval(output.prev_sample)
                timing_info["diffusion_scheduler"] = (
                    timing_info.get("diffusion_scheduler", 0.0)
                    + (time.perf_counter() - sched_t0)
                )

            speech = output.prev_sample
            prev_x0 = output.x0_pred if output.x0_pred is not None else None
            if timing_sync:
                # Materialize per-step output only in timing mode so section-level
                # profiling reflects actual kernel runtime under MLX lazy execution.
                mx.eval(speech)

        return speech

    def generate(
        self,
        text: Union[str, List[str]],
        max_tokens: Optional[int] = None,
        cfg_scale: float = 1.5,
        ddpm_steps: Optional[int] = None,
        voice: Optional[Union[str, Path, List[Tuple[str, str]]]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text.

        Args:
            text: Input text to synthesize (must be a list if voice is a list of tuples)
            max_tokens: Optional max control tokens to generate.
                If None, non-streaming path uses model context budget.
            cfg_scale: Classifier-free guidance scale
            ddpm_steps: Override diffusion inference steps (higher = better quality, slower)
            voice: Either a single voice name/path, or a list
            verbose: Whether to show progress

        Yields:
            GenerationResult containing audio and metadata
        """
        if self.is_non_streaming:
            yield from self._generate_non_streaming(
                text=text if isinstance(text, str) else "\n".join(text),
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                ddpm_steps=ddpm_steps,
                verbose=verbose,
                **kwargs,
            )
            return

        stream_max_tokens = int(max_tokens) if max_tokens is not None else 512

        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

        # Handle multi-speaker dialogue mode
        if isinstance(text, list) and isinstance(voice, list):
            if len(text) != len(voice):
                raise ValueError(
                    f"text and voice lists must have the same length. "
                    f"Got {len(text)} texts and {len(voice)} voices."
                )

            dialogue = list(zip(voice, text))
            yield from self._generate_multi_speaker(
                dialogue=dialogue,
                max_tokens=stream_max_tokens,
                cfg_scale=cfg_scale,
                ddpm_steps=ddpm_steps,
                verbose=verbose,
                **kwargs,
            )
            return

        # Single voice mode - load voice and delegate to single speaker generator
        if voice is not None:
            # Only reload if different
            if not hasattr(self, "_voice_path") or str(voice) != getattr(
                self, "_voice_path"
            ):
                self.load_voice(voice)

        yield from self._generate_single_speaker(
            text=text,
            max_tokens=stream_max_tokens,
            cfg_scale=cfg_scale,
            ddpm_steps=ddpm_steps,
            verbose=verbose,
            **kwargs,
        )

    def _generate_non_streaming(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        cfg_scale: float = 1.0,
        ddpm_steps: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Non-streaming generation loop for `vibevoice` checkpoints."""
        allow_chunking_opt = kwargs.pop("_allow_chunking", kwargs.pop("allow_chunking", None))
        chunked_mode = bool(kwargs.pop("_chunked_mode", False))
        prefill_ref_embeds = kwargs.pop("_prefill_ref_embeds", None)
        prefill_ref_prompt_steps = kwargs.pop("_ref_prompt_steps", None)
        ref_audio = kwargs.pop("ref_audio", None)
        do_sample = kwargs.pop("do_sample", None)
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        top_k = kwargs.pop("top_k", None)
        # The shared `mlx_audio.tts.generate` CLI keeps its generic default
        # repetition_penalty=1.1 for contract stability. Most VibeVoice parity
        # and quality tuning here was exercised with `None` / model-level defaults.
        repetition_penalty = kwargs.pop("repetition_penalty", None)
        use_semantic_feedback = kwargs.pop("use_semantic_feedback", None)
        diffusion_fp32 = bool(kwargs.pop("diffusion_fp32", False))
        requested_cfg_active_ratio = float(kwargs.pop("cfg_active_ratio", 1.0))
        max_length_times = float(kwargs.pop("max_length_times", 2.0))
        refresh_negative = bool(kwargs.pop("refresh_negative", True))
        redecode_final_audio = bool(kwargs.pop("redecode_final_audio", False))
        feedback_acoustic_use_cache = bool(
            kwargs.pop("feedback_acoustic_use_cache", True)
        )
        feedback_semantic_use_cache = bool(
            kwargs.pop("feedback_semantic_use_cache", True)
        )
        trace_jsonl_path = kwargs.pop("trace_jsonl_path", None)
        trace_limit = kwargs.pop("trace_limit", None)
        diffusion_noise_path = kwargs.pop("diffusion_noise_path", None)
        lm_input_dump_path = kwargs.pop("lm_input_dump_path", None)
        feedback_component_dump_path = kwargs.pop("feedback_component_dump_path", None)
        lm_transition_dump_path = kwargs.pop("lm_transition_dump_path", None)
        lm_layer_probe_dump_path = kwargs.pop("lm_layer_probe_dump_path", None)
        lm_layer_probe_steps = kwargs.pop("lm_layer_probe_steps", None)
        prefill_probe_dump_path = kwargs.pop("prefill_probe_dump_path", None)
        if trace_limit is not None:
            trace_limit = int(trace_limit)
        layer_probe_steps = None
        if lm_layer_probe_steps:
            layer_probe_steps = {int(step) for step in lm_layer_probe_steps}
        # Unused generic kwargs passed by the CLI wrapper.
        del kwargs

        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call post_load_hook first.")
        self._resolve_non_streaming_token_ids()
        if self.lm_head is None:
            raise ValueError("lm_head is required for non-streaming vibevoice.")
        if not hasattr(self, "semantic_tokenizer") or self.semantic_tokenizer is None:
            raise ValueError("semantic_tokenizer is required for non-streaming vibevoice.")
        if not hasattr(self, "acoustic_tokenizer") or self.acoustic_tokenizer is None:
            raise ValueError("acoustic_tokenizer is required for non-streaming vibevoice.")

        diffusion_noise_sequence = None
        if diffusion_noise_path:
            diffusion_noise_sequence = _load_diffusion_noise_sequence_mlx(
                diffusion_noise_path,
                latent_dim=self.config.acoustic_vae_dim,
            )

        do_sample = bool(do_sample) if do_sample is not None else False
        if use_semantic_feedback is None:
            use_semantic_feedback = True
        use_negative_cfg = float(cfg_scale) > 1.0 + 1e-6
        cfg_active_ratio = 1.0
        recompute_positive_without_cache = (
            os.getenv("MLX_AUDIO_VIBEVOICE_POS_NO_CACHE", "0") == "1"
        )
        dual_batch_allowed = os.getenv("MLX_AUDIO_VIBEVOICE_DISABLE_DUAL_BATCH", "0") != "1"
        use_dual_lm_batch = (
            use_negative_cfg
            and not refresh_negative
            and not recompute_positive_without_cache
        )
        if not use_negative_cfg:
            # Fast path: negative branch is unnecessary when CFG guidance is disabled.
            refresh_negative = False
            if verbose:
                print("Using fast CFG path: negative branch disabled (cfg_scale <= 1.0).")
        elif use_dual_lm_batch and verbose:
            if dual_batch_allowed:
                print("Using dual LM batch path for negative branch updates.")
            else:
                print("Dual LM batching disabled by env; using two-pass fallback.")
        if verbose and abs(requested_cfg_active_ratio - 1.0) > 1e-8:
            print("Note: cfg_active_ratio is disabled; using full CFG for all diffusion steps.")
        if recompute_positive_without_cache and verbose:
            print("Diagnostic mode: positive LM branch recomputed without KV cache.")

        parsed_script = self._parse_non_streaming_script(text)
        speaker_order = self._get_non_streaming_speaker_order(parsed_script)
        ref_audio_inputs = None
        if ref_audio is not None:
            ref_audio_inputs = self._normalize_non_streaming_ref_audios(
                ref_audio=ref_audio,
                expected_speakers=len(speaker_order),
            )

        if allow_chunking_opt is None:
            # Match upstream behavior: do not auto-chunk inside core generation.
            # Chunking is opt-in via CLI flag only.
            allow_chunking = False
        else:
            allow_chunking = bool(allow_chunking_opt)

        if allow_chunking and len(speaker_order) > 1:
            allow_chunking = False
            if verbose:
                print("Skipping auto-chunking for multi-speaker non-streaming script.")

        if allow_chunking:
            text_chunks = self._split_non_streaming_text(text)
            if len(text_chunks) > 1:
                if verbose:
                    print(
                        f"Long text detected: split into {len(text_chunks)} chunks for stability."
                    )
                start_time = time.perf_counter()
                all_audio = []
                total_tokens = 0
                shared_ref_embeds = prefill_ref_embeds
                if shared_ref_embeds is not None and isinstance(shared_ref_embeds, mx.array):
                    shared_ref_embeds = [shared_ref_embeds]
                shared_ref_steps = prefill_ref_prompt_steps
                if shared_ref_embeds is None and ref_audio_inputs is not None:
                    shared_ref_embeds = self._prepare_non_streaming_ref_embeddings(
                        ref_audio_inputs
                    )
                    shared_ref_steps = [int(embed.shape[0]) for embed in shared_ref_embeds]
                    if verbose:
                        print(
                            "Using reference voice conditioning "
                            f"({sum(shared_ref_steps)} prompt steps across "
                            f"{len(shared_ref_steps)} speaker(s))."
                        )

                for idx, chunk in enumerate(text_chunks):
                    if verbose:
                        print(f"Generating chunk {idx + 1}/{len(text_chunks)}")
                    subgen = self._generate_non_streaming(
                        text=chunk,
                        max_tokens=max_tokens,
                        cfg_scale=cfg_scale,
                        ddpm_steps=ddpm_steps,
                        verbose=verbose,
                        _allow_chunking=False,
                        _chunked_mode=True,
                        ref_audio=None if shared_ref_embeds is not None else ref_audio_inputs,
                        _prefill_ref_embeds=shared_ref_embeds,
                        _ref_prompt_steps=shared_ref_steps,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        cfg_active_ratio=cfg_active_ratio,
                    )
                    sub_result = next(subgen)
                    all_audio.append(sub_result.audio)
                    total_tokens += sub_result.token_count

                final_audio = (
                    mx.concatenate(all_audio, axis=0) if all_audio else mx.array([])
                )
                elapsed_time = time.perf_counter() - start_time
                samples = final_audio.shape[0] if final_audio.size > 0 else 0
                audio_duration_seconds = samples / self.sample_rate if samples > 0 else 0.0
                duration_mins = int(audio_duration_seconds // 60)
                duration_secs = int(audio_duration_seconds % 60)
                duration_ms = int((audio_duration_seconds % 1) * 1000)
                duration_str = (
                    f"{int(audio_duration_seconds // 3600):02d}:"
                    f"{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"
                )
                rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0.0

                yield GenerationResult(
                    audio=final_audio,
                    samples=samples,
                    sample_rate=self.sample_rate,
                    segment_idx=0,
                    token_count=total_tokens,
                    audio_duration=duration_str,
                    real_time_factor=rtf,
                    prompt={
                        "tokens": total_tokens,
                        "tokens-per-sec": (
                            round(total_tokens / elapsed_time, 2) if elapsed_time > 0 else 0
                        ),
                    },
                    audio_samples={
                        "samples": samples,
                        "samples-per-sec": (
                            round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                        ),
                    },
                    processing_time_seconds=elapsed_time,
                    peak_memory_usage=float(mx.get_peak_memory() / 1e9),
                )
                return

        start_time = time.perf_counter()
        trace_handle = None
        if trace_jsonl_path:
            trace_path = Path(trace_jsonl_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = trace_path.open("w", encoding="utf-8")
        timing_sync = os.getenv("MLX_AUDIO_VIBEVOICE_TIMING", "0") == "1"
        prefill_start = time.perf_counter()
        ref_embeds = prefill_ref_embeds
        if ref_embeds is not None and isinstance(ref_embeds, mx.array):
            ref_embeds = [ref_embeds]
        ref_prompt_steps = prefill_ref_prompt_steps
        if ref_embeds is None and ref_audio_inputs is not None:
            ref_embeds = self._prepare_non_streaming_ref_embeddings(ref_audio_inputs)
            ref_prompt_steps = [int(embed.shape[0]) for embed in ref_embeds]
            if verbose:
                print(
                    "Using reference voice conditioning "
                    f"({sum(ref_prompt_steps)} prompt steps across "
                    f"{len(ref_prompt_steps)} speaker(s))."
                )
        elif ref_embeds is not None and ref_prompt_steps is None:
            ref_prompt_steps = self._coerce_non_streaming_ref_prompt_steps(
                [int(embed.shape[0]) for embed in ref_embeds],
                len(speaker_order),
            )
        elif ref_prompt_steps is not None:
            ref_prompt_steps = self._coerce_non_streaming_ref_prompt_steps(
                ref_prompt_steps,
                len(speaker_order),
            )

        prompt_tokens, speech_input_mask = self._build_non_streaming_prompt_tokens(
            parsed_script=parsed_script,
            ref_prompt_steps=ref_prompt_steps if ref_embeds is not None else None,
        )
        input_ids = mx.array([prompt_tokens], dtype=mx.int32)
        initial_length = int(input_ids.shape[1])
        length_budget = max(1, int(max_length_times * initial_length))
        model_ctx_limit = int(
            getattr(self.language_model.config, "max_position_embeddings", 0) or 0
        )
        if model_ctx_limit > 0:
            context_budget = max(1, model_ctx_limit - initial_length)
        else:
            context_budget = max(1, int(max_tokens) if max_tokens is not None else 1200)
        user_budget = context_budget if max_tokens is None else max(1, int(max_tokens))
        max_steps = min(length_budget, context_budget, user_budget)
        if verbose:
            print(
                f"Max generation steps: {max_steps} "
                f"(length={length_budget}, context={context_budget}, user={user_budget})"
            )

        # Prefill with prompt.
        input_embeds = self.language_model.embed_tokens(input_ids)
        if ref_embeds is not None:
            input_embeds = self._inject_prefill_speech_embeddings(
                input_embeds=input_embeds,
                speech_input_mask=speech_input_mask,
                speech_embeds=ref_embeds,
            )
        lm_input_chunks = [input_embeds.astype(mx.float32)] if lm_input_dump_path else None
        # Use efficient KVCache path for the main generation branch.
        cache = self.language_model.make_cache()
        hidden, cache = self.language_model(inputs_embeds=input_embeds, cache=cache)
        if prefill_probe_dump_path:
            prefill_path = Path(prefill_probe_dump_path)
            prefill_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prefill_path,
                input_ids=np.array(input_ids, dtype=np.int32),
                inputs_embeds=np.array(input_embeds.astype(mx.float32), dtype=np.float32),
                prefill_last_hidden=np.array(
                    hidden[:, -1, :].astype(mx.float32), dtype=np.float32
                ),
            )
        if timing_sync:
            mx.eval(hidden)
        prefill_time = time.perf_counter() - prefill_start
        full_positive_embeds = input_embeds if recompute_positive_without_cache else None

        generated_steps = 0
        diffusion_steps = 0
        audio_chunks = []
        latent_chunks = []
        lm_transition_steps = [] if lm_transition_dump_path else None
        lm_transition_inputs = [] if lm_transition_dump_path else None
        lm_transition_hidden = [] if lm_transition_dump_path else None
        lm_layer_probe_records = [] if lm_layer_probe_dump_path else None
        feedback_component_steps = [] if feedback_component_dump_path else None
        feedback_component_positive = [] if feedback_component_dump_path else None
        feedback_component_negative = [] if feedback_component_dump_path else None
        feedback_component_latent = [] if feedback_component_dump_path else None
        feedback_component_acoustic = [] if feedback_component_dump_path else None
        feedback_component_semantic = [] if feedback_component_dump_path else None
        feedback_component_next = [] if feedback_component_dump_path else None
        finished = False
        token_counts = {}
        neg_cache = None
        neg_hidden = None
        neg_initialized = False
        negative_input_ids = [int(self.speech_start_id)]
        prev_step_embed = None
        acoustic_stream_cache = TokenizerStreamingCache()
        semantic_stream_cache = TokenizerStreamingCache()
        sample_indices = mx.array([0], dtype=mx.int32)
        lm_step_time = 0.0
        neg_step_time = 0.0
        diffusion_time = 0.0
        decode_time = 0.0
        semantic_time = 0.0
        control_time = 0.0
        lm_main_time = 0.0
        lm_dual_time = 0.0
        neg_init_time = 0.0
        neg_refresh_time = 0.0
        detailed_timing = timing_sync and os.getenv(
            "MLX_AUDIO_VIBEVOICE_DETAILED_TIMING", "0"
        ) == "1"
        layer_sync_timing = detailed_timing and os.getenv(
            "MLX_AUDIO_VIBEVOICE_LAYER_SYNC_TIMING", "0"
        ) == "1"
        diffusion_detail_time: dict[str, float] = {}
        lm_main_detail_time: dict[str, float] = {}
        neg_init_detail_time: dict[str, float] = {}
        neg_refresh_detail_time: dict[str, float] = {}
        if layer_sync_timing:
            lm_main_detail_time["__sync__"] = True
            neg_init_detail_time["__sync__"] = True
            neg_refresh_detail_time["__sync__"] = True
        dual_batch_reported = False
        dual_batch_fallback_reported = False
        semantic_multiframe_reported = False

        if use_dual_lm_batch:
            # Initialize negative branch with the start token so diffusion can use
            # a valid negative condition on the first control step.
            neg_cache = self.language_model.make_cache()
            init_neg_ids = mx.array([[int(self.speech_start_id)]], dtype=mx.int32)
            init_neg_emb = self.language_model.embed_tokens(init_neg_ids)
            neg_t0 = time.perf_counter()
            neg_hidden, neg_cache = self.language_model(
                inputs_embeds=init_neg_emb,
                cache=neg_cache,
                timing_info=neg_init_detail_time if detailed_timing else None,
            )
            if timing_sync:
                mx.eval(neg_hidden)
            neg_elapsed = time.perf_counter() - neg_t0
            neg_step_time += neg_elapsed
            neg_init_time += neg_elapsed
            neg_initialized = True

        while not finished and generated_steps < max_steps:
            control_t0 = time.perf_counter() if detailed_timing else None
            logits = self.lm_head(hidden[:, -1, :])
            token_selection = self._select_non_streaming_next_token(
                logits=logits,
                allow_speech_start=True,
                allow_termination=True,
                do_sample=do_sample,
                temperature=(1.0 if not do_sample else float(temperature or 0.7)),
                top_p=(None if top_p is None else float(top_p)),
                top_k=(None if top_k is None else int(top_k)),
                repetition_penalty=(
                    None if repetition_penalty is None else float(repetition_penalty)
                ),
                token_counts=token_counts,
                return_details=True,
            )
            if isinstance(token_selection, tuple):
                next_token_id, control_scores = token_selection
            else:
                next_token_id, control_scores = int(token_selection), None
            if detailed_timing:
                mx.eval(logits)
                control_time += time.perf_counter() - control_t0
            generated_steps += 1
            token_counts[next_token_id] = token_counts.get(next_token_id, 0) + 1
            if trace_handle is not None and (
                trace_limit is None or generated_steps <= trace_limit
            ):
                _write_trace_line(
                    trace_handle,
                    {
                        "event": "control_step",
                        "control_step": int(generated_steps),
                        "token_id": int(next_token_id),
                        "token_kind": _control_token_kind(
                            int(next_token_id),
                            speech_start_id=self.speech_start_id,
                            speech_end_id=self.speech_end_id,
                            speech_diffusion_id=self.speech_diffusion_id,
                            eos_id=self.eos_id,
                            bos_id=self.bos_id,
                        ),
                        "control_scores": control_scores,
                    },
                )

            if next_token_id == self.speech_diffusion_id:
                positive_condition = hidden[:, -1, :]
                if use_negative_cfg and refresh_negative:
                    # Upstream-like negative branch for batch=1:
                    # advance one step only when current token is diffusion.
                    #
                    # The upstream non-diffusion cache correction is needed for
                    # batched generation because the negative branch is advanced
                    # for every active sample whenever any sample emits a
                    # diffusion token. This MLX path is strictly single-sample,
                    # so non-diffusion steps do not advance the negative branch
                    # and therefore do not require the same correction here.
                    neg_last_token = int(negative_input_ids[-1])
                    if prev_step_embed is None:
                        neg_ids = mx.array([[neg_last_token]], dtype=mx.int32)
                        neg_step_embed = self.language_model.embed_tokens(neg_ids)
                    else:
                        neg_step_embed = prev_step_embed
                    if not neg_initialized or neg_cache is None:
                        neg_cache = self.language_model.make_cache()
                    neg_t0 = time.perf_counter()
                    neg_hidden, neg_cache = self.language_model(
                        inputs_embeds=neg_step_embed,
                        cache=neg_cache,
                        timing_info=neg_refresh_detail_time if detailed_timing else None,
                    )
                    if timing_sync:
                        mx.eval(neg_hidden)
                    neg_elapsed = time.perf_counter() - neg_t0
                    neg_step_time += neg_elapsed
                    neg_refresh_time += neg_elapsed
                    neg_initialized = True
                    negative_condition = neg_hidden[:, -1, :]
                    negative_input_ids.append(int(self.speech_diffusion_id))
                elif use_negative_cfg:
                    negative_condition = (
                        neg_hidden[:, -1, :]
                        if neg_hidden is not None
                        else mx.zeros_like(positive_condition)
                    )
                else:
                    negative_condition = positive_condition
                diff_t0 = time.perf_counter()
                noise_override = None
                if diffusion_noise_sequence is not None:
                    if diffusion_steps >= len(diffusion_noise_sequence):
                        raise ValueError(
                            "diffusion noise sequence is shorter than generated diffusion steps"
                        )
                    noise_override = diffusion_noise_sequence[diffusion_steps]
                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                    ddpm_steps=ddpm_steps,
                    diffusion_fp32=diffusion_fp32,
                    cfg_active_ratio=cfg_active_ratio,
                    noise_override=noise_override,
                    timing_info=diffusion_detail_time if detailed_timing else None,
                )
                if timing_sync:
                    mx.eval(speech_latent)
                speech_latent = mx.expand_dims(speech_latent, axis=1)
                diffusion_time += time.perf_counter() - diff_t0
                diffusion_steps += 1
                if redecode_final_audio:
                    latent_chunks.append(speech_latent)

                acoustic_embed = self.acoustic_connector(speech_latent)
                if use_semantic_feedback:
                    # Decode current latent for semantic feedback into the next step.
                    dec_t0 = time.perf_counter()
                    scaled_latents = (
                        speech_latent
                        / self.speech_scaling_factor
                        - self.speech_bias_factor
                    )
                    decoded_chunk = self.acoustic_tokenizer.decode(
                        scaled_latents,
                        cache=(
                            acoustic_stream_cache
                            if feedback_acoustic_use_cache
                            else None
                        ),
                        sample_indices=(
                            sample_indices if feedback_acoustic_use_cache else None
                        ),
                        use_cache=feedback_acoustic_use_cache,
                    )
                    if timing_sync:
                        mx.eval(decoded_chunk)
                    decoded_wave = decoded_chunk[0, 0, :]
                    audio_chunks.append(decoded_wave)
                    decode_time += time.perf_counter() - dec_t0

                    sem_t0 = time.perf_counter()
                    semantic_features = self.semantic_tokenizer.encode(
                        decoded_chunk,
                        cache=(
                            semantic_stream_cache
                            if feedback_semantic_use_cache
                            else None
                        ),
                        sample_indices=(
                            sample_indices if feedback_semantic_use_cache else None
                        ),
                        use_cache=feedback_semantic_use_cache,
                    ).mean
                    semantic_features_raw = semantic_features
                    if semantic_features.shape[1] == 0:
                        semantic_features_used = semantic_features
                        semantic_embed = mx.zeros_like(acoustic_embed)
                    else:
                        semantic_features_used = semantic_features
                        if semantic_features.shape[1] > 1:
                            if verbose and not semantic_multiframe_reported:
                                print(
                                    "Semantic feedback produced multiple frames "
                                    f"({int(semantic_features.shape[1])}); "
                                    "using last frame."
                                )
                                semantic_multiframe_reported = True
                            semantic_features_used = semantic_features[:, -1:, :]
                        semantic_embed = self.semantic_connector(semantic_features_used)
                    if timing_sync:
                        mx.eval(semantic_features_raw)
                        mx.eval(semantic_features_used)
                    next_embed = acoustic_embed + semantic_embed
                    if (
                        trace_handle is not None
                        and (trace_limit is None or diffusion_steps <= trace_limit)
                    ):
                        _write_trace_line(
                            trace_handle,
                            {
                                "event": "feedback_step",
                                "control_step": int(generated_steps),
                                "diffusion_step": int(diffusion_steps),
                                "positive_condition": _tensor_stats_mlx(positive_condition),
                                "negative_condition": _tensor_stats_mlx(negative_condition),
                                "speech_latent": _tensor_stats_mlx(speech_latent),
                                "scaled_latent": _tensor_stats_mlx(scaled_latents),
                                "decoded_chunk": _tensor_stats_mlx(decoded_chunk),
                                "semantic_features_raw": _tensor_stats_mlx(semantic_features_raw),
                                "semantic_features_used": _tensor_stats_mlx(semantic_features_used),
                                "acoustic_embed": _tensor_stats_mlx(acoustic_embed),
                                "semantic_embed": _tensor_stats_mlx(semantic_embed),
                                "next_embed": _tensor_stats_mlx(next_embed),
                            },
                        )
                    if feedback_component_steps is not None:
                        feedback_component_steps.append(int(diffusion_steps))
                        feedback_component_positive.append(
                            np.array(positive_condition[0, :].astype(mx.float32))
                        )
                        feedback_component_negative.append(
                            np.array(negative_condition[0, :].astype(mx.float32))
                        )
                        feedback_component_latent.append(
                            np.array(speech_latent[0, 0, :].astype(mx.float32))
                        )
                        feedback_component_acoustic.append(
                            np.array(acoustic_embed[0, 0, :].astype(mx.float32))
                        )
                        feedback_component_semantic.append(
                            np.array(semantic_embed[0, 0, :].astype(mx.float32))
                        )
                        feedback_component_next.append(
                            np.array(next_embed[0, 0, :].astype(mx.float32))
                        )
                    semantic_time += time.perf_counter() - sem_t0
                else:
                    latent_chunks.append(speech_latent)
                    next_embed = acoustic_embed
            else:
                next_ids = mx.array([[next_token_id]], dtype=mx.int32)
                next_embed = self.language_model.embed_tokens(next_ids)
                if (
                    next_token_id == self.speech_start_id
                    and use_negative_cfg
                    and refresh_negative
                ):
                    # Align with upstream speech-begin handling for negative branch.
                    neg_cache = self._reset_negative_cache_on_speech_start(neg_cache)
                    neg_initialized = bool(neg_cache is not None)
                    if len(negative_input_ids) > 0:
                        negative_input_ids[-1] = int(self.speech_start_id)
                    else:
                        negative_input_ids = [int(self.speech_start_id)]
                if next_token_id == self.speech_end_id:
                    # Match upstream non-streaming inference:
                    # speech_end resets tokenizer streaming caches but does not
                    # terminate generation. Termination is driven by EOS/max steps.
                    if use_semantic_feedback:
                        acoustic_stream_cache.set_to_zero(sample_indices)
                        semantic_stream_cache.set_to_zero(sample_indices)
                if next_token_id == self.eos_id:
                    finished = True

            if use_dual_lm_batch:
                neg_next_ids = mx.array([[int(next_token_id)]], dtype=mx.int32)
                neg_next_emb = self.language_model.embed_tokens(neg_next_ids)
                lm_t0 = time.perf_counter()
                hidden, cache, neg_hidden, neg_cache, used_dual = self._dual_language_step(
                    pos_embed=next_embed,
                    pos_cache=cache,
                    neg_embed=neg_next_emb,
                    neg_cache=neg_cache,
                )
                prev_step_embed = next_embed
                if timing_sync:
                    mx.eval(hidden)
                    mx.eval(neg_hidden)
                dual_elapsed = time.perf_counter() - lm_t0
                lm_step_time += dual_elapsed
                lm_dual_time += dual_elapsed
                neg_initialized = True
                negative_input_ids.append(int(next_token_id))
                if verbose and used_dual and not dual_batch_reported:
                    print("Dual LM batching active.")
                    dual_batch_reported = True
                if verbose and (not used_dual) and not dual_batch_fallback_reported:
                    print("Dual LM batching unavailable for current step, using fallback.")
                    dual_batch_fallback_reported = True
            else:
                lm_t0 = time.perf_counter()
                probe_step_active = (
                    lm_layer_probe_records is not None
                    and layer_probe_steps is not None
                    and int(generated_steps) in layer_probe_steps
                )
                if recompute_positive_without_cache:
                    if full_positive_embeds is None:
                        full_positive_embeds = next_embed
                    else:
                        full_positive_embeds = mx.concatenate(
                            [full_positive_embeds, next_embed],
                            axis=1,
                        )
                    lm_out = self.language_model(
                        inputs_embeds=full_positive_embeds,
                        cache=None,
                        return_layer_last_hidden=probe_step_active,
                        timing_info=lm_main_detail_time if detailed_timing else None,
                    )
                    if probe_step_active:
                        hidden, _, layer_last_hidden = lm_out
                    else:
                        hidden, _ = lm_out
                    cache = None
                else:
                    lm_out = self.language_model(
                        inputs_embeds=next_embed,
                        cache=cache,
                        return_layer_last_hidden=probe_step_active,
                        timing_info=lm_main_detail_time if detailed_timing else None,
                    )
                    if probe_step_active:
                        hidden, cache, layer_last_hidden = lm_out
                    else:
                        hidden, cache = lm_out
                prev_step_embed = next_embed
                if timing_sync:
                    mx.eval(hidden)
                lm_elapsed = time.perf_counter() - lm_t0
                lm_step_time += lm_elapsed
                lm_main_time += lm_elapsed
                if probe_step_active:
                    lm_layer_probe_records.append(
                        {
                            "control_step": int(generated_steps),
                            "step_input_embed": np.array(
                                next_embed[0, 0, :].astype(mx.float32)
                            ),
                            "layer_last_hidden": np.stack(
                                [np.array(layer[0, :]) for layer in layer_last_hidden],
                                axis=0,
                            ).astype(np.float32),
                            "final_hidden": np.array(
                                hidden[0, -1, :].astype(mx.float32)
                            ),
                        }
                    )

            if lm_transition_steps is not None:
                lm_transition_steps.append(int(generated_steps))
                lm_transition_inputs.append(
                    np.array(next_embed[0, 0, :].astype(mx.float32))
                )
                lm_transition_hidden.append(
                    np.array(hidden[0, -1, :].astype(mx.float32))
                )

            if lm_input_chunks is not None:
                lm_input_chunks.append(next_embed.astype(mx.float32))

            if verbose and generated_steps % 20 == 0:
                print(
                    f"Generated control steps: {generated_steps}/{max_steps} "
                    f"(diffusion: {diffusion_steps})"
                )

        if use_semantic_feedback and not redecode_final_audio:
            if audio_chunks:
                final_audio = mx.concatenate(audio_chunks, axis=0)
            else:
                final_audio = mx.array([])
        else:
            if latent_chunks:
                # Faster non-streaming decode path when semantic feedback is disabled:
                # decode once at the end to avoid per-step streaming-cache overhead.
                all_latents = mx.concatenate(latent_chunks, axis=1)
                scaled_latents = (
                    all_latents / self.speech_scaling_factor - self.speech_bias_factor
                )
                dec_t0 = time.perf_counter()
                decoded_full = self.acoustic_tokenizer.decode(
                    scaled_latents,
                    cache=None,
                    sample_indices=None,
                    use_cache=False,
                )
                if timing_sync:
                    mx.eval(decoded_full)
                final_audio = decoded_full[0, 0, :]
                decode_time += time.perf_counter() - dec_t0
            else:
                final_audio = mx.array([])

        if lm_input_dump_path and lm_input_chunks is not None:
            lm_input_path = Path(lm_input_dump_path)
            lm_input_path.parent.mkdir(parents=True, exist_ok=True)
            lm_inputs = mx.concatenate(lm_input_chunks, axis=1)
            np.savez_compressed(
                lm_input_path,
                inputs_embeds=np.array(lm_inputs, dtype=np.float32),
                prompt_length=np.array([initial_length], dtype=np.int32),
                generated_steps=np.array([generated_steps], dtype=np.int32),
                diffusion_steps=np.array([diffusion_steps], dtype=np.int32),
            )

        if lm_transition_dump_path and lm_transition_steps is not None:
            transition_path = Path(lm_transition_dump_path)
            transition_path.parent.mkdir(parents=True, exist_ok=True)
            hidden_size = int(self.language_model.config.hidden_size)
            input_array = (
                np.stack(lm_transition_inputs, axis=0).astype(np.float32)
                if lm_transition_inputs
                else np.zeros((0, hidden_size), dtype=np.float32)
            )
            hidden_array = (
                np.stack(lm_transition_hidden, axis=0).astype(np.float32)
                if lm_transition_hidden
                else np.zeros((0, hidden_size), dtype=np.float32)
            )
            np.savez_compressed(
                transition_path,
                control_steps=np.array(lm_transition_steps, dtype=np.int32),
                step_input_embed=input_array,
                step_hidden_output=hidden_array,
            )

        if lm_layer_probe_dump_path and lm_layer_probe_records is not None:
            probe_path = Path(lm_layer_probe_dump_path)
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            hidden_size = int(self.language_model.config.hidden_size)
            num_layers = int(self.language_model.config.num_hidden_layers)
            if lm_layer_probe_records:
                control_steps = np.array(
                    [record["control_step"] for record in lm_layer_probe_records],
                    dtype=np.int32,
                )
                input_array = np.stack(
                    [record["step_input_embed"] for record in lm_layer_probe_records],
                    axis=0,
                ).astype(np.float32)
                layer_array = np.stack(
                    [record["layer_last_hidden"] for record in lm_layer_probe_records],
                    axis=0,
                ).astype(np.float32)
                final_array = np.stack(
                    [record["final_hidden"] for record in lm_layer_probe_records],
                    axis=0,
                ).astype(np.float32)
            else:
                control_steps = np.zeros((0,), dtype=np.int32)
                input_array = np.zeros((0, hidden_size), dtype=np.float32)
                layer_array = np.zeros((0, num_layers, hidden_size), dtype=np.float32)
                final_array = np.zeros((0, hidden_size), dtype=np.float32)
            np.savez_compressed(
                probe_path,
                control_steps=control_steps,
                step_input_embed=input_array,
                layer_last_hidden=layer_array,
                final_hidden=final_array,
            )

        if feedback_component_dump_path and feedback_component_steps is not None:
            feedback_path = Path(feedback_component_dump_path)
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            hidden_size = int(self.language_model.config.hidden_size)
            acoustic_array = (
                np.stack(feedback_component_acoustic, axis=0).astype(np.float32)
                if feedback_component_acoustic
                else np.zeros((0, hidden_size), dtype=np.float32)
            )
            positive_array = (
                np.stack(feedback_component_positive, axis=0).astype(np.float32)
                if feedback_component_positive
                else np.zeros((0, hidden_size), dtype=np.float32)
            )
            negative_array = (
                np.stack(feedback_component_negative, axis=0).astype(np.float32)
                if feedback_component_negative
                else np.zeros((0, hidden_size), dtype=np.float32)
            )
            latent_dim = int(self.config.acoustic_vae_dim)
            latent_array = (
                np.stack(feedback_component_latent, axis=0).astype(np.float32)
                if feedback_component_latent
                else np.zeros((0, latent_dim), dtype=np.float32)
            )
            semantic_array = (
                np.stack(feedback_component_semantic, axis=0).astype(np.float32)
                if feedback_component_semantic
                else np.zeros((0, hidden_size), dtype=np.float32)
            )
            next_array = (
                np.stack(feedback_component_next, axis=0).astype(np.float32)
                if feedback_component_next
                else np.zeros((0, hidden_size), dtype=np.float32)
            )
            np.savez_compressed(
                feedback_path,
                diffusion_steps=np.array(feedback_component_steps, dtype=np.int32),
                positive_condition=positive_array,
                negative_condition=negative_array,
                speech_latent=latent_array,
                acoustic_embed=acoustic_array,
                semantic_embed=semantic_array,
                next_embed=next_array,
            )

        if timing_sync:
            mx.eval(final_audio)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        samples = final_audio.shape[0] if final_audio.size > 0 else 0
        audio_duration_seconds = samples / self.sample_rate if samples > 0 else 0.0

        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{int(audio_duration_seconds // 3600):02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"
        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0.0
        if verbose and os.getenv("MLX_AUDIO_VIBEVOICE_TIMING", "0") == "1":
            accounted = (
                prefill_time
                + lm_step_time
                + neg_step_time
                + diffusion_time
                + decode_time
                + semantic_time
            )
            other_time = max(0.0, elapsed_time - accounted)
            print("Timing breakdown:")
            print(
                f"  prefill={prefill_time:.2f}s, lm_step={lm_step_time:.2f}s, "
                f"neg_step={neg_step_time:.2f}s"
            )
            print(
                f"  diffusion={diffusion_time:.2f}s, decode={decode_time:.2f}s, "
                f"semantic={semantic_time:.2f}s, other={other_time:.2f}s"
            )
            if detailed_timing:
                diff_pred = diffusion_detail_time.get("diffusion_pred_head", 0.0)
                diff_sched = diffusion_detail_time.get("diffusion_scheduler", 0.0)
                print("Detailed timing:")
                print(
                    f"  control={control_time:.2f}s, lm_main={lm_main_time:.2f}s, "
                    f"lm_dual={lm_dual_time:.2f}s"
                )
                print(
                    f"  neg_init={neg_init_time:.2f}s, neg_refresh={neg_refresh_time:.2f}s"
                )
                print(
                    f"  diffusion_pred_head={diff_pred:.2f}s, "
                    f"diffusion_scheduler={diff_sched:.2f}s"
                )
                print(
                    f"  pred_noisy_proj={diffusion_detail_time.get('noisy_proj', 0.0):.2f}s, "
                    f"pred_adaln={diffusion_detail_time.get('adaln', 0.0):.2f}s, "
                    f"pred_final_adaln={diffusion_detail_time.get('final_adaln', 0.0):.2f}s"
                )
                print(
                    f"  pred_ffn_gate_up={diffusion_detail_time.get('ffn_gate_up', 0.0):.2f}s, "
                    f"pred_ffn_act={diffusion_detail_time.get('ffn_act', 0.0):.2f}s, "
                    f"pred_ffn_down={diffusion_detail_time.get('ffn_down', 0.0):.2f}s, "
                    f"pred_final_linear={diffusion_detail_time.get('final_linear', 0.0):.2f}s"
                )
                print(
                    f"  lm_main_attn={lm_main_detail_time.get('attn', 0.0):.2f}s, "
                    f"lm_main_mlp={lm_main_detail_time.get('mlp', 0.0):.2f}s"
                )
                print(
                    f"  neg_init_attn={neg_init_detail_time.get('attn', 0.0):.2f}s, "
                    f"neg_init_mlp={neg_init_detail_time.get('mlp', 0.0):.2f}s"
                )
                print(
                    f"  neg_refresh_attn={neg_refresh_detail_time.get('attn', 0.0):.2f}s, "
                    f"neg_refresh_mlp={neg_refresh_detail_time.get('mlp', 0.0):.2f}s"
                )

        if trace_handle is not None:
            trace_handle.close()

        yield GenerationResult(
            audio=final_audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=len(prompt_tokens) + generated_steps,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": len(prompt_tokens),
                "tokens-per-sec": (
                    round(len(prompt_tokens) / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=float(mx.get_peak_memory() / 1e9),
        )

    def _generate_multi_speaker(
        self,
        dialogue: List[Tuple[str, str]],
        max_tokens: int = 512,
        cfg_scale: float = 1.5,
        ddpm_steps: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from multiple speakers.

        Args:
            dialogue: List of (voice_name, text) tuples for each speaker segment
            max_tokens: Maximum tokens per segment
            cfg_scale: Classifier-free guidance scale
            ddpm_steps: Override diffusion inference steps
            verbose: Whether to show progress

        Yields:
            GenerationResult containing combined audio from all speakers
        """
        start_time = time.perf_counter()
        all_audio_segments = []
        total_tokens = 0

        for segment_idx, (voice_name, segment_text) in enumerate(dialogue):
            if verbose:
                print(
                    f"Generating segment {segment_idx + 1}/{len(dialogue)}: {voice_name}"
                )

            # Load the voice for this segment
            self.load_voice(voice_name)

            # Generate audio for this segment (single speaker path)
            for result in self._generate_single_speaker(
                text=segment_text,
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                ddpm_steps=ddpm_steps,
                verbose=verbose,
                **kwargs,
            ):
                all_audio_segments.append(result.audio)
                total_tokens += result.token_count

        # Combine all audio segments
        if all_audio_segments:
            final_audio = mx.concatenate(all_audio_segments)
        else:
            final_audio = mx.array([])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Calculate statistics
        samples = final_audio.shape[0] if final_audio.size > 0 else 0
        audio_duration_seconds = samples / self.sample_rate if samples > 0 else 0

        # Format duration
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{int(audio_duration_seconds // 3600):02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0

        yield GenerationResult(
            audio=final_audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=total_tokens,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": total_tokens,
                "tokens-per-sec": (
                    round(total_tokens / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def _generate_single_speaker(
        self,
        text: str,
        max_tokens: int = 512,
        cfg_scale: float = 1.5,
        ddpm_steps: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech for a single speaker segment (internal method).

        This contains the core generation logic, used by both generate() and
        _generate_multi_speaker().
        """
        start_time = time.perf_counter()

        # Tokenize input
        text_token_ids = self.tokenizer.encode(
            text.strip() + "\n", add_special_tokens=False
        )
        input_ids = mx.array([text_token_ids], dtype=mx.int32)

        batch_size = 1
        seq_len = input_ids.shape[1]

        # Use voice cache if available
        use_voice_cache = hasattr(self, "_voice_lm_cache") and hasattr(
            self, "_voice_tts_cache"
        )

        if use_voice_cache:
            lm_cache = self._voice_lm_cache
            tts_cache = self._voice_tts_cache
            tts_hidden = self._voice_tts_hidden
            neg_hidden = self._voice_neg_tts_hidden
            neg_cache = self._voice_neg_tts_cache
        else:
            lm_cache = None
            tts_cache = None
            tts_hidden = None
            neg_hidden = None
            neg_cache = None

        speech_latents = []
        finished = False
        step = 0
        text_pos = 0
        use_negative_cfg = float(cfg_scale) > 1.0 + 1e-6
        requested_cfg_active_ratio = float(kwargs.pop("cfg_active_ratio", 1.0))
        cfg_active_ratio = 1.0
        if verbose and abs(requested_cfg_active_ratio - 1.0) > 1e-8:
            print("Note: cfg_active_ratio is disabled; using full CFG for all diffusion steps.")

        while not finished and step < max_tokens:
            if text_pos < seq_len:
                cur_text_ids = input_ids[
                    :, text_pos : min(seq_len, text_pos + TTS_TEXT_WINDOW_SIZE)
                ]
                cur_window = cur_text_ids.shape[1]
                text_pos += cur_window

                text_embeds = self.language_model.embed_tokens(cur_text_ids)
                lm_out, lm_cache = self.language_model(
                    inputs_embeds=text_embeds, cache=lm_cache
                )

                text_type = mx.ones((batch_size, cur_window), dtype=mx.int32)
                type_embed = self.tts_input_types(text_type)
                tts_in = lm_out + type_embed
                tts_out, tts_cache = self.tts_language_model(
                    inputs_embeds=tts_in, cache=tts_cache
                )

                if tts_hidden is None:
                    tts_hidden = tts_out
                else:
                    tts_hidden = mx.concatenate([tts_hidden, tts_out], axis=1)

                if use_negative_cfg and (neg_hidden is None or not use_voice_cache):
                    neg_embed = mx.zeros(
                        (batch_size, cur_window, self.config.decoder_config.hidden_size)
                    )
                    neg_type_embed = self.tts_input_types(
                        mx.ones((batch_size, cur_window), dtype=mx.int32)
                    )
                    neg_in = neg_embed + neg_type_embed
                    neg_out, neg_cache = self.tts_language_model(
                        inputs_embeds=neg_in, cache=neg_cache
                    )
                    if neg_hidden is None:
                        neg_hidden = neg_out
                    else:
                        neg_hidden = mx.concatenate([neg_hidden, neg_out], axis=1)

            if tts_hidden is None or (use_negative_cfg and neg_hidden is None):
                break

            for _ in range(TTS_SPEECH_WINDOW_SIZE):
                positive_condition = tts_hidden[:, -1, :]
                if use_negative_cfg and neg_hidden is not None:
                    negative_condition = neg_hidden[:, -1, :]
                else:
                    negative_condition = positive_condition

                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                    ddpm_steps=ddpm_steps,
                    cfg_active_ratio=cfg_active_ratio,
                )
                speech_latent = mx.expand_dims(speech_latent, 1)

                speech_latents.append(speech_latent)

                acoustic_embed = self.acoustic_connector(speech_latent)

                type_embed = self.tts_input_types(
                    mx.zeros((batch_size, 1), dtype=mx.int32)
                )
                tts_input = acoustic_embed + type_embed

                tts_out, tts_cache = self.tts_language_model(
                    inputs_embeds=tts_input,
                    cache=tts_cache,
                )
                tts_hidden = mx.concatenate([tts_hidden, tts_out], axis=1)

                if use_negative_cfg:
                    neg_type_embed = self.tts_input_types(
                        mx.zeros((batch_size, 1), dtype=mx.int32)
                    )
                    neg_input = acoustic_embed + neg_type_embed
                    neg_out, neg_cache = self.tts_language_model(
                        inputs_embeds=neg_input,
                        cache=neg_cache,
                    )
                    neg_hidden = mx.concatenate([neg_hidden, neg_out], axis=1)

                eos_logits = mx.sigmoid(self.tts_eos_classifier(tts_out[:, -1, :]))
                if eos_logits[0].item() > 0.5:
                    finished = True
                    break

                step += 1
                if step >= max_tokens:
                    finished = True
                    break

        if speech_latents:
            speech_latent_seq = mx.concatenate(speech_latents, axis=1)
            scaled_latents = (
                speech_latent_seq / self.speech_scaling_factor - self.speech_bias_factor
            )
            audio = self.acoustic_tokenizer.decode(scaled_latents)
            final_audio = audio.squeeze(1).squeeze(0)
        else:
            final_audio = mx.array([])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        samples = final_audio.shape[0] if final_audio.size > 0 else 0
        audio_duration_seconds = samples / self.sample_rate if samples > 0 else 0

        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{int(audio_duration_seconds // 3600):02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0

        yield GenerationResult(
            audio=final_audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=len(input_ids[0]),
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": len(input_ids[0]),
                "tokens-per-sec": (
                    round(len(input_ids[0]) / elapsed_time, 2)
                    if elapsed_time > 0
                    else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )
