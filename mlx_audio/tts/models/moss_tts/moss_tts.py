from __future__ import annotations

import fnmatch
import os
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.sample_utils import apply_top_k, apply_top_p

from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.utils import load_audio

from .config import ModelConfig
from .processor import MossTTSProcessor
from .qwen3 import Qwen3Model


def _is_quantized_module(module: Any) -> bool:
    if module is None:
        return False
    return (
        hasattr(module, "bits")
        and hasattr(module, "group_size")
        and hasattr(module, "weight")
        and hasattr(module, "scales")
    )


def _is_q8_module(module: Any) -> bool:
    if not _is_quantized_module(module):
        return False
    try:
        return (
            int(getattr(module, "bits")) == 8
            and str(getattr(module, "mode", "affine")) == "affine"
        )
    except Exception:
        return False


def _dequantize_packed_weight_module(module: Any, *, path: str) -> Any:
    if not _is_quantized_module(module):
        return module

    missing: List[str] = []
    for attr in ("weight", "scales", "group_size", "bits"):
        if not hasattr(module, attr):
            missing.append(attr)
    if missing:
        raise RuntimeError(
            f"MOSS-TTS mixed-precision rescue: cannot dequantize {path}; "
            f"missing attributes: {', '.join(missing)}"
        )

    biases = getattr(module, "biases", None)
    if biases is None and hasattr(module, "bias"):
        biases = getattr(module, "bias")

    use_bias = bool(isinstance(biases, mx.array) and biases.ndim == 1)

    try:
        w = mx.dequantize(
            module.weight,
            module.scales,
            biases,
            int(module.group_size),
            int(module.bits),
            mode=str(getattr(module, "mode", "affine")),
        )
    except Exception as exc:
        raise RuntimeError(
            "MOSS-TTS mixed-precision rescue: mx.dequantize failed for "
            f"{path} ({type(module).__name__}): {type(exc).__name__}: {exc}"
        ) from exc

    if (
        isinstance(module, nn.Embedding)
        or path.endswith("embed_tokens")
        or ("embed" in path and w.ndim == 2)
    ):
        out = nn.Embedding(int(w.shape[0]), int(w.shape[1]))
        out.weight = w
        return out

    if isinstance(module, nn.Linear) or w.ndim == 2:
        out_features, in_features = int(w.shape[0]), int(w.shape[1])
        out = nn.Linear(in_features, out_features, bias=use_bias)
        out.weight = w
        if use_bias and isinstance(biases, mx.array):
            out.bias = biases
        return out

    raise RuntimeError(
        "MOSS-TTS mixed-precision rescue: unsupported module type for dequantize "
        f"at {path}: {type(module).__name__}"
    )


# pyright: reportMissingImports=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false


def _apply_delay_pattern_impl(codes: mx.array, pad_code: int) -> mx.array:
    if codes.ndim != 2:
        raise ValueError("codes must have shape (T, NQ)")
    t, nq = int(codes.shape[0]), int(codes.shape[1])
    if nq < 1:
        raise ValueError("NQ must be >= 1")

    delayed = mx.full((t + nq - 1, nq), int(pad_code), dtype=codes.dtype)
    for i in range(nq):
        delayed[i : i + t, i] = codes[:, i]
    return delayed


def _apply_de_delay_pattern_impl(delay_codes: mx.array) -> mx.array:
    if delay_codes.ndim != 2:
        raise ValueError("delay_codes must have shape (T_delay, NQ)")
    t_delay, nq = int(delay_codes.shape[0]), int(delay_codes.shape[1])
    t = t_delay - nq + 1
    if t <= 0:
        return mx.zeros((0, nq), dtype=delay_codes.dtype)

    tokens = mx.full((t, nq), 0, dtype=delay_codes.dtype)
    for i in range(nq):
        tokens[:, i] = delay_codes[i : i + t, i]
    return tokens


def apply_delay_pattern(codes: mx.array, pad_code: int) -> mx.array:
    return _apply_delay_pattern_impl(codes, pad_code)


def apply_de_delay_pattern(delay_codes: mx.array) -> mx.array:
    return _apply_de_delay_pattern_impl(delay_codes)


def _suppress_token_ids(logits: mx.array, token_ids: List[int]) -> mx.array:
    if len(token_ids) == 0:
        return logits
    ids = mx.array(token_ids, dtype=mx.int32)[None, :]
    return mx.put_along_axis(
        logits,
        ids,
        mx.array(float("-inf"), dtype=logits.dtype),
        axis=-1,
    )


def _apply_repetition_penalty(
    logits: mx.array, prev_tokens: mx.array, penalty: float
) -> mx.array:
    if penalty == 1.0:
        return logits
    if prev_tokens.size == 0:
        return logits

    vocab_size = int(logits.shape[-1])
    prev_np = np.array(prev_tokens).reshape(-1)
    if prev_np.size == 0:
        return logits

    unique_np = np.unique(prev_np)
    unique_np = unique_np[(unique_np >= 0) & (unique_np < vocab_size)]
    if unique_np.size == 0:
        return logits

    unique = mx.array(unique_np.astype(np.int32))

    selected = mx.take(logits, unique, axis=-1)
    penalized = mx.where(selected > 0, selected / penalty, selected * penalty)
    return mx.put_along_axis(logits, unique[None, :], penalized, axis=-1)


def sample_token(
    logits: mx.array,
    top_p: float = 1.0,
    top_k: int = 0,
    do_sample: bool = True,
    prev_tokens: Optional[mx.array] = None,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
) -> mx.array:
    if logits.ndim < 1:
        raise ValueError("logits must have at least 1 dimension")

    vocab_size = int(logits.shape[-1])
    if not do_sample or temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)

    if prev_tokens is not None and repetition_penalty != 1.0:
        if logits.ndim == 2:
            logits = _apply_repetition_penalty(logits, prev_tokens, repetition_penalty)
        elif logits.ndim == 3:
            b, h, _ = logits.shape
            out = []
            for head_idx in range(int(h)):
                prev_h = prev_tokens[..., head_idx]
                out.append(
                    _apply_repetition_penalty(
                        logits[:, head_idx, :], prev_h, repetition_penalty
                    )
                )
            logits = mx.stack(out, axis=1)
        else:
            flat = logits.reshape(-1, vocab_size)
            logits = _apply_repetition_penalty(
                flat, prev_tokens, repetition_penalty
            ).reshape(logits.shape)

    scaled = logits
    if temperature != 1.0:
        scaled = scaled * (1.0 / float(temperature))

    original_shape = scaled.shape
    flat = scaled.reshape(-1, vocab_size)
    flat = nn.log_softmax(flat, axis=-1)

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        flat = apply_top_p(flat, float(top_p))
    if top_k is not None and int(top_k) > 0 and int(top_k) < vocab_size:
        flat = apply_top_k(flat, int(top_k))
        # flat = nn.log_softmax(flat, axis=-1)

    tokens = mx.random.categorical(flat).astype(mx.int32)
    return tokens.reshape(original_shape[:-1]).astype(mx.int32)


def _format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _normalize_reference_audio_for_codec(wav: mx.array) -> mx.array:
    wav_np = np.array(wav, dtype=np.float32)
    if wav_np.ndim == 0:
        raise ValueError("reference audio must be at least 1D")

    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=tuple(range(wav_np.ndim - 1)))

    wav_np = wav_np.reshape(-1)
    if wav_np.size == 0:
        return mx.zeros((0,), dtype=mx.float32)

    current_dbfs = 10.0 * np.log10(np.mean(wav_np.astype(np.float64) ** 2) + 1e-9)
    gain = float(-20.0 - current_dbfs)
    gain = max(-3.0, min(gain, 3.0))
    factor = 10.0 ** (gain / 20.0)
    return mx.array(wav_np * factor, dtype=mx.float32)


def _encode_reference_audio_with_codec(
    *,
    codec: Any,
    ref_audio: Any,
    sample_rate: int,
    n_vq: int,
) -> List[mx.array]:
    if codec is None:
        raise ValueError("MOSS-TTS codec not loaded")
    if not bool(getattr(codec, "has_encoder", False)):
        raise ValueError(
            "MOSS-TTS voice cloning requires MLX codec encoder weights; use a codec_path with encoder params"
        )

    wav = load_audio(ref_audio, sample_rate=sample_rate)
    wav = _normalize_reference_audio_for_codec(wav)

    encoded = codec.encode(wav[None, None, :], num_quantizers=int(n_vq))
    audio_codes = encoded["audio_codes"]
    audio_codes_lengths = encoded["audio_codes_lengths"]

    codes_list: List[mx.array] = []
    for i in range(int(audio_codes.shape[1])):
        length_i = int(audio_codes_lengths[i].item())
        codes_i = (
            audio_codes[: int(n_vq), i, :length_i].transpose(1, 0).astype(mx.int32)
        )
        codes_list.append(codes_i)

    return codes_list


def _get_generated_audio_history(
    generation_ids: mx.array, *, generated_start: int, vq_idx: int
) -> mx.array:
    if generation_ids.ndim != 3:
        raise ValueError("generation_ids must have shape (B, T, 1+n_vq)")
    start = max(0, min(int(generated_start), int(generation_ids.shape[1])))
    return generation_ids[:, start:, vq_idx + 1]


def _path_matches_rescue_pattern(path: str, pattern: str) -> bool:
    pat = str(pattern).strip()
    if not pat:
        return False
    return fnmatch.fnmatchcase(path, pat)


def _get_mixed_precision_rescue_patterns(config: Any) -> List[str]:
    patterns = ["language_model.embed_tokens", "lm_heads.*"]

    config_patterns = getattr(config, "mixed_precision_rescue_patterns", None)
    if isinstance(config_patterns, (list, tuple)):
        for item in config_patterns:
            pat = str(item).strip()
            if pat:
                patterns.append(pat)

    env_value = os.environ.get("MOSS_TTS_MIXED_PRECISION_RESCUE", "")
    for item in env_value.split(","):
        pat = item.strip()
        if pat:
            patterns.append(pat)

    deduped: List[str] = []
    seen = set()
    for pat in patterns:
        if pat in seen:
            continue
        seen.add(pat)
        deduped.append(pat)
    return deduped


def _get_module_by_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def _set_module_by_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    current = root
    for part in parts[:-1]:
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    last = parts[-1]
    if last.isdigit():
        current[int(last)] = value
    else:
        setattr(current, last, value)


def _apply_mixed_precision_rescue(model: Any) -> None:
    patterns = _get_mixed_precision_rescue_patterns(getattr(model, "config", None))
    rescue_paths: List[str] = []
    for path, module in model.named_modules():
        if not path:
            continue
        if not _is_q8_module(module):
            continue
        if any(_path_matches_rescue_pattern(path, pat) for pat in patterns):
            rescue_paths.append(path)

    for path in rescue_paths:
        module = _get_module_by_path(model, path)
        replacement = _dequantize_packed_weight_module(module, path=path)
        _set_module_by_path(model, path, replacement)


def _get_experimental_quant_mode(config: Any) -> str:
    env_value = os.environ.get("MOSS_TTS_EXPERIMENTAL_QUANT_MODE", "").strip()
    if env_value:
        return env_value
    return str(getattr(config, "experimental_quant_mode", "")).strip()


def _get_experimental_quant_patterns(config: Any) -> List[str]:
    patterns: List[str] = []

    config_patterns = getattr(config, "experimental_quant_patterns", None)
    if isinstance(config_patterns, (list, tuple)):
        for item in config_patterns:
            pat = str(item).strip()
            if pat:
                patterns.append(pat)

    env_value = os.environ.get("MOSS_TTS_EXPERIMENTAL_QUANT_PATTERNS", "")
    for item in env_value.split(","):
        pat = item.strip()
        if pat:
            patterns.append(pat)

    if not patterns and _get_experimental_quant_mode(config):
        patterns.append("lm_heads.*")

    deduped: List[str] = []
    seen = set()
    for pat in patterns:
        if pat in seen:
            continue
        seen.add(pat)
        deduped.append(pat)
    return deduped


def _requantize_module_with_mode(module: Any, *, path: str, mode: str) -> Any:
    current = module
    if _is_quantized_module(current):
        current = _dequantize_packed_weight_module(current, path=path)

    to_quantized = getattr(current, "to_quantized", None)
    if not callable(to_quantized):
        raise RuntimeError(
            f"MOSS-TTS experimental quantization cannot rewrite {path}; "
            f"module type {type(current).__name__} has no to_quantized()"
        )

    quant_kwargs: Dict[str, Any] = {"mode": mode}
    if mode == "mxfp8":
        quant_kwargs.update({"group_size": 32, "bits": 8})
    return to_quantized(**quant_kwargs)


def _apply_experimental_quantization(model: Any) -> None:
    mode = _get_experimental_quant_mode(getattr(model, "config", None))
    if not mode:
        return

    patterns = _get_experimental_quant_patterns(getattr(model, "config", None))
    if not patterns:
        return

    rewrite_paths: List[str] = []
    for path, module in model.named_modules():
        if not path:
            continue
        if not any(_path_matches_rescue_pattern(path, pat) for pat in patterns):
            continue
        if _is_quantized_module(module) or callable(
            getattr(module, "to_quantized", None)
        ):
            rewrite_paths.append(path)

    for path in rewrite_paths:
        module = _get_module_by_path(model, path)
        replacement = _requantize_module_with_mode(module, path=path, mode=mode)
        _set_module_by_path(model, path, replacement)


def find_last_equal(tensor: mx.array, value: int) -> mx.array:
    if tensor.ndim != 2:
        raise ValueError("find_last_equal expects a rank-2 tensor [B, T]")

    value_arr = mx.array(value, dtype=tensor.dtype)
    mask = mx.equal(tensor, value_arr).astype(mx.int32)
    flipped = mask[:, ::-1]
    flipped_idx = mx.argmax(flipped, axis=1).astype(mx.int32)

    seq_len = int(tensor.shape[1])
    last_idx = (seq_len - 1) - flipped_idx
    batch_idx = mx.arange(int(tensor.shape[0]), dtype=mx.int32)
    actual = tensor[batch_idx, last_idx]
    no_match = actual != value_arr
    return mx.where(no_match, mx.full(last_idx.shape, -1, dtype=mx.int32), last_idx)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tokenizer: Any | None = None
        self.processor: MossTTSProcessor | None = None
        self.codec: MossAudioTokenizer | None = None

        self.language_model = Qwen3Model(self.config.language_config)

        hidden_size = int(self.config.language_config.get("hidden_size", 4096))
        vocab_size = int(self.config.language_config.get("vocab_size", 155648))
        audio_vocab_size = int(self.config.audio_vocab_size) + 1

        self.emb_ext = [
            nn.Embedding(audio_vocab_size, hidden_size) for _ in range(self.config.n_vq)
        ]

        self.lm_heads = [nn.Linear(hidden_size, vocab_size, bias=False)]
        self.lm_heads.extend(
            [
                nn.Linear(hidden_size, audio_vocab_size, bias=False)
                for _ in range(self.config.n_vq)
            ]
        )

    @staticmethod
    def sanitize(weights: Dict):
        if not weights:
            return weights

        sanitized: Dict[str, Any] = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            sanitized[k] = v
        return sanitized

    def get_input_embeddings(self, input_ids: "mx.array") -> "mx.array":
        if input_ids.ndim != 3 or input_ids.shape[-1] != self.config.n_vq + 1:
            raise ValueError(
                "input_ids must have shape (batch, seq_len, 1 + n_vq) for moss_tts_delay"
            )

        x = self.language_model.embed_tokens(input_ids[..., 0])
        for i, emb in enumerate(self.emb_ext):
            x = x + emb(input_ids[..., i + 1])
        return x

    def forward(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[List] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if use_cache is None:
            use_cache = True

        inputs_embeds = self.get_input_embeddings(input_ids)
        cache = past_key_values
        if cache is None and use_cache:
            cache = self.language_model.make_cache()

        hidden_states, cache = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache=cache,
        )

        logits: List[mx.array] = []
        for i, head in enumerate(self.lm_heads):
            cur = head(hidden_states)
            if i > 0:
                cur = mx.put_along_axis(
                    cur,
                    mx.array([cur.shape[-1] - 1], dtype=mx.int32)[None, None, :],
                    mx.array(float("-inf"), dtype=cur.dtype),
                    axis=-1,
                )
            logits.append(cur)

        return {"logits": logits, "past_key_values": cache}

    def model_quant_predicate(self, path: str, module) -> bool:
        skip_patterns = ["codec", "emb_ext", "audio_tokenizer"]
        return not any(pattern in path for pattern in skip_patterns)

    @property
    def sample_rate(self) -> int:
        return self.config.sampling_rate

    @property
    def model_type(self) -> str:
        return "moss_tts_delay"

    @classmethod
    def post_load_hook(cls, model: "Model", model_path) -> "Model":
        model_path = Path(model_path)
        try:
            from transformers import AutoTokenizer

            model.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            model.processor = MossTTSProcessor(
                tokenizer=model.tokenizer,
                config=model.config,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load HF tokenizer from: {model_path}. "
                "Ensure `tokenizer.json` / `tokenizer_config.json` are present."
            ) from exc

        codec_path = getattr(model.config, "codec_path", None)
        if codec_path is None:
            codec_path = str(model_path / "moss-audio-tokenizer-full")
            if not (model_path / "moss-audio-tokenizer-full").exists():
                codec_path = "./moss-audio-tokenizer-full"

        model.codec = MossAudioTokenizer.from_pretrained(str(codec_path))
        if model.processor is not None:
            model.processor.codec = model.codec
        if not bool(getattr(model.codec, "has_encoder", False)):
            candidates = [
                model_path / "moss-audio-tokenizer-full",
                Path("./moss-audio-tokenizer-full"),
            ]
            for c in candidates:
                try:
                    if not c.exists():
                        continue
                    alt = MossAudioTokenizer.from_pretrained(str(c))
                    if bool(getattr(alt, "has_encoder", False)):
                        model.codec = alt
                        break
                except Exception:
                    continue

        return model

    def _encode_reference_audio_to_codes(self, ref_audio: Any) -> List[mx.array]:
        n_vq = int(getattr(self.config, "n_vq", 32))
        return _encode_reference_audio_with_codec(
            codec=self.codec,
            ref_audio=ref_audio,
            sample_rate=self.sample_rate,
            n_vq=n_vq,
        )

    def _is_audio_code_matrix(self, value: Any) -> bool:
        n_vq = int(getattr(self.config, "n_vq", 32))
        if isinstance(value, mx.array):
            return value.ndim == 2 and int(value.shape[1]) == n_vq
        try:
            arr = np.asarray(value)
        except Exception:
            return False
        return arr.ndim == 2 and int(arr.shape[1]) == n_vq

    def _coerce_audio_code_matrix(self, value: Any) -> mx.array:
        if isinstance(value, mx.array):
            return value.astype(mx.int32)
        arr = np.asarray(value)
        return mx.array(arr, dtype=mx.int32)

    def _coerce_audio_reference(self, value: Any) -> Any:
        if isinstance(value, mx.array):
            return value.astype(mx.float32)
        if isinstance(value, np.ndarray):
            return mx.array(value, dtype=mx.float32)
        if isinstance(value, list):
            arr = np.asarray(value)
            if arr.dtype != object:
                return mx.array(arr, dtype=mx.float32)
        return value

    def _normalize_audio_code_entries(self, values: Any) -> List[mx.array]:
        if values is None:
            return []
        if not isinstance(values, (list, tuple)):
            values = [values]

        normalized: List[mx.array] = []
        for item in values:
            if item is None:
                continue
            if self._is_audio_code_matrix(item):
                normalized.append(self._coerce_audio_code_matrix(item))
                continue
            normalized.extend(
                self._encode_reference_audio_to_codes(
                    self._coerce_audio_reference(item)
                )
            )
        return normalized

    def _normalize_reference_entries(
        self, values: Any
    ) -> List[Optional[mx.array]] | None:
        if values is None:
            return None
        if not isinstance(values, (list, tuple)):
            values = [values]

        normalized: List[Optional[mx.array]] = []
        for item in values:
            if item is None:
                normalized.append(None)
                continue
            if self._is_audio_code_matrix(item):
                normalized.append(self._coerce_audio_code_matrix(item))
                continue
            encoded = self._normalize_audio_code_entries([item])
            normalized.extend(encoded)
        return normalized

    def _normalize_conversation_input(
        self, conversation: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        normalized_conversation: List[Dict[str, Any]] = []
        for idx, message in enumerate(conversation):
            if not isinstance(message, dict):
                raise TypeError(f"Conversation message at index {idx} must be a dict.")
            if "role" not in message:
                raise ValueError(
                    f"Conversation message at index {idx} is missing 'role'."
                )

            role = str(message["role"])
            normalized_message = dict(message)

            if role == "user":
                if "reference" in normalized_message:
                    normalized_message["reference"] = self._normalize_reference_entries(
                        normalized_message.get("reference")
                    )
                if "audio_codes_list" in normalized_message:
                    normalized_message["audio_codes_list"] = (
                        self._normalize_audio_code_entries(
                            normalized_message.get("audio_codes_list")
                        )
                    )
            elif role == "assistant":
                normalized_message["audio_codes_list"] = (
                    self._normalize_audio_code_entries(
                        normalized_message.get("audio_codes_list")
                    )
                )
            else:
                raise ValueError(f"Unsupported role in conversation: {role}")

            normalized_conversation.append(normalized_message)

        return normalized_conversation

    @staticmethod
    def apply_delay_pattern(codes: mx.array, pad_code: int) -> mx.array:
        return _apply_delay_pattern_impl(codes, pad_code)

    @staticmethod
    def apply_de_delay_pattern(delay_codes: mx.array) -> mx.array:
        return _apply_de_delay_pattern_impl(delay_codes)

    def generate(
        self,
        text: Optional[Union[str, Sequence[Dict[str, Any]]]] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        gender: Optional[str] = None,
        pitch: Optional[float] = None,
        instruct: Optional[str] = None,
        lang_code: Optional[str] = None,
        ref_audio: Optional[Any] = None,
        ref_text: Optional[str] = None,
        conversation: Optional[Sequence[Dict[str, Any]]] = None,
        mode: str = "generation",
        temperature: float = 1.7,
        top_p: float = 0.8,
        top_k: int = 25,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        repetition_penalty: float = 1.0,
        stream: bool = False,
        streaming_interval: Optional[int] = None,
        max_tokens: int = 4096,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        _ = (voice, speed, gender, pitch, stream, streaming_interval, kwargs)

        conversation_input = conversation
        text_input = text
        if (
            conversation_input is None
            and isinstance(text, Sequence)
            and not isinstance(text, (str, bytes))
        ):
            conversation_input = list(text)
            text_input = None

        if conversation_input is not None and text_input is not None:
            raise ValueError("Provide either `text` or `conversation`, not both.")
        if conversation_input is None and mode != "generation":
            raise ValueError(
                "`mode` is only supported when `conversation` is provided."
            )
        if conversation_input is not None and ref_audio is not None:
            raise ValueError(
                "`ref_audio` is not supported with conversation input; attach audio context inside the conversation messages instead."
            )
        if conversation_input is not None and ref_text is not None:
            raise ValueError(
                "`ref_text` is not supported with conversation input; include the prefix transcript in the user message text instead."
            )

        if self.processor is None:
            audio = mx.zeros((1,), dtype=mx.float32)
            mx.eval(audio)
            result = GenerationResult(
                audio=audio,
                samples=int(audio.shape[0]),
                sample_rate=self.sample_rate,
                segment_idx=0,
                token_count=0,
                audio_duration=_format_duration(
                    float(audio.shape[0]) / self.sample_rate
                ),
                real_time_factor=0.0,
                prompt={
                    "warning": "tokenizer/processor not loaded; returning silence",
                    "text": text_input,
                    "mode": mode,
                },
                audio_samples=int(audio.shape[0]),
                processing_time_seconds=0.0,
                peak_memory_usage=float(mx.get_peak_memory() / 1e9),
                is_streaming_chunk=False,
                is_final_chunk=True,
            )
            yield result
            return

        start_time = time.time()
        if conversation_input is not None:
            normalized_conversation = self._normalize_conversation_input(
                conversation_input
            )
            if mode == "continuation":
                last_message = (
                    normalized_conversation[-1]
                    if len(normalized_conversation) > 0
                    else None
                )
                if (
                    not isinstance(last_message, dict)
                    or last_message.get("role") != "assistant"
                ):
                    raise ValueError(
                        "continuation mode requires the last conversation message to be an assistant message"
                    )
                assistant_audio = last_message.get("audio_codes_list")
                if not isinstance(assistant_audio, list) or len(assistant_audio) == 0:
                    raise ValueError(
                        "continuation mode requires assistant prefix audio in `audio_codes_list`"
                    )
            input_ids = self.processor.prepare_conversation_input(
                normalized_conversation,
                mode=mode,
            )
        else:
            if text_input is None:
                raise ValueError(
                    "`text` must be provided when `conversation` is not used."
                )
            reference_audio_codes: Optional[Sequence[mx.array]] = None
            if ref_audio is not None:
                reference_audio_codes = self._encode_reference_audio_to_codes(ref_audio)
            input_ids = self.processor.prepare_generation_input(
                text_input,
                reference_audio_codes=reference_audio_codes,
                instruction=instruct,
                language=lang_code,
            )
        attention_mask = mx.ones(
            (int(input_ids.shape[0]), int(input_ids.shape[1])), dtype=mx.bool_
        )

        batch_size = int(input_ids.shape[0])
        seq_len = int(input_ids.shape[1])
        n_vq = int(input_ids.shape[2]) - 1

        if verbose:
            ids_np = np.array(input_ids[0], dtype=np.int64)
            print(
                f"[MOSS-TTS] input_ids shape: {input_ids.shape}  "
                f"(batch={batch_size}, seq_len={seq_len}, channels=1+{n_vq})"
            )
            for ch in range(1 + n_vq):
                ch_ids = ids_np[:, ch].tolist()
                label = "text" if ch == 0 else f"vq{ch - 1:02d}"
                formatted = " ".join(f"{v:6d}" for v in ch_ids)
                print(f"[MOSS-TTS] ch[{label}] ({len(ch_ids)}): {formatted}")
            if self.tokenizer:
                decoded_text = self.tokenizer.decode(ids_np[:, 0].tolist())
                print(f"[MOSS-TTS] text channel decoded:\n{decoded_text}")

        if batch_size != 1:
            raise ValueError("moss_tts_delay generate currently supports batch_size=1")

        text_do_sample = text_temperature > 0
        audio_do_sample = temperature > 0
        if not text_do_sample:
            text_temperature = 1.0
        if not audio_do_sample:
            temperature = 1.0

        current_input_ids = input_ids
        current_attention_mask = attention_mask
        generation_ids = input_ids
        generated_start = int(input_ids.shape[1])
        past_key_values = None

        is_stopping = mx.zeros((batch_size,), dtype=mx.bool_)
        audio_lengths = mx.zeros((batch_size,), dtype=mx.int64)

        # Delay-phase state machine
        # ─────────────────────────
        # After the main gen-slot audio generation ends, the model enters a
        # "delay phase" of n_vq steps to flush the staggered codebook tails
        # (codebook k is delayed by k positions, so the last k tokens of
        # codebook k are emitted during the delay phase).
        #
        # delay_active : bool  – whether we are currently in the delay phase
        # delay_count  : int64 – number of delay steps already *completed*
        #                        (0 while inactive; 1 after the first delay
        #                        step finishes its update; …; resets to 0
        #                        when the phase ends after n_vq+1 steps)
        delay_active = mx.zeros((batch_size,), dtype=mx.bool_)
        delay_count = mx.zeros((batch_size,), dtype=mx.int64)

        is_continuation = mx.logical_or(
            input_ids[:, -1, 0] == self.config.audio_start_token_id,
            input_ids[:, -1, 0] == self.config.audio_assistant_gen_slot_token_id,
        )
        audio_start_indices = find_last_equal(
            input_ids[..., 0], int(self.config.audio_start_token_id)
        ).astype(mx.int64)
        audio_start_mask = mx.logical_and(is_continuation, audio_start_indices != -1)
        audio_lengths = mx.where(
            audio_start_mask, seq_len - audio_start_indices, audio_lengths
        )
        is_audio = audio_start_mask

        exclude_when_text = [
            int(self.config.pad_token_id),
            int(self.config.audio_assistant_gen_slot_token_id),
            int(self.config.audio_assistant_delay_slot_token_id),
            int(self.config.audio_end_token_id),
        ]

        vocab_size = int(self.config.language_config.get("vocab_size", 155648))
        audio_vq_vocab = int(self.config.audio_vocab_size) + 1
        pad_code = int(self.config.audio_pad_code)

        allow_audio_text_ids = [
            int(self.config.audio_assistant_gen_slot_token_id),
            int(self.config.audio_assistant_delay_slot_token_id),
        ]

        allow_audio_text_unique_count = int(len(set(allow_audio_text_ids)))
        audio_text_suppress_mask = mx.ones((vocab_size,), dtype=mx.bool_)
        allow_idx = mx.array(allow_audio_text_ids, dtype=mx.int32)
        audio_text_suppress_mask = mx.put_along_axis(
            audio_text_suppress_mask,
            allow_idx,
            mx.zeros(allow_idx.shape, dtype=mx.bool_),
            axis=0,
        )

        audio_text_allow_count = int(
            mx.sum((~audio_text_suppress_mask).astype(mx.int32)).item()
        )

        try:
            for time_step in range(int(max_tokens)):
                is_audio_step = bool(is_audio.item())
                is_stopping_step = bool(is_stopping.item())
                audio_lengths_step = int(audio_lengths.item())
                delay_active_step = bool(delay_active.item())
                delay_count_step = int(delay_count.item())

                allowed_text_token_count = 0
                if is_audio_step:
                    allowed_text_token_count = 2
                    if time_step == 0:
                        allowed_text_token_count = 1
                else:
                    allowed_text_token_count = vocab_size - len(exclude_when_text)
                    if time_step <= n_vq:
                        allowed_text_token_count = allowed_text_token_count - 1
                if allowed_text_token_count < 0:
                    allowed_text_token_count = 0

                out = self.forward(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out["past_key_values"]
                logits = out["logits"]

                text_logits = logits[0][:, -1, :]

                masked_text_logits = text_logits
                if not is_audio_step:
                    masked_text_logits = _suppress_token_ids(
                        masked_text_logits, exclude_when_text
                    )
                else:
                    masked_text_logits = mx.where(
                        audio_text_suppress_mask[None, :],
                        mx.array(float("-inf"), dtype=masked_text_logits.dtype),
                        masked_text_logits,
                    )

                if time_step == 0:
                    masked_text_logits = _suppress_token_ids(
                        masked_text_logits,
                        [int(self.config.audio_assistant_delay_slot_token_id)],
                    )
                if time_step <= n_vq:
                    masked_text_logits = _suppress_token_ids(
                        masked_text_logits, [int(self.config.im_end_token_id)]
                    )

                next_text_token = mx.full(
                    (batch_size,), int(self.config.pad_token_id), dtype=mx.int32
                )

                # ── Delay-phase text-token override ──
                # While in delay and not yet flushed: force delay-slot token
                next_text_token = mx.where(
                    (~is_stopping) & delay_active & (delay_count < n_vq),
                    mx.full(
                        (batch_size,),
                        int(self.config.audio_assistant_delay_slot_token_id),
                        dtype=mx.int32,
                    ),
                    next_text_token,
                )
                # Delay phase fully flushed (delay_count == n_vq): emit audio-end
                is_audio_eos = (~is_stopping) & delay_active & (delay_count == n_vq)
                next_text_token = mx.where(
                    is_audio_eos,
                    mx.full(
                        (batch_size,),
                        int(self.config.audio_end_token_id),
                        dtype=mx.int32,
                    ),
                    next_text_token,
                )
                is_audio = mx.where(
                    is_audio_eos, mx.full((batch_size,), False), is_audio
                )

                # Not in delay phase → sample text freely from the model
                sampling_text_mask = (~is_stopping) & ~delay_active
                if bool(sampling_text_mask.item()):
                    sampled_text = sample_token(
                        masked_text_logits,
                        top_p=text_top_p,
                        top_k=text_top_k,
                        do_sample=text_do_sample,
                        temperature=text_temperature,
                    )
                    next_text_token = mx.where(
                        sampling_text_mask, sampled_text, next_text_token
                    )

                is_audio = mx.logical_or(
                    is_audio,
                    mx.logical_and(
                        sampling_text_mask,
                        next_text_token == int(self.config.audio_start_token_id),
                    ),
                )
                is_stopping = mx.logical_or(
                    is_stopping, next_text_token == int(self.config.im_end_token_id)
                )

                next_audio_tokens = mx.full(
                    (batch_size, n_vq), pad_code, dtype=mx.int32
                )
                pre_audio_mask = (
                    audio_lengths[:, None] > mx.arange(n_vq, dtype=mx.int64)[None, :]
                )

                # ── Delay-step mask: which codebook heads to sample ──
                # Detect the first delay-slot token (gen → delay transition)
                entering_delay = (~delay_active) & (
                    next_text_token
                    == int(self.config.audio_assistant_delay_slot_token_id)
                )
                # 1-based index of the *current* delay step:
                #   entering  → delay_count(0) + 1 = 1  (first delay step)
                #   active    → delay_count(k) + 1       (k-th completed, now on k+1)
                #   otherwise → 0                         (not in delay; mask = all True)
                in_delay_now = delay_active | entering_delay
                delay_step = mx.where(
                    in_delay_now,
                    delay_count + 1,
                    mx.zeros_like(delay_count),
                )

                # At delay_step k, codebooks 0..k-1 are already flushed;
                # only heads with index >= k still need sampling.
                head_indices = mx.arange(n_vq, dtype=mx.int64)[None, :]
                post_audio_mask = mx.where(
                    in_delay_now[:, None],
                    head_indices >= delay_step[:, None],
                    mx.full((batch_size, n_vq), True),
                )
                sampling_audio_mask = pre_audio_mask & post_audio_mask

                selected_audio_heads: List[int] = []
                if bool(sampling_audio_mask[0].any().item()):
                    for vq_idx in range(n_vq):
                        if not bool(sampling_audio_mask[0, vq_idx].item()):
                            continue
                        selected_audio_heads.append(int(vq_idx))

                    if 0 in selected_audio_heads:
                        vq0_logits = logits[1][:, -1, :]
                        if int(vq0_logits.shape[-1]) != audio_vq_vocab:
                            raise ValueError(
                                f"audio head vocab mismatch: got {vq0_logits.shape[-1]}, expected {audio_vq_vocab}"
                            )
                        vq0_logits = _suppress_token_ids(vq0_logits, [pad_code])
                        prev0 = _get_generated_audio_history(
                            generation_ids,
                            generated_start=generated_start,
                            vq_idx=0,
                        )
                        sampled0 = sample_token(
                            vq0_logits,
                            top_p=top_p,
                            top_k=top_k,
                            do_sample=audio_do_sample,
                            prev_tokens=prev0,
                            repetition_penalty=repetition_penalty,
                            temperature=temperature,
                        )
                        next_audio_tokens[0, 0] = sampled0[0]

                    active_tail_heads = [h for h in selected_audio_heads if h > 0]
                    if len(active_tail_heads) > 0:
                        tail_logits = mx.stack(
                            [logits[h + 1][:, -1, :] for h in active_tail_heads],
                            axis=1,
                        )
                        if int(tail_logits.shape[-1]) != audio_vq_vocab:
                            raise ValueError(
                                f"audio head vocab mismatch: got {tail_logits.shape[-1]}, expected {audio_vq_vocab}"
                            )
                        tail_shape = tail_logits.shape
                        tail_logits = _suppress_token_ids(
                            tail_logits.reshape(-1, int(tail_shape[-1])),
                            [pad_code],
                        ).reshape(tail_shape)

                        tail_prev = mx.stack(
                            [
                                _get_generated_audio_history(
                                    generation_ids,
                                    generated_start=generated_start,
                                    vq_idx=h,
                                )
                                for h in active_tail_heads
                            ],
                            axis=-1,
                        )
                        sampled_tail = sample_token(
                            tail_logits,
                            top_p=top_p,
                            top_k=top_k,
                            do_sample=audio_do_sample,
                            prev_tokens=tail_prev,
                            repetition_penalty=repetition_penalty,
                            temperature=temperature,
                        )
                        for i, head_idx in enumerate(active_tail_heads):
                            next_audio_tokens[0, int(head_idx)] = sampled_tail[0, i]

                inc_audio_mask = mx.logical_or(
                    next_text_token == int(self.config.audio_start_token_id),
                    mx.logical_or(
                        next_text_token
                        == int(self.config.audio_assistant_gen_slot_token_id),
                        next_text_token
                        == int(self.config.audio_assistant_delay_slot_token_id),
                    ),
                )
                inc_audio = inc_audio_mask.astype(mx.int64)
                audio_lengths = audio_lengths + inc_audio
                audio_lengths = mx.where(
                    next_text_token == int(self.config.audio_end_token_id),
                    mx.zeros((batch_size,), dtype=mx.int64),
                    audio_lengths,
                )

                # ── Update delay-phase state machine ──
                # Activate on first delay-slot token
                delay_active = delay_active | entering_delay
                # Tick counter for every active step (entering counts as active)
                delay_count = mx.where(delay_active, delay_count + 1, delay_count)
                # Deactivate once counter passes n_vq (audio-end was just emitted)
                delay_done = delay_count > n_vq
                delay_active = delay_active & ~delay_done
                delay_count = mx.where(
                    delay_done, mx.zeros_like(delay_count), delay_count
                )

                current_input_ids = mx.concatenate(
                    [next_text_token[:, None, None], next_audio_tokens[:, None, :]],
                    axis=2,
                )
                current_attention_mask = mx.concatenate(
                    [current_attention_mask, (~is_stopping)[:, None]], axis=1
                )
                generation_ids = mx.concatenate(
                    [generation_ids, current_input_ids], axis=1
                )
                mx.eval(generation_ids)

                if int(is_stopping.sum().item()) == int(batch_size):
                    break

                if time_step > 0 and time_step % 50 == 0:
                    mx.clear_cache()

                if verbose and time_step % 100 == 0:
                    _ = time_step
        except BaseException:
            raise

        elapsed = time.time() - start_time

        start_indices = find_last_equal(
            input_ids[..., 0], int(self.config.im_start_token_id)
        )
        start_idx = int(start_indices.item())
        slice_start = start_idx + 3
        if slice_start < 0:
            slice_start = 0
        if slice_start > int(generation_ids.shape[1]):
            slice_start = int(generation_ids.shape[1])
        generation_tail = generation_ids[:, slice_start:, :]
        start_length = int(input_ids.shape[1]) - int(slice_start)
        if start_length < 0:
            start_length = 0
        if verbose:
            gen_np = np.array(generation_ids[0], dtype=np.int64)
            gen_len = int(generation_ids.shape[1])
            gen_new = gen_len - int(input_ids.shape[1])
            print(
                f"[MOSS-TTS] generation_ids shape: {generation_ids.shape}  "
                f"(seq_len={gen_len}, new_tokens={gen_new})"
            )
            for ch in range(1 + n_vq):
                ch_ids = gen_np[:, ch].tolist()
                label = "text" if ch == 0 else f"vq{ch - 1:02d}"
                formatted = " ".join(f"{v:6d}" for v in ch_ids)
                print(f"[MOSS-TTS] out ch[{label}] ({len(ch_ids)}): {formatted}")
            if self.tokenizer:
                decoded_text = self.tokenizer.decode(gen_np[:, 0].tolist())
                print(f"[MOSS-TTS] out text channel decoded:\n{decoded_text}")

        decoded_messages = self.processor.decode(
            [(start_length, generation_tail[0])], codec=self.codec
        )
        first_message = decoded_messages[0] if len(decoded_messages) > 0 else None
        audio_chunks = [] if first_message is None else list(first_message.audio)
        if len(audio_chunks) == 0:
            audio = mx.zeros((1,), dtype=mx.float32)
            mx.eval(audio)
        else:
            if len(audio_chunks) == 1:
                audio = audio_chunks[0]
            else:
                audio = mx.concatenate(audio_chunks, axis=0)
            mx.eval(audio)

        yield GenerationResult(
            audio=audio,
            samples=int(audio.shape[0]),
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=int(generation_ids.shape[1] - input_ids.shape[1]),
            audio_duration=_format_duration(float(audio.shape[0]) / self.sample_rate),
            real_time_factor=(
                (float(audio.shape[0]) / self.sample_rate) / elapsed
                if elapsed > 0
                else 0.0
            ),
            prompt={
                "tokens_generated": int(generation_ids.shape[1] - input_ids.shape[1]),
                "elapsed_sec": float(elapsed),
                "mode": mode,
                "conversation_messages": (
                    int(len(conversation_input))
                    if conversation_input is not None
                    else 0
                ),
            },
            audio_samples=int(audio.shape[0]),
            processing_time_seconds=float(elapsed),
            peak_memory_usage=float(mx.get_peak_memory() / 1e9),
            is_streaming_chunk=False,
            is_final_chunk=True,
        )
