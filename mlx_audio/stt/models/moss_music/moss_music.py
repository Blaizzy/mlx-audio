import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import ModelArgs as Qwen3Args
from mlx_lm.models.qwen3 import Qwen3Model

from mlx_audio.stt.models.base import STTOutput

from .config import AudioEncoderConfig, ModelConfig
from .processor import MossMusicProcessor, ProcessorOutput


@dataclass
class StreamingResult:
    text: str
    is_final: bool
    start_time: float
    end_time: float
    language: str = "en"
    prompt_tokens: int = 0
    generation_tokens: int = 0


@dataclass(frozen=True)
class _TimeMarker:
    start: float
    end: Optional[float]
    span_start: int
    span_end: int
    raw: str


def sinusoids(length: int, channels: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    max_timescale = 10000.0
    log_timescale = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1).astype(
        dtype
    )


class AudioAttention(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        return self.out_proj(out)


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            eps=config.layer_norm_eps,
        )
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = nn.gelu(self.fc1(x))
        x = self.fc2(x)
        return residual + x


class MossMusicEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.stem_proj = nn.Linear(
            config.downsample_hidden_size * 16,
            config.d_model,
        )
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.out_proj = (
            nn.Linear(config.d_model, config.output_dim, bias=False)
            if config.output_dim != config.d_model
            else None
        )
        self.deepstack_encoder_layer_indexes = list(
            config.deepstack_encoder_layer_indexes or []
        )
        self._deepstack_capture_map = {
            layer_idx: capture_idx
            for capture_idx, layer_idx in enumerate(
                self.deepstack_encoder_layer_indexes
            )
        }
        self._embed_positions = sinusoids(config.max_source_positions, config.d_model)
        mx.eval(self._embed_positions)

    @staticmethod
    def compute_downsampled_length(length: int) -> int:
        def conv_out_len(x: int) -> int:
            return (int(x) - 1) // 2 + 1

        return conv_out_len(conv_out_len(conv_out_len(length)))

    def _attention_mask(self, lengths: mx.array, max_len: int, dtype: mx.Dtype):
        valid = mx.arange(max_len)[None, :] < lengths[:, None]
        mask = mx.where(valid[:, None, None, :], 0.0, -1e9)
        return mask.astype(dtype)

    def __call__(
        self,
        input_features: mx.array,
        feature_lens: Optional[mx.array] = None,
        output_deepstack_hidden_states: bool = True,
    ) -> Tuple[mx.array, List[mx.array]]:
        if input_features.ndim == 2:
            input_features = input_features[None]
        if input_features.ndim != 3:
            raise ValueError(f"Expected [B, n_mels, T], got {input_features.shape}.")

        if feature_lens is None:
            feature_lens = mx.full((input_features.shape[0],), input_features.shape[-1])
        feature_lens = feature_lens.astype(mx.int32)
        downsampled_lengths = mx.array(
            [self.compute_downsampled_length(int(v)) for v in np.array(feature_lens)],
            dtype=mx.int32,
        )

        x = input_features[:, :, :, None]
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        x = nn.gelu(self.conv3(x))
        B, F, T, C = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(B, T, C * F)
        x = self.stem_proj(x)

        max_len = int(np.array(mx.max(downsampled_lengths)))
        if x.shape[1] > max_len:
            x = x[:, :max_len, :]

        pos = self._embed_positions[: x.shape[1]].astype(x.dtype)
        x = x + pos[None]
        attention_mask = self._attention_mask(downsampled_lengths, x.shape[1], x.dtype)

        deepstack_states: List[Optional[mx.array]] = [None] * len(
            self.deepstack_encoder_layer_indexes
        )
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, mask=attention_mask)
            capture_idx = self._deepstack_capture_map.get(layer_idx)
            if output_deepstack_hidden_states and capture_idx is not None:
                deepstack_states[capture_idx] = x

        x = self.layer_norm(x)
        if self.out_proj is not None:
            x = self.out_proj(x)

        ordered = [state for state in deepstack_states if state is not None]
        if self.out_proj is not None:
            ordered = [self.out_proj(state) for state in ordered]
        return x, ordered


class GatedMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Model(nn.Module):
    _TIME_VALUE = (
        r"(?:\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?"
        r"|\d+(?:[.,]\d+)?\s*s"
        r"|\d+(?:[.,]\d+)?)"
    )
    _LINE_TIME_VALUE = (
        r"(?:\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?" r"|\d+(?:[.,]\d+)?\s*s)"
    )
    _TIME_RANGE_SEPARATOR = r"(?:-|\u2013|\u2014|~|to|\u2192)"
    _BRACKETED_TIMESTAMP_RE = re.compile(
        rf"(?P<open>[\[\(<\u3010])\s*"
        rf"(?P<start>{_TIME_VALUE})\s*"
        rf"(?:(?:{_TIME_RANGE_SEPARATOR})\s*(?P<end>{_TIME_VALUE}))?"
        rf"\s*(?P<close>[\]\)>\u3011])",
        flags=re.IGNORECASE,
    )
    _LINE_TIMESTAMP_RE = re.compile(
        rf"(?m)(?:^|\n)\s*"
        rf"(?P<start>{_LINE_TIME_VALUE})\s*"
        rf"(?:(?:{_TIME_RANGE_SEPARATOR})\s*(?P<end>{_LINE_TIME_VALUE}))?"
        r"\s*[:\uff1a\-\u2013\u2014]\s*",
        flags=re.IGNORECASE,
    )

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.audio_encoder = MossMusicEncoder(config.audio_config)
        text_args = Qwen3Args.from_dict(config.language_config.__dict__)
        self.language_model = Qwen3Model(text_args)
        self.audio_adapter = GatedMLP(
            config.audio_config.output_dim,
            config.adapter_hidden_size,
            config.language_config.hidden_size,
        )

        num_deepstack = len(config.audio_config.deepstack_encoder_layer_indexes or [])
        if config.deepstack_num_inject_layers is not None:
            num_deepstack = min(num_deepstack, int(config.deepstack_num_inject_layers))
        self.deepstack_audio_merger_list = [
            GatedMLP(
                config.audio_config.output_dim,
                config.adapter_hidden_size,
                config.language_config.hidden_size,
            )
            for _ in range(num_deepstack)
        ]
        self.lm_head = nn.Linear(
            config.language_config.hidden_size,
            config.language_config.vocab_size,
            bias=False,
        )
        self._processor: Optional[MossMusicProcessor] = None

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in range(len(self.layers))]

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[KVCache]] = None,
        input_embeddings: Optional[mx.array] = None,
        deepstack_embeddings: Optional[List[mx.array]] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.language_model.embed_tokens(input_ids)

        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c)
            if deepstack_embeddings is not None and layer_idx < len(
                deepstack_embeddings
            ):
                h = h + deepstack_embeddings[layer_idx]

        h = self.language_model.norm(h)
        return self.lm_head(h)

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        return not p.startswith("audio_encoder")

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized: Dict[str, mx.array] = {}
        for key, value in weights.items():
            if key == "audio_encoder.embed_positions.inv_timescales":
                continue
            attention_match = re.match(
                r"(audio_encoder\.layers\.\d+)\."
                r"(q_proj|k_proj|v_proj|out_proj)\.(.+)",
                key,
            )
            if attention_match is not None:
                key = (
                    f"{attention_match.group(1)}.self_attn."
                    f"{attention_match.group(2)}.{attention_match.group(3)}"
                )
            if (
                key.startswith("audio_encoder.conv")
                and key.endswith(".weight")
                and value.ndim == 4
                and value.shape[1] != 3
            ):
                value = value.transpose(0, 2, 3, 1)
            sanitized[key] = value
        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model._processor = MossMusicProcessor(model_path, model.config)
        model.audio_encoder._embed_positions = (
            model.audio_encoder._embed_positions.astype(
                model.audio_encoder.conv1.weight.dtype
            )
        )
        mx.eval(model.audio_encoder._embed_positions)
        return model

    def _ensure_processor(self) -> MossMusicProcessor:
        if self._processor is None:
            if self.config.model_path is None:
                raise ValueError("MOSS-Music processor is not loaded.")
            self._processor = MossMusicProcessor(self.config.model_path, self.config)
        return self._processor

    def _encode_audio_chunks(
        self,
        audio_data: mx.array,
        audio_data_seqlens: mx.array,
    ) -> Tuple[mx.array, List[mx.array]]:
        lengths = [int(x) for x in np.array(audio_data_seqlens).reshape(-1)]
        chunk_frames = int(self.config.audio_config.n_window * 2)
        chunks: List[mx.array] = []
        chunk_lengths: List[int] = []

        for mel, length in zip(audio_data, lengths):
            start = 0
            while start < length:
                end = min(start + chunk_frames, length)
                chunks.append(mel[:, start:end])
                chunk_lengths.append(end - start)
                start = end

        if not chunks:
            empty = mx.zeros((0, self.config.audio_config.output_dim))
            return empty, [empty for _ in range(len(self.deepstack_audio_merger_list))]

        max_len = max(chunk_lengths)
        padded_chunks = []
        for chunk in chunks:
            if chunk.shape[1] < max_len:
                chunk = mx.pad(chunk, ((0, 0), (0, max_len - chunk.shape[1])))
            padded_chunks.append(chunk)

        batch = mx.stack(padded_chunks, axis=0)
        out_parts: List[mx.array] = []
        deepstack_parts: List[List[mx.array]] = [
            [] for _ in range(len(self.deepstack_audio_merger_list))
        ]
        conv_batch = max(int(self.config.audio_config.conv_chunksize), 1)

        for start in range(0, len(chunk_lengths), conv_batch):
            end = min(start + conv_batch, len(chunk_lengths))
            batch_lens = chunk_lengths[start:end]
            encoded, deepstack = self.audio_encoder(
                batch[start:end],
                mx.array(batch_lens, dtype=mx.int32),
                output_deepstack_hidden_states=bool(self.deepstack_audio_merger_list),
            )
            for local_idx, raw_len in enumerate(batch_lens):
                down_len = self.audio_encoder.compute_downsampled_length(raw_len)
                out_parts.append(encoded[local_idx, :down_len])
                for ds_idx, ds in enumerate(deepstack[: len(deepstack_parts)]):
                    deepstack_parts[ds_idx].append(ds[local_idx, :down_len])

        audio_embeds = mx.concatenate(out_parts, axis=0)
        deepstack_embeds = [
            mx.concatenate(parts, axis=0) if parts else mx.zeros_like(audio_embeds)
            for parts in deepstack_parts
        ]
        return audio_embeds, deepstack_embeds

    def _build_prompt_embeddings(
        self,
        processed: ProcessorOutput,
    ) -> Tuple[mx.array, mx.array, Optional[List[mx.array]], int]:
        input_ids = processed.input_ids.astype(mx.int32)
        audio_mask = processed.audio_input_mask.astype(mx.bool_)
        text_ids = mx.where(audio_mask, 0, input_ids)
        inputs_embeds = self.language_model.embed_tokens(text_ids[None])

        if processed.audio_data is None or processed.audio_data_seqlens is None:
            return input_ids, inputs_embeds, None, int(input_ids.shape[0])

        audio_features, deepstack_features = self._encode_audio_chunks(
            processed.audio_data.astype(self.audio_encoder.conv1.weight.dtype),
            processed.audio_data_seqlens,
        )
        audio_features = self.audio_adapter(audio_features).astype(inputs_embeds.dtype)
        expected = int(np.array(mx.sum(audio_mask.astype(mx.int32))))
        if expected != int(audio_features.shape[0]):
            raise ValueError(
                f"Audio token count mismatch: prompt has {expected} tokens but "
                f"audio encoder produced {audio_features.shape[0]} embeddings."
            )

        audio_indices = mx.cumsum(audio_mask.astype(mx.int32)) - 1
        audio_indices = mx.clip(audio_indices, 0, max(audio_features.shape[0] - 1, 0))
        gathered_audio = audio_features[audio_indices]
        inputs_embeds = mx.where(
            audio_mask[None, :, None], gathered_audio[None], inputs_embeds
        )

        deepstack_embeddings = []
        for merger, features in zip(
            self.deepstack_audio_merger_list, deepstack_features
        ):
            merged = merger(features).astype(inputs_embeds.dtype)
            gathered = merged[audio_indices]
            zeros = mx.zeros_like(gathered)
            deepstack_embeddings.append(
                mx.where(audio_mask[:, None], gathered, zeros)[None]
            )

        return input_ids, inputs_embeds, deepstack_embeddings, int(input_ids.shape[0])

    def _generate_tokens(
        self,
        prompt_ids: mx.array,
        input_embeddings: mx.array,
        deepstack_embeddings: Optional[List[mx.array]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: Optional[float],
        repetition_context_size: int,
        prefill_step_size: int,
    ) -> Generator[int, None, None]:
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        sampler = make_sampler(temperature, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
        cache = self.make_cache()
        token_history = None

        def model_call(ids, embeds=None, ds=None):
            return self(
                ids[None],
                cache=cache,
                input_embeddings=embeds[None] if embeds is not None else None,
                deepstack_embeddings=[x[None] for x in ds] if ds is not None else None,
            )

        def step(ids, embeds=None, ds=None):
            nonlocal token_history
            logits = model_call(ids, embeds, ds)[:, -1, :]
            if logits_processors and len(ids) > 0:
                token_history = (
                    mx.concat([token_history, ids])
                    if token_history is not None
                    else ids
                )
                for processor in logits_processors:
                    logits = processor(token_history, logits)
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            token = sampler(logprobs)
            mx.eval(token)
            return int(token.item())

        total = int(prompt_ids.shape[0])
        processed = 0
        while total - processed > 1:
            remaining = (total - processed) - 1
            n = min(prefill_step_size, remaining)
            ds_slice = (
                [x[0, processed : processed + n] for x in deepstack_embeddings]
                if deepstack_embeddings is not None
                else None
            )
            model_call(
                prompt_ids[processed : processed + n],
                input_embeddings[0, processed : processed + n],
                ds_slice,
            )
            mx.eval([c.state for c in cache])
            processed += n
            mx.clear_cache()

        ds_tail = (
            [x[0, processed:] for x in deepstack_embeddings]
            if deepstack_embeddings is not None
            else None
        )
        next_token = step(
            prompt_ids[processed:],
            input_embeddings[0, processed:],
            ds_tail,
        )

        for _ in range(max_tokens):
            yield next_token
            next_token = step(mx.array([next_token], dtype=mx.int32))

    @staticmethod
    def _strip_thinking(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"^\s*<think>.*", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _parse_timestamp_seconds(value: str) -> float:
        value = value.strip().lower().replace(",", ".")
        if value.endswith("s"):
            value = value[:-1].strip()
        parts = value.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(value)

    @classmethod
    def _collect_time_markers(cls, text: str) -> List[_TimeMarker]:
        markers: List[_TimeMarker] = []

        for match in cls._BRACKETED_TIMESTAMP_RE.finditer(text):
            try:
                start = cls._parse_timestamp_seconds(match.group("start"))
                end = (
                    cls._parse_timestamp_seconds(match.group("end"))
                    if match.group("end")
                    else None
                )
            except ValueError:
                continue
            markers.append(
                _TimeMarker(
                    start=start,
                    end=end,
                    span_start=match.start(),
                    span_end=match.end(),
                    raw=match.group(0),
                )
            )

        for match in cls._LINE_TIMESTAMP_RE.finditer(text):
            try:
                start = cls._parse_timestamp_seconds(match.group("start"))
                end = (
                    cls._parse_timestamp_seconds(match.group("end"))
                    if match.group("end")
                    else None
                )
            except ValueError:
                continue
            markers.append(
                _TimeMarker(
                    start=start,
                    end=end,
                    span_start=match.start(),
                    span_end=match.end(),
                    raw=match.group(0).strip(),
                )
            )

        markers.sort(key=lambda marker: (marker.span_start, marker.span_end))
        deduped: List[_TimeMarker] = []
        last_end = -1
        for marker in markers:
            if marker.span_start < last_end:
                continue
            deduped.append(marker)
            last_end = marker.span_end
        return deduped

    @staticmethod
    def _clean_segment_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^[\s:\uff1a,;|\-\u2013\u2014>]+", "", text)
        text = re.sub(r"[\s|]+$", "", text)
        return text.strip()

    @classmethod
    def _fallback_segments(
        cls,
        text: str,
        *,
        audio_duration: Optional[float] = None,
        total_time: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        end = audio_duration if audio_duration is not None else total_time
        end = 0.0 if end is None else float(end)
        return [
            {
                "text": text,
                "start": 0.0,
                "end": round(end, 3),
                "kind": "text",
                "marker": None,
            }
        ]

    @classmethod
    def _parse_structured_segments(
        cls,
        text: str,
        *,
        audio_duration: Optional[float] = None,
        total_time: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        markers = cls._collect_time_markers(text)
        if not markers:
            return cls._fallback_segments(
                text,
                audio_duration=audio_duration,
                total_time=total_time,
            )

        segments: List[Dict[str, object]] = []
        for index, marker in enumerate(markers):
            next_marker = markers[index + 1] if index + 1 < len(markers) else None
            body_end = next_marker.span_start if next_marker is not None else len(text)
            segment_text = cls._clean_segment_text(text[marker.span_end : body_end])
            if not segment_text and marker.end is None:
                continue

            end = marker.end
            if end is None and next_marker is not None:
                end = next_marker.start
            if end is None:
                end = audio_duration if audio_duration is not None else total_time
            if end is None:
                end = marker.start
            if end < marker.start:
                end = marker.start

            segments.append(
                {
                    "text": segment_text,
                    "start": round(float(marker.start), 3),
                    "end": round(float(end), 3),
                    "kind": "timestamped_text",
                    "marker": marker.raw,
                }
            )

        if not segments:
            return cls._fallback_segments(
                text,
                audio_duration=audio_duration,
                total_time=total_time,
            )
        return segments

    def generate(
        self,
        audio: Optional[Union[str, mx.array, np.ndarray, Sequence]] = None,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 100,
        prompt: Optional[str] = None,
        strip_thinking: Optional[bool] = None,
        enable_time_marker: Optional[bool] = None,
        prefill_step_size: int = 2048,
        stream: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> Union[STTOutput, Generator[StreamingResult, None, None]]:
        if stream:
            return self._stream_generate(
                audio,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                prompt=prompt,
                strip_thinking=strip_thinking,
                enable_time_marker=enable_time_marker,
                prefill_step_size=prefill_step_size,
                verbose=verbose,
            )

        started = time.time()
        processor = self._ensure_processor()
        processed = processor(
            text=prompt or self.config.default_prompt,
            audio=audio,
            enable_time_marker=enable_time_marker,
        )
        prompt_ids, input_embeds, deepstack, prompt_tokens = (
            self._build_prompt_embeddings(processed)
        )
        prefill_start = time.time()
        tokens: List[int] = []
        for token in self._generate_tokens(
            prompt_ids,
            input_embeds,
            deepstack,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            prefill_step_size=prefill_step_size,
        ):
            if token == self.config.eos_token_id:
                break
            tokens.append(token)
        total_time = time.time() - started
        gen_time = max(time.time() - prefill_start, 1e-9)
        text = processor.decode(tokens, skip_special_tokens=True)
        if self.config.strip_thinking if strip_thinking is None else strip_thinking:
            text = self._strip_thinking(text)
        audio_duration = (
            sum(processed.audio_durations or []) if processed.audio_durations else None
        )
        segments = self._parse_structured_segments(
            text,
            audio_duration=audio_duration,
            total_time=total_time,
        )
        return STTOutput(
            text=text,
            segments=segments,
            prompt_tokens=prompt_tokens,
            generation_tokens=len(tokens),
            total_tokens=prompt_tokens + len(tokens),
            total_time=total_time,
            prompt_tps=0.0,
            generation_tps=len(tokens) / gen_time,
        )

    def _stream_generate(
        self,
        audio,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: Optional[float],
        repetition_context_size: int,
        prompt: Optional[str],
        strip_thinking: Optional[bool],
        enable_time_marker: Optional[bool],
        prefill_step_size: int,
        verbose: bool,
    ) -> Generator[StreamingResult, None, None]:
        processor = self._ensure_processor()
        processed = processor(
            text=prompt or self.config.default_prompt,
            audio=audio,
            enable_time_marker=enable_time_marker,
        )
        prompt_ids, input_embeds, deepstack, prompt_tokens = (
            self._build_prompt_embeddings(processed)
        )
        generated = 0
        for token in self._generate_tokens(
            prompt_ids,
            input_embeds,
            deepstack,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            prefill_step_size=prefill_step_size,
        ):
            if token == self.config.eos_token_id:
                break
            generated += 1
            text = processor.decode([token], skip_special_tokens=True)
            yield StreamingResult(
                text=text,
                is_final=False,
                start_time=0.0,
                end_time=0.0,
                prompt_tokens=prompt_tokens,
                generation_tokens=generated,
            )
        yield StreamingResult(
            text="",
            is_final=True,
            start_time=0.0,
            end_time=0.0,
            prompt_tokens=prompt_tokens,
            generation_tokens=generated,
        )
