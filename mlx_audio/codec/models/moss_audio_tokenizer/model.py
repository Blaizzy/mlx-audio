"""MLX implementation of the MOSS audio tokenizer codec."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from .config import MossAudioTokenizerConfig, load_moss_audio_tokenizer_config
from .decoder import build_moss_audio_tokenizer_decoder_modules
from .encoder import build_moss_audio_tokenizer_encoder_modules
from .quantizer import (
    MossAudioTokenizerResidualLFQ,
    MossAudioTokenizerResidualVQ,
    build_moss_audio_tokenizer_quantizer,
)


@dataclass
class MossAudioTokenizerEncoderOutput:
    audio_codes: Optional[mx.array] = None
    audio_codes_lengths: Optional[mx.array] = None
    encoder_hidden_states: Optional[mx.array] = None


@dataclass
class MossAudioTokenizerDecoderOutput:
    audio: Optional[mx.array] = None
    audio_lengths: Optional[mx.array] = None


@dataclass
class MossAudioTokenizerOutput:
    audio: Optional[mx.array] = None
    audio_lengths: Optional[mx.array] = None
    audio_codes: Optional[mx.array] = None
    audio_codes_lengths: Optional[mx.array] = None


class MossAudioTokenizer(nn.Module):
    """Standalone codec used by the MOSS-TTS family."""

    def __init__(self, config: MossAudioTokenizerConfig):
        super().__init__()
        self.config = config
        self.sampling_rate = config.sampling_rate
        self.downsample_rate = config.downsample_rate
        self.causal_transformer_context_duration = (
            config.causal_transformer_context_duration
        )

        self.encoder = build_moss_audio_tokenizer_encoder_modules(config)
        self.quantizer = build_moss_audio_tokenizer_quantizer(config.quantizer)
        self.decoder = build_moss_audio_tokenizer_decoder_modules(config)

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.downsample_rate

    def _build_caches(self, modules: list[nn.Module]) -> list[Optional[list]]:
        caches: list[Optional[list]] = []
        for module in modules:
            make_cache = getattr(module, "make_cache", None)
            caches.append(make_cache() if callable(make_cache) else None)
        return caches

    def _ensure_audio_layout(self, input_values: mx.array) -> mx.array:
        if input_values.ndim == 1:
            input_values = input_values[None, None, :]
        elif input_values.ndim == 2:
            input_values = input_values[:, None, :]
        elif input_values.ndim != 3:
            raise ValueError(
                f"Expected input_values with 1/2/3 dims, got shape={input_values.shape}"
            )
        return input_values

    def _resolve_requested_quantizers(self, num_quantizers: Optional[int]) -> int:
        configured_quantizers = self.config.quantizer.num_quantizers
        if num_quantizers is None:
            return configured_quantizers
        if num_quantizers <= 0:
            raise ValueError("num_quantizers must be > 0 when provided.")
        if num_quantizers > configured_quantizers:
            raise ValueError(
                f"num_quantizers ({num_quantizers}) must be <= configured "
                f"quantizer count ({configured_quantizers})."
            )
        return int(num_quantizers)

    def _normalize_decode_audio_codes(
        self,
        audio_codes: mx.array,
        *,
        num_quantizers: Optional[int],
    ) -> mx.array:
        configured_quantizers = self.config.quantizer.num_quantizers
        requested_quantizers = self._resolve_requested_quantizers(num_quantizers)

        if audio_codes.ndim not in {2, 3}:
            raise ValueError(
                f"Expected audio_codes with 2 or 3 dims, got shape={audio_codes.shape}"
            )

        candidates: list[tuple[str, int, mx.array]] = []

        def register_candidate(
            orientation: str,
            quantizer_count: int,
            normalized: mx.array,
        ) -> None:
            candidates.append((orientation, quantizer_count, normalized))

        if audio_codes.ndim == 2:
            if audio_codes.shape[0] == configured_quantizers:
                register_candidate(
                    "NQ-first",
                    configured_quantizers,
                    audio_codes[:, None, :],
                )
            if (
                requested_quantizers != configured_quantizers
                and audio_codes.shape[0] == requested_quantizers
            ):
                register_candidate(
                    "NQ-first",
                    requested_quantizers,
                    audio_codes[:, None, :],
                )
            if audio_codes.shape[1] == configured_quantizers:
                register_candidate(
                    "NQ-last",
                    configured_quantizers,
                    audio_codes.transpose(1, 0)[:, None, :],
                )
            if (
                requested_quantizers != configured_quantizers
                and audio_codes.shape[1] == requested_quantizers
            ):
                register_candidate(
                    "NQ-last",
                    requested_quantizers,
                    audio_codes.transpose(1, 0)[:, None, :],
                )
        else:
            if audio_codes.shape[0] == configured_quantizers:
                register_candidate("NQ-first", configured_quantizers, audio_codes)
            if (
                requested_quantizers != configured_quantizers
                and audio_codes.shape[0] == requested_quantizers
            ):
                register_candidate("NQ-first", requested_quantizers, audio_codes)
            if audio_codes.shape[-1] == configured_quantizers:
                register_candidate(
                    "NQ-last",
                    configured_quantizers,
                    audio_codes.transpose(2, 0, 1),
                )
            if (
                requested_quantizers != configured_quantizers
                and audio_codes.shape[-1] == requested_quantizers
            ):
                register_candidate(
                    "NQ-last",
                    requested_quantizers,
                    audio_codes.transpose(2, 0, 1),
                )

        if not candidates:
            shape_contract = (
                "Expected (NQ, T)/(T, NQ) or (NQ, B, T)/(B, T, NQ) with "
                f"NQ in {{{configured_quantizers}"
                + (
                    f", {requested_quantizers}"
                    if requested_quantizers != configured_quantizers
                    else ""
                )
                + "}."
            )
            raise ValueError(
                f"Unrecognized audio_codes layout for shape={audio_codes.shape}. "
                f"{shape_contract}"
            )

        if len(candidates) > 1:
            if num_quantizers is not None:
                # When callers request a quantizer prefix explicitly, prefer layout
                # candidates that already match that quantizer count before applying
                # canonical tie-break rules. This preserves legal NQ-last prefix time axes.
                requested_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[1] == requested_quantizers
                ]
                if len(requested_candidates) == 1:
                    candidates = requested_candidates
                elif len(requested_candidates) > 1:
                    requested_nq_first = [
                        candidate
                        for candidate in requested_candidates
                        if candidate[0] == "NQ-first"
                    ]
                    requested_nq_last = [
                        candidate
                        for candidate in requested_candidates
                        if candidate[0] == "NQ-last"
                    ]

                    # Explicit square ties are ambiguous by shape alone. Keep canonical
                    # NQ-first orientation for these cases so canonical decode inputs
                    # do not get transposed under explicit num_quantizers requests.
                    is_square_requested_tie = False
                    if audio_codes.ndim == 2:
                        is_square_requested_tie = (
                            int(audio_codes.shape[0]) == requested_quantizers
                            and int(audio_codes.shape[1]) == requested_quantizers
                        )
                    else:
                        is_square_requested_tie = (
                            int(audio_codes.shape[0]) == requested_quantizers
                            and int(audio_codes.shape[1]) == 1
                            and int(audio_codes.shape[-1]) == requested_quantizers
                        ) or (
                            int(audio_codes.shape[0]) == 1
                            and int(audio_codes.shape[1]) == requested_quantizers
                            and int(audio_codes.shape[-1]) == requested_quantizers
                        )

                    if is_square_requested_tie and len(requested_nq_first) == 1:
                        candidates = requested_nq_first
                    elif len(requested_nq_last) == 1:
                        candidates = requested_nq_last
                    else:
                        candidates = requested_candidates

            if len(candidates) > 1:
                # Fallback tie-break: prefer canonical layout so encode() -> decode()
                # round-trips remain valid when T == NQ.
                canonical_candidates = [
                    candidate for candidate in candidates if candidate[0] == "NQ-first"
                ]
                if len(canonical_candidates) == 1:
                    candidates = canonical_candidates
                else:
                    candidate_labels = [
                        f"{orientation}[nq={nq}]"
                        for orientation, nq, _ in candidates
                    ]
                    raise ValueError(
                        "Ambiguous audio_codes layout. Multiple interpretations are valid for "
                        f"shape={audio_codes.shape}: {candidate_labels}. "
                        "Use a canonical layout to disambiguate."
                    )

        _, source_quantizers, normalized = candidates[0]
        if requested_quantizers > source_quantizers:
            raise ValueError(
                f"num_quantizers ({requested_quantizers}) must be <= decoded source "
                f"quantizer count ({source_quantizers})."
            )
        if requested_quantizers < source_quantizers:
            normalized = normalized[:requested_quantizers]
        return normalized

    def _encode_frame(
        self,
        input_values: mx.array,
        input_lengths: Optional[mx.array] = None,
        n_quantizers: Optional[int] = None,
        caches: Optional[list[Optional[list]]] = None,
    ) -> MossAudioTokenizerEncoderOutput:
        input_values = self._ensure_audio_layout(input_values)
        batch_size, _, time_steps = input_values.shape

        if input_lengths is None:
            input_lengths = mx.full((batch_size,), time_steps, dtype=mx.int32)
        else:
            input_lengths = input_lengths.astype(mx.int32)

        if time_steps % self.downsample_rate != 0:
            pad_length = self.downsample_rate - (time_steps % self.downsample_rate)
            input_values = mx.pad(input_values, [(0, 0), (0, 0), (0, pad_length)])

        hidden = input_values
        hidden_lengths = input_lengths
        for idx, module in enumerate(self.encoder):
            cache = caches[idx] if caches is not None else None
            hidden, hidden_lengths = module(hidden, hidden_lengths, cache=cache)

        quantizer = self.quantizer
        if not isinstance(
            quantizer, (MossAudioTokenizerResidualLFQ, MossAudioTokenizerResidualVQ)
        ):
            raise TypeError(f"Unsupported quantizer type: {type(quantizer)}")
        _, audio_codes, audio_codes_lengths = quantizer(
            hidden, hidden_lengths, n_quantizers
        )

        return MossAudioTokenizerEncoderOutput(
            audio_codes=audio_codes.astype(mx.int32),
            audio_codes_lengths=audio_codes_lengths.astype(mx.int32),
            encoder_hidden_states=hidden,
        )

    def _decode_frame(
        self,
        audio_codes: mx.array,
        audio_codes_lengths: Optional[mx.array] = None,
        caches: Optional[list[Optional[list]]] = None,
    ) -> MossAudioTokenizerDecoderOutput:
        if audio_codes.ndim != 3:
            raise ValueError(
                "Expected canonical audio_codes layout (NQ, B, T) in _decode_frame, "
                f"got shape={audio_codes.shape}."
            )
        audio_codes = audio_codes.astype(mx.int32)
        _, batch_size, time_steps = audio_codes.shape

        if audio_codes_lengths is None:
            audio_codes_lengths = mx.full((batch_size,), time_steps, dtype=mx.int32)
        else:
            audio_codes_lengths = audio_codes_lengths.astype(mx.int32)

        quantizer = self.quantizer
        if not isinstance(
            quantizer, (MossAudioTokenizerResidualLFQ, MossAudioTokenizerResidualVQ)
        ):
            raise TypeError(f"Unsupported quantizer type: {type(quantizer)}")
        decoded = quantizer.decode_codes(audio_codes)

        audio = decoded
        audio_lengths = audio_codes_lengths
        for idx, module in enumerate(self.decoder):
            cache = caches[idx] if caches is not None else None
            audio, audio_lengths = module(audio, audio_lengths, cache=cache)

        return MossAudioTokenizerDecoderOutput(
            audio=audio,
            audio_lengths=audio_lengths.astype(mx.int32),
        )

    def encode(
        self,
        input_values: mx.array,
        padding_mask: Optional[mx.array] = None,
        num_quantizers: Optional[int] = None,
        return_dict: bool = True,
        chunk_duration: Optional[float] = None,
    ) -> MossAudioTokenizerEncoderOutput | tuple[mx.array, mx.array]:
        input_values = self._ensure_audio_layout(input_values)
        batch_size, _, time_steps = input_values.shape

        if padding_mask is not None:
            input_lengths = mx.sum(padding_mask.astype(mx.int32), axis=-1).astype(mx.int32)
        else:
            input_lengths = mx.full((batch_size,), time_steps, dtype=mx.int32)

        if chunk_duration is None:
            output = self._encode_frame(
                input_values, input_lengths, n_quantizers=num_quantizers
            )
        else:
            if chunk_duration <= 0:
                raise ValueError("chunk_duration must be > 0 when provided.")
            if chunk_duration > self.causal_transformer_context_duration:
                raise ValueError(
                    "chunk_duration must be <= "
                    f"{self.causal_transformer_context_duration}, got {chunk_duration}."
                )
            if batch_size != 1:
                raise ValueError(
                    "Streaming encode currently only supports batch_size=1."
                )

            chunk_length = int(round(chunk_duration * self.sampling_rate))
            if chunk_length <= 0:
                raise ValueError(
                    "chunk_duration is too small and results in chunk_length <= 0."
                )
            if chunk_length % self.downsample_rate != 0:
                raise ValueError(
                    "chunk_duration * sampling_rate must be divisible by downsample_rate. "
                    f"Got chunk_length={chunk_length}, downsample_rate={self.downsample_rate}."
                )

            input_length = int(input_lengths[0])
            if input_length <= chunk_length:
                output = self._encode_frame(
                    input_values[..., :input_length],
                    input_lengths,
                    n_quantizers=num_quantizers,
                )
            else:
                caches = self._build_caches(self.encoder)
                codes_chunks = []
                hidden_chunks = []
                for start_idx in range(0, input_length, chunk_length):
                    input_length_i = min(chunk_length, input_length - start_idx)
                    if input_length_i <= 0:
                        break
                    input_values_i = input_values[
                        ..., start_idx : start_idx + input_length_i
                    ]
                    input_lengths_i = mx.array([input_length_i], dtype=mx.int32)
                    chunk_output = self._encode_frame(
                        input_values_i,
                        input_lengths_i,
                        n_quantizers=num_quantizers,
                        caches=caches,
                    )
                    if chunk_output.audio_codes is None or chunk_output.audio_codes_lengths is None:
                        raise RuntimeError("Internal error: _encode_frame returned empty audio codes.")
                    if chunk_output.encoder_hidden_states is None:
                        raise RuntimeError(
                            "Internal error: _encode_frame returned empty hidden states."
                        )
                    length_i = int(chunk_output.audio_codes_lengths[0])
                    codes_chunks.append(chunk_output.audio_codes[:, :, :length_i])
                    hidden_chunks.append(chunk_output.encoder_hidden_states[:, :, :length_i])

                audio_codes = mx.concatenate(codes_chunks, axis=-1)
                hidden_states = mx.concatenate(hidden_chunks, axis=-1)
                output = MossAudioTokenizerEncoderOutput(
                    audio_codes=audio_codes,
                    audio_codes_lengths=mx.array([audio_codes.shape[-1]], dtype=mx.int32),
                    encoder_hidden_states=hidden_states,
                )

        if not return_dict:
            if output.audio_codes is None or output.audio_codes_lengths is None:
                raise RuntimeError("encode() produced empty outputs.")
            return output.audio_codes, output.audio_codes_lengths
        return output

    def decode(
        self,
        audio_codes: mx.array,
        padding_mask: Optional[mx.array] = None,
        return_dict: bool = True,
        chunk_duration: Optional[float] = None,
        num_quantizers: Optional[int] = None,
    ) -> MossAudioTokenizerDecoderOutput | tuple[mx.array]:
        audio_codes = self._normalize_decode_audio_codes(
            audio_codes,
            num_quantizers=num_quantizers,
        ).astype(mx.int32)

        _, batch_size, time_steps = audio_codes.shape
        if padding_mask is not None:
            codes_lengths = mx.sum(padding_mask.astype(mx.int32), axis=-1).astype(mx.int32)
        else:
            codes_lengths = mx.full((batch_size,), time_steps, dtype=mx.int32)

        if chunk_duration is None:
            output = self._decode_frame(audio_codes, codes_lengths)
        else:
            if chunk_duration <= 0:
                raise ValueError("chunk_duration must be > 0 when provided.")
            if chunk_duration > self.causal_transformer_context_duration:
                raise ValueError(
                    "chunk_duration must be <= "
                    f"{self.causal_transformer_context_duration}, got {chunk_duration}."
                )
            if batch_size != 1:
                raise ValueError(
                    "Streaming decode currently only supports batch_size=1."
                )

            chunk_length = int(round(chunk_duration * self.sampling_rate))
            if chunk_length <= 0:
                raise ValueError(
                    "chunk_duration is too small and results in chunk_length <= 0."
                )
            if chunk_length % self.downsample_rate != 0:
                raise ValueError(
                    "chunk_duration * sampling_rate must be divisible by downsample_rate. "
                    f"Got chunk_length={chunk_length}, downsample_rate={self.downsample_rate}."
                )
            chunk_frame_length = chunk_length // self.downsample_rate

            codes_length = int(codes_lengths[0])
            if codes_length <= chunk_frame_length:
                output = self._decode_frame(audio_codes[..., :codes_length], codes_lengths)
            else:
                caches = self._build_caches(self.decoder)
                wav_chunks = []
                for start_idx in range(0, codes_length, chunk_frame_length):
                    codes_length_i = min(chunk_frame_length, codes_length - start_idx)
                    if codes_length_i <= 0:
                        break
                    codes_i = audio_codes[:, :, start_idx : start_idx + codes_length_i]
                    codes_lengths_i = mx.array([codes_length_i], dtype=mx.int32)
                    chunk_output = self._decode_frame(codes_i, codes_lengths_i, caches=caches)
                    if chunk_output.audio is None or chunk_output.audio_lengths is None:
                        raise RuntimeError("Internal error: _decode_frame returned empty audio.")
                    wav_chunk = chunk_output.audio[:, :, : int(chunk_output.audio_lengths[0])]
                    mx.eval(wav_chunk)
                    wav_chunks.append(wav_chunk)
                    mx.clear_cache()
                wav = mx.concatenate(wav_chunks, axis=-1)
                output = MossAudioTokenizerDecoderOutput(
                    audio=wav,
                    audio_lengths=mx.array([wav.shape[-1]], dtype=mx.int32),
                )

        if not return_dict:
            if output.audio is None:
                raise RuntimeError("decode() produced empty audio.")
            return (output.audio,)
        return output

    def batch_encode(
        self,
        wav_list: list[mx.array],
        num_quantizers: Optional[int] = None,
    ) -> MossAudioTokenizerEncoderOutput:
        if not wav_list:
            raise ValueError("wav_list must contain at least one waveform.")

        normalized = []
        for wav in wav_list:
            if wav.ndim == 2:
                if wav.shape[0] != 1:
                    raise ValueError(
                        "Expected 2D waveform shape (1, T), got shape="
                        f"{wav.shape}."
                    )
                wav = wav.squeeze(0)
            if wav.ndim != 1:
                raise ValueError(
                    f"Each waveform in wav_list must be 1D or (1, T), got {wav.shape}."
                )
            normalized.append(wav)

        batch_size = len(normalized)
        max_length = max(int(w.shape[-1]) for w in normalized)
        input_values = mx.zeros((batch_size, 1, max_length), dtype=mx.float32)
        input_lengths = mx.zeros((batch_size,), dtype=mx.int32)

        for idx, wav in enumerate(normalized):
            length_i = int(wav.shape[-1])
            input_values[idx, 0, :length_i] = wav.astype(mx.float32)
            input_lengths[idx] = length_i

        return self._encode_frame(
            input_values,
            input_lengths,
            n_quantizers=num_quantizers,
        )

    def batch_decode(
        self,
        codes_list: list[mx.array],
        num_quantizers: Optional[int] = None,
    ) -> MossAudioTokenizerDecoderOutput:
        if not codes_list:
            raise ValueError("codes_list must contain at least one code tensor.")

        normalized = [
            self._normalize_decode_audio_codes(c, num_quantizers=num_quantizers).squeeze(1)
            for c in codes_list
        ]
        target_quantizers = int(normalized[0].shape[0])
        if any(int(c.shape[0]) != target_quantizers for c in normalized):
            raise ValueError(
                "All elements in codes_list must resolve to the same quantizer count."
            )

        max_length = max(int(c.shape[-1]) for c in normalized)
        batch_size = len(normalized)
        audio_codes = mx.zeros(
            (target_quantizers, batch_size, max_length),
            dtype=mx.int32,
        )
        audio_codes_lengths = mx.zeros((batch_size,), dtype=mx.int32)

        for idx, codes in enumerate(normalized):
            time_steps = int(codes.shape[-1])
            audio_codes[:, idx, :time_steps] = codes
            audio_codes_lengths[idx] = time_steps

        return self._decode_frame(audio_codes, audio_codes_lengths)

    def streaming_decode(
        self,
        audio_codes: mx.array,
        *,
        chunk_tokens: int = 100,
        num_quantizers: Optional[int] = None,
    ):
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be > 0.")

        audio_codes = self._normalize_decode_audio_codes(
            audio_codes,
            num_quantizers=num_quantizers,
        ).astype(mx.int32)

        _, batch_size, total_tokens = audio_codes.shape
        if batch_size != 1:
            raise ValueError("streaming_decode currently only supports batch_size=1.")

        caches = self._build_caches(self.decoder)
        for start_idx in range(0, total_tokens, chunk_tokens):
            end_idx = min(start_idx + chunk_tokens, total_tokens)
            codes_chunk = audio_codes[:, :, start_idx:end_idx]
            chunk_len = int(codes_chunk.shape[-1])
            output = self._decode_frame(
                codes_chunk,
                mx.array([chunk_len], dtype=mx.int32),
                caches=caches,
            )
            if output.audio is None or output.audio_lengths is None:
                raise RuntimeError("Internal error: _decode_frame returned empty audio.")
            wav_chunk = output.audio[:, :, : int(output.audio_lengths[0])]
            mx.eval(wav_chunk)
            yield wav_chunk
            mx.clear_cache()

    def __call__(
        self,
        input_values: Optional[mx.array] = None,
        padding_mask: Optional[mx.array] = None,
        audio_codes: Optional[mx.array] = None,
        num_quantizers: Optional[int] = None,
        return_dict: bool = True,
    ) -> MossAudioTokenizerOutput | tuple[Optional[mx.array], Optional[mx.array], Optional[mx.array]]:
        output_audio_codes = None
        output_audio_codes_lengths = None
        output_audio = None
        output_audio_lengths = None
        decoded_from_encoded_codes = False

        if input_values is not None:
            encode_output = self.encode(
                input_values,
                padding_mask=padding_mask,
                num_quantizers=num_quantizers,
                return_dict=True,
            )
            if not isinstance(encode_output, MossAudioTokenizerEncoderOutput):
                raise RuntimeError("Internal error: encode() returned unexpected output type.")
            output_audio_codes = encode_output.audio_codes
            output_audio_codes_lengths = encode_output.audio_codes_lengths

            if audio_codes is None:
                audio_codes = output_audio_codes
                decoded_from_encoded_codes = True

        if audio_codes is not None:
            if decoded_from_encoded_codes and output_audio_codes_lengths is not None:
                decode_output = self._decode_frame(audio_codes, output_audio_codes_lengths)
            else:
                decode_output = self.decode(
                    audio_codes,
                    padding_mask=padding_mask,
                    return_dict=True,
                    num_quantizers=num_quantizers,
                )
                if not isinstance(decode_output, MossAudioTokenizerDecoderOutput):
                    raise RuntimeError(
                        "Internal error: decode() returned unexpected output type."
                    )
            output_audio = decode_output.audio
            output_audio_lengths = decode_output.audio_lengths

        if not return_dict:
            return output_audio_codes, output_audio, output_audio_lengths

        return MossAudioTokenizerOutput(
            audio=output_audio,
            audio_lengths=output_audio_lengths,
            audio_codes=output_audio_codes,
            audio_codes_lengths=output_audio_codes_lengths,
        )

    def model_quant_predicate(self, path: str, module) -> bool:
        # Protect codebook embeddings from quantization.
        if isinstance(module, nn.Embedding):
            return False
        return True

    def _sanitize_chunk(
        self,
        weights: Dict[str, mx.array],
        pending_weight_norm: Optional[Dict[str, Dict[str, mx.array]]] = None,
    ) -> tuple[Dict[str, mx.array], Dict[str, Dict[str, mx.array]]]:
        pending = {} if pending_weight_norm is None else dict(pending_weight_norm)
        sanitized: Dict[str, mx.array] = {}
        current_shapes = {
            name: tuple(value.shape) for name, value in tree_flatten(self.parameters())
        }

        for key, value in weights.items():
            if key.endswith(".parametrizations.weight.original0"):
                base = key[: -len(".parametrizations.weight.original0")]
                entry = pending.setdefault(base, {})
                entry["g"] = value
                continue
            if key.endswith(".parametrizations.weight.original1"):
                base = key[: -len(".parametrizations.weight.original1")]
                entry = pending.setdefault(base, {})
                entry["v"] = value
                continue

            new_key = key
            if value.ndim == 3 and new_key in current_shapes:
                if tuple(value.shape) != current_shapes[new_key]:
                    value = value.swapaxes(1, 2)
            sanitized[new_key] = value

        resolved = []
        for base, parts in pending.items():
            if "g" not in parts or "v" not in parts:
                continue
            g = parts["g"].astype(mx.float32)
            v = parts["v"].astype(mx.float32)
            reduce_axes = tuple(range(1, v.ndim))
            norm = mx.sqrt(mx.sum(v**2, axis=reduce_axes, keepdims=True) + 1e-12)
            merged = (g * v / norm).astype(parts["v"].dtype)
            target_key = f"{base}.weight"
            if merged.ndim == 3 and target_key in current_shapes:
                if tuple(merged.shape) != current_shapes[target_key]:
                    merged = merged.swapaxes(1, 2)
            sanitized[target_key] = merged
            resolved.append(base)

        for base in resolved:
            pending.pop(base, None)

        return sanitized, pending

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized, pending = self._sanitize_chunk(weights, pending_weight_norm=None)
        if pending:
            missing = sorted(pending.keys())
            raise ValueError(
                "Missing weight-norm pairs while sanitizing weights for keys: "
                f"{missing}"
            )
        return sanitized

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: Union[str, Path],
        *,
        strict: bool = True,
    ) -> "MossAudioTokenizer":
        model_path = Path(path_or_repo)
        if not model_path.exists():
            model_path = Path(
                snapshot_download(
                    str(path_or_repo),
                    allow_patterns=["*.json", "*.safetensors"],
                )
            )

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json at {model_path}")
        config = load_moss_audio_tokenizer_config(config_path)
        model = cls(config)

        weight_files = sorted(model_path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No *.safetensors files found at {model_path}")

        loaded_keys = set()
        pending: Dict[str, Dict[str, mx.array]] = {}
        for weight_file in weight_files:
            shard_weights = mx.load(weight_file.as_posix(), format="safetensors")
            sanitized, pending = model._sanitize_chunk(
                shard_weights, pending_weight_norm=pending
            )
            if sanitized:
                model.load_weights(list(sanitized.items()), strict=False)
                loaded_keys.update(sanitized.keys())

        if pending:
            unresolved = sorted(pending.keys())
            raise ValueError(
                "Unresolved weight-norm parameter groups across shards: "
                f"{unresolved}"
            )

        if strict:
            expected_keys = {name for name, _ in tree_flatten(model.parameters())}
            missing = sorted(expected_keys - loaded_keys)
            if missing:
                raise ValueError(
                    "Strict load failed: missing parameter weights for keys: "
                    f"{missing[:20]}"
                    + ("..." if len(missing) > 20 else "")
                )

        mx.eval(model.parameters())
        return model

    def save_config(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.config.to_dict(), handle, indent=2, sort_keys=True)
