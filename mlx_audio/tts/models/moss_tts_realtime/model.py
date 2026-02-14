"""MLX runtime model for MOSS-TTS-Realtime."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.tts.models.moss_tts.backbone import MossTTSBackbone
from mlx_audio.tts.models.moss_tts.local_model import MossTTSMLP
from mlx_audio.tts.models.moss_tts.local_transformer import MossTTSLocalTransformer
from mlx_audio.tts.models.moss_tts.sampling import (
    ChannelSamplingConfig,
    resolve_channel_sampling_configs,
    sample_channel_token,
)

from .config import ModelConfig
from .inference import MossTTSRealtimeInference, RealtimeSession
from .processor import MossTTSRealtimeProcessor
from .request import RealtimeNormalizedRequest


def _format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class MossTTSRealtimeCore(nn.Module):
    """Realtime architecture core: global backbone + local RVQ decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding_list = [
            nn.Embedding(config.vocab_size, config.hidden_size),
            *[
                nn.Embedding(config.audio_vocab_size, config.hidden_size)
                for _ in range(config.rvq)
            ],
        ]

        self.backbone = MossTTSBackbone(config.language_config)
        self.local_transformer = MossTTSLocalTransformer(config.local_transformer_config())

        local_hidden_size = self.local_transformer.config.hidden_size
        self.speech_embedding_to_local_mlp = MossTTSMLP(
            input_size=config.hidden_size,
            hidden_size=config.local_config.intermediate_size,
            output_size=local_hidden_size,
        )
        self.local_to_speech_embedding_mlps = [
            MossTTSMLP(
                input_size=local_hidden_size,
                hidden_size=config.local_config.intermediate_size,
                output_size=config.hidden_size,
            )
            for _ in range(config.rvq)
        ]

        self.layer_norm_before_lm_heads = [
            nn.RMSNorm(config.hidden_size, eps=config.language_config.rms_norm_eps)
            for _ in range(config.rvq)
        ]
        self.lm_heads = [
            nn.Linear(config.hidden_size, config.audio_vocab_size, bias=False)
            for _ in range(config.rvq)
        ]

    def make_cache(self):
        return self.backbone.make_cache()

    def _prepare_multi_modal_embeddings(self, input_ids: mx.array) -> mx.array:
        if input_ids.ndim != 3:
            raise ValueError(
                f"Expected input_ids shape (batch, time, channels), got {input_ids.shape}"
            )

        expected_channels = self.config.channels
        if int(input_ids.shape[2]) != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels, got {input_ids.shape[2]}"
            )

        fused = self.embedding_list[0](input_ids[:, :, 0])
        for channel_idx in range(self.config.rvq):
            fused = fused + self.embedding_list[channel_idx + 1](
                input_ids[:, :, channel_idx + 1]
            )
        return fused

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        embeddings = self._prepare_multi_modal_embeddings(input_ids)
        return self.backbone(embeddings, cache=cache)

    def compute_next_audio_logits(
        self,
        global_hidden_state: mx.array,
        *,
        local_input_ids: Optional[mx.array] = None,
    ) -> List[mx.array]:
        """Return per-channel logits for one realtime audio frame."""

        batch_size = int(global_hidden_state.shape[0])
        local_hidden_size = self.local_transformer.config.hidden_size
        local_inputs = mx.zeros((batch_size, 0, local_hidden_size), dtype=global_hidden_state.dtype)
        current_input = self.speech_embedding_to_local_mlp(global_hidden_state)

        logits_per_channel: List[mx.array] = []
        for channel_idx in range(self.config.rvq):
            local_inputs = mx.concatenate([local_inputs, current_input[:, None, :]], axis=1)
            local_outputs = self.local_transformer(local_inputs)
            hidden_state = local_outputs[:, -1, :]
            hidden_state = self.local_to_speech_embedding_mlps[channel_idx](hidden_state)
            hidden_state = self.layer_norm_before_lm_heads[channel_idx](hidden_state)

            logits = self.lm_heads[channel_idx](hidden_state)
            if 0 <= self.config.audio_pad_token < logits.shape[-1]:
                token_ids = mx.arange(logits.shape[-1], dtype=mx.int32)
                pad_mask = token_ids == int(self.config.audio_pad_token)
                logits = mx.where(pad_mask[None, :], -mx.inf, logits)

            logits_per_channel.append(logits)

            if local_input_ids is not None:
                next_token = local_input_ids[:, channel_idx].astype(mx.int32)
            else:
                next_token = mx.argmax(logits, axis=-1).astype(mx.int32)

            current_input = self.embedding_list[channel_idx + 1](next_token)
            current_input = self.speech_embedding_to_local_mlp(current_input)

        return logits_per_channel

    def sample_next_audio_tokens(
        self,
        global_hidden_state: mx.array,
        *,
        generated_history: Optional[mx.array],
        channel_sampling: Sequence[ChannelSamplingConfig],
    ) -> mx.array:
        """Sample one RVQ frame from logits with per-channel policies."""

        if len(channel_sampling) != self.config.rvq:
            raise ValueError(
                f"Expected {self.config.rvq} channel configs, got {len(channel_sampling)}"
            )

        batch_size = int(global_hidden_state.shape[0])
        local_hidden_size = self.local_transformer.config.hidden_size
        local_inputs = mx.zeros((batch_size, 0, local_hidden_size), dtype=global_hidden_state.dtype)
        current_input = self.speech_embedding_to_local_mlp(global_hidden_state)

        sampled: List[mx.array] = []
        for channel_idx in range(self.config.rvq):
            local_inputs = mx.concatenate([local_inputs, current_input[:, None, :]], axis=1)
            local_outputs = self.local_transformer(local_inputs)
            hidden_state = local_outputs[:, -1, :]
            hidden_state = self.local_to_speech_embedding_mlps[channel_idx](hidden_state)
            hidden_state = self.layer_norm_before_lm_heads[channel_idx](hidden_state)

            logits = self.lm_heads[channel_idx](hidden_state)
            if 0 <= self.config.audio_pad_token < logits.shape[-1]:
                token_ids = mx.arange(logits.shape[-1], dtype=mx.int32)
                pad_mask = token_ids == int(self.config.audio_pad_token)
                logits = mx.where(pad_mask[None, :], -mx.inf, logits)

            previous_tokens = None
            if generated_history is not None:
                previous_tokens = generated_history[:, :, channel_idx]

            token = sample_channel_token(
                logits,
                channel_sampling[channel_idx],
                previous_tokens=previous_tokens,
            )
            sampled.append(token)

            current_input = self.embedding_list[channel_idx + 1](token.astype(mx.int32))
            current_input = self.speech_embedding_to_local_mlp(current_input)

        return mx.stack(sampled, axis=-1).astype(mx.int32)


class Model(nn.Module):
    """Loadable MLX model wrapper for MOSS-TTS-Realtime."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self._sample_rate = config.sampling_rate

        self.model = MossTTSRealtimeCore(config)
        self.tokenizer = None
        self.audio_tokenizer: Optional[MossAudioTokenizer] = None
        self.processor: Optional[MossTTSRealtimeProcessor] = None
        self.generation_config: Dict[str, Any] = {}

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def layers(self):
        return self.model.backbone.layers

    def make_cache(self):
        return self.model.make_cache()

    def model_quant_predicate(self, path: str, module) -> bool:
        if isinstance(module, nn.Embedding):
            return False
        skip_patterns = [
            "audio_tokenizer",
            "quantizer",
            "codebook",
            "model.embedding_list",
            "model.lm_heads",
            "model.layer_norm_before_lm_heads",
        ]
        return not any(pattern in path for pattern in skip_patterns)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized: Dict[str, mx.array] = {}
        for key, value in weights.items():
            if "num_batches_tracked" in key:
                continue

            if key.startswith("model.language_model.embed_tokens."):
                new_key = key.replace(
                    "model.language_model.embed_tokens.",
                    "model.embedding_list.",
                    1,
                )
            elif key.startswith("language_model.embed_tokens."):
                new_key = key.replace(
                    "language_model.embed_tokens.",
                    "model.embedding_list.",
                    1,
                )
            elif key.startswith("model.language_model."):
                new_key = key.replace("model.language_model.", "model.backbone.", 1)
            elif key.startswith("language_model."):
                new_key = key.replace("language_model.", "model.backbone.", 1)
            elif key.startswith("model.embed_tokens."):
                new_key = key.replace("model.embed_tokens.", "model.embedding_list.", 1)
            elif key.startswith("embed_tokens."):
                new_key = key.replace("embed_tokens.", "model.embedding_list.", 1)
            elif key.startswith("local_transformer.model."):
                new_key = key.replace(
                    "local_transformer.model.",
                    "model.local_transformer.",
                    1,
                )
            elif key.startswith("model.local_transformer.model."):
                new_key = key.replace(
                    "model.local_transformer.model.",
                    "model.local_transformer.",
                    1,
                )
            elif key.startswith("local_transformer.local_lm_heads."):
                new_key = key.replace(
                    "local_transformer.local_lm_heads.",
                    "model.lm_heads.",
                    1,
                )
            elif key.startswith("model.local_transformer.local_lm_heads."):
                new_key = key.replace(
                    "model.local_transformer.local_lm_heads.",
                    "model.lm_heads.",
                    1,
                )
            elif key.startswith("local_transformer.layer_norm_before_lm_heads."):
                new_key = key.replace(
                    "local_transformer.layer_norm_before_lm_heads.",
                    "model.layer_norm_before_lm_heads.",
                    1,
                )
            elif key.startswith("model.local_transformer.layer_norm_before_lm_heads."):
                new_key = key.replace(
                    "model.local_transformer.layer_norm_before_lm_heads.",
                    "model.layer_norm_before_lm_heads.",
                    1,
                )
            elif key.startswith("local_transformer."):
                new_key = key.replace("local_transformer.", "model.local_transformer.", 1)
            elif key.startswith("speech_embedding_to_local_mlp."):
                new_key = key.replace(
                    "speech_embedding_to_local_mlp.",
                    "model.speech_embedding_to_local_mlp.",
                    1,
                )
            elif key.startswith("local_to_speech_embedding_mlps."):
                new_key = key.replace(
                    "local_to_speech_embedding_mlps.",
                    "model.local_to_speech_embedding_mlps.",
                    1,
                )
            elif key.startswith("layer_norm_before_lm_heads."):
                new_key = key.replace(
                    "layer_norm_before_lm_heads.",
                    "model.layer_norm_before_lm_heads.",
                    1,
                )
            elif key.startswith("lm_heads."):
                new_key = key.replace("lm_heads.", "model.lm_heads.", 1)
            else:
                new_key = key

            sanitized[new_key] = value

        return sanitized

    def _ensure_runtime_ready(self):
        if self.processor is None or self.tokenizer is None:
            raise RuntimeError(
                "Realtime processor is not initialized. Ensure post_load_hook has run."
            )
        if self.processor.audio_tokenizer is None:
            raise RuntimeError(
                "Realtime codec tokenizer is not available."
            )

    def _build_generation_result(
        self,
        audio: mx.array,
        *,
        start_time: float,
        token_count: int,
        segment_idx: int,
        is_streaming_chunk: bool = False,
        is_final_chunk: bool = False,
    ) -> GenerationResult:
        samples = int(audio.shape[0])
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        audio_duration_seconds = samples / self.sample_rate if self.sample_rate > 0 else 0.0
        rtf = audio_duration_seconds / elapsed if elapsed > 0 else 0.0

        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=segment_idx,
            token_count=token_count,
            audio_duration=_format_duration(audio_duration_seconds),
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": token_count / elapsed,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": samples / elapsed,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=is_streaming_chunk,
            is_final_chunk=is_final_chunk,
        )

    def generate(
        self,
        text: Optional[str] = None,
        voice: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.6,
        top_k: int = 30,
        repetition_penalty: float = 1.1,
        max_tokens: int = 1200,
        verbose: bool = False,
        ref_audio: Optional[Any] = None,
        stream: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        del voice, verbose
        self._ensure_runtime_ready()

        request = RealtimeNormalizedRequest.from_generate_kwargs(
            text=text,
            ref_audio=ref_audio,
            include_system_prompt=kwargs.pop("include_system_prompt", None),
            reset_cache=kwargs.pop("reset_cache", None),
            chunk_frames=kwargs.pop("chunk_frames", None),
            overlap_frames=kwargs.pop("overlap_frames", None),
            decode_chunk_duration=kwargs.pop("decode_chunk_duration", 0.32),
            max_pending_frames=kwargs.pop("max_pending_frames", None),
        )

        decode_kwargs = dict(kwargs.pop("decode_kwargs", {}) or {})
        if request.decode_chunk_duration is not None:
            decode_kwargs.setdefault("chunk_duration", request.decode_chunk_duration)

        inferencer = MossTTSRealtimeInference(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            max_length=max_tokens,
        )
        session = RealtimeSession(
            inferencer=inferencer,
            processor=self.processor,
            chunk_frames=request.chunk_frames,
            overlap_frames=request.overlap_frames,
            decode_kwargs=decode_kwargs,
            max_pending_frames=request.max_pending_frames,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=kwargs.pop("do_sample", True),
            repetition_penalty=repetition_penalty,
        )

        prompt_audio_tokens = None
        if request.reference_audio is not None:
            prompt_audio_tokens = self.processor.encode_prompt_audio(request.reference_audio)

        session.reset_turn(
            user_text="",
            user_audio_tokens=prompt_audio_tokens,
            include_system_prompt=request.include_system_prompt,
            reset_cache=request.reset_cache,
        )

        text_token_ids = self.processor.make_text_prefix(
            self.processor.tokens_from_text(request.text, add_special_tokens=False)
        )

        start_time = time.perf_counter()
        token_count = 0
        chunks: list[mx.array] = []

        if text_token_ids:
            token_count += len(text_token_ids)
            chunks.extend(session.push_text_tokens(text_token_ids))
        chunks.extend(session.end_text())
        chunks.extend(session.drain(max_steps=max_tokens))

        session.close()

        if not chunks:
            raise RuntimeError("No realtime audio was generated")

        if stream:
            for idx, chunk in enumerate(chunks):
                yield self._build_generation_result(
                    chunk,
                    start_time=start_time,
                    token_count=token_count,
                    segment_idx=idx,
                    is_streaming_chunk=True,
                    is_final_chunk=idx == len(chunks) - 1,
                )
            return

        merged = mx.concatenate(chunks, axis=0)
        yield self._build_generation_result(
            merged,
            start_time=start_time,
            token_count=token_count,
            segment_idx=0,
            is_streaming_chunk=False,
            is_final_chunk=True,
        )

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for MOSS-TTS-Realtime tokenizer"
            ) from exc

        model.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )

        codec_path_candidates = [
            model_path / "audio_tokenizer",
            model_path / "moss_audio_tokenizer",
            model_path / "codec",
        ]
        for candidate in codec_path_candidates:
            if candidate.exists():
                model.audio_tokenizer = MossAudioTokenizer.from_pretrained(candidate)
                break

        if model.audio_tokenizer is None:
            try:
                model.audio_tokenizer = MossAudioTokenizer.from_pretrained(
                    "OpenMOSS-Team/MOSS-Audio-Tokenizer"
                )
            except Exception:
                model.audio_tokenizer = None

        model.processor = MossTTSRealtimeProcessor(
            tokenizer=model.tokenizer,
            audio_tokenizer=model.audio_tokenizer,
            model_config=model.config,
        )

        generation_config_path = model_path / "generation_config.json"
        if generation_config_path.exists():
            model.generation_config = json.loads(
                generation_config_path.read_text(encoding="utf-8")
            )

        return model


__all__ = ["Model", "ModelConfig", "MossTTSRealtimeCore"]
