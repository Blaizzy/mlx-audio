"""MLX runtime model for MOSS-TTS Local vertical slice (Phase 2)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.moss_audio_tokenizer import MossAudioTokenizer
from mlx_audio.tts.models.base import GenerationResult

from .config import ModelConfig
from .local_model import MossTTSLocalModel
from .processor import MossTTSProcessor, VALID_INPUT_TYPES
from .request import MossNormalizedRequest
from .sampling import resolve_channel_sampling_configs


def _format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class Model(nn.Module):
    """
    MOSS-TTS Local model.

    Delay architecture support is intentionally deferred to Phase 3.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self._sample_rate = config.sampling_rate

        if not config.is_local_variant:
            raise NotImplementedError(
                "Delay architecture (no local_num_layers in config) is planned for "
                "Phase 3. Phase 2 currently supports MOSS-TTS-Local only."
            )

        self.model = MossTTSLocalModel(config)
        self.tokenizer = None
        self.audio_tokenizer: Optional[MossAudioTokenizer] = None
        self.processor: Optional[MossTTSProcessor] = None
        self.generation_config: Dict[str, Union[int, float, bool]] = {}

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
        skip_patterns = (
            "audio_tokenizer",
            "quantizer",
            "codebook",
            "model.lm_heads",
            "model.layer_norm_before_lm_heads",
            "model.embedding_list",
        )
        return not any(pattern in path for pattern in skip_patterns)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized: Dict[str, mx.array] = {}
        for key, value in weights.items():
            if "num_batches_tracked" in key:
                continue

            if key.startswith("model.language_model."):
                new_key = key.replace("model.language_model.", "model.backbone.", 1)
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

            # `language_model.embed_tokens` exists in upstream checkpoints but is
            # unused in Local (input comes from `embedding_list` summation).
            if new_key == "model.backbone.embed_tokens.weight":
                continue

            sanitized[new_key] = value
        return sanitized

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
        audio_duration_seconds = samples / self.sample_rate
        rtf = audio_duration_seconds / elapsed

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

    def _decode_audio_rows(self, audio_rows: List[mx.array]) -> mx.array:
        if self.processor is None:
            raise RuntimeError("processor is not initialized")
        if len(audio_rows) == 0:
            raise RuntimeError("no audio rows available to decode")
        audio_codes = mx.stack(audio_rows, axis=0).astype(mx.int32)
        audio = self.processor.decode_audio_codes(audio_codes, chunk_duration=8.0)
        mx.eval(audio)
        return audio

    def _resolve_generation_request(
        self,
        *,
        text: Optional[str],
        ref_audio: Optional[Union[str, mx.array]],
        ref_text: Optional[str],
        instruct: Optional[str],
        **kwargs,
    ) -> MossNormalizedRequest:
        instruction = kwargs.pop("instruction", None)
        if ref_text is not None:
            transcript_context = f"Reference transcript: {ref_text}"
            instruction = (
                transcript_context
                if instruction is None and instruct is None
                else f"{instruction or instruct}\n{transcript_context}"
            )

        return MossNormalizedRequest.from_generate_kwargs(
            text=text,
            reference=kwargs.pop("reference", None),
            instruction=instruction,
            instruct=instruct,
            ref_audio=ref_audio,
            tokens=kwargs.pop("tokens", None),
            quality=kwargs.pop("quality", None),
            sound_event=kwargs.pop("sound_event", None),
            ambient_sound=kwargs.pop("ambient_sound", None),
            language=kwargs.pop("language", None),
        )

    def _ensure_processor_ready(self):
        if self.processor is None or self.tokenizer is None:
            raise RuntimeError(
                "MOSS processor is not initialized. Ensure tokenizer + codec are "
                "available (post_load_hook)."
            )
        if self.processor.audio_tokenizer is None:
            raise RuntimeError(
                "MOSS audio tokenizer is not available. Provide a checkpoint with "
                "embedded codec files or ensure "
                "`OpenMOSS-Team/MOSS-Audio-Tokenizer` is reachable."
            )

    def generate(
        self,
        text: Optional[str] = None,
        voice: Optional[str] = None,  # Unused, retained for generate() compatibility
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        max_tokens: int = 1200,
        verbose: bool = False,
        ref_audio: Optional[Union[str, mx.array]] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        del voice, verbose
        self._ensure_processor_ready()

        request = self._resolve_generation_request(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            **kwargs,
        )
        input_type = kwargs.pop("input_type", "text")
        if input_type not in VALID_INPUT_TYPES:
            raise ValueError(
                f"Unsupported input_type '{input_type}'. "
                f"Expected one of {sorted(VALID_INPUT_TYPES)}"
            )

        user_message = self.processor.build_user_message(
            **request.to_user_message_kwargs(),
            input_type=input_type,
        )
        batch = self.processor.prepare_generation_inputs(
            user_message,
            n_vq=self.config.n_vq,
            apply_chat_template=True,
        )
        input_ids = batch["input_ids"].astype(mx.int32)

        do_samples = kwargs.get("do_samples", None)
        layers = kwargs.get("layers", None)
        sampling_cfg = resolve_channel_sampling_configs(
            self.config.channels,
            default_temperature=temperature,
            default_top_p=top_p,
            default_top_k=top_k,
            default_repetition_penalty=repetition_penalty,
            do_samples=do_samples,
            layers=layers,
        )

        cache = self.model.make_cache()
        hidden_states = self.model(input_ids, cache=cache, n_vq_for_inference=self.config.n_vq)
        global_hidden = hidden_states[:, -1, :]

        chunk_rows = max(1, int(streaming_interval * 12.5))
        chunk_start_time = time.perf_counter()
        last_yielded_samples = 0
        generated_row_count = 0
        audio_rows: List[mx.array] = []

        for _ in range(max_tokens):
            next_tokens = self.model.sample_next_channels(
                global_hidden,
                input_ids,
                sampling_cfg,
                n_vq_for_inference=self.config.n_vq,
            )
            input_ids = mx.concatenate([input_ids, next_tokens[:, None, :]], axis=1)
            generated_row_count += 1

            text_token = int(next_tokens[0, 0])
            if text_token == self.config.audio_assistant_gen_slot_token_id:
                audio_rows.append(next_tokens[0, 1 : 1 + self.config.n_vq])

            if text_token == self.config.audio_end_token_id:
                break

            hidden_states = self.model(
                next_tokens[:, None, :],
                cache=cache,
                n_vq_for_inference=self.config.n_vq,
            )
            global_hidden = hidden_states[:, -1, :]

            if stream and len(audio_rows) > 0 and len(audio_rows) % chunk_rows == 0:
                decoded = self._decode_audio_rows(audio_rows)
                if decoded.shape[0] > last_yielded_samples:
                    new_audio = decoded[last_yielded_samples:]
                    last_yielded_samples = int(decoded.shape[0])
                    yield self._build_generation_result(
                        new_audio,
                        start_time=chunk_start_time,
                        token_count=generated_row_count,
                        segment_idx=0,
                        is_streaming_chunk=True,
                        is_final_chunk=False,
                    )
                    chunk_start_time = time.perf_counter()
                mx.clear_cache()

        if len(audio_rows) == 0:
            raise RuntimeError(
                "No audio rows were generated. Check prompt formatting/sampling "
                "configuration for this checkpoint."
            )

        decoded = self._decode_audio_rows(audio_rows)
        if stream:
            if decoded.shape[0] > last_yielded_samples:
                final_chunk = decoded[last_yielded_samples:]
            else:
                final_chunk = mx.zeros((0,), dtype=mx.float32)
            yield self._build_generation_result(
                final_chunk,
                start_time=chunk_start_time,
                token_count=generated_row_count,
                segment_idx=0,
                is_streaming_chunk=True,
                is_final_chunk=True,
            )
        else:
            yield self._build_generation_result(
                decoded,
                start_time=chunk_start_time,
                token_count=generated_row_count,
                segment_idx=0,
            )
        mx.clear_cache()

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - import failure is environment specific
            raise RuntimeError("transformers is required for MOSS-TTS tokenizer") from exc

        model.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

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

        model.processor = MossTTSProcessor(
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


__all__ = ["Model", "ModelConfig"]
