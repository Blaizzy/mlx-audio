import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.models.glmasr.glmasr import WhisperEncoderLayer
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import TextModel

from .config import ModelConfig

AUDIO_PAD_TOKEN = "<|audio_pad|>"
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
WHISPER_ENCODER_STRIDE = 2
DEFAULT_PROMPT = (
    "Transcribe the audio into text. Start each segment with the start "
    "timestamp and speaker label ([S01], [S02], [S03], ...), write the "
    "corresponding spoken content, and end each segment with the ending "
    "timestamp to clearly mark the segment range."
)
TRANSCRIPT_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\]"
    r"(?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
    re.DOTALL,
)


@dataclass
class StreamingResult:
    text: str
    is_final: bool
    start_time: float
    end_time: float
    language: str = "en"
    prompt_tokens: int = 0
    generation_tokens: int = 0


class VQAdaptor(nn.Module):
    """MOSS audio-text adaptor: Linear -> SiLU -> Linear -> LayerNorm."""

    def __init__(self, input_dim: int, hidden_size: int, norm_eps: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=norm_eps),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers(x)


class MossWhisperEncoder(nn.Module):
    """HF WhisperEncoder layout, with MLX Conv1d input convention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.conv1 = nn.Conv1d(config.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.max_source_positions, embed_dim)
        self.layers = [
            WhisperEncoderLayer(config, use_rope=False)
            for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(embed_dim)

    def __call__(self, input_features: mx.array) -> mx.array:
        hidden_states = nn.gelu(self.conv1(input_features))
        hidden_states = nn.gelu(self.conv2(hidden_states))

        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states + self.embed_positions.weight[:seq_len]

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.layer_norm(hidden_states)


class MossBackbone(nn.Module):
    """Whisper encoder + Qwen3 decoder backbone with MOSS audio injection."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = TextModel(config.text_config)
        self.whisper_encoder = MossWhisperEncoder(config.audio_config)
        self.vq_adaptor = VQAdaptor(
            input_dim=config.adaptor_input_dim,
            hidden_size=config.text_config.hidden_size,
            norm_eps=config.text_config.rms_norm_eps,
        )

    @property
    def layers(self):
        return self.language_model.layers

    def time_merge(self, features: mx.array) -> mx.array:
        batch_size, seq_len, dim = features.shape
        merge_size = int(self.config.audio_merge_size)
        trim_len = (seq_len // merge_size) * merge_size
        return features[:, :trim_len, :].reshape(
            batch_size, trim_len // merge_size, dim * merge_size
        )

    def get_audio_features(
        self,
        input_features: mx.array,
        audio_feature_lengths: mx.array,
        audio_chunk_mapping: Optional[mx.array] = None,
    ) -> List[mx.array]:
        whisper_features = self.whisper_encoder(input_features)

        lengths = np.array(audio_feature_lengths).astype(np.int64).reshape(-1)
        if audio_chunk_mapping is None:
            mapping = np.zeros((input_features.shape[0],), dtype=np.int64)
        else:
            mapping = np.array(audio_chunk_mapping).astype(np.int64).reshape(-1)

        if len(lengths) != input_features.shape[0]:
            raise ValueError(
                "audio_feature_lengths must contain one length per input feature chunk."
            )
        if len(mapping) != input_features.shape[0]:
            raise ValueError(
                "audio_chunk_mapping must contain one sample index per input feature chunk."
            )

        num_audios = int(mapping.max()) + 1 if len(mapping) else 0
        per_audio_chunks: List[List[mx.array]] = [[] for _ in range(num_audios)]
        for chunk_idx, token_len in enumerate(lengths):
            sample_idx = int(mapping[chunk_idx])
            per_audio_chunks[sample_idx].append(
                whisper_features[chunk_idx : chunk_idx + 1, : int(token_len) * 4]
            )

        adapted = []
        for parts in per_audio_chunks:
            feat = mx.concatenate(parts, axis=1)
            merged = self.time_merge(feat)
            adapted.append(self.vq_adaptor(merged))
        return adapted

    def inject_audio_features(
        self,
        input_ids: mx.array,
        inputs_embeds: mx.array,
        input_features: Optional[mx.array],
        audio_feature_lengths: Optional[mx.array],
        audio_chunk_mapping: Optional[mx.array],
    ) -> mx.array:
        if input_features is None:
            return inputs_embeds
        if audio_feature_lengths is None:
            raise ValueError(
                "audio_feature_lengths must be provided with input_features."
            )

        audio_features = self.get_audio_features(
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
        )
        audio_embeds = mx.concatenate([f.squeeze(0) for f in audio_features], axis=0)
        audio_embeds = audio_embeds.astype(inputs_embeds.dtype)

        is_audio = input_ids == self.config.audio_token_id
        audio_count = int(np.array(is_audio).sum())
        if audio_count != audio_embeds.shape[0]:
            raise ValueError(
                "Audio features and audio tokens do not match: "
                f"tokens: {audio_count}, features: {audio_embeds.shape[0]}"
            )

        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        flat_embeds = inputs_embeds.reshape(-1, hidden_dim)
        audio_indices = np.where(np.array(is_audio).reshape(-1))[0]
        flat_embeds[mx.array(audio_indices, dtype=mx.uint32)] = audio_embeds
        return flat_embeds.reshape(batch_size, seq_len, hidden_dim)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
        input_features: Optional[mx.array] = None,
        audio_feature_lengths: Optional[mx.array] = None,
        audio_chunk_mapping: Optional[mx.array] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        if cache is None or cache[0] is None or cache[0].offset == 0:
            inputs_embeds = self.inject_audio_features(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                input_features=input_features,
                audio_feature_lengths=audio_feature_lengths,
                audio_chunk_mapping=audio_chunk_mapping,
            )

        return self.language_model(inputs_embeds=inputs_embeds, cache=cache)


class Model(nn.Module):
    """MLX implementation of MOSS-Transcribe-Diarize speech-to-text model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = MossBackbone(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = (
            None
            if config.text_config.tie_word_embeddings
            else nn.Linear(config.text_config.hidden_size, self.vocab_size, bias=False)
        )
        self._tokenizer = None
        self._feature_extractor = None
        self.audio_tokens_per_second = 12.5
        self.time_marker_every_seconds = 5
        self.enable_time_marker = True
        self._digit_token_ids = None

    @property
    def layers(self):
        return self.model.layers

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def make_cache(self) -> List[Any]:
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.config.text_config.num_hidden_layers)]

    def __call__(
        self,
        input_ids: mx.array,
        input_embeddings: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        audio_feature_lengths: Optional[mx.array] = None,
        audio_chunk_mapping: Optional[mx.array] = None,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=input_embeddings,
            cache=cache,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
        )
        if self.lm_head is None:
            return self.model.language_model.embed_tokens.as_linear(hidden_states)
        return self.lm_head(hidden_states)

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        return not (
            p.startswith("model.whisper_encoder") or p.startswith("model.vq_adaptor")
        )

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}
        already_converted = any("scales" in k for k in weights)

        for k, v in weights.items():
            if k == "lm_head.weight":
                continue

            new_key = k
            if new_key.startswith("model.vq_adwaptor."):
                new_key = new_key.replace("model.vq_adwaptor.", "model.vq_adaptor.", 1)
            if new_key.startswith("model.vq_adaptor.layers."):
                parts = new_key.split(".")
                if len(parts) > 4 and parts[2] == "layers" and parts[3].isdigit():
                    new_key = ".".join(
                        ["model", "vq_adaptor", "layers", "layers", *parts[3:]]
                    )
            if new_key.startswith("model.vq_adaptor.layers.layers.layers."):
                new_key = new_key.replace(
                    "model.vq_adaptor.layers.layers.layers.",
                    "model.vq_adaptor.layers.layers.",
                    1,
                )

            if (
                not already_converted
                and new_key.startswith("model.whisper_encoder.")
                and "conv" in new_key
                and new_key.endswith(".weight")
                and len(v.shape) == 3
            ):
                v = v.transpose(0, 2, 1)

            sanitized[new_key] = v
        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        import transformers
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            model._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            model._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                str(model_path)
            )
        finally:
            transformers.logging.set_verbosity(prev)

        processor_config_path = model_path / "processor_config.json"
        if processor_config_path.exists():
            processor_config = json.loads(processor_config_path.read_text())
            model.audio_tokens_per_second = float(
                processor_config.get(
                    "audio_tokens_per_second", model.audio_tokens_per_second
                )
            )
            model.time_marker_every_seconds = int(
                processor_config.get(
                    "time_marker_every_seconds", model.time_marker_every_seconds
                )
            )
            model.enable_time_marker = bool(
                processor_config.get("enable_time_marker", model.enable_time_marker)
            )

        digit_token_ids = {}
        for digit in "0123456789":
            ids = model._tokenizer.encode(digit, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Digit {digit!r} is not a single token: {ids}")
            digit_token_ids[digit] = int(ids[0])
        model._digit_token_ids = digit_token_ids

        return model

    def _audio_to_numpy(self, audio: Union[str, mx.array, np.ndarray]) -> np.ndarray:
        from mlx_audio.stt.utils import load_audio

        if isinstance(audio, str):
            audio = load_audio(audio, sr=self.sample_rate)
        if isinstance(audio, mx.array):
            audio = np.array(audio)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.squeeze(audio)
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio, got shape {audio.shape}.")
        if audio.shape[0] == 0:
            raise ValueError("Audio must contain at least one sample.")
        return audio

    @staticmethod
    def _pad_or_trim_audio(audio: np.ndarray, length: int) -> np.ndarray:
        if audio.shape[0] > length:
            audio = audio[:length]
        elif audio.shape[0] < length:
            audio = np.pad(audio, (0, length - audio.shape[0]))
        return audio.astype(np.float32, copy=False)

    def _compute_audio_token_length(self, num_samples: int) -> int:
        hop_length = int(self._feature_extractor.hop_length)
        stride = hop_length * WHISPER_ENCODER_STRIDE * int(self.config.audio_merge_size)
        return (int(num_samples) - 1) // stride + 1

    def _preprocess_audio(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
    ) -> Tuple[mx.array, mx.array, mx.array, List[int], float]:
        if self._feature_extractor is None:
            raise RuntimeError("Feature extractor is not initialized.")

        audios = audio if isinstance(audio, list) else [audio]
        feature_batches = []
        feature_lengths: List[int] = []
        chunk_mapping: List[int] = []
        duration = 0.0

        for audio_idx, item in enumerate(audios):
            wav = self._audio_to_numpy(item)
            duration += float(wav.shape[0]) / float(self.sample_rate)
            n_samples = int(self._feature_extractor.n_samples)
            chunks = []
            for start in range(0, wav.shape[0], n_samples):
                chunk = wav[start : start + n_samples]
                feature_lengths.append(self._compute_audio_token_length(chunk.shape[0]))
                chunks.append(self._pad_or_trim_audio(chunk, n_samples))
                chunk_mapping.append(audio_idx)

            features = self._feature_extractor(
                chunks,
                sampling_rate=self.sample_rate,
                padding="max_length",
                return_tensors="np",
            )["input_features"]
            feature_batches.append(features)

        input_features = mx.array(
            np.concatenate(feature_batches, axis=0).transpose(0, 2, 1)
        )
        return (
            input_features.astype(self.model.whisper_encoder.conv1.weight.dtype),
            mx.array(feature_lengths, dtype=mx.int32),
            mx.array(chunk_mapping, dtype=mx.int32),
            feature_lengths,
            duration,
        )

    def _audio_span_ids(self, audio_seq_len: int) -> List[int]:
        audio_seq_len = int(audio_seq_len)
        if (
            not self.enable_time_marker
            or audio_seq_len <= 0
            or self.time_marker_every_seconds <= 0
        ):
            return [self.config.audio_token_id] * max(audio_seq_len, 0)

        tokens_per_marker = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )
        if tokens_per_marker <= 0:
            return [self.config.audio_token_id] * audio_seq_len

        if self._digit_token_ids is None:
            raise RuntimeError("Digit token ids are not initialized.")

        duration = audio_seq_len / float(self.audio_tokens_per_second)
        output, consumed = [], 0
        for sec in range(
            self.time_marker_every_seconds,
            int(duration) + 1,
            self.time_marker_every_seconds,
        ):
            pos = (sec // self.time_marker_every_seconds) * tokens_per_marker
            segment_len = pos - consumed
            if segment_len > 0:
                output.extend([self.config.audio_token_id] * segment_len)
                consumed += segment_len
            output.extend(self._digit_token_ids[digit] for digit in str(sec))

        remainder = audio_seq_len - consumed
        if remainder > 0:
            output.extend([self.config.audio_token_id] * remainder)
        return output

    def _build_prompt(
        self,
        audio_token_count: int,
        prompt: Optional[str] = None,
    ) -> mx.array:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")

        if prompt is None:
            prompt = DEFAULT_PROMPT

        audio_token = getattr(self._tokenizer, "audio_token", AUDIO_PAD_TOKEN)
        audio_start = getattr(self._tokenizer, "audio_start_token", AUDIO_START_TOKEN)
        audio_end = getattr(self._tokenizer, "audio_end_token", AUDIO_END_TOKEN)

        if audio_token in prompt:
            rendered = prompt
        elif hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": ""},
                        {"type": "text", "text": prompt.strip() or DEFAULT_PROMPT},
                    ],
                }
            ]
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = (
                f"<|im_start|>user\n"
                f"{audio_start}{audio_token}{audio_end}\n"
                f"{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        if rendered.count(audio_token) != 1:
            raise ValueError(
                f"Expected exactly one {audio_token!r} token in the prompt."
            )

        before_audio, after_audio = rendered.split(audio_token, maxsplit=1)
        input_ids = (
            self._tokenizer.encode(before_audio, add_special_tokens=False)
            + self._audio_span_ids(audio_token_count)
            + self._tokenizer.encode(after_audio, add_special_tokens=False)
        )
        return mx.array([input_ids], dtype=mx.int32)

    def _build_inputs_embeds(
        self,
        input_ids: mx.array,
        input_features: mx.array,
        audio_feature_lengths: mx.array,
        audio_chunk_mapping: mx.array,
    ) -> mx.array:
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        return self.model.inject_audio_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
        )

    def _prepare_generation_inputs(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
        prompt: Optional[str],
    ) -> Tuple[mx.array, mx.array, int, float]:
        input_features, audio_lengths, chunk_mapping, feature_lengths, duration = (
            self._preprocess_audio(audio)
        )
        audio_token_count = int(sum(feature_lengths))
        input_ids = self._build_prompt(audio_token_count, prompt)
        inputs_embeds = self._build_inputs_embeds(
            input_ids=input_ids,
            input_features=input_features,
            audio_feature_lengths=audio_lengths,
            audio_chunk_mapping=chunk_mapping,
        )[0]
        mx.eval(inputs_embeds)
        prompt_ids = input_ids[0]
        return prompt_ids, inputs_embeds, int(prompt_ids.shape[0]), duration

    @staticmethod
    def _parse_segments(text: str, fallback_end: float) -> List[dict]:
        segments = []
        for match in TRANSCRIPT_SEGMENT_RE.finditer(text):
            start = float(match.group("start"))
            end = float(match.group("end"))
            if end < start:
                continue
            segment_text = match.group("text").strip()
            speaker = match.group("speaker")
            if not segment_text:
                continue
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": f"[{speaker}] {segment_text}",
                    "speaker_id": speaker,
                }
            )
        if segments:
            return segments
        return [{"start": 0.0, "end": max(fallback_end, 0.0), "text": text}]

    def _eos_token_ids(self) -> set[int]:
        eos_token_id = self._tokenizer.eos_token_id if self._tokenizer else None
        eos_token_ids = {eos_token_id} if isinstance(eos_token_id, int) else set()
        eos_token_ids.update({151643, 151645})
        return eos_token_ids

    def stream_generate(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
        *,
        max_tokens: int = 2048,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[List[Callable]] = None,
        prompt: Optional[str] = None,
        prefill_step_size: int = 4096,
        verbose: bool = False,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        from mlx_lm.generate import generate_step

        prompt_ids, inputs_embeds, _, _ = self._prepare_generation_inputs(audio, prompt)
        eos_token_ids = self._eos_token_ids()

        for token, logprobs in generate_step(
            prompt=prompt_ids,
            input_embeddings=inputs_embeds,
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=prefill_step_size,
        ):
            if int(token) in eos_token_ids:
                break
            yield token, logprobs

    def _stream_transcribe(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
        *,
        max_tokens: int,
        sampler: Optional[Callable[[mx.array], mx.array]],
        logits_processors: Optional[List[Callable]],
        prompt: Optional[str],
        prefill_step_size: int,
        verbose: bool,
    ) -> Generator[StreamingResult, None, None]:
        gen_tokens = 0
        for token, _ in self.stream_generate(
            audio,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt=prompt,
            prefill_step_size=prefill_step_size,
            verbose=verbose,
        ):
            gen_tokens += 1
            text = self._tokenizer.decode([int(token)], skip_special_tokens=True)
            yield StreamingResult(
                text=text,
                is_final=False,
                start_time=0.0,
                end_time=0.0,
                generation_tokens=gen_tokens,
            )

        yield StreamingResult(
            text="",
            is_final=True,
            start_time=0.0,
            end_time=0.0,
            generation_tokens=gen_tokens,
        )

    def generate(
        self,
        audio: Union[str, mx.array, np.ndarray, List[Union[str, mx.array, np.ndarray]]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 100,
        prompt: Optional[str] = None,
        prefill_step_size: int = 4096,
        verbose: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> Union[STTOutput, Generator[StreamingResult, None, None]]:
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        sampler = make_sampler(temperature, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        if stream:
            return self._stream_transcribe(
                audio,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt=prompt,
                prefill_step_size=prefill_step_size,
                verbose=verbose,
            )

        start_time = time.time()
        prefill_start = time.time()
        prompt_ids, inputs_embeds, prompt_tokens, duration = (
            self._prepare_generation_inputs(audio, prompt)
        )
        prefill_time = time.time() - prefill_start
        eos_token_ids = self._eos_token_ids()

        generated_tokens = []
        gen_start = time.time()
        from mlx_lm.generate import generate_step

        for token, _ in generate_step(
            prompt=prompt_ids,
            input_embeddings=inputs_embeds,
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=prefill_step_size,
        ):
            if int(token) in eos_token_ids:
                break
            generated_tokens.append(int(token))

        gen_time = time.time() - gen_start
        text = self._tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        elapsed = time.time() - start_time
        segments = self._parse_segments(text, duration)

        return STTOutput(
            text=text,
            segments=segments,
            prompt_tokens=prompt_tokens,
            generation_tokens=len(generated_tokens),
            total_tokens=prompt_tokens + len(generated_tokens),
            total_time=elapsed,
            prompt_tps=prompt_tokens / prefill_time if prefill_time > 0 else 0.0,
            generation_tps=(len(generated_tokens) / gen_time if gen_time > 0 else 0.0),
        )
