import argparse
import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import mlx.core as mx
import numpy as np
import sounddevice as sd
from mlx_lm.generate import generate as generate_text
from mlx_lm.utils import load as load_llm

from mlx_audio.sts.audio_player import AudioPlayer
from mlx_audio.tts.utils import load_model as load_tts

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class VoicePipelineConfig:
    """Configuration for the realtime Voxtral STT + Pocket TTS voice loop."""

    input_sample_rate: int = 16_000
    output_sample_rate: Optional[int] = None
    input_channels: int = 1
    frame_duration_ms: int = 32
    latency_profile: str = "balanced"

    stt_model: str = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    stt_transcription_delay_ms: Optional[int] = None
    stt_max_decode_tokens_per_step: int = 6
    stt_max_turn_tokens: int = 256
    stt_finalization_max_steps: int = 96

    vad_model: str = "mlx-community/silero-vad"
    vad_start_threshold: float = 0.35
    vad_stop_threshold: float = 0.2
    vad_start_frames: int = 1
    vad_end_silence_ms: int = 600
    vad_max_turn_seconds: float = 30.0
    preroll_ms: int = 250

    turn_model: str = "mlx-community/smart-turn-v3"
    turn_threshold: float = 0.5
    turn_max_incomplete_silence_ms: int = 1600

    response_model: str = "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"
    system_prompt: str = (
        "You are a helpful voice assistant. Respond in natural spoken sentences. Never use markdown, emoji, or lists."
    )

    tts_model: str = "mlx-community/pocket-tts"
    tts_voice: str = "cosette"
    tts_streaming_interval: Optional[float] = None
    tts_temperature: Optional[float] = None

    barge_in: bool = True
    min_barge_in_ms: int = 180
    ignore_playback_echo_ms: int = 450
    echo_delay_min_ms: int = 250
    echo_delay_max_ms: int = 500
    echo_correlation_step_ms: int = 32
    barge_in_min_transcript_chars: int = 2

    play_audio: bool = True
    queue_size: int = 128
    verbose: bool = False

    def __post_init__(self):
        profile = self.latency_profile.lower()
        if self.stt_transcription_delay_ms is None:
            self.stt_transcription_delay_ms = {
                "fast": 240,
                "balanced": 480,
                "quality": 960,
            }.get(profile, 480)

        if self.tts_streaming_interval is None:
            self.tts_streaming_interval = {
                "fast": 0.24,
                "balanced": 0.32,
                "quality": 0.48,
            }.get(profile, 0.32)


@dataclass
class SpeechFrameDecision:
    probability: float
    is_speech: bool
    speech_started: bool = False
    candidate_ended: bool = False


@dataclass
class EndpointDecision:
    complete: bool
    probability: float


class MLXWorkScheduler:
    """Run all MLX work on one worker thread and stream.

    Recent MLX streams are thread-local, so model load, cached state creation,
    and inference must stay on the same worker thread. A generic
    ``asyncio.to_thread`` call can hop between pool threads and later fail when
    evaluating arrays that reference a stream created in another thread.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-audio-sts"
        )
        self._stream = None
        self._initialized = False

    def _run(self, func: Callable[[], Any]) -> Any:
        if not self._initialized:
            self._stream = mx.new_stream(mx.gpu)
            mx.set_default_stream(self._stream)
            self._initialized = True
        with mx.stream(self._stream):
            return func()

    async def run(self, func: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run, func)

    def _synchronize(self) -> None:
        if self._stream is not None:
            mx.synchronize(self._stream)

    async def shutdown(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._synchronize)
        self._executor.shutdown(wait=True, cancel_futures=False)


class FixedSizeAudioChunker:
    def __init__(self, chunk_size: int):
        self.chunk_size = int(chunk_size)
        self._buffer = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return []
        self._buffer = np.concatenate([self._buffer, samples])

        chunks = []
        while self._buffer.shape[0] >= self.chunk_size:
            chunks.append(self._buffer[: self.chunk_size].copy())
            self._buffer = self._buffer[self.chunk_size :]
        return chunks


class PreRollBuffer:
    def __init__(self, max_samples: int):
        self.max_samples = max(0, int(max_samples))
        self._chunks: list[np.ndarray] = []
        self._samples = 0

    def append(self, samples: np.ndarray) -> None:
        if self.max_samples <= 0:
            return
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        self._chunks.append(samples.copy())
        self._samples += samples.size
        while self._samples > self.max_samples and self._chunks:
            extra = self._samples - self.max_samples
            first = self._chunks[0]
            if extra >= first.size:
                self._chunks.pop(0)
                self._samples -= first.size
            else:
                self._chunks[0] = first[extra:]
                self._samples -= extra
                break

    def get(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks).astype(np.float32, copy=False)

    def clear(self) -> None:
        self._chunks.clear()
        self._samples = 0


class SileroSpeechGate:
    """Stateful speech gate using Silero VAD probabilities and hysteresis."""

    def __init__(
        self,
        model,
        *,
        sample_rate: int = 16_000,
        start_threshold: float = 0.35,
        stop_threshold: float = 0.2,
        start_frames: int = 1,
        end_silence_ms: int = 600,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.start_threshold = float(start_threshold)
        self.stop_threshold = float(stop_threshold)
        self.start_frames = max(1, int(start_frames))
        chunk_size = self._chunk_size()
        self.end_silence_frames = max(
            1, int(round((end_silence_ms / 1000.0) * sample_rate / chunk_size))
        )
        self.chunker = FixedSizeAudioChunker(chunk_size)
        self.state = self._initial_state()
        self.in_speech = False
        self._speech_run = 0
        self._silence_run = 0

    def reset(self) -> None:
        self.chunker = FixedSizeAudioChunker(self._chunk_size())
        self.state = self._initial_state()
        self.in_speech = False
        self._speech_run = 0
        self._silence_run = 0

    def process(self, samples: np.ndarray) -> list[SpeechFrameDecision]:
        decisions = []
        for chunk in self.chunker.push(samples):
            probability = self._feed_probability(chunk)
            is_speech = probability >= (
                self.stop_threshold if self.in_speech else self.start_threshold
            )
            started = False
            ended = False

            if is_speech:
                self._speech_run += 1
                self._silence_run = 0
            else:
                self._speech_run = 0
                self._silence_run += 1

            if not self.in_speech and self._speech_run >= self.start_frames:
                self.in_speech = True
                started = True
                self._silence_run = 0

            if self.in_speech and self._silence_run >= self.end_silence_frames:
                self.in_speech = False
                ended = True
                self._speech_run = 0

            decisions.append(
                SpeechFrameDecision(
                    probability=probability,
                    is_speech=is_speech,
                    speech_started=started,
                    candidate_ended=ended,
                )
            )
        return decisions

    def _chunk_size(self) -> int:
        branch = getattr(getattr(self.model, "config", None), "branch_16k", None)
        return int(getattr(branch, "chunk_size", 512))

    def _initial_state(self):
        if hasattr(self.model, "initial_state"):
            return self.model.initial_state(sample_rate=self.sample_rate)
        return None

    def _feed_probability(self, chunk: np.ndarray) -> float:
        probability, self.state = self.model.feed(
            chunk, self.state, sample_rate=self.sample_rate
        )
        if hasattr(probability, "item"):
            return float(probability.item())
        return float(np.asarray(probability).reshape(-1)[0])


class SmartTurnEndpointDetector:
    def __init__(
        self,
        model,
        *,
        sample_rate: int = 16_000,
        threshold: Optional[float] = None,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.threshold = threshold

    def predict(self, audio: np.ndarray) -> EndpointDecision:
        result = self.model.predict_endpoint(
            audio, sample_rate=self.sample_rate, threshold=self.threshold
        )
        return EndpointDecision(
            complete=bool(result.prediction), probability=float(result.probability)
        )


class VoxtralRealtimeTranscriber:
    def __init__(
        self,
        model,
        *,
        transcription_delay_ms: int = 480,
        max_decode_tokens_per_step: int = 6,
        max_turn_tokens: int = 256,
        verbose: bool = False,
    ):
        self.model = model
        self.transcription_delay_ms = transcription_delay_ms
        self.max_decode_tokens_per_step = max_decode_tokens_per_step
        self.max_turn_tokens = max_turn_tokens
        self.verbose = verbose
        self.session = None
        self.text = ""
        self.last_finish_steps = 0
        self.last_finish_seconds = 0.0
        self.last_finish_hit_max_steps = False

    @property
    def active(self) -> bool:
        return self.session is not None and not getattr(self.session, "done", False)

    def start(self) -> None:
        self.session = self.model.create_streaming_session(
            max_tokens=self.max_turn_tokens,
            transcription_delay_ms=self.transcription_delay_ms,
        )
        self.text = ""
        self.last_finish_steps = 0
        self.last_finish_seconds = 0.0
        self.last_finish_hit_max_steps = False

    def feed(self, samples: np.ndarray) -> None:
        if self.session is None or getattr(self.session, "done", False):
            self.start()
        self.session.feed(np.asarray(samples, dtype=np.float32).reshape(-1))

    def step(self) -> list[str]:
        if self.session is None:
            return []
        deltas = self.session.step(max_decode_tokens=self.max_decode_tokens_per_step)
        for delta in deltas:
            self.text += delta
        return deltas

    def close(self) -> None:
        if self.session is not None:
            self.session.close()

    def reset(self) -> None:
        if self.session is not None:
            self.session.close()
        self.session = None
        self.text = ""
        self.last_finish_steps = 0
        self.last_finish_seconds = 0.0
        self.last_finish_hit_max_steps = False

    def finish(self, *, max_steps: int = 96) -> str:
        if self.session is None:
            return self.text.strip()
        start_time = time.monotonic()
        self.close()
        steps = 0
        while not getattr(self.session, "done", True) and steps < max_steps:
            self.step()
            steps += 1
        self.last_finish_steps = steps
        self.last_finish_seconds = time.monotonic() - start_time
        self.last_finish_hit_max_steps = steps >= max_steps and not getattr(
            self.session, "done", True
        )
        if self.verbose and self.last_finish_hit_max_steps:
            logger.warning(
                "event=voxtral_finalization_max_steps max_steps=%s chars=%s",
                max_steps,
                len(self.text.strip()),
            )
        final_text = self.text.strip()
        self.session = None
        return final_text


class LocalLLMResponseEngine:
    def __init__(self, model_name: str, *, system_prompt: str):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.llm = None
        self.tokenizer = None

    def load(self) -> None:
        self.llm, self.tokenizer = load_llm(self.model_name)

    def generate(self, transcript: str, context: Optional[list[dict]] = None) -> str:
        if self.llm is None or self.tokenizer is None:
            self.load()
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": transcript})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, add_generation_prompt=True
        )
        return generate_text(self.llm, self.tokenizer, prompt, verbose=False).strip()


class PocketTTSResponder:
    def __init__(
        self,
        model,
        *,
        voice: str = "cosette",
        streaming_interval: float = 0.32,
        temperature: Optional[float] = None,
    ):
        self.model = model
        self.voice = voice
        self.streaming_interval = streaming_interval
        self.temperature = temperature
        self.sample_rate = int(getattr(model, "sample_rate", 24_000) or 24_000)

    def create_generator(self, text: str) -> Iterable[Any]:
        kwargs: dict[str, Any] = {
            "text": text,
            "voice": self.voice,
            "stream": True,
            "streaming_interval": self.streaming_interval,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return iter(self.model.generate(**kwargs))


class AudioOutputStream:
    def __init__(self, *, sample_rate: int, enabled: bool = True, event_callback=None):
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.player = (
            AudioPlayer(sample_rate=sample_rate, event_callback=event_callback)
            if enabled
            else None
        )

    def queue_audio(self, audio) -> dict[str, Any]:
        samples = self._sample_count(audio)
        if self.player is None:
            return {
                "player_enabled": False,
                "sample_rate": self.sample_rate,
                "samples": samples,
                "duration_ms": samples / self.sample_rate * 1000.0,
            }
        was_playing = self.player.playing
        self.player.queue_audio(audio)
        buffered_samples = self.player.buffered_samples()
        status = {
            "player_enabled": True,
            "player_playing": self.player.playing,
            "player_started": not was_playing and self.player.playing,
            "sample_rate": self.sample_rate,
            "samples": samples,
            "duration_ms": samples / self.sample_rate * 1000.0,
            "buffered_samples": buffered_samples,
            "buffered_ms": buffered_samples / self.sample_rate * 1000.0,
        }
        if hasattr(self.player, "start_buffer_samples"):
            status["start_buffer_samples"] = self.player.start_buffer_samples()
            status["start_buffer_ms"] = self.player.start_buffer_seconds * 1000.0
        return status

    def start_playback_if_buffered(self, *, force=False) -> dict[str, Any]:
        if self.player is None:
            return {
                "player_enabled": False,
                "player_playing": False,
                "player_started": False,
                "sample_rate": self.sample_rate,
                "buffered_samples": 0,
                "buffered_ms": 0.0,
                "forced": force,
            }
        if hasattr(self.player, "start_if_buffered"):
            return self.player.start_if_buffered(force=force)
        return {
            "player_enabled": True,
            "player_playing": self.player.playing,
            "player_started": False,
            "sample_rate": self.sample_rate,
            "buffered_samples": self.player.buffered_samples(),
            "buffered_ms": self.player.buffered_samples() / self.sample_rate * 1000.0,
            "forced": force,
        }

    @staticmethod
    def _sample_count(audio) -> int:
        samples = np.asarray(audio)
        if samples.ndim == 0:
            return 1
        if samples.ndim == 1:
            return int(samples.shape[0])
        if samples.ndim == 2:
            if samples.shape[1] == 1:
                return int(samples.shape[0])
            if samples.shape[0] == 1:
                return int(samples.shape[1])
            if samples.shape[0] <= 8 and samples.shape[0] < samples.shape[1]:
                return int(samples.shape[1])
            return int(samples.shape[0])
        return int(samples.size)

    def flush(self) -> dict[str, Any]:
        if self.player is not None:
            return self.player.flush()
        return {
            "was_playing": False,
            "buffered_samples": 0,
            "buffered_ms": 0.0,
            "last_output_age_ms": None,
        }

    def playback_state(self, now_ms: Optional[float] = None) -> dict[str, Any]:
        if self.player is None:
            return {
                "player_enabled": False,
                "player_playing": False,
                "sample_rate": self.sample_rate,
                "buffered_samples": 0,
                "buffered_ms": 0.0,
                "last_output_callback_monotonic_ms": None,
                "last_output_age_ms": None,
            }
        return self.player.playback_state(now_ms=now_ms)

    def echo_correlation(
        self,
        samples: np.ndarray,
        *,
        input_sample_rate: int,
        input_end_ms: float,
        min_delay_ms: float,
        max_delay_ms: float,
        step_ms: float,
    ) -> dict[str, Any]:
        if self.player is None:
            return {"correlation": 0.0, "delay_ms": None, "method": None}
        return self.player.echo_correlation(
            samples,
            input_sample_rate=input_sample_rate,
            input_end_ms=input_end_ms,
            min_delay_ms=min_delay_ms,
            max_delay_ms=max_delay_ms,
            step_ms=step_ms,
        )

    def stop(self) -> None:
        if self.player is not None:
            self.player.stop()


class VoicePipeline:
    """Realtime voice assistant pipeline using Voxtral Realtime STT and Pocket TTS."""

    def __init__(
        self,
        config: Optional[VoicePipelineConfig] = None,
        *,
        speech_gate=None,
        endpoint_detector=None,
        transcriber: Optional[VoxtralRealtimeTranscriber] = None,
        response_engine: Optional[Any] = None,
        tts_responder: Optional[PocketTTSResponder] = None,
        audio_output: Optional[AudioOutputStream] = None,
    ):
        self.config = config or VoicePipelineConfig()
        self.speech_gate = speech_gate
        self.endpoint_detector = endpoint_detector
        self.transcriber = transcriber
        self.response_engine = response_engine
        self.tts_responder = tts_responder
        self.audio_output = audio_output

        self.input_audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(
            maxsize=self.config.queue_size
        )
        self.transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self.output_audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(
            maxsize=self.config.queue_size
        )
        self.mlx = MLXWorkScheduler()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self._current_tts_task: Optional[asyncio.Task] = None
        self._current_tts_cancel: Optional[asyncio.Event] = None
        self._current_response_text = ""
        self._conversation: list[dict] = []
        self._turn_audio: list[np.ndarray] = []
        self._turn_started_at: Optional[float] = None
        self._candidate_silence_started_at: Optional[float] = None
        self._barge_candidate_started_at: Optional[float] = None
        self._barge_candidate_reason: Optional[str] = None
        self._barge_candidate_waiting_logged = False
        self._tts_output_started_logged = False
        preroll_samples = int(
            self.config.input_sample_rate * self.config.preroll_ms / 1000.0
        )
        self._preroll = PreRollBuffer(preroll_samples)

    def _log_event(self, event: str, **fields: Any) -> None:
        if not self.config.verbose:
            return
        if fields:
            rendered_fields = " ".join(
                f"{key}={self._format_log_value(value)}"
                for key, value in fields.items()
            )
            logger.info("event=%s %s", event, rendered_fields)
            return
        logger.info("event=%s", event)

    @staticmethod
    def _format_log_value(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, str):
            return (
                repr(value)
                if not value or any(char.isspace() for char in value)
                else value
            )
        return str(value)

    def _log_vad_decisions(
        self, decisions: list[SpeechFrameDecision], now: float
    ) -> None:
        if not self.config.verbose or not decisions:
            return
        has_transition = any(
            decision.speech_started or decision.candidate_ended
            for decision in decisions
        )
        if not has_transition:
            return
        last = decisions[-1]
        max_probability = max(decision.probability for decision in decisions)
        self._log_event(
            "vad",
            monotonic_ms=now * 1000.0,
            probability=last.probability,
            max_probability=max_probability,
            frames=len(decisions),
            speech_frames=sum(decision.is_speech for decision in decisions),
            in_speech=bool(getattr(self.speech_gate, "in_speech", False)),
            speech_started=any(decision.speech_started for decision in decisions),
            candidate_ended=any(decision.candidate_ended for decision in decisions),
        )

    def _handle_playback_event(self, event: str, fields: dict[str, Any]) -> None:
        if not self.config.verbose:
            return
        if event == "tts_output_callback":
            if self._tts_output_started_logged and not fields.get("status"):
                return
            self._tts_output_started_logged = True
            event = "tts_output_started"
        log_fields = dict(fields)
        log_fields.setdefault("monotonic_ms", time.monotonic() * 1000.0)
        if self.loop is not None and self.loop.is_running():
            self.loop.call_soon_threadsafe(lambda: self._log_event(event, **log_fields))
        else:
            self._log_event(event, **log_fields)

    def _playback_state(self, now: Optional[float] = None) -> dict[str, Any]:
        now_ms = (now if now is not None else time.monotonic()) * 1000.0
        if self.audio_output is None or not hasattr(
            self.audio_output, "playback_state"
        ):
            return {
                "player_enabled": False,
                "player_playing": False,
                "buffered_samples": 0,
                "buffered_ms": 0.0,
                "last_output_age_ms": None,
            }
        return self.audio_output.playback_state(now_ms=now_ms)

    def _playback_echo_window_active(
        self, state: Optional[dict[str, Any]] = None, now: Optional[float] = None
    ) -> bool:
        state = state or self._playback_state(now)
        last_output_age_ms = state.get("last_output_age_ms")
        recent_output = (
            last_output_age_ms is not None
            and -50.0
            <= float(last_output_age_ms)
            <= self.config.ignore_playback_echo_ms
        )
        return bool(state.get("player_playing") or recent_output)

    def _echo_correlation(self, samples: np.ndarray, now: float) -> dict[str, Any]:
        if self.audio_output is None or not hasattr(
            self.audio_output, "echo_correlation"
        ):
            return {"correlation": 0.0, "delay_ms": None, "method": None}
        return self.audio_output.echo_correlation(
            samples,
            input_sample_rate=self.config.input_sample_rate,
            input_end_ms=now * 1000.0,
            min_delay_ms=self.config.echo_delay_min_ms,
            max_delay_ms=self.config.echo_delay_max_ms,
            step_ms=self.config.echo_correlation_step_ms,
        )

    @staticmethod
    def _normalize_text_for_match(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    def _barge_candidate_text_status(self, text: str) -> tuple[str, str]:
        candidate = self._normalize_text_for_match(text)
        compact_len = len(candidate.replace(" ", ""))
        if compact_len < self.config.barge_in_min_transcript_chars:
            return "too_short", "below_min_chars"

        response = self._normalize_text_for_match(self._current_response_text)
        if response and self._looks_like_playback_text(candidate, response):
            return "echo", "transcript_matches_playback"
        return "speech", "transcript"

    @staticmethod
    def _looks_like_playback_text(candidate: str, response: str) -> bool:
        if not candidate or not response:
            return False
        if response.startswith(candidate) or candidate in response:
            return True

        candidate_tokens = candidate.split()
        response_tokens = response.split()
        if not candidate_tokens or len(candidate_tokens) > len(response_tokens):
            return False
        for index in range(len(response_tokens) - len(candidate_tokens) + 1):
            window = response_tokens[index : index + len(candidate_tokens)]
            if all(
                response_token.startswith(candidate_token)
                for candidate_token, response_token in zip(candidate_tokens, window)
            ):
                return True
        return False

    async def init_models(self):
        if self.speech_gate is None:
            from mlx_audio.vad import load as load_vad

            try:
                self.speech_gate = await self.mlx.run(
                    lambda: SileroSpeechGate(
                        load_vad(self.config.vad_model),
                        sample_rate=self.config.input_sample_rate,
                        start_threshold=self.config.vad_start_threshold,
                        stop_threshold=self.config.vad_stop_threshold,
                        start_frames=self.config.vad_start_frames,
                        end_silence_ms=self.config.vad_end_silence_ms,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load speech VAD model {self.config.vad_model!r}"
                ) from exc

        if self.endpoint_detector is None:
            from mlx_audio.vad import load as load_vad

            try:
                self.endpoint_detector = await self.mlx.run(
                    lambda: SmartTurnEndpointDetector(
                        load_vad(self.config.turn_model),
                        sample_rate=self.config.input_sample_rate,
                        threshold=self.config.turn_threshold,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load Smart Turn model {self.config.turn_model!r}"
                ) from exc

        if self.transcriber is None:
            from mlx_audio.stt.utils import load as load_stt

            stt = await self.mlx.run(lambda: load_stt(self.config.stt_model))
            self.transcriber = VoxtralRealtimeTranscriber(
                stt,
                transcription_delay_ms=int(self.config.stt_transcription_delay_ms),
                max_decode_tokens_per_step=self.config.stt_max_decode_tokens_per_step,
                max_turn_tokens=self.config.stt_max_turn_tokens,
                verbose=self.config.verbose,
            )

        if self.response_engine is None:
            self.response_engine = LocalLLMResponseEngine(
                self.config.response_model, system_prompt=self.config.system_prompt
            )
            await self.mlx.run(self.response_engine.load)

        if self.tts_responder is None:
            tts = await self.mlx.run(lambda: load_tts(self.config.tts_model))
            self.tts_responder = PocketTTSResponder(
                tts,
                voice=self.config.tts_voice,
                streaming_interval=float(self.config.tts_streaming_interval),
                temperature=self.config.tts_temperature,
            )

        if self.audio_output is None:
            output_sr = self.config.output_sample_rate or self.tts_responder.sample_rate
            self.audio_output = AudioOutputStream(
                sample_rate=int(output_sr),
                enabled=self.config.play_audio,
                event_callback=self._handle_playback_event,
            )

    async def start(self):
        self.loop = asyncio.get_running_loop()
        tasks = []
        try:
            await self.init_models()
            tasks = [
                asyncio.create_task(self._listener()),
                asyncio.create_task(self._transcription_stepper()),
                asyncio.create_task(self._response_processor()),
                asyncio.create_task(self._audio_output_processor()),
            ]
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.audio_output is not None:
                self.audio_output.stop()
            await self.mlx.shutdown()

    def _sd_callback(self, indata, frames, _time, status):
        if status:
            self._log_event("input_status", status=str(status))
        audio = self._normalize_input(indata)

        def _enqueue():
            try:
                self.input_audio_queue.put_nowait(audio)
            except asyncio.QueueFull:
                if self.config.verbose:
                    logger.debug("Dropping input audio because queue is full")

        if self.loop is not None:
            self.loop.call_soon_threadsafe(_enqueue)

    def _normalize_input(self, indata) -> np.ndarray:
        audio = np.asarray(indata)
        if np.issubdtype(audio.dtype, np.signedinteger):
            scale = float(abs(np.iinfo(audio.dtype).min))
            audio = audio.astype(np.float32) / scale
        elif np.issubdtype(audio.dtype, np.unsignedinteger):
            info = np.iinfo(audio.dtype)
            midpoint = (float(info.max) + 1.0) / 2.0
            audio = (audio.astype(np.float32) - midpoint) / midpoint
        else:
            audio = audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.clip(audio.reshape(-1), -1.0, 1.0)

    async def _listener(self):
        frame_size = int(
            self.config.input_sample_rate * (self.config.frame_duration_ms / 1000.0)
        )
        stream = sd.InputStream(
            samplerate=self.config.input_sample_rate,
            blocksize=frame_size,
            channels=self.config.input_channels,
            dtype="int16",
            callback=self._sd_callback,
        )
        stream.start()
        self._log_event(
            "listening",
            sample_rate=self.config.input_sample_rate,
            frame_ms=self.config.frame_duration_ms,
            vad_start_threshold=self.config.vad_start_threshold,
            vad_stop_threshold=self.config.vad_stop_threshold,
        )
        try:
            while True:
                samples = await self.input_audio_queue.get()
                await self._process_input_audio(samples)
                self.input_audio_queue.task_done()
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        finally:
            stream.stop()
            stream.close()

    async def _process_input_audio(self, samples: np.ndarray):
        decisions = await self.mlx.run(lambda: self.speech_gate.process(samples))
        if not decisions:
            self._preroll.append(samples)
            return

        now = time.monotonic()
        self._log_vad_decisions(decisions, now)
        if self.transcriber is not None and self.transcriber.active:
            self.transcriber.feed(samples)
            self._turn_audio.append(samples.copy())
        else:
            self._preroll.append(samples)

        for decision in decisions:
            if decision.speech_started:
                await self._handle_speech_started(now, samples)
            elif self._barge_candidate_started_at is not None:
                await self._maybe_confirm_barge_candidate(decision, now)
            if decision.candidate_ended:
                if self._barge_candidate_started_at is not None:
                    await self._suppress_barge_candidate(
                        "ended_before_confirm", now, samples
                    )
                elif self.transcriber is not None:
                    await self._maybe_finalize_turn(now)

        if (
            self.transcriber is not None
            and self.transcriber.active
            and self._turn_started_at is not None
            and now - self._turn_started_at >= self.config.vad_max_turn_seconds
        ):
            self._log_event(
                "max_turn_duration",
                duration_seconds=now - self._turn_started_at,
            )
            await self._finalize_turn()

    async def _handle_speech_started(self, now: float, samples: np.ndarray):
        if self.transcriber is not None and self.transcriber.active:
            await self._start_turn(now)
            return
        playback_state = self._playback_state(now)
        if self._playback_echo_window_active(playback_state, now):
            await self._start_barge_candidate(now)
            correlation = self._echo_correlation(samples, now)
            self._log_event(
                "barge_in_candidate",
                reason=self._barge_candidate_reason,
                min_barge_in_ms=self.config.min_barge_in_ms,
                player_playing=bool(playback_state.get("player_playing")),
                buffered_ms=float(playback_state.get("buffered_ms") or 0.0),
                last_output_age_ms=playback_state.get("last_output_age_ms"),
                echo_correlation=correlation.get("correlation", 0.0),
                echo_delay_ms=correlation.get("delay_ms"),
                echo_method=correlation.get("method"),
            )
            return
        await self._start_turn(now)

    async def _start_barge_candidate(self, now: float) -> None:
        self._barge_candidate_started_at = now
        self._barge_candidate_reason = "playback_echo_window"
        self._barge_candidate_waiting_logged = False
        if self.transcriber is not None and not self.transcriber.active:
            await self._start_transcriber_capture(now)

    async def _maybe_confirm_barge_candidate(
        self, decision: SpeechFrameDecision, now: float
    ):
        if self._barge_candidate_started_at is None or not decision.is_speech:
            return
        elapsed_ms = (now - self._barge_candidate_started_at) * 1000.0
        if elapsed_ms < self.config.min_barge_in_ms:
            return
        if not self._barge_candidate_waiting_logged:
            self._barge_candidate_waiting_logged = True
            self._log_event(
                "barge_in_waiting_for_transcript",
                elapsed_ms=elapsed_ms,
                min_chars=self.config.barge_in_min_transcript_chars,
            )

    async def _evaluate_barge_candidate_transcript(
        self, now: float, samples: Optional[np.ndarray] = None
    ) -> bool:
        if self._barge_candidate_started_at is None or self.transcriber is None:
            return False
        text = getattr(self.transcriber, "text", "").strip()
        status, reason = self._barge_candidate_text_status(text)
        if status == "too_short":
            return False

        sample_window = samples
        if sample_window is None:
            sample_window = (
                self._turn_audio[-1]
                if self._turn_audio
                else np.zeros(0, dtype=np.float32)
            )
        if status == "echo":
            await self._suppress_barge_candidate(
                reason, now, sample_window, transcript=text
            )
            return True

        await self._confirm_barge_candidate(
            now,
            sample_window,
            reason=reason,
            transcript=text,
        )
        return True

    async def _confirm_barge_candidate(
        self,
        now: float,
        samples: np.ndarray,
        *,
        reason: str,
        transcript: Optional[str] = None,
        elapsed_ms: Optional[float] = None,
        correlation: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._barge_candidate_started_at is None:
            return
        candidate_started_at = self._barge_candidate_started_at
        if elapsed_ms is None:
            elapsed_ms = (now - candidate_started_at) * 1000.0
        if correlation is None:
            correlation = self._echo_correlation(samples, now)
        log_fields = {
            "reason": reason,
            "elapsed_ms": elapsed_ms,
            "echo_correlation": correlation.get("correlation", 0.0),
            "echo_delay_ms": correlation.get("delay_ms"),
            "echo_method": correlation.get("method"),
        }
        if transcript is not None:
            log_fields["text"] = transcript
            log_fields["chars"] = len(transcript)
        self._log_event("barge_in_confirmed", **log_fields)

        self._barge_candidate_started_at = None
        self._barge_candidate_reason = None
        self._barge_candidate_waiting_logged = False
        if self.config.barge_in:
            await self._handle_barge_in()
        if self.speech_gate is not None and hasattr(self.speech_gate, "in_speech"):
            self.speech_gate.in_speech = True
        if self.transcriber is not None and not self.transcriber.active:
            await self._start_transcriber_capture(candidate_started_at)
        self._turn_started_at = candidate_started_at
        self._candidate_silence_started_at = None
        self._log_event(
            "speech_started",
            preroll_ms=self.config.preroll_ms,
            vad_start_threshold=self.config.vad_start_threshold,
            vad_start_frames=self.config.vad_start_frames,
        )

    async def _suppress_barge_candidate(
        self,
        reason: str,
        now: float,
        samples: np.ndarray,
        *,
        transcript: Optional[str] = None,
    ) -> None:
        if self._barge_candidate_started_at is None:
            return
        elapsed_ms = (now - self._barge_candidate_started_at) * 1000.0
        correlation = self._echo_correlation(samples, now)
        self._barge_candidate_started_at = None
        self._barge_candidate_reason = None
        self._barge_candidate_waiting_logged = False
        self._turn_audio = []
        self._turn_started_at = None
        self._candidate_silence_started_at = None
        self._preroll.clear()
        if self.transcriber is not None and hasattr(self.transcriber, "reset"):
            await self.mlx.run(lambda: self.transcriber.reset())
        log_fields = {
            "reason": reason,
            "elapsed_ms": elapsed_ms,
            "echo_correlation": correlation.get("correlation", 0.0),
            "echo_delay_ms": correlation.get("delay_ms"),
            "echo_method": correlation.get("method"),
        }
        if transcript is not None:
            log_fields["text"] = transcript
            log_fields["chars"] = len(transcript)
        self._log_event("barge_in_suppressed_echo", **log_fields)

    async def _start_turn(self, now: float):
        self._barge_candidate_started_at = None
        self._barge_candidate_reason = None
        self._barge_candidate_waiting_logged = False
        if self.transcriber is not None and self.transcriber.active:
            self._candidate_silence_started_at = None
            return
        if self.config.barge_in:
            await self._handle_barge_in()
        if self.speech_gate is not None and hasattr(self.speech_gate, "in_speech"):
            self.speech_gate.in_speech = True
        if self.transcriber is None:
            return
        await self._start_transcriber_capture(now)
        self._turn_started_at = now
        self._log_event(
            "speech_started",
            preroll_ms=self.config.preroll_ms,
            vad_start_threshold=self.config.vad_start_threshold,
            vad_start_frames=self.config.vad_start_frames,
        )

    async def _start_transcriber_capture(self, now: float) -> None:
        if self.transcriber is None:
            return
        if not self.transcriber.active:
            await self.mlx.run(lambda: self.transcriber.start())
            preroll = self._preroll.get()
            self._turn_audio = []
            if preroll.size:
                self.transcriber.feed(preroll)
                self._turn_audio.append(preroll.copy())
            self._candidate_silence_started_at = None

    async def _maybe_finalize_turn(self, now: float):
        if not self._turn_audio:
            return
        if self._candidate_silence_started_at is None:
            self._candidate_silence_started_at = now

        audio = np.concatenate(self._turn_audio).astype(np.float32, copy=False)
        decision = await self.mlx.run(lambda: self.endpoint_detector.predict(audio))
        silence_ms = (now - self._candidate_silence_started_at) * 1000.0
        self._log_event(
            "endpoint_candidate",
            complete=decision.complete,
            probability=decision.probability,
            silence_ms=silence_ms,
            audio_ms=audio.size / self.config.input_sample_rate * 1000.0,
        )
        if (
            decision.complete
            or silence_ms >= self.config.turn_max_incomplete_silence_ms
        ):
            await self._finalize_turn()

    async def _finalize_turn(self):
        if self.transcriber is None or self.transcriber.session is None:
            self._log_event("turn_finalization_skipped", reason="no_active_session")
            return
        turn_audio = (
            np.concatenate(self._turn_audio).astype(np.float32, copy=False)
            if self._turn_audio
            else np.zeros(0, dtype=np.float32)
        )
        turn_audio_ms = turn_audio.size / self.config.input_sample_rate * 1000.0
        partial_chars = len(getattr(self.transcriber, "text", ""))
        self._log_event(
            "turn_finalization_started",
            audio_ms=turn_audio_ms,
            partial_chars=partial_chars,
            max_steps=self.config.stt_finalization_max_steps,
            max_turn_tokens=self.config.stt_max_turn_tokens,
        )
        start_time = time.monotonic()
        text = await self.mlx.run(
            lambda: self.transcriber.finish(
                max_steps=self.config.stt_finalization_max_steps
            )
        )
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        finish_steps = getattr(self.transcriber, "last_finish_steps", 0)
        hit_max_steps = getattr(self.transcriber, "last_finish_hit_max_steps", False)
        self._turn_audio = []
        self._turn_started_at = None
        self._candidate_silence_started_at = None
        self._preroll.clear()
        self._log_event(
            "turn_finalized",
            text=text,
            chars=len(text),
            elapsed_ms=elapsed_ms,
            finish_steps=finish_steps,
            hit_max_steps=hit_max_steps,
        )
        if text:
            await self.transcript_queue.put(text)
        else:
            self._log_event("turn_dropped", reason="empty_transcript")

    async def _handle_barge_in(self):
        tts_active = (
            self._current_tts_task is not None and not self._current_tts_task.done()
        )
        queued_audio = self.output_audio_queue.qsize()
        playback_state = self._playback_state()
        if self._current_tts_cancel is not None:
            self._current_tts_cancel.set()
        if tts_active:
            self._current_tts_task.cancel()
        self._clear_output_audio_queue()
        flush_status = {}
        if self.audio_output is not None:
            flush_status = self.audio_output.flush() or {}
        flushed_buffered_samples = int(flush_status.get("buffered_samples") or 0)
        playback_was_active = bool(
            flush_status.get("was_playing")
            or playback_state.get("player_playing")
            or playback_state.get("buffered_samples")
        )
        if tts_active or queued_audio or playback_was_active:
            self._log_event(
                "barge_in",
                cancelled_tts=tts_active,
                flushed_chunks=queued_audio,
                flushed_buffered_samples=flushed_buffered_samples,
                flushed_buffered_ms=float(flush_status.get("buffered_ms") or 0.0),
                playback_was_playing=bool(flush_status.get("was_playing")),
                last_output_age_ms=flush_status.get("last_output_age_ms"),
            )

    def _clear_output_audio_queue(self):
        while True:
            try:
                self.output_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self.output_audio_queue.task_done()

    async def _transcription_stepper(self):
        while True:
            await asyncio.sleep(0.02)
            if self.transcriber is None or not self.transcriber.active:
                continue
            deltas = await self.mlx.run(lambda: self.transcriber.step())
            for delta in deltas:
                if delta:
                    self._log_event("partial_transcript_delta", text=delta)
            if self._barge_candidate_started_at is not None:
                await self._evaluate_barge_candidate_transcript(time.monotonic())

    async def _response_processor(self):
        while True:
            transcript = await self.transcript_queue.get()
            task = asyncio.create_task(self._respond_to_transcript(transcript))
            try:
                await task
            finally:
                self.transcript_queue.task_done()

    async def _respond_to_transcript(self, transcript: str):
        response_text = await self.mlx.run(
            lambda: self.response_engine.generate(transcript, self._conversation)
        )
        response_text = response_text.strip()
        if not response_text:
            return
        self._current_response_text = response_text
        self._log_event("response_ready", text=response_text, chars=len(response_text))
        self._conversation.append({"role": "user", "content": transcript})
        self._conversation.append({"role": "assistant", "content": response_text})
        self._current_tts_cancel = asyncio.Event()
        self._current_tts_task = asyncio.create_task(
            self._speak_response(response_text, self._current_tts_cancel)
        )

    async def _speak_response(self, text: str, cancel_event: asyncio.Event):
        self._log_event("tts_started", chars=len(text))
        self._tts_output_started_logged = False
        generator = self.tts_responder.create_generator(text)
        sentinel = object()
        finished = False
        chunk_count = 0
        sample_count = 0

        def _next_chunk():
            try:
                result = next(generator)
            except StopIteration:
                return sentinel
            return np.asarray(result.audio)

        try:
            while not cancel_event.is_set():
                result = await self.mlx.run(_next_chunk)
                if result is sentinel:
                    finished = True
                    break
                if cancel_event.is_set():
                    break
                await self.output_audio_queue.put(result)
                chunk_count += 1
                sample_count += int(result.shape[0])
            if finished and not cancel_event.is_set():
                await self.output_audio_queue.join()
                self._start_buffered_tts_playback(force=True)
        except asyncio.CancelledError:
            cancel_event.set()
            self._log_tts_done("tts_cancelled", chunk_count, sample_count)
        except Exception as exc:
            self._log_event("tts_error", error=str(exc))
        else:
            self._log_tts_done(
                "tts_finished" if finished else "tts_cancelled",
                chunk_count,
                sample_count,
            )

    def _log_tts_done(self, event: str, chunk_count: int, sample_count: int) -> None:
        sample_rate = int(getattr(self.tts_responder, "sample_rate", 0) or 0)
        fields: dict[str, Any] = {
            "chunks": chunk_count,
            "samples": sample_count,
        }
        if sample_rate > 0:
            fields["duration_ms"] = sample_count / sample_rate * 1000.0
        self._log_event(event, **fields)

    def _start_buffered_tts_playback(self, *, force: bool) -> None:
        if self.audio_output is None or not hasattr(
            self.audio_output, "start_playback_if_buffered"
        ):
            return
        status = self.audio_output.start_playback_if_buffered(force=force) or {}
        if not self.config.verbose or not status.get("player_started"):
            return
        self._log_event(
            "tts_playback_started",
            sample_rate=status.get("sample_rate"),
            buffered_samples=status.get("buffered_samples"),
            buffered_ms=status.get("buffered_ms"),
            start_buffer_ms=status.get("start_buffer_ms"),
            forced=bool(status.get("forced")),
            output_queue_size=self.output_audio_queue.qsize(),
            monotonic_ms=time.monotonic() * 1000.0,
        )

    async def _audio_output_processor(self):
        while True:
            audio = await self.output_audio_queue.get()
            if self.audio_output is not None:
                queued_at = time.monotonic()
                status = self.audio_output.queue_audio(audio) or {}
                if self.config.verbose:
                    status.setdefault(
                        "output_queue_size", self.output_audio_queue.qsize()
                    )
                    status.setdefault("monotonic_ms", queued_at * 1000.0)
                    if status.get("player_started"):
                        self._log_event(
                            "tts_playback_started",
                            sample_rate=status.get("sample_rate"),
                            buffered_samples=status.get("buffered_samples"),
                            buffered_ms=status.get("buffered_ms"),
                            start_buffer_ms=status.get("start_buffer_ms"),
                            forced=False,
                            output_queue_size=status.get("output_queue_size"),
                            monotonic_ms=status.get("monotonic_ms"),
                        )
            self.output_audio_queue.task_done()


async def main():
    parser = argparse.ArgumentParser(
        description="Voice Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stt_model",
        type=str,
        default="mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
        help="STT model",
    )
    parser.add_argument(
        "--tts_model",
        type=str,
        default="mlx-community/pocket-tts",
        help="TTS model",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit",
        help="LLM model",
    )
    parser.add_argument(
        "--vad_model",
        type=str,
        default="mlx-community/silero-vad",
        help="Speech trigger VAD model",
    )
    parser.add_argument(
        "--turn_model",
        type=str,
        default="mlx-community/smart-turn-v3",
        help="Smart Turn endpoint model",
    )
    parser.add_argument(
        "--latency_profile",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Latency profile",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="cosette",
        help="Pocket TTS voice",
    )
    parser.add_argument(
        "--transcription_delay_ms",
        type=int,
        default=None,
        help="Override Voxtral Realtime transcription delay",
    )
    parser.add_argument(
        "--stt_max_turn_tokens",
        type=int,
        default=256,
        help="Maximum Voxtral decode tokens per user turn",
    )
    parser.add_argument(
        "--stt_finalization_max_steps",
        type=int,
        default=96,
        help="Maximum bounded Voxtral finalization steps after endpointing",
    )
    parser.add_argument(
        "--tts_streaming_interval",
        type=float,
        default=None,
        help="Pocket TTS streaming chunk interval in seconds",
    )
    parser.add_argument(
        "--vad_start_threshold",
        type=float,
        default=0.35,
        help="Silero VAD probability needed to start a turn",
    )
    parser.add_argument(
        "--vad_stop_threshold",
        type=float,
        default=0.2,
        help="Silero VAD probability needed to keep an active turn open",
    )
    parser.add_argument(
        "--vad_start_frames",
        type=int,
        default=1,
        help="Consecutive speech frames needed before starting a turn",
    )
    parser.add_argument(
        "--vad_end_silence_ms",
        type=int,
        default=600,
        help="Silence duration before asking Smart Turn to finalize",
    )
    parser.add_argument(
        "--turn_threshold",
        type=float,
        default=0.5,
        help="Smart Turn endpoint probability threshold",
    )
    parser.add_argument(
        "--min_barge_in_ms",
        type=int,
        default=180,
        help="Speech duration required before confirming playback-time barge-in",
    )
    parser.add_argument(
        "--ignore_playback_echo_ms",
        type=int,
        default=450,
        help="Playback echo window after recent output callbacks",
    )
    parser.add_argument(
        "--echo_delay_min_ms",
        type=int,
        default=250,
        help="Minimum expected acoustic echo delay for output-reference matching",
    )
    parser.add_argument(
        "--echo_delay_max_ms",
        type=int,
        default=500,
        help="Maximum expected acoustic echo delay for output-reference matching",
    )
    parser.add_argument(
        "--barge_in_min_transcript_chars",
        type=int,
        default=2,
        help="Minimum partial transcript characters needed to confirm barge-in",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable structured voice pipeline debug logs",
    )
    parser.add_argument(
        "--no_play",
        action="store_true",
        help="Run without playing generated audio",
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    config = VoicePipelineConfig(
        latency_profile=args.latency_profile,
        stt_model=args.stt_model,
        tts_model=args.tts_model,
        tts_voice=args.voice,
        response_model=args.llm_model,
        vad_model=args.vad_model,
        turn_model=args.turn_model,
        vad_start_threshold=args.vad_start_threshold,
        vad_stop_threshold=args.vad_stop_threshold,
        vad_start_frames=args.vad_start_frames,
        vad_end_silence_ms=args.vad_end_silence_ms,
        turn_threshold=args.turn_threshold,
        min_barge_in_ms=args.min_barge_in_ms,
        ignore_playback_echo_ms=args.ignore_playback_echo_ms,
        echo_delay_min_ms=args.echo_delay_min_ms,
        echo_delay_max_ms=args.echo_delay_max_ms,
        barge_in_min_transcript_chars=args.barge_in_min_transcript_chars,
        stt_transcription_delay_ms=args.transcription_delay_ms,
        stt_max_turn_tokens=args.stt_max_turn_tokens,
        stt_finalization_max_steps=args.stt_finalization_max_steps,
        tts_streaming_interval=args.tts_streaming_interval,
        verbose=args.verbose,
        play_audio=not args.no_play,
    )
    pipeline = VoicePipeline(config)
    await pipeline.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
