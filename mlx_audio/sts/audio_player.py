import time
from collections import deque
from threading import Event, Lock

import numpy as np
import sounddevice as sd


class AudioPlayer:
    min_buffer_seconds = 0.5

    def __init__(
        self,
        sample_rate=24_000,
        buffer_size=2048,
        event_callback=None,
        start_buffer_seconds=None,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.event_callback = event_callback
        self.start_buffer_seconds = float(
            self.min_buffer_seconds
            if start_buffer_seconds is None
            else start_buffer_seconds
        )

        self.audio_buffer = deque()
        self.buffer_lock = Lock()
        self.stream: sd.OutputStream | None = None
        self.playing = False
        self.drain_event = Event()
        self.history_lock = Lock()
        self.output_history = deque()
        self.output_history_seconds = 5.0
        self.last_output_callback_monotonic_ms = None
        self.last_output_dac_time = None
        self.last_output_duration_ms = 0.0
        self.last_output_samples = 0
        self.last_flush_monotonic_ms = None
        self.last_flush_buffered_samples = 0
        self.last_flush_was_playing = False

    def _emit_event(self, event, **fields):
        if self.event_callback is None:
            return
        try:
            self.event_callback(event, fields)
        except Exception:
            pass

    def callback(self, outdata, frames, callback_time, status):
        outdata.fill(0)  # initialize the frame with silence
        filled = 0
        should_stop = False
        buffered_samples = 0

        with self.buffer_lock:
            while filled < frames and self.audio_buffer:
                buf = self.audio_buffer[0]
                to_copy = min(frames - filled, len(buf))
                outdata[filled : filled + to_copy, 0] = buf[:to_copy]
                filled += to_copy

                if to_copy == len(buf):
                    self.audio_buffer.popleft()
                else:
                    self.audio_buffer[0] = buf[to_copy:]

            buffered_samples = sum(map(len, self.audio_buffer))
            if not self.audio_buffer and filled < frames:
                self.drain_event.set()
                self.playing = False
                should_stop = True

        if filled:
            callback_monotonic_ms = time.monotonic() * 1000.0
            output_dac_time = getattr(callback_time, "outputBufferDacTime", None)
            current_time = getattr(callback_time, "currentTime", None)
            output_start_ms = (
                float(output_dac_time) * 1000.0
                if output_dac_time is not None
                else callback_monotonic_ms
            )
            output_samples = outdata[:filled, 0].copy()
            self._append_output_history(output_start_ms, output_samples)
            self.last_output_callback_monotonic_ms = callback_monotonic_ms
            self.last_output_dac_time = (
                float(output_dac_time) if output_dac_time is not None else None
            )
            self.last_output_duration_ms = filled / self.sample_rate * 1000.0
            self.last_output_samples = filled
            fields = {
                "samples": filled,
                "frames": frames,
                "duration_ms": filled / self.sample_rate * 1000.0,
                "sample_rate": self.sample_rate,
                "buffered_samples": buffered_samples,
                "buffered_ms": buffered_samples / self.sample_rate * 1000.0,
                "callback_monotonic_ms": callback_monotonic_ms,
            }
            if output_dac_time is not None:
                fields["pa_output_dac_time"] = float(output_dac_time)
            if current_time is not None:
                fields["pa_current_time"] = float(current_time)
            if status:
                fields["status"] = str(status)
            self._emit_event("tts_output_callback", **fields)

        if should_stop:
            raise sd.CallbackStop()

    def start_stream(self):
        if self.stream is not None:
            self.stop_stream()
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.callback,
            blocksize=self.buffer_size,
        )
        self.stream.start()
        self.playing = True
        self.drain_event.clear()

    def stop_stream(self):
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
            self.playing = False

    def buffered_samples(self) -> int:
        return sum(map(len, self.audio_buffer))

    def start_buffer_samples(self) -> int:
        return max(1, int(round(self.sample_rate * self.start_buffer_seconds)))

    def start_if_buffered(self, *, force=False) -> dict:
        buffered_samples = self.buffered_samples()
        needed = 1 if force else self.start_buffer_samples()
        started = False
        if not self.playing and buffered_samples >= needed:
            self.start_stream()
            started = True
        return {
            "player_enabled": True,
            "player_playing": self.playing,
            "player_started": started,
            "sample_rate": self.sample_rate,
            "buffered_samples": buffered_samples,
            "buffered_ms": buffered_samples / self.sample_rate * 1000.0,
            "start_buffer_samples": self.start_buffer_samples(),
            "start_buffer_ms": self.start_buffer_seconds * 1000.0,
            "forced": force,
        }

    def playback_state(self, now_ms=None) -> dict:
        now_ms = time.monotonic() * 1000.0 if now_ms is None else float(now_ms)
        buffered_samples = self.buffered_samples()
        state = {
            "player_enabled": True,
            "player_playing": self.playing,
            "sample_rate": self.sample_rate,
            "buffered_samples": buffered_samples,
            "buffered_ms": buffered_samples / self.sample_rate * 1000.0,
            "last_output_callback_monotonic_ms": self.last_output_callback_monotonic_ms,
            "last_output_age_ms": None,
            "last_output_duration_ms": self.last_output_duration_ms,
            "last_output_samples": self.last_output_samples,
            "last_output_dac_time": self.last_output_dac_time,
            "last_flush_monotonic_ms": self.last_flush_monotonic_ms,
            "last_flush_age_ms": None,
            "last_flush_buffered_samples": self.last_flush_buffered_samples,
            "last_flush_was_playing": self.last_flush_was_playing,
            "start_buffer_samples": self.start_buffer_samples(),
            "start_buffer_ms": self.start_buffer_seconds * 1000.0,
        }
        if self.last_output_callback_monotonic_ms is not None:
            state["last_output_age_ms"] = (
                now_ms - self.last_output_callback_monotonic_ms
            )
        if self.last_flush_monotonic_ms is not None:
            state["last_flush_age_ms"] = now_ms - self.last_flush_monotonic_ms
        return state

    def _append_output_history(self, start_ms: float, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        with self.history_lock:
            self.output_history.append((float(start_ms), samples))
            cutoff_ms = start_ms - self.output_history_seconds * 1000.0
            while self.output_history:
                chunk_start, chunk = self.output_history[0]
                chunk_end = chunk_start + chunk.size / self.sample_rate * 1000.0
                if chunk_end >= cutoff_ms:
                    break
                self.output_history.popleft()

    def echo_correlation(
        self,
        samples,
        *,
        input_sample_rate: int,
        input_end_ms: float,
        min_delay_ms: float,
        max_delay_ms: float,
        step_ms: float = 32.0,
    ) -> dict:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size < 8:
            return {"correlation": 0.0, "delay_ms": None, "method": None}
        duration_ms = samples.size / input_sample_rate * 1000.0
        best = {"correlation": 0.0, "delay_ms": None, "method": None}
        delay = float(min_delay_ms)
        while delay <= max_delay_ms:
            output_end = input_end_ms - delay
            output_start = output_end - duration_ms
            output = self._history_segment(output_start, output_end)
            if output is not None and output.size >= 8:
                output = self._resample(output, samples.size)
                raw = self._normalized_correlation(samples, output)
                env = self._normalized_correlation(
                    self._envelope(samples), self._envelope(output)
                )
                correlation = max(raw, env)
                method = "raw" if raw >= env else "envelope"
                if correlation > best["correlation"]:
                    best = {
                        "correlation": correlation,
                        "delay_ms": delay,
                        "method": method,
                    }
            delay += step_ms
        return best

    def _history_segment(self, start_ms: float, end_ms: float) -> np.ndarray | None:
        pieces = []
        with self.history_lock:
            history = list(self.output_history)
        for chunk_start, chunk in history:
            chunk_end = chunk_start + chunk.size / self.sample_rate * 1000.0
            if chunk_end <= start_ms or chunk_start >= end_ms:
                continue
            start_index = max(
                0, int(round((start_ms - chunk_start) / 1000.0 * self.sample_rate))
            )
            end_index = min(
                chunk.size,
                int(round((end_ms - chunk_start) / 1000.0 * self.sample_rate)),
            )
            if end_index > start_index:
                pieces.append(chunk[start_index:end_index])
        if not pieces:
            return None
        return np.concatenate(pieces)

    @staticmethod
    def _resample(samples: np.ndarray, size: int) -> np.ndarray:
        if samples.size == size:
            return samples.astype(np.float32, copy=False)
        if samples.size <= 1:
            return np.zeros(size, dtype=np.float32)
        source = np.linspace(0.0, 1.0, samples.size)
        target = np.linspace(0.0, 1.0, size)
        return np.interp(target, source, samples).astype(np.float32)

    @staticmethod
    def _envelope(samples: np.ndarray, window: int = 16) -> np.ndarray:
        samples = np.abs(np.asarray(samples, dtype=np.float32).reshape(-1))
        if samples.size < window:
            return samples
        kernel = np.ones(window, dtype=np.float32) / window
        return np.convolve(samples, kernel, mode="same").astype(np.float32)

    @staticmethod
    def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.size != b.size or a.size == 0:
            return 0.0
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-8:
            return 0.0
        return abs(float(np.dot(a, b) / denom))

    def queue_audio(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if not len(samples):
            return
        if samples.ndim == 0:
            samples = samples.reshape(1)
        elif samples.ndim == 2:
            if samples.shape[1] == 1:
                samples = samples[:, 0]
            elif samples.shape[0] == 1:
                samples = samples[0]
            elif samples.shape[0] <= 8 and samples.shape[0] < samples.shape[1]:
                samples = samples.mean(axis=0)
            else:
                samples = samples.mean(axis=1)
        elif samples.ndim > 2:
            samples = samples.reshape(-1)

        with self.buffer_lock:
            self.audio_buffer.append(samples)

        self.start_if_buffered(force=False)

    def wait_for_drain(self):
        return self.drain_event.wait()

    def stop(self):
        if self.playing:
            self.wait_for_drain()
            sd.sleep(100)

            self.stop_stream()
            self.playing = False

    def flush(self):
        """Discard everything and stop playback immediately."""
        was_playing = self.playing
        buffered_samples = self.buffered_samples()
        self.last_flush_monotonic_ms = time.monotonic() * 1000.0
        self.last_flush_buffered_samples = buffered_samples
        self.last_flush_was_playing = was_playing

        with self.buffer_lock:
            self.audio_buffer.clear()
        if was_playing:
            self.stop_stream()
        self.playing = False
        self.drain_event.set()
        return {
            "was_playing": was_playing,
            "buffered_samples": buffered_samples,
            "buffered_ms": buffered_samples / self.sample_rate * 1000.0,
            "last_output_age_ms": self.playback_state()["last_output_age_ms"],
        }
