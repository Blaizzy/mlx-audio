import time
from collections import deque
from threading import Event, Lock

import numpy as np
import sounddevice as sd


class AudioPlayer:
    min_buffer_seconds = 1.5  # with respect to real-time, not the sample rate
    measure_window = 0.25
    ema_alpha = 0.25

    def __init__(self, sample_rate=24_000, buffer_size=2048):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        self.audio_buffer = deque()
        self.buffer_lock = Lock()
        self.stream: sd.OutputStream | None = None
        self.playing = False
        self.drain_event = Event()
        self.drain_event.set()

        self.window_sample_count = 0
        self.window_start = time.perf_counter()
        self.arrival_rate = sample_rate  # assume real-time to start

    def callback(self, outdata, frames, time, status):
        outdata.fill(0)  # initialize the frame with silence
        filled = 0

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

            if not self.audio_buffer and filled < frames:
                self.playing = False
                raise sd.CallbackStop()

    def _on_stream_finished(self):
        # CallbackStop only means all samples have been submitted. PortAudio
        # calls this after the device has actually played the pending output.
        self.playing = False
        self.drain_event.set()

    def start_stream(self):
        # An underrun stops callbacks but leaves the old stream open. Finish
        # and close it before replacing it (and reusing the drain event).
        if self.stream is not None:
            self.stop_stream()

        print("\nStarting audio stream...")
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.callback,
            finished_callback=self._on_stream_finished,
            blocksize=self.buffer_size,
        )
        # Callbacks may run before start() returns, so publish state first.
        self.drain_event.clear()
        self.playing = True
        try:
            self.stream.start()
        except BaseException:
            self.stop_stream(abort=True)
            raise

    def stop_stream(self, *, abort=False):
        try:
            if self.stream is not None:
                try:
                    if abort:
                        self.stream.abort()
                    else:
                        self.stream.stop()
                finally:
                    self.stream.close()
        finally:
            self.stream = None
            self.playing = False
            self.drain_event.set()

    def buffered_samples(self) -> int:
        with self.buffer_lock:
            return sum(map(len, self.audio_buffer))

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

        now = time.perf_counter()

        # arrival-rate statistics
        self.window_sample_count += len(samples)
        if now - self.window_start >= self.measure_window:
            inst_rate = self.window_sample_count / (now - self.window_start)
            self.arrival_rate = (
                inst_rate
                if self.arrival_rate is None
                else self.ema_alpha * inst_rate
                + (1 - self.ema_alpha) * self.arrival_rate
            )
            self.window_sample_count = 0
            self.window_start = now

        with self.buffer_lock:
            self.audio_buffer.append(samples)

        # start playback only when we have enough buffered audio
        needed = int(self.arrival_rate * self.min_buffer_seconds)
        if not self.playing and self.buffered_samples() >= needed:
            self.start_stream()

    def wait_for_drain(self):
        # If the stream never started (e.g. short text whose total audio fell
        # below the min_buffer_seconds threshold), force-start it now so the
        # buffered audio is played instead of hanging forever.
        if not self.playing and self.buffered_samples() > 0:
            self.start_stream()
        # An empty Python buffer can still have output pending at the device.
        if self.stream is not None:
            return self.drain_event.wait()

    def stop(self):
        """Play all queued audio, then close the stream."""
        # wait_for_drain() force-starts the stream when buffered audio never
        # reached min_buffer_seconds, so it must run even if playback has not
        # started; gating on `self.playing` silently dropped short utterances.
        try:
            self.wait_for_drain()
        finally:
            self.stop_stream()

    def flush(self):
        """Discard everything and stop playback immediately."""
        with self.buffer_lock:
            self.audio_buffer.clear()
        self.stop_stream(abort=True)
