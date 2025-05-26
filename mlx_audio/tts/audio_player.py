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
                self.drain_event.set()
                self.playing = False
                raise sd.CallbackStop()

    def start_stream(self):
        print("Starting audio stream...")
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

    def queue_audio(self, samples):
        if not len(samples):
            return

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
            self.audio_buffer.append(np.asarray(samples))

        # start playback only when we have enough buffered audio
        needed = int(self.arrival_rate * self.min_buffer_seconds)
        if not self.playing and self.buffered_samples() >= needed:
            self.start_stream()

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
        if not self.playing:
            return

        with self.buffer_lock:
            self.audio_buffer.clear()
        self.stop_stream()
        self.playing = False
        self.drain_event.set()
