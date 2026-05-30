import asyncio
import logging
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_audio.sts.audio_player import AudioPlayer
from mlx_audio.sts.voice_pipeline import (
    EndpointDecision,
    MLXWorkScheduler,
    PocketTTSResponder,
    PreRollBuffer,
    SileroSpeechGate,
    SpeechFrameDecision,
    VoicePipeline,
    VoicePipelineConfig,
    VoxtralRealtimeTranscriber,
)


class FakeVADModel:
    def __init__(self, probabilities, chunk_size=4):
        self.probabilities = list(probabilities)
        self.config = SimpleNamespace(branch_16k=SimpleNamespace(chunk_size=chunk_size))
        self.calls = []

    def initial_state(self, sample_rate=16000):
        return {"sample_rate": sample_rate}

    def feed(self, chunk, state, sample_rate=16000):
        self.calls.append(np.asarray(chunk).copy())
        probability = self.probabilities.pop(0)
        return np.array([probability], dtype=np.float32), state


class FakeStreamingSession:
    def __init__(self):
        self.done = False
        self.closed = False
        self.feeds = []
        self.steps = 0

    def feed(self, samples):
        self.feeds.append(np.asarray(samples, dtype=np.float32))

    def close(self):
        self.closed = True

    def step(self, max_decode_tokens=4):
        self.steps += 1
        if self.closed:
            self.done = True
            return [" done"]
        return ["hello"] if self.steps == 1 else []


class FakeRealtimeModel:
    def __init__(self):
        self.sessions = []
        self.delay = None
        self.max_tokens = None

    def create_streaming_session(self, max_tokens=None, transcription_delay_ms=None):
        self.delay = transcription_delay_ms
        self.max_tokens = max_tokens
        session = FakeStreamingSession()
        self.sessions.append(session)
        return session


class FakeEndpointDetector:
    def __init__(self, complete=True):
        self.complete = complete
        self.calls = []

    def predict(self, audio):
        self.calls.append(np.asarray(audio, dtype=np.float32))
        return EndpointDecision(complete=self.complete, probability=0.9)


class FakePipelineTranscriber:
    def __init__(self):
        self.session = None
        self.feeds = []
        self.started = 0
        self.text = ""
        self.deltas = []

    @property
    def active(self):
        return self.session is not None

    def start(self):
        self.started += 1
        self.session = object()
        self.text = ""

    def feed(self, samples):
        self.feeds.append(np.asarray(samples, dtype=np.float32))

    def step(self):
        if self.deltas:
            delta = self.deltas.pop(0)
            self.text += delta
            return [delta]
        return []

    def finish(self, max_steps=96):
        self.max_steps = max_steps
        self.session = None
        return "hello there"

    def reset(self):
        self.session = None
        self.text = ""


class EmptyPipelineTranscriber(FakePipelineTranscriber):
    def finish(self, max_steps=96):
        self.max_steps = max_steps
        self.last_finish_steps = 3
        self.last_finish_hit_max_steps = False
        self.session = None
        return ""


class FakeResponseEngine:
    def generate(self, transcript, context=None):
        return f"echo {transcript}"


class FakeTTSModel:
    sample_rate = 24_000

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        yield SimpleNamespace(audio=np.array([0.1, 0.2], dtype=np.float32))


class FakeAudioOutput:
    sample_rate = 24_000

    def __init__(self, *, playing=False, buffered_samples=0, last_output_age_ms=None):
        self.flushed = 0
        self.queued = []
        self.playing = playing
        self.buffered_samples = buffered_samples
        self.last_output_age_ms = last_output_age_ms
        self.echo = {"correlation": 0.0, "delay_ms": None, "method": None}

    def flush(self):
        self.flushed += 1
        was_playing = self.playing
        buffered_samples = self.buffered_samples
        self.playing = False
        self.buffered_samples = 0
        return {
            "was_playing": was_playing,
            "buffered_samples": buffered_samples,
            "buffered_ms": buffered_samples / self.sample_rate * 1000.0,
            "last_output_age_ms": self.last_output_age_ms,
        }

    def queue_audio(self, audio):
        audio = np.asarray(audio)
        self.queued.append(audio)
        samples = int(audio.size)
        self.buffered_samples += samples
        return {
            "player_enabled": True,
            "player_playing": self.playing,
            "player_started": False,
            "sample_rate": self.sample_rate,
            "samples": samples,
            "duration_ms": samples / self.sample_rate * 1000.0,
            "buffered_samples": self.buffered_samples,
            "buffered_ms": self.buffered_samples / self.sample_rate * 1000.0,
        }

    def playback_state(self, now_ms=None):
        return {
            "player_enabled": True,
            "player_playing": self.playing,
            "sample_rate": self.sample_rate,
            "buffered_samples": self.buffered_samples,
            "buffered_ms": self.buffered_samples / self.sample_rate * 1000.0,
            "last_output_age_ms": self.last_output_age_ms,
        }

    def echo_correlation(self, *args, **kwargs):
        return self.echo

    def stop(self):
        pass


class TestVoxtralRealtimePipelinePieces:
    def test_mlx_scheduler_keeps_arrays_on_one_stream_thread(self):
        async def run_scheduler():
            scheduler = MLXWorkScheduler()
            holder = {}
            await scheduler.run(
                lambda: holder.setdefault("x", mx.array([1.0, 2.0]) + 3.0)
            )
            await scheduler.run(lambda: mx.async_eval(holder["x"]))
            value = await scheduler.run(lambda: float(holder["x"][0].item()))
            await scheduler.shutdown()
            return value

        assert asyncio.run(run_scheduler()) == 4.0

    def test_config_expands_latency_profile_defaults(self):
        fast = VoicePipelineConfig(latency_profile="fast")
        balanced = VoicePipelineConfig(latency_profile="balanced")
        quality = VoicePipelineConfig(latency_profile="quality")

        assert fast.stt_transcription_delay_ms == 240
        assert balanced.stt_transcription_delay_ms == 480
        assert quality.stt_transcription_delay_ms == 960
        assert fast.tts_streaming_interval == 0.24
        assert balanced.tts_streaming_interval == 0.32
        assert quality.tts_streaming_interval == 0.48
        assert balanced.vad_start_threshold == 0.35
        assert balanced.vad_stop_threshold == 0.2
        assert balanced.vad_start_frames == 1
        assert balanced.vad_end_silence_ms == 600
        assert balanced.tts_voice == "cosette"
        assert balanced.verbose is False
        assert balanced.min_barge_in_ms == 180
        assert balanced.ignore_playback_echo_ms == 450
        assert balanced.echo_delay_min_ms == 250
        assert balanced.echo_delay_max_ms == 500
        assert balanced.barge_in_min_transcript_chars == 2
        assert balanced.stt_max_turn_tokens == 256
        assert balanced.stt_finalization_max_steps == 96

    def test_preroll_buffer_keeps_only_recent_samples(self):
        buffer = PreRollBuffer(max_samples=5)
        buffer.append(np.array([1, 2, 3], dtype=np.float32))
        buffer.append(np.array([4, 5, 6, 7], dtype=np.float32))

        assert np.allclose(buffer.get(), np.array([3, 4, 5, 6, 7], dtype=np.float32))

    def test_normalize_input_scales_int16_before_channel_mix(self):
        pipeline = VoicePipeline(VoicePipelineConfig(play_audio=False))
        indata = np.array(
            [
                [0],
                [16_384],
                [-16_384],
                [32_767],
                [-32_768],
            ],
            dtype=np.int16,
        )

        audio = pipeline._normalize_input(indata)

        assert audio.dtype == np.float32
        assert np.allclose(
            audio,
            np.array([0.0, 0.5, -0.5, 32_767 / 32_768, -1.0], dtype=np.float32),
        )
        assert not np.all(np.abs(audio[1:]) == 1.0)

    def test_normalize_input_averages_scaled_stereo_int16(self):
        pipeline = VoicePipeline(VoicePipelineConfig(play_audio=False))
        indata = np.array(
            [
                [16_384, 0],
                [-16_384, 0],
            ],
            dtype=np.int16,
        )

        audio = pipeline._normalize_input(indata)

        assert np.allclose(audio, np.array([0.25, -0.25], dtype=np.float32))

    def test_silero_speech_gate_uses_hysteresis(self):
        vad = FakeVADModel([0.7, 0.8, 0.1], chunk_size=4)
        gate = SileroSpeechGate(
            vad,
            sample_rate=16,
            start_threshold=0.6,
            stop_threshold=0.35,
            start_frames=2,
            end_silence_ms=250,
        )

        decisions = gate.process(np.ones(12, dtype=np.float32))

        assert [d.speech_started for d in decisions] == [False, True, False]
        assert [d.candidate_ended for d in decisions] == [False, False, True]
        assert len(vad.calls) == 3

    def test_default_vad_starts_on_single_quiet_frame(self):
        config = VoicePipelineConfig()
        vad = FakeVADModel([0.36], chunk_size=4)
        gate = SileroSpeechGate(
            vad,
            sample_rate=16,
            start_threshold=config.vad_start_threshold,
            stop_threshold=config.vad_stop_threshold,
            start_frames=config.vad_start_frames,
            end_silence_ms=config.vad_end_silence_ms,
        )

        decisions = gate.process(np.ones(4, dtype=np.float32))

        assert decisions[0].speech_started is True
        assert decisions[0].is_speech is True

    def test_voxtral_realtime_transcriber_streams_and_finishes(self):
        model = FakeRealtimeModel()
        transcriber = VoxtralRealtimeTranscriber(
            model,
            transcription_delay_ms=240,
            max_decode_tokens_per_step=3,
            max_turn_tokens=64,
        )

        transcriber.feed(np.array([0.1, 0.2], dtype=np.float32))
        deltas = transcriber.step()
        final = transcriber.finish()

        assert model.delay == 240
        assert model.max_tokens == 64
        assert deltas == ["hello"]
        assert final == "hello done"
        assert transcriber.last_finish_steps == 1
        assert transcriber.last_finish_hit_max_steps is False
        assert np.allclose(model.sessions[0].feeds[0], [0.1, 0.2])

    def test_pocket_tts_responder_passes_streaming_options(self):
        model = FakeTTSModel()
        responder = PocketTTSResponder(
            model, voice="alba", streaming_interval=0.24, temperature=0.5
        )

        chunks = list(responder.create_generator("hello"))

        assert np.allclose(chunks[0].audio, [0.1, 0.2])
        assert model.kwargs["text"] == "hello"
        assert model.kwargs["voice"] == "alba"
        assert model.kwargs["stream"] is True
        assert model.kwargs["streaming_interval"] == 0.24
        assert model.kwargs["temperature"] == 0.5

    def test_audio_player_output_callback_emits_playback_event(self):
        class CallbackTime:
            outputBufferDacTime = 12.5
            currentTime = 12.0

        events = []
        player = AudioPlayer(
            sample_rate=4,
            event_callback=lambda event, fields: events.append((event, fields)),
        )
        player.audio_buffer.append(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        outdata = np.zeros((2, 1), dtype=np.float32)

        player.callback(outdata, 2, CallbackTime(), None)

        assert np.allclose(outdata[:, 0], [0.1, 0.2])
        assert events[0][0] == "tts_output_callback"
        assert events[0][1]["samples"] == 2
        assert events[0][1]["frames"] == 2
        assert events[0][1]["duration_ms"] == 500.0
        assert events[0][1]["buffered_samples"] == 1
        assert events[0][1]["pa_output_dac_time"] == 12.5

    def test_audio_player_flush_clears_buffer_even_before_stream_starts(self):
        player = AudioPlayer(sample_rate=4)
        player.audio_buffer.append(np.array([0.1, 0.2, 0.3], dtype=np.float32))

        status = player.flush()

        assert status["was_playing"] is False
        assert status["buffered_samples"] == 3
        assert player.buffered_samples() == 0
        assert player.playing is False

    def test_audio_player_start_threshold_uses_audio_duration(self, monkeypatch):
        player = AudioPlayer(sample_rate=10, start_buffer_seconds=0.5)
        starts = []

        def fake_start_stream():
            starts.append(True)
            player.playing = True

        monkeypatch.setattr(player, "start_stream", fake_start_stream)

        player.queue_audio(np.ones(5, dtype=np.float32))

        assert starts == [True]
        assert player.playing is True

    def test_audio_player_can_force_start_short_buffer(self, monkeypatch):
        player = AudioPlayer(sample_rate=10, start_buffer_seconds=1.0)
        starts = []

        def fake_start_stream():
            starts.append(True)
            player.playing = True

        monkeypatch.setattr(player, "start_stream", fake_start_stream)
        player.audio_buffer.append(np.ones(3, dtype=np.float32))

        status = player.start_if_buffered(force=True)

        assert starts == [True]
        assert status["player_started"] is True
        assert status["forced"] is True
        assert status["buffered_ms"] == 300.0

    def test_audio_player_echo_correlation_finds_delayed_output(self):
        player = AudioPlayer(sample_rate=16)
        output = np.sin(np.linspace(0, 2 * np.pi, 32)).astype(np.float32)
        player._append_output_history(1000.0, output)

        result = player.echo_correlation(
            output[:16],
            input_sample_rate=16,
            input_end_ms=2500.0,
            min_delay_ms=500,
            max_delay_ms=500,
        )

        assert result["correlation"] > 0.99
        assert result["delay_ms"] == 500

    def test_pipeline_finalizes_turn_with_smart_turn_decision(self):
        class ScriptedGate:
            in_speech = False

            def __init__(self):
                self.calls = 0

            def process(self, _samples):
                self.calls += 1
                if self.calls == 1:
                    return [
                        SpeechFrameDecision(
                            probability=0.9, is_speech=True, speech_started=True
                        )
                    ]
                return [
                    SpeechFrameDecision(
                        probability=0.1, is_speech=False, candidate_ended=True
                    )
                ]

        transcriber = FakePipelineTranscriber()
        endpoint = FakeEndpointDetector(complete=True)
        output = FakeAudioOutput()
        pipeline = VoicePipeline(
            VoicePipelineConfig(play_audio=False),
            speech_gate=ScriptedGate(),
            endpoint_detector=endpoint,
            transcriber=transcriber,
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=output,
        )

        async def run_turn():
            await pipeline._process_input_audio(np.ones(4, dtype=np.float32))
            await pipeline._process_input_audio(np.zeros(4, dtype=np.float32))
            return await pipeline.transcript_queue.get()

        transcript = asyncio.run(run_turn())

        assert transcript == "hello there"
        assert transcriber.started == 1
        assert transcriber.max_steps == 96
        assert len(transcriber.feeds) == 2
        assert len(endpoint.calls) == 1

    def test_pipeline_logs_empty_finalized_turn(self, caplog):
        transcriber = EmptyPipelineTranscriber()
        transcriber.start()
        pipeline = VoicePipeline(
            VoicePipelineConfig(
                play_audio=False,
                verbose=True,
                stt_finalization_max_steps=7,
            ),
            speech_gate=SimpleNamespace(),
            endpoint_detector=FakeEndpointDetector(),
            transcriber=transcriber,
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=FakeAudioOutput(),
        )
        pipeline._turn_audio = [np.ones(16, dtype=np.float32)]

        async def finalize():
            await pipeline._finalize_turn()
            await pipeline.mlx.shutdown()

        with caplog.at_level(logging.INFO, logger="mlx_audio.sts.voice_pipeline"):
            asyncio.run(finalize())

        assert transcriber.max_steps == 7
        assert "event=turn_finalization_started" in caplog.text
        assert "event=turn_finalized" in caplog.text
        assert "chars=0" in caplog.text
        assert "event=turn_dropped reason=empty_transcript" in caplog.text

    def test_audio_output_processor_logs_playback_start_event(self, caplog):
        class StartingAudioOutput(FakeAudioOutput):
            def queue_audio(self, audio):
                status = super().queue_audio(audio)
                status["player_started"] = True
                return status

        pipeline = VoicePipeline(
            VoicePipelineConfig(
                play_audio=False,
                verbose=True,
            ),
            speech_gate=SimpleNamespace(),
            endpoint_detector=SimpleNamespace(),
            transcriber=SimpleNamespace(active=False),
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=StartingAudioOutput(),
        )

        async def run_output_once():
            task = asyncio.create_task(pipeline._audio_output_processor())
            await pipeline.output_audio_queue.put(
                np.array([0.1, 0.2], dtype=np.float32)
            )
            await asyncio.wait_for(pipeline.output_audio_queue.join(), timeout=1.0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await pipeline.mlx.shutdown()

        with caplog.at_level(logging.INFO, logger="mlx_audio.sts.voice_pipeline"):
            asyncio.run(run_output_once())

        assert "event=tts_playback_started" in caplog.text
        assert "output_queue_size=0" in caplog.text

    def test_echo_during_playback_is_suppressed_before_barge_in(self, caplog):
        class ScriptedGate:
            in_speech = False

            def __init__(self):
                self.calls = 0

            def process(self, _samples):
                self.calls += 1
                if self.calls == 1:
                    return [
                        SpeechFrameDecision(
                            probability=0.9, is_speech=True, speech_started=True
                        )
                    ]
                return [
                    SpeechFrameDecision(
                        probability=0.1, is_speech=False, candidate_ended=True
                    )
                ]

        output = FakeAudioOutput(playing=True, last_output_age_ms=100)
        output.echo = {"correlation": 0.9, "delay_ms": 330, "method": "envelope"}
        transcriber = FakePipelineTranscriber()
        pipeline = VoicePipeline(
            VoicePipelineConfig(play_audio=False, verbose=True),
            speech_gate=ScriptedGate(),
            endpoint_detector=FakeEndpointDetector(),
            transcriber=transcriber,
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=output,
        )

        async def run_echo():
            await pipeline._process_input_audio(np.ones(4, dtype=np.float32))
            await pipeline._process_input_audio(np.zeros(4, dtype=np.float32))
            await pipeline.mlx.shutdown()

        with caplog.at_level(logging.INFO, logger="mlx_audio.sts.voice_pipeline"):
            asyncio.run(run_echo())

        assert transcriber.started == 1
        assert transcriber.session is None
        assert output.flushed == 0
        assert "event=barge_in_candidate" in caplog.text
        assert "event=barge_in_suppressed_echo" in caplog.text

    def test_persistent_speech_during_playback_waits_for_transcript(self, caplog):
        class ScriptedGate:
            in_speech = False

            def __init__(self):
                self.calls = 0

            def process(self, _samples):
                self.calls += 1
                return [
                    SpeechFrameDecision(
                        probability=0.9,
                        is_speech=True,
                        speech_started=self.calls == 1,
                    )
                ]

        output = FakeAudioOutput(
            playing=True, buffered_samples=12_000, last_output_age_ms=100
        )
        output.echo = {"correlation": 0.1, "delay_ms": 330, "method": "raw"}
        transcriber = FakePipelineTranscriber()
        pipeline = VoicePipeline(
            VoicePipelineConfig(
                play_audio=False,
                verbose=True,
                min_barge_in_ms=0,
            ),
            speech_gate=ScriptedGate(),
            endpoint_detector=FakeEndpointDetector(),
            transcriber=transcriber,
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=output,
        )

        async def run_barge():
            await pipeline._process_input_audio(np.ones(4, dtype=np.float32))
            await pipeline._process_input_audio(np.ones(4, dtype=np.float32))
            await pipeline.mlx.shutdown()

        with caplog.at_level(logging.INFO, logger="mlx_audio.sts.voice_pipeline"):
            asyncio.run(run_barge())

        assert transcriber.started == 1
        assert output.flushed == 0
        assert "event=barge_in_candidate" in caplog.text
        assert "event=barge_in_waiting_for_transcript" in caplog.text
        assert "event=barge_in_confirmed" not in caplog.text

    def test_playback_candidate_matching_response_text_is_suppressed(self, caplog):
        output = FakeAudioOutput(
            playing=True, buffered_samples=12_000, last_output_age_ms=100
        )
        transcriber = FakePipelineTranscriber()
        transcriber.start()
        transcriber.text = "The cap."
        pipeline = VoicePipeline(
            VoicePipelineConfig(play_audio=False, verbose=True),
            speech_gate=SimpleNamespace(in_speech=True),
            endpoint_detector=FakeEndpointDetector(),
            transcriber=transcriber,
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=output,
        )
        pipeline._current_response_text = "The capital of France is Paris."
        pipeline._barge_candidate_started_at = time.monotonic() - 0.3
        pipeline._barge_candidate_reason = "playback_echo_window"
        pipeline._turn_audio = [np.ones(4, dtype=np.float32)]

        async def evaluate_candidate():
            handled = await pipeline._evaluate_barge_candidate_transcript(
                time.monotonic(), np.ones(4, dtype=np.float32)
            )
            await pipeline.mlx.shutdown()
            return handled

        with caplog.at_level(logging.INFO, logger="mlx_audio.sts.voice_pipeline"):
            handled = asyncio.run(evaluate_candidate())

        assert handled is True
        assert output.flushed == 0
        assert transcriber.session is None
        assert pipeline._turn_audio == []
        assert "event=barge_in_suppressed_echo" in caplog.text
        assert "reason=transcript_matches_playback" in caplog.text

    def test_distinct_playback_candidate_transcript_confirms_barge_in(self, caplog):
        output = FakeAudioOutput(
            playing=True, buffered_samples=12_000, last_output_age_ms=100
        )
        transcriber = FakePipelineTranscriber()
        transcriber.start()
        transcriber.text = "stop"
        pipeline = VoicePipeline(
            VoicePipelineConfig(play_audio=False, verbose=True),
            speech_gate=SimpleNamespace(in_speech=True),
            endpoint_detector=FakeEndpointDetector(),
            transcriber=transcriber,
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=output,
        )
        pipeline._current_response_text = "The capital of France is Paris."
        pipeline._barge_candidate_started_at = time.monotonic() - 0.3
        pipeline._barge_candidate_reason = "playback_echo_window"
        pipeline._turn_audio = [np.ones(4, dtype=np.float32)]

        async def evaluate_candidate():
            handled = await pipeline._evaluate_barge_candidate_transcript(
                time.monotonic(), np.ones(4, dtype=np.float32)
            )
            await pipeline.mlx.shutdown()
            return handled

        with caplog.at_level(logging.INFO, logger="mlx_audio.sts.voice_pipeline"):
            handled = asyncio.run(evaluate_candidate())

        assert handled is True
        assert output.flushed == 1
        assert transcriber.session is not None
        assert pipeline._turn_started_at is not None
        assert pipeline._barge_candidate_started_at is None
        assert "event=barge_in_confirmed" in caplog.text
        assert "reason=transcript" in caplog.text
        assert "event=barge_in" in caplog.text

    def test_vad_logging_emits_probability_events(self, caplog):
        class ScriptedGate:
            in_speech = False

            def process(self, _samples):
                return [
                    SpeechFrameDecision(
                        probability=0.36, is_speech=True, speech_started=True
                    )
                ]

        pipeline = VoicePipeline(
            VoicePipelineConfig(
                play_audio=False,
                verbose=True,
            ),
            speech_gate=ScriptedGate(),
            endpoint_detector=FakeEndpointDetector(),
            transcriber=FakePipelineTranscriber(),
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=FakeAudioOutput(),
        )

        async def run_log():
            await pipeline._process_input_audio(np.ones(4, dtype=np.float32))
            await pipeline.mlx.shutdown()

        with caplog.at_level(logging.INFO, logger="mlx_audio.sts.voice_pipeline"):
            asyncio.run(run_log())

        assert "event=vad" in caplog.text
        assert "probability=0.360" in caplog.text
        assert "speech_started=true" in caplog.text

    def test_logging_is_quiet_without_verbose(self, caplog):
        class ScriptedGate:
            in_speech = False

            def process(self, _samples):
                return [
                    SpeechFrameDecision(
                        probability=0.36, is_speech=True, speech_started=True
                    )
                ]

        pipeline = VoicePipeline(
            VoicePipelineConfig(play_audio=False),
            speech_gate=ScriptedGate(),
            endpoint_detector=FakeEndpointDetector(),
            transcriber=FakePipelineTranscriber(),
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=FakeAudioOutput(),
        )

        async def run_log():
            await pipeline._process_input_audio(np.ones(4, dtype=np.float32))
            await pipeline.mlx.shutdown()

        with caplog.at_level(logging.INFO, logger="mlx_audio.sts.voice_pipeline"):
            asyncio.run(run_log())

        assert "event=vad" not in caplog.text
        assert "event=speech_started" not in caplog.text

    def test_vad_logging_skips_non_transition_silence_by_default(self, caplog):
        class SilentGate:
            in_speech = False

            def process(self, _samples):
                return [SpeechFrameDecision(probability=0.01, is_speech=False)]

        pipeline = VoicePipeline(
            VoicePipelineConfig(
                play_audio=False,
                verbose=True,
            ),
            speech_gate=SilentGate(),
            endpoint_detector=FakeEndpointDetector(),
            transcriber=FakePipelineTranscriber(),
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=FakeAudioOutput(),
        )

        async def run_log():
            await pipeline._process_input_audio(np.zeros(4, dtype=np.float32))
            await pipeline.mlx.shutdown()

        with caplog.at_level(logging.INFO, logger="mlx_audio.sts.voice_pipeline"):
            asyncio.run(run_log())

        assert "event=vad" not in caplog.text

    def test_barge_in_cancels_tts_and_flushes_output(self):
        output = FakeAudioOutput()
        pipeline = VoicePipeline(
            VoicePipelineConfig(play_audio=False),
            speech_gate=SimpleNamespace(),
            endpoint_detector=SimpleNamespace(),
            transcriber=SimpleNamespace(active=False),
            response_engine=FakeResponseEngine(),
            tts_responder=PocketTTSResponder(FakeTTSModel()),
            audio_output=output,
        )

        async def run_barge_in():
            pipeline._current_tts_cancel = asyncio.Event()
            await pipeline.output_audio_queue.put(np.array([0.2], dtype=np.float32))
            await pipeline._handle_barge_in()
            return (
                pipeline._current_tts_cancel.is_set(),
                pipeline.output_audio_queue.empty(),
            )

        was_cancelled, queue_empty = asyncio.run(run_barge_in())

        assert was_cancelled is True
        assert queue_empty is True
        assert output.flushed == 1
