from unittest import mock

import numpy as np
import pytest

from mlx_audio.sts.voice_pipeline import VoicePipeline


class TestVoicePipeline:
    def test_initialization_default_params(self):
        """
        Test that the initialization method initializes the parameters correctly.
        """
        pipeline = VoicePipeline()
        assert pipeline.silence_threshold == 0.03
        assert pipeline.silence_duration == 1.5
        assert pipeline.input_sample_rate == 16_000
        assert pipeline.output_sample_rate == 24_000
        assert pipeline.streaming_interval == 3
        assert pipeline.frame_duration_ms == 30
        assert pipeline.stt_model == "mlx-community/whisper-large-v3-turbo"
        assert pipeline.llm_model == "Qwen/Qwen2.5-0.5B-Instruct-4bit"
        assert pipeline.tts_model == "mlx-community/csm-1b-fp16"

    def test_initialization_custom_params(self):
        """
        Test that the initialization method initializes the parameters correctly.
        """
        pipeline = VoicePipeline(
            silence_threshold=0.05,
            silence_duration=2.0,
            input_sample_rate=8_000,
            output_sample_rate=12_000,
            streaming_interval=5,
            frame_duration_ms=20,
            vad_mode=2,
            stt_model="custom/stt",
            llm_model="custom/llm",
            tts_model="custom/tts",
        )
        assert pipeline.silence_threshold == 0.05
        assert pipeline.silence_duration == 2.0
        assert pipeline.input_sample_rate == 8_000
        assert pipeline.output_sample_rate == 12_000
        assert pipeline.streaming_interval == 5
        assert pipeline.frame_duration_ms == 20
        assert pipeline.stt_model == "custom/stt"
        assert pipeline.llm_model == "custom/llm"
        assert pipeline.tts_model == "custom/tts"

    @mock.patch("mlx_audio.sts.voice_pipeline.load_llm")
    @mock.patch("mlx_audio.sts.voice_pipeline.load_tts")
    @mock.patch("mlx_audio.sts.voice_pipeline.Whisper.from_pretrained")
    async def test_init_models(self, mock_whisper_load, mock_tts_load, mock_llm_load):
        """
        Test that the init_models method initializes the models correctly.
        """
        pipeline = VoicePipeline()

        # Mock the return values of the model loaders
        mock_llm = mock.AsyncMock()
        mock_tokenizer = mock.AsyncMock()
        mock_llm_load.return_value = (mock_llm, mock_tokenizer)

        mock_tts = mock.AsyncMock()
        mock_tts_load.return_value = mock_tts

        mock_stt = mock.AsyncMock()
        mock_whisper_load.return_value = mock_stt

        await pipeline.init_models()

        mock_llm_load.assert_called_once_with(pipeline.llm_model)
        mock_tts_load.assert_called_once_with(pipeline.tts_model)
        mock_whisper_load.assert_called_once_with(pipeline.stt_model)

        assert pipeline.llm is mock_llm
        assert pipeline.tokenizer is mock_tokenizer
        assert pipeline.tts is mock_tts
        assert pipeline.stt is mock_stt

    def test_is_silent_true(self):
        """
        Test that the is_silent method returns True for silent audio frames.
        """
        pipeline = VoicePipeline(silence_threshold=0.1)
        # Create a silent audio frame (very low amplitude)
        silent_audio_data_np = np.random.uniform(-0.01, 0.01, size=480).astype(
            np.float32
        )  # 30ms at 16kHz
        silent_audio_data_bytes = (
            (silent_audio_data_np * 32768.0).astype(np.int16).tobytes()
        )

        assert pipeline._is_silent(silent_audio_data_np) is np.True_
        assert pipeline._is_silent(silent_audio_data_bytes) is np.True_

    def test_is_silent_false(self):
        """
        Test that the is_silent method returns False for non-silent audio frames.
        """
        pipeline = VoicePipeline(silence_threshold=0.001)
        # Create a non-silent audio frame (higher amplitude)
        speech_audio_data_np = np.random.uniform(-2, 2, size=480).astype(np.float32)
        speech_audio_data_bytes = (
            (speech_audio_data_np * 32768.0).astype(np.int16).tobytes()
        )

        assert pipeline._is_silent(speech_audio_data_np) is np.False_
        assert pipeline._is_silent(speech_audio_data_bytes) is np.False_

    @mock.patch("webrtcvad.Vad.is_speech")
    def test_voice_activity_detection_vad_speech(self, mock_is_speech):
        """
        Test that the voice activity detection returns True for speech frames.
        """
        pipeline = VoicePipeline()
        mock_is_speech.return_value = True
        frame = b"\x00\x00" * (16000 * 30 // 1000)  # 30ms of silence at 16kHz, 16-bit
        assert pipeline._voice_activity_detection(frame) is True
        mock_is_speech.assert_called_once_with(frame, pipeline.input_sample_rate)

    @mock.patch("webrtcvad.Vad.is_speech")
    def test_voice_activity_detection_vad_silence(self, mock_is_speech):
        """
        Test that the voice activity detection returns False for silent frames.
        """
        pipeline = VoicePipeline()
        mock_is_speech.return_value = False
        frame = b"\x00\x00" * (16000 * 30 // 1000)
        assert pipeline._voice_activity_detection(frame) is False
        mock_is_speech.assert_called_once_with(frame, pipeline.input_sample_rate)

    @mock.patch("webrtcvad.Vad.is_speech")
    def test_voice_activity_detection_vad_error_fallback_silent(self, mock_is_speech):
        """
        Test that the voice activity detection returns False for silent frames.
        """
        pipeline = VoicePipeline(silence_threshold=0.1)
        mock_is_speech.side_effect = ValueError("VAD error")

        frame_np = np.full(480, 0.001, dtype=np.float32)
        frame_bytes = (frame_np * 32768.0).astype(np.int16).tobytes()

        assert pipeline._voice_activity_detection(frame_bytes) is False
        mock_is_speech.assert_called_once_with(frame_bytes, pipeline.input_sample_rate)

    @mock.patch("webrtcvad.Vad.is_speech")
    def test_voice_activity_detection_vad_error_fallback_speech(self, mock_is_speech):
        pipeline = VoicePipeline(silence_threshold=0.01)
        mock_is_speech.side_effect = ValueError("VAD error")
        frame_np = np.full(480, 0.5, dtype=np.float32)
        frame_bytes = (frame_np * 32768.0).astype(np.int16).tobytes()

        assert pipeline._voice_activity_detection(frame_bytes) is True
        mock_is_speech.assert_called_once_with(frame_bytes, pipeline.input_sample_rate)
