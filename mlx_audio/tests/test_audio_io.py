"""Tests for mlx_audio.audio_io module."""

import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlx_audio.audio_io import read, write

# Check if ffmpeg is available for optional tests
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


class TestAudioIOFormats:
    """Test audio I/O for various formats."""

    @pytest.fixture
    def sample_audio_mono(self):
        """Generate sample mono audio data (1 second, 16000 Hz)."""
        samplerate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(samplerate * duration))
        # Generate a simple sine wave at 440 Hz
        data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return data, samplerate

    @pytest.fixture
    def sample_audio_stereo(self):
        """Generate sample stereo audio data (1 second, 16000 Hz)."""
        samplerate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(samplerate * duration))
        # Generate two sine waves at different frequencies
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        data = np.column_stack((left, right))
        return data, samplerate

    def test_write_read_wav_mono(self, sample_audio_mono, tmp_path):
        """Test writing and reading mono WAV file."""
        data, samplerate = sample_audio_mono
        output_file = tmp_path / "test_mono.wav"

        write(output_file, data, samplerate, format="wav")
        assert output_file.exists()

        read_data, read_samplerate = read(output_file)
        assert read_samplerate == samplerate
        assert read_data.shape[0] == data.shape[0]

    def test_write_read_wav_stereo(self, sample_audio_stereo, tmp_path):
        """Test writing and reading stereo WAV file."""
        data, samplerate = sample_audio_stereo
        output_file = tmp_path / "test_stereo.wav"

        write(output_file, data, samplerate, format="wav")
        assert output_file.exists()

        read_data, read_samplerate = read(output_file)
        assert read_samplerate == samplerate
        assert read_data.shape == data.shape

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_write_read_mp3(self, sample_audio_mono, tmp_path):
        """Test writing and reading MP3 file."""
        data, samplerate = sample_audio_mono
        output_file = tmp_path / "test.mp3"

        write(output_file, data, samplerate, format="mp3")
        assert output_file.exists()

        read_data, read_samplerate = read(output_file)
        assert read_samplerate == samplerate
        # MP3 is lossy and may have padding, so we check shape is reasonable
        # (within 20% or 0.5 seconds, whichever is larger)
        tolerance = max(data.shape[0] * 0.2, samplerate * 0.5)
        assert abs(read_data.shape[0] - data.shape[0]) < tolerance

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_write_read_flac(self, sample_audio_stereo, tmp_path):
        """Test writing and reading FLAC file."""
        data, samplerate = sample_audio_stereo
        output_file = tmp_path / "test.flac"

        write(output_file, data, samplerate, format="flac")
        assert output_file.exists()

        read_data, read_samplerate = read(output_file)
        assert read_samplerate == samplerate
        assert read_data.shape == data.shape

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_write_read_ogg(self, sample_audio_mono, tmp_path):
        """Test writing and reading OGG Vorbis file."""
        data, samplerate = sample_audio_mono
        output_file = tmp_path / "test.ogg"

        write(output_file, data, samplerate, format="ogg")
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify we can read it back via ffmpeg
        read_data, read_samplerate = read(output_file)
        assert read_samplerate == samplerate
        # OGG Vorbis is lossy, check shape is reasonable
        tolerance = max(data.shape[0] * 0.2, samplerate * 0.5)
        assert abs(read_data.shape[0] - data.shape[0]) < tolerance

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_write_read_opus(self, sample_audio_stereo, tmp_path):
        """Test writing and reading Opus file."""
        data, samplerate = sample_audio_stereo
        output_file = tmp_path / "test.opus"

        write(output_file, data, samplerate, format="opus")
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify we can read it back via ffmpeg
        # Note: Opus internally uses 48kHz, so reading may return different sample rate
        read_data, read_samplerate = read(output_file)
        # Opus always decodes to 48kHz, but ffmpeg may resample back to original
        assert read_data.shape[0] > 0  # Just verify we got data

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_write_read_vorbis(self, sample_audio_mono, tmp_path):
        """Test writing and reading Vorbis (OGG) file."""
        data, samplerate = sample_audio_mono
        output_file = tmp_path / "test_vorbis.ogg"

        write(output_file, data, samplerate, format="vorbis")
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify we can read it back
        read_data, read_samplerate = read(output_file)
        assert read_samplerate == samplerate
        tolerance = max(data.shape[0] * 0.2, samplerate * 0.5)
        assert abs(read_data.shape[0] - data.shape[0]) < tolerance

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_write_bytesio_ogg(self, sample_audio_mono):
        """Test writing OGG to BytesIO."""
        data, samplerate = sample_audio_mono
        buffer = io.BytesIO()

        write(buffer, data, samplerate, format="ogg")
        assert buffer.getvalue()  # Should have content

        # Verify we can read it back
        buffer.seek(0)
        read_data, read_samplerate = read(buffer)
        assert read_samplerate == samplerate

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_write_bytesio_opus(self, sample_audio_stereo):
        """Test writing Opus to BytesIO."""
        data, samplerate = sample_audio_stereo
        buffer = io.BytesIO()

        write(buffer, data, samplerate, format="opus")
        assert buffer.getvalue()  # Should have content

        # Verify we can read it back
        buffer.seek(0)
        read_data, read_samplerate = read(buffer)
        assert read_data.shape[0] > 0  # Just verify we got data

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_format_inference_from_extension(self, sample_audio_mono, tmp_path):
        """Test format inference from file extension."""
        data, samplerate = sample_audio_mono

        # Test OGG extension (opus format needs explicit format parameter)
        output_file = tmp_path / "test.ogg"
        write(output_file, data, samplerate)  # No format specified
        assert output_file.exists(), "Failed to create OGG file"

        # Verify file can be read back
        read_data, read_samplerate = read(output_file)
        assert read_samplerate == samplerate

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_int16_input(self, sample_audio_mono, tmp_path):
        """Test writing with int16 input data."""
        data, samplerate = sample_audio_mono
        # Convert to int16
        data_int16 = (data * 32767).astype(np.int16)

        output_file = tmp_path / "test_int16.ogg"
        write(output_file, data_int16, samplerate, format="ogg")
        assert output_file.exists()

        read_data, read_samplerate = read(output_file)
        assert read_samplerate == samplerate

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_float64_input(self, sample_audio_stereo, tmp_path):
        """Test writing with float64 input data."""
        data, samplerate = sample_audio_stereo
        # Convert to float64
        data_float64 = data.astype(np.float64)

        output_file = tmp_path / "test_float64.ogg"
        write(output_file, data_float64, samplerate, format="ogg")
        assert output_file.exists()

        read_data, read_samplerate = read(output_file)
        assert read_samplerate == samplerate


class TestAudioIOEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_clipping(self, tmp_path):
        """Test that values outside [-1, 1] are clipped."""
        samplerate = 16000
        # Create data with values outside [-1, 1]
        data = np.array([1.5, -1.5, 0.5, -0.5], dtype=np.float32)

        output_file = tmp_path / "test_clipped.ogg"
        write(output_file, data, samplerate, format="ogg")
        assert output_file.exists()

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_empty_audio(self, tmp_path):
        """Test handling of very short audio."""
        samplerate = 16000
        # Create very short audio (just a few samples)
        data = np.array([0.0, 0.1, -0.1], dtype=np.float32)

        output_file = tmp_path / "test_short.opus"
        write(output_file, data, samplerate, format="opus")
        assert output_file.exists()

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_different_sample_rates(self, tmp_path):
        """Test various sample rates."""
        data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)

        for samplerate in [8000, 16000, 22050, 44100, 48000]:
            output_file = tmp_path / f"test_{samplerate}.ogg"
            write(output_file, data, samplerate, format="ogg")
            assert output_file.exists(), f"Failed at sample rate {samplerate}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
