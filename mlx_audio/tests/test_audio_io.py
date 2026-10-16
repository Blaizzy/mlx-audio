"""Tests for mlx_audio.audio_io module."""

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlx_audio.audio_io import read, write

# Check if ffmpeg is available for optional tests
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def _ffmpeg_has_encoder(name):
    if not FFMPEG_AVAILABLE:
        return False
    listing = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return f" {name} " in listing.stdout


LIBVORBIS_AVAILABLE = _ffmpeg_has_encoder("libvorbis")


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

    def test_read_wav_target_sample_rate_and_channels(
        self, sample_audio_stereo, tmp_path
    ):
        """Test target-rate loading and decoder-side mono conversion."""
        data, samplerate = sample_audio_stereo
        output_file = tmp_path / "test_resampled_mono.wav"
        target_samplerate = samplerate // 2

        write(output_file, data, samplerate, format="wav")

        read_data, read_samplerate = read(
            output_file,
            dtype="float32",
            sample_rate=target_samplerate,
            nchannels=1,
        )
        assert read_samplerate == target_samplerate
        assert read_data.dtype == np.float32
        assert read_data.ndim == 1
        assert abs(read_data.shape[0] - target_samplerate) <= 1

    @pytest.mark.parametrize("use_buffer", [False, True])
    def test_miniaudio_downsampling_rejects_alias(self, tmp_path, use_buffer):
        """Miniaudio inputs use the high-quality FIR when downsampling."""
        source_rate = 44100
        target_rate = 16000
        t = np.arange(2 * source_rate) / source_rate
        tone = (0.5 * np.sin(2 * np.pi * 12000.0 * t)).astype(np.float32)

        if use_buffer:
            audio_input = io.BytesIO()
            write(audio_input, tone, source_rate, format="wav")
        else:
            audio_input = tmp_path / "out_of_band.wav"
            write(audio_input, tone, source_rate, format="wav")

        loaded, loaded_rate = read(
            audio_input,
            dtype="float32",
            sample_rate=target_rate,
            nchannels=1,
        )
        loaded = loaded[len(loaded) // 4 : -len(loaded) // 4]
        rms = float(np.sqrt(np.mean(loaded.astype(np.float64) ** 2)))

        assert loaded_rate == target_rate
        assert rms < 0.001

    @pytest.mark.parametrize("source_rate,frequency", [(16000, 7000), (24000, 10000)])
    def test_load_audio_upsampling_rejects_images(
        self, tmp_path, source_rate, frequency
    ):
        """Upsampling a reference must not add images above its original bandwidth."""
        import wave

        from mlx_audio.utils import load_audio, resample_audio

        target_rate = 44100
        t = np.arange(2 * source_rate) / source_rate
        pcm = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        path = tmp_path / "upsample.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(source_rate)
            wav.writeframes(pcm.tobytes())

        loaded = np.asarray(load_audio(str(path), sample_rate=target_rate))
        expected = resample_audio(
            pcm.astype(np.float32) / 32768, source_rate, target_rate
        )
        assert loaded.shape == (2 * target_rate,)
        center = loaded[len(loaded) // 4 : -len(loaded) // 4].astype(np.float64)
        spectrum = np.abs(np.fft.rfft(center * np.hanning(len(center)))) ** 2
        freqs = np.fft.rfftfreq(len(center), 1 / target_rate)
        near_tone = np.abs(freqs - frequency) < 50
        image_ratio = (
            spectrum[freqs > source_rate / 2 + 50].sum() / spectrum[near_tone].sum()
        )
        spur_ratio = (
            spectrum[(freqs < source_rate / 2 - 50) & ~near_tone].max()
            / spectrum[near_tone].max()
        )
        assert image_ratio < 1e-6
        assert spur_ratio < 1e-6
        np.testing.assert_allclose(loaded, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("use_buffer", [False, True])
    @pytest.mark.parametrize("channels", [1, 2])
    def test_miniaudio_upsampling_matches_fir(self, tmp_path, use_buffer, channels):
        """Both file and memory inputs retain channel layout through chunked FIR."""
        from mlx_audio.resample import resample_audio_array

        source_rate, target_rate = 16000, 44100
        t = np.arange(2 * source_rate) / source_rate
        tone = (0.5 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)
        if channels == 2:
            tone = np.column_stack((tone, tone * 0.5))
        source = io.BytesIO() if use_buffer else tmp_path / "channels.wav"
        write(source, tone, source_rate, format="wav")
        native, _ = read(source, dtype="float32")
        actual, rate = read(source, dtype="float32", sample_rate=target_rate)
        expected = resample_audio_array(native, source_rate, target_rate, axis=0)
        assert rate == target_rate
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_miniaudio_same_rate_keeps_native_samples(self, tmp_path, monkeypatch):
        """Requesting the source rate does not introduce a FIR conversion."""
        import mlx_audio.resample as resample

        def unexpected_resample(*args, **kwargs):
            raise AssertionError("same-rate decoding must not resample")

        monkeypatch.setattr(resample, "resample_audio_chunks", unexpected_resample)
        t = np.arange(1600) / 16000
        path = tmp_path / "native.wav"
        write(path, np.sin(2 * np.pi * 440 * t).astype(np.float32), 16000, format="wav")
        native, _ = read(path, dtype="float32")
        requested, rate = read(path, dtype="float32", sample_rate=16000)
        assert rate == 16000
        np.testing.assert_array_equal(requested, native)

    def test_stt_load_audio_rejects_alias(self, tmp_path):
        """Protect the public STT loader path reported in issue #870."""
        from mlx_audio.stt.utils import load_audio

        source_rate = 44100
        target_rate = 16000
        t = np.arange(2 * source_rate) / source_rate
        tone = (0.5 * np.sin(2 * np.pi * 12000.0 * t)).astype(np.float32)
        audio_path = tmp_path / "stt_out_of_band.wav"
        write(audio_path, tone, source_rate, format="wav")

        loaded = np.asarray(load_audio(str(audio_path), sr=target_rate))
        loaded = loaded[len(loaded) // 4 : -len(loaded) // 4]
        rms = float(np.sqrt(np.mean(loaded.astype(np.float64) ** 2)))

        assert rms < 0.001

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
    def test_write_read_webm(self, sample_audio_mono, tmp_path):
        """Test writing and reading WebM file."""
        data, samplerate = sample_audio_mono
        output_file = tmp_path / "test.webm"

        write(output_file, data, samplerate, format="webm")
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify we can read it back via ffmpeg
        # Note: WebM with Opus internally uses 48kHz, so reading may return different sample rate
        read_data, read_samplerate = read(output_file)
        assert read_data.shape[0] > 0  # Just verify we got data

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_write_read_webm_stereo(self, sample_audio_stereo, tmp_path):
        """Test writing and reading stereo WebM file."""
        data, samplerate = sample_audio_stereo
        output_file = tmp_path / "test_stereo.webm"

        write(output_file, data, samplerate, format="webm")
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        read_data, read_samplerate = read(output_file)
        assert read_data.shape[0] > 0
        # WebM/Opus may change channel count, just verify data is returned
        assert read_data.ndim >= 1

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
    def test_write_bytesio_webm(self, sample_audio_mono):
        """Test writing WebM to BytesIO and reading it back (simulates browser blob)."""
        data, samplerate = sample_audio_mono
        buffer = io.BytesIO()

        write(buffer, data, samplerate, format="webm")
        assert buffer.getvalue()  # Should have content

        # Verify we can read it back (this is the browser blob path)
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
        data = np.tile([1.5, -1.5, 0.5, -0.5], 1024).astype(np.float32)

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
    @pytest.mark.parametrize("use_buffer", [False, True])
    def test_ffmpeg_without_samples_raises(self, tmp_path, use_buffer):
        """ffmpeg exits 0 and writes nothing for a CAF with no frames."""
        path = _encode_with_ffmpeg(
            tmp_path / "empty.caf", "-c:a", "pcm_s16le", seconds=0
        )
        source = io.BytesIO(path.read_bytes()) if use_buffer else path

        with pytest.raises(RuntimeError, match="no audio samples"):
            read(source)

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


class TestAudioIOCAF:
    """Test reading Core Audio Format, the macOS native recording container."""

    @pytest.fixture
    def caf_file(self, tmp_path):
        """Encode one second of 440 Hz mono audio into a CAF container."""
        samplerate = 16000
        t = np.linspace(0, 1.0, samplerate, endpoint=False)
        pcm = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
        path = tmp_path / "recording.caf"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "s16le",
                "-ar",
                str(samplerate),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-c:a",
                "pcm_s16le",
                str(path),
            ],
            input=pcm.tobytes(),
            capture_output=True,
            check=True,
        )
        return path, samplerate

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_read_caf_path(self, caf_file):
        path, samplerate = caf_file

        data, read_samplerate = read(path)

        assert read_samplerate == samplerate
        assert data.shape[0] > 0

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_read_caf_bytesio(self, caf_file):
        """Uploads arrive as bytes, so the caff magic alone must route to ffmpeg."""
        path, samplerate = caf_file

        data, read_samplerate = read(io.BytesIO(path.read_bytes()))

        assert read_samplerate == samplerate
        assert data.shape[0] > 0


def _encode_with_ffmpeg(
    path, *codec_args, samplerate=16000, channels=1, seconds=1.0, mono=None
):
    """Encode a 440 Hz tone, or the int16 samples in `mono`, with ffmpeg.

    ffmpeg picks the container from the path.
    """
    if mono is None:
        t = np.arange(int(samplerate * seconds)) / samplerate
        mono = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    pcm = np.repeat(mono[:, None], channels, axis=1)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", str(samplerate), "-ac", str(channels)]
        + ["-i", "pipe:0", *codec_args, str(path)],
        input=pcm.tobytes(),
        capture_output=True,
        check=True,
    )
    return path


def _mp4_top_level_atoms(data):
    atoms, offset = [], 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        atoms.append(data[offset + 4 : offset + 8].decode("latin-1"))
        if size < 8:
            break
        offset += size
    return atoms


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
class TestAudioIOBytesThroughFFmpeg:
    """Bytes that need ffmpeg are decoded from a seekable temporary file."""

    def test_m4a_with_moov_at_end(self, tmp_path):
        # Large enough that ffmpeg cannot seek back to mdat inside its pipe buffer.
        path = _encode_with_ffmpeg(
            tmp_path / "moov_last.m4a",
            "-c:a",
            "aac",
            samplerate=44100,
            channels=2,
            seconds=10.0,
        )
        data = path.read_bytes()
        atoms = _mp4_top_level_atoms(data)
        assert atoms.index("moov") > atoms.index("mdat")

        from_bytes, rate = read(io.BytesIO(data), dtype="int16")
        from_path, _ = read(path, dtype="int16")

        assert rate == 44100
        assert from_bytes.shape[0] > 0
        np.testing.assert_array_equal(from_bytes, from_path)

    @pytest.mark.skipif(shutil.which("afconvert") is None, reason="needs afconvert")
    @pytest.mark.parametrize("codec", ["aac", "LEI16"])
    def test_afconvert_caf(self, tmp_path, codec):
        """macOS writes CAF files that ffmpeg cannot read from a pipe."""
        wav = tmp_path / "clip.wav"
        write(wav, np.sin(np.arange(5 * 44100) / 10.0).astype(np.float32), 44100)
        caf = tmp_path / "clip.caf"
        subprocess.run(
            ["afconvert", "-f", "caff", "-d", codec, str(wav), str(caf)],
            capture_output=True,
            check=True,
        )

        from_bytes, _ = read(io.BytesIO(caf.read_bytes()), dtype="int16")
        from_path, _ = read(caf, dtype="int16")

        assert from_bytes.shape[0] > 0
        np.testing.assert_array_equal(from_bytes, from_path)

    def test_temporary_file_is_removed(self, tmp_path, monkeypatch):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(scratch))
        path = _encode_with_ffmpeg(tmp_path / "clip.m4a", "-c:a", "aac")

        read(io.BytesIO(path.read_bytes()))
        with pytest.raises(RuntimeError):
            read(io.BytesIO(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64))

        assert list(scratch.iterdir()) == []


class TestAudioIOFormatDetection:
    """The first bytes pick the decoder; the extension is only a fallback."""

    @staticmethod
    def _assert_detected(path, wrong_suffix, tmp_path):
        expected, rate = read(path, dtype="int16")
        assert expected.shape[0] > 0

        no_suffix = tmp_path / "upload"
        no_suffix.write_bytes(path.read_bytes())
        wrong = tmp_path / f"upload{wrong_suffix}"
        wrong.write_bytes(path.read_bytes())

        for source in (no_suffix, wrong, io.BytesIO(path.read_bytes())):
            data, read_rate = read(source, dtype="int16")
            assert read_rate == rate
            np.testing.assert_array_equal(data, expected)

    def test_wav(self, tmp_path):
        path = tmp_path / "clip.wav"
        write(path, np.sin(np.arange(16000) / 10.0).astype(np.float32), 16000)
        self._assert_detected(path, ".mp3", tmp_path)

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    @pytest.mark.parametrize(
        "name, codec_args, samplerate",
        [
            pytest.param("clip.aiff", ["-c:a", "pcm_s16be"], 16000, id="aiff"),
            pytest.param("clip.aifc", ["-c:a", "pcm_f32be"], 16000, id="aifc"),
            pytest.param("clip.flac", [], 16000, id="flac"),
            pytest.param("clip.caf", ["-c:a", "pcm_s16le"], 16000, id="caf"),
            pytest.param("clip.m4a", ["-c:a", "aac"], 16000, id="m4a"),
            pytest.param("clip.ogg", ["-c:a", "flac"], 16000, id="ogg"),
            pytest.param("clip.mp3", ["-id3v2_version", "0"], 44100, id="mp3-44k"),
            pytest.param("clip.mp3", ["-id3v2_version", "0"], 22050, id="mp3-22k"),
            pytest.param("clip.mp3", ["-id3v2_version", "0"], 16000, id="mp3-16k"),
            pytest.param(
                "clip.mp3", ["-metadata", "title=clip"], 16000, id="mp3-16k-id3"
            ),
            pytest.param("clip.aac", ["-f", "adts"], 16000, id="adts"),
            pytest.param(
                "clip.aac",
                ["-f", "adts", "-write_id3v2", "1", "-metadata", "title=clip"],
                16000,
                id="adts-id3",
            ),
        ],
    )
    def test_ffmpeg_encoded(self, tmp_path, name, codec_args, samplerate):
        path = _encode_with_ffmpeg(tmp_path / name, *codec_args, samplerate=samplerate)
        self._assert_detected(path, ".wav", tmp_path)

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_extension_is_the_fallback(self, tmp_path):
        """Bytes that name no format still decode when the extension does."""
        source = _encode_with_ffmpeg(tmp_path / "source.mp3", "-id3v2_version", "0")
        path = tmp_path / "padded.mp3"
        path.write_bytes(b"\x00" * 16 + source.read_bytes())

        data, rate = read(path)

        assert rate == 16000
        assert data.shape[0] > 0

    @staticmethod
    def _id3_tag(size):
        """An ID3v2.4 tag with `size` zero bytes of content."""
        return (
            b"ID3\x04\x00\x00\x00\x00" + bytes([size >> 7, size & 0x7F]) + bytes(size)
        )

    @pytest.mark.parametrize("padding", [0, 1000], ids=["tags", "tags-padding"])
    def test_detection_looks_past_id3_tags(self, padding):
        """ID3v2 tags and zero padding in front of the signature are skipped."""
        from mlx_audio.audio_io import _detect_format

        tags = self._id3_tag(300) + self._id3_tag(5) + bytes(padding)

        assert _detect_format(io.BytesIO(tags + b"fLaC" + bytes(8))) == "flac"
        # Unknown bytes, or padding past the limit: the caller's fallback wins.
        for rest in (b"unknown bytes", bytes(100_000) + b"fLaC"):
            assert _detect_format(io.BytesIO(tags + rest), after_id3="ogg") == "ogg"

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    @pytest.mark.parametrize(
        "name, codec_args",
        [
            pytest.param("clip.aac", ["-f", "adts"], id="adts"),
            pytest.param("clip.flac", [], id="flac"),
            pytest.param("clip.ogg", ["-c:a", "flac"], id="ogg"),
            pytest.param(
                "clip.ogg",
                ["-c:a", "libvorbis"],
                id="vorbis",
                marks=pytest.mark.skipif(
                    not LIBVORBIS_AVAILABLE, reason="ffmpeg has no libvorbis encoder"
                ),
            ),
            pytest.param("clip.opus", ["-c:a", "libopus"], id="opus"),
        ],
    )
    def test_two_id3_tags(self, tmp_path, name, codec_args):
        """A second ID3v2 tag does not turn the file into an MP3."""
        plain = _encode_with_ffmpeg(tmp_path / name, *codec_args)
        tagged = tmp_path / f"tagged{plain.suffix}"
        tagged.write_bytes(self._id3_tag(300) + self._id3_tag(120) + plain.read_bytes())

        np.testing.assert_array_equal(
            read(tagged, dtype="int16")[0], read(plain, dtype="int16")[0]
        )
        self._assert_detected(tagged, ".wav", tmp_path)

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_padding_after_id3_tag(self, tmp_path):
        """Zero bytes between the tag and the audio do not turn the file into an MP3."""
        plain = _encode_with_ffmpeg(tmp_path / "clip.aac", "-f", "adts")
        tagged = tmp_path / "tagged.aac"
        tagged.write_bytes(self._id3_tag(64) + bytes(417) + plain.read_bytes())

        np.testing.assert_array_equal(
            read(tagged, dtype="int16")[0], read(plain, dtype="int16")[0]
        )
        self._assert_detected(tagged, ".wav", tmp_path)

    def test_extension_names_the_format_after_id3_tag(self, tmp_path, monkeypatch):
        """Unknown bytes after the tag: an .ogg file goes to ffmpeg, not to MP3."""
        from mlx_audio import audio_io

        decoded = []

        def fake_ffmpeg(input_data, **kwargs):
            decoded.append(input_data)
            return np.zeros(16000, dtype=np.int16), 16000, 1

        monkeypatch.setattr(audio_io, "_decode_ffmpeg", fake_ffmpeg)
        path = tmp_path / "clip.ogg"
        path.write_bytes(self._id3_tag(64) + b"unknown bytes")

        data, rate = read(path)

        assert decoded == [path]
        assert (data.shape, rate) == ((16000,), 16000)

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_extension_is_tried_when_the_detected_decoder_fails(self, tmp_path):
        """MP3 in a WAV container: the bytes say WAV, but only other decoders read it."""
        from miniaudio import DecodeError

        riff = _encode_with_ffmpeg(tmp_path / "riff.wav", "-c:a", "libmp3lame")
        assert riff.read_bytes()[:4] == b"RIFF"

        for suffix in (".mp3", ".m4a", ".ogg"):
            path = tmp_path / f"clip{suffix}"
            path.write_bytes(riff.read_bytes())
            data, rate = read(path)
            assert rate == 16000
            assert data.shape[0] > 0

        # No other decoder to try, and the first error is the one raised.
        with pytest.raises(DecodeError):
            read(riff)
        not_m4a = tmp_path / "not_m4a.mp3"
        not_m4a.write_bytes(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64)
        with pytest.raises(RuntimeError):
            read(not_m4a)

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_bytes_with_id3_tag_fall_back_to_mp3(self, tmp_path):
        """An MP3 whose tag is followed by another format's signature still decodes."""
        source = _encode_with_ffmpeg(tmp_path / "source.mp3", "-id3v2_version", "0")
        tagged = self._id3_tag(64) + b"RIFF\x00\x00\x00\x00WAVE" + source.read_bytes()

        data, rate = read(io.BytesIO(tagged))

        assert rate == 16000
        assert data.shape[0] > 0

    @pytest.mark.skipif(
        not LIBVORBIS_AVAILABLE, reason="ffmpeg has no libvorbis encoder"
    )
    def test_vorbis_extension_without_ffmpeg(self, tmp_path, monkeypatch):
        """Ogg goes to ffmpeg; a .vorbis file falls back to miniaudio without it."""
        source = _encode_with_ffmpeg(tmp_path / "clip.ogg", "-c:a", "libvorbis")
        path = tmp_path / "clip.vorbis"
        path.write_bytes(source.read_bytes())
        monkeypatch.setattr(shutil, "which", lambda name: None)

        data, rate = read(path)

        assert rate == 16000
        assert data.shape[0] > 0

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_vorbis_extension_round_trip(self, tmp_path):
        """write() puts FLAC in Ogg for .vorbis; the Ogg bytes route it to ffmpeg."""
        path = tmp_path / "clip.vorbis"
        tone = np.sin(np.arange(16000) / 10.0).astype(np.float32)
        write(path, tone, 16000)

        data, rate = read(path)

        assert rate == 16000
        assert data.shape[0] == tone.shape[0]


class TestAudioIODuration:
    """duration decodes only the start of a file, to the same samples."""

    @pytest.fixture
    def wav_path(self, tmp_path):
        path = tmp_path / "clip.wav"
        rng = np.random.default_rng(0)
        write(path, rng.uniform(-0.5, 0.5, (5 * 44100, 2)).astype(np.float32), 44100)
        return path

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"sample_rate": 16000, "nchannels": 1}, {"sample_rate": 48000}],
        ids=["native", "downsampled", "upsampled"],
    )
    @pytest.mark.parametrize("use_buffer", [False, True])
    def test_wav_matches_full_read(self, wav_path, kwargs, use_buffer):
        full, rate = read(wav_path, dtype="float32", **kwargs)
        source = io.BytesIO(wav_path.read_bytes()) if use_buffer else wav_path

        part, part_rate = read(source, dtype="float32", duration=1.5, **kwargs)

        assert part_rate == rate
        np.testing.assert_array_equal(part, full[: round(1.5 * rate)])

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    @pytest.mark.parametrize(
        "kwargs", [{}, {"sample_rate": 16000, "nchannels": 1}], ids=["native", "16k"]
    )
    @pytest.mark.parametrize("use_buffer", [False, True])
    def test_m4a_matches_full_read(self, tmp_path, kwargs, use_buffer):
        path = _encode_with_ffmpeg(
            tmp_path / "clip.m4a", "-c:a", "aac", samplerate=44100, seconds=5.0
        )
        full, rate = read(path, dtype="float32", **kwargs)
        source = io.BytesIO(path.read_bytes()) if use_buffer else path

        part, part_rate = read(source, dtype="float32", duration=1.5, **kwargs)

        assert part_rate == rate
        np.testing.assert_array_equal(part, full[: round(1.5 * rate)])

    @pytest.mark.skipif(
        not LIBVORBIS_AVAILABLE, reason="ffmpeg has no libvorbis encoder"
    )
    @pytest.mark.parametrize(
        "kwargs", [{}, {"sample_rate": 8000}], ids=["native", "8k"]
    )
    def test_ogg_vorbis_matches_full_read(self, tmp_path, kwargs):
        """ffmpeg's Ogg Vorbis timestamps run late where the block size changes."""
        # A click every 0.1 s makes libvorbis switch between its two block sizes.
        clicks = np.zeros(2 * 16000, dtype=np.int16)
        clicks[::1600] = 20000
        path = _encode_with_ffmpeg(
            tmp_path / "clip.ogg", "-c:a", "libvorbis", mono=clicks
        )
        full, rate = read(path, dtype="int16", **kwargs)

        for step in range(8):
            duration = 1.0 + step * 0.0125
            part, _ = read(path, dtype="int16", duration=duration, **kwargs)
            np.testing.assert_array_equal(part, full[: round(duration * rate)])

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_timestamp_gap_matches_full_read(self, tmp_path):
        """A recording with a pause: its timestamps skip a second, its samples do not."""
        path = _encode_with_ffmpeg(
            tmp_path / "clip.m4a",
            "-af",
            "aselect='not(between(t,1,2))'",
            "-c:a",
            "aac",
            seconds=4.0,
        )
        full, rate = read(path, dtype="int16")
        assert 2.5 * rate < full.shape[0] < 3.5 * rate

        part, _ = read(path, dtype="int16", duration=2.5)

        np.testing.assert_array_equal(part, full[: round(2.5 * rate)])

    @pytest.mark.parametrize(
        "kwargs", [{}, {"sample_rate": 8000}, {"sample_rate": 48000}]
    )
    def test_miniaudio_stops_decoding(self, tmp_path, monkeypatch, kwargs):
        from mlx_audio import audio_io

        path = tmp_path / "long.wav"
        write(path, np.zeros(60 * 16000, dtype=np.float32), 16000)
        stream = audio_io._stream_miniaudio
        chunks = []

        def counting_stream(*args):
            for chunk in stream(*args):
                chunks.append(chunk)
                yield chunk

        monkeypatch.setattr(audio_io, "_stream_miniaudio", counting_stream)
        data, rate = read(path, duration=2.0, **kwargs)

        assert data.shape[0] == 2 * rate
        assert len(chunks) <= 4  # of 60 one-second chunks

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_ffmpeg_gets_time_limit(self, tmp_path, monkeypatch):
        path = _encode_with_ffmpeg(tmp_path / "clip.m4a", "-c:a", "aac", seconds=3.0)
        commands = []
        run = subprocess.run

        def recording_run(cmd, *args, **kwargs):
            commands.append(cmd)
            return run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", recording_run)
        data, rate = read(path, duration=1.0)

        assert data.shape[0] == rate
        assert "-t" in commands[-1]

    @pytest.mark.parametrize(
        "ext",
        [
            "wav",
            pytest.param(
                "m4a",
                marks=pytest.mark.skipif(
                    not FFMPEG_AVAILABLE, reason="ffmpeg not installed"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "kwargs", [{}, {"sample_rate": 16000}], ids=["native", "16k"]
    )
    def test_tiny_and_huge_limits(self, wav_path, tmp_path, ext, kwargs):
        if ext == "m4a":
            path = _encode_with_ffmpeg(
                tmp_path / "clip.m4a", "-c:a", "aac", samplerate=44100, seconds=2.0
            )
        else:
            path = wav_path
        full, _ = read(path, **kwargs)

        assert read(path, duration=1e-5, **kwargs)[0].shape == (0, *full.shape[1:])
        np.testing.assert_array_equal(read(path, duration=1e300, **kwargs)[0], full)

    @pytest.mark.parametrize("duration", [0, -1.0, float("inf"), float("nan")])
    def test_rejects_invalid(self, wav_path, duration):
        with pytest.raises(ValueError, match="duration"):
            read(wav_path, duration=duration)

    def test_load_audio_helpers(self, wav_path):
        from mlx_audio.stt.utils import load_audio as stt_load_audio
        from mlx_audio.utils import load_audio

        assert stt_load_audio(str(wav_path), sr=16000, duration=0.5).shape == (8000,)
        assert load_audio(str(wav_path), sample_rate=16000, duration=0.5).shape == (
            8000,
        )
