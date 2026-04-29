"""Tests for DeepFilterNet STS model."""

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np


class TestDeepFilterNetConfig(unittest.TestCase):
    """Config serialization/default tests."""

    def test_defaults(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNetConfig

        cfg = DeepFilterNetConfig()
        self.assertEqual(cfg.sample_rate, 48000)
        self.assertEqual(cfg.fft_size, 960)
        self.assertEqual(cfg.hop_size, 480)
        self.assertEqual(cfg.nb_erb, 32)
        self.assertEqual(cfg.nb_df, 96)
        self.assertEqual(cfg.freq_bins, 481)

    def test_from_dict_and_to_dict(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNetConfig

        d = {"sample_rate": 44100, "nb_df": 64, "df_order": 3}
        cfg = DeepFilterNetConfig.from_dict(d)
        self.assertEqual(cfg.sample_rate, 44100)
        self.assertEqual(cfg.nb_df, 64)
        self.assertEqual(cfg.df_order, 3)

        out = cfg.to_dict()
        self.assertEqual(out["sample_rate"], 44100)
        self.assertEqual(out["nb_df"], 64)
        self.assertEqual(out["df_order"], 3)


class TestDeepFilterNetForward(unittest.TestCase):
    """Forward-pass shape tests for DF2/DF3 and DF1 backends."""

    def _run_forward(self, config, model_cls):
        model = model_cls(config)
        t = 8
        f = config.fft_size // 2 + 1
        spec = mx.zeros((1, 1, t, f, 2), dtype=mx.float32)
        feat_erb = mx.zeros((1, 1, t, config.nb_erb), dtype=mx.float32)
        feat_df = mx.zeros((1, 1, t, config.nb_df, 2), dtype=mx.float32)

        spec_e, m, lsnr, df_coefs = model(spec, feat_erb, feat_df)
        mx.eval(spec_e, m, lsnr, df_coefs)

        self.assertEqual(spec_e.shape, spec.shape)
        self.assertEqual(m.shape[0], 1)
        self.assertEqual(lsnr.shape[0], 1)
        self.assertEqual(df_coefs.shape[0], 1)

    def test_df2_like_forward(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNet2Config
        from mlx_audio.sts.models.deepfilternet.network import DfNet

        self._run_forward(DeepFilterNet2Config(), DfNet)

    def test_df3_like_forward(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNet3Config
        from mlx_audio.sts.models.deepfilternet.network import DfNet

        self._run_forward(DeepFilterNet3Config(), DfNet)

    def test_df1_forward(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNetConfig
        from mlx_audio.sts.models.deepfilternet.network_df1 import DfNetV1

        self._run_forward(DeepFilterNetConfig(), DfNetV1)


class TestDeepFilterNetRuntimeHelpers(unittest.TestCase):
    """Tests for runtime helper behavior without external weights."""

    def test_vorbis_window_shape(self):
        from mlx_audio.sts.models.deepfilternet.model import DeepFilterNetModel

        w = DeepFilterNetModel._vorbis_window(960)
        mx.eval(w)
        self.assertEqual(w.shape[0], 960)
        self.assertGreaterEqual(float(mx.min(w)), 0.0)
        self.assertLessEqual(float(mx.max(w)), 1.0)

    def test_from_pretrained_rejects_missing_dir(self):
        from mlx_audio.sts.models.deepfilternet.model import DeepFilterNetModel

        with self.assertRaises(Exception):
            DeepFilterNetModel.from_pretrained("/definitely/not/a/model/dir")

    def test_from_pretrained_rejects_file_path(self):
        from mlx_audio.sts.models.deepfilternet.model import DeepFilterNetModel

        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "model.safetensors"
            file_path.touch()
            with self.assertRaises(ValueError):
                DeepFilterNetModel.from_pretrained(str(file_path))

    def test_streamer_rejects_df1_backend(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNetConfig
        from mlx_audio.sts.models.deepfilternet.model import DeepFilterNetModel
        from mlx_audio.sts.models.deepfilternet.network_df1 import DfNetV1

        cfg = DeepFilterNetConfig()
        runtime = DeepFilterNetModel(
            model=DfNetV1(cfg), config=cfg, model_dir=Path(".")
        )
        with self.assertRaises(NotImplementedError):
            runtime.create_streamer()

    def test_streamer_df3_runs(self):
        from mlx_audio.sts.models.deepfilternet.config import DeepFilterNet3Config
        from mlx_audio.sts.models.deepfilternet.model import DeepFilterNetModel
        from mlx_audio.sts.models.deepfilternet.network import DfNet

        cfg = DeepFilterNet3Config()
        runtime = DeepFilterNetModel(model=DfNet(cfg), config=cfg, model_dir=Path("."))
        streamer = runtime.create_streamer()

        # 20 ms of silence at 48 kHz.
        x = np.zeros((960,), dtype=np.float32)
        y1 = streamer.process_chunk(x, is_last=False)
        y2 = streamer.flush()
        self.assertIsInstance(y1, np.ndarray)
        self.assertIsInstance(y2, np.ndarray)


class TestDeepFilterNetIntegration(unittest.TestCase):
    """End-to-end integration tests with real model weights.

    These tests download the pretrained DeepFilterNet3 MLX weights from
    HuggingFace and run inference on the included sample audio file.
    They are skipped when the model cannot be loaded (e.g. no network).
    """

    REPO_ROOT = Path(__file__).resolve().parents[3]
    SAMPLE_AUDIO = REPO_ROOT / "examples" / "denoise" / "noisey_audio_10s.wav"

    @classmethod
    def setUpClass(cls):
        if not cls.SAMPLE_AUDIO.exists():
            raise unittest.SkipTest(f"Sample audio not found: {cls.SAMPLE_AUDIO}")
        try:
            from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

            cls.model = DeepFilterNetModel.from_pretrained()
        except Exception as e:
            raise unittest.SkipTest(f"Could not load pretrained model: {e}")

        from mlx_audio.audio_io import read as audio_read

        audio, sr = audio_read(str(cls.SAMPLE_AUDIO))
        if audio.ndim > 1:
            audio = audio[:, 0]
        cls.audio = audio.astype(np.float32)
        cls.sr = sr
        cls.input_rms = float(np.sqrt(np.mean(cls.audio**2)))

    def test_offline_output_length_and_range(self):
        """Offline enhancement produces output of correct length within [-1, 1]."""
        enhanced = self.model.enhance_array(self.audio)
        self.assertEqual(enhanced.shape[0], self.audio.shape[0])
        self.assertLessEqual(float(np.max(np.abs(enhanced))), 1.0)

    def test_offline_reduces_noise(self):
        """Offline enhancement reduces RMS energy (denoising actually occurred)."""
        enhanced = self.model.enhance_array(self.audio)
        output_rms = float(np.sqrt(np.mean(enhanced**2)))
        self.assertLess(
            output_rms,
            self.input_rms,
            "Enhanced audio RMS should be lower than noisy input RMS",
        )

    def test_offline_output_is_not_silence(self):
        """Offline enhancement produces non-trivial output (not all zeros)."""
        enhanced = self.model.enhance_array(self.audio)
        output_rms = float(np.sqrt(np.mean(enhanced**2)))
        self.assertGreater(output_rms, 0.001, "Enhanced audio should not be silence")

    def test_streaming_output_length_and_range(self):
        """Streaming enhancement produces output of correct length within [-1, 1]."""
        enhanced = self.model.enhance_array_streaming(self.audio)
        self.assertEqual(enhanced.shape[0], self.audio.shape[0])
        self.assertLessEqual(float(np.max(np.abs(enhanced))), 1.0)

    def test_streaming_reduces_noise(self):
        """Streaming enhancement reduces RMS energy."""
        enhanced = self.model.enhance_array_streaming(self.audio)
        output_rms = float(np.sqrt(np.mean(enhanced**2)))
        self.assertLess(
            output_rms,
            self.input_rms,
            "Streaming enhanced audio RMS should be lower than noisy input RMS",
        )

    def test_offline_streaming_correlation(self):
        """Offline and streaming outputs are highly correlated."""
        offline = self.model.enhance_array(self.audio)
        streaming = self.model.enhance_array_streaming(self.audio)
        min_len = min(len(offline), len(streaming))
        corr = float(np.corrcoef(offline[:min_len], streaming[:min_len])[0, 1])
        self.assertGreater(
            corr,
            0.85,
            f"Offline/streaming correlation {corr:.4f} should be > 0.85",
        )

    def test_enhance_file_roundtrip(self):
        """enhance_file writes a valid audio file with correct sample rate."""
        from mlx_audio.audio_io import read as audio_read

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "enhanced.wav"
            self.model.enhance_file(str(self.SAMPLE_AUDIO), str(out_path))
            self.assertTrue(out_path.exists())
            data, sr = audio_read(str(out_path))
            self.assertEqual(sr, self.sr)
            self.assertEqual(len(data), len(self.audio))

    def test_target_parity(self):
        """MLX output closely matches pre-generated PyTorch reference output.

        Compares against a target WAV file generated by the official PyTorch
        DeepFilterNet implementation. This test does NOT require PyTorch to run.
        """
        from mlx_audio.audio_io import read as audio_read

        target_path = (
            self.REPO_ROOT / "examples" / "denoise" / "noisey_audio_10s_target.wav"
        )
        if not target_path.exists():
            self.skipTest(f"Target audio not found: {target_path}")

        target, target_sr = audio_read(str(target_path))
        if target.ndim > 1:
            target = target[:, 0]
        target = target.astype(np.float32)

        mlx_out = self.model.enhance_array(self.audio)

        min_len = min(len(target), len(mlx_out))
        target = target[:min_len]
        mlx_out = mlx_out[:min_len]

        # Correlation (actual ~0.9997, threshold 0.999)
        corr = float(np.corrcoef(target, mlx_out)[0, 1])
        self.assertGreater(
            corr,
            0.999,
            f"Correlation {corr:.6f} should be > 0.999",
        )

        # Signal-to-Error Ratio in dB (actual ~31.6 dB, threshold 25 dB)
        signal_power = float(np.mean(target**2))
        error_power = float(np.mean((target - mlx_out) ** 2))
        ser_db = 10 * np.log10(signal_power / (error_power + 1e-10))
        self.assertGreater(
            ser_db,
            25.0,
            f"SER {ser_db:.1f} dB should be > 25 dB",
        )

        # Mean Absolute Error (actual ~0.001, threshold 0.002)
        mae = float(np.mean(np.abs(target - mlx_out)))
        self.assertLess(
            mae,
            3e-3,
            f"MAE {mae:.6f} should be < 0.003",
        )

        # RMS difference (actual ~0.5%, threshold 1%)
        rms_target = float(np.sqrt(np.mean(target**2)))
        rms_mlx = float(np.sqrt(np.mean(mlx_out**2)))
        rms_diff_pct = abs(rms_target - rms_mlx) / (rms_target + 1e-10) * 100
        self.assertLess(
            rms_diff_pct,
            1.0,
            f"RMS difference {rms_diff_pct:.3f}% should be < 1%",
        )

    def test_pytorch_parity(self):
        """MLX output correlates highly with PyTorch DeepFilterNet output.

        Requires PyTorch and the df package. Skipped when not installed.
        """
        try:
            import torch
            from df.enhance import df_features, init_df
            from df.utils import as_complex
        except ImportError:
            self.skipTest("PyTorch DeepFilterNet (df) not installed")

        from df.model import ModelParams

        # Run PyTorch inference
        model_pt, df_state, _ = init_df(log_level="ERROR")
        model_pt.eval()

        audio_t = torch.from_numpy(self.audio).unsqueeze(0).float()
        n_fft = df_state.fft_size()
        hop = df_state.hop_size()
        audio_padded = torch.nn.functional.pad(audio_t, (0, n_fft))

        p = ModelParams()
        spec, erb_feat, spec_feat = df_features(
            audio_padded, df_state, p.nb_df, device="cpu"
        )

        with torch.no_grad():
            if hasattr(model_pt, "reset_h0"):
                model_pt.reset_h0(batch_size=1, device="cpu")
            enhanced_spec, _, _, _ = model_pt(spec, erb_feat, spec_feat)

        enhanced_complex = as_complex(enhanced_spec.squeeze(1))
        pt_audio = df_state.synthesis(enhanced_complex.numpy())
        pt_audio = np.asarray(pt_audio, dtype=np.float32)
        d = n_fft - hop
        pt_audio = pt_audio[0, d : len(self.audio) + d]

        # Run MLX inference
        mlx_audio = self.model.enhance_array(self.audio)

        min_len = min(len(pt_audio), len(mlx_audio))
        corr = float(np.corrcoef(pt_audio[:min_len], mlx_audio[:min_len])[0, 1])
        self.assertGreater(
            corr,
            0.90,
            f"PyTorch/MLX correlation {corr:.4f} should be > 0.90",
        )


if __name__ == "__main__":
    unittest.main()
