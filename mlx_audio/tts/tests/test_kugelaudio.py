"""Tests for KugelAudio model — config parsing, weight sanitization,
SDE scheduler, and token constraint logic."""

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_audio.tts.models.kugelaudio.config import DiffusionHeadConfig, ModelConfig
from mlx_audio.tts.models.kugelaudio.kugelaudio import (
    SPEECH_DIFFUSION_ID,
    VALID_SPEECH_TOKENS,
    Model,
)
from mlx_audio.tts.models.kugelaudio.scheduler import SDEDPMSolverMultistepScheduler
from mlx_audio.tts.models.vibevoice.scheduler import (
    DPMSolverMultistepScheduler as BaseDPMSolver,
)


class TestConfigParsing(unittest.TestCase):
    """Test ModelConfig.from_dict with real and edge-case configs."""

    def test_from_dict_basic(self):
        """Basic config creation with nested sub-configs."""
        cfg = ModelConfig.from_dict(
            {
                "model_type": "kugelaudio",
                "acoustic_vae_dim": 64,
                "decoder_config": {"hidden_size": 3584, "num_hidden_layers": 28},
                "diffusion_head_config": {"hidden_size": 3584},
                "acoustic_tokenizer_config": {"vae_dim": 64},
            }
        )
        self.assertEqual(cfg.model_type, "kugelaudio")
        self.assertEqual(cfg.acoustic_vae_dim, 64)
        self.assertEqual(cfg.decoder_config.hidden_size, 3584)
        self.assertEqual(cfg.decoder_config.num_hidden_layers, 28)

    def test_from_dict_handles_typo(self):
        """HF config has 'acostic_vae_dim' typo."""
        cfg = ModelConfig.from_dict(
            {
                "acostic_vae_dim": 64,
                "decoder_config": {},
                "diffusion_head_config": {},
                "acoustic_tokenizer_config": {},
            }
        )
        self.assertEqual(cfg.acoustic_vae_dim, 64)

    def test_from_dict_drops_semantic_config(self):
        """semantic_tokenizer_config should be silently dropped."""
        cfg = ModelConfig.from_dict(
            {
                "acoustic_vae_dim": 64,
                "semantic_tokenizer_config": {"vae_dim": 128},
                "decoder_config": {},
                "diffusion_head_config": {},
                "acoustic_tokenizer_config": {},
            }
        )
        self.assertFalse(hasattr(cfg, "semantic_tokenizer_config"))

    def test_from_dict_ignores_unknown_fields(self):
        """Unknown top-level fields should not raise."""
        cfg = ModelConfig.from_dict(
            {
                "acoustic_vae_dim": 64,
                "unknown_field_xyz": True,
                "decoder_config": {},
                "diffusion_head_config": {},
                "acoustic_tokenizer_config": {},
            }
        )
        self.assertEqual(cfg.acoustic_vae_dim, 64)

    def test_diffusion_config_defaults(self):
        """Check default values for diffusion config."""
        cfg = DiffusionHeadConfig()
        self.assertEqual(cfg.ddpm_num_inference_steps, 10)
        self.assertEqual(cfg.ddpm_algorithm_type, "sde-dpmsolver++")
        self.assertEqual(cfg.prediction_type, "v_prediction")


_TINY_CONFIG = {
    "acoustic_vae_dim": 4,
    "tie_word_embeddings": False,
    "decoder_config": {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "max_position_embeddings": 64,
    },
    "diffusion_head_config": {
        "hidden_size": 8,
        "latent_size": 4,
        "head_layers": 1,
        "head_ffn_ratio": 2.0,
    },
    "acoustic_tokenizer_config": {
        "vae_dim": 4,
        "encoder_n_filters": 4,
        "decoder_n_filters": 4,
        "encoder_ratios": [2],
        "encoder_depths": "1",
    },
}


def _make_tiny_model():
    """Create a minimal KugelAudio model for sanitize testing."""
    return Model(ModelConfig.from_dict(dict(_TINY_CONFIG)))


class TestWeightSanitize(unittest.TestCase):
    """Test weight key remapping and transpositions in sanitize()."""

    def test_strips_model_prefix(self):
        """Keys with 'model.' prefix should be mapped to unprefixed."""
        model = _make_tiny_model()
        params = dict(tree_flatten(model.parameters()))
        real_key = next(k for k in params if k.startswith("language_model."))
        fake_weights = {f"model.{real_key}": params[real_key]}
        result = model.sanitize(fake_weights)
        self.assertIn(real_key, result)

    def test_skips_encoder_and_semantic_keys(self):
        """Encoder/semantic weights should be silently dropped."""
        model = _make_tiny_model()
        weights = {
            "model.semantic_tokenizer.foo": mx.zeros((2,)),
            "model.semantic_connector.bar": mx.zeros((2,)),
            "model.acoustic_tokenizer.encoder.baz": mx.zeros((2,)),
        }
        result = model.sanitize(weights)
        self.assertEqual(len(result), 0)

    def test_fixes_sequential_indexing(self):
        """PyTorch '.mlp.0.' should become '.mlp.layers.0.' for MLX."""
        model = _make_tiny_model()
        params = dict(tree_flatten(model.parameters()))
        mlp_key = next((k for k in params if "t_embedder.mlp.layers." in k), None)
        if mlp_key:
            pytorch_key = "model." + mlp_key.replace(".mlp.layers.", ".mlp.")
            fake_weights = {pytorch_key: params[mlp_key]}
            result = model.sanitize(fake_weights)
            self.assertIn(mlp_key, result)

    def test_preserves_quantization_metadata(self):
        """Quantization keys (.scales, .biases) must not be dropped."""
        model = _make_tiny_model()
        weights = {
            "language_model.layers.0.self_attn.q_proj.scales": mx.zeros((4,)),
            "language_model.layers.0.self_attn.q_proj.biases": mx.zeros((4,)),
        }
        result = model.sanitize(weights)
        self.assertEqual(len(result), 2)

    def test_transposes_linear_weights(self):
        """Linear weights with mismatched shape should be transposed."""
        model = _make_tiny_model()
        params = dict(tree_flatten(model.parameters()))
        key = "lm_head.weight"
        if key in params:
            target_shape = params[key].shape
            fake = mx.zeros(tuple(reversed(target_shape)))
            result = model.sanitize({key: fake})
            self.assertEqual(result[key].shape, target_shape)


class TestSDEScheduler(unittest.TestCase):
    """Test that SDE scheduler diverges from deterministic base."""

    def test_inherits_from_base(self):
        """SDEDPMSolverMultistepScheduler should subclass the base."""
        self.assertTrue(issubclass(SDEDPMSolverMultistepScheduler, BaseDPMSolver))

    def test_sde_adds_noise(self):
        """SDE variant should produce different results from deterministic."""
        mx.random.seed(42)
        sde = SDEDPMSolverMultistepScheduler(
            num_train_timesteps=100, prediction_type="v_prediction"
        )
        det = BaseDPMSolver(num_train_timesteps=100, prediction_type="v_prediction")

        sde.set_timesteps(5)
        det.set_timesteps(5)

        sample = mx.ones((1, 4)) * 0.5
        model_output = mx.ones((1, 4)) * 0.1

        sde_result = sde.step(model_output, sde.timesteps[0], sample)
        det_result = det.step(model_output, det.timesteps[0], sample)

        mx.eval(sde_result.prev_sample, det_result.prev_sample)

        diff = mx.abs(sde_result.prev_sample - det_result.prev_sample).sum().item()
        self.assertGreater(diff, 0.0)

    def test_sde_step_output_shape(self):
        """Output shape should match input shape."""
        sched = SDEDPMSolverMultistepScheduler(num_train_timesteps=100)
        sched.set_timesteps(5)

        sample = mx.zeros((2, 8))
        output = mx.zeros((2, 8))

        result = sched.step(output, sched.timesteps[0], sample)
        mx.eval(result.prev_sample)

        self.assertEqual(result.prev_sample.shape, (2, 8))
        self.assertIsNotNone(result.x0_pred)

    def test_reset_clears_state(self):
        """reset() should zero out step index and lower order count."""
        sched = SDEDPMSolverMultistepScheduler(num_train_timesteps=100)
        sched.set_timesteps(5)
        sched._step_index = 3  # pylint: disable=protected-access
        sched.lower_order_nums = 2
        sched.reset()
        self.assertIsNone(sched._step_index)  # pylint: disable=protected-access
        self.assertEqual(sched.lower_order_nums, 0)


class TestTokenConstraint(unittest.TestCase):
    """Test the token constraint masking logic."""

    def test_constraint_mask_allows_valid_tokens(self):
        """Only VALID_SPEECH_TOKENS should survive the mask."""
        vocab_size = 152064
        logits = mx.zeros((1, vocab_size))

        constraint_mask = mx.full(logits.shape, float("-inf"))
        valid_indices = mx.array(VALID_SPEECH_TOKENS)
        constraint_mask[:, valid_indices] = 0.0
        masked = logits + constraint_mask
        mx.eval(masked)

        for tid in VALID_SPEECH_TOKENS:
            self.assertEqual(masked[0, tid].item(), 0.0)

        self.assertEqual(masked[0, 0].item(), float("-inf"))
        self.assertEqual(masked[0, 1000].item(), float("-inf"))

    def test_argmax_selects_highest_valid_token(self):
        """argmax on masked logits should pick the boosted valid token."""
        vocab_size = 152064
        logits = mx.zeros((1, vocab_size))

        logits = logits.at[0, SPEECH_DIFFUSION_ID].add(mx.array(10.0))

        constraint_mask = mx.full(logits.shape, float("-inf"))
        valid_indices = mx.array(VALID_SPEECH_TOKENS)
        constraint_mask[:, valid_indices] = 0.0
        masked = logits + constraint_mask

        selected = mx.argmax(masked, axis=-1).item()
        self.assertEqual(selected, SPEECH_DIFFUSION_ID)


if __name__ == "__main__":
    unittest.main()
