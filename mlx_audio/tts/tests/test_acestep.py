import unittest
import mlx.core as mx
from mlx_audio.tts.models.acestep.config import AceStepConfig, AceStepDiTConfig, AceStepVAEConfig
from mlx_audio.tts.models.acestep.acestep import AceStepTTAModel

class TestAceStep(unittest.TestCase):
    def setUp(self):
        dit_config = AceStepDiTConfig(
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            in_channels=16,
            audio_acoustic_hidden_dim=16,
            layer_types=["self", "cross"],
            text_hidden_dim=64,
            num_lyric_encoder_hidden_layers=1,
        )
        vae_config = AceStepVAEConfig(
            encoder_hidden_size=16,
            downsampling_ratios=[2, 2],
            channel_multiples=[1, 2],
            decoder_channels=16,
            decoder_input_channels=16,
        )
        self.config = AceStepConfig(dit_config=dit_config, vae_config=vae_config)
        self.model = AceStepTTAModel(self.config)

    def test_condition_encoder(self):
        text_hs = mx.random.normal((1, 10, 64))
        text_mask = mx.ones((1, 10))
        # lyric_hs coming out of LLM is ALREADY embedded, so it's [B, L, D], NOT token IDs!
        lyric_hs = mx.random.normal((1, 15, 64))
        lyric_mask = mx.ones((1, 15))
        
        enc_out, enc_mask = self.model.encoder(
            text_hidden_states=text_hs,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hs,
            lyric_attention_mask=lyric_mask,
        )
        
        self.assertEqual(enc_out.shape[0], 1)
        self.assertEqual(enc_out.shape[1], 25)
        self.assertEqual(enc_out.shape[2], 64)
        self.assertEqual(enc_mask.shape[1], 25)

if __name__ == "__main__":
    unittest.main()
