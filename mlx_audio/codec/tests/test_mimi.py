import unittest
from unittest.mock import patch

import mlx.core as mx

from ..models.mimi.mimi import Mimi, mimi_202407
from ..models.mimi.modules.conv import ConvTranspose1d
from ..models.mimi.modules.quantization import EuclideanCodebook


class TestMimi(unittest.TestCase):
    def test_mimi_model(self):
        """Test Mimi model encoding and decoding."""
        model = Mimi(mimi_202407(32))

        audio = mx.zeros((1, 1, 120_000))
        codes = model.encode(audio)
        self.assertEqual(codes.shape, (1, 32, 63))

        audio_out = model.decode(codes)
        self.assertEqual(audio_out.shape, (1, 1, 120_960))

    def test_convtranspose_materializes_expanded_weight(self):
        with patch("mlx_audio.codec.models.mimi.modules.conv.mx.eval") as eval_mock:
            layer = ConvTranspose1d(4, 4, 3, groups=4)

        args = eval_mock.call_args.args
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], layer._expanded_weight)

    def test_codebook_materializes_derived_lookup_arrays(self):
        with patch(
            "mlx_audio.codec.models.mimi.modules.quantization.mx.eval"
        ) as eval_mock:
            codebook = EuclideanCodebook(dim=4, codebook_size=8)

        args = eval_mock.call_args.args
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], codebook._embedding)
        self.assertIs(args[1], codebook._c2)

    def test_from_pretrained_materializes_loaded_parameters(self):
        with (
            patch(
                "mlx_audio.codec.models.mimi.mimi.hf_hub_download",
                return_value="weights.safetensors",
            ),
            patch.object(Mimi, "load_pytorch_weights", return_value=None),
            patch("mlx_audio.codec.models.mimi.mimi.mx.eval") as eval_mock,
        ):
            Mimi.from_pretrained("test/repo")

        self.assertTrue(
            any(
                call.args and isinstance(call.args[0], dict)
                for call in eval_mock.mock_calls
            )
        )


if __name__ == "__main__":
    unittest.main()
