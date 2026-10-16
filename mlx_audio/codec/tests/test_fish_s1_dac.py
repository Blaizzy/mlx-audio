import threading
import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from ..models.fish_s1_dac.fish_s1_dac import (
    DAC,
    DownsampleResidualVectorQuantize,
    Identity,
    ModelArgs,
    WindowLimitedTransformer,
)


def _tiny_streaming_dac(block_size: int = 64, semantic_codebook_size: int = 32) -> DAC:
    mx.random.seed(0)
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=16,
        n_codebooks=2,
        codebook_dim=4,
        codebook_size=16,
        semantic_codebook_size=semantic_codebook_size,
        downsample_factor=(2,),
        pre_module=Identity(),
        post_module=WindowLimitedTransformer(
            causal=True,
            window_size=4,
            input_dim=16,
            config=ModelArgs(
                block_size=block_size,
                n_layer=2,
                n_head=2,
                dim=16,
                intermediate_size=32,
                head_dim=8,
            ),
        ),
    )
    return DAC(
        encoder_dim=4,
        encoder_rates=[2, 2],
        latent_dim=16,
        decoder_dim=16,
        decoder_rates=[2, 2],
        quantizer=quantizer,
        sample_rate=44100,
        causal=True,
        encoder_transformer_layers=[0, 0],
        decoder_transformer_layers=[0, 0],
        transformer_general_config=lambda **kw: None,
    )


def _random_codes(frames: int) -> mx.array:
    mx.random.seed(1)
    semantic = mx.random.randint(0, 32, (1, 1, frames))
    residual = mx.random.randint(0, 16, (1, 2, frames))
    return mx.concatenate([semantic, residual], axis=1)


def _stream(model: DAC, codes: mx.array, chunk: int) -> mx.array:
    model.reset_streaming_state()
    chunks = [
        model.streaming_step(codes[:, :, start : start + chunk])
        for start in range(0, codes.shape[-1], chunk)
    ]
    return mx.concatenate(chunks, axis=-1)


class TestFishS1DAC(unittest.TestCase):
    def test_tiny_encode_decode(self):
        quantizer = DownsampleResidualVectorQuantize(
            input_dim=16,
            n_codebooks=2,
            codebook_dim=4,
            codebook_size=16,
            semantic_codebook_size=32,
            downsample_factor=(2,),
            pre_module=Identity(),
            post_module=Identity(),
        )

        model = DAC(
            encoder_dim=4,
            encoder_rates=[2, 2],
            latent_dim=16,
            decoder_dim=16,
            decoder_rates=[2, 2],
            quantizer=quantizer,
            sample_rate=44100,
            causal=True,
            encoder_transformer_layers=[0, 0],
            decoder_transformer_layers=[0, 0],
            transformer_general_config=lambda **kw: None,
        )

        audio = mx.zeros((1, 1, 128), dtype=mx.float32)
        indices, feature_lengths = model.encode(audio)
        self.assertEqual(indices.shape[0], 1)
        self.assertEqual(indices.shape[1], 3)  # semantic + residual quantizers

        decoded, decoded_lengths = model.decode(indices, feature_lengths)
        self.assertEqual(tuple(decoded.shape), (1, 1, 128))
        self.assertEqual(int(decoded_lengths[0]), 128)

        z_q = model.encode_zq(audio)
        self.assertEqual(tuple(z_q.shape), (1, 16, 16))
        recon = model.decode_zq(z_q)
        self.assertEqual(tuple(recon.shape), (1, 1, 128))

    def test_streaming_step_matches_full_decode(self):
        # CPU float32 keeps results independent of shape-dependent GPU kernels.
        with mx.stream(mx.cpu):
            model = _tiny_streaming_dac()
            codes = _random_codes(23)
            full, _ = model.decode(codes, mx.array([23]))

            for chunk in (1, 2, 5, 23):
                streamed = _stream(model, codes, chunk)
                self.assertEqual(streamed.shape, full.shape)
                self.assertLess(mx.abs(streamed - full).max().item(), 1e-5, chunk)

    def test_streaming_state_resets_between_streams(self):
        with mx.stream(mx.cpu):
            model = _tiny_streaming_dac()
            codes = _random_codes(9)
            first = _stream(model, codes, 3)
            second = _stream(model, codes, 3)
            self.assertLess(mx.abs(first - second).max().item(), 1e-6)

    def test_streaming_continues_past_rope_table(self):
        # A stream longer than block_size rebases RoPE positions onto the
        # cached window; with float32 tables that matches a larger table.
        with mx.stream(mx.cpu):
            codes = _random_codes(40)
            reference, _ = _tiny_streaming_dac(block_size=64).decode(
                codes, mx.array([40])
            )
            streamed = _stream(_tiny_streaming_dac(block_size=12), codes, 3)
            self.assertLess(mx.abs(streamed - reference).max().item(), 1e-4)

    def test_streaming_rejects_chunk_larger_than_rope_table(self):
        with mx.stream(mx.cpu):
            model = _tiny_streaming_dac(block_size=12)
            model.reset_streaming_state()
            with self.assertRaises(ValueError):
                model.streaming_step(_random_codes(13))

    def test_folded_kernels_match_and_refold_after_new_weights(self):
        with mx.stream(mx.cpu):
            codes = _random_codes(9)
            audio = mx.random.normal((1, 1, 128))
            model = _tiny_streaming_dac()
            unfolded, _ = model.decode(codes, mx.array([9]))
            unfolded_zq = model.encode_zq(audio)
            model.fold_weight_norm()
            folded, _ = model.decode(codes, mx.array([9]))
            self.assertTrue(mx.array_equal(folded, unfolded).item())
            self.assertTrue(mx.array_equal(model.encode_zq(audio), unfolded_zq).item())

            scaled = [
                (k, v * 1.5 if k.endswith("weight_g") else v)
                for k, v in tree_flatten(model.parameters())
            ]
            model.load_weights(scaled)
            model.fold_weight_norm()
            fresh = _tiny_streaming_dac()
            fresh.load_weights(scaled)
            refolded, _ = model.decode(codes, mx.array([9]))
            expected, _ = fresh.decode(codes, mx.array([9]))
            self.assertGreater(mx.abs(refolded - folded).max().item(), 1e-3)
            self.assertTrue(mx.array_equal(refolded, expected).item())

    def test_folded_codec_decodes_on_another_thread(self):
        # Folded kernels are shared across threads, which each decode on their
        # own stream; a lazy kernel would be bound to the folding thread.
        model = _tiny_streaming_dac()
        codes = _random_codes(9)
        mx.eval(model.parameters(), codes)
        model.fold_weight_norm()
        errors = []

        def decode():
            try:
                with mx.stream(mx.new_stream(mx.default_device())):
                    mx.eval(model.decode(codes, mx.array([9]))[0])
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=decode)
        worker.start()
        worker.join()
        self.assertEqual(errors, [])

    def test_folded_codec_compiles_with_captured_state(self):
        model = _tiny_streaming_dac()
        codes = _random_codes(9)
        mx.eval(model.parameters(), codes)
        model.fold_weight_norm()
        lengths = mx.array([9])
        compiled = mx.compile(
            lambda c: model.decode(c, lengths)[0], inputs=model, outputs=model
        )
        self.assertTrue(
            mx.allclose(compiled(codes), model.decode(codes, lengths)[0], atol=1e-5)
        )

    def test_checkpoint_causal_masks_are_dropped(self):
        # PyTorch checkpoints carry square causal-mask buffers. A strict load
        # fails if one survives sanitize or the model still expects one.
        model = _tiny_streaming_dac()
        weights = dict(tree_flatten(model.parameters()))
        weights["quantizer.post_module.causal_mask"] = mx.ones((64, 64), dtype=mx.bool_)
        model.load_weights(list(model.sanitize(weights).items()), strict=True)


if __name__ == "__main__":
    unittest.main()
