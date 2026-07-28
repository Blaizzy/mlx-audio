import mlx.core as mx
import pytest

from mlx_audio.tts.models.kitten_tts.istftnet import SineGen as KittenSineGen
from mlx_audio.tts.models.kokoro.istftnet import SineGen as KokoroSineGen


@pytest.mark.parametrize("sine_gen_cls", [KokoroSineGen, KittenSineGen])
def test_sinegen_matches_f0_length_after_interpolation(sine_gen_cls):
    sine_gen = sine_gen_cls(24000, upsample_scale=300, harmonic_num=8)
    f0 = mx.ones((1, 2, 1)) * 120

    sine_waves, uv, noise = sine_gen(f0)

    assert sine_waves.shape == (1, 2, 9)
    assert uv.shape == (1, 2, 1)
    assert noise.shape == (1, 2, 9)
