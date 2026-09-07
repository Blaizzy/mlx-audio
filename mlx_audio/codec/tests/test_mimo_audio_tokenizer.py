import json
from dataclasses import asdict, replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.codec import MiMoAudioTokenizer
from mlx_audio.codec.models.mimo_audio_tokenizer import ModelConfig
from mlx_audio.codec.models.mimo_audio_tokenizer.audio import (
    load_waveform,
    log_mel_spectrogram,
    same_istft,
)
from mlx_audio.codec.models.mimo_audio_tokenizer.convert import convert


def tiny_config(**kwargs):
    return replace(
        ModelConfig(
            d_model=16,
            n_mels=4,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            vocoder_dim=8,
            vocoder_intermediate_dim=16,
            vocoder_num_layers=1,
            vocoder_attention_heads=2,
            num_quantizers=2,
            codebook_size=[8, 4],
            nfft=32,
            window_size=32,
            hop_length=8,
            encoder_skip_layer_id=1,
        ),
        **kwargs,
    )


def test_config_defaults_and_validation():
    config = ModelConfig()
    assert config.sampling_rate == 24000
    assert config.num_quantizers == 20
    assert config.codebook_size == [1024, 1024] + [128] * 18
    assert config.downsample_rate == 960
    assert ModelConfig.from_dict(asdict(tiny_config())) == tiny_config()
    with pytest.raises(ValueError, match="codebook_size"):
        tiny_config(num_quantizers=3)


def test_encode_decode_with_synthetic_weights():
    model = MiMoAudioTokenizer(tiny_config())
    wave = mx.sin(mx.arange(200, dtype=mx.float32) / 7) * 0.1
    mel = log_mel_spectrogram(wave, model.config)
    features = model.encoder(mel[None])[0]
    codes = model.encode(wave, sample_rate=24000)
    audio = model.decode(codes)
    assert mel.shape == (26, model.config.n_mels)
    assert features.shape == (7, model.config.d_model)
    assert codes.shape == (model.num_quantizers, 7)
    assert mx.issubdtype(codes.dtype, mx.integer)
    assert audio.shape == (7 * model.config.downsample_rate,)
    assert bool(mx.all(mx.isfinite(audio)))


def test_all_twenty_codebooks_and_eight_codebook_prefix():
    model = MiMoAudioTokenizer(
        tiny_config(num_quantizers=20, codebook_size=[8, 8] + [4] * 18)
    )
    features = mx.random.normal((11, model.config.d_model))
    full = model.encoder.quantizer.encode(features)
    prefix = model.encoder.quantizer.encode(features, 8)
    assert full.shape == (20, 11)
    np.testing.assert_array_equal(np.asarray(full[:8]), np.asarray(prefix))
    assert model.decode(full).shape == (11 * model.config.downsample_rate,)
    assert model.decode(prefix).shape == (11 * model.config.downsample_rate,)


def test_segment_padding_and_lengths():
    model = MiMoAudioTokenizer(tiny_config())
    mel = mx.random.normal((35, model.config.n_mels))
    # 16 -> 4 codes, 16 -> 4 codes, 3 -> 1 code.
    codes = model.encode_mels(mel, segment_size=16)
    assert codes.shape == (2, 9)
    assert model.decode(codes, segment_size=4).shape == (
        9 * model.config.downsample_rate,
    )
    assert model.decode(codes, length=27).shape == (27,)
    assert model.decode(mx.zeros((2, 0), dtype=mx.int32)).shape == (0,)
    short = model.encode(mx.array([0.0]), sample_rate=24000)
    assert short.shape[1] > 0


@pytest.mark.parametrize(
    "codes",
    [
        mx.array([[8]]),
        mx.array([[-1]]),
        mx.array([[0.5]]),
        mx.zeros((3, 4), dtype=mx.int32),
    ],
)
def test_invalid_codes_rejected(codes):
    with pytest.raises(ValueError):
        MiMoAudioTokenizer(tiny_config()).decode(codes)


def test_invalid_audio_rejected():
    for wave in ([], [float("nan")], np.zeros((1, 1, 2))):
        with pytest.raises(ValueError):
            load_waveform(wave, 24000, 24000)
    with pytest.raises(ValueError, match="sample_rate"):
        load_waveform([0.0, 1.0], None, 24000)
    assert load_waveform(np.ones((120, 2)), 12000, 24000).shape == (240,)


def test_same_istft_length_and_silence():
    wave = same_istft(mx.zeros((7, 17), dtype=mx.complex64), 32, 8)
    assert wave.shape == (56,)
    np.testing.assert_array_equal(np.asarray(wave), np.zeros(56))


def test_sanitize_source_weight_layouts():
    model = MiMoAudioTokenizer(tiny_config())
    weights = model.sanitize(
        {
            "encoder.conv1.weight": mx.zeros((16, 4, 3)),
            "encoder.down_sample_layer.0.weight": mx.zeros((16, 16, 2)),
            "decoder.dconv2.conv.weight": mx.zeros((16, 4, 3)),
            "encoder.quantizer.vq.layers.0._codebook.embed": mx.zeros((8, 16)),
            "encoder.quantizer.vq.layers.0._codebook.embed_avg": mx.zeros((8, 16)),
        }
    )
    assert weights["encoder.conv1.weight"].shape == (16, 3, 4)
    assert weights["encoder.down_sample_layer.layers.0.weight"].shape == (16, 2, 16)
    assert weights["decoder.dconv2.conv.weight"].shape == (4, 3, 16)
    assert weights["encoder.quantizer.codebooks.0.weight"].shape == (8, 16)
    assert len(weights) == 4


def test_strict_converted_loading_with_synthetic_weights(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    model = MiMoAudioTokenizer(tiny_config(weights_format="mlx"))
    (source / "config.json").write_text(json.dumps(asdict(model.config)))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    output = convert(str(source), tmp_path / "converted")
    original = MiMoAudioTokenizer.from_pretrained(source, dtype=None)
    restored = MiMoAudioTokenizer.from_pretrained(output, dtype=None)
    assert restored.config.weights_format == "mlx"
    left, right = dict(tree_flatten(original.parameters())), dict(
        tree_flatten(restored.parameters())
    )
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(np.asarray(left[name]), np.asarray(right[name]))
    missing = dict(right)
    missing.pop("decoder.layer_norm.weight")
    mx.save_safetensors(str(output / "model.safetensors"), missing)
    for name, weight in mx.load(str(output / "model.safetensors")).items():
        np.testing.assert_array_equal(np.asarray(weight), np.asarray(right[name]))
    with pytest.raises(ValueError, match=r"Missing.*parameters"):
        MiMoAudioTokenizer.from_pretrained(output)
