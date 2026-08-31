import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.granite_speech5.config import (
    EncoderConfig,
    ModelConfig,
)
from mlx_audio.stt.models.granite_speech5.granite_speech5 import (
    Model,
    compute_deltas,
    compute_features,
    ctc_collapse,
)
from mlx_audio.stt.utils import MODEL_REMAPPING
from mlx_audio.utils import get_model_class, get_model_name_parts


def tiny_config(**encoder_overrides):
    values = {
        "vocab_size": 16,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_mel_bins": 4,
        "head_dim": 8,
        "max_position_embeddings": 16,
        "context_size": 8,
        "conv_kernel_size": 3,
        "conv_expansion_factor": 2,
        "subsample_layers": [0, 1],
    }
    values.update(encoder_overrides)
    encoder = EncoderConfig(**values)
    return ModelConfig(vocab_size=encoder.vocab_size, encoder_config=encoder)


def test_config_parses_native_hf_schema():
    config = ModelConfig.from_dict(
        {
            "model_type": "granite_speech5_ctc",
            "vocab_size": 32,
            "encoder_config": {
                "vocab_size": 32,
                "hidden_size": 24,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 3,
                "num_key_value_heads": 3,
                "head_dim": 8,
                "num_mel_bins": 6,
                "subsample_layers": [0],
            },
        }
    )

    assert config.model_type == "granite_speech5_ctc"
    assert config.encoder_config.hidden_size == 24
    assert config.encoder_config.subsample_layers == [0]


def test_exact_model_type_mapping_beats_older_granite_name_match():
    module, resolved_type = get_model_class(
        model_type="granite_speech5_ctc",
        model_name=get_model_name_parts("granite-speech-5.0-470m-turboctc"),
        category="stt",
        model_remapping=MODEL_REMAPPING,
    )

    assert resolved_type == "granite_speech5"
    assert module.Model is Model


def test_forward_applies_two_subsampling_blocks():
    model = Model(tiny_config())
    features = mx.zeros((1, 19, 16))
    logits = model(features)
    mx.eval(logits)

    assert logits.shape == (1, 4, 16)


def test_sanitize_transposes_only_pytorch_depthwise_kernels():
    pytorch_kernel = mx.zeros((8, 1, 3))
    mlx_kernel = mx.zeros((8, 3, 1))
    weights = {
        "encoder.layers.0.conv.depthwise_conv.weight": pytorch_kernel,
        "encoder.layers.1.conv.depthwise_conv.weight": mlx_kernel,
        "encoder.layers.0.conv.norm.num_batches_tracked": mx.array(0),
        "encoder.input_linear.weight": mx.zeros((4, 4)),
    }

    sanitized = Model.sanitize(weights)

    assert sanitized["encoder.layers.0.conv.depthwise_conv.weight"].shape == (
        8,
        3,
        1,
    )
    assert sanitized["encoder.layers.1.conv.depthwise_conv.weight"].shape == (
        8,
        3,
        1,
    )
    assert not any("num_batches_tracked" in key for key in sanitized)


def test_compute_deltas_uses_replicated_edges():
    features = mx.array([[1.0], [3.0], [7.0]])
    deltas = compute_deltas(features)
    mx.eval(deltas)

    np.testing.assert_allclose(
        np.asarray(deltas).reshape(-1),
        [1.0, 3.0, 2.0],
    )


def test_feature_extractor_shape_includes_partial_frame_pair():
    waveform = mx.zeros((321,), dtype=mx.float32)
    features = compute_features(waveform, num_mel_bins=4)
    mx.eval(features)

    assert features.shape == (1, 16)


def test_ctc_collapse_deduplicates_before_removing_blanks():
    tokens = ctc_collapse(mx.array([0, 1, 1, 0, 1, 2, 2, 0]))
    mx.eval(tokens)

    assert tokens.tolist() == [1, 1, 2]
