import mlx.core as mx

from mlx_audio.lm.models import gemma3_text
from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.sts.models.nemotron_voicechat import Model, ModelConfig
from mlx_audio.sts.models.nemotron_voicechat.convert import _quantize
from mlx_audio.sts.utils import infer_model_type_from_config


def mini_config():
    llm = {
        "model_type": "nemotron_h",
        "vocab_size": 64,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 3,
        "max_position_embeddings": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "attention_bias": False,
        "mamba_num_heads": 2,
        "mamba_head_dim": 4,
        "mamba_proj_bias": False,
        "ssm_state_size": 8,
        "conv_kernel": 4,
        "n_groups": 1,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "hybrid_override_pattern": ["M", "*", "-"],
    }
    return {
        "data": {
            "source_sample_rate": 16_000,
            "target_sample_rate": 22_050,
            "frame_length": 0.08,
        },
        "mlx_audio": {"llm_config": llm, "char_vocab_size": 4},
        "model": {
            "inference_speaker_name": "Aria",
            "stt": {
                "model": {
                    "pretrained_llm": "local",
                    "perception": {
                        "output_dim": 8,
                        "preprocessor": {
                            "features": 8,
                            "n_fft": 16,
                            "window_size": 0.001,
                            "window_stride": 0.0005,
                        },
                        "encoder": {
                            "feat_in": 8,
                            "n_layers": 1,
                            "d_model": 8,
                            "n_heads": 2,
                            "ff_expansion_factor": 2,
                            "subsampling_factor": 2,
                            "subsampling_conv_channels": 4,
                            "conv_kernel_size": 3,
                            "att_context_size": [4, 0],
                        },
                    },
                }
            },
            "speech_generation": {
                "model": {
                    "codec_config": {
                        "base_hidden_size": 8,
                        "channel_mult": [1],
                        "rates": [2],
                        "num_blocks": 1,
                        "kernel_size": 3,
                        "latent_size": 4,
                        "n_fft": 4,
                        "hop_length": 2,
                        "num_quantizers": 2,
                        "codebook_size": 8,
                    },
                    "tts_config": {
                        "backbone_config": {
                            "hidden_size": 8,
                            "intermediate_size": 16,
                            "num_hidden_layers": 2,
                            "num_attention_heads": 2,
                            "num_key_value_heads": 2,
                            "head_dim": 4,
                            "sliding_window": 16,
                        },
                        "latent_size": 4,
                        "codebook_size": 8,
                        "num_quantizers": 2,
                        "mog_head_config": {
                            "intermediate_size": 16,
                            "num_layers": 1,
                            "num_predictions": 8,
                            "low_rank": 2,
                        },
                    },
                }
            },
        },
    }


class MiniTokenizer:
    bos_token_id = 5
    eos_token_id = 6

    def get_vocab(self):
        return {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "ab": 4,
            "<s>": 5,
            "</s>": 6,
            "<SPECIAL_12>": 7,
        }


def gemma_args(num_hidden_layers=1):
    return gemma3_text.ModelArgs(
        model_type="gemma3_text",
        hidden_size=8,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=8,
        sliding_window=16,
        sliding_window_pattern=1,
    )


def test_gemma3_preserves_external_embedding_scale():
    model = gemma3_text.Gemma3Model(gemma_args())
    embeddings = mx.random.normal((2, 3, 8))

    output = model(None, input_embeddings=embeddings)
    expected = embeddings
    mask = create_attention_mask(expected)
    for layer in model.layers:
        expected = layer(expected, mask)
    expected = model.norm(expected)

    assert mx.allclose(output, expected).item()


def test_gemma3_batched_cache_matches_full_forward():
    model = gemma3_text.Gemma3Model(gemma_args())
    embeddings = mx.random.normal((2, 4, 8))
    full_output = model(None, input_embeddings=embeddings)
    cache = [KVCache()]
    prefix_output = model(None, cache=cache, input_embeddings=embeddings[:, :3])
    mx.eval(prefix_output)
    cached_output = model(None, cache=cache, input_embeddings=embeddings[:, 3:])
    mx.eval(full_output, cached_output)

    assert mx.allclose(cached_output, full_output[:, 3:], atol=1e-2, rtol=1e-2).item()


def test_official_config_is_detected():
    assert (
        infer_model_type_from_config({"model": {"stt": {}, "speech_generation": {}}})
        == "nemotron_voicechat"
    )


def test_stt_and_tts_incremental_steps():
    model = Model(ModelConfig.from_dict(mini_config()))
    model.tts_model.set_tokenizer(MiniTokenizer())

    state = model.stt_model.initialize(
        mx.zeros((320,), dtype=mx.float32), None, pad_id=7
    )
    text_token = model.stt_model.step(
        0,
        state,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
    )
    previous_codes, cache = model.tts_model.initialize(batch_size=1, pad_id=7, eos_id=6)
    codes, cache = model.tts_model.tts_model.generate_step(
        text_token,
        previous_codes,
        cache,
        text_eos_id=6,
        silence_codes=model.tts_model.codec_silence_tokens[None, None],
        guidance_enabled=True,
    )
    mx.eval(text_token, codes)

    assert text_token.shape == (1, 1)
    assert codes.shape == (1, 1, 2)
    assert len(cache) == 2
    assert (
        model.tts_model.tts_model.embed_subword.subword_flag_emb.is_continuation[
            4
        ].item()
        == 1
    )
    assert (
        model.tts_model.tts_model.embed_subword.bos_eos_emb.special_flags[5].item() == 1
    )
    assert (
        model.tts_model.tts_model.embed_subword.bos_eos_emb.special_flags[6].item() == 2
    )


def test_sanitize_convolution_layouts():
    model = Model(ModelConfig.from_dict(mini_config()))
    weights = {
        "stt_model.perception.encoder.pre_encode.conv.0.weight": mx.zeros((4, 1, 3, 3)),
        "stt_model.llm.layers.0.mixer.conv1d.weight": mx.zeros((12, 1, 4)),
        "tts_model.tts_model.rvq_embs": mx.zeros((2, 8, 4)),
        "tts_model.tts_model.audio_prompt_projection_W": mx.zeros((8, 8)),
        "stt_model.rnnt_decoder.prediction.embed.weight": mx.zeros((8, 4)),
    }
    converted = model.sanitize(weights)

    assert converted["stt_model.perception.encoder.pre_encode.conv.0.weight"].shape == (
        4,
        3,
        3,
        1,
    )
    assert converted["stt_model.llm.layers.0.mixer.conv1d.weight"].shape == (12, 4, 1)
    assert converted["tts_model.tts_model.rvq_embs"].shape == (2, 8, 4)
    assert "tts_model.tts_model.audio_prompt_projection_W" not in converted
    assert not any(key.startswith("stt_model.rnnt_") for key in converted)


def test_quantize_only_supported_linear_weights():
    weights = {
        "stt_model.llm.linear.weight": mx.zeros((8, 64)),
        "stt_model.llm.linear.bias": mx.zeros((8,)),
        "stt_model.perception.linear.weight": mx.zeros((8, 64)),
        "stt_model.llm.conv.weight": mx.zeros((8, 3, 4)),
    }
    quantized = _quantize(weights, group_size=64, bits=4)

    assert "stt_model.llm.linear.scales" in quantized
    assert "stt_model.llm.linear.biases" in quantized
    assert quantized["stt_model.llm.linear.bias"].shape == (8,)
    assert quantized["stt_model.perception.linear.weight"].shape == (8, 64)
    assert quantized["stt_model.llm.conv.weight"].shape == (8, 3, 4)
