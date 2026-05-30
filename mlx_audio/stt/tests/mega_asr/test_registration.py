def test_megaasrconfig_from_dict_defaults():
    from mlx_audio.stt.models.mega_asr import MegaASRConfig

    cfg = MegaASRConfig.from_dict({"model_type": "mega_asr"})
    assert cfg.model_type == "mega_asr"
    assert cfg.router_weights == "extras/router.safetensors"
    assert cfg.lora_weights == "extras/lora.safetensors"


def test_megaasrconfig_from_dict_custom_paths():
    from mlx_audio.stt.models.mega_asr import MegaASRConfig

    cfg = MegaASRConfig.from_dict(
        {
            "model_type": "mega_asr",
            "router_weights": "custom/router.safetensors",
            "lora_weights": "custom/lora.safetensors",
        }
    )
    assert cfg.router_weights == "custom/router.safetensors"
    assert cfg.lora_weights == "custom/lora.safetensors"


def test_to_qwen3_dict_sets_correct_type():
    from mlx_audio.stt.models.mega_asr import MegaASRConfig

    cfg = MegaASRConfig.from_dict({"model_type": "mega_asr"})
    q3cfg = cfg.to_qwen3_dict()
    assert q3cfg["model_type"] == "qwen3_asr"


def test_megaasrconfig_nested_configs():
    from mlx_audio.stt.models.mega_asr import MegaASRConfig

    cfg = MegaASRConfig.from_dict(
        {
            "model_type": "mega_asr",
            "audio_config": {"num_mel_bins": 128, "d_model": 1024},
            "text_config": {"vocab_size": 151936, "num_hidden_layers": 28},
        }
    )
    assert cfg.audio_config is not None
    assert cfg.text_config is not None
    assert cfg.audio_config.num_mel_bins == 128
    assert cfg.audio_config.d_model == 1024
    assert cfg.text_config.vocab_size == 151936
    assert cfg.text_config.num_hidden_layers == 28
