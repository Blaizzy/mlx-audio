from mlx_audio.registry import (
    SUPPORTED_MODEL_TYPES,
    classify_model,
    is_supported_model,
    kinds,
    model_type_from_config,
)


def test_kinds_discovered_with_voice_first():
    discovered = kinds()
    assert {"tts", "stt", "sts"} <= set(discovered)
    assert discovered[:3] == ("tts", "stt", "sts")
    assert "music" in discovered


def test_supported_types_are_populated():
    assert len(SUPPORTED_MODEL_TYPES["tts"]) > 10
    assert len(SUPPORTED_MODEL_TYPES["stt"]) > 10
    assert "kokoro" in SUPPORTED_MODEL_TYPES["tts"]
    assert "whisper" in SUPPORTED_MODEL_TYPES["stt"]
    assert "phonon" in SUPPORTED_MODEL_TYPES["stt"]
    assert "minimax_music3" in SUPPORTED_MODEL_TYPES["music"]


def test_classify_text_to_speech():
    assert classify_model("csm", "Marvis-AI/marvis-tts-250m") == "tts"
    assert classify_model("kokoro", "prince-canuma/Kokoro-82M") == "tts"
    assert classify_model("dia", "nari-labs/Dia-1.6B") == "tts"
    assert classify_model("bark", "suno/bark") == "tts"


def test_phonon_tts_is_identified_by_config():
    from mlx_audio.utils import get_model_category, get_model_name_parts

    model_type = model_type_from_config({"flow_lm": {}, "mimi": {}})
    assert model_type == "pocket_tts"
    assert classify_model(model_type, "gradium/phonon-31m-en") == "tts"
    assert classify_model(model_type, "gradium/phonon-40m-en") == "tts"
    assert (
        get_model_category(model_type, get_model_name_parts("gradium/phonon-40m-es"))
        == "tts"
    )


def test_phonon_tts_explicit_type_takes_precedence_over_name():
    from mlx_audio.utils import get_model_category, get_model_name_parts

    model_type = model_type_from_config(
        {"model_type": "gradium_phonon_tts", "flow_lm": {}, "mimi": {}}
    )
    assert model_type in SUPPORTED_MODEL_TYPES["tts"]
    assert classify_model(model_type) == "tts"
    assert classify_model(model_type, "gradium/phonon-40m-es") == "tts"
    assert (
        get_model_category(model_type, get_model_name_parts("gradium/phonon-40m-es"))
        == "tts"
    )


def test_classify_speech_to_text():
    assert classify_model("tdt", "mlx-community/parakeet-tdt-0.6b-v2") == "stt"
    assert classify_model("whisper", "openai/whisper-base") == "stt"
    assert classify_model("moonshine", "UsefulSensors/moonshine-base") == "stt"
    assert classify_model("qwen3_asr", "Qwen/Qwen3-ASR") == "stt"
    assert (
        classify_model(
            "granite_speech5_ctc",
            "ibm-granite/granite-speech-5.0-470m-turboctc",
        )
        == "stt"
    )
    assert classify_model("", "FermionResearch/Phonon-1") == "stt"


def test_classify_other_voice_kinds():
    assert classify_model("moshi", "kyutai/moshiko") == "sts"
    assert classify_model("nemotron_voicechat", "") == "sts"
    assert classify_model("mimi", "kyutai/mimi") == "codec"


def test_classify_music_model():
    assert (
        classify_model("minimax_music3", "mlx-community/MiniMax-Music3-mxfp8")
        == "music"
    )


def test_repo_name_recovers_generic_model_type():
    # parakeet's config model_type is "tdt"; the family is recovered from the name.
    assert classify_model("tdt", "parakeet") == "stt"


def test_language_models_are_not_audio():
    assert classify_model("llama", "meta-llama/Llama-3.1-8B") is None
    assert classify_model("qwen3", "Qwen/Qwen3-8B") is None
    assert classify_model("gemma3", "google/gemma-3-4b") is None
    assert classify_model("", "") is None
    assert not is_supported_model("foobar", "acme/whatever")
