"""Shared STS entrypoint tests without downloads or large checkpoints."""

import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from mlx_audio import audio_io
from mlx_audio.sts import generate as cli
from mlx_audio.sts.models.base import STSOutput


@pytest.mark.parametrize("speech", [False, True])
def test_generate_returns_result_and_writes_text_and_audio(tmp_path, speech):
    result = STSOutput(
        audio=np.linspace(-0.5, 0.5, 80, dtype=np.float32) if speech else None,
        text="Hello.",
        sample_rate=16000,
    )
    calls = []

    def generate(audio, **kwargs):
        calls.append((audio, kwargs))
        return result

    output = tmp_path / "nested" / "response.wav"
    actual = cli.generate_audio(
        model=SimpleNamespace(generate=generate),
        text="Say hello",
        temperature=0.9,
        gen_kwargs={"temperature": 0.0, "max_new_tokens": 7},
        output_path=output,
    )
    assert actual is result
    assert calls == [
        (None, {"text": "Say hello", "temperature": 0.0, "max_new_tokens": 7})
    ]
    assert output.with_suffix(".txt").read_text() == "Hello.\n"
    assert output.exists() == speech
    if speech:
        audio, rate = audio_io.read(str(output))
        assert rate == 16000
        np.testing.assert_allclose(audio, result.audio, atol=2 / 32768)


def test_cli_preserves_model_defaults_and_loads_arbitrary_model_name(
    tmp_path, monkeypatch
):
    calls = []

    def generate(
        audio, *, task="model_default", temperature=0.3, voice="default_voice"
    ):
        calls.append((audio, task, temperature, voice))
        return STSOutput(text="Answer")

    def load_model(path, **kwargs):
        assert path == "unrecognized/repository"
        assert kwargs == {"strict": True, "revision": "test"}
        return SimpleNamespace(generate=generate)

    monkeypatch.setattr(cli, "load_model", load_model)
    output = tmp_path / "output.wav"
    cli.main(
        [
            "--model",
            "unrecognized/repository",
            "--output-path",
            str(output),
            "--model-kwargs",
            '{"strict": true, "revision": "test"}',
        ]
    )
    assert calls == [(None, "model_default", 0.3, "default_voice")]
    assert output.with_suffix(".txt").read_text() == "Answer\n"


@pytest.mark.parametrize("field", ["messages", "segments", "prompt_examples"])
def test_cli_reads_json_input_and_passes_generation_options(
    tmp_path, monkeypatch, field
):
    value = [{"role": "user", "content": "Hi"}]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(value))
    captured = {}
    monkeypatch.setattr(cli, "generate_audio", lambda **kwargs: captured.update(kwargs))
    cli.main(
        [
            "--model",
            "any/model",
            f"--{field.replace('_', '-')}",
            str(path),
            "--natural-instruction",
            "--audio-tokenizer-path",
            "codec",
            "--gen-kwargs",
            '{"max_new_tokens": 12}',
            "--version",
            "2",
        ]
    )
    assert captured == {
        "model": "any/model",
        field: value,
        "read_text_only": False,
        "audio_tokenizer_path": "codec",
        "gen_kwargs": {"max_new_tokens": 12},
        "model_kwargs": {"version": 2},
    }


def test_model_path_objects_and_loading_options(tmp_path, monkeypatch):
    model_path = tmp_path / "checkpoint"
    calls = []

    def load_model(path, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace(generate=lambda audio: STSOutput(text="ok"))

    monkeypatch.setattr(cli, "load_model", load_model)
    cli.generate_audio(
        model=model_path, model_kwargs={"lazy": True}, output_path=tmp_path / "out.wav"
    )
    assert calls == [(model_path, {"lazy": True})]


def test_named_stems_and_default_separation_filenames(tmp_path):
    from mlx_audio.sts.models.dialogue_sidon import Model

    calls = []

    def separate(audio, **kwargs):
        calls.append((audio, kwargs))
        return SimpleNamespace(speakers=np.zeros((2, 12)), sample_rate=24000)

    model = SimpleNamespace(separate=separate, output_suffix=Model.output_suffix)
    model.generate = lambda audio, **kwargs: Model.generate(model, audio, **kwargs)
    source = tmp_path / "input.wav"
    result = cli.generate_audio(model=model, audio=source, seed=42, num_steps=8)
    assert result.audio is None and result.text is None
    assert set(result.audios) == {"speaker_1", "speaker_2"}
    assert calls == [
        (
            source,
            {
                "sample_rate": None,
                "num_steps": 8,
                "seed": 42,
                "chunk_seconds": 20.0,
                "overlap_seconds": 5.0,
            },
        )
    ]
    for i in (1, 2):
        waveform, rate = audio_io.read(
            str(tmp_path / f"input_separated_speaker_{i}.wav")
        )
        assert waveform.shape == (12,) and rate == 24000
    assert not (tmp_path / "input_separated.txt").exists()


@pytest.mark.parametrize("stream", [False, True])
def test_deepfilternet_adapter_matches_existing_enhancement_paths(tmp_path, stream):
    from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

    source = tmp_path / "noisy.wav"
    stereo = np.stack([np.linspace(-0.25, 0.25, 48), np.zeros(48)], axis=-1)
    audio_io.write(str(source), stereo, 48000)
    calls = []

    def enhance(audio, **kwargs):
        calls.append((audio, kwargs))
        return audio * 0.5

    model = SimpleNamespace(
        config=SimpleNamespace(sample_rate=48000),
        enhance_array=enhance,
        enhance_array_streaming=enhance,
        output_suffix=DeepFilterNetModel.output_suffix,
    )
    model.generate = MethodType(DeepFilterNetModel.generate, model)
    result = cli.generate_audio(
        model=model, audio=source, stream=stream, chunk_samples=960
    )
    expected_kwargs = (
        {"chunk_samples": 960, "pad_end_frames": 3, "compensate_delay": True}
        if stream
        else {}
    )
    assert calls[0][1] == expected_kwargs
    np.testing.assert_allclose(calls[0][0], stereo[:, 0], atol=4e-5)
    assert result.sample_rate == 48000
    saved, rate = audio_io.read(str(tmp_path / "noisy_enhanced.wav"))
    assert saved.shape == (48,) and rate == 48000
    assert not (tmp_path / "noisy_enhanced.txt").exists()


def test_mossformer_adapter_passes_chunking_and_validates_array_sample_rate(tmp_path):
    from mlx_audio.sts.models.mossformer2_se import MossFormer2SEModel

    calls = []

    def enhance(audio, **kwargs):
        calls.append((audio, kwargs))
        return np.zeros(8)

    model = SimpleNamespace(config=SimpleNamespace(sample_rate=48000), enhance=enhance)
    result = MossFormer2SEModel.generate(model, tmp_path / "input.wav", chunked=True)
    assert calls == [(str(tmp_path / "input.wav"), {"chunked": True})]
    assert result.sample_rate == 48000 and result.audio.shape == (8,)
    with pytest.raises(ValueError, match="Expected 48000"):
        MossFormer2SEModel.generate(model, np.zeros(8), sample_rate=16000)


def test_deepfilternet_adapter_rejects_wrong_sample_rate_and_empty_audio():
    from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

    model = SimpleNamespace(config=SimpleNamespace(sample_rate=48000))
    with pytest.raises(ValueError, match="Expected 48000"):
        DeepFilterNetModel.generate(model, np.zeros(8), sample_rate=16000)
    with pytest.raises(ValueError, match="nonempty"):
        DeepFilterNetModel.generate(model, np.zeros(0))


def test_unsupported_streaming_fails_before_generation():
    def generate(audio):
        pytest.fail("Unsupported streaming should fail before generating")

    with pytest.raises(ValueError, match="streaming"):
        cli.generate_audio(model=SimpleNamespace(generate=generate), stream=True)


def test_stream_false_works_for_offline_models(tmp_path):
    result = STSOutput(text="ok")
    model = SimpleNamespace(generate=lambda audio: result)
    assert (
        cli.generate_audio(model=model, stream=False, output_path=tmp_path / "out.wav")
        is result
    )


@pytest.mark.parametrize("flag", ["--gen-kwargs", "--model-kwargs"])
def test_kwargs_require_json_objects(flag):
    with pytest.raises(SystemExit):
        cli.parse_args([flag, "[]"])


def test_invalid_model_and_legacy_outputs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be provided"):
        cli.generate_audio()
    with pytest.raises(TypeError, match="does not implement"):
        cli.generate_audio(model=object())
    model = SimpleNamespace(generate=lambda audio: np.zeros(8))
    with pytest.raises(TypeError, match="audio and sample_rate"):
        cli.generate_audio(model=model, output_path=tmp_path / "out.wav")
    with pytest.raises(ValueError, match="model_kwargs"):
        cli.generate_audio(model=model, model_kwargs={"strict": True})


@pytest.mark.parametrize("version", [1, 2, 3])
def test_loader_selects_version_without_leaking_options_to_config(
    tmp_path, monkeypatch, version
):
    from mlx_audio.sts import utils

    version_dir = tmp_path / f"v{version}"
    version_dir.mkdir()
    (version_dir / "config.json").write_text(
        json.dumps({"model_version": f"DeepFilterNet{version}"})
    )
    captured = {}
    monkeypatch.setattr(
        utils, "base_load_model", lambda **kwargs: captured.update(kwargs) or "loaded"
    )
    assert utils.load_model(tmp_path, version=version, subfolder="ignored") == "loaded"
    assert captured["model_path"] == version_dir
    assert "version" not in captured and "subfolder" not in captured


def test_loader_subfolder_metadata_and_missing_version(tmp_path, monkeypatch):
    from mlx_audio.sts import utils

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "config.json").write_text('{"model_type": "ming_v3_audio"}')
    captured = {}
    monkeypatch.setattr(
        utils, "base_load_model", lambda **kwargs: captured.update(kwargs)
    )
    utils.load_model(tmp_path, subfolder="nested")
    assert captured["model_path"] == nested
    assert captured["model_type"] == "ming_v3_audio"
    versioned = tmp_path / "v1"
    versioned.mkdir()
    (versioned / "config.json").write_text('{"model_type": "ming_v3_audio"}')
    with pytest.raises(ValueError, match="DeepFilterNet"):
        utils.load_model(tmp_path, version=1)
    with pytest.raises(ValueError, match="Choose from"):
        utils.load_model(tmp_path, version=4)
    with pytest.raises(FileNotFoundError, match="v2"):
        utils.load_model(tmp_path, version=2)
