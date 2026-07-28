from dataclasses import dataclass
from unittest.mock import patch

from mlx_audio.stt.eval.runner import _batched, run_seed_tts_eval, run_stt_wer_eval
from mlx_audio.stt.eval.schema import STTEvalSample
from mlx_audio.stt.eval.seed_tts import SeedTTSSample


@dataclass
class FakeOutput:
    text: str


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, audio_path, language=None, temperature=None):
        self.calls.append(
            {"audio_path": audio_path, "language": language, "temperature": temperature}
        )
        return FakeOutput(text="hello world")


def test_run_seed_tts_eval_with_fake_model(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    sample = SeedTTSSample(
        utt_id="utt",
        audio_path=audio_path,
        reference_text="Hello, world!",
        source_path="en/wavs/utt.wav",
    )
    fake_model = FakeModel()

    with patch(
        "mlx_audio.stt.eval.runner.iter_seed_tts_english_samples",
        return_value=iter([sample]),
    ):
        summary = run_seed_tts_eval(
            model=fake_model,
            output_dir=tmp_path / "out",
            language="en",
            gen_kwargs={"temperature": 0.0, "unused": "ignored"},
        )

    assert summary["num_samples"] == 1
    assert summary["metrics"] == ["wer"]
    assert summary["wer_micro"] == 0.0
    assert fake_model.calls == [
        {
            "audio_path": str(audio_path),
            "language": "en",
            "temperature": 0.0,
        }
    ]
    predictions = (tmp_path / "out" / "predictions.jsonl").read_text()
    assert '"hypothesis": "hello world"' in predictions


def test_run_stt_wer_eval_accepts_bring_your_own_samples(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    sample = STTEvalSample(
        utt_id="utt",
        audio_path=audio_path,
        reference_text="Hello, world!",
        source_path="custom/sample.wav",
        metadata={"speaker": "abc"},
    )

    summary = run_stt_wer_eval(
        model=FakeModel(),
        samples=[sample],
        output_dir=tmp_path / "out",
        dataset_name="custom",
        dataset_split="test",
    )

    assert summary["dataset_name"] == "custom"
    assert summary["dataset_split"] == "test"
    assert summary["wer_micro"] == 0.0
    prediction = (tmp_path / "out" / "predictions.jsonl").read_text()
    assert '"metadata": {"speaker": "abc"}' in prediction


def test_run_stt_wer_eval_accepts_explicit_metrics(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    sample = STTEvalSample(
        utt_id="utt",
        audio_path=audio_path,
        reference_text="Hello, world!",
    )

    summary = run_stt_wer_eval(
        model=FakeModel(),
        samples=[sample],
        output_dir=tmp_path / "out",
        dataset_name="custom",
        metrics=["WER", "wer"],
    )

    assert summary["metrics"] == ["wer"]


def test_run_stt_wer_eval_rejects_unsupported_metrics(tmp_path):
    try:
        run_stt_wer_eval(
            model=FakeModel(),
            samples=[],
            output_dir=tmp_path / "out",
            dataset_name="custom",
            metrics=["cer"],
        )
    except ValueError as exc:
        assert "Unsupported metric" in str(exc)
    else:
        raise AssertionError("Expected unsupported metric failure")


def test_run_seed_tts_eval_dataset_batch_size_processes_all_samples(tmp_path):
    samples = []
    for idx in range(3):
        audio_path = tmp_path / f"sample-{idx}.wav"
        audio_path.write_bytes(b"not a real wav")
        samples.append(
            SeedTTSSample(
                utt_id=f"utt-{idx}",
                audio_path=audio_path,
                reference_text="Hello, world!",
                source_path=f"en/wavs/utt-{idx}.wav",
            )
        )
    fake_model = FakeModel()

    with patch(
        "mlx_audio.stt.eval.runner.iter_seed_tts_english_samples",
        return_value=iter(samples),
    ):
        summary = run_seed_tts_eval(
            model=fake_model,
            output_dir=tmp_path / "out",
            dataset_batch_size=2,
        )

    assert summary["num_samples"] == 3
    assert summary["dataset_batch_size"] == 2
    assert len(fake_model.calls) == 3
    predictions = (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()
    assert len(predictions) == 3


def test_batched_yields_final_partial_batch():
    assert list(_batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_run_seed_tts_eval_rejects_invalid_dataset_batch_size(tmp_path):
    try:
        run_seed_tts_eval(
            model=FakeModel(),
            output_dir=tmp_path / "out",
            dataset_batch_size=0,
        )
    except ValueError as exc:
        assert "dataset_batch_size" in str(exc)
    else:
        raise AssertionError("Expected invalid dataset_batch_size failure")


def test_run_seed_tts_eval_rejects_empty_hypotheses(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    sample = SeedTTSSample(
        utt_id="utt",
        audio_path=audio_path,
        reference_text="Hello",
        source_path="en/wavs/utt.wav",
    )

    class EmptyModel:
        def generate(self, audio_path):
            return FakeOutput(text="")

    with patch(
        "mlx_audio.stt.eval.runner.iter_seed_tts_english_samples",
        return_value=iter([sample]),
    ):
        try:
            run_seed_tts_eval(model=EmptyModel(), output_dir=tmp_path / "out")
        except RuntimeError as exc:
            assert "hypotheses were empty" in str(exc)
        else:
            raise AssertionError("Expected empty hypothesis failure")
