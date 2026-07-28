from pathlib import Path
from unittest.mock import patch

from mlx_audio.stt.eval.seed_tts import (
    _streaming_audio_path_to_relpath,
    iter_seed_tts_english_samples,
    materialize_streamed_audio,
    parse_seed_tts_meta_line,
)


def test_parse_seed_tts_meta_line_derives_target_wav():
    entry = parse_seed_tts_meta_line(
        "common_voice_en_a-common_voice_en_b|Prompt text|prompt-wavs/a.wav|Target text"
    )
    assert entry.utt_id == "common_voice_en_a-common_voice_en_b"
    assert entry.target_text == "Target text"
    assert entry.target_wav == "en/wavs/common_voice_en_a-common_voice_en_b.wav"


def test_parse_seed_tts_meta_line_accepts_explicit_target_wav():
    entry = parse_seed_tts_meta_line(
        "utt.wav|Prompt|prompt-wavs/p.wav|Target|en/wavs/custom.wav"
    )
    assert entry.utt_id == "utt"
    assert entry.target_wav == "en/wavs/custom.wav"


def test_streaming_audio_path_to_relpath_for_hf_path():
    relpath = _streaming_audio_path_to_relpath(
        "hf://datasets/zhaochenyang20/seed-tts-eval@abc/en/wavs/utt.wav"
    )
    assert relpath == "en/wavs/utt.wav"


def test_materialize_streamed_audio_uses_inline_bytes(tmp_path):
    path = materialize_streamed_audio(
        audio={"bytes": b"RIFFfake", "path": "hf://unused/en/wavs/utt.wav"},
        audio_path="hf://unused/en/wavs/utt.wav",
        cache_dir=tmp_path,
        utt_id="utt",
    )
    assert path == tmp_path / "utt.wav"
    assert path.read_bytes() == b"RIFFfake"


def test_iter_seed_tts_english_samples_filters_prompt_rows(tmp_path):
    rows = [
        {"audio": {"path": "hf://repo@abc/en/prompt-wavs/prompt.wav", "bytes": b"p"}},
        {"audio": {"path": "hf://repo@abc/en/wavs/utt.wav", "bytes": b"RIFFtarget"}},
    ]
    references = {
        "utt": parse_seed_tts_meta_line("utt|Prompt|prompt-wavs/prompt.wav|Hello")
    }

    with (
        patch(
            "mlx_audio.stt.eval.seed_tts.load_seed_tts_english_references",
            return_value=references,
        ),
        patch("mlx_audio.stt.eval.seed_tts._load_streaming_dataset", return_value=rows),
    ):
        samples = list(iter_seed_tts_english_samples(audio_cache_dir=tmp_path, limit=1))

    assert len(samples) == 1
    assert samples[0].utt_id == "utt"
    assert samples[0].reference_text == "Hello"
    assert Path(samples[0].audio_path).read_bytes() == b"RIFFtarget"
