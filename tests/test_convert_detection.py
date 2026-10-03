from pathlib import Path

from mlx_audio import convert


def test_path_detection_prefers_specific_model_family(monkeypatch):
    patterns = {
        convert.Domain.TTS: {"vibevoice": {"vibevoice"}},
        convert.Domain.STT: {"vibevoice_asr": {"vibevoice-asr", "vibevoice_asr"}},
    }
    monkeypatch.setattr(
        convert,
        "get_detection_hints",
        lambda domain: {"path_patterns": patterns.get(domain, {})},
    )

    for checkpoint in (
        "microsoft/VibeVoice-ASR-Streaming-1.5B",
        "microsoft/VibeVoice-ASR-Streaming-7B",
    ):
        match = convert._match_by_path(Path(checkpoint))
        assert match == (convert.Domain.STT, "vibevoice_asr")
