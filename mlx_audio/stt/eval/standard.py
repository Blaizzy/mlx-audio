from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

from .schema import STTEvalSample

ID_COLUMNS = ("utt_id", "id", "sample_id")
AUDIO_COLUMNS = ("audio_path", "path", "audio")
REFERENCE_COLUMNS = ("reference_text", "text", "transcript")


def sample_from_standard_row(
    row: Mapping[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
) -> STTEvalSample:
    """Build an eval sample from a row with common STT dataset columns.

    Accepted column aliases:
    - ID: `utt_id`, `id`, `sample_id`
    - Audio: `audio_path`, `path`, `audio`
    - Reference: `reference_text`, `text`, `transcript`
    """
    utt_id = _first_present(row, ID_COLUMNS)
    audio_value = _first_present(row, AUDIO_COLUMNS)
    reference_text = _first_present(row, REFERENCE_COLUMNS)

    if isinstance(audio_value, Mapping):
        audio_value = audio_value.get("path")

    if not utt_id:
        raise ValueError(f"Standard eval row is missing an ID column: {ID_COLUMNS}")
    if not audio_value:
        raise ValueError(
            f"Standard eval row is missing an audio column: {AUDIO_COLUMNS}"
        )
    if reference_text is None:
        raise ValueError(
            f"Standard eval row is missing a reference column: {REFERENCE_COLUMNS}"
        )

    audio_path = Path(str(audio_value))
    if base_dir is not None and not audio_path.is_absolute():
        audio_path = Path(base_dir) / audio_path

    metadata = {
        key: value
        for key, value in row.items()
        if key
        not in {
            *ID_COLUMNS,
            *AUDIO_COLUMNS,
            *REFERENCE_COLUMNS,
        }
    }

    return STTEvalSample(
        utt_id=str(utt_id),
        audio_path=audio_path,
        reference_text=str(reference_text),
        source_path=str(audio_value),
        metadata=metadata,
    )


def iter_standard_eval_samples(
    rows: Iterable[Mapping[str, Any]],
    *,
    base_dir: Optional[str | Path] = None,
):
    for row in rows:
        yield sample_from_standard_row(row, base_dir=base_dir)


def _first_present(row: Mapping[str, Any], columns: tuple[str, ...]):
    for column in columns:
        if column in row:
            return row[column]
    return None
