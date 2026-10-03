"""Materialize Phonon's checksummed byte-plane transport archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download

PHONON_CONFIG_SCHEMA = "fermion.phonon/1"
PHONON_RELEASE_FORMATS = frozenset(
    {
        # The published family uses the original internal format name; the
        # release packer in the public repository uses the renamed spelling.
        "sttg1a-byteplane-tar-zstd-v1",
        "phonon-byteplane-tar-zstd-v1",
    }
)
_MARKER = ".mlx_audio_phonon.json"
_BLOCK_SIZE = 16 << 20


def is_phonon_model(model_path: Path, config: dict[str, Any]) -> bool:
    if config.get("config_schema") == PHONON_CONFIG_SCHEMA:
        return True
    manifest_path = model_path / "packed_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return str(manifest.get("format", "")).startswith("sttg1a-")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_BLOCK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in name:
        raise RuntimeError(f"unsafe path in Phonon archive: {name!r}")
    return path


def _join_byte_planes(data: bytes, meta: dict[str, Any]) -> bytes:
    """Reverse the per-tensor byte-plane transport transform."""

    original_bytes = int(meta["original_bytes"])
    base = int(meta["base"])
    payload_bytes = int(meta["payload_bytes"])
    if len(data) != original_bytes or not 0 <= base <= original_bytes:
        raise RuntimeError("invalid Phonon byte-plane payload size")
    if payload_bytes != original_bytes - base:
        raise RuntimeError("invalid Phonon byte-plane payload metadata")

    raw = np.frombuffer(data, dtype=np.uint8)
    output = np.empty(original_bytes, dtype=np.uint8)
    output[:base] = raw[:base]
    source = base
    cursor = 0
    for span in meta["plan"]:
        if len(span) != 3:
            raise RuntimeError("invalid Phonon byte-plane span")
        start, end, itemsize = map(int, span)
        if not (cursor <= start <= end <= payload_bytes) or itemsize <= 1:
            raise RuntimeError("invalid Phonon byte-plane span")
        if (end - start) % itemsize:
            raise RuntimeError("misaligned Phonon byte-plane span")
        if start > cursor:
            width = start - cursor
            output[base + cursor : base + start] = raw[source : source + width]
            source += width
        width = end - start
        per_plane = width // itemsize
        block = raw[source : source + width].reshape(itemsize, per_plane)
        output[base + start : base + end] = block.T.reshape(-1)
        source += width
        cursor = end

    remaining = payload_bytes - cursor
    if remaining:
        output[base + cursor :] = raw[source : source + remaining]
        source += remaining
    if source != len(raw):
        raise RuntimeError("Phonon byte-plane payload was not consumed exactly")
    return output.tobytes()


def _extract_archive(archive: Path, destination: Path, expected_bytes: int) -> None:
    try:
        import zstandard
    except ImportError as error:
        raise ImportError(
            "Phonon archives require zstandard; install mlx-audio[stt]"
        ) from error

    destination.mkdir(parents=True, exist_ok=True)
    manifest = None
    index: dict[str, dict[str, Any]] = {}
    written: set[str] = set()
    total_bytes = 0

    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        raise RuntimeError(
                            f"unsupported member in Phonon archive: {member.name!r}"
                        )
                    handle = tar.extractfile(member)
                    if handle is None:
                        raise RuntimeError(
                            f"could not read Phonon archive member: {member.name!r}"
                        )
                    blob = handle.read()
                    if len(blob) != member.size:
                        raise RuntimeError(
                            f"truncated Phonon archive member: {member.name!r}"
                        )

                    if manifest is None:
                        if member.name != "bps_manifest.json":
                            raise RuntimeError(
                                "bps_manifest.json must be first in a Phonon archive"
                            )
                        manifest = json.loads(blob)
                        if manifest.get("release_format") not in PHONON_RELEASE_FORMATS:
                            raise RuntimeError("unsupported Phonon release format")
                        rows = manifest.get("files", [])
                        for row in rows:
                            name = str(row.get("path", ""))
                            _safe_relative_path(name)
                            if not name or name in index:
                                raise RuntimeError(
                                    "duplicate or empty path in Phonon manifest"
                                )
                            index[name] = row
                        continue

                    transformed = member.name.endswith(".bps")
                    relative = member.name[:-4] if transformed else member.name
                    _safe_relative_path(relative)
                    if relative not in index or relative in written:
                        raise RuntimeError(
                            f"unexpected Phonon archive member: {member.name!r}"
                        )
                    row = index[relative]
                    if transformed != ("transform" in row):
                        raise RuntimeError(
                            f"transform mismatch for Phonon member: {relative!r}"
                        )
                    data = (
                        _join_byte_planes(blob, row["transform"])
                        if transformed
                        else blob
                    )
                    expected_size = int(row["original_bytes"])
                    expected_sha = str(row["original_sha256"])
                    if len(data) != expected_size:
                        raise RuntimeError(
                            f"size mismatch for Phonon member {relative}"
                        )
                    if hashlib.sha256(data).hexdigest() != expected_sha:
                        raise RuntimeError(
                            f"SHA-256 mismatch for Phonon member {relative}"
                        )

                    path = _safe_relative_path(relative)
                    target = destination.joinpath(*path.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    written.add(relative)
                    total_bytes += len(data)

    if manifest is None:
        raise RuntimeError("empty Phonon archive")
    missing = set(index) - written
    if missing:
        raise RuntimeError(f"Phonon archive is missing members: {sorted(missing)}")
    if total_bytes != expected_bytes:
        raise RuntimeError(
            f"Phonon unpacked size mismatch: expected {expected_bytes}, got {total_bytes}"
        )


def _default_cache_root() -> Path:
    override = os.environ.get("MLX_AUDIO_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "mlx_audio"


def _is_complete(path: Path, archive_sha256: str) -> bool:
    marker = path / _MARKER
    try:
        metadata = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("archive_sha256") == archive_sha256
        and (path / "config.json").is_file()
        and (path / "packed_manifest.json").is_file()
        and any(path.glob("*.safetensors"))
    )


def prepare_model_path(
    model_path: Path,
    *,
    source: str | None = None,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Return an ordinary model directory for a Phonon repo or archive directory."""

    model_path = Path(model_path).expanduser().resolve()
    if (model_path / "packed_manifest.json").is_file() and any(
        model_path.glob("*.safetensors")
    ):
        return model_path

    outer_config_path = model_path / "config.json"
    if not outer_config_path.is_file():
        raise FileNotFoundError(f"Phonon config not found at {model_path}")
    outer_config = json.loads(outer_config_path.read_text())
    if outer_config.get("config_schema") != PHONON_CONFIG_SCHEMA:
        raise RuntimeError("not a supported Phonon release repository")

    artifact = outer_config.get("artifact", {})
    filename = str(artifact.get("filename", ""))
    archive_sha256 = str(artifact.get("sha256", ""))
    expected_bytes = int(artifact.get("unpacked_bytes", 0))
    if not filename.endswith(".bps.tar.zst") or len(archive_sha256) != 64:
        raise RuntimeError("invalid Phonon artifact metadata")
    if expected_bytes <= 0:
        raise RuntimeError("invalid Phonon unpacked size")

    archive = model_path / filename
    if not archive.is_file():
        repo_id = source or outer_config.get("model_id")
        if not repo_id:
            raise FileNotFoundError(f"Phonon archive not found: {archive}")
        model_path = Path(
            snapshot_download(
                str(repo_id),
                revision=revision,
                allow_patterns=[filename, "*.json", "*.txt", "README.md"],
                force_download=force_download,
            )
        )
        archive = model_path / filename
    if not archive.is_file():
        raise FileNotFoundError(f"Phonon archive not found: {archive}")
    if _sha256_file(archive) != archive_sha256:
        raise RuntimeError(f"SHA-256 mismatch for Phonon archive {archive.name}")

    destination = _default_cache_root() / "phonon" / archive_sha256
    if _is_complete(destination, archive_sha256):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{archive_sha256[:12]}-", dir=destination.parent)
    )
    try:
        _extract_archive(archive, temporary, expected_bytes)
        marker = {
            "archive_sha256": archive_sha256,
            "model_id": outer_config.get("model_id"),
            "profile": artifact.get("profile"),
        }
        (temporary / _MARKER).write_text(json.dumps(marker, indent=2) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
