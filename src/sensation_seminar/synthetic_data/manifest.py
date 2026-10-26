"""Canonical JSONL manifest construction and loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .mapping import SUPPORTED_EXTENSIONS, artifact_reason, map_synthetic_filename
from .validation import eligibility, inspect_image, supervision_for

SCHEMA_VERSION = "1.1"


def _image_index(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
            result[path.stem] = path
    return result


def _portable_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def discover_synthetic_images(synthetic_root: Path) -> Iterator[tuple[str, Path]]:
    """Yield supported files recursively in deterministic group/path order."""
    groups = sorted(
        (path for path in synthetic_root.iterdir() if path.is_dir()),
        key=lambda path: (int(path.name[5:]) if path.name.startswith("group") and path.name[5:].isdigit() else 10**9, path.name),
    )
    for group in groups:
        for path in sorted(group.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield group.name, path


def build_manifest_records(
    data_root: str | Path,
    supervision_config: Mapping[str, Any],
    *,
    repository_root: str | Path = ".",
    content_hashes: bool = False,
) -> Iterator[dict[str, Any]]:
    """Build records without modifying either real or synthetic source data."""
    data_root = Path(data_root)
    repository_root = Path(repository_root)
    real_root = data_root / "sensation_ds_v2_1_1152x768"
    synthetic_root = data_root / "synthetic-data"
    split_images = {
        split: _image_index(real_root / split / "images")
        for split in ("training", "validation", "testing")
    }
    split_stems = {split: set(images) for split, images in split_images.items()}

    for source_group, path in discover_synthetic_images(synthetic_root):
        relpath = _portable_path(path, repository_root)
        artifact = artifact_reason(path.relative_to(synthetic_root))
        mapping = map_synthetic_filename(path.name, split_stems)
        width, height, mode, decode_error = inspect_image(path)
        supervision = supervision_for(
            source_group,
            relpath,
            (width, height) if width is not None and height is not None else None,
            supervision_config,
        )
        reasons = [reason for reason in (artifact, decode_error, mapping.reason_code) if reason]
        eligible = eligibility(
            artifact=artifact,
            decode_error=decode_error,
            mapping_error=mapping.reason_code,
            supervision=supervision,
        )
        real_image = (
            split_images[mapping.real_split][mapping.original_id]
            if mapping.original_id and mapping.real_split
            else None
        )
        real_mask = (
            real_root / mapping.real_split / "masks" / f"{mapping.original_id}.png"
            if mapping.original_id and mapping.real_split
            else None
        )
        identity = f"{source_group}\0{relpath}".encode("utf-8")
        yield {
            "schema_version": SCHEMA_VERSION,
            "synthetic_id": f"sha256:{hashlib.sha256(identity).hexdigest()}",
            "source_group": source_group,
            "source_relpath": relpath,
            "original_filename": path.name,
            "extension": path.suffix.lower(),
            "original_id": mapping.original_id,
            "real_image_relpath": _portable_path(real_image, repository_root) if real_image else None,
            "real_mask_relpath": _portable_path(real_mask, repository_root) if real_mask else None,
            "real_split": mapping.real_split,
            "generation_tokens": list(mapping.generation_tokens),
            "variant": mapping.variant,
            "width": width,
            "height": height,
            "mode": mode,
            "content_sha256": _file_sha256(path) if content_hashes and decode_error is None else None,
            "validity": {"status": "valid" if not reasons else "invalid", "reason_codes": reasons},
            "eligibility": eligible,
            "supervision": supervision,
            "generation_metadata": {},
        }


def write_jsonl(records: Iterable[Mapping[str, Any]], output: str | Path) -> int:
    """Atomically write deterministic JSONL and return its record count."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(output)
    return count


def load_manifest(path: str | Path, *, eligibility_filter: str | None = None) -> list[dict[str, Any]]:
    """Load JSONL and optionally retain `analysis` or `segmentation_training` rows."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported manifest schema on line {line_number}")
            if eligibility_filter and not record.get("eligibility", {}).get(eligibility_filter, False):
                continue
            records.append(record)
    return records
