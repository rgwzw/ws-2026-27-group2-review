#!/usr/bin/env python3
"""Generate Group 2 top/bottom examples and selected-source distribution."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "synthetic-data-seminar/manifest.jsonl"
RESULTS_DIR = PROJECT_ROOT / "group-2/results/fallback-v1"
SCORES_PATH = RESULTS_DIR / "scores.jsonl"
SUBSET_PATH = RESULTS_DIR / "perceptual_quality_budget_0.25x.jsonl"
EXAMPLES_PATH = RESULTS_DIR / "top_bottom_examples.jpg"
DISTRIBUTION_PATH = RESULTS_DIR / "source_group_distribution.csv"


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_image_path(source_relpath: str) -> Path:
    source_path = Path(source_relpath)

    if source_path.is_absolute() and source_path.is_file():
        return source_path

    project_relative = PROJECT_ROOT / source_path
    if project_relative.is_file():
        return project_relative

    parts = source_path.parts
    if "synthetic-data" in parts:
        suffix = parts[parts.index("synthetic-data") :]
        relocated = PROJECT_ROOT / "datasets" / Path(*suffix)
        if relocated.is_file():
            return relocated

    raise FileNotFoundError(f"Could not resolve image: {source_relpath}")


def load_font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scores = [
        (row["synthetic_id"], float(row["score"]))
        for row in iter_jsonl(SCORES_PATH)
    ]
    scores.sort(key=lambda item: (-item[1], item[0]))

    subset_rows = list(iter_jsonl(SUBSET_PATH))
    subset_ids = [row["synthetic_id"] for row in subset_rows]

    top = scores[:10]
    bottom = list(reversed(scores[-10:]))
    wanted_ids = set(subset_ids)
    wanted_ids.update(synthetic_id for synthetic_id, _ in top + bottom)

    manifest = {}
    for row in iter_jsonl(MANIFEST_PATH):
        synthetic_id = row["synthetic_id"]
        if synthetic_id in wanted_ids:
            manifest[synthetic_id] = row

    missing = wanted_ids.difference(manifest)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} required IDs from manifest")

    distribution = Counter(
        manifest[synthetic_id].get("source_group", "unknown")
        for synthetic_id in subset_ids
    )

    with DISTRIBUTION_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_group", "selected_count", "selected_fraction"])
        for source_group, count in sorted(distribution.items()):
            writer.writerow([source_group, count, f"{count / len(subset_ids):.6f}"])

    examples = [
        ("TOP", rank, synthetic_id, score)
        for rank, (synthetic_id, score) in enumerate(top, start=1)
    ] + [
        ("BOTTOM", rank, synthetic_id, score)
        for rank, (synthetic_id, score) in enumerate(bottom, start=1)
    ]

    columns = 5
    tile_width = 320
    image_height = 213
    label_height = 52
    tile_height = image_height + label_height
    rows = (len(examples) + columns - 1) // columns

    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(13)
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    for index, (kind, rank, synthetic_id, score) in enumerate(examples):
        record = manifest[synthetic_id]
        image_path = resolve_image_path(record["source_relpath"])

        with Image.open(image_path) as image:
            tile = ImageOps.fit(
                image.convert("RGB"),
                (tile_width, image_height),
                method=resampling,
            )

        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(tile, (x, y))
        draw.rectangle(
            (x, y + image_height, x + tile_width, y + tile_height),
            fill="white",
        )

        source_group = record.get("source_group", "unknown")
        filename = image_path.name
        if len(filename) > 35:
            filename = filename[:32] + "..."
        label = (
            f"{kind} {rank:02d} | score={score:.4f}\n"
            f"{source_group} | {filename}"
        )
        draw.multiline_text(
            (x + 5, y + image_height + 5),
            label,
            fill="black",
            font=font,
            spacing=2,
        )

    canvas.save(EXAMPLES_PATH, quality=92, optimize=True)

    print(f"Created: {EXAMPLES_PATH}")
    print(f"Created: {DISTRIBUTION_PATH}")
    print("\nSelected 548 source-group distribution:")
    for source_group, count in sorted(distribution.items()):
        print(f"{source_group}: {count} ({count / len(subset_ids):.2%})")


if __name__ == "__main__":
    main()

