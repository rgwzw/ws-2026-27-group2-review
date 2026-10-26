"""Command-line entry point for the non-destructive synthetic-data index."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sensation_seminar.config import load_yaml
from sensation_seminar.synthetic_data.manifest import build_manifest_records, write_jsonl


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, default=Path("datasets"))
    result.add_argument("--output", type=Path, default=Path("synthetic-data-seminar/manifest.jsonl"))
    result.add_argument(
        "--supervision-config",
        type=Path,
        default=Path("configs/data/synthetic_supervision.yaml"),
    )
    result.add_argument("--repository-root", type=Path, default=Path("."))
    result.add_argument("--content-hashes", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_yaml(args.supervision_config)
    counts: Counter[str] = Counter()

    def observed_records():
        for record in build_manifest_records(
            args.data_root,
            config,
            repository_root=args.repository_root,
            content_hashes=args.content_hashes,
        ):
            counts["records"] += 1
            if record["eligibility"]["analysis"]:
                counts["analysis_eligible"] += 1
            if record["eligibility"]["segmentation_training"]:
                counts["training_eligible"] += 1
            yield record

    written = write_jsonl(observed_records(), args.output)
    print(
        f"wrote {written} records to {args.output} "
        f"(analysis={counts['analysis_eligible']}, training={counts['training_eligible']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
