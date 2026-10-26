"""Create the reusable deterministic synthetic comparison subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from sensation_seminar.synthetic_data.manifest import load_manifest, write_jsonl
from sensation_seminar.synthetic_data.selection import canonical_checksum, random_training_subset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=Path("synthetic-data-seminar/manifest.jsonl"))
    result.add_argument("--method", choices=["random"], default="random")
    result.add_argument("--budget", choices=["0.25x", "0.50x"], required=True)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    records = load_manifest(args.manifest)
    selected = random_training_subset(records, budget=args.budget, seed=args.seed)
    output = args.output or Path(
        f"synthetic-data-seminar/subsets/{args.method}_budget_{args.budget}.jsonl"
    )
    write_jsonl(selected, output)
    print(f"wrote {len(selected)} records to {output} ({canonical_checksum(selected)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
