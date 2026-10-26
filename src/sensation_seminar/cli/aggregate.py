"""Aggregate every valid result bundle found below a directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sensation_seminar.results.aggregate import aggregate_results, load_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("baseline/results"))
    parser.add_argument("--output", type=Path, default=Path("baseline/aggregate.json"))
    args = parser.parse_args(argv)
    aggregate = aggregate_results(load_results(args.results_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(aggregate)} aggregate rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
