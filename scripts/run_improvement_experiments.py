#!/usr/bin/env python3
"""Run nested four-track FMRG improvement experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmrg_submission.experiments import (  # noqa: E402
    load_cached_aligned_data,
    run_experiments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cache_dir = args.cache_dir or args.raw_dir.parent / "cache"
    data = load_cached_aligned_data(
        cache_dir,
        args.raw_dir,
        aligned_output_dir=args.output_dir / "tables",
    )
    result = run_experiments(data, args.output_dir)
    summary = {
        "incumbent_mae_mm": result["incumbent"]["metrics"][
            "track_balanced_width_mae_mm"
        ],
        "candidate_mae_mm": result["candidates"]["nested_metrics"][
            "track_balanced_width_mae_mm"
        ],
        "promoted": result["promotion"]["promoted"],
        "selected": result["selected"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
