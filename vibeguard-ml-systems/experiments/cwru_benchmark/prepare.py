"""CLI: download the official CWRU files and build fixed feature windows."""

from __future__ import annotations

import argparse
import json

from experiments.cwru_benchmark.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/cwru/raw")
    parser.add_argument("--output-dir", default="data/cwru")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use files already in --raw-dir.",
    )
    args = parser.parse_args()
    result = prepare_dataset(args.raw_dir, args.output_dir, download=not args.no_download)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
