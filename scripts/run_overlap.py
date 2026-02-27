#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.overlap import run_overlap_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["output"]["dir"])
    run_overlap_analysis(
        config=config,
        patch_dir=root / "patch",
        prune_dir=root / "prune",
        output_dir=root / "overlap",
    )


if __name__ == "__main__":
    main()
