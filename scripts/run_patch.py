#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data import load_capability_data
from src.model import load_model
from src.patch import run_activation_patching


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    bundle = load_model(config, debug=args.debug)
    data = load_capability_data(config, debug=args.debug)
    run_activation_patching(
        model=bundle.model,
        dataset=data,
        config=config,
        output_dir=Path(config["output"]["dir"]) / "patch",
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
