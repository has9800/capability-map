from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

REQUIRED_KEYS = {"prompt", "correct_answer", "incorrect_answer"}
CAPABILITIES = ("reasoning", "recall", "syntax", "honesty")


Example = Dict[str, str]
Dataset = Dict[str, List[Example]]


def _validate_examples(examples: List[Example], name: str) -> None:
    if not isinstance(examples, list):
        raise ValueError(f"{name} must contain a list of examples")
    for i, ex in enumerate(examples):
        missing = REQUIRED_KEYS - set(ex.keys())
        if missing:
            raise ValueError(f"{name}[{i}] missing required keys: {sorted(missing)}")
        for key in REQUIRED_KEYS:
            value = ex[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}[{i}].{key} must be a non-empty string")


def load_capability_data(config: dict, debug: bool = False) -> Dataset:
    data_cfg = config["data"]
    dataset: Dataset = {}

    if debug:
        sample_path = Path("data/sample.json")
        payload = json.loads(sample_path.read_text())
        for capability in CAPABILITIES:
            examples = payload[capability]
            _validate_examples(examples, capability)
            dataset[capability] = examples
            print(f"Loaded {capability}: {len(examples)} examples (debug sample)")
        return dataset

    for capability in CAPABILITIES:
        path = Path(data_cfg[f"{capability}_path"])
        examples = json.loads(path.read_text())
        _validate_examples(examples, capability)
        dataset[capability] = examples
        print(f"Loaded {capability}: {len(examples)} examples from {path}")

    return dataset
