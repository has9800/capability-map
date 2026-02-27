from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .data import CAPABILITIES


def _significant(ci_low: np.ndarray, ci_high: np.ndarray) -> np.ndarray:
    return np.logical_or(ci_low > 0, ci_high < 0)


def run_overlap_analysis(config: dict, patch_dir: Path, prune_dir: Path, output_dir: Path):
    overlap_cfg = config["overlap"]
    percentile = float(overlap_cfg.get("importance_percentile", 80))
    require_significant = bool(overlap_cfg.get("require_significant", True))

    effects: Dict[str, np.ndarray] = {}
    important_masks: Dict[str, np.ndarray] = {}

    for cap in CAPABILITIES:
        eff = np.load(patch_dir / f"{cap}_effects.npy")
        effects[cap] = eff
        threshold = np.percentile(np.abs(eff), percentile)
        important = np.abs(eff) >= threshold
        if require_significant:
            ci_low = np.load(patch_dir / f"{cap}_ci_low.npy")
            ci_high = np.load(patch_dir / f"{cap}_ci_high.npy")
            important = np.logical_and(important, _significant(ci_low, ci_high))
        important_masks[cap] = important

    n_layers, n_heads = next(iter(effects.values())).shape
    labels: List[Dict[str, object]] = []

    for l in range(n_layers):
        for h in range(n_heads):
            active = [cap for cap in CAPABILITIES if important_masks[cap][l, h]]
            if not active:
                label = "redundant"
            elif len(active) >= 3:
                label = "general"
            elif len(active) == 1:
                label = f"{active[0]}-only"
            else:
                label = "+".join(sorted(active))
            labels.append({"layer": l, "head": h, "label": label, "capabilities": active})

    pairwise = {}
    for a, b in product(CAPABILITIES, repeat=2):
        a_mask = important_masks[a]
        b_mask = important_masks[b]
        denom = max(1, int(a_mask.sum()))
        pairwise[f"{a}_to_{b}"] = float(np.logical_and(a_mask, b_mask).sum() / denom)

    pruning_summary = {}
    for cap in CAPABILITIES:
        with (prune_dir / f"{cap}_summary.json").open() as f:
            pruning_summary[cap] = json.load(f)

    shared_rh = np.logical_and(important_masks["reasoning"], important_masks["honesty"])
    shared_heads = [(int(i // n_heads), int(i % n_heads)) for i in np.where(shared_rh.reshape(-1))[0]]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "head_labels.json").open("w") as f:
        json.dump(labels, f, indent=2)
    with (output_dir / "overlap_matrix.json").open("w") as f:
        json.dump(pairwise, f, indent=2)

    summary_lines = [
        "Capability overlap summary",
        f"Total heads: {n_layers * n_heads}",
        f"Reasoning-honesty shared heads: {len(shared_heads)}",
        f"Shared heads list: {shared_heads}",
        "Pruning cliff points:",
    ]
    for cap in CAPABILITIES:
        summary_lines.append(f"- {cap}: {pruning_summary[cap].get('cliff_point')}")
    (output_dir / "overlap_summary.txt").write_text("\n".join(summary_lines))
