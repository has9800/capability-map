from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .data import CAPABILITIES, Example


def _accuracy(model, examples: Sequence[Example], ablated_heads: Sequence[Tuple[int, int]]) -> float:
    ablated_set = set(ablated_heads)

    def ablate_hook(z: torch.Tensor, hook):
        layer = hook.layer()
        for h in range(z.shape[2]):
            if (layer, h) in ablated_set:
                z[:, :, h, :] = 0.0
        return z

    hooks = [(f"blocks.{l}.attn.hook_z", ablate_hook) for l in range(model.cfg.n_layers)]
    correct = 0
    for ex in examples:
        logits = model.run_with_hooks(ex["prompt"], fwd_hooks=hooks)
        correct_id = model.to_single_token(ex["correct_answer"])
        incorrect_id = model.to_single_token(ex["incorrect_answer"])
        correct += int(logits[0, -1, correct_id] > logits[0, -1, incorrect_id])
    return correct / max(1, len(examples))


def _curve_for_order(model, examples, ordered_heads, plateau_batch: int, cliff_batch: int, floor: float):
    curve = []
    removed: List[Tuple[int, int]] = []
    total = len(ordered_heads)
    idx = 0

    while idx < total:
        acc = _accuracy(model, examples, removed)
        n_remaining = total - len(removed)
        curve.append({"n_remaining": n_remaining, "accuracy": acc})
        if acc <= floor:
            break
        batch = plateau_batch if acc > 0.9 else cliff_batch
        remove_now = ordered_heads[idx : idx + batch]
        removed.extend(remove_now)
        idx += len(remove_now)

    return curve


def run_progressive_pruning(model, dataset: Dict[str, List[Example]], config: dict, input_dir: Path, output_dir: Path, debug: bool = False):
    prune_cfg = config["prune"]
    plateau_batch = 50 if debug else int(prune_cfg.get("batch_size_plateau", 10))
    cliff_batch = 1 if debug else int(prune_cfg.get("batch_size_cliff", 1))
    floor = float(prune_cfg.get("accuracy_floor", 0.0))
    n_random = int(prune_cfg.get("n_random_baselines", 5))

    output_dir.mkdir(parents=True, exist_ok=True)

    for capability in CAPABILITIES:
        effects = np.load(input_dir / f"{capability}_effects.npy")
        n_layers, n_heads = effects.shape
        heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        ordering = np.argsort(np.abs(effects).reshape(-1))
        specific_order = [heads[i] for i in ordering]

        curves = {
            "capability_specific": _curve_for_order(
                model,
                dataset[capability],
                specific_order,
                plateau_batch,
                cliff_batch,
                floor,
            )
        }

        random_curves = []
        for _ in tqdm(range(n_random), desc=f"{capability} random baselines"):
            shuffled = heads.copy()
            np.random.shuffle(shuffled)
            random_curves.append(
                _curve_for_order(model, dataset[capability], shuffled, plateau_batch, cliff_batch, floor)
            )
        curves["random_baselines"] = random_curves

        cliff = next((p for p in curves["capability_specific"] if p["accuracy"] < 0.7), None)
        summary = {
            "cliff_point": cliff,
            "minimum_viable_heads": min((p["n_remaining"] for p in curves["capability_specific"] if p["accuracy"] >= 0.7), default=0),
        }

        with (output_dir / f"{capability}_curves.json").open("w") as f:
            json.dump(curves, f, indent=2)
        with (output_dir / f"{capability}_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
