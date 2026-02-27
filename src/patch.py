from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .data import CAPABILITIES, Example


@dataclass
class PatchResult:
    effects: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    n_prompts, n_layers, n_heads = values.shape
    if n_bootstrap <= 0:
        mean = values.mean(axis=0)
        return mean, mean
    samples = np.zeros((n_bootstrap, n_layers, n_heads), dtype=np.float32)
    for b in range(n_bootstrap):
        idx = np.random.randint(0, n_prompts, size=n_prompts)
        samples[b] = values[idx].mean(axis=0)
    lo = np.percentile(samples, 100 * (alpha / 2), axis=0)
    hi = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def _collect_attribution_effects(model, clean_prompt: str, corrupted_prompt: str, correct_id: int, incorrect_id: int) -> torch.Tensor:
    n_layers = model.cfg.n_layers
    names_filter = lambda name: name.endswith("attn.hook_z")

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_prompt, names_filter=names_filter, return_type="logits")

    model.zero_grad()
    # Preferred route: cache forward and backward tensors through TransformerLens hooks.
    try:
        cache_dict, fwd_hooks, bwd_hooks = model.get_caching_hooks(names_filter=names_filter, incl_bwd=True)
        with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, reset_hooks_end=False):
            corrupted_logits = model(corrupted_prompt, return_type="logits")

        metric = corrupted_logits[0, -1, correct_id] - corrupted_logits[0, -1, incorrect_id]
        metric.backward()

        effects = []
        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.attn.hook_z"
            clean_z = clean_cache[hook_name]
            corrupted_z = cache_dict[hook_name]
            grad_z = cache_dict[f"{hook_name}_grad"]
            delta = clean_z - corrupted_z
            head_effects = (grad_z * delta).sum(dim=(0, 1, 3))
            effects.append(head_effects)
        model.reset_hooks()
        stacked = torch.stack(effects, dim=0)

    except AttributeError:
        # Fallback route for older TransformerLens versions.
        retained = {}

        def retain_grad_hook(tensor: torch.Tensor, hook):
            tensor.retain_grad()
            retained[hook.name] = tensor
            return tensor

        retain_hooks = [(f"blocks.{layer}.attn.hook_z", retain_grad_hook) for layer in range(n_layers)]
        corrupted_logits = model.run_with_hooks(
            corrupted_prompt,
            fwd_hooks=retain_hooks,
            reset_hooks_end=False,
            return_type="logits",
        )

        metric = corrupted_logits[0, -1, correct_id] - corrupted_logits[0, -1, incorrect_id]
        metric.backward()

        effects = []
        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.attn.hook_z"
            clean_z = clean_cache[hook_name]
            corrupted_z = retained[hook_name]
            grad_z = corrupted_z.grad
            if grad_z is None:
                raise RuntimeError(f"Missing gradient for {hook_name}; attribution patching failed")
            delta = clean_z - corrupted_z.detach()
            head_effects = (grad_z * delta).sum(dim=(0, 1, 3))
            effects.append(head_effects)
        model.reset_hooks()
        stacked = torch.stack(effects, dim=0)

    # Keep sign convention consistent with previous activation patching:
    # negative means the head supports the correct answer.
    return stacked


def run_activation_patching(
    model,
    dataset: Dict[str, List[Example]],
    config: dict,
    output_dir: Path,
    debug: bool = False,
) -> Dict[str, PatchResult]:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    patch_cfg = config["patch"]
    n_bootstrap = 0 if debug else int(patch_cfg.get("n_bootstrap", 1000))

    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, PatchResult] = {}

    for capability in CAPABILITIES:
        examples = dataset[capability]
        n = len(examples)
        per_prompt = np.zeros((n, n_layers, n_heads), dtype=np.float32)
        print(f"Patching capability: {capability} ({n} prompts)")

        corrupted_indices = np.random.permutation(n)
        if n > 1:
            same = corrupted_indices == np.arange(n)
            corrupted_indices[same] = (corrupted_indices[same] + 1) % n

        for i, example in enumerate(tqdm(examples, desc=f"{capability} prompts")):
            corrupted = examples[int(corrupted_indices[i])]
            clean_prompt = example["prompt"]
            corrupted_prompt = corrupted["prompt"]

            correct_id = model.to_single_token(example["correct_answer"])
            incorrect_id = model.to_single_token(example["incorrect_answer"])
            effects = _collect_attribution_effects(
                model=model,
                clean_prompt=clean_prompt,
                corrupted_prompt=corrupted_prompt,
                correct_id=correct_id,
                incorrect_id=incorrect_id,
            )
            per_prompt[i] = effects.detach().cpu().numpy()

        effects = per_prompt.mean(axis=0)
        ci_low, ci_high = bootstrap_ci(per_prompt, n_bootstrap=n_bootstrap)

        np.save(output_dir / f"{capability}_effects.npy", effects)
        np.save(output_dir / f"{capability}_ci_low.npy", ci_low)
        np.save(output_dir / f"{capability}_ci_high.npy", ci_high)

        flat = np.argsort(np.abs(effects).reshape(-1))[-20:][::-1]
        top_heads = [(int(idx // n_heads), int(idx % n_heads), float(effects.reshape(-1)[idx])) for idx in flat]
        print(f"Top 20 most necessary heads for {capability} (layer, head, effect):")
        for entry in top_heads:
            print(entry)

        results[capability] = PatchResult(effects=effects, ci_low=ci_low, ci_high=ci_high)

    return results
