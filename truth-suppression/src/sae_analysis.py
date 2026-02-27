"""SAE utilities for truth-suppression analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import PromptPair

RELEASE_CANDIDATES = [
    "pythia-1.4b-deduped-res-sm",
    "pythia-160m-deduped-res-sm",
    "pythia-70m-deduped-res-sm",
]
MODEL_NAME_BY_RELEASE = {
    "pythia-1.4b-deduped-res-sm": "EleutherAI/pythia-1.4b-deduped",
    "pythia-160m-deduped-res-sm": "EleutherAI/pythia-160m-deduped",
    "pythia-70m-deduped-res-sm": "EleutherAI/pythia-70m-deduped",
}


@dataclass
class SAEContext:
    release: str
    model_name: str
    available_layers: List[int]
    sae_by_layer: Dict[int, SAE]


def try_load_first_layer(release: str, device: str) -> bool:
    try:
        SAE.from_pretrained(release=release, sae_id="blocks.0.hook_resid_post", device=device)
        return True
    except Exception:
        return False


def resolve_release(device: str = "cpu", candidates: Sequence[str] = RELEASE_CANDIDATES) -> str:
    for release in candidates:
        if try_load_first_layer(release=release, device=device):
            return release
    raise RuntimeError(f"No usable SAE release found in candidates: {candidates}")


def load_model(model_name: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def discover_layers(release: str, max_layers: int, device: str = "cpu") -> List[int]:
    layers: List[int] = []
    for layer in range(max_layers):
        sae_id = f"blocks.{layer}.hook_resid_post"
        try:
            SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
            layers.append(layer)
        except Exception:
            continue
    return layers


def load_saes(release: str, layers: Iterable[int], device: str) -> Dict[int, SAE]:
    saes: Dict[int, SAE] = {}
    for layer in layers:
        sae_id = f"blocks.{layer}.hook_resid_post"
        sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        sae.eval()
        saes[layer] = sae
    return saes


def build_sae_context(device: str = "cpu", max_layers: int = 40) -> SAEContext:
    release = resolve_release(device=device)
    model_name = MODEL_NAME_BY_RELEASE[release]
    layers = discover_layers(release=release, max_layers=max_layers, device=device)
    if not layers:
        raise RuntimeError(f"No layers found for SAE release {release}.")
    sae_by_layer = load_saes(release=release, layers=layers, device=device)
    return SAEContext(release=release, model_name=model_name, available_layers=layers, sae_by_layer=sae_by_layer)


@torch.inference_mode()
def extract_topk_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae_by_layer: Dict[int, SAE],
    prompt: str,
    layers: Sequence[int],
    top_k: int,
    device: str,
) -> Dict[int, Dict[int, float]]:
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**toks, output_hidden_states=True)
    hidden_states = out.hidden_states
    result: Dict[int, Dict[int, float]] = {}

    for layer in layers:
        resid_post = hidden_states[layer + 1][:, -1, :]
        acts = sae_by_layer[layer].encode(resid_post).squeeze(0)
        k = min(top_k, acts.shape[0])
        values, indices = torch.topk(torch.abs(acts), k=k)
        _ = values  # keep symmetry with explicit indexing below
        feature_map = {int(idx): float(acts[idx].item()) for idx in indices}
        result[layer] = feature_map
    return result


def dense_from_sparse(feature_map: Dict[int, float], size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    for idx, val in feature_map.items():
        if 0 <= idx < size:
            vec[idx] = val
    return vec


def compare_pair(
    truth_features: Dict[int, Dict[int, float]],
    secret_features: Dict[int, Dict[int, float]],
    d_sae_by_layer: Dict[int, int],
) -> Dict[int, Dict[str, List[Tuple[int, float]]]]:
    output: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}
    for layer, d_sae in d_sae_by_layer.items():
        truth_dense = dense_from_sparse(truth_features.get(layer, {}), d_sae)
        secret_dense = dense_from_sparse(secret_features.get(layer, {}), d_sae)
        diff = truth_dense - secret_dense
        top_truth = np.argsort(diff)[-10:][::-1]
        top_secret = np.argsort(diff)[:10]
        output[layer] = {
            "top_truth_features": [(int(i), float(diff[i])) for i in top_truth if diff[i] > 0],
            "top_secret_features": [(int(i), float(diff[i])) for i in top_secret if diff[i] < 0],
        }
    return output


def run_dataset_analysis(
    pairs: Sequence[PromptPair],
    context: SAEContext,
    device: str,
    top_k: int = 100,
) -> dict:
    tokenizer, model = load_model(context.model_name, device=device)
    d_sae_by_layer = {layer: int(context.sae_by_layer[layer].cfg.d_sae) for layer in context.available_layers}

    pair_results = []
    aggregate_diffs: Dict[int, np.ndarray] = {
        layer: np.zeros(d_sae_by_layer[layer], dtype=np.float64) for layer in context.available_layers
    }

    for pair in tqdm(pairs, desc="Analyzing prompt pairs"):
        truth_features = extract_topk_features(
            model=model,
            tokenizer=tokenizer,
            sae_by_layer=context.sae_by_layer,
            prompt=pair.truth_prompt,
            layers=context.available_layers,
            top_k=top_k,
            device=device,
        )
        secret_features = extract_topk_features(
            model=model,
            tokenizer=tokenizer,
            sae_by_layer=context.sae_by_layer,
            prompt=pair.secret_prompt,
            layers=context.available_layers,
            top_k=top_k,
            device=device,
        )

        for layer in context.available_layers:
            d_sae = d_sae_by_layer[layer]
            t = dense_from_sparse(truth_features.get(layer, {}), d_sae)
            s = dense_from_sparse(secret_features.get(layer, {}), d_sae)
            aggregate_diffs[layer] += (t - s)

        per_layer_diff = compare_pair(
            truth_features=truth_features,
            secret_features=secret_features,
            d_sae_by_layer=d_sae_by_layer,
        )

        pair_results.append(
            {
                "pair_id": pair.pair_id,
                "topic": pair.topic,
                "question": pair.question,
                "answer": pair.answer,
                "truth_features_topk": truth_features,
                "secret_features_topk": secret_features,
                "per_layer_differentials": per_layer_diff,
            }
        )

    num_pairs = max(1, len(pairs))
    aggregate = {}
    for layer, diff_vec in aggregate_diffs.items():
        mean_diff = diff_vec / num_pairs
        top = np.argsort(np.abs(mean_diff))[-50:][::-1]
        aggregate[layer] = {
            "top_abs_diff_features": [
                {"feature": int(i), "mean_diff": float(mean_diff[i]), "abs_mean_diff": float(abs(mean_diff[i]))}
                for i in top
                if mean_diff[i] != 0
            ]
        }

    return {
        "release": context.release,
        "model_name": context.model_name,
        "available_layers": context.available_layers,
        "top_k": top_k,
        "pair_results": pair_results,
        "aggregate": aggregate,
    }


def save_results(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
