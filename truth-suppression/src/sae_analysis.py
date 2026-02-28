"""SAE utilities for truth-suppression analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sparsify import Sae
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import PromptPair

SAE_REPO = "EleutherAI/sae-pythia-410m-65k"
MODEL_NAME = "EleutherAI/pythia-410m"
NUM_LAYERS = 24
D_MODEL = 1024
NUM_SAE_FEATURES = 65536


@dataclass
class SAEContext:
    sae_repo: str
    model_name: str
    available_layers: List[int]
    sae_by_layer: Dict[int, Sae]


def load_model(model_name: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def discover_layers(sae_repo: str, max_layers: int, device: str = "cpu") -> List[int]:
    layers: List[int] = []
    for layer in range(max_layers):
        hookpoint = f"layers.{layer}"
        try:
            sae = Sae.load_from_hub(sae_repo, hookpoint=hookpoint)
            sae.to(device)
            layers.append(layer)
        except Exception:
            continue
    return layers


def load_saes(sae_repo: str, layers: Sequence[int], device: str) -> Dict[int, Sae]:
    saes: Dict[int, Sae] = {}
    for layer in layers:
        hookpoint = f"layers.{layer}"
        sae = Sae.load_from_hub(sae_repo, hookpoint=hookpoint)
        sae.to(device)
        sae.eval()
        saes[layer] = sae
    return saes


def build_sae_context(device: str = "cpu", max_layers: int = NUM_LAYERS) -> SAEContext:
    layers = discover_layers(sae_repo=SAE_REPO, max_layers=max_layers, device=device)
    if not layers:
        raise RuntimeError(f"No layers found for SAE repo {SAE_REPO}.")
    sae_by_layer = load_saes(sae_repo=SAE_REPO, layers=layers, device=device)
    return SAEContext(
        sae_repo=SAE_REPO,
        model_name=MODEL_NAME,
        available_layers=layers,
        sae_by_layer=sae_by_layer,
    )


@torch.inference_mode()
def extract_features_for_layer(sae: Sae, hidden_state: torch.Tensor, top_k: int = 100) -> Dict[int, float]:
    """Extract top-k SAE features from a single hidden state vector."""
    output = sae.encode(hidden_state.unsqueeze(0))
    indices = output.top_indices[0].detach().cpu().tolist()
    acts = output.top_acts[0].detach().cpu().tolist()

    feature_map = dict(zip(indices, acts))
    sorted_feats = sorted(feature_map.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return dict(sorted_feats)


@torch.inference_mode()
def extract_topk_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae_by_layer: Dict[int, Sae],
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
        resid_post = hidden_states[layer + 1][:, -1, :].squeeze(0)
        result[layer] = extract_features_for_layer(sae=sae_by_layer[layer], hidden_state=resid_post, top_k=top_k)
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
    d_sae_by_layer = {layer: int(context.sae_by_layer[layer].num_latents) for layer in context.available_layers}

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
        "sae_repo": context.sae_repo,
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
