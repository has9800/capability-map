"""Visualization helpers for truth suppression experiments."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _get_prob_curve(run: Dict) -> np.ndarray:
    return np.array([layer["truth_prob"] for layer in run["layers"]], dtype=float)


def save_pair_plot(topic: str, truth_run: Dict, secret_run: Dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    truth_curve = _get_prob_curve(truth_run)
    secret_curve = _get_prob_curve(secret_run)
    layers = np.arange(len(truth_curve))

    plt.figure(figsize=(8, 4))
    plt.plot(layers, truth_curve, color="royalblue", label="truth condition")
    plt.plot(layers, secret_curve, color="firebrick", label="secret condition")
    plt.xlabel("Layer index")
    plt.ylabel("P(true token)")
    plt.title(f"Truth-token probability across layers: {topic}")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, f"pair_{topic}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_heatmap(all_results: List[Dict], out_path: str) -> str:
    topics = [r["topic"] for r in all_results]
    matrix = []
    for r in all_results:
        truth_curve = _get_prob_curve(r["truth_condition"])
        secret_curve = _get_prob_curve(r["secret_condition"])
        matrix.append(truth_curve - secret_curve)

    data = np.vstack(matrix)
    plt.figure(figsize=(12, max(6, len(topics) * 0.35)))
    sns.heatmap(
        data,
        cmap="bwr",
        center=0.0,
        yticklabels=topics,
        xticklabels=np.arange(data.shape[1]),
    )
    plt.xlabel("Layer")
    plt.ylabel("Contrastive pair")
    plt.title("Suppression heatmap: P(true|truth) - P(true|secret)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()
    return out_path


def save_average_suppression_curve(all_results: List[Dict], out_path: str) -> Dict:
    diffs = []
    for r in all_results:
        truth_curve = _get_prob_curve(r["truth_condition"])
        secret_curve = _get_prob_curve(r["secret_condition"])
        diffs.append(truth_curve - secret_curve)

    diff_arr = np.vstack(diffs)
    avg_curve = diff_arr.mean(axis=0)
    peak_layer = int(np.argmax(avg_curve))

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(avg_curve)), avg_curve, color="purple")
    plt.axvline(peak_layer, linestyle="--", color="black", alpha=0.7, label=f"peak={peak_layer}")
    plt.xlabel("Layer")
    plt.ylabel("Average suppression")
    plt.title("Average suppression curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

    return {
        "average_suppression": avg_curve.tolist(),
        "peak_layer": peak_layer,
        "peak_value": float(avg_curve[peak_layer]),
        "path": out_path,
    }
