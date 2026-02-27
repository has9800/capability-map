"""Visualization utilities for SAE truth-suppression analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def _aggregate_matrix(results: dict, top_n: int = 20) -> Tuple[np.ndarray, List[str], List[int]]:
    layers = results["available_layers"]
    aggregate = results["aggregate"]
    row_labels = []
    rows = []
    for layer in layers:
        entries = aggregate[str(layer)]["top_abs_diff_features"] if str(layer) in aggregate else aggregate[layer]["top_abs_diff_features"]
        for item in entries[:top_n]:
            row_labels.append(f"L{layer}:F{item['feature']}")
            row = np.zeros(len(layers))
            row[layers.index(layer)] = item["mean_diff"]
            rows.append(row)
    if not rows:
        return np.zeros((1, len(layers))), ["none"], layers
    return np.vstack(rows), row_labels, layers


def plot_feature_differential_heatmap(results: dict, output_dir: Path) -> None:
    mat, row_labels, layers = _aggregate_matrix(results, top_n=10)
    plt.figure(figsize=(12, max(6, len(row_labels) * 0.2)))
    sns.heatmap(mat, cmap="coolwarm", center=0, yticklabels=row_labels, xticklabels=layers)
    plt.title("Feature differential heatmap (truth - secret)")
    plt.xlabel("Layer")
    plt.ylabel("Top differential features")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_differential_heatmap.png", dpi=180)
    plt.close()


def _compute_feature_trajectories(results: dict, sign: str = "positive", top_n: int = 5) -> Dict[int, Dict[int, Tuple[float, float]]]:
    layers = results["available_layers"]
    pair_results = results["pair_results"]

    global_scores = Counter()
    for layer in layers:
        layer_key = str(layer)
        entries = results["aggregate"][layer_key]["top_abs_diff_features"] if layer_key in results["aggregate"] else results["aggregate"][layer]["top_abs_diff_features"]
        for item in entries:
            diff = item["mean_diff"]
            if (sign == "positive" and diff > 0) or (sign == "negative" and diff < 0):
                global_scores[(layer, item["feature"])] = abs(diff)

    selected = [x for x, _ in global_scores.most_common(top_n)]
    traj: Dict[int, Dict[int, Tuple[float, float]]] = defaultdict(dict)

    for layer, feat in selected:
        t_vals, s_vals = [], []
        for pair in pair_results:
            t = pair["truth_features_topk"].get(str(layer), pair["truth_features_topk"].get(layer, {}))
            s = pair["secret_features_topk"].get(str(layer), pair["secret_features_topk"].get(layer, {}))
            t_vals.append(float(t.get(str(feat), t.get(feat, 0.0))))
            s_vals.append(float(s.get(str(feat), s.get(feat, 0.0))))
        traj[layer][feat] = (float(np.mean(t_vals)), float(np.mean(s_vals)))

    return traj


def _plot_trajectory(results: dict, output_dir: Path, sign: str, filename: str, title: str) -> None:
    layers = results["available_layers"]
    traj = _compute_feature_trajectories(results, sign=sign, top_n=5)
    plt.figure(figsize=(10, 6))

    for layer, feature_map in traj.items():
        for feat, _ in feature_map.items():
            truth_line = []
            secret_line = []
            for l in layers:
                t, s = traj.get(l, {}).get(feat, (0.0, 0.0))
                truth_line.append(t)
                secret_line.append(s)
            plt.plot(layers, truth_line, linestyle="-", label=f"F{feat} truth")
            plt.plot(layers, secret_line, linestyle="--", label=f"F{feat} secret")

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Mean activation")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
    plt.close()


def plot_per_pair_summary(results: dict, output_dir: Path) -> None:
    pair_rows = []
    for pair in results["pair_results"]:
        max_layer = None
        max_gap = -1.0
        top_feature = None
        for layer, details in pair["per_layer_differentials"].items():
            feats = details["top_truth_features"]
            if feats and feats[0][1] > max_gap:
                max_gap = feats[0][1]
                max_layer = layer
                top_feature = feats[0][0]
        pair_rows.append((pair["pair_id"], int(max_layer) if max_layer is not None else -1, float(max_gap), top_feature))

    labels = [x[0] for x in pair_rows]
    vals = [x[2] for x in pair_rows]
    plt.figure(figsize=(12, 5))
    sns.barplot(x=labels, y=vals, color="steelblue")
    plt.title("Per-pair max truth-feature suppression gap")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Max positive differential")
    plt.tight_layout()
    plt.savefig(output_dir / "per_pair_summary.png", dpi=180)
    plt.close()


def plot_feature_cooccurrence(results: dict, output_dir: Path) -> None:
    layers = results["available_layers"]
    pair_results = results["pair_results"]
    feature_counts = Counter()

    for pair in pair_results:
        seen = set()
        for layer in layers:
            diff_info = pair["per_layer_differentials"].get(str(layer), pair["per_layer_differentials"].get(layer, {}))
            for feat, _ in diff_info.get("top_truth_features", [])[:5]:
                seen.add((int(layer), int(feat)))
        for item in seen:
            feature_counts[item] += 1

    top = feature_counts.most_common(20)
    labels = [f"L{l}:F{f}" for (l, f), _ in top]
    vals = [count for _, count in top]

    plt.figure(figsize=(12, 5))
    sns.barplot(x=labels, y=vals, color="darkorange")
    plt.title("Feature co-occurrence across prompt pairs")
    plt.ylabel("Pair count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_cooccurrence.png", dpi=180)
    plt.close()


def create_all_plots(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_feature_differential_heatmap(results, output_dir)
    _plot_trajectory(
        results,
        output_dir,
        sign="positive",
        filename="truth_feature_trajectory.png",
        title="Truth feature trajectory (top +diff features)",
    )
    _plot_trajectory(
        results,
        output_dir,
        sign="negative",
        filename="suppression_trigger_trajectory.png",
        title="Suppression-trigger feature trajectory (top -diff features)",
    )
    plot_per_pair_summary(results, output_dir)
    plot_feature_cooccurrence(results, output_dir)
