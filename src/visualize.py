from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import CAPABILITIES


def save_patch_heatmaps(patch_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = make_subplots(rows=1, cols=4, subplot_titles=list(CAPABILITIES))
    for i, cap in enumerate(CAPABILITIES, start=1):
        eff = np.load(patch_dir / f"{cap}_effects.npy")
        fig.add_trace(
            go.Heatmap(z=eff, colorscale="RdBu", zmid=0, colorbar=dict(title="Effect")),
            row=1,
            col=i,
        )
    fig.update_layout(title="Per-capability head effects")
    fig.write_html(output_dir / "patch_heatmaps.html")


def save_pruning_curves(prune_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    for cap in CAPABILITIES:
        curves = json.loads((prune_dir / f"{cap}_curves.json").read_text())
        spec = curves["capability_specific"]
        x = [p["n_remaining"] for p in spec]
        y = [p["accuracy"] for p in spec]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=f"{cap} specific"))

        random_curves: List[List[Dict[str, float]]] = curves["random_baselines"]
        min_len = min(len(c) for c in random_curves)
        if min_len > 0:
            x_rand = [random_curves[0][i]["n_remaining"] for i in range(min_len)]
            arr = np.array([[c[i]["accuracy"] for i in range(min_len)] for c in random_curves])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            fig.add_trace(go.Scatter(x=x_rand, y=mean, mode="lines", name=f"{cap} random mean", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=x_rand, y=mean + std, mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=x_rand,
                    y=mean - std,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    name=f"{cap} random ± std",
                )
            )
    fig.update_layout(title="Progressive pruning curves", xaxis_title="Heads remaining", yaxis_title="Accuracy")
    fig.write_html(output_dir / "pruning_curves.html")


def save_overlap_plots(overlap_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = json.loads((overlap_dir / "head_labels.json").read_text())
    max_layer = max(item["layer"] for item in labels)
    max_head = max(item["head"] for item in labels)
    categories = sorted({item["label"] for item in labels})
    cat_to_id = {cat: i for i, cat in enumerate(categories)}

    z = np.zeros((max_layer + 1, max_head + 1), dtype=int)
    for item in labels:
        z[item["layer"], item["head"]] = cat_to_id[item["label"]]

    heatmap = go.Figure(
        data=go.Heatmap(z=z, colorscale="Viridis", colorbar=dict(title="Label ID"))
    )
    heatmap.update_layout(title="Head label map")
    heatmap.write_html(output_dir / "overlap_labels.html")

    counts = {cat: 0 for cat in categories}
    for item in labels:
        counts[item["label"]] += 1
    bar = go.Figure(data=go.Bar(x=list(counts.keys()), y=list(counts.values())))
    bar.update_layout(title="Head count per label", xaxis_title="Label", yaxis_title="Count")
    bar.write_html(output_dir / "overlap_counts.html")
