"""Run truth-suppression SAE experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from dataset import get_prompt_pairs
from sae_analysis import build_sae_context, run_dataset_analysis, save_results
from visualize import create_all_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--output", default="truth-suppression/results/sae_results.json")
    parser.add_argument("--plot-dir", default="truth-suppression/results")
    args = parser.parse_args()

    print("[1/4] Loading dataset...")
    pairs = get_prompt_pairs()
    print(f"Loaded {len(pairs)} prompt pairs.")

    print("[2/4] Resolving SAE release and loading SAEs...")
    context = build_sae_context(device=args.device, max_layers=args.max_layers)
    print(f"Using release: {context.release}")
    print(f"Using model: {context.model_name}")
    print(f"Layers covered: {context.available_layers}")

    print("[3/4] Running SAE feature extraction and pairwise contrasts...")
    results = run_dataset_analysis(pairs=pairs, context=context, device=args.device, top_k=args.top_k)
    results["run_metadata"] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_pairs": len(pairs),
    }

    output_path = Path(args.output)
    save_results(results, output_path)
    print(f"Saved results to {output_path}")

    print("[4/4] Generating visualizations...")
    create_all_plots(results=results, output_dir=Path(args.plot_dir))
    print("Done.")


if __name__ == "__main__":
    main()
