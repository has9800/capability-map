"""Main runner for truth suppression detection with a logit-lens analysis."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from dataset import attach_token_ids, get_contrastive_pairs
from logit_lens import analyze_prompt_by_layer
from visualize import save_average_suppression_curve, save_heatmap, save_pair_plot


MODEL_NAME = "EleutherAI/pythia-1.4b"


def _ensure_dirs(root: str) -> dict:
    paths = {
        "root": root,
        "results": os.path.join(root, "results"),
        "plots": os.path.join(root, "results", "plots"),
        "data": os.path.join(root, "data"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def run_experiment(project_root: str, top_k: int = 5) -> dict:
    paths = _ensure_dirs(project_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print(f"Using device: {device}")
    pairs = attach_token_ids(get_contrastive_pairs(), tokenizer)

    print(f"Running {len(pairs)} contrastive pairs over {model.config.num_hidden_layers} layers...")
    all_results = []

    for pair in tqdm(pairs, desc="Pairs"):
        truth_run = analyze_prompt_by_layer(
            model=model,
            tokenizer=tokenizer,
            prompt=pair.truth_prompt,
            true_token_id=pair.track_token_id,
            top_k=top_k,
            device=device,
        )
        secret_run = analyze_prompt_by_layer(
            model=model,
            tokenizer=tokenizer,
            prompt=pair.secret_prompt,
            true_token_id=pair.track_token_id,
            top_k=top_k,
            device=device,
        )

        plot_path = save_pair_plot(pair.topic, truth_run, secret_run, paths["plots"])

        truth_probs = [x["truth_prob"] for x in truth_run["layers"]]
        secret_probs = [x["truth_prob"] for x in secret_run["layers"]]
        suppression = [t - s for t, s in zip(truth_probs, secret_probs)]
        max_layer = int(max(range(len(suppression)), key=lambda i: suppression[i]))

        pair_result = {
            "topic": pair.topic,
            "true_answer": pair.true_answer,
            "track_token": pair.track_token,
            "track_token_id": pair.track_token_id,
            "truth_condition": truth_run,
            "secret_condition": secret_run,
            "max_suppression_layer": max_layer,
            "max_suppression_value": suppression[max_layer],
            "pair_plot": plot_path,
            "summary": {
                "truth_prob_at_max": truth_probs[max_layer],
                "secret_prob_at_max": secret_probs[max_layer],
                "truth_final_pred": truth_run["final_pred_token"],
                "secret_final_pred": secret_run["final_pred_token"],
            },
        }
        all_results.append(pair_result)

        print(
            f"[{pair.topic}] max suppression layer={max_layer}, "
            f"truth={truth_probs[max_layer]:.5f}, secret={secret_probs[max_layer]:.5f}, "
            f"final tokens: truth='{truth_run['final_pred_token']}', secret='{secret_run['final_pred_token']}'"
        )

    heatmap_path = save_heatmap(all_results, os.path.join(paths["results"], "suppression_heatmap.png"))
    avg_curve_stats = save_average_suppression_curve(
        all_results, os.path.join(paths["results"], "average_suppression_curve.png")
    )

    output = {
        "model": MODEL_NAME,
        "device": device,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "num_pairs": len(all_results),
        "num_layers": model.config.num_hidden_layers,
        "results": all_results,
        "aggregate": {
            "heatmap_path": heatmap_path,
            "average_curve": avg_curve_stats,
        },
    }

    out_json = os.path.join(paths["results"], "truth_suppression_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== Summary Stats ===")
    for item in all_results:
        summary = item["summary"]
        print(
            f"- {item['topic']}: max suppression at layer {item['max_suppression_layer']} "
            f"(truth={summary['truth_prob_at_max']:.5f}, secret={summary['secret_prob_at_max']:.5f}) | "
            f"final truth token='{summary['truth_final_pred']}' | "
            f"final secret token='{summary['secret_final_pred']}'"
        )

    print(f"\nSaved JSON results: {out_json}")
    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved average suppression curve: {avg_curve_stats['path']}")

    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Truth suppression detection in Pythia-1.4B")
    parser.add_argument(
        "--project-root",
        default=os.path.dirname(os.path.dirname(__file__)),
        help="Root of the truth-suppression project.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k tokens to store per layer.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(project_root=args.project_root, top_k=args.top_k)
