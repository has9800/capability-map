"""Check which Pythia-410M SAE layers are available in Sparsify."""

from __future__ import annotations

import argparse
import json

from sae_analysis import MODEL_NAME, NUM_LAYERS, SAE_REPO, discover_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-layers", type=int, default=NUM_LAYERS)
    args = parser.parse_args()

    layers = discover_layers(sae_repo=SAE_REPO, max_layers=args.max_layers, device=args.device)
    report = {
        "sae_repo": SAE_REPO,
        "model_name": MODEL_NAME,
        "max_layers_checked": args.max_layers,
        "available_layers": layers,
        "num_available_layers": len(layers),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
