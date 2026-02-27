"""Check which Pythia SAE releases/layers are available in SAELens."""

from __future__ import annotations

import argparse
import json

from sae_analysis import MODEL_NAME_BY_RELEASE, RELEASE_CANDIDATES, discover_layers, try_load_first_layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-layers", type=int, default=40)
    args = parser.parse_args()

    report = []
    for release in RELEASE_CANDIDATES:
        ok = try_load_first_layer(release=release, device=args.device)
        item = {
            "release": release,
            "model_name": MODEL_NAME_BY_RELEASE.get(release),
            "available": ok,
            "layers": [],
        }
        if ok:
            item["layers"] = discover_layers(release=release, max_layers=args.max_layers, device=args.device)
        report.append(item)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
