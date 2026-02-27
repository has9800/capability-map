from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer


@dataclass
class ModelBundle:
    model: HookedTransformer
    n_layers: int
    n_heads: int


def load_model(config: dict, debug: bool = False) -> ModelBundle:
    model_cfg = config["model"]
    model_name = model_cfg["debug_model"] if (debug or model_cfg.get("debug", False)) else model_cfg["name"]
    device = model_cfg.get("device", "cuda")
    dtype_name = model_cfg.get("dtype", "float32")
    dtype = getattr(torch, dtype_name)

    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"Loaded {model_name} with {n_layers} layers x {n_heads} heads = {n_layers * n_heads} total heads")
    return ModelBundle(model=model, n_layers=n_layers, n_heads=n_heads)
