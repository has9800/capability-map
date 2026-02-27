"""Logit lens utilities for GPT-NeoX/Pythia models."""

from __future__ import annotations

from typing import Dict, List

import torch


@torch.no_grad()
def layer_logits_from_hidden(model, hidden_state: torch.Tensor) -> torch.Tensor:
    """Project one hidden vector to vocabulary logits using final layer norm + unembedding."""
    normed = model.gpt_neox.final_layer_norm(hidden_state)
    return model.embed_out(normed)


@torch.no_grad()
def analyze_prompt_by_layer(
    model,
    tokenizer,
    prompt: str,
    true_token_id: int,
    top_k: int = 5,
    device: str = "cpu",
) -> Dict:
    """Return truth-token probability/rank and top-k tokens for each layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)

    layer_stats: List[Dict] = []

    for layer_idx, hidden in enumerate(outputs.hidden_states):
        last_hidden = hidden[0, -1, :]
        logits = layer_logits_from_hidden(model, last_hidden)
        probs = torch.softmax(logits, dim=-1)

        true_prob = probs[true_token_id].item()
        sorted_indices = torch.argsort(logits, descending=True)
        true_rank = (sorted_indices == true_token_id).nonzero(as_tuple=False)[0].item() + 1

        top_vals, top_idx = torch.topk(probs, k=top_k)
        top_tokens = []
        for p, idx in zip(top_vals.tolist(), top_idx.tolist()):
            token = tokenizer.decode([idx]).replace("\n", "\\n")
            top_tokens.append({"token": token, "token_id": idx, "prob": p})

        layer_stats.append(
            {
                "layer": layer_idx,
                "truth_prob": true_prob,
                "truth_rank": int(true_rank),
                "top_tokens": top_tokens,
            }
        )

    final_logits = outputs.logits[0, -1, :]
    final_token_id = int(torch.argmax(final_logits).item())

    return {
        "prompt": prompt,
        "layers": layer_stats,
        "final_pred_token_id": final_token_id,
        "final_pred_token": tokenizer.decode([final_token_id]).replace("\n", "\\n"),
    }
