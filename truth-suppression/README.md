# Truth Suppression Detection in Pythia-1.4B

This project probes where truth representations are suppressed in `EleutherAI/pythia-1.4b` when prompts are framed as secrets.

## Structure

```
truth-suppression/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── logit_lens.py
│   ├── experiment.py
│   └── visualize.py
├── data/
├── results/
└── notebooks/
    └── analysis.ipynb
```

## Method

For each contrastive pair (truth prompt vs secret prompt), the experiment:

1. Runs the model with `output_hidden_states=True`.
2. For every layer hidden state at the last token position, applies:
   - `model.gpt_neox.final_layer_norm`
   - `model.embed_out`
3. Computes:
   - Probability of the true answer token
   - Rank of the true answer token
   - Top-5 predicted tokens
4. Compares curves between truth and secret conditions.

Suppression signal per layer is:

`P(true token | truth condition) - P(true token | secret condition)`

## Run

```bash
cd truth-suppression
python -m pip install -r requirements.txt
python src/experiment.py
```

Artifacts are written to `results/`:

- `truth_suppression_results.json`
- `suppression_heatmap.png`
- `average_suppression_curve.png`
- `plots/pair_<topic>.png`

## Notes

- Uses CUDA automatically when available, otherwise CPU.
- Tracks the first token of each correct answer using the tokenizer at runtime.
- Includes 20 contrastive pairs across geography, science, math, and history.
