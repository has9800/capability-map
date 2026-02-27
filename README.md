# Transformer Capability Map

A modular pipeline for mapping which GPT attention heads are important for reasoning, factual recall, syntax processing, and honesty.

## Pipeline

1. **Activation patching** (`scripts/run_patch.py`)
2. **Progressive pruning** (`scripts/run_prune.py`)
3. **Cross-capability overlap analysis** (`scripts/run_overlap.py`)

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_patch.py --config config.yaml --debug
python scripts/run_prune.py --config config.yaml --debug
python scripts/run_overlap.py --config config.yaml
```

## Data

Create these JSON files in `data/`:

- `reasoning.json`
- `recall.json`
- `syntax.json`
- `honesty.json`

Each file uses this format:

```json
[
  {
    "prompt": "If all dogs are animals and all animals are mortal, are dogs mortal? Answer:",
    "correct_answer": "Yes",
    "incorrect_answer": "No"
  }
]
```

A test fixture is available at `data/sample.json`.
