# GRPO (Experimental)

This folder contains the **experimental** GRPO training path for query expansion.
It is not part of the default production pipeline.

## Files

- `grpo.yaml` – experimental GRPO hyperparameters
- `grpo.py` – standalone GRPO training script

## Run

```bash
# Recommended default: run from repo root
cd /home/tobi/qmd
uv run finetune/experiments/grpo/grpo.py

# Or use unified entrypoint (deprecated in main pipeline):
uv run train.py grpo --config finetune/experiments/grpo/grpo.yaml
```

## Notes

- Current mainline focuses on SFT-only quality and benchmarks.
- Keep this workflow isolated unless you are explicitly experimenting with
  reinforcement-learning refinement.
