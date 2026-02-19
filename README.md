# gp_code3 — Predictive framework project

Project for developing the **predictive framework** (interpretability metrics → predict cross-dataset performance). ECG is the baseline modality; reference code is grouped in one folder; the rest is written from scratch.

## Layout

```
gp_code3/
├── ECG_repo/          # Reference: full ECG_repo (Leinonen et al.) — look here for pipeline, data prep, models
├── configs/           # Your configs (YAML, etc.)
├── data/              # Your data, splits, CSVs
├── docs/              # Notes and documentation
├── experiments/       # Model outputs, logs, predictions
├── notebooks/         # Your notebooks
├── src/               # Your code (dataloaders, models, training, metrics)
├── requirements.txt
└── README.md
```

- **ECG_repo/** — Copy of the ECG repository. Use it as reference only (data handling, training loop, evaluation). Do not depend on it from your code; reimplement what you need under `src/`, `notebooks/`, etc.
- **Everything else** — Empty placeholders. Add your own code, configs, and data here.

## Goals

1. Implement a working ECG baseline (or adapt ideas from `ECG_repo/`).
2. Add attention and compute interpretability metrics.
3. Run the predictive framework: combined training → metrics → predict cross-dataset performance.

## Getting started

1. Create a virtualenv and install deps as you add them to `requirements.txt`.
2. Put data under `data/` and add your pipelines in `src/`.
3. Use `ECG_repo/` only as reference (see `ECG_repo/README.md` for their pipeline).
