# Virchow2 training on preprocessed PCam

## Model architecture (generalizable)

- **Backbone (frozen):** Virchow2, a ViT-based histology foundation model (timm / Hugging Face). It expects 224×224 RGB input and outputs a sequence of tokens. We take the **class token** and the **mean of patch tokens**, concatenated → **2560-dimensional** embedding. All backbone parameters are **frozen** (no gradients), so the representation stays the one learned by the foundation model on histology data. This supports generalization: the same features can be reused on other H&E datasets.

- **Head (trainable):** A single **Linear(2560, 1)** layer that maps the 2560-d embedding to one logit for binary classification (metastasis present vs absent). Only this head is trained on PCam. The loss is binary cross-entropy with logits (BCEWithLogitsLoss).

- **Why frozen:** Training only the head keeps the foundation features fixed and adapts only the decision layer to PCam. That reduces overfitting to PCam-specific nuisances and makes it easier to reuse the same backbone (and preprocessing pipeline) on other datasets (e.g. CAMELYON17) by training a new head or fine-tuning later.

## Data

- **Source:** Preprocessed PCam under `pcam_data/<subdir>/` (default `preprocessed/`): `train_x.h5`, `train_y.h5`, `valid_x.h5`, `valid_y.h5`. Patches are 96×96, float32, in [0, 1] (after quality filter, stain normalization, value normalization).
- **Multi-reference run:** If you used `scripts/preprocess_pcam_to_h5.py --stain-multi-ref --preprocessed-subdir preprocessed_multi_ref`, point training at that folder, e.g. `--preprocessed-dir pcam_data/preprocessed_multi_ref`.
- **At train time:** Each patch is resized to 224×224 with **bicubic** interpolation, then normalized with ImageNet mean/std (as required by Virchow2). No extra augmentation in the script.

## Checkpoint system

- After **every epoch** the script saves `experiments/virchow2_preprocessed/checkpoint.pt` with: `epoch`, `model_state_dict`, `optimizer_state_dict`, `best_val_acc`, and the last epoch’s train/val loss and accuracy.
- **Resume:** Run with `--resume` to load the latest `checkpoint.pt` and continue from the **next** epoch. No need to delete anything; the script overwrites `checkpoint.pt` each epoch.
- The best model by validation accuracy is saved separately as `model_best.pt` (weights only). Final metrics are written to `metrics.json`.

## How to run

```bash
# From project root, first run (10 epochs by default, same as baseline)
python scripts/train_virchow_preprocessed.py --preprocessed-dir pcam_data/preprocessed

# After a multi-reference preprocess (separate output folder)
python scripts/train_virchow_preprocessed.py --preprocessed-dir pcam_data/preprocessed_multi_ref

# Resume after a crash or stop
python scripts/train_virchow_preprocessed.py --preprocessed-dir pcam_data/preprocessed --resume
```

Optional: `--epochs`, `--batch-size`, `--out-dir`, `--lr` (head learning rate, default 1e-3).

Virchow2 is downloaded from Hugging Face (`paige-ai/Virchow2`). If the model is gated, set `HF_TOKEN` or run `huggingface-cli login` first.
