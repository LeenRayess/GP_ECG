# CAMELYON17 Patch Dataset Workflow

This project currently has CAM17 in **patch form** (already extracted), not WSI/XML form:

- `cam17_original/data/camelyon17_v1.0/metadata.csv`
- `cam17_original/data/camelyon17_v1.0/patches/patient_XXX_node_Y/*.png`

## 1) Export CAM17 patches to PCam-style H5

From repo root:

```bash
python scripts/camelyon17/prepare_cam17_patches_to_h5.py \
  --dataset-dir "cam17_original/data/camelyon17_v1.0" \
  --out-dir "data/cam17_preprocessed_raw_h5" \
  --test-fraction-from-train 0.1 \
  --seed 42
```

Outputs:

- `data/cam17_preprocessed_raw_h5/train_x.h5`, `train_y.h5`
- `data/cam17_preprocessed_raw_h5/valid_x.h5`, `valid_y.h5`
- `data/cam17_preprocessed_raw_h5/test_x.h5`, `test_y.h5`
- `data/cam17_preprocessed_raw_h5/manifest.json`

Split policy used by exporter:

- `metadata split == 1` -> validation
- `metadata split == 0` -> train/test pool
- test built by holding out a fraction of `patient_node` groups from split 0 (prevents leakage)

## 2) Remove overlaps between CAM17 and PCam

Run exact dedup by SHA-256 patch hash against PCam:

```bash
python scripts/camelyon17/dedup_cam17_vs_pcam.py \
  --cam17-h5-dir "data/cam17_preprocessed_raw_h5" \
  --pcam-data-dir "pcam_data" \
  --out-dir "data/cam17_dedup_vs_pcam" \
  --write-filtered-h5
```

Outputs:

- `data/cam17_dedup_vs_pcam/cam17_kept_indices_train.npy`
- `data/cam17_dedup_vs_pcam/cam17_kept_indices_valid.npy`
- `data/cam17_dedup_vs_pcam/cam17_kept_indices_test.npy`
- `data/cam17_dedup_vs_pcam/manifest.json`
- optional filtered H5 in `data/cam17_dedup_vs_pcam/filtered_h5/`

## 3) Use filtered H5 for downstream preprocessing/training

When `--write-filtered-h5` is used, point later scripts to:

- `data/cam17_dedup_vs_pcam/filtered_h5`

This ensures CAM17 samples overlapping with PCam are excluded.
