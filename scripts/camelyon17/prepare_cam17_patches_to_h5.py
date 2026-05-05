"""Export CAMELYON17 v1.0 patch dataset to PCam-style H5 splits.

Expected dataset layout (already present in this repo):
  cam17_original/data/camelyon17_v1.0/
    metadata.csv
    patches/patient_XXX_node_Y/patch_patient_XXX_node_Y_x_<x>_y_<y>.png

This exporter adapts to the downloaded patch dataset (no WSI/XML required).
It writes:
  <out-dir>/train_x.h5, train_y.h5
  <out-dir>/valid_x.h5, valid_y.h5
  <out-dir>/test_x.h5,  test_y.h5
  <out-dir>/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image

CHUNK_SIZE = 2048
PATCH_SIZE = 96


def _atomic_json_dump(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _normalize_patient_node(v) -> str:
    try:
        return str(int(v))
    except Exception:
        s = str(v).strip()
        return str(int(s)) if s.isdigit() else s


def _patch_path(dataset_dir: Path, patient: str, node: str, x_coord: int, y_coord: int) -> Path:
    p3 = f"{int(patient):03d}" if patient.isdigit() else patient
    node_s = str(int(node)) if str(node).isdigit() else str(node)
    return (
        dataset_dir
        / "patches"
        / f"patient_{p3}_node_{node_s}"
        / f"patch_patient_{p3}_node_{node_s}_x_{int(x_coord)}_y_{int(y_coord)}.png"
    )


def _read_patch_rgb_u8(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    if arr.shape != (PATCH_SIZE, PATCH_SIZE, 3):
        raise ValueError(f"Unexpected patch shape {arr.shape} at {path}")
    return arr


def _load_metadata(dataset_dir: Path) -> pd.DataFrame:
    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.csv at {meta_path}")
    df = pd.read_csv(meta_path)
    required = {"patient", "node", "x_coord", "y_coord", "tumor", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {sorted(missing)}")

    out = df.copy()
    out["patient"] = out["patient"].map(_normalize_patient_node)
    out["node"] = out["node"].map(_normalize_patient_node)
    out["x_coord"] = out["x_coord"].astype(int)
    out["y_coord"] = out["y_coord"].astype(int)
    out["tumor"] = out["tumor"].astype(int)
    out["split"] = out["split"].astype(int)
    out["patient_node"] = out["patient"] + "_" + out["node"]
    return out


def _assign_splits(
    df: pd.DataFrame,
    *,
    test_fraction_from_train: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    if not (0.0 < test_fraction_from_train < 1.0):
        raise ValueError("--test-fraction-from-train must be in (0,1)")

    # Existing CAM17 metadata split: 0/1
    # Keep split==1 as validation.
    # Build test from split==0 at patient_node granularity to avoid leakage.
    is_valid = df["split"].to_numpy() == 1
    train_pool_idx = np.flatnonzero(~is_valid)
    valid_idx = np.flatnonzero(is_valid)

    pool = df.iloc[train_pool_idx]
    groups = sorted(pool["patient_node"].unique().tolist())
    if not groups:
        raise ValueError("No split==0 samples available for train/test construction.")

    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    n_test_groups = max(1, int(round(len(groups) * test_fraction_from_train)))
    test_groups = set(groups[:n_test_groups])

    in_test = pool["patient_node"].isin(test_groups).to_numpy()
    test_idx = train_pool_idx[np.flatnonzero(in_test)]
    train_idx = train_pool_idx[np.flatnonzero(~in_test)]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Train/test split is empty; adjust --test-fraction-from-train.")

    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def _write_split_h5(
    dataset_dir: Path,
    df: pd.DataFrame,
    indices: np.ndarray,
    split_name: str,
    out_dir: Path,
) -> Dict[str, float]:
    out_x = out_dir / f"{split_name}_x.h5"
    out_y = out_dir / f"{split_name}_y.h5"

    n = int(len(indices))
    with h5py.File(out_x, "w") as fx, h5py.File(out_y, "w") as fy:
        xds = fx.create_dataset(
            "x",
            shape=(n, PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
            chunks=(1, PATCH_SIZE, PATCH_SIZE, 3),
            compression="gzip",
        )
        yds = fy.create_dataset("y", shape=(n,), dtype=np.float32, compression="gzip")

        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            batch_idx = indices[start:end]
            batch_x = np.zeros((end - start, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            batch_y = np.zeros((end - start,), dtype=np.float32)
            for i, ridx in enumerate(batch_idx):
                r = df.iloc[int(ridx)]
                p = _patch_path(dataset_dir, r["patient"], r["node"], int(r["x_coord"]), int(r["y_coord"]))
                batch_x[i] = _read_patch_rgb_u8(p)
                batch_y[i] = float(r["tumor"])
            xds[start:end] = batch_x
            yds[start:end] = batch_y

    y = np.asarray(df.iloc[indices]["tumor"].to_numpy(), dtype=np.float32)
    return {
        "n_samples": int(n),
        "n_positive": int((y >= 0.5).sum()),
        "n_negative": int((y < 0.5).sum()),
        "frac_positive": float((y >= 0.5).mean()) if n > 0 else 0.0,
    }


def _verify_paths_exist(dataset_dir: Path, df: pd.DataFrame, sample_n: int = 2000) -> Tuple[int, int]:
    n = len(df)
    if n == 0:
        return 0, 0
    rng = np.random.RandomState(0)
    take = min(sample_n, n)
    ids = rng.choice(n, size=take, replace=False)
    miss = 0
    for ridx in ids:
        r = df.iloc[int(ridx)]
        p = _patch_path(dataset_dir, r["patient"], r["node"], int(r["x_coord"]), int(r["y_coord"]))
        if not p.is_file():
            miss += 1
    return take, miss


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CAM17 patch dataset into PCam-style train/valid/test H5.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="cam17_original/data/camelyon17_v1.0",
        help="Folder containing metadata.csv and patches/ (default matches current repo layout).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/cam17_preprocessed_raw_h5",
        help="Output folder for train/valid/test H5 and manifest.json.",
    )
    parser.add_argument(
        "--test-fraction-from-train",
        type=float,
        default=0.1,
        help="Fraction of split==0 patient_node groups held out as test.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test group split.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metadata from:", dataset_dir / "metadata.csv")
    df = _load_metadata(dataset_dir)
    sampled, missing = _verify_paths_exist(dataset_dir, df, sample_n=2000)
    print(f"Sample path check: missing {missing}/{sampled}")
    if missing > 0:
        raise FileNotFoundError("Some metadata rows do not map to patch files; inspect dataset extraction.")

    split_idx = _assign_splits(
        df,
        test_fraction_from_train=float(args.test_fraction_from_train),
        seed=int(args.seed),
    )

    print("Writing H5 splits...")
    stats = {}
    for split in ("train", "valid", "test"):
        print(f"  -> {split} ({len(split_idx[split])} samples)")
        stats[split] = _write_split_h5(dataset_dir, df, split_idx[split], split, out_dir)

    manifest = {
        "version": 1,
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "source_metadata_rows": int(len(df)),
        "source_columns": list(df.columns),
        "split_policy": {
            "valid_from_metadata_split_eq_1": True,
            "test_from_metadata_split_eq_0_patient_node_groups_fraction": float(args.test_fraction_from_train),
            "seed": int(args.seed),
        },
        "stats": stats,
    }
    _atomic_json_dump(out_dir / "manifest.json", manifest)
    print("Done. Wrote", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
