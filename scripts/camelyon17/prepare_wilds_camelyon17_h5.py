"""Convert WILDS CAMELYON17 patches (optionally from manifest subset) to H5 splits.

This lets existing H5-based training scripts run without major refactors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image

PATCH_SIZE = 96
CHUNK_SIZE = 2048


def _read_patch_u8(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    if arr.shape != (PATCH_SIZE, PATCH_SIZE, 3):
        raise ValueError(f"Unexpected patch shape {arr.shape} at {path}")
    return arr


def _row_to_patch_path(dataset_dir: Path, row: pd.Series) -> Path:
    patient = str(row["patient"])
    node = int(row["node"])
    x = int(row["x_coord"])
    y = int(row["y_coord"])
    return (
        dataset_dir
        / "patches"
        / f"patient_{patient}_node_{node}"
        / f"patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png"
    )


def _load_source_df(dataset_dir: Path, manifest_csv: str | None) -> pd.DataFrame:
    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.csv: {meta_path}")
    full = pd.read_csv(meta_path, index_col=0, dtype={"patient": "str"})
    if manifest_csv is None:
        df = full.copy()
    else:
        mpath = Path(manifest_csv).resolve()
        if not mpath.is_file():
            raise FileNotFoundError(f"Missing manifest csv: {mpath}")
        subset = pd.read_csv(mpath, index_col=0, dtype={"patient": "str"})
        needed = {"tumor", "split_name", "patient", "node", "x_coord", "y_coord"}
        miss = needed - set(subset.columns)
        if miss:
            raise ValueError(f"Manifest missing columns: {sorted(miss)}")
        df = subset.copy()
    return df


def _split_indices(df: pd.DataFrame, valid_source: str) -> Dict[str, np.ndarray]:
    # valid_source: "id_val" (in-domain validation) or "val" (OOD validation)
    allowed = {"id_val", "val"}
    if valid_source not in allowed:
        raise ValueError(f"--valid-source must be one of {sorted(allowed)}")

    split_col = "split_name"
    if split_col not in df.columns:
        # Reconstruct WILDS official split behavior from raw metadata.csv:
        # 1) decode existing split IDs
        # 2) override by center (center==1 -> val, center==2 -> test)
        split_map = {0: "train", 1: "id_val", 2: "test", 3: "val"}
        if "split" not in df.columns:
            raise ValueError("Need either split_name or split column in source dataframe.")
        df[split_col] = df["split"].map(split_map)
        if "center" in df.columns:
            center = df["center"].astype(int)
            df.loc[center == 1, split_col] = "val"
            df.loc[center == 2, split_col] = "test"

    train_idx = np.flatnonzero(df[split_col].to_numpy() == "train")
    valid_idx = np.flatnonzero(df[split_col].to_numpy() == valid_source)
    test_idx = np.flatnonzero(df[split_col].to_numpy() == "test")
    ood_val_idx = np.flatnonzero(df[split_col].to_numpy() == "val")
    id_val_idx = np.flatnonzero(df[split_col].to_numpy() == "id_val")

    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        counts = df[split_col].value_counts(dropna=False).to_dict()
        raise ValueError(f"One of train/valid/test splits is empty. split_name counts: {counts}")

    return {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
        "id_val": id_val_idx,
        "ood_val": ood_val_idx,
    }


def _write_h5_split(dataset_dir: Path, df: pd.DataFrame, indices: np.ndarray, split: str, out_dir: Path) -> Dict[str, float]:
    x_path = out_dir / f"{split}_x.h5"
    y_path = out_dir / f"{split}_y.h5"
    n = int(len(indices))
    with h5py.File(x_path, "w") as fx, h5py.File(y_path, "w") as fy:
        xds = fx.create_dataset(
            "x",
            shape=(n, PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.float32,
            chunks=(1, PATCH_SIZE, PATCH_SIZE, 3),
            compression="gzip",
        )
        yds = fy.create_dataset("y", shape=(n,), dtype=np.float32, compression="gzip")

        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            bidx = indices[start:end]
            bx = np.zeros((end - start, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
            by = np.zeros((end - start,), dtype=np.float32)
            for i, ridx in enumerate(bidx):
                row = df.iloc[int(ridx)]
                p = _row_to_patch_path(dataset_dir, row)
                bx[i] = _read_patch_u8(p).astype(np.float32) / 255.0
                by[i] = float(int(row["tumor"]))
            xds[start:end] = bx
            yds[start:end] = by

    y = df.iloc[indices]["tumor"].to_numpy().astype(np.float32)
    return {
        "n_samples": n,
        "n_positive": int((y >= 0.5).sum()),
        "n_negative": int((y < 0.5).sum()),
        "frac_positive": float((y >= 0.5).mean()) if n > 0 else 0.0,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert WILDS CAMELYON17 to H5 splits.")
    p.add_argument("--dataset-dir", type=str, default="data/wilds/camelyon17_v1.0")
    p.add_argument("--manifest-csv", type=str, default=None, help="Optional subset manifest csv (e.g., pilot_subset_manifest.csv).")
    p.add_argument("--out-dir", type=str, default="data/wilds/camelyon17_h5_pilot")
    p.add_argument("--valid-source", type=str, default="id_val", choices=["id_val", "val"])
    p.add_argument("--write-aux-splits", action="store_true", help="Also write id_val and ood_val H5 pairs.")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_source_df(dataset_dir, args.manifest_csv)
    split_idx = _split_indices(df, valid_source=args.valid_source)

    stats: Dict[str, dict] = {}
    for split in ("train", "valid", "test"):
        print(f"Writing {split} ({len(split_idx[split])} samples)")
        stats[split] = _write_h5_split(dataset_dir, df, split_idx[split], split, out_dir)

    if args.write_aux_splits:
        for split in ("id_val", "ood_val"):
            if len(split_idx[split]) > 0:
                print(f"Writing {split} ({len(split_idx[split])} samples)")
                stats[split] = _write_h5_split(dataset_dir, df, split_idx[split], split, out_dir)

    manifest = {
        "version": 1,
        "dataset_dir": str(dataset_dir),
        "manifest_csv": str(Path(args.manifest_csv).resolve()) if args.manifest_csv else None,
        "out_dir": str(out_dir),
        "valid_source": args.valid_source,
        "stats": stats,
    }
    _atomic_json(out_dir / "manifest.json", manifest)
    print("Done. Wrote", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
