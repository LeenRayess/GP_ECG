"""Remove CAM17 patches that duplicate PCam patches (pixel hash exact match).

Inputs:
- CAM17 H5 splits produced by prepare_cam17_patches_to_h5.py
- PCam raw dataset via keras_pcam loader (pcam-master)

Outputs:
- <out-dir>/cam17_kept_indices_{train,valid,test}.npy
- <out-dir>/manifest.json
- Optional filtered CAM17 H5 copies via --write-filtered-h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Set

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PCAM_MASTER = PROJECT_ROOT / "pcam-master"
if PCAM_MASTER.is_dir():
    sys.path.insert(0, str(PCAM_MASTER))

try:
    from keras_pcam.dataset.pcam import load_data
except ImportError:
    load_data = None

CHUNK_SIZE = 2000


def _atomic_json_dump(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _sha256_u8_rgb(arr: np.ndarray) -> str:
    if arr.ndim != 3:
        raise ValueError(f"Expected image tensor HxWxC, got shape {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.asarray(np.clip(arr, 0, 255), dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _collect_pcam_hashes(data_dir: Path, chunk_size: int) -> Set[str]:
    if load_data is None:
        raise ImportError("keras_pcam loader unavailable. Ensure pcam-master is present in project root.")
    (train_x, _, _), (valid_x, _, _), (test_x, _, _) = load_data(data_dir=str(data_dir))
    hset: Set[str] = set()
    for split_name, x_ds in (("train", train_x), ("valid", valid_x), ("test", test_x)):
        n = len(x_ds)
        print(f"Hashing PCam {split_name}: {n} patches")
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch = np.asarray(x_ds[start:end])
            for i in range(batch.shape[0]):
                hset.add(_sha256_u8_rgb(batch[i]))
    return hset


def _cam17_h5_paths(cam17_h5_dir: Path, split: str) -> tuple[Path, Path]:
    return cam17_h5_dir / f"{split}_x.h5", cam17_h5_dir / f"{split}_y.h5"


def _dedup_split_against_hashes(x_path: Path, pcam_hashes: Set[str], chunk_size: int) -> np.ndarray:
    with h5py.File(x_path, "r") as f:
        x = f["x"]
        n = x.shape[0]
        keep = np.ones((n,), dtype=bool)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch = np.asarray(x[start:end])
            for i in range(batch.shape[0]):
                if _sha256_u8_rgb(batch[i]) in pcam_hashes:
                    keep[start + i] = False
    return np.flatnonzero(keep).astype(np.int64)


def _write_filtered_h5(src_x: Path, src_y: Path, kept_idx: np.ndarray, out_x: Path, out_y: Path) -> None:
    with h5py.File(src_x, "r") as fx_in, h5py.File(src_y, "r") as fy_in:
        x_in = fx_in["x"]
        y_in = fy_in["y"]
        n = int(len(kept_idx))
        with h5py.File(out_x, "w") as fx_out, h5py.File(out_y, "w") as fy_out:
            x_out = fx_out.create_dataset(
                "x",
                shape=(n,) + x_in.shape[1:],
                dtype=x_in.dtype,
                chunks=(1,) + x_in.shape[1:],
                compression="gzip",
            )
            y_out = fy_out.create_dataset("y", shape=(n,), dtype=np.float32, compression="gzip")
            for start in range(0, n, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, n)
                idx = kept_idx[start:end]
                x_out[start:end] = np.asarray(x_in[idx])
                y_out[start:end] = np.asarray(y_in[idx]).reshape(-1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate CAM17 patches against PCam by SHA-256 hash.")
    parser.add_argument(
        "--cam17-h5-dir",
        type=str,
        default="data/cam17_preprocessed_raw_h5",
        help="Directory containing CAM17 train/valid/test *_x.h5 and *_y.h5.",
    )
    parser.add_argument(
        "--pcam-data-dir",
        type=str,
        default="pcam_data",
        help="PCam data root passed to keras_pcam load_data().",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/cam17_dedup_vs_pcam",
        help="Output directory for kept indices and manifest.",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size for hashing.")
    parser.add_argument(
        "--write-filtered-h5",
        action="store_true",
        help="Also write filtered CAM17 H5 files under <out-dir>/filtered_h5.",
    )
    args = parser.parse_args()

    cam17_h5_dir = Path(args.cam17_h5_dir).resolve()
    pcam_data_dir = Path(args.pcam_data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        x_path, y_path = _cam17_h5_paths(cam17_h5_dir, split)
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(f"Missing CAM17 H5 for split '{split}' in {cam17_h5_dir}")

    print("Collecting PCam hashes...")
    pcam_hashes = _collect_pcam_hashes(pcam_data_dir, chunk_size=int(args.chunk_size))
    print("PCam unique hashes:", len(pcam_hashes))

    summary: Dict[str, Dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        x_path, y_path = _cam17_h5_paths(cam17_h5_dir, split)
        with h5py.File(x_path, "r") as f:
            n_orig = int(f["x"].shape[0])
        kept_idx = _dedup_split_against_hashes(x_path, pcam_hashes, chunk_size=int(args.chunk_size))
        np.save(out_dir / f"cam17_kept_indices_{split}.npy", kept_idx)
        n_kept = int(len(kept_idx))
        n_removed = int(n_orig - n_kept)
        summary[split] = {"n_original": n_orig, "n_kept": n_kept, "n_removed_vs_pcam": n_removed}
        print(f"{split}: original={n_orig} kept={n_kept} removed_vs_pcam={n_removed}")

        if args.write_filtered_h5:
            filtered = out_dir / "filtered_h5"
            filtered.mkdir(parents=True, exist_ok=True)
            _write_filtered_h5(
                x_path,
                y_path,
                kept_idx,
                filtered / f"{split}_x.h5",
                filtered / f"{split}_y.h5",
            )

    manifest = {
        "version": 1,
        "cam17_h5_dir": str(cam17_h5_dir),
        "pcam_data_dir": str(pcam_data_dir),
        "pcam_unique_hashes": int(len(pcam_hashes)),
        "summary": summary,
        "indices_files": {s: f"cam17_kept_indices_{s}.npy" for s in ("train", "valid", "test")},
        "write_filtered_h5": bool(args.write_filtered_h5),
    }
    _atomic_json_dump(out_dir / "manifest.json", manifest)
    print("Done. Wrote", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
