"""
PCam preprocessing pipeline: apply data-quality checks, deduplicate, and Macenko stain normalization.

Steps:
  1. Exclude anomaly indices (zero_std, all_black, all_white, high_black/white_ratio, low_tissue, low_blur, low_contrast).
  2. Exclude patches with tissue content below threshold (default 25%).
  3. Deduplicate: keep one representative per duplicate group.
  4. Fit Macenko normalizer on a reference patch; apply to all kept images when saving or on-the-fly.

Output:
  - Manifest JSON (indices to use per split, reference index, options).
  - Optional: preprocessed H5 files with stain-normalized patches (--save-h5).

Run from project root:
  python scripts/preprocess_pcam.py --data-dir pcam_data [--out-dir preprocessed_pcam] [--save-h5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Allow importing data_quality_report from same directory
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    import data_quality_report as dq
except ImportError:
    dq = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

# Preprocessing defaults (aligned with data quality checks)
TISSUE_PCT_THRESHOLD = 0.25   # exclude patches with < 25% tissue
BALANCE_TRAIN = True          # under-sample negatives to match positives
CHUNK_SIZE = 2000             # for re-scan when building kept indices (match dq report)


def _indices_to_keep_for_split(
    n_total: int,
    anomaly_indices: list[int],
    duplicate_groups: list[list[int]],
) -> list[int]:
    """Return list of indices to keep: exclude anomalies and all but one from each duplicate group."""
    exclude = set(anomaly_indices)
    for group in duplicate_groups:
        # Keep first index in group, exclude the rest
        for idx in group[1:]:
            exclude.add(idx)
    return [i for i in range(n_total) if i not in exclude]


def run_detection_and_compute_kept(
    data_dir: Path,
    tissue_pct_threshold: float = TISSUE_PCT_THRESHOLD,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[dict, dict]:
    """
    Load data, run anomaly/duplicate detection per split, compute kept indices.
    Returns (splits_data, kept_per_split).
    splits_data = {"train": (x_ds, y_ds, meta), ...}; kept_per_split = {"train": [...], ...}.
    """
    if dq is None or dq.pcam_load_data is None:
        raise ImportError("PCam loader and data_quality_report required.")

    (train_x, train_y, meta_train), (valid_x, valid_y, meta_valid), (test_x, test_y, meta_test) = dq.pcam_load_data(
        data_dir=str(data_dir)
    )
    splits_data = {
        "train": (train_x, train_y, meta_train),
        "valid": (valid_x, valid_y, meta_valid),
        "test": (test_x, test_y, meta_test),
    }

    kept = {}
    for split in ("train", "valid", "test"):
        x_ds, y_ds, meta = splits_data[split]
        n_total = len(x_ds)
        anomalies, hash_list, _, _ = dq._detect_anomalies_and_duplicates(
            x_ds,
            chunk_size=chunk_size,
            std_threshold=dq.STD_ZERO_THRESHOLD,
            bw_ratio_threshold=dq.BLACK_WHITE_PIXEL_RATIO_THRESHOLD,
            split_name=split,
            tissue_pct_threshold=tissue_pct_threshold,
            background_gray_threshold=dq.BACKGROUND_GRAY_THRESHOLD,
        )
        anomaly_indices = {a["index"] for a in anomalies}
        dup_groups = dq._find_duplicate_groups(hash_list)
        keep_set = set(range(n_total)) - anomaly_indices
        for group in dup_groups:
            # keep first, drop rest
            for idx in group[1:]:
                keep_set.discard(idx)
        indices = sorted(keep_set)
        kept[split] = indices

    return splits_data, kept


def get_macenko_normalizer(reference_rgb_uint8: np.ndarray):
    """Build Macenko normalizer fitted on reference image (HxWx3 uint8)."""
    try:
        from staintools import MacenkoNormalizer
    except ImportError:
        raise ImportError("staintools is required for Macenko. Install with: pip install staintools")
    normalizer = MacenkoNormalizer()
    normalizer.fit(reference_rgb_uint8)
    return normalizer


def apply_macenko_to_patch(patch: np.ndarray, normalizer) -> np.ndarray:
    """Apply Macenko normalizer to one patch. patch: uint8 HxWx3; returns uint8 HxWx3."""
    if patch.max() <= 1.0:
        patch = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
    return normalizer.transform(patch)


def run_preprocessing(
    data_dir: Path,
    out_dir: Path,
    tissue_pct_threshold: float = TISSUE_PCT_THRESHOLD,
    apply_stain_norm: bool = True,
    save_h5: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """
    Full preprocessing: compute kept indices, optionally save manifest and preprocessed H5 with Macenko.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_data, kept = run_detection_and_compute_kept(
        data_dir,
        tissue_pct_threshold=tissue_pct_threshold,
        chunk_size=chunk_size,
    )

    # Reference for Macenko: first kept training patch (uint8)
    train_x = splits_data["train"][0]
    ref_idx = kept["train"][0]
    ref_patch = np.asarray(train_x[ref_idx])
    if ref_patch.max() <= 1:
        ref_patch = (np.clip(ref_patch, 0, 1) * 255).astype(np.uint8)
    else:
        ref_patch = np.clip(ref_patch, 0, 255).astype(np.uint8)

    normalizer = None
    if apply_stain_norm:
        normalizer = get_macenko_normalizer(ref_patch)

    manifest = {
        "version": 1,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "tissue_pct_threshold": tissue_pct_threshold,
        "stain_normalization": "macenko" if apply_stain_norm else None,
        "reference_index_train": ref_idx,
        "splits": {},
    }

    for split in ("train", "valid", "test"):
        indices = kept[split]
        x_ds, y_ds, meta = splits_data[split]
        n_orig = len(x_ds)
        manifest["splits"][split] = {
            "n_original": n_orig,
            "n_kept": len(indices),
            "indices_file": f"{split}_indices.npy",
            "labels_file": f"{split}_labels.npy",
        }
        if meta is not None and len(indices) > 0:
            try:
                manifest["splits"][split]["meta_kept"] = meta.iloc[indices].to_dict(orient="list")
            except Exception:
                manifest["splits"][split]["meta_kept"] = None

    # Save indices and labels for each split (training can load these)
    for split in ("train", "valid", "test"):
        _, y_ds, _ = splits_data[split]
        y = np.asarray(y_ds).flatten()
        ind = kept[split]
        labels = y[ind]
        np.save(out_dir / f"{split}_indices.npy", np.array(ind, dtype=np.int64))
        np.save(out_dir / f"{split}_labels.npy", labels)

    manifest_path = out_dir / "preprocess_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    if save_h5:
        try:
            import h5py
        except ImportError:
            print("h5py not installed; skipping --save-h5.", file=sys.stderr)
        else:
            for split in ("train", "valid", "test"):
                indices = kept[split]
                if not indices:
                    continue
                x_ds = splits_data[split][0]
                n_kept = len(indices)
                sample = np.asarray(x_ds[indices[0]])
                shape = (n_kept,) + sample.shape
                suffix = "normalized" if normalizer else "filtered"
                out_h5 = out_dir / f"{split}_x_{suffix}.h5"
                with h5py.File(out_h5, "w") as f:
                    dset = f.create_dataset("x", shape=shape, dtype=np.uint8)
                    for i, idx in enumerate(tqdm(indices, desc=f"{split} save")):
                        patch = np.asarray(x_ds[idx])
                        if patch.max() <= 1:
                            patch = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
                        else:
                            patch = np.clip(patch, 0, 255).astype(np.uint8)
                        if normalizer:
                            patch = apply_macenko_to_patch(patch, normalizer)
                        dset[i] = patch
                print(f"Saved {out_h5}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="PCam preprocessing: quality filters, dedup, Macenko stain norm")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "pcam_data")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "preprocessed_pcam")
    parser.add_argument("--tissue-pct", type=float, default=TISSUE_PCT_THRESHOLD,
                        help="Exclude patches with tissue %% below this (default 0.25)")
    parser.add_argument("--no-stain-norm", action="store_true", help="Skip Macenko stain normalization")
    parser.add_argument("--save-h5", action="store_true", help="Write preprocessed H5 with stain-normalized patches")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Error: data dir not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    run_preprocessing(
        args.data_dir,
        args.out_dir,
        tissue_pct_threshold=args.tissue_pct,
        apply_stain_norm=not args.no_stain_norm,
        save_h5=args.save_h5,
        chunk_size=args.chunk_size,
    )
    print("Preprocessing done.")


if __name__ == "__main__":
    main()
