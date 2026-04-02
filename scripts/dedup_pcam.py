"""PCam deduplication by SHA-256 of raw patch bytes; one index kept per identical group.

Writes manifest.json and {train,valid,test}_kept_indices.npy unless --write-h5 is used to emit a
loader-compatible tree. Use --verify to re-hash kept indices and check counts against the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PCAM_MASTER = PROJECT_ROOT / "pcam-master"
if PCAM_MASTER.is_dir():
    sys.path.insert(0, str(PCAM_MASTER))

try:
    from keras_pcam.dataset.pcam import load_data
except ImportError as e:
    load_data = None
    _import_err = e
else:
    _import_err = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

CHUNK_SIZE = 2000


def _compute_hash_list(x_ds, chunk_size: int, split_name: str):
    """Return list of (index, sha256_hex) for each image in x_ds."""
    n = len(x_ds)
    hash_list = []
    for start in tqdm(range(0, n, chunk_size), desc=f"{split_name} hash", unit="chunk"):
        end = min(start + chunk_size, n)
        batch = np.asarray(x_ds[start:end])
        for i in range(batch.shape[0]):
            idx = start + i
            raw = batch[i].tobytes()
            h = hashlib.sha256(raw).hexdigest()
            hash_list.append((idx, h))
    return hash_list


def _find_duplicate_groups(hash_list):
    """Return list of groups (each group = sorted list of indices with same hash)."""
    h2idx = defaultdict(list)
    for idx, h in hash_list:
        h2idx[h].append(idx)
    return [sorted(idxs) for idxs in h2idx.values() if len(idxs) > 1]


def _kept_indices(n_total: int, duplicate_groups: list[list[int]]) -> np.ndarray:
    """Indices to keep: all indices minus duplicates (keep first of each group)."""
    exclude = set()
    for group in duplicate_groups:
        for idx in group[1:]:  # keep group[0], drop the rest
            exclude.add(idx)
    return np.array([i for i in range(n_total) if i not in exclude], dtype=np.int64)


def run_dedup(data_dir: Path, chunk_size: int = CHUNK_SIZE):
    """
    Load PCam from data_dir, compute duplicate groups per split, return kept indices.
    Returns (splits_data, kept_per_split).
    splits_data = {"train": (x_ds, y_ds, meta), ...}
    kept_per_split = {"train": array of indices, ...}
    """
    if load_data is None:
        raise ImportError("PCam loader not available.") from _import_err

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    (train_x, train_y, meta_train), (valid_x, valid_y, meta_valid), (test_x, test_y, meta_test) = load_data(
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
        hash_list = _compute_hash_list(x_ds, chunk_size, split)
        dup_groups = _find_duplicate_groups(hash_list)
        kept[split] = _kept_indices(n_total, dup_groups)
        n_dup = sum(len(g) - 1 for g in dup_groups)
        print(f"  {split}: n_original={n_total}, n_duplicate_images_removed={n_dup}, n_kept={len(kept[split])}")

    return splits_data, kept


def verify_dedup(out_dir: Path, chunk_size: int = CHUNK_SIZE) -> bool:
    """Re-hash kept indices; require unique hashes and manifest count consistency."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
        return False
    with open(manifest_path) as f:
        manifest = json.load(f)
    data_dir = Path(manifest["data_dir"])
    if not data_dir.is_dir():
        print(f"Error: data dir not found: {data_dir}", file=sys.stderr)
        return False

    if load_data is None:
        print("Error: PCam loader not available.", file=sys.stderr)
        return False

    print("Loading data (H5 handles only; images read during verify)...")
    (train_x, train_y, _), (valid_x, valid_y, _), (test_x, test_y, _) = load_data(data_dir=str(data_dir))
    splits_data = {"train": train_x, "valid": valid_x, "test": test_x}
    ok = True
    for split in ("train", "valid", "test"):
        indices_path = out_dir / f"{split}_kept_indices.npy"
        if not indices_path.exists():
            print(f"  {split}: missing {indices_path}", file=sys.stderr)
            ok = False
            continue
        kept = np.load(indices_path)
        info = manifest["splits"].get(split, {})
        n_orig = info.get("n_original")
        n_kept = info.get("n_kept")
        n_removed = info.get("n_removed")
        if n_orig is None or n_kept is None:
            ok = False
            continue
        if len(kept) != n_kept:
            print(f"  {split}: FAIL len(kept)={len(kept)} != manifest n_kept={n_kept}", file=sys.stderr)
            ok = False
        if n_orig != n_kept + n_removed:
            print(f"  {split}: FAIL n_original != n_kept + n_removed ({n_orig} vs {n_kept} + {n_removed})", file=sys.stderr)
            ok = False
        x_ds = splits_data[split]
        n_total = len(x_ds)
        if n_total != n_orig:
            print(f"  {split}: FAIL current data len={n_total} != manifest n_original={n_orig}", file=sys.stderr)
            ok = False
        # Re-hash all kept indices and ensure no two have the same hash (can be slow: reads every kept patch)
        hashes = []
        for start in tqdm(range(0, len(kept), chunk_size), desc=f"  {split} verify", unit="chunk"):
            end = min(start + chunk_size, len(kept))
            batch_idx = kept[start:end]
            batch = np.asarray(x_ds[batch_idx])
            for j in range(len(batch_idx)):
                hashes.append(hashlib.sha256(batch[j].tobytes()).hexdigest())
        n_unique = len(set(hashes))
        if n_unique != len(kept):
            print(f"  {split}: FAIL {len(kept) - n_unique} duplicate hashes among kept (expected 0)", file=sys.stderr)
            ok = False
        else:
            print(f"  {split}: OK n_original={n_orig} n_kept={n_kept} n_removed={n_removed} unique_hashes_among_kept={n_unique}")
    return ok


def write_h5_layout(out_dir: Path, splits_data: dict, kept_per_split: dict):
    """Emit .h5/.csv compatible with keras_pcam load_data(out_dir)."""
    import h5py

    layout = [
        ("train", "training", "camelyonpatch_level_2_split_train"),
        ("valid", "val", "camelyonpatch_level_2_split_valid"),
        ("test", "test", "camelyonpatch_level_2_split_test"),
    ]
    for split_key, subdir, prefix in layout:
        x_ds, y_ds, meta = splits_data[split_key]
        indices = kept_per_split[split_key]
        if len(indices) == 0:
            continue
        out_sub = out_dir / subdir
        out_sub.mkdir(parents=True, exist_ok=True)

        sample_x = np.asarray(x_ds[indices[0]])
        sample_y = np.asarray(y_ds[indices[0]])
        if hasattr(sample_y, "shape") and sample_y.shape:
            y_shape = (len(indices),) + tuple(sample_y.shape)
        else:
            sample_y = np.asarray(y_ds[indices[0]]).flatten()
            y_shape = (len(indices),)

        x_path = out_sub / f"{prefix}_x.h5"
        y_path = out_sub / f"{prefix}_y.h5"
        with h5py.File(x_path, "w") as f:
            dset = f.create_dataset("x", shape=(len(indices),) + sample_x.shape, dtype=sample_x.dtype)
            for i, idx in enumerate(tqdm(indices, desc=f"write {split_key} x")):
                dset[i] = np.asarray(x_ds[idx])
        y_arr = np.asarray(y_ds[indices]).flatten()
        with h5py.File(y_path, "w") as f:
            f.create_dataset("y", data=y_arr)

        if meta is not None:
            meta_kept = meta.iloc[indices]
            meta_path = out_sub / f"{prefix}_meta.csv"
            meta_kept.to_csv(meta_path, index=False)

        print(f"  Wrote {out_sub}/ ({len(indices)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate PCam by pixel hash; optionally write new .h5 for loader.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "pcam_data",
                        help="Input pcam_data directory (default: project_root/pcam_data)")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "pcam_dedup",
                        help="Output directory for manifest and index files (default: project_root/pcam_dedup)")
    parser.add_argument("--write-h5", action="store_true",
                        help="Also write new .h5 and .csv so load_data(out_dir) works as drop-in")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Chunk size for hashing (default 2000)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing dedup output: re-hash kept indices and check no duplicates among kept, counts match")
    args = parser.parse_args()

    if args.verify:
        ok = verify_dedup(args.out_dir, chunk_size=args.chunk_size)
        sys.exit(0 if ok else 1)

    if not args.data_dir.is_dir():
        print(f"Error: data dir not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Deduplicating PCam (hash-based, keep one per duplicate group)...")
    splits_data, kept = run_dedup(args.data_dir, chunk_size=args.chunk_size)

    for split in ("train", "valid", "test"):
        np.save(args.out_dir / f"{split}_kept_indices.npy", kept[split])

    manifest = {
        "version": 1,
        "data_dir": str(args.data_dir.resolve()),
        "out_dir": str(args.out_dir),
        "chunk_size": args.chunk_size,
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        x_ds = splits_data[split][0]
        n_orig = len(x_ds)
        n_kept = len(kept[split])
        manifest["splits"][split] = {
            "n_original": n_orig,
            "n_kept": n_kept,
            "n_removed": n_orig - n_kept,
            "indices_file": f"{split}_kept_indices.npy",
        }
    manifest_path = args.out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")

    if args.write_h5:
        print("Writing H5 layout (loader-compatible)...")
        write_h5_layout(args.out_dir, splits_data, kept)
        print("Use with: load_data(data_dir=r'{}')".format(args.out_dir))

    print("Done.")


if __name__ == "__main__":
    main()
