import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


def to_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return np.asarray(np.clip(arr, 0.0, 1.0) * 255.0, dtype=np.uint8)


def hash_split(x_path: Path, chunk_size: int) -> set[str]:
    hs: set[str] = set()
    with h5py.File(x_path, "r") as f:
        x = f["x"]
        n = int(x.shape[0])
        print(f"[hash] {x_path.name}: {n} samples", flush=True)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            b = np.asarray(x[s:e])
            for i in range(b.shape[0]):
                u = to_u8(b[i])
                hs.add(hashlib.sha256(u.tobytes()).hexdigest())
    return hs


def overlap_count(x_path: Path, ref_hashes: set[str], chunk_size: int) -> tuple[int, int]:
    n = 0
    ov = 0
    with h5py.File(x_path, "r") as f:
        x = f["x"]
        n = int(x.shape[0])
        print(f"[match] {x_path.name}: {n} samples", flush=True)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            b = np.asarray(x[s:e])
            for i in range(b.shape[0]):
                u = to_u8(b[i])
                h = hashlib.sha256(u.tobytes()).hexdigest()
                if h in ref_hashes:
                    ov += 1
    return n, ov


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pcam-dir", type=Path, default=Path("pcam_data/preprocessed_macenko_benchmark_style"))
    p.add_argument(
        "--cam-dir",
        type=Path,
        default=Path("data/wilds/camelyon17_h5_full_oodval/preprocessed_macenko_benchmark_style"),
    )
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--out-json", type=Path, default=Path("reports/data_integrity/preprocessed_overlap_exact.json"))
    args = p.parse_args()

    pcam_splits = ["train", "valid", "test"]
    cam_splits = ["train", "valid", "test"]

    pcam_hashes_union: set[str] = set()
    pcam_split_hash_counts = {}
    for sp in pcam_splits:
        hs = hash_split(args.pcam_dir / f"{sp}_x.h5", args.chunk_size)
        pcam_hashes_union |= hs
        pcam_split_hash_counts[sp] = len(hs)
        print(f"[done-hash] pcam {sp}: {len(hs)} unique", flush=True)

    cam_hashes_union: set[str] = set()
    cam_split_hash_counts = {}
    for sp in cam_splits:
        hs = hash_split(args.cam_dir / f"{sp}_x.h5", args.chunk_size)
        cam_hashes_union |= hs
        cam_split_hash_counts[sp] = len(hs)
        print(f"[done-hash] cam17 {sp}: {len(hs)} unique", flush=True)

    cam_vs_pcam = {}
    for sp in cam_splits:
        n, ov = overlap_count(args.cam_dir / f"{sp}_x.h5", pcam_hashes_union, args.chunk_size)
        cam_vs_pcam[sp] = {"n_samples": n, "n_overlap_with_any_pcam": ov, "overlap_rate": (ov / n if n else 0.0)}
        print(f"[done-match] cam17 {sp}: overlap={ov}/{n}", flush=True)

    pcam_vs_cam = {}
    for sp in pcam_splits:
        n, ov = overlap_count(args.pcam_dir / f"{sp}_x.h5", cam_hashes_union, args.chunk_size)
        pcam_vs_cam[sp] = {"n_samples": n, "n_overlap_with_any_cam17": ov, "overlap_rate": (ov / n if n else 0.0)}
        print(f"[done-match] pcam {sp}: overlap={ov}/{n}", flush=True)

    out = {
        "created_at": datetime.now().isoformat(),
        "pcam_dir": str(args.pcam_dir),
        "cam17_dir": str(args.cam_dir),
        "hashing_space": "sha256(uint8_rgb_bytes)",
        "pcam_unique_hashes_by_split": pcam_split_hash_counts,
        "cam17_unique_hashes_by_split": cam_split_hash_counts,
        "pcam_unique_hashes_union": len(pcam_hashes_union),
        "cam17_unique_hashes_union": len(cam_hashes_union),
        "cam17_split_overlap_vs_any_pcam": cam_vs_pcam,
        "pcam_split_overlap_vs_any_cam17": pcam_vs_cam,
        "union_intersection_hashes": len(pcam_hashes_union.intersection(cam_hashes_union)),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
