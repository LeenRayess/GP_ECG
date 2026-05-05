"""
Create fixed PCam subsets and preprocess them with multiple stain strategies.

Methods generated:
  1) macenko
  2) reinhard
  3) vahadane
  4) adaptive_single_ref
  5) adaptive_multi_ref
  6) adaptive_multi_ref_aug

Each method writes:
  <out-root>/<method>/{train,valid,test}_x.h5
  <out-root>/<method>/{train,valid,test}_y.h5

adaptive_multi_ref_aug applies augmentation to TRAIN split only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

import preprocess_histopath_h5 as pp

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it


def _to_float_label(y_ds, idx: int) -> float:
    return float(np.asarray(y_ds[idx]).reshape(-1)[0])


def _balanced_indices(y_path: Path, n_samples: int, seed: int) -> np.ndarray:
    with h5py.File(y_path, "r") as f:
        y = np.asarray(f["y"][:]).reshape(-1)
    pos = np.where(y >= 0.5)[0]
    neg = np.where(y < 0.5)[0]
    half = n_samples // 2
    rng = np.random.RandomState(seed)
    n_pos = min(half, len(pos))
    n_neg = min(n_samples - n_pos, len(neg))
    p = rng.choice(pos, size=n_pos, replace=False)
    n = rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([p, n]).astype(np.int64)
    rng.shuffle(idx)
    return idx


def _quality_filter_indices(x_path: Path, indices: np.ndarray) -> np.ndarray:
    kept = []
    with h5py.File(x_path, "r") as f:
        x = f["x"]
        for idx in tqdm(indices, desc=f"quality {x_path.stem}", unit="patch", leave=False):
            p01 = pp.to_01(np.asarray(x[int(idx)]))
            passed, _ = pp.passes_quality(p01)
            if passed:
                kept.append(int(idx))
    return np.asarray(kept, dtype=np.int64)


def _write_xy(
    x_src: Path,
    y_src: Path,
    indices: np.ndarray,
    out_x: Path,
    out_y: Path,
    transform_fn,
):
    out_x.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(x_src, "r") as fx, h5py.File(y_src, "r") as fy, h5py.File(out_x, "w") as ox, h5py.File(out_y, "w") as oy:
        x_ds = fx["x"]
        y_ds = fy["y"]
        ox.create_dataset("x", shape=(len(indices), 96, 96, 3), dtype=np.float32, chunks=(1, 96, 96, 3), compression="gzip")
        oy.create_dataset("y", shape=(len(indices),), dtype=np.float32)
        for i, idx in enumerate(tqdm(indices, desc=f"write {out_x.parent.name}/{out_x.stem}", unit="patch", leave=False)):
            p = np.asarray(x_ds[int(idx)])
            p01 = pp.to_01(p)
            ox["x"][i] = transform_fn(p01)
            oy["y"][i] = _to_float_label(y_ds, int(idx))


def _load_progress(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_steps": []}


def _save_progress(path: Path, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _is_done(state: dict, step: str) -> bool:
    return step in state.get("completed_steps", [])


def _mark_done(path: Path, state: dict, step: str) -> None:
    if step not in state["completed_steps"]:
        state["completed_steps"].append(step)
    _save_progress(path, state)


def _h5_pair_complete(out_x: Path, out_y: Path, expected_n: int) -> bool:
    if not out_x.exists() or not out_y.exists():
        return False
    try:
        with h5py.File(out_x, "r") as fx, h5py.File(out_y, "r") as fy:
            return fx["x"].shape[0] == expected_n and fy["y"].shape[0] == expected_n
    except Exception:
        return False


def _fit_single_ref(train_x_path: Path, ref_config_path: Path):
    with open(ref_config_path, "r", encoding="utf-8") as f:
        ref_idx = int(json.load(f)["reference_train_index"])
    with h5py.File(train_x_path, "r") as f:
        ref_patch = np.asarray(f["x"][ref_idx])
    ref_u8 = pp.to_uint8(ref_patch)
    from staintools.preprocessing.luminosity_standardizer import LuminosityStandardizer
    from staintools import StainNormalizer, ReinhardColorNormalizer

    ref_fit = LuminosityStandardizer.standardize(ref_u8.copy())
    mac = StainNormalizer(method="macenko")
    mac.fit(ref_fit)
    rei = ReinhardColorNormalizer()
    rei.fit(ref_u8)
    vah = StainNormalizer(method="vahadane")
    vah.fit(ref_fit)
    return mac, rei, vah


def _classical_transform(normalizer, lum_std_fn):
    def _fn(p01: np.ndarray) -> np.ndarray:
        pu8 = pp.to_uint8(p01)
        try:
            pstd = lum_std_fn(pu8)
            out_u8 = normalizer.transform(pstd)
            return np.clip(out_u8.astype(np.float32) / 255.0, 0.0, 1.0)
        except Exception:
            return pp._luminosity_only_norm(p01, lum_std_fn)

    return _fn


def _simple_aug(p01: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    x = p01.copy()
    gain = rng.uniform(0.9, 1.1)
    bias = rng.uniform(-0.03, 0.03)
    x = np.clip(x * gain + bias, 0.0, 1.0)
    sat = rng.uniform(0.9, 1.1)
    mean = x.mean(axis=2, keepdims=True)
    x = np.clip((x - mean) * sat + mean, 0.0, 1.0)
    return x.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-method stain-normalized H5 subsets for benchmarking.")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "pcam_data"))
    parser.add_argument("--out-root", type=str, default=str(Path(__file__).resolve().parent.parent / "pcam_data" / "benchmark_preprocessed"))
    parser.add_argument("--train-n", type=int, default=40000)
    parser.add_argument("--valid-n", type=int, default=8000)
    parser.add_argument("--test-n", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref-config", type=str, default=str(Path(__file__).resolve().parent.parent / "experiments" / "stain_reference" / "stain_reference.json"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    ref_config = Path(args.ref_config)
    progress_path = out_root / "prepare_progress.json"

    train_x = data_dir / "training" / "camelyonpatch_level_2_split_train_x.h5"
    train_y = data_dir / "training" / "camelyonpatch_level_2_split_train_y.h5"
    valid_x = data_dir / "val" / "camelyonpatch_level_2_split_valid_x.h5"
    valid_y = data_dir / "val" / "camelyonpatch_level_2_split_valid_y.h5"
    test_x = data_dir / "test" / "camelyonpatch_level_2_split_test_x.h5"
    test_y = data_dir / "test" / "camelyonpatch_level_2_split_test_y.h5"

    for p in [train_x, train_y, valid_x, valid_y, test_x, test_y, ref_config]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    out_root.mkdir(parents=True, exist_ok=True)
    state = _load_progress(progress_path)

    idx_raw_path = out_root / "subset_indices.json"
    idx_qf_path = out_root / "subset_indices_quality_filtered.json"
    if idx_raw_path.exists() and idx_qf_path.exists():
        with open(idx_raw_path, "r", encoding="utf-8") as f:
            split_indices = {k: np.asarray(v, dtype=np.int64) for k, v in json.load(f).items()}
        with open(idx_qf_path, "r", encoding="utf-8") as f:
            split_indices_qf = {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in json.load(f).items()}
        print("Loaded existing subset index files.")
    else:
        print("Sampling balanced subsets...")
        split_indices = {
            "train": _balanced_indices(train_y, args.train_n, args.seed),
            "valid": _balanced_indices(valid_y, args.valid_n, args.seed + 1),
            "test": _balanced_indices(test_y, args.test_n, args.seed + 2),
        }
        print("Applying shared quality filter...")
        split_indices_qf = {
            "train": _quality_filter_indices(train_x, split_indices["train"]),
            "valid": _quality_filter_indices(valid_x, split_indices["valid"]),
            "test": _quality_filter_indices(test_x, split_indices["test"]),
        }
        split_indices_qf = {k: np.sort(v) for k, v in split_indices_qf.items()}
        with open(idx_raw_path, "w", encoding="utf-8") as f:
            json.dump({k: v.tolist() for k, v in split_indices.items()}, f, indent=2)
        with open(idx_qf_path, "w", encoding="utf-8") as f:
            json.dump({k: v.tolist() for k, v in split_indices_qf.items()}, f, indent=2)
        _mark_done(progress_path, state, "subset_indices_ready")

    mac, rei, vah = _fit_single_ref(train_x, ref_config)
    from staintools.preprocessing.luminosity_standardizer import LuminosityStandardizer

    lum_std = LuminosityStandardizer()
    classic = {
        "macenko": _classical_transform(mac, lum_std.standardize),
        "reinhard": _classical_transform(rei, lum_std.standardize),
        "vahadane": _classical_transform(vah, lum_std.standardize),
    }

    split_paths = {
        "train": (train_x, train_y),
        "valid": (valid_x, valid_y),
        "test": (test_x, test_y),
    }

    # Prioritize faster methods first; run Vahadane last.
    for method in ("macenko", "reinhard"):
        tfm = classic[method]
        method_dir = out_root / method
        for split, (x_path, y_path) in split_paths.items():
            step = f"classic:{method}:{split}"
            idx = split_indices_qf[split]
            out_x = method_dir / f"{split}_x.h5"
            out_y = method_dir / f"{split}_y.h5"
            if _is_done(state, step) or _h5_pair_complete(out_x, out_y, len(idx)):
                print(f"Skip {step} (already complete).")
                _mark_done(progress_path, state, step)
                continue
            print(f"Run {step} ...")
            _write_xy(x_path, y_path, idx, out_x, out_y, tfm)
            _mark_done(progress_path, state, step)

    # Adaptive single reference (existing pipeline behavior).
    single_dir = out_root / "adaptive_single_ref"
    single_dir.mkdir(parents=True, exist_ok=True)
    mac0, rei0, ref_idx, ref_mean_rgb = pp.get_normalizers_and_threshold(str(data_dir), str(train_x), str(ref_config))
    for split, (x_path, y_path) in split_paths.items():
        step = f"adaptive_single_ref:{split}"
        out_x = single_dir / f"{split}_x.h5"
        out_y = single_dir / f"{split}_y.h5"
        idx = split_indices_qf[split]
        if _is_done(state, step) or _h5_pair_complete(out_x, out_y, len(idx)):
            print(f"Skip {step} (already complete).")
            _mark_done(progress_path, state, step)
            continue
        print(f"Run {step} ...")
        pp.process_split(
            split_name=split,
            x_path=str(x_path),
            y_path=str(y_path),
            out_dir=str(single_dir),
            macenko=mac0,
            reinhard=rei0,
            ref_mean_rgb=ref_mean_rgb,
            candidate_indices=idx,
            resume=False,
            stain_router=None,
            ref_packs=None,
        )
        _mark_done(progress_path, state, step)

    # Adaptive multi-reference.
    multi_dir = out_root / "adaptive_multi_ref"
    multi_dir.mkdir(parents=True, exist_ok=True)
    router = pp.fit_multi_ref_router(str(train_x), train_candidate_indices=split_indices_qf["train"])
    packs = pp.build_ref_packs(str(train_x), pp.MULTI_REF_INDICES)
    for split, (x_path, y_path) in split_paths.items():
        step = f"adaptive_multi_ref:{split}"
        out_x = multi_dir / f"{split}_x.h5"
        out_y = multi_dir / f"{split}_y.h5"
        idx = split_indices_qf[split]
        if _is_done(state, step) or _h5_pair_complete(out_x, out_y, len(idx)):
            print(f"Skip {step} (already complete).")
            _mark_done(progress_path, state, step)
            continue
        print(f"Run {step} ...")
        pp.process_split(
            split_name=split,
            x_path=str(x_path),
            y_path=str(y_path),
            out_dir=str(multi_dir),
            macenko=packs[0]["macenko"],
            reinhard=packs[0]["reinhard"],
            ref_mean_rgb=packs[0]["ref_mean_rgb"],
            candidate_indices=idx,
            resume=False,
            stain_router=router,
            ref_packs=packs,
        )
        _mark_done(progress_path, state, step)

    # Adaptive multi-reference + augmentation (train only augmented).
    multi_aug_dir = out_root / "adaptive_multi_ref_aug"
    multi_aug_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed + 999)
    for split in ("train", "valid", "test"):
        step = f"adaptive_multi_ref_aug:{split}"
        src_x = multi_dir / f"{split}_x.h5"
        src_y = multi_dir / f"{split}_y.h5"
        out_x = multi_aug_dir / f"{split}_x.h5"
        out_y = multi_aug_dir / f"{split}_y.h5"
        expected_n = int(len(split_indices_qf[split]))
        if _is_done(state, step) or _h5_pair_complete(out_x, out_y, expected_n):
            print(f"Skip {step} (already complete).")
            _mark_done(progress_path, state, step)
            continue
        print(f"Run {step} ...")
        with h5py.File(src_x, "r") as fx, h5py.File(src_y, "r") as fy, h5py.File(out_x, "w") as ox, h5py.File(out_y, "w") as oy:
            x = fx["x"]
            y = np.asarray(fy["y"][:], dtype=np.float32)
            n = x.shape[0]
            ox.create_dataset("x", shape=(n, 96, 96, 3), dtype=np.float32, chunks=(1, 96, 96, 3), compression="gzip")
            oy.create_dataset("y", data=y)
            for i in tqdm(range(n), desc=f"write adaptive_multi_ref_aug/{split}", unit="patch", leave=False):
                p = np.asarray(x[i], dtype=np.float32)
                ox["x"][i] = _simple_aug(p, rng) if split == "train" else p
        _mark_done(progress_path, state, step)

    # Run Vahadane last.
    method = "vahadane"
    tfm = classic[method]
    method_dir = out_root / method
    for split, (x_path, y_path) in split_paths.items():
        step = f"classic:{method}:{split}"
        idx = split_indices_qf[split]
        out_x = method_dir / f"{split}_x.h5"
        out_y = method_dir / f"{split}_y.h5"
        if _is_done(state, step) or _h5_pair_complete(out_x, out_y, len(idx)):
            print(f"Skip {step} (already complete).")
            _mark_done(progress_path, state, step)
            continue
        print(f"Run {step} ...")
        _write_xy(x_path, y_path, idx, out_x, out_y, tfm)
        _mark_done(progress_path, state, step)

    print("Done. Output root:", out_root)
    print("Methods:", ["macenko", "reinhard", "vahadane", "adaptive_single_ref", "adaptive_multi_ref", "adaptive_multi_ref_aug"])
    print("Shared subset sizes after quality filter:",
          {k: int(len(v)) for k, v in split_indices_qf.items()})
    print("Progress file:", progress_path)


if __name__ == "__main__":
    main()

