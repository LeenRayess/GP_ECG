"""Comprehensive dataset audit for PCam and CAMELYON17 (WILDS) H5 splits.

Outputs report-ready artifacts (JSON/CSV/plots) for methodology/results sections.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class SplitPaths:
    x_path: Path
    y_path: Path
    split_name: str


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_split_pairs(dataset_dir: Path, split_order: List[str]) -> List[SplitPaths]:
    pairs: List[SplitPaths] = []
    for split in split_order:
        x_path = dataset_dir / f"{split}_x.h5"
        y_path = dataset_dir / f"{split}_y.h5"
        if x_path.is_file() and y_path.is_file():
            pairs.append(SplitPaths(x_path=x_path, y_path=y_path, split_name=split))
    return pairs


def _compute_label_stats(y_values: np.ndarray) -> Dict[str, Any]:
    y = y_values.reshape(-1).astype(np.float64)
    uniq, cnt = np.unique(y, return_counts=True)
    counts = {str(float(k)): int(v) for k, v in zip(uniq, cnt)}
    n = int(y.size)
    pos = int(np.sum(y > 0.5))
    neg = int(n - pos)
    frac_pos = float(pos / n) if n > 0 else math.nan
    return {
        "n_samples": n,
        "n_positive": pos,
        "n_negative": neg,
        "frac_positive": frac_pos,
        "label_value_counts": counts,
    }


def _sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    take = min(n, max_n)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    idx = sorted(idx[:take])
    return np.asarray(idx, dtype=np.int64)


def _img_to_float01(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _per_image_stats(img01: np.ndarray) -> Dict[str, float]:
    # img01: H, W, 3 in [0,1]
    r = img01[:, :, 0]
    g = img01[:, :, 1]
    b = img01[:, :, 2]
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat = np.where(maxc > 1e-8, (maxc - minc) / np.maximum(maxc, 1e-8), 0.0)
    tissue_proxy = float(np.mean(sat > 0.12))
    black_ratio = float(np.mean(gray < 0.05))
    blue_dom_ratio = float(np.mean(b > r))
    pink_proxy = float(np.mean((r > 0.55) & (g > 0.35) & (b > 0.55)))
    return {
        "mean_r": float(np.mean(r)),
        "mean_g": float(np.mean(g)),
        "mean_b": float(np.mean(b)),
        "std_r": float(np.std(r)),
        "std_g": float(np.std(g)),
        "std_b": float(np.std(b)),
        "mean_gray": float(np.mean(gray)),
        "std_gray": float(np.std(gray)),
        "sat_mean": float(np.mean(sat)),
        "tissue_proxy_ratio_sat_gt_0p12": tissue_proxy,
        "black_ratio_gray_lt_0p05": black_ratio,
        "blue_dominance_ratio_b_gt_r": blue_dom_ratio,
        "pink_proxy_ratio": pink_proxy,
        "min_pixel": float(np.min(img01)),
        "max_pixel": float(np.max(img01)),
    }


def _sample_duplicate_rate(x_ds: h5py.Dataset, sample_idx: np.ndarray) -> Dict[str, Any]:
    if sample_idx.size == 0:
        return {"sampled": 0, "unique_hashes": 0, "duplicate_fraction": None}
    hashes: List[str] = []
    for i in sample_idx:
        arr = np.asarray(x_ds[int(i)])
        h = hashlib.md5(arr.tobytes()).hexdigest()
        hashes.append(h)
    counter = Counter(hashes)
    n_dup = int(sum(v - 1 for v in counter.values() if v > 1))
    return {
        "sampled": int(sample_idx.size),
        "unique_hashes": int(len(counter)),
        "duplicate_fraction": float(n_dup / sample_idx.size),
    }


def _aggregate_metric(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def analyze_split(
    split: SplitPaths,
    sample_n: int,
    seed: int,
    out_dir: Path,
    dataset_name: str,
) -> Dict[str, Any]:
    split_dir = out_dir / dataset_name / split.split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(split.x_path, "r") as fx, h5py.File(split.y_path, "r") as fy:
        x_key = "x" if "x" in fx else list(fx.keys())[0]
        y_key = "y" if "y" in fy else list(fy.keys())[0]
        x_ds = fx[x_key]
        y_ds = fy[y_key]

        n_x = int(x_ds.shape[0])
        n_y = int(y_ds.shape[0])
        n = min(n_x, n_y)
        sampled_idx = _sample_indices(n=n, max_n=sample_n, seed=seed)

        y_all = np.asarray(y_ds[:]).reshape(-1)
        label_stats = _compute_label_stats(y_all)

        per_image_rows: List[Dict[str, Any]] = []
        agg_lists: Dict[str, List[float]] = defaultdict(list)
        for idx in sampled_idx:
            img = np.asarray(x_ds[int(idx)])
            img01 = _img_to_float01(img)
            st = _per_image_stats(img01)
            row = {"index": int(idx), "label": float(y_all[int(idx)])}
            row.update(st)
            per_image_rows.append(row)
            for k, v in st.items():
                agg_lists[k].append(float(v))

        dup_info = _sample_duplicate_rate(x_ds, sampled_idx)

        per_image_csv = split_dir / "sampled_per_image_stats.csv"
        with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
            if per_image_rows:
                writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_image_rows)
            else:
                f.write("index,label\n")

        summary = {
            "split": split.split_name,
            "x_file": str(split.x_path),
            "y_file": str(split.y_path),
            "x_file_size_bytes": split.x_path.stat().st_size,
            "y_file_size_bytes": split.y_path.stat().st_size,
            "x_key": x_key,
            "y_key": y_key,
            "x_shape": tuple(int(v) for v in x_ds.shape),
            "y_shape": tuple(int(v) for v in y_ds.shape),
            "x_dtype": str(x_ds.dtype),
            "y_dtype": str(y_ds.dtype),
            "n_x": n_x,
            "n_y": n_y,
            "label_stats": label_stats,
            "sampled_n": int(sampled_idx.size),
            "sampled_idx_head": sampled_idx[:20].tolist(),
            "sampled_metrics_aggregate": {k: _aggregate_metric(v) for k, v in agg_lists.items()},
            "sample_duplicate_audit": dup_info,
            "per_image_stats_csv": str(per_image_csv),
            "y_keys_available": [str(k) for k in fy.keys()],
        }

    _write_json(split_dir / "summary.json", summary)
    return summary


def _plot_label_balance(dataset_name: str, split_summaries: List[Dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        return
    splits = [s["split"] for s in split_summaries]
    pos = [s["label_stats"]["n_positive"] for s in split_summaries]
    neg = [s["label_stats"]["n_negative"] for s in split_summaries]
    x = np.arange(len(splits))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, neg, w, label="negative")
    ax.bar(x + w / 2, pos, w, label="positive")
    ax.set_title(f"{dataset_name}: class balance by split")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=20)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sample_rgb_mean(dataset_name: str, split_summaries: List[Dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        return
    splits = [s["split"] for s in split_summaries]
    mr = [s["sampled_metrics_aggregate"]["mean_r"]["mean"] for s in split_summaries]
    mg = [s["sampled_metrics_aggregate"]["mean_g"]["mean"] for s in split_summaries]
    mb = [s["sampled_metrics_aggregate"]["mean_b"]["mean"] for s in split_summaries]
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, mr, marker="o", label="R")
    ax.plot(x, mg, marker="o", label="G")
    ax.plot(x, mb, marker="o", label="B")
    ax.set_title(f"{dataset_name}: sampled mean channel intensity by split")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean intensity [0,1]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_dataset(
    dataset_name: str,
    dataset_dir: Path,
    split_order: List[str],
    sample_n: int,
    seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    pairs = _find_split_pairs(dataset_dir=dataset_dir, split_order=split_order)
    summaries: List[Dict[str, Any]] = []
    for i, pair in enumerate(pairs):
        summaries.append(
            analyze_split(
                split=pair,
                sample_n=sample_n,
                seed=seed + i * 10007,
                out_dir=out_dir,
                dataset_name=dataset_name,
            )
        )

    dataset_out_dir = out_dir / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    _plot_label_balance(dataset_name, summaries, dataset_out_dir / "plot_class_balance.png")
    _plot_sample_rgb_mean(dataset_name, summaries, dataset_out_dir / "plot_sampled_rgb_mean.png")

    preprocess_report = _read_json_if_exists(dataset_dir / "preprocess_report.json")
    manifest = _read_json_if_exists(dataset_dir / "manifest.json")

    aggregate = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "splits_found": [s["split"] for s in summaries],
        "n_total_samples_across_found_splits": int(sum(s["label_stats"]["n_samples"] for s in summaries)),
        "split_summaries": summaries,
        "manifest_present": manifest is not None,
        "preprocess_report_present": preprocess_report is not None,
        "manifest_path": str(dataset_dir / "manifest.json"),
        "preprocess_report_path": str(dataset_dir / "preprocess_report.json"),
    }
    if manifest is not None:
        aggregate["manifest_excerpt"] = {
            k: manifest.get(k) for k in ("version", "out_dir", "valid_source", "stats") if k in manifest
        }
    if preprocess_report is not None:
        aggregate["preprocess_report_excerpt"] = {
            k: preprocess_report.get(k)
            for k in ("pipeline", "config", "dataset_summary")
            if k in preprocess_report
        }

    _write_json(dataset_out_dir / "dataset_summary.json", aggregate)
    return aggregate


def _write_comparison_csv(dataset_summaries: List[Dict[str, Any]], out_csv: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for ds in dataset_summaries:
        dname = ds["dataset_name"]
        for s in ds["split_summaries"]:
            row = {
                "dataset": dname,
                "split": s["split"],
                "n_samples": s["label_stats"]["n_samples"],
                "n_positive": s["label_stats"]["n_positive"],
                "n_negative": s["label_stats"]["n_negative"],
                "frac_positive": s["label_stats"]["frac_positive"],
                "sampled_mean_r": s["sampled_metrics_aggregate"]["mean_r"]["mean"],
                "sampled_mean_g": s["sampled_metrics_aggregate"]["mean_g"]["mean"],
                "sampled_mean_b": s["sampled_metrics_aggregate"]["mean_b"]["mean"],
                "sampled_tissue_proxy_mean": s["sampled_metrics_aggregate"][
                    "tissue_proxy_ratio_sat_gt_0p12"
                ]["mean"],
                "sampled_black_ratio_mean": s["sampled_metrics_aggregate"]["black_ratio_gray_lt_0p05"]["mean"],
                "sampled_blue_dom_mean": s["sampled_metrics_aggregate"]["blue_dominance_ratio_b_gt_r"]["mean"],
                "sampled_duplicate_fraction": s["sample_duplicate_audit"]["duplicate_fraction"],
            }
            rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset", "split"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Audit PCam and CAMELYON17 (WILDS) H5 datasets for report writing.")
    p.add_argument(
        "--pcam-dir",
        type=Path,
        default=Path("pcam_data/preprocessed_macenko_benchmark_style"),
        help="Directory containing PCam split H5 files (train/valid/test).",
    )
    p.add_argument(
        "--cam17-dir",
        type=Path,
        default=Path("data/wilds/camelyon17_h5_full_oodval/preprocessed_macenko_benchmark_style"),
        help="Directory containing CAMELYON17(WILDS) split H5 files.",
    )
    p.add_argument(
        "--cam17-splits",
        type=str,
        default="train,valid,test",
        help="Comma-separated CAMELYON17 split names to inspect if present (e.g., train,valid,test,id_val,ood_val).",
    )
    p.add_argument("--sample-n", type=int, default=4000, help="Sampled patches per split for pixel-level diagnostics.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/dataset_audit"),
        help="Base output directory. A timestamped run folder is created inside.",
    )
    args = p.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"audit_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pcam_summary = analyze_dataset(
        dataset_name="pcam",
        dataset_dir=args.pcam_dir,
        split_order=["train", "valid", "test"],
        sample_n=args.sample_n,
        seed=args.seed,
        out_dir=run_dir,
    )
    cam17_summary = analyze_dataset(
        dataset_name="camelyon17",
        dataset_dir=args.cam17_dir,
        split_order=[s.strip() for s in args.cam17_splits.split(",") if s.strip()],
        sample_n=args.sample_n,
        seed=args.seed + 123,
        out_dir=run_dir,
    )

    dataset_summaries = [pcam_summary, cam17_summary]
    _write_comparison_csv(dataset_summaries, run_dir / "comparison_by_split.csv")

    master = {
        "created_at": datetime.now().isoformat(),
        "sample_n_per_split": args.sample_n,
        "seed": args.seed,
        "pcam_dir": str(args.pcam_dir),
        "cam17_dir": str(args.cam17_dir),
        "outputs": {
            "pcam": str(run_dir / "pcam" / "dataset_summary.json"),
            "camelyon17": str(run_dir / "camelyon17" / "dataset_summary.json"),
            "comparison_csv": str(run_dir / "comparison_by_split.csv"),
        },
        "datasets": dataset_summaries,
    }
    _write_json(run_dir / "master_summary.json", master)
    print(f"[done] Dataset audit saved to: {run_dir}")


if __name__ == "__main__":
    main()
