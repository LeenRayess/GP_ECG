"""
Compare stain-related statistics on two preprocessed runs using the SAME definitions.

No re-preprocessing: reads existing *_x.h5 and preprocess_report.json from each folder.

For each run we report mean L1 distance between each patch's mean RGB (in [0,1]) and fixed
reference vectors. References are taken from each run's report (old ref vs new primary ref),
then BOTH runs are scored against BOTH refs so numbers are directly comparable.

Usage (from repo root):
  python scripts/compare_preprocessed_stain_metrics.py \\
    --old-dir pcam_data/preprocessed --new-dir pcam_data/preprocessed_multi_ref --split train
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import h5py
import numpy as np


def load_ref_rgb(report_path):
    with open(report_path, encoding="utf-8") as f:
        r = json.load(f)
    cfg = r["config"]
    ref = np.array(cfg["ref_mean_rgb"], dtype=np.float64)
    idx = cfg.get("ref_train_idx")
    return ref, idx


def sample_indices(n_total, max_samples, seed):
    rng = np.random.RandomState(seed)
    m = min(max_samples, n_total)
    return np.sort(rng.choice(n_total, size=m, replace=False))


def mean_rgb_batch(x_ds, positions, chunk=512):
    """x_ds: h5 dataset (n,96,96,3). Return (len(positions), 3) float64 means."""
    out = np.empty((len(positions), 3), dtype=np.float64)
    for a in range(0, len(positions), chunk):
        b = min(a + chunk, len(positions))
        idx = positions[a:b]
        batch = np.array(x_ds[idx], dtype=np.float64)
        out[a:b] = batch.mean(axis=(1, 2))
    return out


def mean_l1_to_refs(means, refs):
    """
    means: (N, 3)
    refs: list of (3,) arrays
    Returns dict name -> mean L1 distance
    """
    out = {}
    for i, ref in enumerate(refs):
        ref = ref.reshape(1, 3)
        d = np.abs(means - ref).sum(axis=1)
        out[i] = float(d.mean())
    return out


def main():
    ap = argparse.ArgumentParser(description="Comparable stain metrics from two preprocessed H5 dirs.")
    ap.add_argument("--old-dir", type=str, default="pcam_data/preprocessed")
    ap.add_argument("--new-dir", type=str, default="pcam_data/preprocessed_multi_ref")
    ap.add_argument("--split", type=str, default="train", choices=("train", "valid", "test"))
    ap.add_argument("--max-samples", type=int, default=20000, help="Random patches per run (same indices if same n).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    old_dir = os.path.abspath(args.old_dir)
    new_dir = os.path.abspath(args.new_dir)
    split = args.split
    x_name = "{}_x.h5".format(split)
    old_x = os.path.join(old_dir, x_name)
    new_x = os.path.join(new_dir, x_name)
    old_rep = os.path.join(old_dir, "preprocess_report.json")
    new_rep = os.path.join(new_dir, "preprocess_report.json")

    for p in (old_x, new_x, old_rep, new_rep):
        if not os.path.isfile(p):
            print("Missing:", p, file=sys.stderr)
            sys.exit(1)

    ref_old, idx_old = load_ref_rgb(old_rep)
    ref_new, idx_new = load_ref_rgb(new_rep)

    with h5py.File(old_x, "r") as fo, h5py.File(new_x, "r") as fn:
        no, nn = fo["x"].shape[0], fn["x"].shape[0]
        if no != nn:
            print("Error: different patch counts {} vs {} - not comparable.".format(no, nn), file=sys.stderr)
            sys.exit(1)
        n = no
        pos = sample_indices(n, args.max_samples, args.seed)
        print("Split: {} | n_in_h5={} | sampled={} (seed={})".format(split, n, len(pos), args.seed))
        print("Ref 'old' (from {} report): idx={} RGB={}".format(args.old_dir, idx_old, ref_old.tolist()))
        print("Ref 'new' (from {} report): idx={} RGB={}".format(args.new_dir, idx_new, ref_new.tolist()))
        print()

        means_old = mean_rgb_batch(fo["x"], pos)
        means_new = mean_rgb_batch(fn["x"], pos)

    refs = [ref_old, ref_new]
    labels = ["ref_old_report", "ref_new_report"]

    l1_old = mean_l1_to_refs(means_old, refs)
    l1_new = mean_l1_to_refs(means_new, refs)

    std_old = means_old.std(axis=0)
    std_new = means_new.std(axis=0)

    print("Mean L1( patch_mean_RGB, ref ) - lower = closer to that ref in mean RGB")
    print("-" * 72)
    print("{:22} {:>18} {:>18}".format("metric", "old_preprocessed", "new_preprocessed"))
    for j, name in enumerate(labels):
        print(
            "{:22} {:>18.4f} {:>18.4f}".format(
                "L1 vs " + name, l1_old[j], l1_new[j]
            )
        )

    print()
    print("Spread of patch mean RGB over the same sample (std per channel)")
    print("-" * 72)
    print("{:8} {:>12} {:>12} {:>12}   {:>12} {:>12} {:>12}".format("", "std_R_old", "std_G_old", "std_B_old", "std_R_new", "std_G_new", "std_B_new"))
    print(
        "         {:12.4f} {:12.4f} {:12.4f}   {:12.4f} {:12.4f} {:12.4f}".format(
            std_old[0], std_old[1], std_old[2], std_new[0], std_new[1], std_new[2]
        )
    )

    delta = np.abs(means_old - means_new).sum(axis=1)
    print()
    print("Paired same-index patches: mean L1 between old and new patch mean RGB: {:.4f}".format(float(delta.mean())))
    print("(Large value = runs disagree strongly on average color center.)")


if __name__ == "__main__":
    main()
