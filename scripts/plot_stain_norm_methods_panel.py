#!/usr/bin/env python3
"""
One PCam training patch → Macenko / Reinhard / Vahadane panel (same reference as stain benchmark).

Workflow (efficient):
  1. Fit Macenko, Reinhard, and Vahadane on **only** the reference tile at
     `reference_train_index` from `stain_reference.json` (same as `prepare_stain_benchmark_h5.py`).
     No other patches are normalized at this step.
  2. Walk the **training** H5 in order from `--start`, convert each row to [0,1], run QC; the
     **first** patch that passes is transformed three ways and plotted.

Run from repo root:
  python scripts/plot_stain_norm_methods_panel.py

Requires: h5py, numpy, matplotlib, scipy, staintools.

Display defaults reduce blockiness: patches are upsampled with bicubic (``--upsample``, default 4)
for the figure only; PCam remains 96×96 in the H5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import preprocess_histopath_h5 as pp
from prepare_stain_benchmark_h5 import _classical_transform, _fit_single_ref
from staintools.preprocessing.luminosity_standardizer import LuminosityStandardizer

# Pipeline defaults (mirror preprocess_histopath_h5.passes_quality gates)
_DEFAULT_SOLID = pp.SOLID_COLOR_STD
_DEFAULT_BLACK = pp.HIGH_BLACK_RATIO
_DEFAULT_TISSUE = pp.LOW_TISSUE_THRESHOLD

# Stricter: more texture, less black, more tissue required
_STRICT_SOLID = 0.06
_STRICT_BLACK = 0.35
_STRICT_TISSUE = 0.45


def passes_quality_custom(
    patch_01: np.ndarray,
    solid_std: float,
    high_black_ratio: float,
    low_tissue_threshold: float,
) -> Tuple[bool, Optional[str]]:
    """Same logic as preprocess_histopath_h5.passes_quality, with tunable thresholds."""
    gray = patch_01.mean(axis=2)
    gray_std = float(np.std(gray))
    if gray_std < solid_std:
        return False, "solid_color"
    n_elems = patch_01.size
    ratio_black = float(np.sum(patch_01 <= 0.1) / n_elems)
    if ratio_black >= high_black_ratio:
        return False, "high_black"
    tissue = pp.tissue_pct_final(patch_01)
    if tissue < low_tissue_threshold:
        return False, "low_tissue"
    return True, None


def first_passing_index_sequential(
    x_path: Path,
    start: int,
    max_scan: int,
    solid_std: float,
    high_black_ratio: float,
    low_tissue_threshold: float,
) -> Tuple[int, np.ndarray]:
    """First index i in [start, start+max_scan) with i < n that passes QC; returns (idx, p01)."""
    with h5py.File(x_path, "r") as f:
        n = int(f["x"].shape[0])
        x = f["x"]
        if start < 0 or start >= n:
            raise ValueError(f"--start {start} out of range [0, {n})")
        for offset in range(max_scan):
            idx = start + offset
            if idx >= n:
                break
            p01 = pp.to_01(np.asarray(x[idx]))
            ok, _ = passes_quality_custom(p01, solid_std, high_black_ratio, low_tissue_threshold)
            if ok:
                return int(idx), p01
    raise RuntimeError(
        f"No patch passed QC in indices [{start}, {min(start + max_scan, n)}) "
        f"(n={n}). Loosen thresholds, increase --max-scan, or lower --start."
    )


def main() -> None:
    root = _SCRIPT_DIR.parent
    p = argparse.ArgumentParser(
        description="Sequential QC scan → first pass; ref-fit Macenko/Reinhard/Vahadane panel."
    )
    p.add_argument(
        "--train-x",
        type=Path,
        default=root / "pcam_data" / "training" / "camelyonpatch_level_2_split_train_x.h5",
        help="PCam training images H5.",
    )
    p.add_argument(
        "--ref-config",
        type=Path,
        default=root / "experiments" / "stain_reference" / "stain_reference.json",
        help="JSON with reference_train_index.",
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Use this training index (must pass QC). If set, no sequential scan.",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="First training index to consider when scanning (default 0).",
    )
    p.add_argument(
        "--max-scan",
        type=int,
        default=200_000,
        help="Max indices to try in order from --start (exclusive upper bound by count).",
    )
    p.add_argument(
        "--strict-qc",
        action="store_true",
        help="Stricter QC than preprocessing defaults (more texture, less black, more tissue).",
    )
    p.add_argument(
        "--qc-solid-std",
        type=float,
        default=None,
        help="Reject if gray std below this (default: pipeline 0.04, or strict 0.06 with --strict-qc).",
    )
    p.add_argument(
        "--qc-high-black",
        type=float,
        default=None,
        help="Reject if fraction of pixels <=0.1 exceeds this (default 0.5, strict 0.35).",
    )
    p.add_argument(
        "--qc-low-tissue",
        type=float,
        default=None,
        help="Reject if tissue fraction below this (default 0.35, strict 0.45).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "reports" / "figures" / "theoretical_stain_norm_macenko_reinhard_vahadane_panel.png",
        help="Output .png (PDF written alongside).",
    )
    p.add_argument(
        "--upsample",
        type=int,
        default=4,
        metavar="N",
        help="Integer edge upscale for display only (bicubic via scipy.zoom). Native PCam is 96×96; "
        "N=4 → 384×384 per panel. Set 1 for raw pixels (blocky on screen).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for PNG/PDF (default 300).",
    )
    p.add_argument(
        "--fig-width",
        type=float,
        default=14.0,
        help="Figure width in inches (height scales with one row).",
    )
    args = p.parse_args()

    if args.strict_qc:
        solid, black, tissue = _STRICT_SOLID, _STRICT_BLACK, _STRICT_TISSUE
    else:
        solid, black, tissue = _DEFAULT_SOLID, _DEFAULT_BLACK, _DEFAULT_TISSUE
    if args.qc_solid_std is not None:
        solid = args.qc_solid_std
    if args.qc_high_black is not None:
        black = args.qc_high_black
    if args.qc_low_tissue is not None:
        tissue = args.qc_low_tissue

    if not args.train_x.exists():
        raise FileNotFoundError(f"Missing training H5: {args.train_x}")
    if not args.ref_config.exists():
        raise FileNotFoundError(
            f"Missing {args.ref_config}. Create it or pass --ref-config "
            "(see notebooks/stain_reference_selection.ipynb)."
        )

    # 1) Fit normalizers on reference tile only (no full-corpus work).
    mac, rei, vah = _fit_single_ref(args.train_x, args.ref_config)
    with open(args.ref_config, "r", encoding="utf-8") as f:
        ref_idx = int(json.load(f)["reference_train_index"])

    # 2) Pick one candidate patch (sequential QC unless --index).
    if args.index is not None:
        with h5py.File(args.train_x, "r") as f:
            n_total = int(f["x"].shape[0])
            idx = int(args.index)
            if idx < 0 or idx >= n_total:
                raise ValueError(f"--index {idx} out of range [0, {n_total})")
            p01 = pp.to_01(np.asarray(f["x"][idx]))
        ok, reason = passes_quality_custom(p01, solid, black, tissue)
        if not ok:
            raise ValueError(f"Index {idx} fails QC ({reason}) under current thresholds.")
    else:
        idx, p01 = first_passing_index_sequential(
            args.train_x, args.start, args.max_scan, solid, black, tissue
        )

    lum = LuminosityStandardizer()
    tfm_mac = _classical_transform(mac, lum.standardize)
    tfm_rei = _classical_transform(rei, lum.standardize)
    tfm_vah = _classical_transform(vah, lum.standardize)

    raw = np.clip(p01, 0.0, 1.0)
    out_mac = tfm_mac(p01)
    out_rei = tfm_rei(p01)
    out_vah = tfm_vah(p01)

    panels = [
        ("Original", raw),
        ("Macenko", out_mac),
        ("Reinhard", out_rei),
        ("Vahadane", out_vah),
    ]

    def _for_display(im: np.ndarray) -> np.ndarray:
        """Upsample RGB [0,1] for publication display (does not change stored patch resolution)."""
        im = np.clip(im.astype(np.float64), 0.0, 1.0)
        n = int(args.upsample)
        if n <= 1:
            return im.astype(np.float32)
        return np.clip(zoom(im, (n, n, 1), order=3).astype(np.float32), 0.0, 1.0)

    h_in = float(args.fig_width) / 4.0 + 0.55
    fig, axes = plt.subplots(1, 4, figsize=(float(args.fig_width), h_in), dpi=int(args.dpi))
    for ax, (title, im) in zip(axes, panels):
        disp = _for_display(im)
        ax.imshow(disp, interpolation="bilinear")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_kw = dict(bbox_inches="tight", pad_inches=0.08, dpi=int(args.dpi))
    fig.savefig(args.out, **save_kw)
    pdf_path = args.out.with_suffix(".pdf")
    fig.savefig(pdf_path, **save_kw)
    plt.close(fig)
    print("Wrote", args.out)
    print("Wrote", pdf_path)
    print(
        f"Patch index {idx} (ref tile for normalizers: {ref_idx}); "
        f"display upsample={args.upsample}×, dpi={args.dpi}."
    )


if __name__ == "__main__":
    main()
