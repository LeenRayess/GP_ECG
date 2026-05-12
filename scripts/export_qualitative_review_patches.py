"""
Export patch PNGs + CSV roster for manual qualitative error review.

Aligns with docs/qualitative_error_analysis_protocol.md (buckets, n=20, seed 42)
and docs/final_methodology.md Ch.6 / §5.3 (confidence c, entropy H, tau_H = 90th
percentile of H on the evaluated split, threshold 0.5 on calibrated p).

Inputs:
  - test_predictions.npz from evaluate_virchow_preprocessed_test_colab.py
    (expects y_true, prob_after_temperature; optional prob_mc_std)
  - test_x.h5 for preprocessed patches (dataset "x", same row order as eval)
  - optional raw test_x.h5 (same indices) for side-by-side raw vs preprocessed

Example:
  python scripts/export_qualitative_review_patches.py ^
    --npz experiments/virchow_colab/evals_cross_domain/pcam_trained_on_cam17_test/test_predictions.npz ^
    --preprocessed-test-x pcam_data/preprocessed_macenko_benchmark_style/test_x.h5 ^
    --raw-test-x pcam_data/test/camelyonpatch_level_2_split_test_x.h5 ^
    --out-dir reports/qualitative_error_analysis/C2_pcam_to_cam17 ^
    --condition-id C2 ^
    --direction PCam_to_CAMELYON17
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _h5_patch_to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """H x W x 3 -> uint8."""
    x = np.asarray(arr)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"expected HxWx3 patch, got shape {x.shape}")
    if x.dtype == np.uint8:
        return x
    xf = x.astype(np.float32)
    if xf.max() <= 1.0 + 1e-6:
        xf = np.clip(xf * 255.0, 0.0, 255.0)
    else:
        xf = np.clip(xf, 0.0, 255.0)
    return np.round(xf).astype(np.uint8)


def _read_x_row(h5_path: Path, idx: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "x" not in f:
            raise KeyError(f'{h5_path}: no dataset "x"')
        return np.array(f["x"][idx])


def _save_patch_png(arr: np.ndarray, out_path: Path, resize: Optional[int]) -> None:
    rgb = _h5_patch_to_uint8_rgb(arr)
    im = Image.fromarray(rgb, mode="RGB")
    if resize is not None and (im.size[0] != resize or im.size[1] != resize):
        im = im.resize((resize, resize), Image.Resampling.BICUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def _sample_mask_indices(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx
    if idx.size <= n:
        return idx
    return rng.choice(idx, size=n, replace=False)


def export_qualitative_review(
    npz_path: Path,
    preprocessed_test_x: Path,
    out_dir: Path,
    *,
    raw_test_x: Optional[Path] = None,
    condition_id: str = "",
    direction: str = "",
    seed: int = 42,
    n_per_bucket: int = 20,
    png_size: int = 224,
    include_confident_errors: bool = False,
) -> Path:
    """Sample buckets, write PNGs, CSVs, gallery.html. Returns *out_dir*."""
    npz_path = Path(npz_path)
    pre_x = Path(preprocessed_test_x)
    raw_x = Path(raw_test_x) if raw_test_x is not None else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_root = out_dir / "figures"

    data = np.load(npz_path, allow_pickle=True)
    if "y_true" not in data or "prob_after_temperature" not in data:
        raise ValueError(
            f"NPZ missing keys: have {sorted(data.files)}; need at least y_true, prob_after_temperature"
        )
    y = data["y_true"].reshape(-1).astype(np.int64)
    p = data["prob_after_temperature"].reshape(-1).astype(np.float64)
    n = y.size
    if p.size != n:
        raise ValueError("y_true and prob_after_temperature length mismatch")

    y_hat = (p >= 0.5).astype(np.int64)
    c = np.maximum(p, 1.0 - p)
    H = _binary_entropy(p)
    tau_H = float(np.percentile(H, 90))

    mc_std = None
    if "prob_mc_std" in data.files:
        mc_std = data["prob_mc_std"].reshape(-1).astype(np.float64)

    fp = (y_hat == 1) & (y == 0)
    fn = (y_hat == 0) & (y == 1)
    he_err = (y_hat != y) & (H >= tau_H)
    he_ok = (y_hat == y) & (H >= tau_H)
    ce = (y_hat != y) & (c >= 0.9)

    rng = np.random.default_rng(seed)

    plan: List[Tuple[str, np.ndarray, int]] = [
        ("FP", fp, 1001),
        ("FN", fn, 1002),
        ("high_entropy_error", he_err, 1003),
        ("high_entropy_correct", he_ok, 1004),
    ]
    if include_confident_errors:
        plan.append(("confident_error", ce, 1005))

    selected: List[Dict[str, object]] = []
    summary_rows = []

    for bucket_name, mask, seed_off in plan:
        sub = np.random.default_rng(seed + seed_off)
        idxs = _sample_mask_indices(mask, n_per_bucket, sub)
        avail = int(np.sum(mask))
        taken = len(idxs)
        summary_rows.append(
            {
                "bucket": bucket_name,
                "available_n": avail,
                "sampled_n": taken,
                "shortfall": max(0, n_per_bucket - taken),
            }
        )
        for j, i in enumerate(idxs):
            i = int(i)
            case_id = f"{bucket_name}_{j:03d}"
            pre_arr = _read_x_row(pre_x, i)
            resize = None if png_size <= 0 else int(png_size)
            rel_pre = fig_root / bucket_name / f"{case_id}_preprocessed.png"
            _save_patch_png(pre_arr, rel_pre, resize)

            rel_raw_path = ""
            if raw_x is not None:
                raw_arr = _read_x_row(raw_x, i)
                rel_raw = fig_root / bucket_name / f"{case_id}_raw.png"
                _save_patch_png(raw_arr, rel_raw, resize)
                rel_raw_path = str(rel_raw)

            row = {
                "case_id": case_id,
                "h5_index": i,
                "bucket": bucket_name,
                "y_true": int(y[i]),
                "y_hat": int(y_hat[i]),
                "p_cal": float(p[i]),
                "confidence": float(c[i]),
                "entropy": float(H[i]),
                "tau_H_90pct_split": tau_H,
                "figure_preprocessed": str(rel_pre),
                "figure_raw": rel_raw_path,
                "condition_id": condition_id,
                "direction": direction,
                "npz_source": str(npz_path.resolve()),
            }
            if mc_std is not None:
                row["prob_mc_std"] = float(mc_std[i])
            selected.append(row)

    csv_path = out_dir / "selected_cases.csv"
    if selected:
        keys = list(selected[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(selected)
    else:
        csv_path.write_text("h5_index,bucket,note\n", encoding="utf-8")

    with open(out_dir / "bucket_sampling_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tau_H": tau_H,
                "n_total": n,
                "buckets": summary_rows,
                "threshold": 0.5,
                "probability_used": "prob_after_temperature (calibrated)",
            },
            f,
            indent=2,
        )

    checklist = [
        "tissue_scarcity",
        "artifact_burden",
        "borderline_morphology",
        "small_focus_lesion",
        "color_stain_atypia",
        "patch_context_limit",
        "free_text_note",
    ]
    review_path = out_dir / "review_labels_template.csv"
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["case_id", "h5_index", "bucket", "review_order"]
            + [f"chk_{c}_Present_Absent_Unclear" for c in checklist[:-1]]
            + ["free_text_note"]
        )
        order = list(range(len(selected)))
        rng.shuffle(order)
        for ro, k in enumerate(order):
            r = selected[k]
            w.writerow([r["case_id"], r["h5_index"], r["bucket"], ro + 1] + [""] * len(checklist))

    _write_index_html(out_dir, selected, condition_id, direction)

    readme = out_dir / "README_REVIEW.txt"
    readme.write_text(
        "\n".join(
            [
                "1) Open gallery.html in a browser (images + captions).",
                "2) Open review_labels_template.csv in Excel/Sheets.",
                "3) For each case_id, mark checklist items Present / Absent / Unclear",
                "   (exact rubric + definitions: docs/qualitative_error_analysis_protocol.md section 6).",
                "4) Review order: use column review_order (randomized within export).",
                "5) After finishing, save as review_labels_completed.csv in this folder.",
                "",
                f"Condition ID (for thesis): {condition_id or '(set condition_id)'}",
                f"Direction label: {direction or '(set direction)'}",
                "",
                "If raw images are missing, only preprocessed PNGs were written; pass raw_test_x",
                "with test_x.h5 whose rows align with the same indices as preprocessed test.",
            ]
        ),
        encoding="utf-8",
    )

    print("Wrote:", csv_path)
    print("Wrote:", review_path)
    print("Wrote:", out_dir / "gallery.html")
    print("Wrote:", readme)
    print("Wrote:", out_dir / "bucket_sampling_summary.json")
    return out_dir


def _write_index_html(
    out_dir: Path,
    rows: List[Dict[str, object]],
    condition_id: str,
    direction: str,
) -> None:
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        buckets.setdefault(str(r["bucket"]), []).append(r)

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Qualitative review</title>",
        "<style>body{font-family:sans-serif;margin:16px;} .b{margin-top:28px;} img{border:1px solid #ccc;margin:4px;} .cap{font-size:12px;max-width:520px;}</style>",
        "</head><body>",
        f"<h1>Qualitative error review</h1><p><b>Condition</b>: {condition_id} &nbsp; <b>Direction</b>: {direction}</p>",
        "<p>Open <code>selected_cases.csv</code> and <code>review_labels_template.csv</code> while viewing images. "
        "Checklist wording: <code>docs/qualitative_error_analysis_protocol.md</code> §6.</p>",
    ]
    for bname, items in sorted(buckets.items()):
        parts.append(f"<div class='b'><h2>{bname}</h2>")
        for r in items:
            rel = str(Path(r["figure_preprocessed"]).relative_to(out_dir)).replace("\\", "/")
            raw_html = ""
            if r.get("figure_raw"):
                relr = str(Path(r["figure_raw"]).relative_to(out_dir)).replace("\\", "/")
                raw_html = f"<br/><img src='{relr}' width='224'/><div class='cap'>raw</div>"
            parts.append(
                f"<div style='display:inline-block;vertical-align:top;margin:8px;'>"
                f"<img src='{rel}' width='224'/>{raw_html}"
                f"<div class='cap'>idx={r['h5_index']} y={r['y_true']} pred={r['y_hat']} "
                f"p={float(r['p_cal']):.3f} c={float(r['confidence']):.3f} H={float(r['entropy']):.3f}"
                f"</div></div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    p = out_dir / "gallery.html"
    p.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export patches for qualitative error analysis.")
    ap.add_argument("--npz", type=str, required=True, help="test_predictions.npz from Virchow eval")
    ap.add_argument("--preprocessed-test-x", type=str, required=True, help="test_x.h5 (same order as NPZ)")
    ap.add_argument("--raw-test-x", type=str, default="", help="optional raw test_x.h5 (same indices)")
    ap.add_argument("--out-dir", type=str, required=True, help="e.g. reports/qualitative_error_analysis/C2_...")
    ap.add_argument("--condition-id", type=str, default="", help="e.g. C1..C4")
    ap.add_argument("--direction", type=str, default="", help="e.g. PCam_to_CAMELYON17")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-bucket", type=int, default=20, help="target samples per bucket (protocol: 20)")
    ap.add_argument("--png-size", type=int, default=224, help="export square size (0 = native H5 resolution)")
    ap.add_argument("--include-confident-errors", action="store_true", help="extra bucket CE (methodology §6)")
    args = ap.parse_args()

    raw = Path(args.raw_test_x) if args.raw_test_x.strip() else None
    export_qualitative_review(
        Path(args.npz),
        Path(args.preprocessed_test_x),
        Path(args.out_dir),
        raw_test_x=raw,
        condition_id=args.condition_id,
        direction=args.direction,
        seed=args.seed,
        n_per_bucket=args.n_per_bucket,
        png_size=args.png_size,
        include_confident_errors=args.include_confident_errors,
    )


if __name__ == "__main__":
    main()
