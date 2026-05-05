"""Histopath patch H5 preprocessing (PatchCamelyon + WILDS CAMELYON17): quality filter, stain norm, [0,1] H5.

Reads layout under --data-dir: PCam (training/val/test camelyonpatch_*.h5) or WILDS
(--layout wilds: train_x.h5, valid_x.h5, test_x.h5 + _y). Optional --dedup-dir (PCam only).

Stain normalizers are fit on the patch at reference_train_index from --ref-config, read from
--reference-train-x-h5 when set (use PCam train H5 for PCam-aligned stain on WILDS); otherwise from
this dataset's train_x.h5. Outputs under data_dir/<preprocessed_subdir>/: *_x.h5, *_y.h5,
*_normalizer_used.npy, *_purple_fallback.npy, *_qa_samples.npz, manifest.json, preprocess_report.json."""

from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import h5py
from scipy.ndimage import uniform_filter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, desc=None, **kwargs):
        return x

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pcam-master"))

SOLID_COLOR_STD = 0.04
HIGH_BLACK_RATIO = 0.5
LOW_TISSUE_THRESHOLD = 0.35
PURPLE_PERCENTILE = 2.0
BLUE_DOM_PERCENTILE = 25.0
GUARDRAIL_MEAN_R_MIN = 0.08
GUARDRAIL_PINK_PCT_MIN = 0.01
GUARDRAIL_MEAN_SAT_MIN = 0.03
QA_SAMPLES_PER_SPLIT = 60
CLASS_BALANCE_ALERT_ABS_DIFF = 0.03
SAT_THRESHOLD = 0.12
VAR_THRESHOLD = 0.003
EDGE_LO, EDGE_HI = 0.12, 0.45
LOCAL_VAR_WIN = 11

CHUNK_SIZE = 5000

MULTI_REF_INDICES = [237219, 240880, 162772]
MULTI_REF_MERGE_PATCH_IDX = 137467
MULTI_REF_K_CLUSTERS = 4
MULTI_REF_FIT_SAMPLE = 20000


def otsu_threshold_01(gray):
    """Otsu threshold for [0,1] float grayscale. Returns threshold in [0,1]."""
    flat = np.asarray(gray).ravel().astype(np.float64)
    hist, _ = np.histogram(flat, bins=256, range=(0, 1))
    bin_centers = (np.arange(256) + 0.5) / 256.0
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    sigma_b_sq_max = -1.0
    best_t = 0.5
    for t in range(1, 255):
        w0, w1 = p[:t].sum(), p[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (p[:t] * bin_centers[:t]).sum() / w0
        mu1 = (p[t:] * bin_centers[t:]).sum() / w1
        sigma_b_sq = w0 * w1 * (mu0 - mu1) ** 2
        if sigma_b_sq > sigma_b_sq_max:
            sigma_b_sq_max = sigma_b_sq
            best_t = bin_centers[t]
    return best_t


def saturation_from_rgb(rgb):
    """RGB [0,1], return S per pixel (HSV saturation)."""
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    return np.where(mx > 1e-8, (mx - mn) / np.maximum(mx, 1e-8), 0.0)


def tissue_pct_final(patch_01):
    """Combined mix: saturation OR local variance; Otsu on edge cases. Returns tissue fraction [0,1]."""
    gray = np.clip(patch_01.mean(axis=2).astype(np.float64), 0, 1)
    n_px = gray.size
    sat = saturation_from_rgb(patch_01)
    tissue_pct_sat = float(np.sum(sat > SAT_THRESHOLD) / n_px)
    var_map = uniform_filter(gray ** 2, size=LOCAL_VAR_WIN, mode="nearest") - \
              uniform_filter(gray, size=LOCAL_VAR_WIN, mode="nearest") ** 2
    tissue_pct_var = float(np.sum(var_map > VAR_THRESHOLD) / n_px)
    tissue_pct_combined = float(np.sum((sat > SAT_THRESHOLD) | (var_map > VAR_THRESHOLD)) / n_px)
    if EDGE_LO <= tissue_pct_combined <= EDGE_HI:
        p5, p95 = np.percentile(gray, 5), np.percentile(gray, 95)
        span = max(p95 - p5, 1e-8)
        g = np.clip((gray - p5) / span, 0, 1)
        t = otsu_threshold_01(g)
        otsu_norm = float(np.sum(g < t) / n_px)
        return max(tissue_pct_combined, otsu_norm)
    return tissue_pct_combined


def passes_quality(patch_01):
    """(passed, reason). reason is first failure: 'solid_color', 'high_black', 'low_tissue', or None if passed."""
    gray = patch_01.mean(axis=2)
    gray_std = float(np.std(gray))
    if gray_std < SOLID_COLOR_STD:
        return False, "solid_color"
    n_elems = patch_01.size
    ratio_black = float(np.sum(patch_01 <= 0.1) / n_elems)
    if ratio_black >= HIGH_BLACK_RATIO:
        return False, "high_black"
    tissue = tissue_pct_final(patch_01)
    if tissue < LOW_TISSUE_THRESHOLD:
        return False, "low_tissue"
    return True, None


def _routing_feature_row(p01):
    """Seven features for KMeans multi-reference routing."""
    m = p01.mean(axis=(0, 1)).astype(np.float64)
    tissue = tissue_pct_final(p01)
    blue_dom = float((p01[:, :, 2] > p01[:, :, 0]).mean())
    pink_pct = float((p01[:, :, 0] > p01[:, :, 2]).mean())
    mean_sat = float(saturation_from_rgb(p01).mean())
    return np.array([tissue, m[0], m[1], m[2], blue_dom, pink_pct, mean_sat], dtype=np.float64)


def to_uint8(p):
    """Convert patch to uint8. p: [0,1] or [0,255]."""
    if p.max() > 1.0:
        return np.clip(p, 0, 255).astype(np.uint8)
    return (np.clip(p, 0, 1) * 255).astype(np.uint8)


def to_01(p):
    """Convert patch to float [0,1]. p: uint8 [0,255] or float [0,1]."""
    a = p.astype(np.float64)
    if p.max() > 1.0:
        return np.clip(a / 255.0, 0.0, 1.0)
    return np.clip(a, 0.0, 1.0)


def _mean_r_pink_pct(patch_01):
    """Return (mean_r, pink_pct) for a patch (float [0,1]). Used for percentile-based purple detection."""
    mean_r = float(patch_01[:, :, 0].mean())
    pink_pct = float((patch_01[:, :, 0] > patch_01[:, :, 2]).mean())
    return mean_r, pink_pct


def _mean_saturation(patch_01):
    """Mean HSV-like saturation in [0,1] for guardrail checks."""
    mx = patch_01.max(axis=2)
    mn = patch_01.min(axis=2)
    sat = np.zeros_like(mx, dtype=np.float64)
    valid = (mx > 1e-8) & np.isfinite(mx)
    np.divide(mx - mn, mx, out=sat, where=valid)
    return float(sat.mean())


def _fails_post_norm_guardrails(patch_01):
    """Detect implausible stain-normalized outputs and trigger luminosity-only fallback."""
    mean_r, pink_pct = _mean_r_pink_pct(patch_01)
    mean_sat = _mean_saturation(patch_01)
    return (
        (mean_r < GUARDRAIL_MEAN_R_MIN) or
        (pink_pct < GUARDRAIL_PINK_PCT_MIN) or
        (mean_sat < GUARDRAIL_MEAN_SAT_MIN)
    )


def _luminosity_only_norm(patch_01, luminosity_standardizer):
    """Return patch normalized with luminosity standardization only (no Macenko/Reinhard), float32 [0,1]."""
    pu8 = to_uint8(patch_01)
    try:
        pstd = luminosity_standardizer(pu8)
    except Exception:
        pstd = pu8
    return np.clip(pstd.astype(np.float32) / 255.0, 0.0, 1.0)


def compute_blue_dom_threshold(x_path, kept_indices, percentile=25.0):
    """Per-split percentile of (B>R) fraction for Macenko vs Reinhard routing."""
    kept = np.asarray(kept_indices)
    n_kept = len(kept)
    n_sample = min(30000, n_kept)
    sample_pos = np.random.RandomState(42).choice(n_kept, size=n_sample, replace=False)
    indices_to_sample = kept[sample_pos]
    blue_dom_list = []
    for start in range(0, len(indices_to_sample), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(indices_to_sample))
        batch_idx = np.asarray(indices_to_sample[start:end])
        order = np.argsort(batch_idx, kind="mergesort")
        sorted_idx = batch_idx[order]
        with h5py.File(x_path, "r") as f:
            chunk_sorted = np.array(f["x"][sorted_idx])
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        chunk = chunk_sorted[inv]
        for i in range(chunk.shape[0]):
            p = chunk[i]
            p01 = to_01(p)
            blue_dom_list.append((p01[:, :, 2] > p01[:, :, 0]).mean())
    return float(np.percentile(blue_dom_list, percentile))


def get_normalizers_and_threshold(
    data_dir,
    train_x_path,
    ref_config_path,
    train_candidate_indices=None,
    reference_train_x_path=None,
):
    """Fit Macenko/Reinhard from stain_reference.json index.

    reference_train_x_path: if set, load x[reference_train_index] from this H5 (e.g. PCam train_x)
    so WILDS patches are normalized with the same fitted normalizers as PCam. If None, use train_x_path.
    """
    from staintools import StainNormalizer, ReinhardColorNormalizer
    from staintools.preprocessing.luminosity_standardizer import LuminosityStandardizer

    with open(ref_config_path) as f:
        ref_config = json.load(f)
    ref_idx = ref_config["reference_train_index"]

    ref_h5 = reference_train_x_path if reference_train_x_path else train_x_path
    with h5py.File(ref_h5, "r") as f:
        n_ref = int(f["x"].shape[0])
        if ref_idx < 0 or ref_idx >= n_ref:
            raise IndexError(
                "reference_train_index {} out of range for reference H5 (n={}) {}".format(ref_idx, n_ref, ref_h5)
            )
        ref_patch = np.array(f["x"][ref_idx])
    ref_01 = to_01(ref_patch)
    ref_uint8 = to_uint8(ref_patch)

    ref_fit = LuminosityStandardizer.standardize(ref_uint8.copy())
    macenko = StainNormalizer(method="macenko")
    macenko.fit(ref_fit)
    reinhard = ReinhardColorNormalizer()
    reinhard.fit(ref_uint8)
    ref_mean_rgb = (float(ref_01[:, :, 0].mean()), float(ref_01[:, :, 1].mean()), float(ref_01[:, :, 2].mean()))

    return macenko, reinhard, ref_idx, ref_mean_rgb


def build_ref_packs(train_x_path, ref_indices):
    """Fit Macenko + Reinhard per reference index. Returns list of dicts: macenko, reinhard, ref_mean_rgb, idx."""
    from staintools import StainNormalizer, ReinhardColorNormalizer
    from staintools.preprocessing.luminosity_standardizer import LuminosityStandardizer

    packs = []
    for ref_idx in ref_indices:
        with h5py.File(train_x_path, "r") as f:
            ref_patch = np.array(f["x"][ref_idx])
        ref_01 = to_01(ref_patch)
        ref_uint8 = to_uint8(ref_patch)
        ref_fit = LuminosityStandardizer.standardize(ref_uint8.copy())
        macenko = StainNormalizer(method="macenko")
        macenko.fit(ref_fit)
        reinhard = ReinhardColorNormalizer()
        reinhard.fit(ref_uint8)
        ref_mean_rgb = tuple(float(x) for x in ref_01.mean(axis=(0, 1)))
        packs.append({"idx": int(ref_idx), "macenko": macenko, "reinhard": reinhard, "ref_mean_rgb": ref_mean_rgb})
    return packs


def fit_multi_ref_router(train_x_path, train_candidate_indices=None):
    """KMeans(4) on train features; merge cluster maps to nearest ref by RGB; others via Hungarian assignment."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import linear_sum_assignment

    with h5py.File(train_x_path, "r") as f:
        n_file = f["x"].shape[0]
    cand = np.arange(n_file, dtype=np.int64) if train_candidate_indices is None else np.asarray(train_candidate_indices, dtype=np.int64)
    n_take = min(MULTI_REF_FIT_SAMPLE, len(cand))
    rng = np.random.RandomState(42)
    pick = rng.choice(len(cand), size=n_take, replace=False)
    sample_idx = np.sort(cand[pick])

    rows = []
    with h5py.File(train_x_path, "r") as f:
        x = f["x"]
        for idx in sample_idx:
            p01 = to_01(np.array(x[int(idx)]))
            if not passes_quality(p01)[0]:
                continue
            rows.append(_routing_feature_row(p01))
    if len(rows) < MULTI_REF_K_CLUSTERS:
        raise ValueError("Not enough quality-passed samples for multi-ref KMeans (got {})".format(len(rows)))

    X = np.stack(rows, axis=0)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=MULTI_REF_K_CLUSTERS, random_state=42, n_init=20)
    kmeans.fit(Xz)
    centers = kmeans.cluster_centers_

    with h5py.File(train_x_path, "r") as f:
        p_disc = to_01(np.array(f["x"][MULTI_REF_MERGE_PATCH_IDX]))
    z_disc = scaler.transform(_routing_feature_row(p_disc).reshape(1, -1))[0]
    merge_cluster = int(np.argmin(np.sum((centers - z_disc) ** 2, axis=1)))

    ref_z = []
    for ri in MULTI_REF_INDICES:
        with h5py.File(train_x_path, "r") as f:
            pr = to_01(np.array(f["x"][ri]))
        ref_z.append(scaler.transform(_routing_feature_row(pr).reshape(1, -1))[0])
    ref_z = np.stack(ref_z, axis=0)

    remaining = sorted([c for c in range(MULTI_REF_K_CLUSTERS) if c != merge_cluster])
    C = np.zeros((3, 3), dtype=np.float64)
    for i, c in enumerate(remaining):
        for j in range(3):
            C[i, j] = float(np.linalg.norm(centers[c] - ref_z[j]))
    row_ind, col_ind = linear_sum_assignment(C)
    cluster_to_pack = {}
    for r, c in zip(row_ind, col_ind):
        cluster_to_pack[int(remaining[r])] = int(c)

    print("Multi-ref router: merge_cluster={} (patch idx {}); cluster_to_pack={}".format(
        merge_cluster, MULTI_REF_MERGE_PATCH_IDX, cluster_to_pack))
    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "merge_cluster": merge_cluster,
        "cluster_to_pack": cluster_to_pack,
        "multi_ref_indices": list(MULTI_REF_INDICES),
        "merge_patch_idx": MULTI_REF_MERGE_PATCH_IDX,
    }


def choose_ref_pack_index(p01, router, ref_packs):
    """Return pack index 0..2; merge-cluster patches use closest ref by mean RGB (hue)."""
    z = router["scaler"].transform(_routing_feature_row(p01).reshape(1, -1))
    cl = int(router["kmeans"].predict(z)[0])
    if cl == router["merge_cluster"]:
        m = p01.mean(axis=(0, 1)).astype(np.float64)
        best_j, best_d = 0, 1e9
        for j, pack in enumerate(ref_packs):
            r = np.array(pack["ref_mean_rgb"], dtype=np.float64)
            d = float(np.abs(m - r).sum())
            if d < best_d:
                best_d, best_j = d, j
        return best_j
    return int(router["cluster_to_pack"][cl])


NORM_MACENKO = 0
NORM_REINHARD = 1
NORM_MACENKO_FALLBACK = 2
NORM_REINHARD_FALLBACK = 3
NORM_LUMINOSITY_ONLY = 4

def normalize_patch(patch_01, macenko, reinhard, blue_dom_threshold, luminosity_standardizer, return_after_stain=False, return_which_normalizer=False):
    """Stain normalize then value norm to [0,1]. Returns float32 (96,96,3).
    If return_after_stain: (out, norm_u8). If return_which_normalizer: also return method code (0-4).
    """
    bd = (patch_01[:, :, 2] > patch_01[:, :, 0]).mean()
    use_reinhard = bd < blue_dom_threshold
    pu8 = to_uint8(patch_01)
    try:
        pstd = luminosity_standardizer(pu8)
    except Exception:
        pstd = pu8
    which = NORM_LUMINOSITY_ONLY
    try:
        norm_u8 = reinhard.transform(pstd) if use_reinhard else macenko.transform(pstd)
        which = NORM_REINHARD if use_reinhard else NORM_MACENKO
    except Exception:
        try:
            norm_u8 = macenko.transform(pstd) if use_reinhard else reinhard.transform(pstd)
            which = NORM_MACENKO_FALLBACK if use_reinhard else NORM_REINHARD_FALLBACK
        except Exception:
            norm_u8 = pstd
            which = NORM_LUMINOSITY_ONLY
    out = np.clip(norm_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    if which != NORM_LUMINOSITY_ONLY and _fails_post_norm_guardrails(out):
        out = _luminosity_only_norm(patch_01, luminosity_standardizer)
        norm_u8 = np.clip((out * 255.0), 0, 255).astype(np.uint8)
        which = NORM_LUMINOSITY_ONLY
    if return_after_stain and return_which_normalizer:
        return out, norm_u8, which
    if return_after_stain:
        return out, norm_u8
    if return_which_normalizer:
        return out, which
    return out


def normalize_patch_macenko_benchmark_style(
    patch_01,
    macenko,
    luminosity_standardizer,
    return_after_stain=False,
    return_which_normalizer=False,
):
    """Benchmark-style classical Macenko: luminosity standardize -> Macenko, else luminosity-only."""
    pu8 = to_uint8(patch_01)
    try:
        pstd = luminosity_standardizer(pu8)
    except Exception:
        pstd = pu8
    try:
        norm_u8 = macenko.transform(pstd)
        which = NORM_MACENKO
    except Exception:
        norm_u8 = pstd
        which = NORM_LUMINOSITY_ONLY
    out = np.clip(norm_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    if return_after_stain and return_which_normalizer:
        return out, norm_u8, which
    if return_after_stain:
        return out, norm_u8
    if return_which_normalizer:
        return out, which
    return out


def _l1_dist_mean_rgb(mean_rgb, ref_mean_rgb):
    return sum(abs(a - b) for a, b in zip(mean_rgb, ref_mean_rgb))


def _normalizer_usage_dict(normalizer_used):
    arr = np.array(normalizer_used, dtype=np.uint8)
    n = len(arr)
    if n == 0:
        return {
            "macenko": 0, "reinhard": 0, "macenko_fallback": 0, "reinhard_fallback": 0, "luminosity_only": 0,
            "macenko_rate": 0.0, "reinhard_rate": 0.0, "macenko_fallback_rate": 0.0, "reinhard_fallback_rate": 0.0, "luminosity_only_rate": 0.0,
        }
    counts = {
        "macenko": int((arr == NORM_MACENKO).sum()),
        "reinhard": int((arr == NORM_REINHARD).sum()),
        "macenko_fallback": int((arr == NORM_MACENKO_FALLBACK).sum()),
        "reinhard_fallback": int((arr == NORM_REINHARD_FALLBACK).sum()),
        "luminosity_only": int((arr == NORM_LUMINOSITY_ONLY).sum()),
    }
    counts.update({
        "macenko_rate": float(counts["macenko"] / n),
        "reinhard_rate": float(counts["reinhard"] / n),
        "macenko_fallback_rate": float(counts["macenko_fallback"] / n),
        "reinhard_fallback_rate": float(counts["reinhard_fallback"] / n),
        "luminosity_only_rate": float(counts["luminosity_only"] / n),
    })
    return counts


def _label_stats_for_indices(y_path, indices):
    """Compute positive/negative counts for given indices without loading full y into memory."""
    idx = np.asarray(indices)
    if len(idx) == 0:
        return {"n_positive": 0, "n_negative": 0, "frac_positive": 0.0}
    n_pos = 0
    with h5py.File(y_path, "r") as f:
        y = f["y"]
        for start in range(0, len(idx), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(idx))
            batch_idx = idx[start:end]
            vals = np.array(y[batch_idx]).reshape(-1)
            n_pos += int((vals >= 0.5).sum())
    n = int(len(idx))
    n_neg = int(n - n_pos)
    return {"n_positive": int(n_pos), "n_negative": int(n_neg), "frac_positive": float(n_pos / n) if n else 0.0}


def print_preprocess_status(out_dir):
    """Inspect preprocessed/ state and print what finished / what to do (e.g. after PC restart)."""
    if not os.path.isdir(out_dir):
        print("Output dir does not exist yet: {}".format(out_dir))
        print("  -> Run the script without --status to run the full pipeline.")
        return
    print("Preprocess status for: {}".format(os.path.abspath(out_dir)))
    print("-" * 60)
    splits = ("train", "valid", "test")
    for split in splits:
        ckpt = os.path.join(out_dir, "checkpoint_step1_{}.npz".format(split))
        x_path = os.path.join(out_dir, "{}_x.h5".format(split))
        y_path = os.path.join(out_dir, "{}_y.h5".format(split))
        has_ckpt = os.path.isfile(ckpt)
        n_kept_expected = None
        if has_ckpt:
            try:
                data = np.load(ckpt, allow_pickle=False)
                n_kept_expected = len(data["kept_indices"])
            except Exception as e:
                print("[{}] Step 1 checkpoint present but failed to read: {}".format(split, e))
        n_x = n_y = None
        if os.path.isfile(x_path):
            try:
                with h5py.File(x_path, "r") as f:
                    n_x = f["x"].shape[0]
            except Exception as e:
                print("[{}] {} exists but error reading: {}".format(split, x_path, e))
        if os.path.isfile(y_path):
            try:
                with h5py.File(y_path, "r") as f:
                    n_y = f["y"].shape[0]
            except Exception as e:
                print("[{}] {} exists but error reading: {}".format(split, y_path, e))
        step1_done = has_ckpt and n_kept_expected is not None
        output_ok = (n_x is not None and n_y is not None and n_x == n_y and
                     (n_kept_expected is None or n_x == n_kept_expected))
        if step1_done and output_ok:
            state = "DONE (Step 1 + Step 2-3)"
        elif step1_done and (n_x is None or n_y is None or n_x != n_kept_expected):
            state = "Step 1 done; Step 2-3 incomplete or missing (output n={})".format(n_x if n_x is not None else "?")
        elif not step1_done and (n_x is not None or n_y is not None):
            state = "No Step 1 checkpoint; partial output (n={}) - delete _x.h5/_y.h5 and run without --resume for this split, or run full pipeline".format(n_x or n_y)
        else:
            state = "not started"
        print("[{}] Step 1 checkpoint: {}  |  output x: {}  y: {}  |  {}".format(
            split, "yes" if has_ckpt else "no",
            "n={}".format(n_x) if n_x is not None else "missing",
            "n={}".format(n_y) if n_y is not None else "missing",
            state))
        if n_kept_expected is not None and n_x is not None and n_x != n_kept_expected and n_x > 0:
            print("       -> Partial output ({} vs expected {}). Delete {} and {} then run with --resume to redo Step 2-3 only.".format(
                n_x, n_kept_expected, os.path.basename(x_path), os.path.basename(y_path)))
    print("-" * 60)
    print("Recommendation:")
    all_done = all(
        os.path.isfile(os.path.join(out_dir, "{}_x.h5".format(s))) and
        os.path.isfile(os.path.join(out_dir, "{}_y.h5".format(s)))
        for s in splits)
    if all_done:
        try:
            with h5py.File(os.path.join(out_dir, "train_x.h5"), "r") as f:
                n = f["x"].shape[0]
            print("  All splits have output H5 files. If sizes look correct above, preprocessing is complete.")
            print("  If manifest.json is missing, run once without --status to write it.")
        except Exception:
            pass
    else:
        print("  Run with --resume to continue. Step 1 will be skipped where checkpoint exists;")
        print("  Step 2-3 will be skipped where output H5 already has the expected size.")
        print("  If a split shows partial output, delete that split's _x.h5 and _y.h5 in preprocessed/")
        print("  then run again with --resume so Step 2-3 are redone (Step 1 still skipped).")


def process_split(split_name, x_path, y_path, out_dir, macenko, reinhard, ref_mean_rgb, candidate_indices=None, resume=False,
                   stain_router=None, ref_packs=None, stain_mode="adaptive"):
    """Quality filter pass then stain/value write; optional multi-ref via stain_router + ref_packs."""
    from staintools.preprocessing.luminosity_standardizer import LuminosityStandardizer

    ckpt_step1 = os.path.join(out_dir, "checkpoint_step1_{}.npz".format(split_name))
    out_x = os.path.join(out_dir, "{}_x.h5".format(split_name))
    out_y = os.path.join(out_dir, "{}_y.h5".format(split_name))

    with h5py.File(x_path, "r") as f:
        n_total_file = f["x"].shape[0]

    if candidate_indices is None:
        candidate_indices = np.arange(n_total_file)
    else:
        candidate_indices = np.asarray(candidate_indices)
        if candidate_indices.max() >= n_total_file or candidate_indices.min() < 0:
            raise ValueError("candidate_indices out of range for {} (n_total={})".format(split_name, n_total_file))
    n_total = len(candidate_indices)
    class_balance_before = _label_stats_for_indices(y_path, candidate_indices)

    if resume and os.path.isfile(ckpt_step1):
        data = np.load(ckpt_step1, allow_pickle=False)
        kept = np.array(data["kept_indices"])
        n_solid = int(data["n_solid"])
        n_black = int(data["n_black"])
        n_low_tissue = int(data["n_low_tissue"])
        n_total = int(data["n_total"])
        n_kept = len(kept)
        if os.path.isfile(out_x) and os.path.isfile(out_y):
            with h5py.File(out_x, "r") as f:
                if f["x"].shape[0] == n_kept:
                    print("[{}] Resuming: Step 1 (from checkpoint), Step 2-3 (already done, skipping).".format(split_name))
                    with h5py.File(out_y, "r") as fy:
                        y_arr = np.array(fy["y"][:])
                    n_pos = int((y_arr >= 0.5).sum())
                    n_neg = int((y_arr < 0.5).sum())
                    return {
                        "kept_indices": kept.tolist(), "n_kept": n_kept, "n_total": n_total,
                        "class_balance": {"n_positive": n_pos, "n_negative": n_neg, "frac_positive": n_pos / n_kept if n_kept else 0},
                        "class_balance_before": class_balance_before,
                        "step1_removed": {"solid_color": n_solid, "high_black": n_black, "low_tissue": n_low_tissue},
                        "step2_blue_dom_threshold": None,
                        "step2_purple_fallback_count": None,
                        "step2_sample_dist_before": None, "step2_sample_dist_after": None,
                        "step2_sample_mean_r_before": None, "step2_sample_mean_r_after": None,
                        "step2_sample_pink_pct_before": None, "step2_sample_pink_pct_after": None,
                        "step2_sample_purple_only_after": None, "step2_sample_std_mean_rgb_after": None,
                        "step2_sample_outlier_rate_after": None, "step2_sample_n": None,
                        "step3_sample_min_max_mean": None,
                    }
        print("[{}] Resuming: Step 1 (from checkpoint), Step 2-3 ...".format(split_name))
    else:
        print("[{}] Step 1: Quality filter (over {} candidates) ...".format(split_name, n_total))
        kept = []
        n_solid = n_black = n_low_tissue = 0
        for start in tqdm(range(0, n_total, CHUNK_SIZE), desc="  quality scan", leave=False):
            end = min(start + CHUNK_SIZE, n_total)
            batch_idx = np.asarray(candidate_indices[start:end])
            order = np.argsort(batch_idx, kind="mergesort")
            sorted_idx = batch_idx[order]
            with h5py.File(x_path, "r") as f:
                chunk_sorted = np.array(f["x"][sorted_idx])
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            chunk = chunk_sorted[inv]
            for i in range(chunk.shape[0]):
                idx = int(batch_idx[i])
                p = chunk[i]
                p01 = to_01(p)
                passed, reason = passes_quality(p01)
                if passed:
                    kept.append(idx)
                else:
                    if reason == "solid_color":
                        n_solid += 1
                    elif reason == "high_black":
                        n_black += 1
                    else:
                        n_low_tissue += 1
        kept = np.array(kept)
        n_kept = len(kept)
        n_removed = n_total - n_kept
        print("  Step 1 summary: kept {} / {}  |  removed: {} (solid_color: {}, high_black: {}, low_tissue: {})".format(
            n_kept, n_total, n_removed, n_solid, n_black, n_low_tissue))
        np.savez_compressed(ckpt_step1, kept_indices=kept, n_solid=np.int64(n_solid), n_black=np.int64(n_black),
                            n_low_tissue=np.int64(n_low_tissue), n_total=np.int64(n_total))
        print("  Checkpoint saved: {}".format(os.path.basename(ckpt_step1)))
        if n_kept == 0:
            print("  WARNING: no patches kept for {}; skipping write.".format(split_name))
            return {"kept_indices": [], "n_kept": 0, "n_total": n_total,
                    "class_balance_before": class_balance_before,
                    "step1_removed": {"solid_color": n_solid, "high_black": n_black, "low_tissue": n_low_tissue}}

    print("[{}] Step 2-3: Stain normalization + value [0,1] ...".format(split_name))
    blue_dom_threshold = None
    if stain_mode == "adaptive":
        blue_dom_threshold = compute_blue_dom_threshold(x_path, kept, BLUE_DOM_PERCENTILE)
        print("  blue_dom_threshold ({}th pct, this split): {:.4f}".format(int(BLUE_DOM_PERCENTILE), blue_dom_threshold))
    else:
        print("  stain mode: macenko (benchmark-style classical transform)")
    n_sample = min(2000, n_kept)
    sample_positions = set(np.random.RandomState(42).choice(n_kept, size=n_sample, replace=False).tolist())
    n_qa = min(QA_SAMPLES_PER_SPLIT, n_kept)
    qa_positions = sorted(np.random.RandomState(123).choice(n_kept, size=n_qa, replace=False).tolist()) if n_qa > 0 else []
    before_mean_rgb = []
    after_stain_mean_rgb = []
    before_pink_pct = []
    after_pink_pct = []
    after_value_min = []
    after_value_max = []
    after_value_mean = []

    lum_std = LuminosityStandardizer()
    sample_dist_before = []
    sample_dist_after = []
    with h5py.File(x_path, "r") as f_x_in, h5py.File(y_path, "r") as f_y_in, \
         h5py.File(out_x, "w") as f_x, h5py.File(out_y, "w") as f_y:
        x_data = f_x_in["x"]
        y_data = f_y_in["y"]
        f_x.create_dataset("x", shape=(n_kept, 96, 96, 3), dtype=np.float32, chunks=(1, 96, 96, 3), compression="gzip")
        f_y.create_dataset("y", shape=(n_kept,), dtype=np.float32)

        kept_labels = []
        normalizer_used = []
        mean_r_per_patch = []
        pink_pct_per_patch = []
        for pos, idx in enumerate(tqdm(kept, desc="  stain+value", leave=False)):
            p = np.array(x_data[idx])
            label = float(np.array(y_data[idx]).flatten()[0])
            kept_labels.append(label)
            p01 = np.clip(p.astype(np.float64) / 255.0, 0, 1) if p.max() > 1 else np.clip(p.astype(np.float64), 0, 1)
            if stain_router is not None and ref_packs is not None:
                pj = choose_ref_pack_index(p01, stain_router, ref_packs)
                pack = ref_packs[pj]
                mac_u, rei_u, ref_m = pack["macenko"], pack["reinhard"], pack["ref_mean_rgb"]
            else:
                mac_u, rei_u, ref_m = macenko, reinhard, ref_mean_rgb
            in_sample = pos in sample_positions
            if in_sample:
                if stain_mode == "adaptive":
                    preprocessed, norm_u8, which_norm = normalize_patch(
                        p01, mac_u, rei_u, blue_dom_threshold, lum_std.standardize,
                        return_after_stain=True, return_which_normalizer=True
                    )
                else:
                    preprocessed, norm_u8, which_norm = normalize_patch_macenko_benchmark_style(
                        p01, mac_u, lum_std.standardize, return_after_stain=True, return_which_normalizer=True
                    )
                normalizer_used.append(which_norm)
                mr, pp = _mean_r_pink_pct(preprocessed)
                mean_r_per_patch.append(mr)
                pink_pct_per_patch.append(pp)
                bm = p01.mean(axis=(0, 1))
                am = preprocessed.mean(axis=(0, 1))
                before_mean_rgb.append(tuple(bm))
                after_stain_mean_rgb.append(tuple(am))
                ref_vec = np.array(ref_m, dtype=np.float64)
                sample_dist_before.append(float(np.abs(bm - ref_vec).sum()))
                sample_dist_after.append(float(np.abs(am - ref_vec).sum()))
                before_pink_pct.append(float((p01[:, :, 0] > p01[:, :, 2]).mean()))
                after_pink_pct.append(pp)
                after_value_min.append(float(preprocessed.min()))
                after_value_max.append(float(preprocessed.max()))
                after_value_mean.append(float(preprocessed.mean()))
            else:
                if stain_mode == "adaptive":
                    preprocessed, which_norm = normalize_patch(
                        p01, mac_u, rei_u, blue_dom_threshold, lum_std.standardize,
                        return_which_normalizer=True
                    )
                else:
                    preprocessed, which_norm = normalize_patch_macenko_benchmark_style(
                        p01, mac_u, lum_std.standardize, return_which_normalizer=True
                    )
                normalizer_used.append(which_norm)
                mr, pp = _mean_r_pink_pct(preprocessed)
                mean_r_per_patch.append(mr)
                pink_pct_per_patch.append(pp)
            f_x["x"][pos] = preprocessed
            f_y["y"][pos] = label

        mean_r_all = np.array(mean_r_per_patch, dtype=np.float64)
        pink_pct_all = np.array(pink_pct_per_patch, dtype=np.float64)
        if stain_mode == "adaptive":
            thresh_mean_r = np.percentile(mean_r_all, PURPLE_PERCENTILE)
            thresh_pink_pct = np.percentile(pink_pct_all, PURPLE_PERCENTILE)
            purple_fallback_indices = np.where((mean_r_all <= thresh_mean_r) | (pink_pct_all <= thresh_pink_pct))[0].tolist()
            print("  Purple (percentile-based): thresh mean_r={:.4f}, pink_pct={:.4f} -> {} patches below threshold".format(
                thresh_mean_r, thresh_pink_pct, len(purple_fallback_indices)))
            for pos in tqdm(purple_fallback_indices, desc="  purple->luminosity", leave=False):
                idx = int(kept[pos])
                p = np.array(x_data[idx])
                p01 = np.clip(p.astype(np.float64) / 255.0, 0, 1) if p.max() > 1 else np.clip(p.astype(np.float64), 0, 1)
                preprocessed = _luminosity_only_norm(p01, lum_std.standardize)
                f_x["x"][pos] = preprocessed
                normalizer_used[pos] = NORM_LUMINOSITY_ONLY
        else:
            thresh_mean_r = None
            thresh_pink_pct = None
            purple_fallback_indices = []

        normalizer_path = os.path.join(out_dir, "{}_normalizer_used.npy".format(split_name))
        np.save(normalizer_path, np.array(normalizer_used, dtype=np.uint8))
        print("  Saved normalizer-per-patch:", os.path.basename(normalizer_path))
        purple_path = os.path.join(out_dir, "{}_purple_fallback.npy".format(split_name))
        np.save(purple_path, np.array(purple_fallback_indices, dtype=np.int64))
        print("  All-purple (percentile-based) replaced with luminosity-only: {} patches -> {}".format(len(purple_fallback_indices), os.path.basename(purple_path)))

        if n_qa > 0:
            qa_orig = []
            qa_norm = []
            qa_labels = []
            qa_orig_idx = []
            qa_norm_code = []
            for pos in qa_positions:
                idx = int(kept[pos])
                p = np.array(x_data[idx])
                p01 = np.clip(p.astype(np.float64) / 255.0, 0, 1) if p.max() > 1 else np.clip(p.astype(np.float64), 0, 1)
                qa_orig.append((p01 * 255.0).clip(0, 255).astype(np.uint8))
                qa_norm.append((np.array(f_x["x"][pos]) * 255.0).clip(0, 255).astype(np.uint8))
                qa_labels.append(float(np.array(y_data[idx]).flatten()[0]))
                qa_orig_idx.append(idx)
                qa_norm_code.append(int(normalizer_used[pos]))
            qa_path = os.path.join(out_dir, "{}_qa_samples.npz".format(split_name))
            np.savez_compressed(
                qa_path,
                original_idx=np.array(qa_orig_idx, dtype=np.int64),
                output_pos=np.array(qa_positions, dtype=np.int64),
                labels=np.array(qa_labels, dtype=np.float32),
                normalizer_used=np.array(qa_norm_code, dtype=np.uint8),
                original_u8=np.array(qa_orig, dtype=np.uint8),
                normalized_u8=np.array(qa_norm, dtype=np.uint8),
            )
            print("  Saved QA sample pack:", os.path.basename(qa_path), "(n={})".format(n_qa))
        else:
            qa_path = None

    before_arr = np.array(before_mean_rgb)
    after_arr = np.array(after_stain_mean_rgb)
    if sample_dist_before:
        dist_before = float(np.mean(sample_dist_before))
        dist_after = float(np.mean(sample_dist_after))
    else:
        ref = np.array(ref_mean_rgb)
        dist_before = float(np.mean(np.abs(before_arr - ref).sum(axis=1)))
        dist_after = float(np.mean(np.abs(after_arr - ref).sum(axis=1)))
    before_r = before_arr[:, 0]
    before_b = before_arr[:, 2]
    after_r = after_arr[:, 0]
    after_b = after_arr[:, 2]
    before_pink = np.array(before_pink_pct)
    after_pink = np.array(after_pink_pct)
    purple_only_after = np.sum((after_r < 0.2) | (after_pink < 0.05))
    mean_r_before = float(np.mean(before_r))
    mean_r_after = float(np.mean(after_r))
    mean_pink_before = float(np.mean(before_pink))
    mean_pink_after = float(np.mean(after_pink))
    std_r_after = float(np.std(after_arr[:, 0]))
    std_g_after = float(np.std(after_arr[:, 1]))
    std_b_after = float(np.std(after_arr[:, 2]))
    outlier_after = np.sum((after_r < 0.2) | (after_r > 0.8) | (after_b < 0.2) | (after_b > 0.8))
    outlier_rate_after = float(outlier_after / len(after_r)) if len(after_r) else 0.0

    print("  Step 2 (stain) effectiveness (sample n={}):".format(len(before_mean_rgb)))
    print("    Closer to ref:  mean L1 dist to ref  before: {:.4f}  after: {:.4f}  (lower after = better)".format(dist_before, dist_after))
    print("    Color preserved:  mean R before/after: {:.4f} / {:.4f}   pink_pct (R>B) before/after: {:.4f} / {:.4f}  (no big drop = pink not lost)".format(
        mean_r_before, mean_r_after, mean_pink_before, mean_pink_after))
    if stain_mode == "adaptive":
        print("    Purple (percentile-based): thresh mean_r={:.4f}, pink_pct={:.4f} -> {} patches replaced with luminosity-only (stored data)".format(
            thresh_mean_r, thresh_pink_pct, len(purple_fallback_indices)))
    else:
        print("    Purple replacement: disabled in macenko benchmark-style mode")
    print("    Sample purple count (fixed 0.2/0.05): {} / {}  (reference only; stored data uses percentile)".format(
        int(purple_only_after), len(after_r)))
    print("  Step 3 (value) effectiveness (sample n={}): min={:.4f}, max={:.4f}, mean={:.4f}  (expect [0,1])".format(
        len(after_value_min), np.mean(after_value_min), np.mean(after_value_max), np.mean(after_value_mean)))

    kept_labels_arr = np.array(kept_labels)
    n_positive = int((kept_labels_arr >= 0.5).sum())
    n_negative = int((kept_labels_arr < 0.5).sum())
    class_balance_after = {"n_positive": n_positive, "n_negative": n_negative, "frac_positive": float(n_positive / n_kept) if n_kept else 0.0}
    frac_before = float(class_balance_before["frac_positive"])
    frac_after = float(class_balance_after["frac_positive"])
    abs_diff = float(abs(frac_after - frac_before))
    if abs_diff > CLASS_BALANCE_ALERT_ABS_DIFF:
        print("  QA ALERT: class balance shifted after filtering (pos frac before {:.4f} -> after {:.4f}, abs diff {:.4f})".format(
            frac_before, frac_after, abs_diff))
    usage = _normalizer_usage_dict(normalizer_used)
    print("  Normalizer usage: Mac {:.1%}, Rein {:.1%}, Mac_fb {:.1%}, Rein_fb {:.1%}, Lum {:.1%}".format(
        usage["macenko_rate"], usage["reinhard_rate"], usage["macenko_fallback_rate"], usage["reinhard_fallback_rate"], usage["luminosity_only_rate"]))

    return {
        "kept_indices": kept.tolist(),
        "n_kept": n_kept,
        "n_total": n_total,
        "class_balance": class_balance_after,
        "class_balance_before": class_balance_before,
        "qa_class_balance_abs_diff": abs_diff,
        "qa_class_balance_alert": bool(abs_diff > CLASS_BALANCE_ALERT_ABS_DIFF),
        "qa_samples_file": os.path.basename(qa_path) if qa_path else None,
        "step1_removed": {"solid_color": n_solid, "high_black": n_black, "low_tissue": n_low_tissue},
        "step2_normalizer_usage": usage,
        "step2_blue_dom_threshold": float(blue_dom_threshold) if blue_dom_threshold is not None else None,
        "step2_purple_fallback_count": len(purple_fallback_indices),
        "step2_purple_percentile": PURPLE_PERCENTILE,
        "step2_purple_thresh_mean_r": float(thresh_mean_r) if thresh_mean_r is not None else None,
        "step2_purple_thresh_pink_pct": float(thresh_pink_pct) if thresh_pink_pct is not None else None,
        "step2_sample_dist_before": float(dist_before),
        "step2_sample_dist_after": float(dist_after),
        "step2_sample_mean_r_before": mean_r_before,
        "step2_sample_mean_r_after": mean_r_after,
        "step2_sample_pink_pct_before": mean_pink_before,
        "step2_sample_pink_pct_after": mean_pink_after,
        "step2_sample_purple_only_after": int(purple_only_after),
        "step2_sample_std_mean_rgb_after": (std_r_after, std_g_after, std_b_after),
        "step2_sample_outlier_rate_after": outlier_rate_after,
        "step2_sample_n": len(before_mean_rgb),
        "step3_sample_min_max_mean": (float(np.mean(after_value_min)), float(np.mean(after_value_max)), float(np.mean(after_value_mean))),
        "step2_stain_mode": "multi_ref" if stain_router is not None else "single",
        "step2_multi_ref_indices": list(MULTI_REF_INDICES) if stain_router is not None else None,
        "step2_multi_ref_merge_patch_idx": MULTI_REF_MERGE_PATCH_IDX if stain_router is not None else None,
        "step2_multi_ref_merge_cluster": int(stain_router["merge_cluster"]) if stain_router is not None else None,
        "step2_multi_ref_cluster_to_pack": {str(k): int(v) for k, v in stain_router["cluster_to_pack"].items()} if stain_router is not None else None,
        "step2_stain_method": stain_mode,
    }


def main():
    global PURPLE_PERCENTILE, BLUE_DOM_PERCENTILE
    global GUARDRAIL_MEAN_R_MIN, GUARDRAIL_PINK_PCT_MIN, GUARDRAIL_MEAN_SAT_MIN
    global QA_SAMPLES_PER_SPLIT, CLASS_BALANCE_ALERT_ABS_DIFF

    parser = argparse.ArgumentParser(description="Preprocess histopath patch H5 (PCam or WILDS layout) and write new H5 files.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="PCam: pcam_data root. WILDS (--layout wilds): folder containing train_x.h5, valid_x.h5, test_x.h5 (+ _y). Default: PROJECT_ROOT/pcam_data")
    parser.add_argument("--layout", type=str, default="pcam", choices=("pcam", "wilds"),
                        help="Input H5 naming: pcam (training/*/camelyonpatch_*.h5) or wilds (train_x.h5, valid_x.h5, test_x.h5 in data-dir).")
    parser.add_argument(
        "--reference-train-x-h5",
        type=str,
        default=None,
        help="Optional train_x.h5 used only to read the reference patch x[reference_train_index] for fitting "
        "Macenko/Reinhard. For --layout wilds, defaults to pcam_data/training/camelyonpatch_level_2_split_train_x.h5 "
        "if that file exists (same stain fit as PCam). Otherwise fits from this dataset's train_x.h5.",
    )
    parser.add_argument("--no-quality", action="store_true",
                        help="Skip quality filter (keep all patches)")
    parser.add_argument("--ref-config", type=str, default=None,
                        help="Path to stain_reference.json (default: experiments/stain_reference/stain_reference.json)")
    parser.add_argument("--dedup-dir", type=str, default=None,
                        help="Path to dedup output dir (e.g. pcam_dedup). Must contain train/valid/test_kept_indices.npy. Default: PROJECT_ROOT/pcam_dedup (used if that dir exists).")
    parser.add_argument("--purple-percentile", type=float, default=PURPLE_PERCENTILE,
                        help="Percentile for purple-tail replacement per split (default: {})".format(PURPLE_PERCENTILE))
    parser.add_argument("--blue-dom-percentile", type=float, default=BLUE_DOM_PERCENTILE,
                        help="Percentile for per-split blue_dom threshold (default: {})".format(BLUE_DOM_PERCENTILE))
    parser.add_argument("--guardrail-mean-r-min", type=float, default=GUARDRAIL_MEAN_R_MIN,
                        help="Guardrail min mean R after normalization (default: {})".format(GUARDRAIL_MEAN_R_MIN))
    parser.add_argument("--guardrail-pink-pct-min", type=float, default=GUARDRAIL_PINK_PCT_MIN,
                        help="Guardrail min pink_pct after normalization (default: {})".format(GUARDRAIL_PINK_PCT_MIN))
    parser.add_argument("--guardrail-mean-sat-min", type=float, default=GUARDRAIL_MEAN_SAT_MIN,
                        help="Guardrail min mean saturation after normalization (default: {})".format(GUARDRAIL_MEAN_SAT_MIN))
    parser.add_argument("--qa-samples-per-split", type=int, default=QA_SAMPLES_PER_SPLIT,
                        help="How many before/after QA samples to save per split (default: {})".format(QA_SAMPLES_PER_SPLIT))
    parser.add_argument("--class-balance-alert-diff", type=float, default=CLASS_BALANCE_ALERT_ABS_DIFF,
                        help="Raise QA alert if |pos_frac_after - pos_frac_before| exceeds this (default: {})".format(CLASS_BALANCE_ALERT_ABS_DIFF))
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints: skip Step 1 if checkpoint exists; skip a split entirely if its output H5 already exists.")
    parser.add_argument("--status", action="store_true",
                        help="Only inspect preprocessed/ state (checkpoints and output H5) and print what finished / what to do. Use after an interrupted run.")
    parser.add_argument("--preprocessed-subdir", type=str, default="preprocessed_macenko_benchmark_style",
                        help="Output folder under data_dir (default: preprocessed). Use e.g. preprocessed_multi_ref to keep separate from a prior run.")
    parser.add_argument("--stain-multi-ref", action="store_true",
                        help="Use 3 fixed references + KMeans routing; cluster matching patch %d maps to closest ref by hue (mean RGB)." % MULTI_REF_MERGE_PATCH_IDX)
    parser.add_argument("--stain-mode", type=str, default="macenko", choices=("adaptive", "macenko"),
                        help="adaptive: current Macenko/Reinhard routing + purple replacement. macenko: benchmark-style classical Macenko only.")
    args = parser.parse_args()
    PURPLE_PERCENTILE = float(args.purple_percentile)
    BLUE_DOM_PERCENTILE = float(args.blue_dom_percentile)
    GUARDRAIL_MEAN_R_MIN = float(args.guardrail_mean_r_min)
    GUARDRAIL_PINK_PCT_MIN = float(args.guardrail_pink_pct_min)
    GUARDRAIL_MEAN_SAT_MIN = float(args.guardrail_mean_sat_min)
    QA_SAMPLES_PER_SPLIT = int(max(0, args.qa_samples_per_split))
    CLASS_BALANCE_ALERT_ABS_DIFF = float(max(0.0, args.class_balance_alert_diff))

    if args.dedup_dir is None and args.layout == "pcam":
        default_dedup = os.path.join(PROJECT_ROOT, "pcam_dedup")
        if os.path.isdir(default_dedup) and os.path.isfile(os.path.join(default_dedup, "train_kept_indices.npy")):
            args.dedup_dir = default_dedup
            print("Using default dedup dir:", args.dedup_dir)

    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, "pcam_data")
    out_dir = os.path.join(data_dir, args.preprocessed_subdir)
    if args.status:
        print_preprocess_status(out_dir)
        sys.exit(0)
    if not os.path.isdir(data_dir):
        print("Error: data_dir not found:", data_dir)
        sys.exit(1)

    if args.layout == "wilds":
        train_x_path = os.path.join(data_dir, "train_x.h5")
        train_y_path = os.path.join(data_dir, "train_y.h5")
        valid_x_path = os.path.join(data_dir, "valid_x.h5")
        valid_y_path = os.path.join(data_dir, "valid_y.h5")
        test_x_path = os.path.join(data_dir, "test_x.h5")
        test_y_path = os.path.join(data_dir, "test_y.h5")
    else:
        training_dir = os.path.join(data_dir, "training")
        val_dir = os.path.join(data_dir, "val")
        test_dir = os.path.join(data_dir, "test")
        if not os.path.isdir(training_dir):
            print("Error: expected training/ under data_dir. Layout: data_dir/training/, val/, test/")
            sys.exit(1)

        train_x_path = os.path.join(training_dir, "camelyonpatch_level_2_split_train_x.h5")
        train_y_path = os.path.join(training_dir, "camelyonpatch_level_2_split_train_y.h5")
        valid_x_path = os.path.join(val_dir, "camelyonpatch_level_2_split_valid_x.h5")
        valid_y_path = os.path.join(val_dir, "camelyonpatch_level_2_split_valid_y.h5")
        test_x_path = os.path.join(test_dir, "camelyonpatch_level_2_split_test_x.h5")
        test_y_path = os.path.join(test_dir, "camelyonpatch_level_2_split_test_y.h5")

    for p in [train_x_path, train_y_path, valid_x_path, valid_y_path, test_x_path, test_y_path]:
        if not os.path.isfile(p):
            print("Error: missing", p)
            sys.exit(1)

    pcam_default_ref_x = os.path.join(PROJECT_ROOT, "pcam_data", "training", "camelyonpatch_level_2_split_train_x.h5")
    if args.reference_train_x_h5:
        reference_train_x_path = os.path.abspath(args.reference_train_x_h5)
        if not os.path.isfile(reference_train_x_path):
            print("Error: --reference-train-x-h5 not found:", reference_train_x_path)
            sys.exit(1)
    elif args.layout == "wilds" and os.path.isfile(pcam_default_ref_x):
        reference_train_x_path = pcam_default_ref_x
        print("WILDS: fitting stain normalizers from PCam train H5 (default):", reference_train_x_path)
    else:
        reference_train_x_path = None
        if args.layout == "wilds":
            print(
                "Warning: PCam train_x not at default path; fitting stain reference from WILDS train_x.h5 "
                "(index matches JSON but patch is NOT the PCam reference). Install PCam H5 or pass --reference-train-x-h5."
            )

    ref_config_path = args.ref_config or os.path.join(PROJECT_ROOT, "experiments", "stain_reference", "stain_reference.json")
    if not args.stain_multi_ref and not os.path.isfile(ref_config_path):
        print("Error: ref_config not found:", ref_config_path)
        sys.exit(1)
    if args.stain_multi_ref and args.stain_mode != "adaptive":
        print("Error: --stain-multi-ref is only supported with --stain-mode adaptive.")
        sys.exit(1)

    dedup_dir = args.dedup_dir
    candidate_indices_per_split = None
    if dedup_dir:
        dedup_dir = os.path.abspath(dedup_dir)
        if not os.path.isdir(dedup_dir):
            print("Error: --dedup-dir not found:", dedup_dir)
            sys.exit(1)
        print("Using deduplicated indices from:", dedup_dir)
        candidate_indices_per_split = {}
        for split in ("train", "valid", "test"):
            path = os.path.join(dedup_dir, "{}_kept_indices.npy".format(split))
            if not os.path.isfile(path):
                print("Error: missing", path)
                sys.exit(1)
            candidate_indices_per_split[split] = np.load(path)
            print("  {}: {} candidates".format(split, len(candidate_indices_per_split[split])))

    os.makedirs(out_dir, exist_ok=True)
    print("Preprocessing: Step 1 (quality) -> Step 2 (stain norm) -> Step 3 (value [0,1])")
    print("Output dir:", out_dir)
    if args.resume:
        print("Resume: enabled (will skip Step 1 when checkpoint exists; skip split when output H5 exists)")

    stain_router = None
    ref_packs = None
    if args.stain_multi_ref:
        print("Stain mode: MULTI-REF (indices {}, merge cluster from patch {})".format(
            MULTI_REF_INDICES, MULTI_REF_MERGE_PATCH_IDX))
        train_cand = candidate_indices_per_split.get("train") if candidate_indices_per_split else None
        stain_router = fit_multi_ref_router(train_x_path, train_candidate_indices=train_cand)
        ref_packs = build_ref_packs(train_x_path, MULTI_REF_INDICES)
        macenko, reinhard, ref_idx, ref_mean_rgb = ref_packs[0]["macenko"], ref_packs[0]["reinhard"], ref_packs[0]["idx"], ref_packs[0]["ref_mean_rgb"]
        print("Primary ref (pack 0) index:", ref_idx, "| mean RGB:", [round(x, 4) for x in ref_mean_rgb])
    else:
        print("Loading reference and fitting normalizers ...")
        macenko, reinhard, ref_idx, ref_mean_rgb = get_normalizers_and_threshold(
            data_dir, train_x_path, ref_config_path, reference_train_x_path=reference_train_x_path
        )
        print("Reference index:", ref_idx, "| ref mean RGB:", [round(x, 4) for x in ref_mean_rgb])

    if args.no_quality:
        def process_split_no_quality(split_name, x_path, y_path, candidate_indices=None):
            with h5py.File(x_path, "r") as f:
                n_total_file = f["x"].shape[0]
            if candidate_indices is None:
                candidate_indices = np.arange(n_total_file)
            else:
                candidate_indices = np.asarray(candidate_indices)
            n_kept = len(candidate_indices)
            kept = candidate_indices
            blue_dom_threshold = compute_blue_dom_threshold(x_path, kept, BLUE_DOM_PERCENTILE)
            print("[{}] No quality filter: processing all {} patches. blue_dom_threshold ({}th pct): {:.4f}. Step 2-3: stain + value ...".format(
                split_name, n_kept, int(BLUE_DOM_PERCENTILE), blue_dom_threshold))
            out_x = os.path.join(out_dir, "{}_x.h5".format(split_name))
            out_y = os.path.join(out_dir, "{}_y.h5".format(split_name))
            with h5py.File(y_path, "r") as f:
                y_data = f["y"]
            from staintools.preprocessing.luminosity_standardizer import LuminosityStandardizer
            lum_std = LuminosityStandardizer()
            labels = []
            with h5py.File(out_x, "w") as f_x, h5py.File(out_y, "w") as f_y:
                f_x.create_dataset("x", shape=(n_kept, 96, 96, 3), dtype=np.float32, chunks=(1, 96, 96, 3), compression="gzip")
                f_y.create_dataset("y", shape=(n_kept,), dtype=np.float32)
                for pos in tqdm(range(n_kept), desc="  stain+value", leave=False):
                    idx = int(kept[pos])
                    with h5py.File(x_path, "r") as f:
                        p = np.array(f["x"][idx])
                    label = float(np.array(y_data[idx]).flatten()[0])
                    labels.append(label)
                    p01 = np.clip(p.astype(np.float64) / 255.0, 0, 1) if p.max() > 1 else np.clip(p.astype(np.float64), 0, 1)
                    if stain_router is not None and ref_packs is not None:
                        pj = choose_ref_pack_index(p01, stain_router, ref_packs)
                        pk = ref_packs[pj]
                        preprocessed = normalize_patch(p01, pk["macenko"], pk["reinhard"], blue_dom_threshold, lum_std.standardize)
                    else:
                        preprocessed = normalize_patch(p01, macenko, reinhard, blue_dom_threshold, lum_std.standardize)
                    f_x["x"][pos] = preprocessed
                    f_y["y"][pos] = label
            arr = np.array(labels)
            n_pos = int((arr >= 0.5).sum())
            n_neg = int((arr < 0.5).sum())
            return {"kept_indices": kept.tolist(), "n_kept": n_kept, "n_total": n_kept,
                    "class_balance_before": {"n_positive": n_pos, "n_negative": n_neg, "frac_positive": float(n_pos / n_kept) if n_kept else 0},
                    "class_balance": {"n_positive": n_pos, "n_negative": n_neg, "frac_positive": float(n_pos / n_kept) if n_kept else 0},
                    "qa_class_balance_abs_diff": 0.0, "qa_class_balance_alert": False}

        manifest = {
            "train": process_split_no_quality("train", train_x_path, train_y_path, candidate_indices_per_split.get("train") if candidate_indices_per_split else None),
            "valid": process_split_no_quality("valid", valid_x_path, valid_y_path, candidate_indices_per_split.get("valid") if candidate_indices_per_split else None),
            "test": process_split_no_quality("test", test_x_path, test_y_path, candidate_indices_per_split.get("test") if candidate_indices_per_split else None),
        }
    else:
        manifest = {
            "train": process_split("train", train_x_path, train_y_path, out_dir, macenko, reinhard, ref_mean_rgb,
                                   candidate_indices=candidate_indices_per_split.get("train") if candidate_indices_per_split else None, resume=args.resume,
                                   stain_router=stain_router, ref_packs=ref_packs, stain_mode=args.stain_mode),
            "valid": process_split("valid", valid_x_path, valid_y_path, out_dir, macenko, reinhard, ref_mean_rgb,
                                   candidate_indices=candidate_indices_per_split.get("valid") if candidate_indices_per_split else None, resume=args.resume,
                                   stain_router=stain_router, ref_packs=ref_packs, stain_mode=args.stain_mode),
            "test": process_split("test", test_x_path, test_y_path, out_dir, macenko, reinhard, ref_mean_rgb,
                                  candidate_indices=candidate_indices_per_split.get("test") if candidate_indices_per_split else None, resume=args.resume,
                                  stain_router=stain_router, ref_packs=ref_packs, stain_mode=args.stain_mode),
        }

    manifest["config"] = {
        "layout": args.layout,
        "data_dir": data_dir,
        "reference_train_x_h5": os.path.abspath(reference_train_x_path) if reference_train_x_path else None,
        "preprocessed_subdir": args.preprocessed_subdir,
        "dedup_dir": os.path.abspath(args.dedup_dir) if args.dedup_dir else None,
        "stain_mode": args.stain_mode,
        "stain_multi_ref": bool(args.stain_multi_ref),
        "multi_ref_indices": list(MULTI_REF_INDICES) if args.stain_multi_ref else None,
        "multi_ref_merge_patch_idx": MULTI_REF_MERGE_PATCH_IDX if args.stain_multi_ref else None,
        "multi_ref_merge_cluster": int(stain_router["merge_cluster"]) if args.stain_multi_ref and stain_router else None,
        "multi_ref_cluster_to_pack": ({str(k): int(v) for k, v in stain_router["cluster_to_pack"].items()} if args.stain_multi_ref and stain_router else None),
        "blue_dom_per_split": True,
        "blue_dom_percentile": BLUE_DOM_PERCENTILE,
        "ref_train_idx": ref_idx,
        "ref_mean_rgb": list(ref_mean_rgb),
        "quality_filter": not args.no_quality,
        "solid_color_std": SOLID_COLOR_STD,
        "high_black_ratio": HIGH_BLACK_RATIO,
        "low_tissue_threshold": LOW_TISSUE_THRESHOLD,
        "purple_percentile": PURPLE_PERCENTILE,
        "guardrail_mean_r_min": GUARDRAIL_MEAN_R_MIN,
        "guardrail_pink_pct_min": GUARDRAIL_PINK_PCT_MIN,
        "guardrail_mean_sat_min": GUARDRAIL_MEAN_SAT_MIN,
        "qa_samples_per_split": QA_SAMPLES_PER_SPLIT,
        "class_balance_alert_abs_diff": CLASS_BALANCE_ALERT_ABS_DIFF,
        "patch_shape": [96, 96, 3],
        "value_range": [0, 1],
        "normalizer_used_codes": {"0": "macenko", "1": "reinhard", "2": "macenko_fallback", "3": "reinhard_fallback", "4": "luminosity_only"},
    }

    total_before = sum(manifest[s]["n_total"] for s in ("train", "valid", "test"))
    total_kept = sum(manifest[s]["n_kept"] for s in ("train", "valid", "test"))
    total_removed = total_before - total_kept
    step1 = manifest["train"].get("step1_removed", {})
    manifest["dataset_summary"] = {
        "total_patches_before": total_before,
        "total_patches_after": total_kept,
        "total_removed": total_removed,
        "removed_solid_color": sum(manifest[s].get("step1_removed", {}).get("solid_color", 0) for s in ("train", "valid", "test")),
        "removed_high_black": sum(manifest[s].get("step1_removed", {}).get("high_black", 0) for s in ("train", "valid", "test")),
        "removed_low_tissue": sum(manifest[s].get("step1_removed", {}).get("low_tissue", 0) for s in ("train", "valid", "test")),
        "train": {"n_kept": manifest["train"]["n_kept"], "n_total": manifest["train"]["n_total"], "class_balance": manifest["train"].get("class_balance")},
        "valid": {"n_kept": manifest["valid"]["n_kept"], "n_total": manifest["valid"]["n_total"], "class_balance": manifest["valid"].get("class_balance")},
        "test": {"n_kept": manifest["test"]["n_kept"], "n_total": manifest["test"]["n_total"], "class_balance": manifest["test"].get("class_balance")},
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Done. Wrote", manifest_path)

    report = {
        "pipeline": ["quality_filter", "stain_normalization_macenko_reinhard", "value_normalization_0_1"],
        "config": manifest["config"],
        "dataset_summary": manifest["dataset_summary"],
        "per_split": {
            s: {
                "n_before": manifest[s]["n_total"],
                "n_after": manifest[s]["n_kept"],
                "n_removed": manifest[s]["n_total"] - manifest[s]["n_kept"],
                "class_balance_before": manifest[s].get("class_balance_before"),
                "class_balance": manifest[s].get("class_balance"),
                "qa_class_balance_abs_diff": manifest[s].get("qa_class_balance_abs_diff"),
                "qa_class_balance_alert": manifest[s].get("qa_class_balance_alert"),
                "qa_samples_file": manifest[s].get("qa_samples_file"),
                "step1_removed": manifest[s].get("step1_removed"),
                "step2_blue_dom_threshold": manifest[s].get("step2_blue_dom_threshold"),
                "step2_normalizer_usage": manifest[s].get("step2_normalizer_usage"),
                "step2_dist_to_ref_before": manifest[s].get("step2_sample_dist_before"),
                "step2_dist_to_ref_after": manifest[s].get("step2_sample_dist_after"),
                "step2_mean_r_before_after": (manifest[s].get("step2_sample_mean_r_before"), manifest[s].get("step2_sample_mean_r_after")),
                "step2_pink_pct_before_after": (manifest[s].get("step2_sample_pink_pct_before"), manifest[s].get("step2_sample_pink_pct_after")),
                "step2_purple_only_after": manifest[s].get("step2_sample_purple_only_after"),
                "step2_purple_fallback_count": manifest[s].get("step2_purple_fallback_count"),
                "step2_std_mean_rgb_after": manifest[s].get("step2_sample_std_mean_rgb_after"),
                "step2_outlier_rate_after": manifest[s].get("step2_sample_outlier_rate_after"),
                "step2_sample_n": manifest[s].get("step2_sample_n"),
                "step3_min_max_mean": manifest[s].get("step3_sample_min_max_mean"),
            }
            for s in ("train", "valid", "test")
        },
    }
    report_path = os.path.join(out_dir, "preprocess_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote preprocess_report.json:", report_path)


if __name__ == "__main__":
    main()
