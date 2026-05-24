"""
Bootstrap CIs and paired permutation tests for Virchow C1–C4 held-out test exports.

Reads experiments/virchow_colab/evals_cross_domain/*/test_predictions.npz
Uses calibrated probabilities (prob_after_temperature) by default.

Methodology alignment (docs/final_methodology.md §7):
  - Case-level bootstrap B=2000, 95% percentile CIs
  - Transfer drops: paired bootstrap within replicate (in-domain vs external)
  - Paired permutation (10k sign flips) for head-to-head on same test patches
  - Benjamini–Hochberg q=0.05 on predefined confirmatory permutation tests

Outputs: reports/inference/virchow_test_inference.json (+ summary CSV)
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from train_virchow_preprocessed_colab import (  # noqa: E402
    compute_classification_metrics,
    expected_calibration_error,
)

DEFAULT_EVAL_ROOT = Path("experiments/virchow_colab/evals_cross_domain")
DEFAULT_OUT = Path("reports/inference/virchow_test_inference.json")

CONDITIONS = {
    "C1": ("pcam_trained_on_pcam_test", "PCam", "PCam", True),
    "C2": ("pcam_trained_on_cam17_test", "PCam", "CAMELYON17", False),
    "C3": ("cam17_trained_on_cam17_test", "CAMELYON17", "CAMELYON17", True),
    "C4": ("cam17_trained_on_pcam_test", "CAMELYON17", "PCam", False),
}

METRIC_KEYS = [
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "brier_score",
    "log_loss",
]

EPS = 1e-12


def _load_npz(eval_root: Path, folder: str) -> Tuple[np.ndarray, np.ndarray]:
    p = eval_root / folder / "test_predictions.npz"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    z = np.load(p)
    y = np.asarray(z["y_true"], dtype=np.float64).reshape(-1)
    if "prob_after_temperature" in z.files:
        prob = np.asarray(z["prob_after_temperature"], dtype=np.float64).reshape(-1)
    else:
        prob = np.asarray(z["prob_sigmoid"], dtype=np.float64).reshape(-1)
    return y, prob


def _metric_bundle(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    m = compute_classification_metrics(y, prob, threshold=0.5)
    ece = expected_calibration_error(y, prob, n_bins=15)
    out: Dict[str, float] = {}
    for k in METRIC_KEYS:
        v = m.get(k)
        out[k] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else float("nan")
    out["ece_15"] = float(ece.get("ece", float("nan")))
    return out


def _fast_ece(y: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (prob >= lo) & (prob <= hi if i == n_bins - 1 else prob < hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        ece += (cnt / max(n, 1)) * abs(float(prob[m].mean()) - float(y[m].mean()))
    return float(ece)


def _metrics_on_sample(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    """Metrics for one bootstrap draw (no sklearn except AUC/AP)."""
    prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
    pred = (prob >= 0.5).astype(np.float64)
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    out: Dict[str, float] = {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "precision": prec,
        "recall": rec,
        "f1": (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0,
        "balanced_accuracy": 0.5
        * (rec + (tn / (tn + fp) if (tn + fp) > 0 else 0.0)),
        "brier_score": float(np.mean((prob - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob))),
        "ece_15": _fast_ece(y, prob),
    }
    if len(np.unique(y)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["average_precision"] = float(average_precision_score(y, prob))
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out


def _bootstrap_chunk(args: Tuple[np.ndarray, np.ndarray, List[np.ndarray]]) -> Dict[str, List[float]]:
    y, prob, index_chunks = args
    stores: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS + ["ece_15"]}
    for idx in index_chunks:
        mb = _metrics_on_sample(y[idx], prob[idx])
        for k in stores:
            v = mb.get(k, float("nan"))
            if not np.isnan(v):
                stores[k].append(v)
    return stores


def _bootstrap_metrics(
    y: np.ndarray,
    prob: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    workers: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    point = _metric_bundle(y, prob)
    all_idx = [rng.integers(0, n, size=n) for _ in range(n_boot)]

    n_workers = max(1, min(workers, n_boot))
    chunk_size = (n_boot + n_workers - 1) // n_workers
    chunks = [all_idx[i : i + chunk_size] for i in range(0, n_boot, chunk_size)]
    stores: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS + ["ece_15"]}

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_bootstrap_chunk, (y, prob, ch)) for ch in chunks]
        for fut in as_completed(futs):
            part = fut.result()
            for k in stores:
                stores[k].extend(part[k])

    out: Dict[str, Dict[str, float]] = {"point": point, "ci95": {}}
    for k, vals in stores.items():
        if not vals:
            out["ci95"][k] = {"low": None, "high": None}
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out["ci95"][k] = {
            "low": float(np.percentile(arr, 2.5)),
            "high": float(np.percentile(arr, 97.5)),
        }
    return out


def _paired_drop_chunk(args: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]) -> Tuple[
    Dict[str, List[float]], Dict[str, List[float]]
]:
    y, p_in, p_ext, _y2, index_chunks = args
    abs_store: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS + ["ece_15"]}
    rel_store: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS + ["ece_15"]}
    for idx in index_chunks:
        mi = _metrics_on_sample(y[idx], p_in[idx])
        me = _metrics_on_sample(y[idx], p_ext[idx])
        for k in abs_store:
            a, b = mi.get(k, float("nan")), me.get(k, float("nan"))
            if np.isnan(a) or np.isnan(b):
                continue
            abs_store[k].append(a - b)
            rel_store[k].append((a - b) / (a + EPS))
    return abs_store, rel_store


def _paired_transfer_drop(
    y_in: np.ndarray,
    p_in: np.ndarray,
    y_ext: np.ndarray,
    p_ext: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    workers: int,
) -> Dict[str, Any]:
    if len(y_in) != len(y_ext):
        raise ValueError("In-domain and external test sets differ in size; cannot pair transfer drops.")
    if not np.allclose(y_in, y_ext):
        raise ValueError("Labels differ between in-domain and external arrays on same test domain.")

    rng = np.random.default_rng(seed)
    n = len(y_in)
    point_in = _metric_bundle(y_in, p_in)
    point_ext = _metric_bundle(y_ext, p_ext)

    all_idx = [rng.integers(0, n, size=n) for _ in range(n_boot)]
    n_workers = max(1, min(workers, n_boot))
    chunk_size = (n_boot + n_workers - 1) // n_workers
    chunks = [all_idx[i : i + chunk_size] for i in range(0, n_boot, chunk_size)]

    abs_store: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS + ["ece_15"]}
    rel_store: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS + ["ece_15"]}

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(_paired_drop_chunk, (y_in, p_in, p_ext, y_ext, ch))
            for ch in chunks
        ]
        for fut in as_completed(futs):
            a_part, r_part = fut.result()
            for k in abs_store:
                abs_store[k].extend(a_part[k])
                rel_store[k].extend(r_part[k])

    def _summ(point_in_v: float, point_ext_v: float, store: List[float]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "point_in": point_in_v,
            "point_ext": point_ext_v,
            "point_delta": point_in_v - point_ext_v if not (np.isnan(point_in_v) or np.isnan(point_ext_v)) else None,
        }
        if store:
            arr = np.asarray(store, dtype=np.float64)
            out["ci95_low"] = float(np.percentile(arr, 2.5))
            out["ci95_high"] = float(np.percentile(arr, 97.5))
            out["excludes_zero"] = bool(out["ci95_low"] > 0 or out["ci95_high"] < 0)
        else:
            out["ci95_low"] = out["ci95_high"] = None
            out["excludes_zero"] = None
        return out

    abs_out = {k: _summ(point_in[k], point_ext[k], abs_store[k]) for k in abs_store}
    rel_out = {k: _summ(point_in[k], point_ext[k], rel_store[k]) for k in rel_store}
    return {"delta_abs": abs_out, "delta_rel": rel_out}


def _per_case_brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def _per_case_nll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _paired_permutation_p(
    d: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> float:
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    obs = float(np.mean(d))
    rng = np.random.default_rng(seed)
    n = len(d)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs(float(np.mean(d * signs))) >= abs(obs):
            count += 1
    return (1.0 + count) / (1.0 + n_perm)


def _benjamini_hochberg(pvals: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i][1])
    adj = [1.0] * m
    prev = 1.0
    for rank_rev, idx in enumerate(reversed(order)):
        r = m - rank_rev
        p = pvals[idx][1]
        val = min(prev, p * m / r)
        adj[idx] = val
        prev = val
    return [
        {"test_id": pvals[i][0], "p": pvals[i][1], "p_bh": adj[i], "reject_bh_q005": adj[i] <= 0.05}
        for i in range(m)
    ]


def run(
    eval_root: Path,
    out_json: Path,
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
    workers: int,
) -> Dict[str, Any]:
    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cid, (folder, *_rest) in CONDITIONS.items():
        data[cid] = _load_npz(eval_root, folder)

    report: Dict[str, Any] = {
        "eval_root": str(eval_root.resolve()),
        "n_bootstrap": n_boot,
        "n_permutation": n_perm,
        "seed": seed,
        "workers": workers,
        "probability_branch": "prob_after_temperature",
        "per_condition": {},
        "transfer_drops": {},
        "drop_asymmetry": {},
        "paired_permutation_tests": [],
        "benjamini_hochberg_confirmatory": [],
    }

    for cid, (folder, train_dom, test_dom, _in_dom) in CONDITIONS.items():
        y, p = data[cid]
        print(f"Bootstrap {cid} (n={len(y)}) …", flush=True)
        report["per_condition"][cid] = {
            "folder": folder,
            "trained_on": train_dom,
            "tested_on": test_dom,
            "n_test": int(len(y)),
            "metrics": _bootstrap_metrics(
                y, p, n_boot=n_boot, seed=seed + hash(cid) % 10_000, workers=workers
            ),
        }

    # PCam-trained: C1 in-domain, C2 external (different test domains — unpaired sizes)
    report["transfer_drops"]["PCam_trained"] = {
        "in_domain": "C1",
        "external": "C2",
        "note": "Different test splits (PCam vs CAM17); drop CIs are marginal per arm, not paired.",
        "marginal_delta_abs": {},
    }
    m1 = report["per_condition"]["C1"]["metrics"]["point"]
    m2 = report["per_condition"]["C2"]["metrics"]["point"]
    for k in METRIC_KEYS + ["ece_15"]:
        a, b = m1.get(k, float("nan")), m2.get(k, float("nan"))
        if not (np.isnan(a) or np.isnan(b)):
            report["transfer_drops"]["PCam_trained"]["marginal_delta_abs"][k] = {
                "point_in": a,
                "point_ext": b,
                "point_delta_in_minus_ext": a - b,
                "ci95_in": report["per_condition"]["C1"]["metrics"]["ci95"].get(k),
                "ci95_ext": report["per_condition"]["C2"]["metrics"]["ci95"].get(k),
            }

    # CAM17-trained: C3 vs C4 — different test sizes, same note
    report["transfer_drops"]["CAMELYON17_trained"] = {
        "in_domain": "C3",
        "external": "C4",
        "note": "Different test splits; marginal deltas only.",
        "marginal_delta_abs": {},
    }
    m3 = report["per_condition"]["C3"]["metrics"]["point"]
    m4 = report["per_condition"]["C4"]["metrics"]["point"]
    for k in METRIC_KEYS + ["ece_15"]:
        a, b = m3.get(k, float("nan")), m4.get(k, float("nan"))
        if not (np.isnan(a) or np.isnan(b)):
            report["transfer_drops"]["CAMELYON17_trained"]["marginal_delta_abs"][k] = {
                "point_in": a,
                "point_ext": b,
                "point_delta_in_minus_ext": a - b,
                "ci95_in": report["per_condition"]["C3"]["metrics"]["ci95"].get(k),
                "ci95_ext": report["per_condition"]["C4"]["metrics"]["ci95"].get(k),
            }

    # Paired transfer on SAME test domain: compare models on identical patches
    # PCam test: C1 (PCam train) vs C4 (CAM17 train)
    y1, p1 = data["C1"]
    y4, p4 = data["C4"]
    print("Paired transfer drop on PCam test (C1 vs C4) …", flush=True)
    report["transfer_drops"]["paired_on_PCam_test"] = _paired_transfer_drop(
        y1, p1, y4, p4, n_boot=n_boot, seed=seed + 101, workers=workers
    )
    report["transfer_drops"]["paired_on_PCam_test"]["interpretation"] = (
        "Each replicate: M from C1 minus M from C4 on the same resampled PCam test patches "
        "(PCam-trained vs CAMELYON17-trained)."
    )

    # CAM17 test: C3 vs C2
    y3, p3 = data["C3"]
    y2, p2 = data["C2"]
    print("Paired transfer drop on CAM17 test (C3 vs C2) …", flush=True)
    report["transfer_drops"]["paired_on_CAM17_test"] = _paired_transfer_drop(
        y3, p3, y2, p2, n_boot=n_boot, seed=seed + 202, workers=workers
    )
    report["transfer_drops"]["paired_on_CAM17_test"]["interpretation"] = (
        "Each replicate: M from C3 minus M from C2 on the same resampled CAM17 test patches."
    )

    # Asymmetry: (drop_PCam_train_on_PCam_test) vs (drop_CAM17_train_on_CAM17_test) — use paired drops ROC/acc
    # Bootstrap difference of marginal drops is approximate; use paired-on-same-test contrasts instead.
    for metric in ("roc_auc", "accuracy", "brier_score", "ece_15"):
        d_pcam = report["transfer_drops"]["paired_on_PCam_test"]["delta_abs"][metric]
        d_cam = report["transfer_drops"]["paired_on_CAM17_test"]["delta_abs"][metric]
        report["drop_asymmetry"][metric] = {
            "paired_PCam_test_delta_C1_minus_C4": d_pcam.get("point_delta"),
            "paired_CAM17_test_delta_C3_minus_C2": d_cam.get("point_delta"),
            "note": "Asymmetry read qualitatively from transfer narrative; formal joint test not prespecified.",
        }

    # Head-to-head permutation on same test (confirmatory family: brier + log-loss components)
    perm_tests: List[Dict[str, Any]] = []
    pvals_for_bh: List[Tuple[str, float]] = []

    def _add_perm(test_id: str, y: np.ndarray, pa: np.ndarray, pb: np.ndarray, label: str) -> None:
        db = _per_case_brier(y, pa) - _per_case_brier(y, pb)
        dn = _per_case_nll(y, pa) - _per_case_nll(y, pb)
        pb_b = _paired_permutation_p(db, n_perm=n_perm, seed=seed + hash(test_id + "b") % 99991)
        pb_n = _paired_permutation_p(dn, n_perm=n_perm, seed=seed + hash(test_id + "n") % 99991)
        perm_tests.append(
            {
                "test_id": f"{test_id}_brier",
                "comparison": label,
                "metric": "mean per-case Brier contribution (A minus B)",
                "p_value": pb_b,
                "tier": "confirmatory",
            }
        )
        perm_tests.append(
            {
                "test_id": f"{test_id}_log_loss",
                "comparison": label,
                "metric": "mean per-case NLL contribution (A minus B)",
                "p_value": pb_n,
                "tier": "confirmatory",
            }
        )
        pvals_for_bh.append((f"{test_id}_brier", pb_b))
        pvals_for_bh.append((f"{test_id}_log_loss", pb_n))

    _add_perm(
        "PCam_test_C1_vs_C4",
        y1,
        p1,
        p4,
        "PCam-trained (C1) vs CAMELYON17-trained (C4) on PCam test",
    )
    _add_perm(
        "CAM17_test_C3_vs_C2",
        y3,
        p3,
        p2,
        "CAMELYON17-trained (C3) vs PCam-trained (C2) on CAMELYON17 test",
    )

    report["paired_permutation_tests"] = perm_tests
    report["benjamini_hochberg_confirmatory"] = _benjamini_hochberg(pvals_for_bh)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = out_json.with_suffix(".csv")
    _write_summary_csv(report, csv_path)
    return report


def _write_summary_csv(report: Dict[str, Any], path: Path) -> None:
    lines = ["section,condition,metric,point,ci95_low,ci95_high"]
    for cid, block in report["per_condition"].items():
        pt = block["metrics"]["point"]
        ci = block["metrics"]["ci95"]
        for k in METRIC_KEYS + ["ece_15"]:
            c = ci.get(k, {})
            lines.append(
                f"bootstrap,{cid},{k},{pt.get(k)},{c.get('low')},{c.get('high')}"
            )
    for block_name in ("paired_on_PCam_test", "paired_on_CAM17_test"):
        td = report["transfer_drops"].get(block_name, {}).get("delta_abs", {})
        for k, v in td.items():
            if isinstance(v, dict) and "point_delta" in v:
                lines.append(
                    f"paired_transfer,{block_name},{k},{v.get('point_delta')},"
                    f"{v.get('ci95_low')},{v.get('ci95_high')}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap / permutation inference for Virchow test exports.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutation", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for bootstrap chunks.")
    args = parser.parse_args()

    print(f"Eval root: {args.eval_root}", flush=True)
    print(
        f"Bootstrap B={args.n_bootstrap}, permutation={args.n_permutation}, "
        f"seed={args.seed}, workers={args.workers}",
        flush=True,
    )
    report = run(
        args.eval_root,
        args.out,
        n_boot=args.n_bootstrap,
        n_perm=args.n_permutation,
        seed=args.seed,
        workers=args.workers,
    )
    print(f"Wrote {args.out}", flush=True)
    print(f"Wrote {args.out.with_suffix('.csv')}", flush=True)
    # Headline
    c4 = report["per_condition"]["C4"]["metrics"]["point"]
    print(f"C4 calibrated ROC-AUC point {c4['roc_auc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
