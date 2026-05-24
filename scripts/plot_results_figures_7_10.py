"""Generate Results figures 7–10 (stain benchmark, confusion matrices, error pools)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

STAIN_JSON = REPO / "reports" / "stain_benchmark_c5_c6_metrics.json"
EVAL_ROOT = REPO / "experiments" / "virchow_colab" / "evals_cross_domain"
QUAL_ROOT = REPO / "reports" / "qualitative_error_analysis" / "virchow_c1_c4"

VIRCHOW_PANELS = {
    "C1": ("pcam_trained_on_pcam_test", "C1: PCam → PCam"),
    "C2": ("pcam_trained_on_cam17_test", "C2: PCam → CAMELYON17"),
    "C3": ("cam17_trained_on_cam17_test", "C3: CAMELYON17 → CAMELYON17"),
    "C4": ("cam17_trained_on_pcam_test", "C4: CAMELYON17 → PCam"),
}

QUAL_DIRS = {
    "C1": "C1_pcam_trained_on_pcam_test",
    "C2": "C2_pcam_trained_on_cam17_test",
    "C3": "C3_cam17_trained_on_cam17_test",
    "C4": "C4_cam17_trained_on_pcam_test",
}

# Shallow CNN preprocessed @ 0.5 (final_results.md Table 11) — fallback if metrics JSON absent
CNN_CONFUSION_FALLBACK = {
    "PCam → PCam": dict(tp=10350, tn=13766, fp=998, fn=2590),
    "PCam → CAMELYON17": dict(tp=40998, tn=36054, fp=4659, fn=1470),
    "CAMELYON17 → CAMELYON17": dict(tp=38620, tn=39770, fp=943, fn=3848),
    "CAMELYON17 → PCam": dict(tp=4686, tn=14638, fp=126, fn=8254),
}


def _save(fig: plt.Figure, stem: str) -> None:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote:", png)
    print("Wrote:", pdf)


def _cm_array(cm: dict[str, int]) -> np.ndarray:
    """Rows = true class (neg, pos); cols = predicted (neg, pos)."""
    return np.array(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        dtype=float,
    )


def _load_virchow_cm(folder: str) -> dict[str, int]:
    path = EVAL_ROOT / folder / "test_metrics_detailed.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["metrics_test_prob_after_temperature"]["confusion_at_threshold"]


def _plot_confusion_grid(
    panels: list[tuple[str, dict[str, int]]],
    suptitle: str,
    stem: str,
    *,
    normalize_rows: bool = True,
) -> None:
    n = len(panels)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 3.4 * nrows), squeeze=False)
    vmax = 1.0 if normalize_rows else None

    for ax, (title, cm) in zip(axes.flat, panels):
        arr = _cm_array(cm)
        if normalize_rows:
            row_sum = arr.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            display = arr / row_sum
            fmt = ".0%"
            cbar_label = "Row fraction"
        else:
            display = arr
            fmt = ".0f"
            cbar_label = "Count"

        im = ax.imshow(display, cmap="Blues", vmin=0, vmax=vmax if vmax else display.max())
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred −", "Pred +"], fontsize=9)
        ax.set_yticklabels(["True −", "True +"], fontsize=9)
        ax.set_title(title, fontsize=10)

        for i in range(2):
            for j in range(2):
                val = display[i, j]
                count = int(arr[i, j])
                if normalize_rows:
                    text = f"{val:.1%}\n(n={count:,})"
                else:
                    text = f"{count:,}"
                color = "white" if val > 0.55 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    for ax in axes.flat[len(panels) :]:
        ax.axis("off")

    fig.subplots_adjust(right=0.88, top=0.92 if nrows > 1 else 0.88, hspace=0.35, wspace=0.25)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax, label=cbar_label)
    fig.suptitle(suptitle, fontsize=11)
    _save(fig, stem)


def _stain_arm_short(arm: str) -> str:
    if "Macenko" in arm:
        return "Macenko"
    if "Reinhard" in arm:
        return "Reinhard"
    if "Vahadane" in arm:
        return "Vahadane"
    if "single-reference" in arm:
        return "Adaptive 1-ref"
    if "augmentation" in arm:
        return "Adaptive multi + aug"
    if "multi-reference" in arm:
        return "Adaptive multi-ref"
    return arm


def plot_figure7_stain() -> None:
    with open(STAIN_JSON, encoding="utf-8") as f:
        rows = json.load(f)["rows"]

    labels = [f"{r['condition']}: {_stain_arm_short(r['arm'])}" for r in rows]
    colors = ["#4C72B0" if r["condition"] == "C5" else "#DD8452" for r in rows]
    order = np.argsort([r["test_roc_auc"] for r in rows])
    labels = [labels[i] for i in order]
    colors = [colors[i] for i in order]
    rows = [rows[i] for i in order]

    metrics = [
        ("val_roc_auc", "Validation ROC-AUC"),
        ("test_roc_auc", "Test ROC-AUC"),
        ("test_accuracy", "Test accuracy @ 0.5"),
        ("test_f1", "Test F1 @ 0.5"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), sharey=True)
    y = np.arange(len(labels))
    macenko_test_auc = next(r["test_roc_auc"] for r in rows if "Macenko" in r["arm"])

    for ax, (key, title) in zip(axes.flat, metrics):
        vals = [r[key] for r in rows]
        lo, hi = min(vals), max(vals)
        pad = max(0.02, (hi - lo) * 0.12)
        ax.barh(y, vals, color=colors, edgecolor="0.3", linewidth=0.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Score")
        if key == "test_roc_auc":
            ax.axvline(macenko_test_auc, color="#2ca02c", ls="--", lw=1, alpha=0.85)
        for yi, val in zip(y, vals):
            ax.text(val + pad * 0.08, yi, f"{val:.3f}", va="center", fontsize=7)

    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels(labels, fontsize=8)
    for ax in axes[1, :]:
        ax.set_yticklabels([])

    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor="#4C72B0", label="C5 classical"),
            Patch(facecolor="#DD8452", label="C6 adaptive"),
            plt.Line2D([0], [0], color="#2ca02c", ls="--", label="Macenko test ROC-AUC"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Stain-handling benchmark (shallow CNN, PCam subset)", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, "figure7_stain_benchmark_metrics")


def plot_figure8_virchow() -> None:
    panels = []
    for cid, (folder, title) in VIRCHOW_PANELS.items():
        cm = _load_virchow_cm(folder)
        panels.append((title, cm))
    _plot_confusion_grid(
        panels,
        "Virchow2 confusion matrices (calibrated, threshold 0.5)",
        "figure8_virchow_confusion_matrices_c1_c4",
    )


def plot_figure9_cnn() -> None:
    panels = [(k, v) for k, v in CNN_CONFUSION_FALLBACK.items()]
    _plot_confusion_grid(
        panels,
        "Shallow CNN confusion matrices (preprocessed, calibrated @ 0.5)",
        "figure9_cnn_confusion_matrices_preprocessed",
    )


def plot_figure10_error_pools() -> None:
    conditions = []
    fn_avail = []
    fp_avail = []
    he_err = []

    for cid, subdir in QUAL_DIRS.items():
        path = QUAL_ROOT / subdir / "bucket_sampling_summary.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        buckets = {b["bucket"]: b["available_n"] for b in data["buckets"]}
        conditions.append(cid)
        fn_avail.append(buckets["FN"])
        fp_avail.append(buckets["FP"])
        he_err.append(buckets["high_entropy_error"])

    x = np.arange(len(conditions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x - width, fn_avail, width, label="FN pool", color="#C44E52")
    ax.bar(x, fp_avail, width, label="FP pool", color="#4C72B0")
    ax.bar(x + width, he_err, width, label="High-entropy error pool", color="#8172B2")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Patches available on full test split")
    ax.set_xlabel("Virchow2 condition")
    ax.set_title("Qualitative review error pools (before sampling)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(50, 10000)
    for i, (fn, fp) in enumerate(zip(fn_avail, fp_avail)):
        if fn > 500:
            ax.annotate(f"{fn:,}", (x[i] - width, fn), ha="center", va="bottom", fontsize=7, rotation=90)
    fig.tight_layout()
    _save(fig, "figure10_qualitative_error_pools_c1_c4")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
        }
    )
    plot_figure7_stain()
    plot_figure8_virchow()
    plot_figure9_cnn()
    plot_figure10_error_pools()
    print("Done.")


if __name__ == "__main__":
    main()
