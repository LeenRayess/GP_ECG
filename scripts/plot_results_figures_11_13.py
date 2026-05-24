"""Generate Results figures 11–13 (transfer grid, raw vs preprocessed, transfer gap bars)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

# Preprocessed CNN @ 0.5 calibrated (Tables 10 / 14 layout)
TRANSFER_ROC = {
    ("PCam", "PCam"): (0.948, 0.981),
    ("PCam", "CAMELYON17"): (0.985, 0.997),
    ("CAMELYON17", "CAMELYON17"): (0.985, 0.997),
    ("CAMELYON17", "PCam"): (0.856, 0.950),
}

CNN_CONFUSION = {
    ("PCam", "PCam"): dict(tp=10350, fn=2590),
    ("PCam", "CAMELYON17"): dict(tp=40998, fn=1470),
    ("CAMELYON17", "CAMELYON17"): dict(tp=38620, fn=3848),
    ("CAMELYON17", "PCam"): dict(tp=4686, fn=8254),
}

# Raw vs preprocessed CNN (Tables 9–10, §7.3); raw recall from metrics / Table 13 back-calculation
RAW_PREPROC = {
    "PCam → PCam": dict(roc_raw=0.880, roc_pre=0.948, recall_raw=0.614, recall_pre=0.800),
    "PCam → CAMELYON17": dict(roc_raw=0.984, roc_pre=0.985, recall_raw=0.930, recall_pre=0.965),
    "CAMELYON17 → CAMELYON17": dict(roc_raw=0.724, roc_pre=0.985, recall_raw=0.16, recall_pre=0.91),
    "CAMELYON17 → PCam": dict(roc_raw=0.676, roc_pre=0.856, recall_raw=0.075, recall_pre=0.362),
}

# Table 19: external minus in-domain ROC-AUC
TRANSFER_GAP = {
    "PCam-trained": dict(virchow=0.016, cnn=0.037),
    "CAMELYON17-trained": dict(virchow=-0.046, cnn=-0.129),
}


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print("Wrote:", path)
    plt.close(fig)


def _prep_recall() -> None:
    for key, cm in CNN_CONFUSION.items():
        r = key
        label = f"{r[0]} → {r[1]}"
        if label in RAW_PREPROC:
            RAW_PREPROC[label]["recall_pre"] = round(cm["tp"] / (cm["tp"] + cm["fn"]), 3)


def plot_figure11_transfer_grid() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharey=True)
    panels = [
        (("PCam", "PCam"), axes[0, 0], "Train PCam\nTest PCam (C1)"),
        (("PCam", "CAMELYON17"), axes[0, 1], "Train PCam\nTest CAMELYON17 (C2)"),
        (("CAMELYON17", "PCam"), axes[1, 0], "Train CAMELYON17\nTest PCam (C4)"),
        (("CAMELYON17", "CAMELYON17"), axes[1, 1], "Train CAMELYON17\nTest CAMELYON17 (C3)"),
    ]
    for (train, test), ax, title in panels:
        cnn, vir = TRANSFER_ROC[(train, test)]
        vals = [cnn, vir]
        bars = ax.bar(["CNN", "Virchow2"], vals, color=["#4C72B0", "#DD8452"], width=0.55)
        ax.set_ylim(0.82, 1.01)
        ax.set_ylabel("Test ROC-AUC")
        ax.set_title(title, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("Cross-domain transfer (study-pipeline inputs, calibrated @ 0.5)", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "figure11_transfer_grid_roc_auc_cnn_virchow")


def plot_figure12_raw_vs_preprocessed() -> None:
    _prep_recall()
    labels = list(RAW_PREPROC.keys())
    x = np.arange(len(labels))
    w = 0.35

    fig, (ax_roc, ax_rec) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    roc_raw = [RAW_PREPROC[k]["roc_raw"] for k in labels]
    roc_pre = [RAW_PREPROC[k]["roc_pre"] for k in labels]
    ax_roc.bar(x - w / 2, roc_raw, w, label="Raw uint8", color="#9e9e9e")
    ax_roc.bar(x + w / 2, roc_pre, w, label="Study pipeline", color="#2ca02c")
    ax_roc.set_ylabel("ROC-AUC")
    ax_roc.set_title("Ranking metric")
    ax_roc.set_ylim(0.5, 1.02)
    ax_roc.legend(fontsize=8, loc="lower right")

    rec_raw = [RAW_PREPROC[k]["recall_raw"] for k in labels]
    rec_pre = [RAW_PREPROC[k]["recall_pre"] for k in labels]
    ax_rec.bar(x - w / 2, rec_raw, w, label="Raw uint8", color="#9e9e9e")
    ax_rec.bar(x + w / 2, rec_pre, w, label="Study pipeline", color="#2ca02c")
    for i, v in enumerate(rec_raw):
        ax_rec.text(x[i] - w / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    for i, v in enumerate(rec_pre):
        ax_rec.text(x[i] + w / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    ax_rec.set_ylabel("Recall @ 0.5")
    ax_rec.set_title("Threshold metric")
    ax_rec.set_ylim(0, 1.05)
    ax_rec.legend(fontsize=8, loc="lower right")

    for ax in (ax_roc, ax_rec):
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace(" → ", "\n→ ") for k in labels], fontsize=7)

    fig.suptitle("Shallow CNN: raw public patches vs study preprocessing pipeline", fontsize=11, y=1.03)
    fig.tight_layout()
    _save(fig, "figure12_cnn_raw_vs_preprocessed_roc_recall")


def plot_figure13_transfer_gap() -> None:
    origins = list(TRANSFER_GAP.keys())
    x = np.arange(len(origins))
    w = 0.35
    vir = [TRANSFER_GAP[o]["virchow"] for o in origins]
    cnn = [TRANSFER_GAP[o]["cnn"] for o in origins]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(x - w / 2, vir, w, label="Virchow2", color="#DD8452")
    ax.bar(x + w / 2, cnn, w, label="Shallow CNN", color="#4C72B0")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["PCam-trained\n(C2−C1 / ext−in)", "CAMELYON17-trained\n(C4−C3 / ext−in)"], fontsize=8)
    ax.set_ylabel("Δ ROC-AUC (external − in-domain)")
    ax.set_title("Cross-domain ranking drop by training origin")
    for xi, v in zip(x - w / 2, vir):
        ax.text(xi, v + (0.008 if v >= 0 else -0.015), f"{v:+.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, cnn):
        ax.text(xi, v + (0.008 if v >= 0 else -0.015), f"{v:+.3f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "figure13_transfer_gap_roc_auc_by_origin")


def main() -> None:
    plt.rcParams.update({"font.size": 10})
    plot_figure11_transfer_grid()
    plot_figure12_raw_vs_preprocessed()
    plot_figure13_transfer_gap()
    print("Done.")


if __name__ == "__main__":
    main()
