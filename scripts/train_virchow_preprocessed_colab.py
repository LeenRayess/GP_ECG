"""Colab-ready Virchow2 training with robust checkpointing to Google Drive.

Uncertainty: optional dropout before the linear head + MC Dropout at eval (epistemic proxy).
Metrics: AUC-ROC, AUC-PR, Brier, log loss, ECE; optional temperature scaling on val.
Exports: metrics_history.json, metrics_final_detailed.json, val_predictions.npz, run_manifest.json.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
except ImportError:
    roc_auc_score = None  # type: ignore

EPOCHS = 10
BATCH_SIZE = 64
EMBED_DIM = 2560
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pct_milestones_crossed(batch_1based: int, n_batches: int, printed: set) -> List[int]:
    """Newly crossed 10%..100%% batch milestones; mutates printed."""
    out: List[int] = []
    if n_batches <= 0:
        return out
    frac = batch_1based / float(n_batches)
    for k in range(10, 101, 10):
        if k in printed:
            continue
        if frac + 1e-12 >= k / 100.0:
            printed.add(k)
            out.append(k)
    return out


class PreprocessedPCamDataset(Dataset):
    """HDF5 PCam patches; resize to 224 and ImageNet-normalize."""

    def __init__(self, x_path, y_path, resize_size=224):
        import h5py

        self.x_path = x_path
        self.y_path = y_path
        self.resize_size = resize_size
        with h5py.File(y_path, "r") as f:
            self.n = f["y"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        import h5py

        with h5py.File(self.x_path, "r") as fx, h5py.File(self.y_path, "r") as fy:
            x = np.array(fx["x"][idx], dtype=np.float32)
            y = float(np.array(fy["y"][idx]).reshape(-1)[0])

        x = torch.from_numpy(x).permute(2, 0, 1)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.resize_size, self.resize_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        x = x.clamp(0, 1)
        mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=x.dtype).view(3, 1, 1)
        x = (x - mean) / std
        return x, torch.tensor(y, dtype=torch.float32)


def get_embedding(backbone, x):
    """x: (B, 3, 224, 224). Returns (B, 2560)."""
    with torch.no_grad():
        out = backbone(x)
    class_tok = out[:, 0]
    patch_tok = out[:, 5:]
    patch_mean = patch_tok.mean(dim=1)
    return torch.cat([class_tok, patch_mean], dim=-1)


class VirchowClassifier(nn.Module):
    """Frozen Virchow2 + optional dropout (for MC Dropout) + Linear(2560, 1)."""

    def __init__(self, backbone, embed_dim=EMBED_DIM, head_dropout_p=0.0):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head_dropout_p = float(head_dropout_p)
        self.dropout = nn.Dropout(self.head_dropout_p) if self.head_dropout_p > 0 else None
        self.head = nn.Linear(embed_dim, 1)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward_logits(self, x: torch.Tensor, *, mc_dropout: bool = False) -> torch.Tensor:
        """If mc_dropout=True, dropout stays in train mode; backbone always eval."""
        self.backbone.eval()
        emb = get_embedding(self.backbone, x)
        if self.dropout is not None:
            if mc_dropout:
                self.dropout.train()
            else:
                self.dropout.eval()
            emb = self.dropout(emb)
        return self.head(emb).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(x, mc_dropout=False)


def _confusion_from_probs(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    pred = (p >= threshold).astype(np.float64)
    y = y.reshape(-1)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Rich scalar metrics for binary problems; safe with sklearn missing or degenerate labels."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)

    out: Dict[str, Any] = {}
    cm = _confusion_from_probs(y_true, y_prob, threshold)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    out["confusion_at_threshold"] = cm
    out["threshold"] = threshold
    out["accuracy"] = float((tp + tn) / max(tp + tn + fp + fn, 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    out["precision"] = float(prec)
    out["recall"] = float(rec)
    out["f1"] = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    if roc_auc_score is None:
        out["roc_auc"] = None
        out["average_precision"] = None
        out["brier_score"] = None
        out["log_loss"] = None
        return out

    uniq = np.unique(y_true)
    if len(uniq) < 2:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["average_precision"] = float(average_precision_score(y_true, y_prob))
    out["brier_score"] = float(brier_score_loss(y_true, y_prob))
    try:
        out["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        out["log_loss"] = float("nan")

    # Balanced accuracy
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    out["balanced_accuracy"] = float(0.5 * (sens + spec))
    return out


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Standard ECE for binary: bin by predicted probability."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (y_prob >= lo) & (y_prob <= hi)
        else:
            m = (y_prob >= lo) & (y_prob < hi)
        cnt = int(m.sum())
        if cnt == 0:
            bin_details.append({"bin": [float(lo), float(hi)], "n": 0, "conf": None, "acc": None})
            continue
        conf = float(y_prob[m].mean())
        acc = float(y_true[m].mean())
        w = cnt / max(n, 1)
        ece += w * abs(conf - acc)
        bin_details.append({"bin": [float(lo), float(hi)], "n": cnt, "conf": conf, "acc": acc})
    return {"ece": float(ece), "n_bins": n_bins, "bins": bin_details}


def fit_temperature_binary(
    logits: np.ndarray,
    y_true: np.ndarray,
    grid: int = 60,
) -> Dict[str, Any]:
    """Grid search T>0 minimizing binary cross-entropy on logits/T."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    Ts = np.logspace(math.log10(0.2), math.log10(5.0), grid)
    best_T, best_nll = 1.0, float("inf")
    for T in Ts:
        z = logits / T
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        nll = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        nll_m = float(nll.mean())
        if nll_m < best_nll:
            best_nll = nll_m
            best_T = float(T)
    z0 = logits
    p0 = 1.0 / (1.0 + np.exp(-z0))
    p0 = np.clip(p0, 1e-7, 1.0 - 1e-7)
    nll_raw = float(-(y_true * np.log(p0) + (1 - y_true) * np.log(1 - p0)).mean())
    return {"temperature": best_T, "nll_logits_raw_mean": nll_raw, "nll_after_T_mean": best_nll}


@torch.no_grad()
def collect_logits_labels(
    model: VirchowClassifier,
    loader: DataLoader,
    device: torch.device,
    desc: str = "collect",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    n_batches = len(loader)
    printed: set = set()
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        logits = model.forward_logits(x, mc_dropout=False)
        logits_list.append(logits.cpu())
        labels_list.append(y)
        b = batch_idx + 1
        for pct in _pct_milestones_crossed(b, n_batches, printed):
            print("  [{}] {}%  batch {}/{}".format(desc, pct, b, n_batches))
    logits_np = torch.cat(logits_list, dim=0).numpy()
    labels_np = torch.cat(labels_list, dim=0).numpy().reshape(-1)
    return logits_np, labels_np


def collect_mc_dropout_probs(
    model: VirchowClassifier,
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
    desc: str = "mc",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns mean_prob, std_prob, stack (n_samples x n_val)."""
    if model.dropout is None or n_samples <= 0:
        raise ValueError("MC Dropout requires head dropout > 0 and n_samples > 0")
    all_means = []
    for si in range(n_samples):
        logits_list: List[torch.Tensor] = []
        n_batches = len(loader)
        printed: set = set()
        tag = "{} sample {}/{}".format(desc, si + 1, n_samples)
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                lg = model.forward_logits(x, mc_dropout=True)
            logits_list.append(lg.cpu())
            b = batch_idx + 1
            for pct in _pct_milestones_crossed(b, n_batches, printed):
                print("  [{}] {}%  batch {}/{}".format(tag, pct, b, n_batches))
        logits_ep = torch.cat(logits_list, dim=0)
        prob_ep = torch.sigmoid(logits_ep).numpy().reshape(-1)
        all_means.append(prob_ep)
    stack = np.stack(all_means, axis=0)
    mean_p = stack.mean(axis=0)
    std_p = stack.std(axis=0)
    return mean_p, std_p, stack


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    total_epochs,
):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    n_batches = len(loader)
    tag = "Epoch {}/{} train".format(epoch + 1, total_epochs)
    printed: set = set()
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += (pred == y).sum().item()
        n += bsz

        running_loss = total_loss / max(n, 1)
        running_acc = correct / max(n, 1)
        bi = batch_idx + 1
        for pct in _pct_milestones_crossed(bi, n_batches, printed):
            print(
                "  [{}] {}%  batch {}/{}  loss: {:.4f}  acc: {:.4f}".format(
                    tag, pct, bi, n_batches, running_loss, running_acc
                )
            )

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate_epoch(
    model: VirchowClassifier,
    loader: DataLoader,
    device: torch.device,
    desc: str = "val",
) -> Tuple[float, float, Dict[str, Any], Dict[str, Any]]:
    """Returns loss, acc, confusion @0.5, sklearn-style metrics dict (no MC)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    n_batches = len(loader)
    printed: set = set()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model.forward_logits(x, mc_dropout=False)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total_loss += loss.item() * x.size(0)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        correct += (pred == y).sum().item()
        n += x.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
        bi = batch_idx + 1
        running_loss = total_loss / max(n, 1)
        running_acc = correct / max(n, 1)
        for pct in _pct_milestones_crossed(bi, n_batches, printed):
            print(
                "  [{}] {}%  batch {}/{}  loss: {:.4f}  acc: {:.4f}".format(
                    desc, pct, bi, n_batches, running_loss, running_acc
                )
            )

    logits_np = torch.cat(all_logits, dim=0).numpy().reshape(-1)
    labels_np = torch.cat(all_labels, dim=0).numpy().reshape(-1)
    prob_np = 1.0 / (1.0 + np.exp(-logits_np))

    pred = (prob_np >= 0.5).astype(np.float64)
    tn = int(((pred == 0) & (labels_np == 0)).sum())
    fp = int(((pred == 1) & (labels_np == 0)).sum())
    fn = int(((pred == 0) & (labels_np == 1)).sum())
    tp = int(((pred == 1) & (labels_np == 1)).sum())
    cm = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    metrics = compute_classification_metrics(labels_np, prob_np, threshold=0.5)
    metrics["ece_15"] = expected_calibration_error(labels_np, prob_np, n_bins=15)

    return total_loss / max(n, 1), correct / max(n, 1), cm, metrics


def print_confusion_and_metrics(tp, tn, fp, fn, split_name="val"):
    print("\n  [{}] Confusion matrix (predicted columns, true rows):".format(split_name))
    print("                    pred_neg  pred_pos")
    print("  true_neg (0)      {:>8}  {:>8}".format(tn, fp))
    print("  true_pos (1)      {:>8}  {:>8}".format(fn, tp))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print("  precision (pos): {:.4f}  recall (pos): {:.4f}  F1: {:.4f}".format(prec, rec, f1))


def _atomic_json_dump(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _save_epoch_checkpoint(out_dir: Path, payload: dict, epoch: int, save_every_epoch_copy: bool) -> None:
    ckpt_last = out_dir / "checkpoint_last.pt"
    torch.save(payload, ckpt_last)
    if save_every_epoch_copy:
        ckpt_epoch = out_dir / "checkpoints" / "checkpoint_epoch_{:03d}.pt".format(epoch + 1)
        ckpt_epoch.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ckpt_epoch)


def _load_resume_checkpoint(out_dir: Path, device: torch.device):
    preferred = [out_dir / "checkpoint_last.pt", out_dir / "checkpoint.pt"]
    for p in preferred:
        if p.is_file():
            return torch.load(p, map_location=device), p
    return None, None


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def export_final_artifacts(
    model: VirchowClassifier,
    valid_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    mc_samples: int,
    save_val_npz: bool,
) -> None:
    """After training: full val logits, temperature fit, optional MC, npz + detailed JSON."""
    print("\nExporting final validation artifacts ...")
    logits_np, labels_np = collect_logits_labels(model, valid_loader, device, desc="final val logits")
    prob_np = 1.0 / (1.0 + np.exp(-logits_np))

    temp_info = fit_temperature_binary(logits_np, labels_np)
    T = temp_info["temperature"]
    logits_scaled = logits_np / T
    prob_cal = 1.0 / (1.0 + np.exp(-logits_scaled))

    metrics_raw = compute_classification_metrics(labels_np, prob_np, threshold=0.5)
    metrics_cal = compute_classification_metrics(labels_np, prob_cal, threshold=0.5)
    ece_raw = expected_calibration_error(labels_np, prob_np, n_bins=15)
    ece_cal = expected_calibration_error(labels_np, prob_cal, n_bins=15)

    export: Dict[str, Any] = {
        "temperature_scaling_val": temp_info,
        "metrics_val_prob_raw_sigmoid": metrics_raw,
        "metrics_val_prob_after_temperature": metrics_cal,
        "ece_15_bins_raw": ece_raw,
        "ece_15_bins_temperature_scaled": ece_cal,
    }

    mc_mean = mc_std = None
    mc_stack = None
    if mc_samples > 0 and model.dropout is not None:
        mc_mean, mc_std, mc_stack = collect_mc_dropout_probs(
            model, valid_loader, device, mc_samples, desc="mc val"
        )
        export["mc_dropout"] = {
            "n_samples": mc_samples,
            "prob_mean_summary": {
                "mean": float(mc_mean.mean()),
                "std": float(mc_mean.std()),
            },
            "epistemic_std_summary": {
                "mean": float(mc_std.mean()),
                "std": float(mc_std.std()),
            },
        }
        metrics_mc = compute_classification_metrics(labels_np, mc_mean, threshold=0.5)
        export["metrics_val_mc_dropout_mean_prob"] = metrics_mc
        export["ece_15_bins_mc_mean_prob"] = expected_calibration_error(labels_np, mc_mean, n_bins=15)

    _atomic_json_dump(out_dir / "metrics_final_detailed.json", _json_safe(export))

    run_manifest = {
        "created_by": "train_virchow_preprocessed_colab.py export_final_artifacts",
        "n_val": int(len(labels_np)),
        "files_written": [
            "metrics_final_detailed.json",
            "temperature_fit.json",
            "val_predictions.npz (if enabled)",
        ],
    }
    _atomic_json_dump(out_dir / "run_manifest.json", run_manifest)
    _atomic_json_dump(out_dir / "temperature_fit.json", _json_safe(temp_info))

    if save_val_npz:
        npz_payload = {
            "y_true": labels_np.astype(np.float32),
            "logits": logits_np.astype(np.float32),
            "prob_sigmoid": prob_np.astype(np.float32),
            "logits_temperature_scaled": logits_scaled.astype(np.float32),
            "prob_after_temperature": prob_cal.astype(np.float32),
            "temperature_T": np.float32(T),
        }
        if mc_mean is not None:
            npz_payload["prob_mc_mean"] = mc_mean.astype(np.float32)
            npz_payload["prob_mc_std"] = mc_std.astype(np.float32)
        out_npz = out_dir / "val_predictions.npz"
        np.savez_compressed(out_npz, **npz_payload)
        print("Wrote", out_npz, "(arrays: {})".format(", ".join(npz_payload.keys())))

    print("Wrote", out_dir / "metrics_final_detailed.json")


def main():
    parser = argparse.ArgumentParser(description="Colab-ready Virchow2 (frozen) training on preprocessed PCam.")
    parser.add_argument("--preprocessed-dir", type=str, required=True, help="Folder with train_x/y.h5 and valid_x/y.h5")
    parser.add_argument("--out-dir", type=str, required=True, help="Output folder (use Google Drive path on Colab)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_last.pt in --out-dir")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (Colab typically 2-4)")
    parser.add_argument("--save-every-epoch-copy", action="store_true", help="Also save checkpoints/checkpoint_epoch_XXX.pt")
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.2,
        help="Dropout before linear head (for MC Dropout). Use 0 to disable (old checkpoint compatibility).",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=30,
        help="MC Dropout forward passes on val at export (0=skip). Only if head-dropout>0.",
    )
    parser.add_argument(
        "--no-save-val-preds",
        action="store_true",
        help="Do not write val_predictions.npz (large). Metrics JSON still written.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip final detailed export (faster exit; not recommended for documentation).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x = os.path.join(args.preprocessed_dir, "train_x.h5")
    train_y = os.path.join(args.preprocessed_dir, "train_y.h5")
    valid_x = os.path.join(args.preprocessed_dir, "valid_x.h5")
    valid_y = os.path.join(args.preprocessed_dir, "valid_y.h5")
    for p in [train_x, train_y, valid_x, valid_y]:
        if not os.path.isfile(p):
            print("Error: missing", p)
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = PreprocessedPCamDataset(train_x, train_y)
    valid_ds = PreprocessedPCamDataset(valid_x, valid_y)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    try:
        from timm.layers import SwiGLUPacked
        import timm
    except ImportError:
        print("Error: install timm and dependencies (pip install timm)")
        sys.exit(1)

    print("Loading Virchow2 backbone (hf-hub:paige-ai/Virchow2) ...")
    try:
        backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        backbone = backbone.to(device).eval()
        model = VirchowClassifier(backbone, head_dropout_p=args.head_dropout).to(device)
    except Exception as e:
        print("Error: Virchow2 load failed:", e)
        print("Tip: in Colab, run `huggingface-cli login` first if model access is gated.")
        sys.exit(1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)

    history_path = out_dir / "metrics_history.json"
    run_config_path = out_dir / "run_config.json"
    checkpoint_path_compat = out_dir / "checkpoint.pt"

    run_config = {
        "script": "train_virchow_preprocessed_colab.py",
        "preprocessed_dir": str(Path(args.preprocessed_dir).resolve()),
        "out_dir": str(out_dir.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "device": str(device),
        "head_dropout": args.head_dropout,
        "mc_samples_export": args.mc_samples,
        "save_val_predictions_npz": not args.no_save_val_preds,
    }
    _atomic_json_dump(run_config_path, run_config)

    history: List[Dict[str, Any]] = []
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    start_epoch = 0
    best_val_acc = -1.0

    if args.resume:
        ckpt, ckpt_path = _load_resume_checkpoint(out_dir, device)
        if ckpt is not None:
            _inc = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            _mk = getattr(_inc, "missing_keys", None) if _inc is not None else None
            _uk = getattr(_inc, "unexpected_keys", None) if _inc is not None else None
            if _mk or _uk:
                print("load_state_dict (strict=False) missing_keys:", _mk, "unexpected_keys:", _uk)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_val_acc = float(ckpt.get("best_val_acc", -1.0))
            print("Resumed from:", ckpt_path)
            print("Start epoch:", start_epoch, "| best_val_acc:", best_val_acc)
        else:
            print("Resume requested but no checkpoint found; starting from epoch 0.")

    if start_epoch >= args.epochs:
        print("Training already complete for requested epochs.")
        if not args.skip_export:
            export_final_artifacts(
                model,
                valid_loader,
                device,
                out_dir,
                mc_samples=args.mc_samples if args.head_dropout > 0 else 0,
                save_val_npz=not args.no_save_val_preds,
            )
        return

    print("\nTraining epochs {}..{} (inclusive), train samples: {}, val samples: {}".format(
        start_epoch + 1, args.epochs, len(train_ds), len(valid_ds)
    ))

    for epoch in range(start_epoch, args.epochs):
        print("\n" + "=" * 60)
        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        print("=" * 60)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_loss, val_acc, val_cm, val_metrics = evaluate_epoch(
            model, valid_loader, device, desc="Epoch {}/{} val".format(epoch + 1, args.epochs)
        )

        # Compact metrics for history (drop huge ECE bin list)
        val_metrics_compact = dict(val_metrics)
        ece_val = val_metrics.get("ece_15", {})
        if isinstance(ece_val, dict):
            val_metrics_compact["ece_15"] = {"ece": ece_val.get("ece"), "n_bins": ece_val.get("n_bins")}
        else:
            val_metrics_compact["ece_15"] = None

        print("\n  Epoch {} summary: train loss {:.4f}  train acc {:.4f}  val loss {:.4f}  val acc {:.4f}".format(
            epoch + 1, train_loss, train_acc, val_loss, val_acc
        ))
        if val_metrics.get("roc_auc") is not None:
            print("  val ROC-AUC: {:.4f}  PR-AUC: {:.4f}  Brier: {:.4f}".format(
                val_metrics.get("roc_auc", float("nan")),
                val_metrics.get("average_precision", float("nan")),
                val_metrics.get("brier_score", float("nan")),
            ))
        print_confusion_and_metrics(val_cm["tp"], val_cm["tn"], val_cm["fp"], val_cm["fn"], split_name="val")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "model_best.pt")
            print("  -> new best val acc, saved model_best.pt")

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_metrics": val_metrics_compact,
            "head_dropout": args.head_dropout,
        }
        _save_epoch_checkpoint(out_dir, checkpoint_payload, epoch, args.save_every_epoch_copy)
        torch.save(checkpoint_payload, checkpoint_path_compat)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc_so_far": best_val_acc,
                "val_confusion": val_cm,
                "val_metrics": val_metrics_compact,
            }
        )
        _atomic_json_dump(history_path, _json_safe(history))
        _atomic_json_dump(out_dir / "run_progress.json", {"last_completed_epoch": epoch + 1, "best_val_acc": best_val_acc})

        # Per-epoch rich metrics (without full ECE bin arrays)
        _atomic_json_dump(
            out_dir / "metrics_epoch_{:03d}.json".format(epoch + 1),
            _json_safe({"epoch": epoch + 1, "val_metrics": val_metrics_compact, "val_confusion": val_cm}),
        )

        print("  checkpoint saved:", out_dir / "checkpoint_last.pt")

    metrics_final = {
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "preprocessed_dir": str(Path(args.preprocessed_dir).resolve()),
        "out_dir": str(out_dir.resolve()),
        "head_dropout": args.head_dropout,
    }
    _atomic_json_dump(out_dir / "metrics_final.json", metrics_final)
    print("\nDone training. Best val acc:", best_val_acc)

    # Reload best weights for export (matches val selection criterion)
    best_path = out_dir / "model_best.pt"
    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("Loaded model_best.pt for final export.")
    if not args.skip_export:
        export_final_artifacts(
            model,
            valid_loader,
            device,
            out_dir,
            mc_samples=args.mc_samples if args.head_dropout > 0 else 0,
            save_val_npz=not args.no_save_val_preds,
        )
    print("Saved:", out_dir / "metrics_final.json")


if __name__ == "__main__":
    main()
