"""
Train the same baseline CNN on benchmark H5 folders (fair comparison across stain methods).

Architecture matches notebooks/pcam_baseline_training.ipynb (Keras functional baseline),
implemented in PyTorch (NHWC numpy loads -> NCHW tensors). Uses CUDA when available.

Checkpoints: best_val_auc / last / per-epoch as *.pt (state_dict only).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaselineCNN(nn.Module):
    """
    Keras parity: Conv32-same-MP, Conv64-same-MP, Conv128-same-MP, Conv256-same,
    GlobalAvgPool, Dense128+Dropout0.5, output logit (use BCEWithLogitsLoss).
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def _nhwc_to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()
    return t.to(device, non_blocking=True)


def load_h5_xy(method_dir: Path):
    paths = {
        "train_x": method_dir / "train_x.h5",
        "train_y": method_dir / "train_y.h5",
        "val_x": method_dir / "valid_x.h5",
        "val_y": method_dir / "valid_y.h5",
        "test_x": method_dir / "test_x.h5",
        "test_y": method_dir / "test_y.h5",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k}: {p}")

    def read_pair(xp, yp):
        with h5py.File(xp, "r") as fx, h5py.File(yp, "r") as fy:
            x = np.asarray(fx["x"][:], dtype=np.float32)
            y = np.asarray(fy["y"][:], dtype=np.float32).reshape(-1)
        return x, y

    train_x, train_y = read_pair(paths["train_x"], paths["train_y"])
    val_x, val_y = read_pair(paths["val_x"], paths["val_y"])
    test_x, test_y = read_pair(paths["test_x"], paths["test_y"])
    return train_x, train_y, val_x, val_y, test_x, test_y


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": spec,
        "sensitivity": sens,
        "auc": auc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _print_confusion(title: str, metrics: dict[str, Any]) -> None:
    tp, tn, fp, fn = metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]
    print(f"\n  [{title}] Confusion (true rows, pred cols)")
    print(f"              pred_0   pred_1")
    print(f"    true_0      {tn:6d}   {fp:6d}")
    print(f"    true_1      {fn:6d}   {tp:6d}")


def _json_safe_dump(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe_dump(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe_dump(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if (np.isnan(x) or np.isinf(x)) else x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def predict_proba_batches(
    model: nn.Module,
    x_nhwc: np.ndarray,
    batch_size: int,
    device: torch.device,
    desc: str = "predict",
) -> np.ndarray:
    model.eval()
    n = x_nhwc.shape[0]
    outs = []
    with torch.no_grad():
        for start in tqdm(
            range(0, n, batch_size),
            desc=desc,
            unit="batch",
            leave=False,
            total=(n + batch_size - 1) // batch_size,
        ):
            end = min(start + batch_size, n)
            xb = _nhwc_to_tensor(x_nhwc[start:end], device)
            logits = model(xb)
            prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            outs.append(prob)
    return np.concatenate(outs, axis=0)


class EpochEvalTracker:
    """Per-epoch val metrics, confusion matrix, checkpoints (.pt)."""

    def __init__(
        self,
        model: nn.Module,
        val_x,
        val_y,
        test_x,
        test_y,
        out_dir: Path,
        device: torch.device,
        pred_batch_size: int,
        eval_test_each_epoch: bool,
    ):
        self.model = model
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y
        self.out_dir = out_dir
        self.device = device
        self.pred_batch_size = pred_batch_size
        self.eval_test_each_epoch = eval_test_each_epoch
        self.history: list[dict] = []
        self.best_auc = -1.0
        out_dir.mkdir(parents=True, exist_ok=True)

    def save_weights(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def on_epoch_end(self, epoch: int, train_loss: float, train_acc: float) -> None:
        print(f"\n--- Epoch {epoch + 1} evaluation ---")
        val_prob = predict_proba_batches(
            self.model,
            self.val_x,
            self.pred_batch_size,
            self.device,
            desc=f"val ep{epoch + 1}",
        )
        val_m = _binary_metrics(self.val_y, val_prob)
        _print_confusion("val", val_m)
        print(
            "  val metrics:",
            "acc={:.4f} bal_acc={:.4f} prec={:.4f} rec={:.4f} f1={:.4f} "
            "spec={:.4f} sens={:.4f} auc={:.4f} mcc={:.4f}".format(
                val_m["accuracy"],
                val_m["balanced_accuracy"],
                val_m["precision"],
                val_m["recall"],
                val_m["f1"],
                val_m["specificity"],
                val_m["sensitivity"],
                val_m["auc"],
                val_m["mcc"],
            ),
        )

        record = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val": val_m,
        }

        if self.eval_test_each_epoch:
            test_prob = predict_proba_batches(
                self.model,
                self.test_x,
                self.pred_batch_size,
                self.device,
                desc=f"test ep{epoch + 1}",
            )
            test_m = _binary_metrics(self.test_y, test_prob)
            _print_confusion("test", test_m)
            record["test"] = test_m

        self.history.append(record)
        hist_path = self.out_dir / "metrics_per_epoch.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe_dump(self.history), f, indent=2)
        print(f"  Saved: {hist_path}")

        last_w = self.out_dir / "weights_last.pt"
        self.save_weights(last_w)
        print(f"  Checkpoint: {last_w}")

        ep_w = self.out_dir / f"weights_epoch_{epoch + 1:03d}.pt"
        self.save_weights(ep_w)

        auc = val_m["auc"]
        best_w = self.out_dir / "weights_best_val_auc.pt"
        if not np.isnan(auc) and auc > self.best_auc:
            self.best_auc = auc
            self.save_weights(best_w)
            print(f"  New best val AUC={auc:.4f} -> {best_w}")
        elif not best_w.exists():
            self.save_weights(best_w)
            print(f"  Seeded best checkpoint -> {best_w}")

        prog = {
            "last_completed_epoch": int(epoch + 1),
            "best_val_auc": float(self.best_auc)
            if not np.isnan(self.best_auc)
            else None,
            "epochs_done": len(self.history),
        }
        with open(self.out_dir / "run_progress.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe_dump(prog), f, indent=2)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in tqdm(loader, desc="train", leave=False, unit="batch"):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb).squeeze(-1)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        bs = xb.size(0)
        total_loss += loss.item() * bs
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += (pred == yb).sum().item()
        total += bs
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_method(
    method_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    resume: bool,
    fresh: bool,
    pred_batch_size: int,
    eval_test_each_epoch: bool,
) -> None:
    _set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(
            "GPU: CUDA available — {} (device {})".format(
                torch.cuda.get_device_name(0), device
            )
        )
    else:
        print("GPU: CUDA not available — using CPU.")

    print(f"Loading H5 from {method_dir} ...")
    train_x, train_y, val_x, val_y, test_x, test_y = load_h5_xy(method_dir)
    print(
        f"  shapes train {train_x.shape} val {val_x.shape} test {test_x.shape}"
    )

    out_dir = Path(out_dir)
    if fresh:
        if out_dir.exists():
            print(f"--fresh: removing previous run directory:\n  {out_dir}")
            shutil.rmtree(out_dir)
        resume = False

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "backend": "pytorch",
        "method_dir": str(method_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "pred_batch_size": pred_batch_size,
        "eval_test_each_epoch": eval_test_each_epoch,
        "model": "baseline_cnn_notebook (PyTorch)",
        "checkpoints": "*.pt (state_dict)",
    }
    if fresh or not resume or not (out_dir / "run_config.json").exists():
        with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    x_train_t = torch.from_numpy(np.ascontiguousarray(train_x)).permute(0, 3, 1, 2).float()
    y_train_t = torch.from_numpy(np.ascontiguousarray(train_y)).float()
    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = BaselineCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tracker = EpochEvalTracker(
        model,
        val_x,
        val_y,
        test_x,
        test_y,
        out_dir,
        device,
        pred_batch_size,
        eval_test_each_epoch,
    )

    start_epoch = 0
    last_w = out_dir / "weights_last.pt"
    hist_path = out_dir / "metrics_per_epoch.json"
    prog_path = out_dir / "run_progress.json"

    if resume and hist_path.exists():
        with open(hist_path, encoding="utf-8") as f:
            tracker.history = json.load(f)
        start_epoch = len(tracker.history)
        if tracker.history:
            aus = []
            for h in tracker.history:
                a = h["val"].get("auc")
                if a is None:
                    continue
                af = float(a)
                if not np.isnan(af):
                    aus.append(af)
            tracker.best_auc = max(aus) if aus else -1.0
        if prog_path.exists():
            with open(prog_path, encoding="utf-8") as f:
                prog_epoch = int(json.load(f).get("last_completed_epoch", 0))
            if prog_epoch != start_epoch:
                print(
                    "Warning: run_progress last_completed_epoch={} != len(metrics_per_epoch)={}; "
                    "using metrics length for resume.".format(prog_epoch, start_epoch)
                )
        if last_w.exists():
            model.load_state_dict(torch.load(last_w, map_location=device))
            print(
                "Resume: loaded {}  |  start_epoch={} (epochs in metrics_per_epoch.json)".format(
                    last_w, start_epoch
                )
            )
        else:
            print(
                "Warning: metrics_per_epoch.json exists but weights_last.pt missing; "
                "starting from epoch 0."
            )
            start_epoch = 0
            tracker.history = []
            tracker.best_auc = -1.0
    elif resume and last_w.exists():
        model.load_state_dict(torch.load(last_w, map_location=device))
        if prog_path.exists():
            with open(prog_path, encoding="utf-8") as f:
                start_epoch = int(json.load(f).get("last_completed_epoch", 0))
        print(
            "Resume: loaded {}  |  start_epoch={} (from run_progress; no metrics file)".format(
                last_w, start_epoch
            )
        )

    if start_epoch >= epochs:
        print("No remaining epochs (already completed).")
        return

    print(f"Training epochs {start_epoch + 1} .. {epochs}")
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(
            "\nEpoch {}/{}  train_loss={:.4f}  train_acc={:.4f}".format(
                epoch + 1, epochs, train_loss, train_acc
            )
        )
        tracker.on_epoch_end(epoch, train_loss, train_acc)

    best_path = out_dir / "weights_best_val_auc.pt"
    if not best_path.exists():
        best_path = out_dir / "weights_last.pt"
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"\nFinal evaluation using {best_path.name}")
    test_prob = predict_proba_batches(
        model, test_x, pred_batch_size, device, desc="test final"
    )
    test_m = _binary_metrics(test_y, test_prob)
    _print_confusion("test (final)", test_m)
    print(
        "  test:",
        "acc={:.4f} auc={:.4f} f1={:.4f} mcc={:.4f}".format(
            test_m["accuracy"], test_m["auc"], test_m["f1"], test_m["mcc"]
        ),
    )
    with open(out_dir / "test_metrics_final.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe_dump(test_m), f, indent=2)
    print(f"Saved {out_dir / 'test_metrics_final.json'}")


DEFAULT_METHODS = (
    "macenko",
    "reinhard",
    "vahadane",
    "adaptive_single_ref",
    "adaptive_multi_ref",
    "adaptive_multi_ref_aug",
)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Train baseline CNN (PyTorch) on benchmark H5 per stain method."
    )
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=str(root / "pcam_data" / "benchmark_preprocessed"),
        help="Folder containing per-method subdirs with train/valid/test *_x/y.h5",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Single method folder name (e.g. macenko). Default: all methods with train_x.h5.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(root / "experiments" / "benchmark_cnn_runs"),
        help="Output root: <runs-root>/<method>/",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from weights_last.pt (+ metrics_per_epoch.json if present)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete this method's run directory under --runs-root before training.",
    )
    parser.add_argument(
        "--pred-batch-size",
        type=int,
        default=256,
        help="Batch size for val/test prediction",
    )
    parser.add_argument(
        "--eval-test-each-epoch",
        action="store_true",
        help="Compute test metrics every epoch (slower)",
    )
    args = parser.parse_args()

    bench = Path(args.benchmark_root)
    if not bench.is_dir():
        print("Error: benchmark root not found:", bench, file=sys.stderr)
        sys.exit(1)

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.method:
        methods = [args.method]
    else:
        methods = []
        for m in DEFAULT_METHODS:
            d = bench / m
            if (d / "train_x.h5").exists():
                methods.append(m)
        if not methods:
            print("Error: no method folders with train_x.h5 under", bench, file=sys.stderr)
            sys.exit(1)
        print("Methods to train:", methods)

    for m in methods:
        method_dir = bench / m
        out_dir = runs_root / m
        print("\n" + "=" * 72)
        print(f"METHOD: {m}")
        print("=" * 72)
        try:
            train_one_method(
                method_dir=method_dir,
                out_dir=out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                resume=args.resume,
                fresh=args.fresh,
                pred_batch_size=args.pred_batch_size,
                eval_test_each_epoch=args.eval_test_each_epoch,
            )
        except FileNotFoundError as e:
            print(f"Skip {m}: {e}", file=sys.stderr)

    print("\nAll requested runs finished. Runs root:", runs_root)


if __name__ == "__main__":
    main()
