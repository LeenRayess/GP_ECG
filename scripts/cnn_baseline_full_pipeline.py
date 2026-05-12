"""
Full-data baseline CNN pipeline (methodology-aligned).

Trains four models on full train / validation splits:
  1) PCam — raw (no stain normalization; uint8 patches scaled to [0,1])
  2) PCam — Macenko benchmark-style preprocessed H5
  3) CAMELYON17 (WILDS full OOD-val pack) — raw
  4) CAMELYON17 — Macenko benchmark-style preprocessed H5

Architecture matches docs/final_methodology.md §4.1 (same depth/width/head/dropout policy
as the historical PyTorch baseline in this repo): four conv stages 32→64→128→256, ReLU,
max-pool on the first three stages, global average pooling, dense 128 + dropout 0.5,
binary logit; Adam on BCE-with-logits.

After training, each model is evaluated:
  - in-domain: held-out test split of its training *domain*, with the *same* preprocessing
    arm as training (raw→raw, preprocessed→preprocessed)
  - cross-domain: held-out test split of the other domain, again with matched preprocessing
    (raw PCam model scored on raw CAM17 test, etc.)

Temperature scaling T>0 is fit on training-domain validation logits only (grid search,
same routine as Virchow scripts); T is fixed and applied to both in-domain and external
tests. Decisions use threshold 0.5 on calibrated probabilities (methodology §5.2).

Outputs (default root):
  experiments/cnn_baseline_full/<run_id>/
    run_manifest.json          # resolved paths, git-less env note, arms
    <arm>/                     # arm in {pcam_raw, pcam_preprocessed, cam17_raw, cam17_preprocessed}
      hparams.json
      metrics_per_epoch.json
      weights_epoch_EEE.pt     # one per completed epoch (model+optim+meta)
      weights_best_val_auc.pt  # best val ROC-AUC snapshot
      last_checkpoint.pt       # latest epoch for --resume
      temperature_fit.json
    evaluation/
      metrics_all.json         # nested tables + transfer deltas
      metrics_summary.csv

Example (Git Bash):
  cd /c/GP_ECG && source .venv/Scripts/activate
  python scripts/cnn_baseline_full_pipeline.py --epochs 10

Colab: set repo root (data under same tree), then run the same module, e.g.
  export GP_ECG_ROOT=/content/GP_ECG
  python scripts/cnn_baseline_full_pipeline.py --repo-root /content/GP_ECG

Progress: same batch milestone style as `train_virchow_preprocessed_colab` (batch 1 start,
then 10%, 20%, …, 100% with loss/acc on train), plus explicit `[cnn_baseline] step:` phase
lines for each arm, val passes, temperature fit, and eval. DataLoader uses num_workers=0.
Default train/eval batch size is 320; default --epochs is 10.
"""

from __future__ import annotations

print("[cnn_baseline] module: stdlib next, then numpy/torch (Colab: first load can take minutes)…", flush=True)

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

print("[cnn_baseline] importing numpy, torch…", flush=True)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


NUM_WORKERS = 0  # fixed (Colab / HDF5 safety)

_SCRIPT_FILE = Path(__file__).resolve()
_DEFAULT_REPO_ROOT = _SCRIPT_FILE.parent.parent
_SCRIPTS_DIR = _SCRIPT_FILE.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

print("[cnn_baseline] importing metrics helpers from train_virchow_preprocessed_colab…", flush=True)
from train_virchow_preprocessed_colab import (  # noqa: E402
    _pct_milestones_crossed,
    compute_classification_metrics,
    expected_calibration_error,
    fit_temperature_binary,
)

print("[cnn_baseline] imports finished.", flush=True)


def _banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{title}\n{line}", flush=True)


def _step(msg: str) -> None:
    print(f"[cnn_baseline] step: {msg}", flush=True)


# ---------------------------------------------------------------------------
# Architecture (methodology §4.1)
# ---------------------------------------------------------------------------


class BaselineCNN(nn.Module):
    """Conv 32→64→128→256, GAP, Dense128 + Dropout(0.5), binary logit."""

    def __init__(self) -> None:
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


# ---------------------------------------------------------------------------
# Data: HDF5 random access (full splits; no full-RAM load)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitPaths:
    train_x: Path
    train_y: Path
    val_x: Path
    val_y: Path
    test_x: Path
    test_y: Path
    arm_name: str
    """raw | preprocessed — controls /255 scaling heuristics."""
    input_kind: str


def _resolve_repo_root(repo_root: Optional[Path]) -> Path:
    env = os.environ.get("GP_ECG_ROOT", "").strip()
    if repo_root is not None:
        return repo_root.resolve()
    if env:
        return Path(env).resolve()
    return _DEFAULT_REPO_ROOT


def _default_pcam_raw_paths(repo: Path) -> SplitPaths:
    root = repo / "pcam_data"
    return SplitPaths(
        train_x=root / "training" / "camelyonpatch_level_2_split_train_x.h5",
        train_y=root / "training" / "camelyonpatch_level_2_split_train_y.h5",
        val_x=root / "val" / "camelyonpatch_level_2_split_valid_x.h5",
        val_y=root / "val" / "camelyonpatch_level_2_split_valid_y.h5",
        test_x=root / "test" / "camelyonpatch_level_2_split_test_x.h5",
        test_y=root / "test" / "camelyonpatch_level_2_split_test_y.h5",
        arm_name="pcam_raw",
        input_kind="raw",
    )


def _default_pcam_preprocessed_paths(repo: Path) -> SplitPaths:
    root = repo / "pcam_data" / "preprocessed_macenko_benchmark_style"
    return SplitPaths(
        train_x=root / "train_x.h5",
        train_y=root / "train_y.h5",
        val_x=root / "valid_x.h5",
        val_y=root / "valid_y.h5",
        test_x=root / "test_x.h5",
        test_y=root / "test_y.h5",
        arm_name="pcam_preprocessed",
        input_kind="preprocessed",
    )


def _default_cam17_raw_paths(repo: Path) -> SplitPaths:
    root = repo / "data" / "wilds" / "camelyon17_h5_full_oodval"
    return SplitPaths(
        train_x=root / "train_x.h5",
        train_y=root / "train_y.h5",
        val_x=root / "valid_x.h5",
        val_y=root / "valid_y.h5",
        test_x=root / "test_x.h5",
        test_y=root / "test_y.h5",
        arm_name="cam17_raw",
        input_kind="raw",
    )


def _default_cam17_preprocessed_paths(repo: Path) -> SplitPaths:
    root = repo / "data" / "wilds" / "camelyon17_h5_full_oodval" / "preprocessed_macenko_benchmark_style"
    return SplitPaths(
        train_x=root / "train_x.h5",
        train_y=root / "train_y.h5",
        val_x=root / "valid_x.h5",
        val_y=root / "valid_y.h5",
        test_x=root / "test_x.h5",
        test_y=root / "test_y.h5",
        arm_name="cam17_preprocessed",
        input_kind="preprocessed",
    )


def _require_paths(sp: SplitPaths) -> None:
    for k in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y"):
        p = getattr(sp, k)
        if not p.is_file():
            raise FileNotFoundError(f"[{sp.arm_name}] missing {k}: {p}")


def _h5_len(y_path: Path) -> int:
    import h5py

    with h5py.File(str(y_path), "r") as fy:
        return int(fy["y"].shape[0])


def _split_counts(sp: SplitPaths) -> Dict[str, int]:
    return {
        "n_train": _h5_len(sp.train_y),
        "n_val": _h5_len(sp.val_y),
        "n_test": _h5_len(sp.test_y),
    }


class H5PatchDataset(Dataset):
    """Reads patches from NHWC H5 ['x'] and ['y'].

    If reopen_each_sample is False (recommended with num_workers=0), keeps HDF5
    files open for fast random access. If True, opens per read (safe with multiprocessing).
    """

    def __init__(self, x_path: Path, y_path: Path, input_kind: str, reopen_each_sample: bool) -> None:
        import h5py

        self.x_path = str(x_path.resolve())
        self.y_path = str(y_path.resolve())
        self.input_kind = input_kind
        self.reopen_each_sample = reopen_each_sample
        self._fx: Any = None
        self._fy: Any = None
        with h5py.File(self.y_path, "r") as fy:
            self.n = int(fy["y"].shape[0])
        if not reopen_each_sample:
            self._fx = h5py.File(self.x_path, "r")
            self._fy = h5py.File(self.y_path, "r")

    def __len__(self) -> int:
        return self.n

    def _read_idx(self, idx: int) -> Tuple[np.ndarray, float]:
        if self.reopen_each_sample:
            import h5py

            with h5py.File(self.x_path, "r") as fx, h5py.File(self.y_path, "r") as fy:
                x = np.asarray(fx["x"][idx], dtype=np.float32)
                y = float(np.asarray(fy["y"][idx]).reshape(()))
        else:
            x = np.asarray(self._fx["x"][idx], dtype=np.float32)
            y = float(np.asarray(self._fy["y"][idx]).reshape(()))
        return x, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self._read_idx(idx)
        if self.input_kind == "raw":
            if x.max() > 1.5:
                x *= 1.0 / 255.0
        else:
            if x.max() > 1.5:
                x *= 1.0 / 255.0
        x = np.clip(x, 0.0, 1.0)
        xt = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        yt = torch.tensor(y, dtype=torch.float32)
        return xt, yt

    def __del__(self) -> None:
        try:
            if self._fx is not None:
                self._fx.close()
            if self._fy is not None:
                self._fy.close()
        except Exception:
            pass


def _collate_xy(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


# ---------------------------------------------------------------------------
# Metrics / calibration
# ---------------------------------------------------------------------------


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    n_batches = len(loader)
    printed: set = set()
    for batch_idx, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(-1)
        logits_list.append(logits.detach().float().cpu().numpy())
        labels_list.append(yb.numpy().reshape(-1))
        if batch_idx == 1:
            print(f"  [{desc}] batch 1/{n_batches} start", flush=True)
        for pct in _pct_milestones_crossed(batch_idx, n_batches, printed):
            print(f"  [{desc}] {pct}%  batch {batch_idx}/{n_batches}", flush=True)
    return np.concatenate(logits_list), np.concatenate(labels_list)


def _eval_split(
    logits: np.ndarray,
    y: np.ndarray,
    temperature: float,
) -> Dict[str, Any]:
    prob_raw = _sigmoid_stable(logits)
    z = logits / float(temperature)
    prob_cal = _sigmoid_stable(z)
    metrics_raw = compute_classification_metrics(y, prob_raw, threshold=0.5)
    metrics_cal = compute_classification_metrics(y, prob_cal, threshold=0.5)
    ece_cal = expected_calibration_error(y, prob_cal, n_bins=15)
    return {
        "n": int(len(y)),
        "metrics_prob_raw_sigmoid": metrics_raw,
        "metrics_prob_after_temperature_threshold_0p5": metrics_cal,
        "ece_15_bins_calibrated": ece_cal,
    }


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json_dump(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _auc_only(y: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(np.int32).reshape(-1)
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def train_one_arm(
    sp: SplitPaths,
    out_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    resume: bool,
) -> Dict[str, Any]:
    _step(f"arm={sp.arm_name} validate H5 paths exist")
    _require_paths(sp)
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seeds(seed)

    hparams = {
        "arm": sp.arm_name,
        "input_kind": sp.input_kind,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "num_workers": NUM_WORKERS,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
        "selection_metric": "val_roc_auc_raw_sigmoid",
        "paths": {
            "train_x": str(sp.train_x),
            "train_y": str(sp.train_y),
            "val_x": str(sp.val_x),
            "val_y": str(sp.val_y),
            "test_x": str(sp.test_x),
            "test_y": str(sp.test_y),
        },
    }
    _step(f"arm={sp.arm_name} write hparams.json -> {out_dir / 'hparams.json'}")
    _atomic_json_dump(out_dir / "hparams.json", hparams)

    pin = device.type == "cuda"
    reopen = NUM_WORKERS > 0
    _step(f"arm={sp.arm_name} open train HDF5 (may block on slow disk): {sp.train_x}")
    train_ds = H5PatchDataset(sp.train_x, sp.train_y, sp.input_kind, reopen_each_sample=reopen)
    _step(f"arm={sp.arm_name} open val HDF5: {sp.val_x}")
    val_ds = H5PatchDataset(sp.val_x, sp.val_y, sp.input_kind, reopen_each_sample=reopen)
    _step(f"arm={sp.arm_name} build DataLoaders (n_train={len(train_ds)} n_val={len(val_ds)})")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=False,
        collate_fn=_collate_xy,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=False,
        collate_fn=_collate_xy,
    )

    print(
        f"[{sp.arm_name}] DataLoaders ready: train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} (first train batch may be slow from disk)",
        flush=True,
    )

    _step(f"arm={sp.arm_name} init BaselineCNN + Adam on {device}")
    model = BaselineCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_auc = -1.0
    history: List[Dict[str, Any]] = []

    last_ckpt = out_dir / "last_checkpoint.pt"
    if resume and last_ckpt.is_file():
        _step(f"arm={sp.arm_name} load resume checkpoint {last_ckpt}")
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = int(ck.get("next_epoch", 0))
        best_auc = float(ck.get("best_val_auc", -1.0))
        history = ck.get("history", [])
        print(f"[{sp.arm_name}] Resuming from epoch {start_epoch + 1} (0-based resume index).", flush=True)

    print(
        f"\nTraining epochs {start_epoch + 1}..{epochs} (inclusive), "
        f"train samples: {len(train_ds)}, val samples: {len(val_ds)}",
        flush=True,
    )

    for epoch in range(start_epoch, epochs):
        _banner(f"{sp.arm_name}  Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0
        n_seen = 0
        correct = 0
        n_batches = len(train_loader)
        tag = f"{sp.arm_name} train ep{epoch+1}/{epochs}"
        printed: set = set()
        _step(f"arm={sp.arm_name} epoch {epoch+1} train loop start ({n_batches} batches)")
        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(-1)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            bs = xb.size(0)
            running_loss += float(loss.item()) * bs
            n_seen += bs
            pred = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((pred == yb).sum().item())
            rloss = running_loss / max(n_seen, 1)
            racc = correct / max(n_seen, 1)
            if batch_idx == 1:
                print(
                    f"  [{tag}] batch 1/{n_batches} start  loss: {rloss:.4f}  acc: {racc:.4f}",
                    flush=True,
                )
            for pct in _pct_milestones_crossed(batch_idx, n_batches, printed):
                print(
                    f"  [{tag}] {pct}%  batch {batch_idx}/{n_batches}  loss: {rloss:.4f}  acc: {racc:.4f}",
                    flush=True,
                )

        train_loss = running_loss / max(n_seen, 1)
        train_acc = correct / max(n_seen, 1)

        _step(f"arm={sp.arm_name} epoch {epoch+1} val logits (per-epoch validation)")
        val_logits, val_y = collect_logits(model, val_loader, device, desc=f"{sp.arm_name} val ep{epoch+1}")
        val_prob = _sigmoid_stable(val_logits)
        val_auc = _auc_only(val_y, val_prob)

        rec = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_roc_auc_raw_sigmoid": float(val_auc),
        }
        history.append(rec)
        _atomic_json_dump(out_dir / "metrics_per_epoch.json", _json_safe(history))

        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), out_dir / "weights_best_val_auc.pt")
            print(f"  [{sp.arm_name}] -> new best val ROC-AUC, saved weights_best_val_auc.pt", flush=True)

        ep_path = out_dir / f"weights_epoch_{epoch + 1:03d}.pt"
        _step(f"arm={sp.arm_name} epoch {epoch+1} save weights + last_checkpoint")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_roc_auc_raw_sigmoid": val_auc,
                "train_loss": train_loss,
            },
            ep_path,
        )
        torch.save(
            {
                "next_epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_val_auc": best_auc,
                "history": history,
            },
            last_ckpt,
        )
        print(
            f"\n  [{sp.arm_name}] Epoch {epoch+1} summary: train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_auc={val_auc:.4f}  best_val_auc={best_auc:.4f}",
            flush=True,
        )

    # Temperature on val with best weights
    best_w = out_dir / "weights_best_val_auc.pt"
    if not best_w.is_file():
        torch.save(model.state_dict(), best_w)
    _step(f"arm={sp.arm_name} load best weights for temperature fit: {best_w}")
    model.load_state_dict(torch.load(best_w, map_location=device))
    _step(f"arm={sp.arm_name} collect val logits for temperature grid search")
    val_logits, val_y = collect_logits(model, val_loader, device, desc=f"{sp.arm_name} val (temperature fit)")
    _step(f"arm={sp.arm_name} fit_temperature_binary (n_val={len(val_y)})")
    tinfo = fit_temperature_binary(val_logits, val_y.astype(np.float64), grid=80)
    T = float(tinfo["temperature"])
    _atomic_json_dump(
        out_dir / "temperature_fit.json",
        {
            "temperature": T,
            "fit_split": "source_validation_only",
            "arm": sp.arm_name,
            "n_val": int(len(val_y)),
            "nll_logits_raw_mean": tinfo.get("nll_logits_raw_mean"),
            "nll_after_T_mean": tinfo.get("nll_after_T_mean"),
        },
    )
    print(f"[{sp.arm_name}] temperature T={T:.4f}  wrote temperature_fit.json", flush=True)
    _step(f"arm={sp.arm_name} training arm complete out_dir={out_dir}")

    return {"temperature": T, "best_val_auc": best_auc, "out_dir": str(out_dir)}


def _load_model_for_eval(ckpt: Path, device: torch.device) -> BaselineCNN:
    m = BaselineCNN().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    m.eval()
    return m


def evaluate_all(
    arms: Dict[str, SplitPaths],
    run_dir: Path,
    temperatures: Dict[str, float],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    """Each trained arm: in-domain + cross-domain tests (matched preprocessing)."""
    _banner("EVALUATION: in-domain + external test (per trained arm)")
    pin = device.type == "cuda"
    out_eval = run_dir / "evaluation"
    out_eval.mkdir(parents=True, exist_ok=True)

    # Map arm -> SplitPaths for tests only
    test_surfaces: Dict[str, SplitPaths] = {
        "pcam_raw": arms["pcam_raw"],
        "pcam_preprocessed": arms["pcam_preprocessed"],
        "cam17_raw": arms["cam17_raw"],
        "cam17_preprocessed": arms["cam17_preprocessed"],
    }

    def make_loader(sp: SplitPaths, split: str) -> DataLoader:
        if split == "test":
            ds = H5PatchDataset(sp.test_x, sp.test_y, sp.input_kind, reopen_each_sample=(NUM_WORKERS > 0))
        else:
            raise ValueError(split)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin,
            persistent_workers=False,
            collate_fn=_collate_xy,
        )

    results: Dict[str, Any] = {}

    plan = [
        ("pcam_raw", "pcam_raw", "cam17_raw"),
        ("pcam_preprocessed", "pcam_preprocessed", "cam17_preprocessed"),
        ("cam17_raw", "cam17_raw", "pcam_raw"),
        ("cam17_preprocessed", "cam17_preprocessed", "pcam_preprocessed"),
    ]

    for bi, (train_arm, in_key, ext_key) in enumerate(plan, start=1):
        ckpt = run_dir / train_arm / "weights_best_val_auc.pt"
        _banner(f"EVAL block {bi}/4 — model trained as: {train_arm}")
        _step(f"load checkpoint {ckpt}")
        T = float(temperatures[train_arm])
        model = _load_model_for_eval(ckpt, device)
        block: Dict[str, Any] = {"trained_arm": train_arm, "temperature": T, "tests": {}}

        for tag, surf_key in (("in_domain_test", in_key), ("external_test", ext_key)):
            sp = test_surfaces[surf_key]
            _step(f"eval {train_arm} -> {tag}  surface={surf_key}  test_x={sp.test_x.name}")
            loader = make_loader(sp, "test")
            _step(
                f"eval {train_arm} -> {tag}  n_samples={len(loader.dataset)}  "
                f"batches={len(loader)}  T={T:.4f}"
            )
            logits, y = collect_logits(model, loader, device, desc=f"eval {train_arm} -> {tag}")
            block["tests"][tag] = {
                "surface": surf_key,
                "split_paths": {
                    "test_x": str(sp.test_x),
                    "test_y": str(sp.test_y),
                },
                **_eval_split(logits, y, T),
            }
            _step(f"eval {train_arm} -> {tag}  done (ROC-AUC cal in metrics block)")

        # Transfer deltas on calibrated ROC-AUC / PR-AUC (higher better)
        m_in = block["tests"]["in_domain_test"]["metrics_prob_after_temperature_threshold_0p5"].get("roc_auc")
        m_ex = block["tests"]["external_test"]["metrics_prob_after_temperature_threshold_0p5"].get("roc_auc")
        pr_in = block["tests"]["in_domain_test"]["metrics_prob_after_temperature_threshold_0p5"].get(
            "average_precision"
        )
        pr_ex = block["tests"]["external_test"]["metrics_prob_after_temperature_threshold_0p5"].get(
            "average_precision"
        )

        def dh(m1: Any, m2: Any) -> Dict[str, float]:
            a = float(m1) if m1 is not None else float("nan")
            b = float(m2) if m2 is not None else float("nan")
            if math.isnan(a) or math.isnan(b):
                return {"abs_m_in_minus_m_ext": float("nan"), "rel_drop": float("nan")}
            return {"abs_m_in_minus_m_ext": float(a - b), "rel_drop": float((a - b) / (abs(a) + 1e-8))}

        block["transfer_degradation_calibrated"] = {
            "roc_auc": dh(m_in, m_ex),
            "pr_auc": dh(pr_in, pr_ex),
        }
        results[train_arm] = block
        print(f"[cnn_baseline] finished eval block for trained_arm={train_arm}", flush=True)

    _step(f"write {out_eval / 'metrics_all.json'}")
    _atomic_json_dump(out_eval / "metrics_all.json", _json_safe(results))

    # CSV summary
    import csv

    csv_path = out_eval / "metrics_summary.csv"
    _step(f"write {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "trained_arm",
                "test_name",
                "surface",
                "n",
                "roc_auc_cal",
                "pr_auc_cal",
                "brier_cal",
                "ece_cal",
            ]
        )
        for train_arm, block in results.items():
            for test_name in ("in_domain_test", "external_test"):
                t = block["tests"][test_name]
                mc = t["metrics_prob_after_temperature_threshold_0p5"]
                ece = t.get("ece_15_bins_calibrated", {}) or {}
                w.writerow(
                    [
                        train_arm,
                        test_name,
                        t.get("surface"),
                        t.get("n"),
                        mc.get("roc_auc"),
                        mc.get("average_precision"),
                        mc.get("brier_score"),
                        ece.get("ece"),
                    ]
                )
    _step("evaluation CSV complete")
    return results


def _print_gpu_banner() -> torch.device:
    print("=== GPU / device check ===", flush=True)
    print("torch:", torch.__version__, flush=True)
    print("torch.version.cuda:", getattr(torch.version, "cuda", None), flush=True)
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        print("CUDA: available", flush=True)
        print("device[0]:", torch.cuda.get_device_name(0), flush=True)
        print("device_count:", torch.cuda.device_count(), flush=True)
    else:
        print("CUDA: not available — training will use CPU (very slow).", flush=True)
    device = torch.device("cuda" if cuda_ok else "cpu")
    print("Selected device:", device, flush=True)
    print("==========================\n", flush=True)
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Full baseline CNN train + cross-domain eval (4 arms).")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per arm (default 10 matches train_virchow_preprocessed_colab.EPOCHS for comparability).",
    )
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--eval-batch-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repo root containing pcam_data/ and data/wilds/ (default: parent of scripts/ or GP_ECG_ROOT env).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Subfolder name under experiments/cnn_baseline_full/. Default: timestamp.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Root for all baseline runs (default: <repo>/experiments/cnn_baseline_full). Use Drive path on Colab.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume each arm from last_checkpoint.pt if present.")
    parser.add_argument("--skip-train", action="store_true", help="Only run evaluation (expects existing run folder).")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only run training phases (no evaluation block).",
    )
    args = parser.parse_args()

    print(
        f"[cnn_baseline] main() starting {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} UTC",
        flush=True,
    )

    device = _print_gpu_banner()

    repo = _resolve_repo_root(Path(args.repo_root) if args.repo_root else None)
    print(f"Repo root (data + scripts): {repo}\n", flush=True)

    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else (repo / "experiments" / "cnn_baseline_full")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    arms_list = [
        _default_pcam_raw_paths(repo),
        _default_pcam_preprocessed_paths(repo),
        _default_cam17_raw_paths(repo),
        _default_cam17_preprocessed_paths(repo),
    ]
    arms: Dict[str, SplitPaths] = {sp.arm_name: sp for sp in arms_list}
    for sp in arms_list:
        _step(f"verify H5 paths for arm={sp.arm_name}")
        _require_paths(sp)

    print("[cnn_baseline] paths OK; writing manifest (opens each train_y H5 once for counts)…", flush=True)
    manifest_path = run_dir / "run_manifest.json"
    if not (args.skip_train and manifest_path.is_file()):
        for sp in arms_list:
            c = _split_counts(sp)
            _step(f"manifest counts {sp.arm_name}: {c}")
        manifest = {
            "run_id": run_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repo_root": str(repo),
            "arms": {
                k: {
                    **{kk: str(vv) for kk, vv in asdict(v).items() if isinstance(vv, Path)},
                    "split_counts": _split_counts(v),
                }
                for k, v in arms.items()
            },
            "cli": vars(args),
        }
        _atomic_json_dump(manifest_path, manifest)
        _step(f"wrote manifest {manifest_path}")
    else:
        print(f"[info] Keeping existing {manifest_path} (--skip-train).", flush=True)

    temperatures: Dict[str, float] = {}

    if not args.skip_train:
        for ai, sp in enumerate(arms_list, start=1):
            _banner(f"TRAINING ARM {ai}/4: {sp.arm_name}")
            a_dir = run_dir / sp.arm_name
            info = train_one_arm(
                sp,
                a_dir,
                device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                resume=args.resume,
            )
            temperatures[sp.arm_name] = float(info["temperature"])
        _banner("All 4 training arms finished")
    else:
        _step("--skip-train: loading temperature_fit.json per arm")
        for sp in arms_list:
            p = run_dir / sp.arm_name / "temperature_fit.json"
            if not p.is_file():
                raise FileNotFoundError(f"--skip-train requires {p}")
            with open(p, "r", encoding="utf-8") as f:
                temperatures[sp.arm_name] = float(json.load(f)["temperature"])

    if not args.skip_eval:
        evaluate_all(
            arms=arms,
            run_dir=run_dir,
            temperatures=temperatures,
            device=device,
            batch_size=args.eval_batch_size,
        )
        print(f"\nWrote evaluation to: {run_dir / 'evaluation'}", flush=True)
    else:
        _step("--skip-eval set; skipping evaluate_all")
    print(f"\nRun directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
