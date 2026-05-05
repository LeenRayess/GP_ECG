"""Evaluate trained Virchow2 head on reserved preprocessed PCam test split.

Reads test_x.h5/test_y.h5 from --preprocessed-dir and model artifacts from --run-dir.
Writes test metrics JSON (and optional NPZ with per-sample predictions) to --run-dir.
"""

from __future__ import print_function

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from timm.layers import SwiGLUPacked
    import timm
except ImportError:
    timm = None
    SwiGLUPacked = None

from train_virchow_preprocessed_colab import (  # type: ignore
    PreprocessedPCamDataset,
    VirchowClassifier,
    collect_logits_labels,
    collect_mc_dropout_probs,
    compute_classification_metrics,
    expected_calibration_error,
    print_confusion_and_metrics,
)


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
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _load_temperature(run_dir: Path) -> float:
    p = run_dir / "temperature_fit.json"
    if not p.is_file():
        return 1.0
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        t = float(d.get("temperature", 1.0))
        if t <= 0:
            return 1.0
        return t
    except Exception:
        return 1.0


def _load_model_weights(model: VirchowClassifier, run_dir: Path, device: torch.device) -> Path:
    best = run_dir / "model_best.pt"
    ckpt_last = run_dir / "checkpoint_last.pt"
    ckpt_compat = run_dir / "checkpoint.pt"

    if best.is_file():
        state = torch.load(best, map_location=device)
        model.load_state_dict(state, strict=False)
        return best

    for p in (ckpt_last, ckpt_compat):
        if p.is_file():
            state = torch.load(p, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
                return p
    raise FileNotFoundError("Could not find model weights in run dir.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reserved PCam test split with trained Virchow head.")
    parser.add_argument("--preprocessed-dir", type=str, required=True, help="Folder with test_x.h5 and test_y.h5")
    parser.add_argument("--run-dir", type=str, required=True, help="Training output folder with model_best.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-dropout", type=float, default=None, help="Override head dropout used at training.")
    parser.add_argument("--mc-samples", type=int, default=0, help="MC dropout passes on test (0 disables).")
    parser.add_argument("--no-save-test-preds", action="store_true", help="Skip writing test_predictions.npz.")
    args = parser.parse_args()

    if timm is None:
        print("Error: install timm and dependencies (pip install timm)")
        sys.exit(1)

    pre_dir = Path(args.preprocessed_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    test_x = pre_dir / "test_x.h5"
    test_y = pre_dir / "test_y.h5"
    for p in (test_x, test_y):
        if not p.is_file():
            print("Error: missing", p)
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    run_cfg = run_dir / "run_config.json"
    inferred_dropout = 0.0
    if run_cfg.is_file():
        try:
            with open(run_cfg, "r", encoding="utf-8") as f:
                rc = json.load(f)
            inferred_dropout = float(rc.get("head_dropout", 0.0))
        except Exception:
            inferred_dropout = 0.0
    head_dropout = inferred_dropout if args.head_dropout is None else float(args.head_dropout)
    print("Head dropout:", head_dropout)

    test_ds = PreprocessedPCamDataset(str(test_x), str(test_y))
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    print("Loading Virchow2 backbone (hf-hub:paige-ai/Virchow2) ...")
    backbone = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    backbone = backbone.to(device).eval()
    model = VirchowClassifier(backbone, head_dropout_p=head_dropout).to(device)
    loaded_from = _load_model_weights(model, run_dir, device)
    print("Loaded model weights from:", loaded_from)

    logits_np, labels_np = collect_logits_labels(model, test_loader, device, desc="test logits")
    logits_np = logits_np.reshape(-1)
    labels_np = labels_np.reshape(-1)
    prob_np = 1.0 / (1.0 + np.exp(-logits_np))

    T = _load_temperature(run_dir)
    logits_scaled = logits_np / T
    prob_cal = 1.0 / (1.0 + np.exp(-logits_scaled))

    metrics_raw = compute_classification_metrics(labels_np, prob_np, threshold=0.5)
    metrics_cal = compute_classification_metrics(labels_np, prob_cal, threshold=0.5)
    ece_raw = expected_calibration_error(labels_np, prob_np, n_bins=15)
    ece_cal = expected_calibration_error(labels_np, prob_cal, n_bins=15)

    cm = metrics_raw.get("confusion_at_threshold", {})
    print_confusion_and_metrics(
        int(cm.get("tp", 0)),
        int(cm.get("tn", 0)),
        int(cm.get("fp", 0)),
        int(cm.get("fn", 0)),
        split_name="test",
    )
    print(
        "  test ROC-AUC: {:.4f}  PR-AUC: {:.4f}  Brier: {:.4f}".format(
            float(metrics_raw.get("roc_auc", float("nan"))),
            float(metrics_raw.get("average_precision", float("nan"))),
            float(metrics_raw.get("brier_score", float("nan"))),
        )
    )

    out: Dict[str, Any] = {
        "n_test": int(len(labels_np)),
        "checkpoint_loaded": str(loaded_from.resolve()),
        "temperature_used": float(T),
        "head_dropout": float(head_dropout),
        "metrics_test_prob_raw_sigmoid": metrics_raw,
        "metrics_test_prob_after_temperature": metrics_cal,
        "ece_15_bins_raw": ece_raw,
        "ece_15_bins_temperature_scaled": ece_cal,
    }

    if args.mc_samples > 0 and head_dropout > 0:
        mc_mean, mc_std, _ = collect_mc_dropout_probs(
            model, test_loader, device, n_samples=args.mc_samples, desc="mc test"
        )
        out["mc_dropout"] = {
            "n_samples": int(args.mc_samples),
            "prob_mean_summary": {"mean": float(mc_mean.mean()), "std": float(mc_mean.std())},
            "epistemic_std_summary": {"mean": float(mc_std.mean()), "std": float(mc_std.std())},
            "metrics_test_mc_dropout_mean_prob": compute_classification_metrics(labels_np, mc_mean, threshold=0.5),
            "ece_15_bins_mc_mean_prob": expected_calibration_error(labels_np, mc_mean, n_bins=15),
        }
    elif args.mc_samples > 0 and head_dropout <= 0:
        print("Warning: --mc-samples > 0 but head_dropout <= 0, skipping MC Dropout.")

    out_json = run_dir / "test_metrics_detailed.json"
    _atomic_json_dump(out_json, _json_safe(out))
    print("Wrote", out_json)

    compact = {
        "n_test": int(len(labels_np)),
        "checkpoint_loaded": str(loaded_from.resolve()),
        "temperature_used": float(T),
        "accuracy_raw": metrics_raw.get("accuracy"),
        "roc_auc_raw": metrics_raw.get("roc_auc"),
        "pr_auc_raw": metrics_raw.get("average_precision"),
        "brier_raw": metrics_raw.get("brier_score"),
        "ece_raw": ece_raw.get("ece"),
    }
    compact_json = run_dir / "test_metrics.json"
    _atomic_json_dump(compact_json, _json_safe(compact))
    print("Wrote", compact_json)

    if not args.no_save_test_preds:
        npz_payload = {
            "y_true": labels_np.astype(np.float32),
            "logits": logits_np.astype(np.float32),
            "prob_sigmoid": prob_np.astype(np.float32),
            "logits_temperature_scaled": logits_scaled.astype(np.float32),
            "prob_after_temperature": prob_cal.astype(np.float32),
            "temperature_T": np.float32(T),
        }
        if "mc_dropout" in out:
            npz_payload["prob_mc_mean"] = mc_mean.astype(np.float32)
            npz_payload["prob_mc_std"] = mc_std.astype(np.float32)
        out_npz = run_dir / "test_predictions.npz"
        np.savez_compressed(out_npz, **npz_payload)
        print("Wrote", out_npz, "(arrays: {})".format(", ".join(npz_payload.keys())))


if __name__ == "__main__":
    main()
