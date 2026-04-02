"""
Train the same baseline CNN on benchmark H5 folders (fair comparison across stain methods).

Uses the same architecture as notebooks/pcam_baseline_training.ipynb (build_baseline_cnn).
Each run: fixed seed, same hyperparameters, data already normalized in [0,1] from prepare_stain_benchmark_h5.

Long-run: per-epoch checkpoints, best-by-val-AUC weights, JSON metrics history, resume support.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

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
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def build_baseline_cnn(input_shape=(96, 96, 3)):
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        import keras
        from keras import layers

    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out)


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


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
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


def predict_in_batches(model, x: np.ndarray, batch_size: int, desc: str = "predict") -> np.ndarray:
    n = x.shape[0]
    outs = []
    for start in tqdm(
        range(0, n, batch_size),
        desc=desc,
        unit="batch",
        leave=False,
        total=(n + batch_size - 1) // batch_size,
    ):
        end = min(start + batch_size, n)
        outs.append(model.predict(x[start:end], verbose=0))
    return np.concatenate(outs, axis=0)


class EpochEvalCallback:
    """Holds val/test arrays and state; use with Keras Callback wrapper below."""

    def __init__(
        self,
        val_x,
        val_y,
        test_x,
        test_y,
        out_dir: Path,
        pred_batch_size: int,
        eval_test_each_epoch: bool,
    ):
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y
        self.out_dir = out_dir
        self.pred_batch_size = pred_batch_size
        self.eval_test_each_epoch = eval_test_each_epoch
        self.history: list[dict] = []
        self.best_auc = -1.0
        out_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, model, epoch, logs=None):
        logs = logs or {}
        print(f"\n--- Epoch {epoch + 1} evaluation ---")

        val_prob = predict_in_batches(
            model, self.val_x, self.pred_batch_size, desc=f"val ep{epoch + 1}"
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
            "train_loss": float(logs.get("loss", 0.0)),
            "train_accuracy": float(logs.get("accuracy", 0.0)),
            "val": val_m,
        }

        if self.eval_test_each_epoch:
            test_prob = predict_in_batches(
                model, self.test_x, self.pred_batch_size, desc=f"test ep{epoch + 1}"
            )
            test_m = _binary_metrics(self.test_y, test_prob)
            _print_confusion("test", test_m)
            record["test"] = test_m

        self.history.append(record)
        hist_path = self.out_dir / "metrics_per_epoch.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe_dump(self.history), f, indent=2)
        print(f"  Saved: {hist_path}")

        last_w = self.out_dir / "weights_last.weights.h5"
        model.save_weights(last_w)
        print(f"  Checkpoint: {last_w}")

        ep_w = self.out_dir / f"weights_epoch_{epoch + 1:03d}.weights.h5"
        model.save_weights(ep_w)

        auc = val_m["auc"]
        if not np.isnan(auc) and auc > self.best_auc:
            self.best_auc = auc
            best_w = self.out_dir / "weights_best_val_auc.weights.h5"
            model.save_weights(best_w)
            print(f"  New best val AUC={auc:.4f} -> {best_w}")

        prog = {
            "last_completed_epoch": int(epoch + 1),
            "best_val_auc": float(self.best_auc),
            "epochs_done": len(self.history),
        }
        with open(self.out_dir / "run_progress.json", "w", encoding="utf-8") as f:
            json.dump(prog, f, indent=2)


def _make_epoch_callback(keras_mod, eval_state: EpochEvalCallback):
    class _EpochCb(keras_mod.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            eval_state.on_epoch_end(self.model, epoch, logs)

    return _EpochCb()


def train_one_method(
    method_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    resume: bool,
    pred_batch_size: int,
    eval_test_each_epoch: bool,
) -> None:
    try:
        from tensorflow import keras
    except ImportError:
        import keras

    _set_seeds(seed)
    print(f"Loading H5 from {method_dir} ...")
    train_x, train_y, val_x, val_y, test_x, test_y = load_h5_xy(method_dir)
    print(
        f"  shapes train {train_x.shape} val {val_x.shape} test {test_x.shape}"
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "method_dir": str(method_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "pred_batch_size": pred_batch_size,
        "eval_test_each_epoch": eval_test_each_epoch,
        "model": "baseline_cnn_notebook",
    }
    if not resume or not (out_dir / "run_config.json").exists():
        with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    model = build_baseline_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    initial_epoch = 0
    last_w = out_dir / "weights_last.weights.h5"
    if resume and last_w.exists():
        model.load_weights(last_w)
        prog_path = out_dir / "run_progress.json"
        if prog_path.exists():
            with open(prog_path, encoding="utf-8") as f:
                p = json.load(f)
            initial_epoch = int(p.get("last_completed_epoch", 0))
        print(f"Resume: loaded {last_w}, continuing from epoch {initial_epoch}")

    cb = EpochEvalCallback(
        val_x,
        val_y,
        test_x,
        test_y,
        out_dir,
        pred_batch_size,
        eval_test_each_epoch,
    )
    if resume and (out_dir / "metrics_per_epoch.json").exists():
        with open(out_dir / "metrics_per_epoch.json", encoding="utf-8") as f:
            cb.history = json.load(f)
        if cb.history:
            cb.best_auc = max(
                (h["val"]["auc"] for h in cb.history if not np.isnan(h["val"]["auc"])),
                default=-1.0,
            )

    epoch_cb = _make_epoch_callback(keras, cb)

    remaining = max(0, epochs - initial_epoch)
    if remaining == 0:
        print("No remaining epochs (already completed).")
        return

    print(f"Training epochs {initial_epoch + 1} .. {epochs} (remaining {remaining})")
    # No validation_data here: avoids an extra full pass on val each epoch; metrics come from EpochEvalCallback only.
    model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        shuffle=True,
        callbacks=[epoch_cb],
    )

    best_path = out_dir / "weights_best_val_auc.weights.h5"
    if not best_path.exists():
        best_path = out_dir / "weights_last.weights.h5"
    model.load_weights(best_path)
    print(f"\nFinal evaluation using {best_path.name}")
    test_prob = predict_in_batches(model, test_x, pred_batch_size, desc="test final")
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
    parser = argparse.ArgumentParser(description="Train baseline CNN on benchmark H5 per stain method.")
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
        help="Single method folder name (e.g. macenko). Default: train all known methods present.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(root / "experiments" / "benchmark_cnn_runs"),
        help="Root for outputs experiments/benchmark_cnn_runs/<method>/",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from weights_last.weights.h5")
    parser.add_argument("--pred-batch-size", type=int, default=256, help="Batch size for metric prediction")
    parser.add_argument(
        "--eval-test-each-epoch",
        action="store_true",
        help="Also compute test metrics+CM every epoch (slower, stricter comparability log)",
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
                pred_batch_size=args.pred_batch_size,
                eval_test_each_epoch=args.eval_test_each_epoch,
            )
        except FileNotFoundError as e:
            print(f"Skip {m}: {e}", file=sys.stderr)

    print("\nAll requested runs finished. Runs root:", runs_root)


if __name__ == "__main__":
    main()
