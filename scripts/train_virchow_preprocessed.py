"""Train a frozen Virchow2 backbone + linear head on H5 from preprocess_pcam_to_h5.

Expects train_x/y.h5 and valid_x/y.h5 (96×96 float [0,1]); loader resizes to 224 and ImageNet-normalizes.
Checkpoint: out-dir/checkpoint.pt; --resume continues. Gated model: HF_TOKEN or huggingface-cli login."""

from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, desc=None, **kwargs):
        return x

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

EPOCHS = 10
BATCH_SIZE = 64
EMBED_DIM = 2560
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
            y = float(np.array(fy["y"][idx]).flatten()[0])
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
    """Frozen Virchow2 + trainable Linear(2560, 1)."""

    def __init__(self, backbone, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(embed_dim, 1)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        emb = get_embedding(self.backbone, x)
        return self.head(emb).squeeze(-1)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs, log_every_batches=500):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    n_batches = len(loader)
    pbar = tqdm(loader, desc="Epoch {}/{} train".format(epoch + 1, total_epochs), leave=True, unit="batch")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        b = x.size(0)
        total_loss += loss.item() * b
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += (pred == y).sum().item()
        n += b
        running_loss = total_loss / n
        running_acc = correct / n
        pbar.set_postfix(loss="{:.4f}".format(running_loss), acc="{:.4f}".format(running_acc))
        if log_every_batches is not None and log_every_batches > 0 and (batch_idx + 1) % log_every_batches == 0:
            print("  [train] batch {}/{}  loss: {:.4f}  acc: {:.4f}".format(
                batch_idx + 1, n_batches, running_loss, running_acc))
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, desc="val"):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    all_preds = []
    all_labels = []
    for x, y in tqdm(loader, desc=desc, leave=False, unit="batch"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += (pred == y).sum().item()
        n += x.size(0)
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tp = ((preds == 1) & (labels == 1)).sum()
    return total_loss / max(n, 1), correct / max(n, 1), {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def print_confusion_and_metrics(tp, tn, fp, fn, split_name="val"):
    """Print 2x2 confusion matrix and precision, recall, F1."""
    print("\n  [{}] Confusion matrix (predicted columns, true rows):".format(split_name))
    print("                    pred_neg  pred_pos")
    print("  true_neg (0)      {:>8}  {:>8}".format(tn, fp))
    print("  true_pos (1)      {:>8}  {:>8}".format(fn, tp))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print("  precision (pos): {:.4f}  recall (pos): {:.4f}  F1: {:.4f}".format(prec, rec, f1))


def main():
    parser = argparse.ArgumentParser(description="Train Virchow2 (frozen) on preprocessed PCam.")
    parser.add_argument("--preprocessed-dir", type=str, default=None,
                        help="Path to preprocessed dir (train_x.h5, train_y.h5, valid_*.h5). Default: PROJECT_ROOT/pcam_data/preprocessed")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size (default: 64)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where to save checkpoints and metrics. Default: experiments/virchow2_preprocessed")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in out-dir")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the head (default: 1e-3)")
    parser.add_argument("--log-every", type=int, default=500,
                        help="Print train loss/acc every N batches (0=disable, default: 500)")
    args = parser.parse_args()

    preprocessed_dir = args.preprocessed_dir or os.path.join(PROJECT_ROOT, "pcam_data", "preprocessed")
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "experiments", "virchow2_preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    train_x = os.path.join(preprocessed_dir, "train_x.h5")
    train_y = os.path.join(preprocessed_dir, "train_y.h5")
    valid_x = os.path.join(preprocessed_dir, "valid_x.h5")
    valid_y = os.path.join(preprocessed_dir, "valid_y.h5")
    for p in [train_x, train_y, valid_x, valid_y]:
        if not os.path.isfile(p):
            print("Error: missing", p)
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = PreprocessedPCamDataset(train_x, train_y)
    valid_ds = PreprocessedPCamDataset(valid_x, valid_y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    try:
        from timm.layers import SwiGLUPacked
        import timm
    except ImportError:
        print("Error: install timm and timm dependencies (e.g. pip install timm)")
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
        model = VirchowClassifier(backbone).to(device)
        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            logits = model(dummy)
        assert logits.shape == (1,), "expected logits shape (1,), got {}".format(logits.shape)
        print("Virchow2 loaded successfully (dummy forward pass OK, logits shape {}).".format(logits.shape))
    except Exception as e:
        print("Error: Virchow2 failed to load or forward pass failed:", e)
        sys.exit(1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_acc = -1.0
    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")

    if args.resume and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", -1.0)
        print("Resumed from epoch", start_epoch - 1, "| best_val_acc so far:", best_val_acc)
    else:
        if args.resume:
            print("No checkpoint found at", checkpoint_path, "; starting from epoch 0")

    n_train = len(train_ds)
    n_val = len(valid_ds)
    print("\nTraining: {} epochs (from {} to {}), train samples: {}, val samples: {}".format(
        args.epochs - start_epoch, start_epoch + 1, args.epochs, n_train, n_val))
    print("  train batches/epoch: {}, val batches: {}  (batch_size={})".format(
        len(train_loader), len(valid_loader), args.batch_size))
    epoch_range = range(start_epoch, args.epochs)
    for epoch in tqdm(epoch_range, desc="Epochs", unit="epoch", leave=True):
        print("\n" + "=" * 60)
        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        print("=" * 60)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.epochs, log_every_batches=args.log_every if args.log_every else None,
        )
        val_loss, val_acc, val_cm = evaluate(model, valid_loader, device, desc="Epoch {}/{} val".format(epoch + 1, args.epochs))
        print("\n  Epoch {} summary:  train loss: {:.4f}  train acc: {:.4f}  val loss: {:.4f}  val acc: {:.4f}".format(
            epoch + 1, train_loss, train_acc, val_loss, val_acc))
        print_confusion_and_metrics(
            val_cm["tp"], val_cm["tn"], val_cm["fp"], val_cm["fn"], split_name="val"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
            print("  -> new best val acc, saved model_best.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, checkpoint_path)
        print("  checkpoint saved:", checkpoint_path)

    metrics = {
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "preprocessed_dir": os.path.abspath(preprocessed_dir),
        "out_dir": os.path.abspath(out_dir),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nDone. Best val acc:", best_val_acc, "| metrics saved to", os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
