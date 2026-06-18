#!/usr/bin/env python3
"""Export a side-by-side PCam pair (normal + tumor) for slides — labels only, no indices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


def resolve_train_x_path(data_dir: Path) -> Path:
    organized = data_dir / "training" / "camelyonpatch_level_2_split_train_x.h5"
    flat = data_dir / "camelyonpatch_level_2_split_train_x.h5"
    if organized.is_file():
        return organized
    if flat.is_file():
        return flat
    raise FileNotFoundError(
        f"No train x HDF5 under {data_dir} (expected training/… or flat …_train_x.h5)."
    )


def load_patch_rgb(train_x_path: Path, idx: int) -> np.ndarray:
    with h5py.File(train_x_path, "r") as fx:
        x = np.asarray(fx["x"][idx], dtype=np.float32)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))
    if x.max() <= 1.0:
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    else:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def upscale_patch(
    rgb: np.ndarray,
    size: int,
    *,
    sharpen: bool = False,
) -> np.ndarray:
    """PCam tiles are 96×96; upscale with Lanczos (optional mild unsharp for slide pop)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected H×W×3 RGB, got {rgb.shape}")
    im = Image.fromarray(rgb, mode="RGB")
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    if sharpen:
        im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=52, threshold=2))
    return np.asarray(im)


def auto_upscale_side(fig_w_inch: float, dpi: int) -> int:
    """~Match each subplot's raster width (1 row x 2 cols, typical margins) so PNG isn't upscaled twice."""
    panel_w_frac = 0.435
    px = int(panel_w_frac * fig_w_inch * dpi)
    return max(2048, min(4096, px))


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    default_data = repo / "pcam_data"
    default_out = repo / "reports" / "figures"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=default_data, help="PCam root (training/ or flat .h5)")
    p.add_argument("--out-dir", type=Path, default=default_out, help="Output directory")
    p.add_argument("--normal-idx", type=int, default=183419)
    p.add_argument("--tumor-idx", type=int, default=87283)
    p.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure DPI for raster export (PNG / PDF raster); higher = sharper file",
    )
    p.add_argument(
        "--upscale",
        type=int,
        default=0,
        metavar="PX",
        help="Square side in pixels after Lanczos (0 = auto from figure width and DPI)",
    )
    p.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Disable mild unsharp after upscale (default: on)",
    )
    args = p.parse_args()

    train_x_path = resolve_train_x_path(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "pcam_slide_normal_tumor_pair"
    out_png = args.out_dir / f"{stem}.png"
    out_pdf = args.out_dir / f"{stem}.pdf"

    # Large canvas + high DPI for maximum slide raster quality
    fig_w, fig_h = 15.0, 6.75
    if args.upscale > 0:
        side = args.upscale
    else:
        side = auto_upscale_side(fig_w, args.dpi)

    sharpen = not args.no_sharpen
    normal = upscale_patch(
        load_patch_rgb(train_x_path, args.normal_idx), side, sharpen=sharpen
    )
    tumor = upscale_patch(
        load_patch_rgb(train_x_path, args.tumor_idx), side, sharpen=sharpen
    )

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=args.dpi)
    fig.patch.set_facecolor("white")

    label_fs = 26
    for ax, img, lab in zip(axes, (normal, tumor), ("Normal", "Tumor")):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(lab, fontsize=label_fs, fontweight="bold", pad=14)
        ax.axis("off")

    plt.subplots_adjust(left=0.022, right=0.978, top=0.89, bottom=0.045, wspace=0.085)
    fig.savefig(
        out_png,
        bbox_inches="tight",
        facecolor="white",
        dpi=args.dpi,
        pil_kwargs={"compress_level": 3},
    )
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white", dpi=args.dpi)
    plt.close(fig)
    print(f"Upscale: {side}x{side} px, DPI={args.dpi}, sharpen={sharpen}")
    print("Wrote:", out_png.resolve())
    print("Wrote:", out_pdf.resolve())


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
