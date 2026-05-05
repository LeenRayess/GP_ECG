"""Shared helpers for CAMELYON17 PCam-style patch processing."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_PATCH_PX = 96
DEFAULT_CENTER_PX = 32


def parse_camelyon_xml(xml_path: Path) -> List[np.ndarray]:
    """Parse CAMELYON polygon XML into level-0 coordinate arrays."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    polys: List[np.ndarray] = []
    for ann in root.findall(".//Annotation"):
        pts = []
        for c in ann.findall(".//Coordinate"):
            try:
                x = float(c.attrib["X"])
                y = float(c.attrib["Y"])
            except Exception:
                continue
            pts.append((x, y))
        if len(pts) >= 3:
            polys.append(np.asarray(pts, dtype=np.float64))
    return polys


def rasterize_tumor_mask_for_patch(
    polygons_level0: Sequence[np.ndarray],
    x0_level0: int,
    y0_level0: int,
    w_level0: int,
    h_level0: int,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Rasterize tumor polygons into a binary mask for one patch."""
    img = Image.new("L", (out_w, out_h), 0)
    draw = ImageDraw.Draw(img)
    x1 = x0_level0 + w_level0
    y1 = y0_level0 + h_level0
    sx = out_w / float(w_level0)
    sy = out_h / float(h_level0)

    for poly in polygons_level0:
        bx0, by0 = float(np.min(poly[:, 0])), float(np.min(poly[:, 1]))
        bx1, by1 = float(np.max(poly[:, 0])), float(np.max(poly[:, 1]))
        if bx1 < x0_level0 or by1 < y0_level0 or bx0 > x1 or by0 > y1:
            continue
        coords = [((p[0] - x0_level0) * sx, (p[1] - y0_level0) * sy) for p in poly]
        if len(coords) >= 3:
            draw.polygon(coords, fill=1, outline=1)
    return np.asarray(img, dtype=np.uint8)


def center_label_from_mask(mask_u8: np.ndarray, center_px: int = DEFAULT_CENTER_PX, min_frac: float = 0.0) -> Tuple[int, float]:
    """PCam-like center-window labeling: positive if center window tumor fraction > min_frac."""
    h, w = mask_u8.shape
    cx, cy = w // 2, h // 2
    half = center_px // 2
    center = mask_u8[cy - half : cy + half, cx - half : cx + half]
    frac = float(center.mean())
    return (1 if frac > min_frac else 0), frac
