import hashlib
import random

import h5py
import numpy as np


def tissue_ratio(u8_img: np.ndarray) -> float:
    r, g, b = u8_img[:, :, 0], u8_img[:, :, 1], u8_img[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b).astype(np.float32)
    mn = np.minimum(np.minimum(r, g), b).astype(np.float32)
    sat = np.where(mx > 1e-6, (mx - mn) / mx, 0.0)
    return float((sat > 0.12).mean())


def h(u8_img: np.ndarray) -> str:
    return hashlib.sha256(u8_img.tobytes()).hexdigest()


def main():
    random.seed(42)
    n = 20000
    pcam = "pcam_data/preprocessed_macenko_benchmark_style/train_x.h5"
    cam = "data/wilds/camelyon17_h5_full_oodval/preprocessed_macenko_benchmark_style/train_x.h5"

    hs_all = set()
    hs_tissue = set()
    with h5py.File(pcam, "r") as f:
        x = f["x"]
        idx = random.sample(range(x.shape[0]), n)
        for i in idx:
            arr = np.asarray(x[i], dtype=np.float32)
            u = np.asarray(np.clip(arr, 0, 1) * 255, dtype=np.uint8)
            hh = h(u)
            hs_all.add(hh)
            if tissue_ratio(u) >= 0.35:
                hs_tissue.add(hh)

    ov_all = 0
    ov_tissue = 0
    tissue_n = 0
    with h5py.File(cam, "r") as f:
        x = f["x"]
        idx = random.sample(range(x.shape[0]), n)
        for i in idx:
            arr = np.asarray(x[i], dtype=np.float32)
            u = np.asarray(np.clip(arr, 0, 1) * 255, dtype=np.uint8)
            hh = h(u)
            if hh in hs_all:
                ov_all += 1
            if tissue_ratio(u) >= 0.35:
                tissue_n += 1
                if hh in hs_tissue:
                    ov_tissue += 1

    print(f"sample_overlap_all={ov_all}/{n} ({ov_all/n:.6f})")
    denom = max(tissue_n, 1)
    print(f"sample_overlap_tissue={ov_tissue}/{denom} ({ov_tissue/denom:.6f}), tissue_n={tissue_n}")


if __name__ == "__main__":
    main()
