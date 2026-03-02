# Supervisor meeting prep — PCam baseline, deduplication, and project status

**Purpose:** Summarise methodology, rationale, alternatives, and results for the PCam baseline training and deduplication pipeline. Use this to walk through the work and answer questions.

---

## 1. Project context and goal

- **Project:** Histopathology (PatchCamelyon first; CAMELYON17 later) for metastasis detection.
- **Phased plan:** (1) Define task and metrics → (2) Data setup and baseline on PCam → (3) Augmentation, domain robustness, uncertainty, explainability → (4) Then move to CAMELYON17 for external validation.
- **Current focus:** PCam baseline established; deduplication done; data quality tracked; next steps are training on dedup data, stain robustness, and transfer learning (Virchow2).

---

## 2. PCam baseline training

### 2.1 Task and data

- **Task:** Binary classification on 96×96 RGB patches: **negative (0)** vs **positive (1)** — “metastasis in the center 32×32 region” (PCam definition).
- **Data:** Official splits only (no custom split).
  - **Train:** 262,144 | **Validation:** 32,768 | **Test:** 32,768
- **Source:** Loaded via `load_data(data_dir=pcam_data)` from HDF5 + CSV (from `pcam-master`).

### 2.2 Methodology

**Preprocessing**

- Pixel normalization: `x = x / 255.0` (input range [0, 1]).
- No augmentation in the current baseline (to be added later).
- Batches streamed from HDF5 with a generator (no full load into RAM).

**Model**

- **Architecture:** Small CNN (baseline, not SOTA).
  - 4 Conv blocks: 32 → 64 → 128 → 256 filters (3×3, ReLU, same padding), each followed by 2×2 MaxPool.
  - Global average pooling → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid).
- **Rationale:** Simple, fast to train, interpretable; establishes a performance floor before adding augmentation, dedup, or transfer learning.
- **Training:** Adam, lr=1e-3; binary cross-entropy; metric: accuracy. Best model selected by **validation accuracy** (ModelCheckpoint, save weights only).
- **Epochs:** 10 (configurable).
- **Batch size:** 64. Steps per epoch = n_train // 64 (e.g. 4096 train, 512 val).

**Evaluation**

- **Metrics:** AUC-ROC, accuracy, confusion matrix (TP, FP, TN, FN) on validation and test.
- **Outputs:** Best weights (`model_best.weights.h5`), training CSV log, `metrics.json` (val/test AUC, accuracy, confusion matrices), and a confusion-matrix figure.

### 2.3 Results (from a full run)

- **Validation:** AUC ≈ 0.934, Accuracy ≈ 0.861.
- **Test:** AUC ≈ 0.877, Accuracy ≈ 0.801.
- Val confusion matrix (rows = true, cols = pred): e.g. [[15173, 1226], [3325, 13044]] (TN, FP, FN, TP).
- Test confusion matrix: e.g. [[15224, 1167], [5332, 11045]] — test is harder (more FN), as expected.

**Where results live:** `experiments/pcam_baseline/` — `model_best.weights.h5`, `metrics.json`, `training_log.csv`, `confusion_matrices.png`.

### 2.4 Why this baseline design

- **Simple CNN:** Reproducible, quick to iterate, no pretraining; later we can compare to Virchow2/transfer learning.
- **Official splits only:** Matches the PCam benchmark; no extra tuning of the split.
- **AUC + accuracy + confusion matrix:** AUC for class imbalance; accuracy for interpretability; CM to spot systematic bias (e.g. one-sided errors).
- **No augmentation yet:** Clean baseline; augmentation (and stain robustness) will be the next comparison.

### 2.5 What we did *not* do (and why)

- **No custom train/val split:** Keeps evaluation comparable to the literature.
- **No heavy architecture (ResNet, etc.):** Deliberately lightweight; heavier models come with transfer learning.
- **No augmentation in this run:** To isolate effect of augmentation in a later step.

---

## 3. PCam deduplication (dedup)

### 3.1 What it does

- **Input:** Same PCam data (train/valid/test) loaded via the official loader.
- **Process:** For each split, compute a **SHA-256 hash** of the raw bytes of each patch. Group indices by hash; within each group with more than one index, **keep one** (e.g. first) and drop the rest.
- **Output (default):**
  - **Manifest:** `manifest.json` (data path, per-split counts: n_original, n_kept, n_removed, index file names).
  - **Per split:** `train_kept_indices.npy`, `valid_kept_indices.npy`, `test_kept_indices.npy` — 1D arrays of indices to use. Training then uses “load data, then index with these arrays” (no new HDF5 by default).
- **Verification:** `--verify` re-hashes only the kept indices and checks (1) no duplicate hashes among kept, (2) n_original = n_kept + n_removed.

**Script:** `scripts/dedup_pcam.py`. Run from project root:
```bash
python scripts/dedup_pcam.py --data-dir pcam_data --out-dir pcam_dedup
python scripts/dedup_pcam.py --out-dir pcam_dedup --verify
```

### 3.2 Rationale for this method

- **Exact duplicate definition:** Two patches are duplicates **only if they are byte-identical** (same pixels). SHA-256 of raw patch bytes gives that. Collision probability is negligible, so we do not remove “similar” patches by mistake.
- **Why remove duplicates at all:** Duplicates (a) inflate effective dataset size and can overstate generalisation, (b) can cause train/val/test leakage if the same patch appears in more than one split, (c) waste compute. Removing them gives a cleaner, more honest evaluation.
- **Why hash-based (exact) rather than “near-duplicate” (e.g. perceptual hash, embeddings):** We want a conservative, unambiguous rule: only drop when patches are identical. Near-duplicate detectors introduce thresholds and possible false positives; for a first pass we keep the method simple and explainable.
- **Why keep indices instead of writing new HDF5 by default:** Saves disk and keeps a single source of truth (original HDF5). We can still write new HDF5 with `--write-h5` if we want a drop-in loader later.

### 3.3 Other methods considered

| Approach | Pros | Cons | Why we didn’t use it (for now) |
|----------|------|------|----------------------------------|
| **Hash (SHA-256) exact** | No false positives, simple, verifiable | Only catches pixel-identical duplicates | **Chosen:** Safe, interpretable, no tuning. |
| Perceptual / embedding similarity | Can find “visually similar” patches | Threshold choice, risk of removing valid variation, more complex | Deferred; can add later if we want to study near-duplicates. |
| Leave duplicates in | No preprocessing | Leakage risk, inflated metrics, wasted compute | We want a cleaner benchmark. |
| Write new HDF5 always | Drop-in loader | Extra disk, duplication | Optional via `--write-h5`; default is indices only. |

### 3.4 Typical numbers (from PCam)

- Train: large number of duplicate groups (e.g. tens of thousands of groups, ~57k duplicate images removed); valid/test: fewer but non-zero.
- Exact figures are in `pcam_dedup/manifest.json` after a run.

### 3.5 Integration with training

- **Current baseline:** Trained on **all** samples (no dedup filter). So baseline numbers are “with duplicates.”
- **Next step:** Train the same (or same-style) baseline using **only** `train_kept_indices` (and val/test kept indices for evaluation) and report “baseline after dedup” for a fair comparison and to avoid leakage.

---

## 4. Data quality and investigation

- **Dataset investigation notebook:** `notebooks/pcam_data_investigation.ipynb` — file inventory, sizes, dimensions, label counts, metadata, **and** data quality (anomalies, duplicates, blur/contrast, stain stats).
- **Quality report (in same notebook):** We run a scan that flags:
  - **Anomalies:** zero_std, all_black, all_white, high black/white ratio, low blur, low contrast, low tissue.
  - **Duplicate groups:** same as in dedup (hash-based).
  - **Per-patch stats:** ratio_white, ratio_black (strict and relaxed for display), so we can set thresholds later (e.g. exclude very white/blank patches).
- **Output:** Report can be saved to JSON (e.g. `reports/pcam_data_quality_report.json`) with per-split anomaly counts, duplicate counts, and optional anomaly indices/details. This supports “filter by quality” in a later training pipeline (e.g. exclude anomaly indices or high-white patches) and is separate from the dedup script (dedup = exact duplicates only; quality = broader flags for optional filtering).

---

## 5. Reproducibility and artefacts

- **Baseline:** `notebooks/pcam_baseline_training.ipynb` — run in order; experiment dir `experiments/pcam_baseline/` holds weights, metrics, and figures.
- **Dedup:** `scripts/dedup_pcam.py` + manifest and `.npy` indices in `pcam_dedup/` (or custom `--out-dir`).
- **Data:** `pcam_data/` (extracted from `pcamv1/` via `extract_pcam.py`); not in git (size).

---

## 6. Next steps (for discussion)

1. **Baseline on dedup data:** Train the same CNN using `*_kept_indices.npy`; report val/test AUC and accuracy and compare to “baseline with duplicates.”
2. **Stain robustness:** Prefer **stain augmentation** (and optionally normalisation as one augmentation) so the model is stain-invariant rather than tied to one normaliser; goal: “any stain in the future.”
3. **Transfer learning (Virchow2):** Backbone + small head on 2560-d embeddings; 96→224 resize; use same dedup indices and evaluation protocol for comparability.
4. **Uncertainty and explainability:** Add after we have a stable baseline and dedup comparison (as in the original phased plan).
5. **CAMELYON17:** After PCam pipeline and metrics are fixed (baseline, dedup, augmentation, transfer learning), define the external evaluation setup and run the same pipeline there.

---

## 7. Short “elevator” summary

- **Baseline:** Simple CNN on PCam, official splits; val AUC ~0.93, test AUC ~0.88; metrics and confusion matrices saved.
- **Dedup:** Hash-based (SHA-256) exact duplicate removal; output = manifest + kept indices per split; verification step ensures no non-duplicate removed and counts consistent.
- **Rationale:** Baseline = fast, interpretable floor; dedup = avoid leakage and inflated metrics; both support a clean comparison when we add augmentation, transfer learning, and later CAMELYON17.

You can use this doc to walk through the methodology, justify choices, and align on next steps (dedup baseline, stain robustness, Virchow2) in the meeting.
