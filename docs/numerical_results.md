# Numerical results hub

Single place to look up **numbers that already exist on this machine** (mostly under `reports/`) plus **simple derived quantities** (differences, rates). Virchow **held-out test** scalars in §10 are ingested from `experiments/virchow_colab/evals_cross_domain/*/test_metrics_detailed.json`.

**Canonical narrative tables** also live in `docs/thesis_results_section.md`; §10.2 here is kept aligned to those JSON exports (re-run ingestion if you replace the files).

---

## 0. Source file index

| Topic | Primary path(s) |
| --- | --- |
| Dataset audit (shapes, labels, sampled RGB stats) | `reports/dataset_audit/audit_20260506_205634/master_summary.json`, `comparison_by_split.csv`, per-split `summary.json` |
| Exact-hash overlap PCam ↔ CAMELYON17 | `reports/data_integrity/preprocessed_overlap_exact.json` |
| Stain CNN benchmark (C5/C6) | `reports/stain_benchmark_c5_c6_metrics.json` |
| PCam raw-split anomaly scan (large JSON) | `reports/pcam_data_quality_report.json` |
| Virchow **held-out test** metrics (Table 8–9) | `test_metrics_detailed.json` + `test_metrics.json` from eval (see §10.2); often under `…/evals_cross_domain/<name>/` if you used `colab_eval_virchow_all_pending_tests.ipynb` |
| Virchow **bootstrap / permutation** (§10.3–10.5) | `reports/inference/virchow_test_inference.json`, `virchow_test_inference.csv` — from `scripts/virchow_test_bootstrap_inference.py` on `test_predictions.npz` |
| Virchow **training** end-of-run (mostly **val**) | Training run folder: `metrics_final.json`, `metrics_final_detailed.json`, `temperature_fit.json` (not a substitute for test JSON unless you also ran test eval into that folder) |
| CNN baseline full pipeline (§9) | `experiments/cnn_baseline_20260512_160329/evaluation/metrics_all.json`, `metrics_summary.csv`; per-arm `metrics_per_epoch.json`, `temperature_fit.json` |
| Qualitative sampling (pool sizes) | `reports/qualitative_error_analysis/virchow_c1_c4/*/bucket_sampling_summary.json` |
| Qualitative **human review** (checklist) | `reports/qualitative_error_analysis/virchow_c1_c4/*/review_labels_template.csv` (50 cases × 4 conditions; six items, no free-text) |

---

## 1. Dataset counts (before curation)

Public / WILDS catalogue sizes (same as Table 1 in `thesis_results_section.md`).

| Split | PCam patches | CAMELYON17 patches |
| --- | ---: | ---: |
| Train | 262,144 | 302,436 |
| Valid | 32,768 | 34,904 |
| Test | 32,768 | 85,054 |
| **All** | **327,680** | **422,394** |

---

## 2. PCam deduplication (exact content)

| Split | Before | After | Removed (dup collapse) |
| --- | ---: | ---: | ---: |
| Train | 262,144 | 220,025 | 42,119 |
| Valid | 32,768 | 28,108 | 4,660 |
| Test | 32,768 | 29,383 | 3,385 |
| **All** | **327,680** | **277,516** | **50,164** |

CAMELYON17 was **not** deduplicated in this pipeline; sizes stay as §1.

---

## 3. Quality filter + Macenko/Reinhard + \[0,1\] (full corpora)

From `preprocess_report.json` excerpts embedded in `master_summary.json` (audit 2026-05-06).

### 3.1 Pooled removal reasons

| Dataset | Candidates | Retained | Removed | Low tissue | Solid colour | High black |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCam | 277,516 | 262,574 | 14,942 | 12,451 | 2,478 | 13 |
| CAMELYON17 | 422,394 | 413,864 | 8,530 | 6,834 | 810 | 886 |

**Removal rate:** PCam \(14{,}942/277{,}516 = 5.38\%\); CAMELYON17 \(8{,}530/422{,}394 = 2.02\%\).

### 3.2 By split (candidates → retained)

| Split | PCam cand. | PCam kept | PCam drop | CAM17 cand. | CAM17 kept | CAM17 drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 220,025 | 208,355 | 11,670 | 302,436 | 296,294 | 6,142 |
| Valid | 28,108 | 26,515 | 1,593 | 34,904 | 34,389 | 515 |
| Test | 29,383 | 27,704 | 1,679 | 85,054 | 83,181 | 1,873 |

### 3.3 Class balance after filter (kept patches)

| Split | PCam \(n_+\) | PCam \(n_-\) | \(f_+\) | CAM17 \(n_+\) | CAM17 \(n_-\) | \(f_+\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 88,828 | 119,527 | 0.426 | 150,867 | 145,427 | 0.509 |
| Valid | 11,715 | 14,800 | 0.442 | 17,427 | 16,962 | 0.507 |
| Test | 12,940 | 14,764 | 0.467 | 42,468 | 40,713 | 0.511 |

### 3.4 Preprocessing pipeline label (both corpora)

`["quality_filter", "stain_normalization_macenko_reinhard", "value_normalization_0_1"]` — i.e. **Macenko + Reinhard machinery**, **\[0,1\]** float patches, **PCam train ref** (`ref_train_idx` 89340) documented in same JSON.

---

## 4. Cross-dataset exact duplicate (SHA256 of uint8 RGB)

`reports/data_integrity/preprocessed_overlap_exact.json`

| Quantity | Value |
| --- | ---: |
| PCam unique hashes (union splits) | 262,540 |
| CAM17 unique hashes (union splits) | 413,860 |
| Hashes in **both** unions | 1 |

Directional overlaps (patch count in source split with hash seen anywhere in other corpus):

| CAM17 split | \(n\) | overlap vs any PCam |
| --- | ---: | ---: |
| train | 296,294 | 4 |
| valid | 34,389 | 0 |
| test | 83,181 | 1 |

| PCam split | \(n\) | overlap vs any CAM17 |
| --- | ---: | ---: |
| train | 208,355 | 2 |
| valid | 26,515 | 0 |
| test | 27,704 | 0 |

**Rates** are \(\sim 10^{-5}\) (see JSON for float).

Thesis text: **3** duplicate training pairs *within* CAM17 only — see `thesis_results_section.md` §3.

---

## 5. Sampled appearance stats (audit, 4000 patches / split)

From `comparison_by_split.csv` (means over stratified sample).

| dataset | split | n | frac_pos | mean_R | mean_G | mean_B | tissue_proxy | black_ratio | blue_dom | dup_frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pcam | train | 208,355 | 0.426 | 0.715 | 0.579 | 0.798 | 0.733 | 3.69e-4 | 0.762 | 0.0 |
| pcam | valid | 26,515 | 0.442 | 0.714 | 0.576 | 0.797 | 0.738 | 3.55e-4 | 0.762 | 0.0 |
| pcam | test | 27,704 | 0.467 | 0.718 | 0.580 | 0.799 | 0.731 | 4.01e-4 | 0.768 | 0.0 |
| camelyon17 | train | 296,294 | 0.509 | 0.757 | 0.548 | 0.816 | 0.825 | 2.47e-4 | 0.601 | 0.0 |
| camelyon17 | valid | 34,389 | 0.507 | 0.686 | 0.490 | 0.769 | 0.850 | 7.28e-5 | 0.733 | 0.0 |
| camelyon17 | test | 83,181 | 0.511 | 0.718 | 0.514 | 0.790 | 0.812 | 2.25e-4 | 0.740 | 0.0 |

---

## 6. PCam **raw** split quality scan (separate from §3 pipeline)

`reports/pcam_data_quality_report.json` — **large** file listing anomaly indices.

**Train split headline (from JSON):** `n_samples` 262,144; balanced 131,072 / 131,072; `n_anomalous` 41,579 with flags such as `low_tissue` 40,157, `low_contrast` 2,768, `low_blur_score` 1,373, etc.  
Use that JSON for any table of raw-scan counts; not re-pasted here in full.

---

## 7. Stain-method CNN benchmark (C5 / C6, PCam benchmark splits)

Full precision: `reports/stain_benchmark_c5_c6_metrics.json`.

| ID | Arm | Val ROC-AUC | Test ROC-AUC | Δ (test−val) | Test acc | Test F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C5 | Macenko | 0.917670 | 0.925253 | **+0.007583** | 0.812245 | 0.841959 |
| C5 | Reinhard | 0.929304 | 0.917288 | −0.016016 | 0.806386 | 0.792434 |
| C5 | Vahadane | 0.918210 | 0.916754 | −0.001456 | 0.822317 | 0.844590 |
| C6 | Adaptive single-ref | 0.911668 | 0.912844 | +0.001176 | 0.828901 | 0.845656 |
| C6 | Adaptive multi-ref | 0.921192 | 0.881453 | −0.039739 | 0.800987 | 0.812527 |
| C6 | Adaptive multi-ref + aug | 0.924074 | 0.902942 | −0.021132 | 0.814483 | 0.820281 |

**Best test ROC-AUC:** Macenko (0.9253). **Best test acc / F1 on this CNN:** adaptive single-ref (0.829 / 0.846).

---

## 8. Resize kernel (96→224) — bicubic vs bilinear

Documented in `thesis_results_section.md` §8: bicubic preferred; **tabular scores not duplicated** in repo JSON — add when pulled from the resize comparison run.

---

## 9. CNN baseline (four arms, in-domain + cross-domain)

**Run:** `experiments/cnn_baseline_20260512_160329` (`run_id` `run_20260512_160329` in manifest; Colab `cnn_baseline_full_pipeline.py`, **10 epochs**, batch **320**, Adam **lr=0.001**, seed **42**, resumed training).

**Arms:** shallow CNN (methodology §4.1) on (1) PCam raw, (2) PCam Macenko preprocessed, (3) CAMELYON17 raw, (4) CAMELYON17 Macenko preprocessed. Each arm: train on that domain’s train split, select checkpoint by **best validation ROC-AUC (raw sigmoid)**, fit temperature on **source validation only**, evaluate **in-domain test** and **external test** (matched preprocessing: raw→raw, preprocessed→preprocessed).

**Outputs:** `evaluation/metrics_all.json` (full tables + `transfer_degradation_calibrated`), `evaluation/metrics_summary.csv`.

### 9.1 Preprocessed arms — aligned to Virchow C1–C4

Same **post-QC Macenko** test rows as Virchow (\(n_{\mathrm{test}}\) = 27,704 PCam, 83,181 CAMELYON17). Calibrated metrics @ threshold **0.5**.

| ID | Train → Test | CNN arm | \(T\) | ROC-AUC | PR-AUC | Acc | Bal acc | F1 | Brier | Log loss | ECE\(_{15}\) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| — | PCam → PCam | `pcam_preprocessed` | 1.4727 | 0.9480 | 0.9458 | 0.8705 | 0.8661 | 0.8523 | 0.0936 | 0.3006 | 0.0429 |
| — | PCam → CAM17 | ↑ external | 1.4727 | 0.9849 | 0.9874 | 0.9263 | 0.9255 | 0.9305 | 0.0535 | 0.1816 | 0.0576 |
| — | CAM17 → CAM17 | `cam17_preprocessed` | 1.7334 | 0.9847 | 0.9880 | 0.9424 | 0.9431 | 0.9416 | 0.0446 | 0.1576 | 0.0268 |
| — | CAM17 → PCam | ↑ external | 1.7334 | 0.8560 | 0.8672 | 0.6975 | 0.6768 | 0.5279 | 0.2308 | 0.7991 | 0.2516 |

**Confusion @ 0.5 (calibrated):**

| Train → Test | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| PCam → PCam | 10,350 | 13,766 | 998 | 2,590 |
| PCam → CAM17 | 40,998 | 36,054 | 4,659 | 1,470 |
| CAM17 → CAM17 | 38,620 | 39,770 | 943 | 3,848 |
| CAM17 → PCam | 4,686 | 14,638 | 126 | 8,254 |

**Transfer degradation** (\(\Delta_{\mathrm{abs}} = M_{\mathrm{in}} - M_{\mathrm{ext}}\) on calibrated metrics, from JSON):

| Arm | ROC-AUC \(\Delta_{\mathrm{in}-\mathrm{ext}}\) | PR-AUC \(\Delta_{\mathrm{in}-\mathrm{ext}}\) | Read |
| --- | ---: | ---: | --- |
| `pcam_preprocessed` | −0.0369 | −0.0416 | External CAM17 **better** than in-domain PCam (ranking) |
| `cam17_preprocessed` | +0.1287 | +0.1208 | Large **drop** on external PCam test |

### 9.2 Raw arms (public test counts; not directly comparable to Virchow)

Virchow runs use **preprocessed** corpora only. Raw CNN uses **full public** test splits (PCam test **32,768**; CAM17 test **85,054** before the same QC shrink).

| Train → Test | \(n_{\mathrm{test}}\) | \(T\) | ROC-AUC | Acc | Brier | ECE\(_{15}\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCam → PCam | 32,768 | 1.3574 | 0.8803 | 0.7799 | 0.1611 | 0.1390 |
| PCam → CAM17 | 85,054 | 1.3574 | 0.9841 | 0.9360 | 0.0473 | 0.0128 |
| CAM17 → CAM17 | 85,054 | 4.4247 | 0.7237 | 0.5539 | 0.2681 | 0.2299 |
| CAM17 → PCam | 32,768 | 4.4247 | 0.6759 | 0.5206 | 0.3084 | 0.2803 |

**Interpretation (raw):** CAMELYON17-trained **raw** model is near chance on CAM17 in-domain test (accuracy **0.55**, recall **0.16** on positives) — stain/domain gap without Macenko is severe. PCam raw in-domain is moderate (ROC-AUC **0.88**). Preprocessing is necessary for a fair domain-matched CNN baseline.

### 9.3 CNN training snapshot (validation ROC-AUC, raw sigmoid)

Best val ROC-AUC epoch (from `metrics_per_epoch.json`): `pcam_preprocessed` **0.947** (epoch 10); `cam17_preprocessed` **0.985** (epoch 10); `pcam_raw` **0.931** (epoch 9); `cam17_raw` **0.724** (epoch 10).

---

## 10. Virchow2 (C1–C4): test metrics and derived contrasts

### 10.1 Test set sizes (post-QC preprocessed rows)

| Test domain | \(n_{\mathrm{test}}\) |
| --- | ---: |
| PCam | 27,704 |
| CAMELYON17 | 83,181 |

### 10.2 Reported scalars — ingested from `test_metrics_detailed.json`

**Source paths (this repo):** each row is read from `experiments/virchow_colab/evals_cross_domain/<folder>/test_metrics_detailed.json` (output of `scripts/evaluate_virchow_preprocessed_test_colab.py`). Folder names match `colab_eval_virchow_all_pending_tests.ipynb`.

| Condition | Subfolder |
| --- | --- |
| C1 PCam → PCam | `pcam_trained_on_pcam_test` |
| C2 PCam → CAM17 | `pcam_trained_on_cam17_test` |
| C3 CAM17 → CAM17 | `cam17_trained_on_cam17_test` |
| C4 CAM17 → PCam | `cam17_trained_on_pcam_test` |

**Calibrated probabilities** = `metrics_test_prob_after_temperature` (temperature \(T\) from the **training-domain** fit, stored as `temperature_used` in JSON). **Raw** = `metrics_test_prob_raw_sigmoid`. Threshold **0.5** for confusion / accuracy / F1 / etc.

#### 10.2a Calibrated test metrics (\(T\) scaled logits → sigmoid)

| ID | Train → Test | \(n_{\mathrm{test}}\) | \(T\) | ROC-AUC | PR-AUC | Acc | Bal acc | Prec | Recall | F1 | Brier | Log loss | ECE\(_{15}\) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | PCam → PCam | 27,704 | 1.2783 | 0.9813 | 0.9820 | 0.9367 | 0.9339 | 0.9702 | 0.8919 | 0.9294 | 0.0486 | 0.1702 | 0.0200 |
| C2 | PCam → CAM17 | 83,181 | 1.2783 | 0.9971 | 0.9977 | 0.9840 | 0.9841 | 0.9867 | 0.9820 | 0.9843 | 0.0134 | 0.0559 | 0.0100 |
| C3 | CAM17 → CAM17 | 83,181 | 1.5900 | 0.9966 | 0.9972 | 0.9798 | 0.9798 | 0.9795 | 0.9810 | 0.9803 | 0.0174 | 0.0712 | 0.0206 |
| C4 | CAM17 → PCam | 27,704 | 1.5900 | 0.9502 | 0.9530 | 0.8237 | 0.8117 | 0.9877 | 0.6304 | 0.7696 | 0.1363 | 0.4716 | 0.1600 |

#### 10.2b Raw sigmoid test metrics (no temperature)

| ID | ROC-AUC | PR-AUC | Acc | Bal acc | Prec | Recall | F1 | Brier | Log loss | ECE\(_{15}\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 0.9813 | 0.9820 | 0.9367 | 0.9339 | 0.9702 | 0.8919 | 0.9294 | 0.0490 | 0.1751 | 0.0268 |
| C2 | 0.9971 | 0.9977 | 0.9840 | 0.9841 | 0.9867 | 0.9820 | 0.9843 | 0.0131 | 0.0531 | 0.0025 |
| C3 | 0.9966 | 0.9972 | 0.9798 | 0.9798 | 0.9795 | 0.9810 | 0.9803 | 0.0161 | 0.0654 | 0.0077 |
| C4 | 0.9502 | 0.9530 | 0.8237 | 0.8117 | 0.9877 | 0.6304 | 0.7696 | 0.1485 | 0.6582 | 0.1666 |

#### 10.2c Confusion @ 0.5 (calibrated branch; same counts as raw for these runs)

| ID | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| C1 | 11,541 | 14,409 | 355 | 1,399 |
| C2 | 41,702 | 40,149 | 564 | 766 |
| C3 | 41,662 | 39,841 | 872 | 806 |
| C4 | 8,157 | 14,662 | 102 | 4,783 |

**Note:** PCam-trained arms (C1, C2) share the same \(T = 1.2783\); CAM17-trained arms (C3, C4) share \(T = 1.5900\) — each comes from that model’s training-domain validation fit, then applied on every test domain.

### 10.3 Bootstrap 95% percentile CIs (calibrated test)

**Source:** `reports/inference/virchow_test_inference.json` — case-level bootstrap \(B = 2000\), seed 42, on `prob_after_temperature` in each `test_predictions.npz` (methodology §7). Format: point \([\)2.5th, 97.5th\(]\) percentile.

| ID | ROC-AUC | PR-AUC | Accuracy | Brier | Log loss | ECE\(_{15}\) |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 0.9813 [0.9799, 0.9827] | 0.9820 [0.9807, 0.9834] | 0.9367 [0.9337, 0.9396] | 0.0486 [0.0467, 0.0504] | 0.1702 [0.1641, 0.1763] | 0.0200 [0.0180, 0.0227] |
| C2 | 0.9971 [0.9967, 0.9974] | 0.9977 [0.9975, 0.9979] | 0.9840 [0.9831, 0.9849] | 0.0134 [0.0129, 0.0140] | 0.0559 [0.0539, 0.0580] | 0.0100 [0.0094, 0.0109] |
| C3 | 0.9966 [0.9962, 0.9969] | 0.9972 [0.9970, 0.9974] | 0.9798 [0.9789, 0.9808] | 0.0174 [0.0168, 0.0180] | 0.0712 [0.0694, 0.0732] | 0.0206 [0.0198, 0.0215] |
| C4 | 0.9502 [0.9477, 0.9526] | 0.9530 [0.9505, 0.9553] | 0.8237 [0.8192, 0.8283] | 0.1363 [0.1329, 0.1395] | 0.4716 [0.4591, 0.4835] | 0.1600 [0.1562, 0.1639] |

Full CIs for precision, recall, F1, and balanced accuracy are in the JSON/CSV.

### 10.4 Paired bootstrap transfer contrasts (same test patches)

Each replicate resamples **the same** test indices and computes \(\Delta = M_{\text{in}} - M_{\text{ext}}\) (methodology §7: in-domain minus external on a shared test surface).

**PCam test** — C1 (PCam-trained) minus C4 (CAMELYON17-trained), \(n = 27{,}704\):

| Metric | \(\Delta\) point | 95% CI for \(\Delta\) | CI excludes 0? |
| --- | ---: | --- | :---: |
| ROC-AUC | +0.0311 | [0.0290, 0.0331] | yes |
| Accuracy | +0.1130 | [0.1091, 0.1172] | yes |
| Brier | −0.0877 | [−0.0905, −0.0852] | yes |
| Log loss | −0.3014 | [−0.3113, −0.2921] | yes |
| ECE\(_{15}\) | −0.1401 | [−0.1428, −0.1368] | yes |

**CAMELYON17 test** — C3 (CAMELYON17-trained) minus C2 (PCam-trained), \(n = 83{,}181\):

| Metric | \(\Delta\) point | 95% CI for \(\Delta\) | CI excludes 0? |
| --- | ---: | --- | :---: |
| ROC-AUC | −0.00049 | [−0.00081, −0.00017] | yes |
| Accuracy | −0.00418 | [−0.00512, −0.00326] | yes |
| Brier | +0.00394 | [0.00345, 0.00447] | yes |
| Log loss | +0.0153 | [0.0136, 0.0171] | yes |
| ECE\(_{15}\) | +0.0106 | [0.00969, 0.0114] | yes |

**Marginal** in-domain vs external on **different** test splits (C1 vs C2, C3 vs C4): point \(\Delta\) only in §10.6 — not paired; each arm has its own bootstrap CI in §10.3.

### 10.5 Paired permutation tests and Benjamini–Hochberg

**Source:** same inference JSON; \(10{,}000\) sign-flips on per-patch score differences; confirmatory family = Brier and log-loss (NLL) head-to-head on shared test patches.

| Test ID | Comparison | \(p\) | \(p_{\mathrm{BH}}\) (\(q=0.05\)) | Reject? |
| --- | --- | ---: | ---: | :---: |
| PCam test, Brier | C1 vs C4 | \(< 10^{-4}\) | \(< 10^{-4}\) | yes |
| PCam test, log loss | C1 vs C4 | \(< 10^{-4}\) | \(< 10^{-4}\) | yes |
| CAM17 test, Brier | C3 vs C2 | \(< 10^{-4}\) | \(< 10^{-4}\) | yes |
| CAM17 test, log loss | C3 vs C2 | \(< 10^{-4}\) | \(< 10^{-4}\) | yes |

Re-run: `python scripts/virchow_test_bootstrap_inference.py --workers 4`

### 10.6 Simple cross-domain deltas (point estimates, calibrated @ 0.5)

**PCam-trained**

| Metric | In-domain (C1, PCam test) | External (C2, CAM17 test) | Δ (C2−C1) |
| --- | ---: | ---: | ---: |
| ROC-AUC | 0.9813 | 0.9971 | +0.0157 |
| Accuracy | 0.9367 | 0.9840 | +0.0473 |

**CAMELYON17-trained**

| Metric | In-domain (C3, CAM17 test) | External (C4, PCam test) | Δ (C4−C3) |
| --- | ---: | ---: | ---: |
| ROC-AUC | 0.9966 | 0.9502 | −0.0463 |
| Accuracy | 0.9798 | 0.8237 | −0.1562 |

Interpretation: asymmetric transfer at **accuracy** on PCam test when training on CAM17; ROC-AUC on PCam test remains high (0.95) but below in-domain CAM17. Paired bootstrap on PCam test (§10.4) shows C1 beats C4 on discrimination and accuracy; on CAM17 test, C3 vs C2 differs only slightly on ROC-AUC but remains statistically detectable at this \(n\).

### 10.7 Uncertainty / confidence definitions (evaluation)

- **Confidence** \(c = \max(\hat p, 1-\hat p)\) on **calibrated** \(\hat p\) after temperature.
- **Entropy** \(H\) of Bernoulli(\(\hat p\)); **\(\tau_H\)** = 90th percentile of \(H\) **on that test split**.
- **Buckets** (qualitative export): FP, FN, high-entropy error, high-entropy correct, optional confident error (\(c \ge 0.9\) & wrong).

### 10.8 CNN vs Virchow2 (preprocessed Macenko, same \(n_{\mathrm{test}}\))

Fair comparison: **preprocessed** arms only (§9.1 vs §10.2). Virchow = frozen Virchow2 + linear head; CNN = shallow 4-block baseline (same pipeline rules: val temperature, test @ 0.5).

| ID | Condition | Metric | Virchow | CNN | Virchow − CNN |
| --- | --- | --- | ---: | ---: | ---: |
| C1 | PCam → PCam | ROC-AUC | 0.9813 | 0.9480 | +0.0333 |
| C1 | PCam → PCam | Accuracy | 0.9367 | 0.8705 | +0.0662 |
| C1 | PCam → PCam | Brier | 0.0486 | 0.0936 | −0.0450 |
| C2 | PCam → CAM17 | ROC-AUC | 0.9971 | 0.9849 | +0.0122 |
| C2 | PCam → CAM17 | Accuracy | 0.9840 | 0.9263 | +0.0577 |
| C3 | CAM17 → CAM17 | ROC-AUC | 0.9966 | 0.9847 | +0.0119 |
| C3 | CAM17 → CAM17 | Accuracy | 0.9798 | 0.9424 | +0.0374 |
| C4 | CAM17 → PCam | ROC-AUC | 0.9502 | 0.8560 | +0.0942 |
| C4 | CAM17 → PCam | Accuracy | 0.8237 | 0.6975 | +0.1262 |
| C4 | CAM17 → PCam | Brier | 0.1363 | 0.2308 | −0.0945 |
| C4 | CAM17 → PCam | ECE\(_{15}\) | 0.1600 | 0.2516 | −0.0916 |

**Cross-domain ROC-AUC change** (external minus in-domain; positive = higher on external):

| Training origin | Virchow | CNN |
| --- | ---: | ---: |
| PCam | +0.0157 (C2−C1) | +0.0369 |
| CAMELYON17 | −0.0463 (C4−C3) | −0.1287 |

**Synthesis (for Results text):**

1. **Virchow2 is stronger on every preprocessed cell** in the table: higher ROC-AUC and accuracy, lower Brier on PCam test for both training origins. The largest gap is **C4** (CAM17-trained → PCam test): Virchow keeps ROC-AUC **0.95** and accuracy **0.82** vs CNN **0.86** / **0.70**, with much worse CNN calibration (ECE **0.25** vs **0.16**).

2. **Transfer pattern is qualitatively similar** but **steeper for CNN** on the hard direction: both show higher ROC on external CAM17 when trained on PCam; both drop on external PCam when trained on CAM17. CNN’s CAM17→PCam ROC drop (**0.13** absolute) is about **2.8×** Virchow’s (**0.05**).

3. **Preprocessing matters for CNN more than for Virchow:** raw CAM17-trained CNN in-domain ROC-AUC is only **0.72** (§9.2); preprocessed CNN reaches **0.98** — Virchow was only evaluated preprocessed, so the headline model comparison is §9.1 vs §10.2, not raw CNN.

4. **Stain benchmark CNN** (§7, subset PCam) peaked at test ROC-AUC **0.925** (Macenko C5); the full **pcam_preprocessed** CNN reaches **0.948** in-domain — different splits and corpus size, but same architecture family.

Virchow bootstrap CIs (§10.3) are **not** recomputed for CNN here; add `scripts/virchow_test_bootstrap_inference.py`-style bootstrap on CNN test NPZ if you need symmetric uncertainty bands.

---

## 11. Qualitative error analysis

### 11.1 Sampling pool (full test split)

\(n_{\mathrm{test}}\) and \(\tau_H\) from each `bucket_sampling_summary.json`; `available_n` = patches in bucket on full test; `sampled_n` = 10 (protocol).

| Condition | \(n_{\mathrm{test}}\) | \(\tau_H\) | FP avail | FN avail | HE err | HE ok | CE avail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 27,704 | 0.5240 | 355 | 1,399 | 929 | 1,842 | 423 |
| C2 | 83,181 | 0.2325 | 564 | 766 | 1,035 | 7,284 | 399 |
| C3 | 83,181 | 0.3577 | 872 | 806 | 1,278 | 7,041 | 362 |
| C4 | 27,704 | 0.5928 | 102 | 4,783 | 1,257 | 1,514 | 2,125 |

### 11.2 Human checklist review (completed)

**Source:** `reports/qualitative_error_analysis/virchow_c1_c4/<condition>/review_labels_template.csv` (exported from filled spreadsheets; same columns as protocol §6). **\(n = 50\)** reviewed cases per condition (**10** per bucket: FP, FN, high-entropy error, high-entropy correct, confident error). Counts are **Present** / \(k\) of \(n=10\) in bucket; no “Unclear” marks in completed files.

Checklist items: (1) tissue scarcity, (2) artifact burden, (3) borderline morphology, (4) small-focus lesion, (5) color/stain atypia, (6) patch-context limitation.

#### C1 — PCam-trained → PCam test

| Bucket | Tissue | Artifact | Borderline | Small focus | Stain atypia | Context limit |
| --- | --- | --- | --- | --- | --- | --- |
| FP | 1/10 | 5/10 | 4/10 | 2/10 | 3/10 | 2/10 |
| FN | 2/10 | 2/10 | 4/10 | 5/10 | 2/10 | **8/10** |
| HE error | 2/10 | 2/10 | 7/10 | 4/10 | 2/10 | 4/10 |
| HE correct | 1/10 | 2/10 | 6/10 | 2/10 | 2/10 | 2/10 |
| Conf. error | 1/10 | 4/10 | 3/10 | 2/10 | 4/10 | 2/10 |

#### C2 — PCam-trained → CAMELYON17 test

| Bucket | Tissue | Artifact | Borderline | Small focus | Stain atypia | Context limit |
| --- | --- | --- | --- | --- | --- | --- |
| FP | 2/10 | 4/10 | 4/10 | 2/10 | **6/10** | 2/10 |
| FN | 2/10 | 3/10 | 4/10 | 4/10 | **6/10** | 6/10 |
| HE error | 2/10 | 3/10 | **8/10** | 4/10 | 4/10 | 4/10 |
| HE correct | 1/10 | 2/10 | 6/10 | 2/10 | 4/10 | 3/10 |
| Conf. error | 1/10 | 4/10 | 4/10 | 3/10 | 5/10 | 2/10 |

#### C3 — CAMELYON17-trained → CAMELYON17 test

| Bucket | Tissue | Artifact | Borderline | Small focus | Stain atypia | Context limit |
| --- | --- | --- | --- | --- | --- | --- |
| FP | 2/10 | 6/10 | 4/10 | 2/10 | 3/10 | 2/10 |
| FN | 2/10 | 2/10 | 4/10 | 4/10 | 2/10 | 4/10 |
| HE error | 2/10 | 3/10 | 7/10 | 5/10 | 2/10 | 4/10 |
| HE correct | 1/10 | 2/10 | 6/10 | 2/10 | 2/10 | 2/10 |
| Conf. error | 1/10 | 5/10 | 4/10 | 3/10 | 3/10 | 2/10 |

#### C4 — CAMELYON17-trained → PCam test (external)

| Bucket | Tissue | Artifact | Borderline | Small focus | Stain atypia | Context limit |
| --- | --- | --- | --- | --- | --- | --- |
| FP | 2/10 | 4/10 | 4/10 | 3/10 | 4/10 | 2/10 |
| FN | 2/10 | 2/10 | 4/10 | **6/10** | 4/10 | 6/10 |
| HE error | 2/10 | 3/10 | **8/10** | 4/10 | 4/10 | 5/10 |
| HE correct | 1/10 | 2/10 | 6/10 | 3/10 | 2/10 | 3/10 |
| Conf. error | 1/10 | 4/10 | 4/10 | 3/10 | 4/10 | 2/10 |

### 11.3 Cross-condition patterns (FN and high-entropy error buckets)

| Pattern (Present in FN bucket) | C1 | C2 | C3 | C4 |
| --- | ---: | ---: | ---: | ---: |
| Patch-context limitation | 8/10 | 6/10 | 4/10 | 6/10 |
| Small-focus lesion | 5/10 | 4/10 | 4/10 | **6/10** |
| Color/stain atypia | 2/10 | **6/10** | 2/10 | 4/10 |
| Borderline morphology (HE **error** bucket) | 7/10 | **8/10** | 7/10 | **8/10** |

### 11.4 Interpretation (linked to §10 metrics)

1. **C1 false negatives** — Patch-context limitation was marked Present in **8/10** FN reviews (vs 2/10 in FP), consistent with missed positives where the central \(32\times32\) tumour focus may be hard to judge at patch scale despite strong overall ROC-AUC (§10.2).

2. **External CAMELYON17 (C2)** — **Color/stain atypia** was common in FN and FP (**6/10** each), aligning with cross-domain appearance shift while numeric transfer on CAM17 test remains excellent (ROC-AUC 0.997).

3. **C4 external PCam** — The test split has **4,783** FN patches available (§11.1); sampled FN cases show **6/10** small-focus lesion and **6/10** context limitation, matching the large accuracy drop (§10.6) and high ECE (§10.2) when CAMELYON17-trained weights are applied to PCam.

4. **High-entropy errors** — **Borderline morphology** was Present in **7–8/10** HE-error cases in every condition, supporting that ambiguous morphology drives uncertain wrong predictions rather than artifacts alone.

5. **Scope** — These are **descriptive** prevalences in \(n=10\) per bucket samples; they support but do not replace quantitative metrics in §10. Figures: `…/figures/<bucket>/` under each condition folder.

---

## 12. Items intentionally **not** duplicated here

- Full **Virchow** train/val **loss curves** (per-epoch): in run `metrics_history` / TensorBoard exports — too large for this hub; point figures to supplement.
- **Reliability diagrams** (figures from `ece_15_bins_*` in `test_metrics_detailed.json`).
- **Per-patch** raw PCam anomaly index lists: `pcam_data_quality_report.json` only.

---

## 13. Changelog

| Date | Note |
| --- | --- |
| 2026-05-12 | Initial hub from `reports/` audit, overlap, stain JSON, thesis Virchow partials, qualitative bucket summaries. |
| 2026-05-12 | §10.2–10.3: filled Virchow C1–C4 from `experiments/virchow_colab/evals_cross_domain/*/test_metrics_detailed.json` (calibrated + raw + confusion; deltas). |
| 2026-05-15 | §10.3–10.5: bootstrap CIs, paired transfer \(\Delta\) CIs, permutation + BH from `reports/inference/virchow_test_inference.json`. |
| 2026-05-15 | §9: CNN baseline from `cnn_baseline_20260512_160329`; §10.8 CNN vs Virchow comparison. |
| 2026-05-15 | §11.2–11.4: qualitative checklist prevalences from completed `review_labels_template.csv` (C1–C4). |

When you add new JSON from Colab, append a short subsection with the file path and paste the headline scalars so this document stays the one-stop table of contents.
