# RESULTS

This chapter reports measured outcomes for the pipeline frozen in the Methodology document: split integrity and overlap screening, quality filtering and stain handling where those steps produced counts or benchmark scores, resize policy evidence, final cohort sizes used for training and testing, and classifier performance for the registered conditions C1–C8. The order follows the same logic as the Methodology chapter so that each procedural claim can be tied to a table or figure here. Primary emphasis is placed on the Virchow2 conditions C1–C4 on Macenko benchmark-style preprocessed held-out tests, with temperature scaling applied under the validation-only rule and decisions at probability 0.5 as defined earlier. Secondary material covers the conventional CNN reference (C7), the stain-method benchmark family (C5–C6) when reported in full, bootstrap and permutation summaries planned under Chapter 7, and the structured qualitative review (C8) once bucket tallies and exemplar panels are finalized. Past tense is used for empirical statements; definitions stay in the Methodology chapter and are not repeated as equations here.

**Figure R.0 (optional overview).** Placeholder for a single results-overview schematic that mirrors Figure 1 in the Methodology but annotates only measured stages (which subsections below hold numbers versus pending items). Omit if redundant with the thesis figure list.

---

## 1. Pipeline evidence: integrity, filtering, benchmarks, and final cohorts

Cross-dataset exact duplicate screening on SHA-256 hashes of stored uint8 RGB patch bytes for the Macenko benchmark-style preprocessed packs showed vanishingly small overlap rates between PCam and CAMELYON17 on every split, with a single hash in the union intersection table across the two corpora. The directional checks are summarized in Table R.1; rates are reported with respect to the CAMELYON17 or PCam sample count on each split as indicated. These figures support the claim that external evaluation is not trivially duplicated pixel-for-pixel across domains at the preprocessed stage, while still acknowledging that near-duplicates below exact identity are not captured by this test.

**Table R.1. Exact-hash overlap between Macenko preprocessed PCam and CAMELYON17 (selected rows).** Values are taken from `reports/data_integrity/preprocessed_overlap_exact.json` (run timestamp 2026-05-06).

| Comparison | Split | Samples in source split | Patches with hash also present in other corpus | Rate |
|---|---|---:|---:|---|
| CAMELYON17 vs any PCam | train | 296294 | 4 | 1.35×10⁻⁵ |
| CAMELYON17 vs any PCam | valid | 34389 | 0 | 0 |
| CAMELYON17 vs any PCam | test | 83181 | 1 | 1.20×10⁻⁵ |
| PCam vs any CAMELYON17 | train | 208355 | 2 | 9.60×10⁻⁶ |
| PCam vs any CAMELYON17 | valid | 26515 | 0 | 0 |
| PCam vs any CAMELYON17 | test | 27704 | 0 | 0 |

Within-PCam duplicate handling at the raw stage and quality filtering with fixed tissue, black-fraction, and solid-colour gates were applied before stain normalization as described in the Methodology; the corresponding before-and-after patch counts per split and per exclusion reason should appear in Table R.2 once consolidated from preprocessing manifests and QC logs. **Figure R.1** is reserved for split-wise bar charts of removals by rule (solid-colour, high-black, low-tissue) and, if space allows, paired histograms of tissue proxy or black fraction before versus after filtering on a fixed random sample of patches.

The stain-method benchmark on the controlled PCam subset indices (train 40k, validation 8k, test 16k targets before QC, shared indices across methods) compared Macenko, Reinhard, Vahadane, and the adaptive routing variants under the shared small CNN; headline discrimination metrics and any distance-to-reference summaries used to justify stability belong in **Table R.3** and **Figure R.2** (for example a grouped bar chart of validation ROC-AUC or the agreed primary benchmark metric per method, plus one diagnostic panel for mean RGB movement after normalization). The bicubic versus bilinear resize comparison on the same benchmark configuration should be reduced to **Table R.4** and a small **Figure R.3** (paired bars or difference-with-95%-CI plot), with one sentence recording the informal sharpness check that supported bicubic for all downstream 96→224 resizing.

After integrity, QC, stain normalization, and value scaling, the Macenko preprocessed HDF5 splits used for Virchow2 development contain the unique hash counts per split already implied by the overlap file (PCam train 208354, valid 26515, test 27704; CAMELYON17 train 296291, valid 34389, test 83181 unique hashes in the preprocessed packs at the time of that audit). **Table R.5** should reconcile these counts with the actual dataset lengths seen by the training loaders and with any updated totals if later filtering changed them; footnotes must explain any deviation from Table 2.3 in the Methodology. **Figure R.4** can show class balance bars for train, validation, and test on each dataset after all preprocessing if not already shown in the Data chapter.

---

## 2. Registered conditions, artefact locations, and completion status

Conditions C1–C8 tie models, preprocessing arms, and evaluation surfaces to a single index used in tables and figure captions. **Table R.6** lists the intended mapping from the Methodology, the artefact directories used in this repository for the completed Virchow runs, and a completion note for items still pending at the time of drafting. Paths are given relative to the repository root; Colab absolute paths inside JSON artefacts are ignored for thesis traceability.

**Table R.6. Condition index and artefact traceability (representative paths).**

| ID | Short description | Primary artefact location (this repo) | Status |
|---|---|---|---|
| C1 | Virchow2 PCam train → PCam test (Macenko) | `experiments/virchow_colab/evals_cross_domain/pcam_trained_on_pcam_test/` | Test metrics and predictions exported |
| C2 | Virchow2 PCam train → CAMELYON17 test (Macenko) | `experiments/virchow_colab/evals_cross_domain/pcam_trained_on_cam17_test/` | Test metrics and predictions exported |
| C3 | Virchow2 CAMELYON17 train → CAMELYON17 test (Macenko) | `experiments/virchow_colab/evals_cross_domain/cam17_trained_on_cam17_test/` | Test metrics and predictions exported |
| C4 | Virchow2 CAMELYON17 train → PCam test (Macenko) | `experiments/virchow_colab/evals_cross_domain/cam17_trained_on_pcam_test/` | Test metrics and predictions exported |
| C5–C6 | Stain-method CNN benchmark on PCam subsets | Benchmark run logs / notebooks (to be cited from final extraction) | Pending consolidation in thesis tables |
| C7 | CNN baseline transfer (matched preprocessing arms) | `experiments/cnn_baseline_full/` when full pipeline run completes | Pending or partial relative to Virchow |
| C8 | Qualitative error review on deterministic test outputs | `reports/qualitative_error_analysis/virchow_c1_c4/` after notebook export | In progress with structured checklist |

Training runs that produced the heads and validation temperatures are stored under `experiments/virchow_colab/virchow_macenko_bench_run_01/` (PCam) and `experiments/virchow_colab/virchow_wilds_preprocessed_run_01/` (CAMELYON17); each holds `run_config.json`, `metrics_history.json`, `metrics_final_detailed.json`, and `temperature_fit.json` for audit of the validation-only selection boundary.

---

## 3. Virchow2 held-out test performance (C1–C4) and transfer summary

All four tests used the frozen Virchow2 encoder with trainable linear head, head dropout as recorded in `run_config.json`, and batch inference with head dropout disabled for the main reported probabilities. Temperature \(T\) was read from each training run’s `temperature_fit.json` and applied to logits on the relevant held-out test without refitting. **Table R.7** reports discrimination and reliability scalars after temperature scaling on the test split (`metrics_test_prob_after_temperature` in each folder’s `test_metrics_detailed.json`); Brier score and log loss are included because they summarize probability quality under domain shift alongside ROC-AUC and average precision. Accuracy is shown because thresholded behaviour at 0.5 is used for confusion-based summaries, but the primary discrimination endpoints remain ROC-AUC and PR-AUC as stated in the Methodology.

**Table R.7. Virchow2 test performance after temperature scaling (conditions C1–C4).**

| Condition | Train domain | Test domain | \(n_{\text{test}}\) | ROC-AUC | PR-AUC | Accuracy | Brier | Log loss | ECE (15 bins) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | PCam | PCam | 27704 | 0.9813 | 0.9820 | 0.9367 | 0.0486 | 0.1702 | 0.0200 |
| C2 | PCam | CAMELYON17 | 83181 | 0.9971 | 0.9977 | 0.9840 | 0.0134 | 0.0559 | 0.0100 |
| C3 | CAMELYON17 | CAMELYON17 | 83181 | 0.9966 | 0.9972 | 0.9798 | 0.0174 | 0.0712 | 0.0206 |
| C4 | CAMELYON17 | PCam | 27704 | 0.9502 | 0.9530 | 0.8237 | 0.1363 | 0.4716 | 0.1600 |

The transfer contrasts that match the primary endpoint wording are summarized in **Table R.8** using the same sign conventions as the Methodology for metrics where higher is better (absolute drop \(M_{\text{in}}-M_{\text{ext}}\)) and for Brier and log loss where lower is better (external minus in-domain change). For the PCam-trained model, in-domain ROC-AUC on PCam test (C1) is lower than external ROC-AUC on CAMELYON17 test (C2), so the conventional “drop” is negative on this scalar; the table states the arithmetic difference honestly and leaves interpretation to the Discussion. For the CAMELYON17-trained model, ROC-AUC and PR-AUC fall from C3 to C4 while Brier and log loss rise, which matches the expected pattern of harder external behaviour on PCam under this protocol.

**Table R.8. Transfer contrasts for Virchow2 (point estimates, same tests as Table R.7).**

| Training domain | Contrast | \(\Delta\) ROC-AUC | \(\Delta\) PR-AUC | \(\Delta\) Accuracy | \(\Delta\) Brier (ext−in) | \(\Delta\) Log loss (ext−in) | \(\Delta\) ECE (ext−in) |
|---|---|---:|---:|---:|---:|---:|---:|
| PCam | C1 vs C2 (in − ext) | −0.0158 | −0.0157 | −0.0473 | −0.0352 | −0.1143 | −0.0100 |
| CAMELYON17 | C3 vs C4 (in − ext) | 0.0464 | 0.0442 | 0.1561 | 0.1189 | 0.4004 | 0.1394 |

**Figure R.5** should collect reliability diagrams or calibration-style plots for at least one in-domain and one external surface per training direction (for example C1 versus C2 and C3 versus C4), using the binned accuracy–confidence pairs already stored beside ECE in `test_metrics_detailed.json` if exported, or recomputed from `test_predictions.npz` for cleaner visuals. **Figure R.6** can show paired forest-style intervals once bootstrap resampling on patch-level outcomes is completed under Chapter 7; until then the figure caption should read “point estimates only” if the graphic is omitted.

---

## 4. CNN baseline, extended statistical inference, and qualitative synthesis

The conventional CNN baseline under matched preprocessing arms (C7) should mirror the Virchow reporting layout: one block for in-domain and external tests in each training direction, with the same reliability metrics and the same temperature policy if applicable to that model family. Artefacts are expected under `experiments/cnn_baseline_full/<run_id>/evaluation/` when the full four-arm pipeline finishes; **Table R.9** is reserved for that block so that Virchow and CNN rows can be aligned on identical test surfaces without mixing training code paths.

Bootstrap percentile intervals for ROC-AUC, PR-AUC, Brier, log loss, and ECE on each test split, together with paired permutation tests where two models share identical patch sets, belong in **Table R.10** once \(B=2000\) replicates and the planned multiplicity adjustment are executed exactly as in Chapter 7. Until those computations are frozen, the table should be omitted rather than filled with provisional numbers. The same rule applies to any McNemar-style or paired AUC tests if they are added beyond the original plan; they must be named and referenced to the preregistered hierarchy of claims.

The structured qualitative review (C8) follows the bucket definitions and sampling targets in the qualitative protocol: false positives, false negatives, high-entropy errors, high-entropy correct calls, and optionally confident errors, with up to twenty cases per bucket per transfer direction where available. **Table R.11** will summarize bucket availability, sampled \(n\), and shortfall, and **Figure R.7** will show representative panels (raw and preprocessed side by side, with label, calibrated probability, confidence, and entropy text) for the patterns that dominate after checklist coding. Pattern prevalence tables (\(\hat{p}_{r,b}=k_{r,b}/n_b\) for checklist item \(r\) in bucket \(b\)) complete the qualitative evidence without substituting for the quantitative endpoints.

---

## 5. Synthesis

Taken together, the integrity table and the planned QC and benchmark tables close the loop between preprocessing choices and the sizes of the tensors actually consumed by the models. The Virchow2 block (Tables R.7–R.8) answers the prespecified transfer questions with held-out test material only, using a single calibrated probability scale per training run and a fixed 0.5 decision rule. The remaining rows of the thesis Results chapter should be populated as the CNN baseline, bootstrap and permutation layers, and qualitative exports reach the same archival standard as the Virchow artefacts cited here, so that the Discussion can refer exclusively to frozen tables and figures without reopening undocumented notebook state.
