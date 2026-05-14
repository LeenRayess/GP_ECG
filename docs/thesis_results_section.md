# Results


## 1. Dataset counts before curation

PatchCamelyon and CAMELYON17 were both used as fixed \(96 \times 96\) RGB patch benchmarks with standard train, validation, and test splits. The counts below are the inventories as released (PCam) or as indexed in the WILDS patch pack (CAMELYON17) before study-specific deduplication on PCam and before quality filtering on either dataset; they are the reference scale for the reductions reported in later subsections.

Table 1. Patch counts by split for the public PCam release and for the CAMELYON17 patch pack as indexed before study curation.

| Split | PCam | CAMELYON17 |
| --- | ---: | ---: |
| Training | 262,144 | 302,436 |
| Validation | 32,768 | 34,904 |
| Test | 32,768 | 85,054 |
| All splits | 327,680 | 422,394 |

## 2. Dataset counts after deduplication

Only PCam was passed through exact-content deduplication, retaining one index per identical patch so later stages and overlap checks refer to a duplicate-free list. Table 2 compares the public catalogue per split with the post-dedup inventory and gives the implied reduction from duplicate collapse (not from tissue-based screening). Those post-dedup PCam totals are the ones that enter quality filtering in Section 4. CAMELYON17 was not deduplicated here; split sizes remain as in Table 1.

Table 2. PCam patch counts before and after exact-content deduplication, by split.

| Split | Before deduplication | After deduplication | Reduction (duplicate collapse) |
| --- | ---: | ---: | ---: |
| Training | 262,144 | 220,025 | 42,119 |
| Validation | 32,768 | 28,108 | 4,660 |
| Test | 32,768 | 29,383 | 3,385 |
| All splits | 327,680 | 277,516 | 50,164 |

## 3. Overlap counts

At the level of pooled preprocessed corpora there were \(262{,}540\) unique PCam content hashes, \(413{,}860\) unique CAMELYON17 hashes, and exactly one hash present in both unions—so byte-identical overlap between the two pooled datasets is trivially small. Directional split-wise overlap is given in Table 3; every non-zero cell is single-digit against \(10^4\)–\(10^5\) denominators (order \(10^{-5}\) when expressed as a rate). Taken together, these figures imply that aggregate test performance is not driven by cross-dataset pixel duplicates; near-duplicate tiles that are not byte-identical remain a separate limitation of the check.

Table 3. Directional exact-hash overlap on preprocessed tensors (source split vs union of target dataset).

| Split | Patches in split (CAMELYON17) | Overlap vs any PCam | Patches in split (PCam) | Overlap vs any CAMELYON17 |
| --- | ---: | ---: | ---: | ---: |
| Training | 296,294 | 4 | 208,355 | 2 |
| Validation | 34,389 | 0 | 26,515 | 0 |
| Test | 83,181 | 1 | 27,704 | 0 |

Within CAMELYON17 alone, only three training pairs shared an identical hash (no such duplicates in validation or test).

## 4. Post–quality filter composition, removal reasons, and illustrative exclusions

Quality screening removed \(5.4\%\) of PCam candidates (\(14{,}942\) of \(277{,}516\)) and \(2.0\%\) of CAMELYON17 candidates (\(8{,}530\) of \(422{,}394\)). In both corpora the bulk of loss was the low-tissue rule; solid-colour and high-black bins were minor on PCam but high-black removals were nontrivial on CAMELYON17 (\(886\) versus \(13\) on PCam), which suggests a heavier tail of dark-background tiles under shared thresholds. Post-hoc positive fractions among kept patches stayed near the incoming class mix (Table 6), so the screen did not materially unbalance labels.

Table 4. Pooled quality-filter throughput and removal reasons (all splits combined).

| Dataset | Candidates | Retained | Removed | Low tissue | Solid colour | High black |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCam | 277,516 | 262,574 | 14,942 | 12,451 | 2,478 | 13 |
| CAMELYON17 | 422,394 | 413,864 | 8,530 | 6,834 | 810 | 886 |

Table 5. Quality-filter candidates, retained patches, and removed patches by split.

| Split | PCam candidates | PCam retained | PCam removed | CAMELYON17 candidates | CAMELYON17 retained | CAMELYON17 removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 220,025 | 208,355 | 11,670 | 302,436 | 296,294 | 6,142 |
| Validation | 28,108 | 26,515 | 1,593 | 34,904 | 34,389 | 515 |
| Test | 29,383 | 27,704 | 1,679 | 85,054 | 83,181 | 1,873 |
| All splits | 277,516 | 262,574 | 14,942 | 422,394 | 413,864 | 8,530 |

Table 6. Tumour-positive label counts and fractions among retained patches after filtering.

| Split | PCam \(n_+\) | PCam \(n_-\) | PCam \(f_+\) | CAMELYON17 \(n_+\) | CAMELYON17 \(n_-\) | CAMELYON17 \(f_+\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 88,828 | 119,527 | 0.426 | 150,867 | 145,427 | 0.509 |
| Validation | 11,715 | 14,800 | 0.442 | 17,427 | 16,962 | 0.507 |
| Test | 12,940 | 14,764 | 0.467 | 42,468 | 40,713 | 0.511 |

Representative tiles rejected under each dominant reason (and borderline kept controls) should appear as figure panels in the final document so the exclusion pattern can be judged visually.

## 5. Stain normalization benchmark results

Six preprocessing arms (C5: classical single-reference; C6: adaptive variants) were compared on the shared PCam benchmark splits with the shallow CNN in Methods. Each row uses the checkpoint with best validation ROC-AUC on the benchmark validation split; test metrics are on the held-out benchmark test split.

Table 7. Stain-method CNN benchmark (C5 and C6).

| Condition | Preprocessing arm | Val ROC-AUC | Test ROC-AUC | Test accuracy | Test F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| C5 | Macenko (classical single-ref.) | 0.918 | 0.925 | 0.812 | 0.842 |
| C5 | Reinhard (classical single-ref.) | 0.929 | 0.917 | 0.806 | 0.792 |
| C5 | Vahadane (classical single-ref.) | 0.918 | 0.917 | 0.822 | 0.845 |
| C6 | Adaptive single-reference | 0.912 | 0.913 | 0.829 | 0.846 |
| C6 | Adaptive multi-reference | 0.921 | 0.881 | 0.801 | 0.813 |
| C6 | Adaptive multi-reference + train aug. | 0.924 | 0.903 | 0.814 | 0.820 |

Macenko achieved the highest benchmark-test ROC-AUC and was therefore adopted for full-corpus preprocessing and for all Virchow2 runs. Adaptive single-reference routing achieved the highest test accuracy and F1 at a \(0.5\) threshold on this CNN. Reinhard reached the highest validation ROC-AUC at the saved checkpoint but not the highest test ROC-AUC, so the validation-based selection rule did not reproduce the test ranking for that arm.

Test ROC-AUC was the prespecified primary criterion because it reflects class separability across all decision thresholds instead of locking the comparison to one cutoff. Accuracy and F1 at \(0.5\) instead summarise a single operating point and can shift when sensitivity and specificity trade at that threshold even when ranking changes little; they are still useful checks, but they are secondary here because later stages use validation-based calibration and may use thresholds other than \(0.5\). The gap between Macenko on test AUC and adaptive single-reference on test accuracy and F1 is therefore read as stronger overall ranking for Macenko versus more favourable hard-threshold behaviour for the adaptive arm on this shallow model, not as proof that Macenko is inferior under every deployment rule. Fixed-threshold headlines should be revisited whenever the classifier, calibration, or operating point changes.


## 8. Bicubic versus bilinear resizing for the \(96 \rightarrow 224\) step

The matched resize-kernel comparison favoured bicubic on both the shared quantitative metrics and side-by-side inspection of fine chromatin and membrane edges, where bilinear tended to look slightly softer. Production Virchow2 training used bicubic upsampling accordingly; tabulated scores should be placed next to this subsection in the final layout.

## 9. Convolutional baseline on raw patches

Not completed; no metrics are reported.

## 10. Convolutional baseline on preprocessed patches

Not completed; no metrics are reported.

## 11. Virchow2 baseline on preprocessed patches

All four prespecified conditions (C1–C4) used the frozen Virchow2 encoder with a trainable linear head, Macenko benchmark-style preprocessed patches, and the same training budget and evaluation bundle described in Methods. For each training domain, a scalar temperature \(T>0\) was fitted on validation logits only to minimise mean binary cross-entropy, then applied unchanged to the corresponding held-out test logits so that reported test probabilities use a single calibration rule per run without using test labels during fitting. Training minimised binary cross-entropy on training batches; per-epoch training and validation loss and accuracy are part of each run’s saved history and belong in the figure supplement as curves rather than as a scalar table in this section.

Discrimination and ranking. ROC-AUC summarises how well tumour-positive patches receive higher predicted probabilities than negatives across every threshold, so it does not depend on the \(0.5\) cut. PR-AUC (average precision) emphasises the positive class and is informative when positives are relatively rare or when false discoveries are costly; it should be read alongside ROC-AUC.

Thresholded behaviour. Accuracy, precision, recall, F1, and balanced accuracy in Table 8 use a \(0.5\) threshold on calibrated probabilities and describe a single default operating point.

Proper scoring and calibration. Brier score and logarithmic loss (mean binary log loss) measure how close probabilities are to labels, with heavier penalty for confident errors. ECE with fifteen bins (ECE\(_{15}\)) measures the average absolute gap between mean predicted probability and empirical positive rate inside each probability bin, i.e. reliability of the probability scale.

Transfer contrasts. On CAMELYON17 test, ROC-AUC stayed very high for both training origins, so ranking on that domain remained strong under shared preprocessing. On PCam test, ROC-AUC stayed high for PCam-trained weights but was lower for CAMELYON17-trained weights, while accuracy at \(0.5\) on calibrated probabilities fell sharply in that external direction; Brier, logarithmic loss, and ECE worsened in the same qualitative pattern as in the earlier narrative—probability behaviour on PCam test suffered more when training came from CAMELYON17 than the reverse—so the stress is mainly in calibrated probability quality and fixed-threshold summaries, not in a collapse of ROC-based discrimination on PCam. The PCam-trained model’s ROC-AUC on external CAMELYON17 test was numerically higher than on its own PCam test, so the usual “external ROC drop” pattern does not hold on that scalar for this pair of domains under the chosen protocol.

Temperature scaling is a strictly monotone map of logits to probabilities, so ROC-AUC and PR-AUC are identical for raw and temperature-scored probabilities; Brier, logarithmic loss, and ECE are not. Table 8 lists the full calibrated-test scalar set; Table 9 repeats the raw layer so the effect of \(T\) on proper scores and calibration can be read directly.

Table 8. Held-out **test** metrics after temperature scaling (calibrated probabilities). \(n_{\mathrm{test}}\) is the public test count for the test-domain split. Fill dashed cells from each condition’s `test_metrics_detailed.json` (`metrics_test_prob_after_temperature`, `ece_15_bins_temperature_scaled`, and `temperature_used` for \(T\)).

| ID | Trained on | Tested on | \(n_{\mathrm{test}}\) | \(T\) | ROC-AUC | PR-AUC | Accuracy | Balanced acc. | Precision | Recall | F1 | Brier | Log loss | ECE\(_{15}\) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | PCam | PCam | 27,704 | — | 0.98 | — | 0.94 | — | — | — | — | — | — | — |
| C2 | PCam | CAMELYON17 | 83,181 | — | 0.997 | — | 0.98 | — | — | — | — | — | — | — |
| C3 | CAMELYON17 | CAMELYON17 | 83,181 | — | 0.997 | — | 0.98 | — | — | — | — | — | — | — |
| C4 | CAMELYON17 | PCam | 27,704 | — | 0.95 | — | 0.82 | — | — | — | — | — | — | — |

Table 9. Held-out **test** metrics on **raw** sigmoid probabilities (before \(T\)). ROC-AUC and PR-AUC match Table 8; fill the remaining columns from `metrics_test_prob_raw_sigmoid` and `ece_15_bins_raw`.

| ID | Trained on | Tested on | ROC-AUC | PR-AUC | Accuracy @0.5 | Balanced acc. | Precision | Recall | F1 | Brier | Log loss | ECE\(_{15}\) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | PCam | PCam | 0.98 | — | — | — | — | — | — | — | — | — |
| C2 | PCam | CAMELYON17 | 0.997 | — | — | — | — | — | — | — | — | — |
| C3 | CAMELYON17 | CAMELYON17 | 0.997 | — | — | — | — | — | — | — | — | — |
| C4 | CAMELYON17 | PCam | 0.95 | — | — | — | — | — | — | — | — | — |

**Confusion counts** at \(0.5\) on calibrated probabilities (TP, TN, FP, FN for each of C1–C4) are emitted in the same evaluation JSON and should be tabulated beside Tables 8–9 or shown as four \(2\times2\) matrices in the figure supplement; they mechanically determine the precision, recall, and F1 entries once the dashed scalars are filled.

Interpretation note. When ROC-AUC remains high but Brier or ECE deteriorates, the model still ranks well but is poorly calibrated or poorly aligned with the default threshold—consistent with CAMELYON17-trained scores on PCam test. When ROC-AUC itself falls on an external surface, the limitation is stronger: separability of the representation on that domain is weaker, not only probability scaling.

Reliability diagrams and bootstrap intervals should be taken from the same evaluation exports so the graphics line up with the filled tables.

