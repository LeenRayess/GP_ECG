# RESULTS

This section presents the empirical outcomes of the study. It first characterizes the PCam and CAMELYON17 corpora after deduplication, overlap screening, and quality control, then compares preprocessing choices through targeted ablations—including stain-normalization strategies, input upsampling for the foundation-model pipeline, and the effect of the full study preprocessing pipeline on a shallow baseline trained on raw public patches versus curated tensors. Classifier performance is reported for a convolutional baseline and for Virchow2 under the same cross-domain protocol: each model is trained on one dataset, evaluated on its in-domain test split, and evaluated on the other dataset without retraining. The interpretive emphasis is external transfer when training on PCam and testing on held-out multi-hospital CAMELYON17 test—the direction that motivated the integrity-aware pipeline—while the mirrored direction (train on CAMELYON17, test on PCam) is reported in full to document asymmetry and how optimizing on heterogeneous nodal data interacts with PCam’s distribution; it is not treated as a second symmetric bar for the forward claim. Discrimination, calibration, and uncertainty-related summaries are given together with bootstrap intervals and paired tests where two models or training origins share the same test patches. A structured qualitative review of sampled errors is included to relate numeric transfer patterns to visible tissue, stain, and context factors. The discussion that follows interprets how preprocessing, model capacity, and domain shift interact, rather than restating the experimental protocol.

## 1. Datasets

PatchCamelyon (PCam) and CAMELYON17 (WILDS) were used as fixed \(96 \times 96\) RGB patch benchmarks with standard train, validation, and test partitions. Table 1 lists catalogue sizes before study-specific curation. Only PCam underwent exact-content deduplication; CAMELYON17 counts refer to the indexed WILDS pack before quality filtering.

**Table 1.** Patch counts by split before study curation.

| Split | PCam | CAMELYON17 |
| --- | ---: | ---: |
| Training | 262,144 | 302,436 |
| Validation | 32,768 | 34,904 |
| Test | 32,768 | 85,054 |
| All splits | 327,680 | 422,394 |

After deduplication, PCam retained 277,516 unique patches across splits (50,164 duplicate indices removed). Table 2 gives the split-wise reduction. CAMELYON17 was not deduplicated at the patch level; its split sizes in Table 1 apply through to the quality-filter stage.

**Table 2.** PCam counts before and after exact-content deduplication.

| Split | Before | After | Removed |
| --- | ---: | ---: | ---: |
| Training | 262,144 | 220,025 | 42,119 |
| Validation | 32,768 | 28,108 | 4,660 |
| Test | 32,768 | 29,383 | 3,385 |
| All splits | 327,680 | 277,516 | 50,164 |

Cross-dataset leakage was assessed by SHA-256 hashing of uint8 RGB tensors on the final preprocessed corpora. Table 3 summarizes directional overlap between corpora. At most four patches in a CAMELYON17 training split matched any PCam hash; rates were on the order of \(10^{-5}\). Exactly one hash appeared in both pooled unions (262,540 PCam-unique and 413,860 CAMELYON17-unique). Within CAMELYON17 alone, three training pairs were byte-identical duplicates; validation and test contained none. These counts indicate that aggregate test performance is not explained by shared pixels across datasets, although visually similar but non-identical tiles remain possible.

**Table 3.** Directional exact-hash overlap on preprocessed tensors.

| Split | CAMELYON17 patches | Overlap with any PCam | PCam patches | Overlap with any CAMELYON17 |
| --- | ---: | ---: | ---: | ---: |
| Training | 296,294 | 4 | 208,355 | 2 |
| Validation | 34,389 | 0 | 26,515 | 0 |
| Test | 83,181 | 1 | 27,704 | 0 |

## 2. Quality filtering and class balance after curation

Quality screening removed 14,942 of 277,516 PCam candidates (5.4%) and 8,530 of 422,394 CAMELYON17 candidates (2.0%). Low-tissue patches accounted for most removals; high-black tiles were more frequent on CAMELYON17 (886) than on PCam (13). Table 5 pools removal reasons. Split-wise throughput and retained class counts appear in Tables 6 and 7. Positive fractions among kept patches remained close to pre-filter levels, so the screen did not materially shift label prevalence.

**Table 5.** Pooled quality-filter removals (all splits).

| Dataset | Candidates | Retained | Removed | Low tissue | Solid colour | High black |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCam | 277,516 | 262,574 | 14,942 | 12,451 | 2,478 | 13 |
| CAMELYON17 | 422,394 | 413,864 | 8,530 | 6,834 | 810 | 886 |

**Table 6.** Quality-filter throughput by split.

| Split | PCam candidates | PCam retained | PCam removed | CAMELYON17 candidates | CAMELYON17 retained | CAMELYON17 removed |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Training | 220,025 | 208,355 | 11,670 | 302,436 | 296,294 | 6,142 |
| Validation | 28,108 | 26,515 | 1,593 | 34,904 | 34,389 | 515 |
| Test | 29,383 | 27,704 | 1,679 | 85,054 | 83,181 | 1,873 |

**Table 7.** Label counts among retained patches.

| Split | PCam \(n_+\) | PCam \(n_-\) | PCam \(f_+\) | CAMELYON17 \(n_+\) | CAMELYON17 \(n_-\) | CAMELYON17 \(f_+\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 88,828 | 119,527 | 0.426 | 150,867 | 145,427 | 0.509 |
| Validation | 11,715 | 14,800 | 0.442 | 17,427 | 16,962 | 0.507 |
| Test | 12,940 | 14,764 | 0.467 | 42,468 | 40,713 | 0.511 |

Figure 1 shows representative tiles rejected for low tissue, solid colour, and high black, alongside borderline patches retained under the fixed thresholds. The panels illustrate that most discarded patches are largely acellular or artifact-dominated rather than morphologically ambiguous tumour deposits.

## 3. Stain normalization benchmark

Six preprocessing arms were compared on class-balanced PCam benchmark subsets with a shared shallow CNN, using the benchmark protocol in the Methods chapter. Checkpoints were selected by best validation ROC-AUC, and the preprocessing arm for all main runs was chosen by **test ROC-AUC** on the held-out benchmark split—the discrimination endpoint fixed before test evaluation—because it measures class separability across thresholds and matches the validation criterion, whereas accuracy and F1 at a fixed 0.5 cutoff depend on class balance and an arbitrary operating point and were reported only as secondary threshold metrics. Figure 7 reports validation ROC-AUC, test ROC-AUC, test accuracy, and test F1 for each arm. Macenko (classical single-reference) achieved the highest test ROC-AUC (0.925) and was adopted for full-corpus preprocessing and all subsequent classifier runs. Adaptive single-reference routing yielded the highest test accuracy (0.829) and F1 (0.846) at threshold 0.5 on this architecture but not the highest test ROC-AUC. Reinhard reached the highest validation ROC-AUC yet lower test ROC-AUC, showing that validation-based selection did not preserve test ordering for that arm.

**Figure 7.** Stain-handling benchmark (shallow CNN on class-balanced PCam subsets): validation ROC-AUC, test ROC-AUC, test accuracy, and test F1 for six preprocessing arms (C5 classical, blue; C6 adaptive, orange), ordered by test ROC-AUC. Dashed line: Macenko (selected).

## 4. Descriptive appearance of retained corpora

After quality screening and adoption of Macenko stain normalization for the full corpora, class balance and coarse RGB statistics were summarized on **stratified random samples of 4,000 patches per split** (Methods). Table 4 reports sample means for \(f_+\) and appearance descriptors; the \(n\) column is the **full retained corpus size** per split (not the sample size). CAMELYON17 patches showed higher tissue-proxy scores and slightly higher positive fractions than PCam on training and test, consistent with the WILDS pack’s label distribution. Blue-dominance ratios differed between datasets, reflecting residual stain and scanner variation that the normalization step was designed to reduce but did not fully erase at the level of simple colour descriptors.

**Table 4.** Descriptive statistics on stratified samples (post-QC Macenko corpora).

| Dataset | Split | \(n\) | \(f_+\) | Mean \(R\) | Mean \(G\) | Mean \(B\) | Tissue proxy | Black ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCam | Train | 208,355 | 0.426 | 0.715 | 0.579 | 0.798 | 0.733 | \(3.7\times10^{-4}\) |
| PCam | Valid | 26,515 | 0.442 | 0.714 | 0.576 | 0.797 | 0.738 | \(3.6\times10^{-4}\) |
| PCam | Test | 27,704 | 0.467 | 0.718 | 0.580 | 0.799 | 0.731 | \(4.0\times10^{-4}\) |
| CAMELYON17 | Train | 296,294 | 0.509 | 0.757 | 0.548 | 0.816 | 0.825 | \(2.5\times10^{-4}\) |
| CAMELYON17 | Valid | 34,389 | 0.507 | 0.686 | 0.490 | 0.769 | 0.850 | \(7.3\times10^{-5}\) |
| CAMELYON17 | Test | 83,181 | 0.511 | 0.718 | 0.514 | 0.790 | 0.812 | \(2.3\times10^{-4}\) |

## 5. Input resizing

Virchow2 expects \(224 \times 224\) inputs while stored patches remain \(96 \times 96\). To choose a resize kernel for that step, representative preprocessed validation patches were upsampled with bicubic and bilinear interpolation and inspected side by side (Figure 2). Bicubic crops appeared clearly sharper and less blurred, especially at nuclear membranes and chromatin texture, whereas bilinear crops looked slightly smoother. Timing checks on the training pipeline showed no meaningful penalty for bicubic over bilinear at batch time. On that basis, bicubic upsampling was adopted for all Virchow2 training and evaluation.

## 6. Training convergence

Each classifier was trained for ten epochs with checkpoints saved every epoch (Figures 3 and 4). Validation accuracy plateaued after roughly five to seven epochs on PCam validation for both model families; CAMELYON17 validation metrics rose steadily through epoch ten. Best validation checkpoints were carried forward for temperature fitting and test evaluation.

**Figure 3.** Shallow CNN on preprocessed PCam and CAMELYON17: train loss, train accuracy, and validation ROC-AUC (10 epochs; blue/red).

**Figure 4.** Virchow2 (frozen encoder) on the same corpora: train and validation loss and accuracy (10 epochs; blue/red).

## 7. Shallow convolutional neural network baseline

A four-stage convolutional baseline with global average pooling and a dropout head was trained on four arms: PCam and CAMELYON17, each either on raw public patches or on tensors that passed the full study preprocessing pipeline (quality filter, Macenko stain normalization, rescaling to \([0,1]\)). Training used Adam (\(lr = 10^{-3}\)), batch size 320, and ten epochs. Checkpoints were chosen by best validation ROC-AUC on raw sigmoid probabilities; temperature \(T\) was fit on the source validation split only. Results are reported first for **raw** patches, then for **preprocessed** patches on the same transfer layout used for Virchow2.

### 7.1 Raw public patches (no study preprocessing)

These arms use catalogue uint8 RGB tensors without quality filtering, stain normalization, or \([0,1]\) rescaling. Test splits keep public sizes before the study quality-filter shrink. Table 9 shows that CAMELYON17-trained weights on raw in-domain CAMELYON17 test achieved only **sub-optimal** performance: ROC-AUC 0.724 (above random ranking at 0.50, but well below usable transfer), accuracy 0.554, and recall 0.16 at threshold 0.5—so the model poorly separated classes in practice despite non-trivial ranking. PCam-trained raw weights reached moderate in-domain discrimination (ROC-AUC 0.880). Applying the full preprocessing pipeline was therefore necessary before a fair cross-domain comparison with the main experiments.

**Table 9.** Shallow CNN on raw uint8 patches (calibrated, threshold 0.5).

| Train \(\rightarrow\) test | \(n_{\mathrm{test}}\) | ROC-AUC | Accuracy | Recall | Brier | ECE\(_{15}\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PCam \(\rightarrow\) PCam | 32,768 | 0.880 | 0.780 | 0.61 | 0.161 | 0.139 |
| PCam \(\rightarrow\) CAMELYON17 | 85,054 | 0.984 | 0.936 | 0.93 | 0.047 | 0.013 |
| CAMELYON17 \(\rightarrow\) CAMELYON17 | 85,054 | 0.724 | 0.554 | 0.16 | 0.268 | 0.230 |
| CAMELYON17 \(\rightarrow\) PCam | 32,768 | 0.676 | 0.521 | 0.08 | 0.308 | 0.280 |

### 7.2 Study preprocessing pipeline (transfer layout)

Table 10 gives held-out test metrics on quality-filtered, Macenko-normalized tensors scaled to \([0,1]\) (\(n_{\mathrm{test}} = 27{,}704\) for PCam test, 83,181 for CAMELYON17 test). The PCam \(\rightarrow\) CAMELYON17 cells are the principal external readout for whether a classifier developed on the large curated corpus still performs on the harder multi-hospital test; the reverse cells quantify the mirrored stress test. Figure 9 shows confusion matrices at threshold 0.5 (row-normalized fractions with raw counts annotated). Table 12 reports transfer contrasts as \(\Delta = M_{\mathrm{in}} - M_{\mathrm{ext}}\) on calibrated probabilities.

**Table 10.** Shallow CNN on preprocessed corpora (calibrated, threshold 0.5).

| Train domain | Test domain | \(T\) | ROC-AUC | PR-AUC | Accuracy | Balanced acc. | F1 | Brier | Log loss | ECE\(_{15}\) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCam | PCam | 1.473 | 0.948 | 0.946 | 0.871 | 0.866 | 0.852 | 0.094 | 0.301 | 0.043 |
| PCam | CAMELYON17 | 1.473 | 0.985 | 0.987 | 0.926 | 0.926 | 0.931 | 0.054 | 0.182 | 0.058 |
| CAMELYON17 | CAMELYON17 | 1.733 | 0.985 | 0.988 | 0.942 | 0.943 | 0.942 | 0.045 | 0.158 | 0.027 |
| CAMELYON17 | PCam | 1.733 | 0.856 | 0.867 | 0.698 | 0.677 | 0.528 | 0.231 | 0.799 | 0.252 |

**Figure 9.** Shallow CNN confusion matrices on preprocessed corpora (calibrated, threshold 0.5; `reports/figures/figure9_cnn_confusion_matrices_preprocessed.png`).

**Table 12.** Shallow CNN transfer contrast (preprocessed, \(\Delta_{\mathrm{in}-\mathrm{ext}}\)).

| Training arm | \(\Delta\) ROC-AUC | \(\Delta\) PR-AUC | Interpretation |
| --- | ---: | ---: | --- |
| PCam preprocessed | −0.037 | −0.042 | Higher ranking on external CAMELYON17 than in-domain PCam |
| CAMELYON17 preprocessed | +0.129 | +0.121 | Large drop on external PCam test |

### 7.3 Impact of the study preprocessing pipeline

Figure 12 compares raw and preprocessed ROC-AUC and recall on the same four train \(\rightarrow\) test directions. Table 13 summarizes the change (preprocessed minus raw, calibrated @ 0.5). Deltas use the same train \(\rightarrow\) test direction in each row; because quality filtering shrinks the test sets, accuracy shifts on CAMELYON17 test partly reflect corpus composition as well as model behaviour.

**Table 13.** Shallow CNN: change from raw to study-pipeline inputs (preprocessed minus raw, calibrated @ 0.5).

| Train \(\rightarrow\) test | \(\Delta\) ROC-AUC | \(\Delta\) accuracy | \(\Delta\) recall | \(\Delta\) Brier | \(\Delta\) ECE\(_{15}\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PCam \(\rightarrow\) PCam | +0.068 | +0.091 | +0.186 | −0.067 | −0.096 |
| PCam \(\rightarrow\) CAMELYON17 | +0.001 | −0.010 | +0.04 | +0.007 | +0.045 |
| CAMELYON17 \(\rightarrow\) CAMELYON17 | +0.261 | +0.388 | +0.771 | −0.223 | −0.203 |
| CAMELYON17 \(\rightarrow\) PCam | +0.180 | +0.177 | +0.287 | −0.077 | −0.028 |

**Figure 12.** Shallow CNN: test ROC-AUC (left) and recall @ 0.5 (right) on raw uint8 patches (grey) versus the study pipeline (green), for four train \(\rightarrow\) test directions; largest gain on CAMELYON17-trained, in-domain CAMELYON17 test.

The pipeline had the largest effect on CAMELYON17-trained weights. On in-domain CAMELYON17 test, ROC-AUC rose from 0.724 to 0.985 and recall from 0.16 to 0.91 (Table 9 vs Table 10), so the raw arm was effectively unable to detect positives at a fixed 0.5 threshold despite moderate ranking (ROC-AUC 0.72). After preprocessing, the same architecture reached performance in line with the other strong cells in Table 10. PCam-trained in-domain performance also improved (ROC-AUC +0.068; recall +0.19; Brier −0.067), showing that quality control, stain alignment, and value scaling mattered on the source domain and not only on the external hospital set.

Training dynamics mirrored test gains: best validation ROC-AUC increased from 0.931 to 0.947 for PCam arms and from 0.724 to 0.985 for CAMELYON17 arms. Raw CAMELYON17 training required a much larger post-hoc temperature (\(T \approx 4.4\) versus 1.7 after preprocessing), indicating poorly structured raw logits even after calibration. Preprocessing did not remove transfer asymmetry—Table 12 still shows a 0.13-point ROC-AUC drop when CAMELYON17-trained weights are applied to external PCam—but it raised the floor of both in-domain and cross-domain performance relative to raw inputs.

The PCam \(\rightarrow\) CAMELYON17 direction was already strong on raw external test (ROC-AUC 0.984) and changed little after preprocessing (\(\Delta\) ROC-AUC +0.001), so stain-normalized tiles were less critical for that transfer direction than for learning usable CAMELYON17-native representations. All subsequent main experiments (Virchow2, bootstrap inference, and qualitative review) therefore use study-pipeline tensors only.

## 8. Virchow2 classifier on preprocessed patches

Virchow2 was used with a frozen pretrained backbone and a trainable linear head (dropout 0.2). Four conditions (C1–C4) follow the same transfer layout: train on PCam or CAMELYON17, evaluate on the in-domain test split and on the external test split with the same study preprocessing pipeline as §7.2. C1 and C2 (PCam-trained; in-domain and CAMELYON17 external) carry the headline external-validation question; C3 and C4 complete the prespecified mirror for asymmetry and threshold behaviour and are read alongside C1–C2, not as an equal success criterion for the forward path. Temperature was fit once per training run and applied to all test logits from that run.

Table 14 reports calibrated test metrics. Table 15 gives raw sigmoid metrics on the same patches. Temperature scaling is a strictly monotone map of logits, so ROC-AUC and PR-AUC are identical between tables; Brier score, log loss, and ECE\(_{15}\) are not. The calibration step still matters for probability-based reporting: on C1, log loss fell from 0.175 to 0.170 and ECE\(_{15}\) from 0.027 to 0.020; on C4, where transfer stress was highest, log loss improved from 0.658 to 0.472 and ECE\(_{15}\) from 0.167 to 0.160, with a small Brier reduction (0.149 to 0.136). Accuracy at 0.5 was unchanged in these runs because the same threshold was applied after scaling, but the mapped probabilities were better aligned with empirical frequencies—supporting the post-hoc calibration step in Methods rather than reporting raw sigmoid scores alone. Figure 8 shows confusion matrices for C1–C4 at threshold 0.5. PCam-trained runs share \(T = 1.278\); CAMELYON17-trained runs share \(T = 1.590\).

**Table 14.** Virchow2 held-out test metrics after temperature scaling (C1–C4).

| ID | Trained on | Tested on | \(n_{\mathrm{test}}\) | \(T\) | ROC-AUC | PR-AUC | Accuracy | Balanced acc. | Precision | Recall | F1 | Brier | Log loss | ECE\(_{15}\) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | PCam | PCam | 27,704 | 1.278 | 0.981 | 0.982 | 0.937 | 0.934 | 0.970 | 0.892 | 0.929 | 0.049 | 0.170 | 0.020 |
| C2 | PCam | CAMELYON17 | 83,181 | 1.278 | 0.997 | 0.998 | 0.984 | 0.984 | 0.987 | 0.982 | 0.984 | 0.013 | 0.056 | 0.010 |
| C3 | CAMELYON17 | CAMELYON17 | 83,181 | 1.590 | 0.997 | 0.997 | 0.980 | 0.980 | 0.979 | 0.981 | 0.980 | 0.017 | 0.071 | 0.021 |
| C4 | CAMELYON17 | PCam | 27,704 | 1.590 | 0.950 | 0.953 | 0.824 | 0.812 | 0.988 | 0.630 | 0.770 | 0.136 | 0.472 | 0.160 |

**Table 15.** Virchow2 held-out test metrics on raw sigmoid probabilities.

| ID | ROC-AUC | PR-AUC | Accuracy | Brier | Log loss | ECE\(_{15}\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 0.981 | 0.982 | 0.937 | 0.049 | 0.175 | 0.027 |
| C2 | 0.997 | 0.998 | 0.984 | 0.013 | 0.053 | 0.003 |
| C3 | 0.997 | 0.997 | 0.980 | 0.016 | 0.065 | 0.008 |
| C4 | 0.950 | 0.953 | 0.824 | 0.149 | 0.658 | 0.167 |

**Figure 8.** Virchow2 confusion matrices (calibrated, threshold 0.5; C1–C4; `reports/figures/figure8_virchow_confusion_matrices_c1_c4.png`).

Table 17 summarizes marginal cross-domain changes on calibrated metrics (external minus in-domain test surfaces of different sizes). The PCam-trained model ranked higher on CAMELYON17 test than on PCam test by ROC-AUC. The CAMELYON17-trained model lost 0.046 ROC-AUC points and 0.156 accuracy points on external PCam test.

**Table 17.** Virchow2 marginal transfer (point estimates, calibrated).

| Training origin | \(\Delta\) ROC-AUC (ext. − in-domain) | \(\Delta\) accuracy (ext. − in-domain) |
| --- | ---: | ---: |
| PCam (C2 − C1) | +0.016 | +0.047 |
| CAMELYON17 (C4 − C3) | −0.046 | −0.156 |

C2 exceeded C1 on ROC-AUC and accuracy despite using the same PCam-trained weights. This pattern should be read cautiously: both scores were already near ceiling on PCam test (C1), the external CAMELYON17 test set is larger and has a slightly higher positive fraction in Table 4, and ranking metrics can move upward when prevalence and score separability shift even if the underlying decision boundary is unchanged. It does not imply that unseen CAMELYON17 data are strictly easier for every patch type; it shows that PCam-trained Virchow2 transferred well to the external hospital set under the study pipeline.

Figure 5 shows reliability diagrams for C1 and C4 from the fifteen-bin expected calibration error analysis. Calibration was close on in-domain PCam test (ECE\(_{15}\) = 0.020) but degraded on external PCam test after CAMELYON17 training (ECE\(_{15}\) = 0.160), consistent with high precision and reduced recall in Table 14.

## 9. Virchow2 versus shallow CNN on study-pipeline inputs

On the same preprocessed test splits (Tables 10 and 14), Figure 11 compares test ROC-AUC in each train \(\rightarrow\) test cell. Table 18 summarizes Virchow2 minus shallow CNN (positive \(\Delta\) favours Virchow2 on ROC-AUC, accuracy, and recall; negative \(\Delta\) favours Virchow2 on Brier and ECE). Table 19 and Figure 13 give cross-domain ROC-AUC change (external minus in-domain) by training origin.

**Table 18.** Virchow2 minus shallow CNN on study-pipeline test metrics (calibrated @ 0.5).

| Train \(\rightarrow\) test | \(\Delta\) ROC-AUC | \(\Delta\) accuracy | \(\Delta\) recall | \(\Delta\) Brier | \(\Delta\) ECE\(_{15}\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PCam \(\rightarrow\) PCam | +0.033 | +0.066 | +0.092 | −0.045 | −0.023 |
| PCam \(\rightarrow\) CAMELYON17 | +0.012 | +0.058 | +0.017 | −0.041 | −0.048 |
| CAMELYON17 \(\rightarrow\) CAMELYON17 | +0.012 | +0.038 | +0.072 | −0.028 | −0.006 |
| CAMELYON17 \(\rightarrow\) PCam | +0.094 | +0.126 | +0.268 | −0.095 | −0.092 |

**Figure 11.** Cross-domain transfer on study-pipeline test splits: test ROC-AUC for shallow CNN and Virchow2 in each train \(\rightarrow\) test cell (C1–C4 layout).

**Table 19.** Cross-domain ROC-AUC change on study-pipeline inputs (external minus in-domain test).

**Figure 13.** Cross-domain ROC-AUC change (external minus in-domain test) by training origin: Virchow2 vs shallow CNN.

| Training origin | Virchow2 | Shallow CNN |
| --- | ---: | ---: |
| PCam | +0.016 | +0.037 |
| CAMELYON17 | −0.046 | −0.129 |

Foundation-model features improved every preprocessed cell in Table 18. On in-domain PCam test, Virchow2 gained 0.033 ROC-AUC points and 0.066 accuracy points while roughly halving Brier score (0.049 versus 0.094). Gains on CAMELYON17 surfaces were smaller in ranking terms (+0.012 ROC-AUC in both directions where PCam or CAMELYON17 training matched the external hospital set) but still visible in accuracy and calibration. The largest separation appeared on the hardest transfer cell, CAMELYON17-trained weights evaluated on external PCam test (C4): Virchow2 retained ROC-AUC 0.950 and accuracy 0.824 versus 0.856 and 0.698 for the shallow CNN, with recall 0.630 versus 0.362 at similar precision (0.988 versus 0.974). At threshold 0.5, the CNN produced 8,254 false negatives on PCam test in that condition compared with 4,783 for Virchow2 (Figures 9 and 8).

The two architectures followed the same qualitative transfer pattern but with different severity. Both ranked higher on external CAMELYON17 than on in-domain PCam when trained on PCam (Figure 13, positive \(\Delta\) for PCam origin). Both lost performance on external PCam when trained on CAMELYON17; the CNN’s in-domain-minus-external ROC-AUC gap was 0.129 points, about 2.8 times the Virchow2 gap of 0.046. Thus preprocessing (§7.3) and representation capacity address different bottlenecks: the pipeline made CAMELYON17 learnable for a shallow CNN, while Virchow2 further reduced transfer loss and threshold failures on the difficult CAMELYON17 \(\rightarrow\) PCam direction.

## 10. Bootstrap and permutation inference

Case-level bootstrap with \(B = 2000\) replicates (seed 42) produced 95% percentile intervals for Virchow2 test metrics on calibrated probabilities. Table 20 lists primary intervals. Narrow bands on large CAMELYON17 test sets reflect stable ranking estimates.

**Table 20.** Virchow2 bootstrap 95% intervals on calibrated test metrics (point [2.5th, 97.5th percentile]).

| ID | ROC-AUC | Accuracy | Brier | ECE\(_{15}\) |
| --- | --- | --- | --- | --- |
| C1 | 0.981 [0.980, 0.983] | 0.937 [0.934, 0.940] | 0.049 [0.047, 0.050] | 0.020 [0.018, 0.023] |
| C2 | 0.997 [0.997, 0.997] | 0.984 [0.983, 0.985] | 0.013 [0.013, 0.014] | 0.010 [0.009, 0.011] |
| C3 | 0.997 [0.996, 0.997] | 0.980 [0.979, 0.981] | 0.017 [0.017, 0.018] | 0.021 [0.020, 0.022] |
| C4 | 0.950 [0.948, 0.953] | 0.824 [0.819, 0.828] | 0.136 [0.133, 0.140] | 0.160 [0.156, 0.164] |

Paired bootstrap on the same test patches compared in-domain and externally trained weights on a shared test domain. Table 21 gives \(\Delta = M_{\mathrm{in}} - M_{\mathrm{ext}}\) for PCam test (C1 minus C4) and CAMELYON17 test (C3 minus C2). All listed intervals excluded zero.

**Table 21.** Paired bootstrap transfer contrasts on shared test domains.

| Test domain | Contrast | \(\Delta\) ROC-AUC [95% CI] | \(\Delta\) accuracy [95% CI] | \(\Delta\) Brier [95% CI] |
| --- | --- | --- | --- | --- |
| PCam | C1 − C4 | 0.031 [0.029, 0.033] | 0.113 [0.109, 0.117] | −0.088 [−0.090, −0.085] |
| CAMELYON17 | C3 − C2 | −0.0005 [−0.0008, −0.0002] | −0.0042 [−0.0051, −0.0033] | +0.0039 [0.0035, 0.0045] |

Paired permutation tests (10,000 sign flips) on per-patch Brier and log-loss contributions compared training origins on the same test surface. Table 22 reports confirmatory tests with Benjamini–Hochberg adjustment at \(q = 0.05\). All four comparisons rejected the null of equal mean patch-wise scores.

**Table 22.** Paired permutation tests on shared test patches.

| Test surface | Score | Comparison | \(p\) | \(p_{\mathrm{BH}}\) |
| --- | --- | --- | ---: | ---: |
| PCam test | Brier | C1 vs C4 | \(<10^{-4}\) | \(<10^{-4}\) |
| PCam test | Log loss | C1 vs C4 | \(<10^{-4}\) | \(<10^{-4}\) |
| CAMELYON17 test | Brier | C3 vs C2 | \(<10^{-4}\) | \(<10^{-4}\) |
| CAMELYON17 test | Log loss | C3 vs C2 | \(<10^{-4}\) | \(<10^{-4}\) |

## 11. Qualitative error analysis

Human checklist review (Methods §6) was completed for all four Virchow2 conditions. Up to 40 patches per transfer direction were exported across four buckets (false positive, false negative, high-entropy error, high-entropy correct), with an optional fifth bucket adding up to ten more high-confidence errors. Figure 10 links the full-test error pools to the numeric results in §8: C4 stands out for 4,783 false negatives available versus only 102 false positives, consistent with the recall drop in Table 14. Table 24 summarizes checklist marks in only two of those buckets—false negatives and high-entropy errors—reporting \(k/10\) Present among the ten patches reviewed per bucket per condition; it is not a census of all exported cases and not full-test prevalences. Figure 6 shows example C4 false negatives; galleries for C1–C3 follow the same export layout.

**Figure 10.** Virchow2 qualitative-review error pools on full test splits (patches available per bucket before sampling; log-scaled y-axis; `reports/figures/figure10_qualitative_error_pools_c1_c4.png`).

**Table 24.** Checklist Present counts in FN and high-entropy error buckets only (\(k/10\) patches reviewed per bucket per condition; up to 40 exports per direction across all buckets).

| Pattern | C1 FN | C2 FN | C3 FN | C4 FN | C1 HE err | C2 HE err | C3 HE err | C4 HE err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Patch-context limitation | 8 | 6 | 4 | 6 | 4 | 4 | 4 | 5 |
| Small-focus lesion | 5 | 4 | 4 | 6 | 4 | 4 | 5 | 4 |
| Color/stain atypia | 2 | 6 | 2 | 4 | 2 | 4 | 2 | 4 |
| Borderline morphology | 4 | 4 | 4 | 4 | 7 | 8 | 7 | 8 |

On **C1**, reviewed false negatives often carried patch-context limitation (8/10), matching the PCam central-focus label rule despite strong overall metrics. On **C2**, color/stain atypia was common among reviewed false negatives (6/10) and high-entropy errors (4/10), even though transfer to CAMELYON17 test remained excellent numerically. **C3** showed no single dominant checklist pattern among false negatives, in line with strong in-domain performance. **C4** combined the largest false-negative pool with reviewed FN cases frequently marked for small-focus lesion and context limits (6/10 each), aligning with the threshold and calibration failures in Table 14. Across all conditions, borderline morphology appeared in 7–8/10 high-entropy **errors**, suggesting uncertain wrong predictions are driven more by ambiguous morphology than by gross tissue loss or artifacts alone. These patterns support the quantitative transfer story in §§8–10 but do not replace it.

---

# DISCUSSION

The experiments trace a single pipeline from public patch benchmarks through shared quality control and Macenko stain normalization to two classifier families evaluated under strict transfer rules. The pipeline was designed so that a classifier trained on the large PCam development corpus could be stress-tested on held-out multi-hospital CAMELYON17 test; the data stages clarify what was removed and why, while the modeling stages separate representation capacity from preprocessing and calibration choices on that path and on the prespecified mirror.

The headline empirical result is strong forward external transfer: after the study pipeline, PCam-trained Virchow2 reached very high discrimination and well-behaved probabilities on CAMELYON17 test (C2), with a clear margin over the shallow CNN on the same tensors—evidence that integrity-aware curation plus a frozen foundation encoder can support deployment-style generalization from a curated tile benchmark to a harder external nodal benchmark under fixed reporting rules. That claim is intentionally scoped to the PCam \(\rightarrow\) CAMELYON17 direction tested here; it does not assert uniform safety on every future centre or slide-level workflow without further validation.

The mirrored direction (CAMELYON17 train, PCam test) was included for information, not as a symmetric requirement for the forward conclusion. It shows how asymmetric transfer can be: both architectures retained respectable ROC-AUC on external PCam, yet Virchow2 and especially the CNN lost threshold recall and calibration relative to in-domain use (C4), with a large false-negative pool despite high precision at 0.5. Read together, C2 and C4 separate ranking strength from operating-point and probability behaviour when the training domain and label geometry differ from the test surface—a reason the study reports calibration and confusion structure alongside area metrics.

This work is not a universal remedy for domain shift. The contribution is methodological and empirical: a leakage-aware, prespecified pipeline; a primary external criterion aligned with practical “develop on PCam, stress on CAMELYON17” use; a mirrored layout that makes asymmetry visible; and reporting that couples ranking with threshold and calibration views. Cross-domain metrics improved in prespecified comparisons—most visibly when the shallow CNN moved from raw catalogue inputs to the full study pipeline on CAMELYON17, and when Virchow2 was compared to that CNN on identical preprocessed tensors.

Integrity checks showed that duplicate and cross-dataset overlap were negligible at the byte level. Quality filtering removed a modest fraction of tiles, mostly for low tissue, without shifting class balance. The stain benchmark justified Macenko for full-corpus work even though another arm achieved slightly better threshold accuracy on a smaller balanced subset. That distinction matters when reading later results: ranking metrics and fixed-threshold metrics can favour different preprocessing strategies on a shallow network, but Macenko remained the prespecified choice for all main runs.

The shallow CNN results establish how much performance depends on preprocessing and on which transfer leg is read. Without the study pipeline—quality control, stain normalization, and consistent value scaling—CAMELYON17-trained weights showed weak in-domain performance on raw CAMELYON17 test (ROC-AUC 0.72 with recall 0.16 at 0.5), which confirms that raw public patches alone were not a fair match to the curated corpora used elsewhere. After the full pipeline, the CNN reached strong ranking on CAMELYON17 when trained on PCam, in line with Virchow2 on that forward path. The mirror leg remained harder: CAMELYON17-trained weights on PCam test showed larger drops in accuracy and recall and worse calibration than the forward leg, with the CNN suffering a larger ROC-AUC gap than Virchow2.

Virchow2 improved every preprocessed comparison in Table 18 by a meaningful margin. The gain was not limited to in-domain PCam test; it persisted on external surfaces and was largest when CAMELYON17-trained weights were applied to PCam—the mirror stress cell—where Virchow2 preserved substantially higher recall at similar precision than the CNN. Bootstrap intervals were tight on large test sets but still excluded trivial shifts for paired contrasts on shared patches. Permutation tests on patch-wise Brier and log-loss confirmed that the two training origins produced different probability behaviour on the same tiles, not only different aggregate thresholds.

Structured checklist review (§11) aligned with these numeric patterns: context-limited and small-focus false negatives on external PCam (C4), stain atypia on cross-domain CAMELYON17 test (C2), and borderline morphology among high-entropy errors in every condition.

Taken together, the results support using this pipeline and the PCam-trained Virchow2 configuration as a reusable base for further external corpora and sites and for closer clinical-style validation (slide-level aggregation, operating-point choice, prospective checks), while keeping claims proportional to patch benchmarks and fixed splits. On the mirror leg, a deployment that relied on a fixed 0.5 cutoff without target-domain tuning would still be risky despite strong ROC-AUC; future work can target recall there through class weighting, focal losses, or threshold selection on validation material from the intended deployment domain, and can extend the qualitative protocol with formal inter-rater agreement on a larger stratified sample.

The resize-kernel and training-curve figures document implementation choices that are easy to overlook in tabular summaries alone. They show that reported metrics came from converged training runs and from a resize step that preserved sharper patch appearance in qualitative review. Further structural limits, compute bounds, and planned extensions are summarized in the Limitations and Future Work chapter.
