# Interim Results

## 1. Scope of currently available results

At the interim stage, three result blocks are available and sufficiently mature for reporting: (i) quantitative outcomes of data curation and preprocessing, (ii) baseline CNN performance on validation and test sets, and (iii) a direct comparison between stain-normalization Trial 1 (single-reference configuration) and Trial 2 (multi-reference configuration) using matched summary-color metrics. Full Virchow2 comparative model results are not yet available because training could not be completed under current GPU constraints; therefore, the model-comparison part of this section is intentionally limited to the baseline CNN.

## 2. Data curation and preprocessing outcomes

### 2.1 Dataset size after deduplication-aware filtering

After candidate restriction and quality filtering, the total number of usable patches decreased from 277,516 to 262,574, corresponding to 14,942 removed patches. Removal reasons were dominated by low tissue occupancy, with much smaller contributions from solid-color and high-black artifacts. Specifically, 12,451 removals were due to low tissue, 2,478 to solid-color behavior, and 13 to high-black behavior. This distribution indicates that the filtering stage is primarily acting as intended on histologically uninformative regions rather than aggressively rejecting potentially useful tissue-containing patches.

The split-wise retention and the decomposition of removals by reason are visualized in **Figure 1**.

At split level, the retained sample sizes were:

| Split | Candidates before filtering | Kept after filtering | Removed |
|---|---:|---:|---:|
| Train | 220,025 | 208,355 | 11,670 |
| Validation | 28,108 | 26,515 | 1,593 |
| Test | 29,383 | 27,704 | 1,679 |

An important observation is that these retained counts were identical between Trial 1 and Trial 2. Therefore, any downstream differences between the two trials arise from stain-processing behavior rather than from changes in dataset composition.

### 2.2 Class-balance behavior after filtering

Post-filter class fractions remained stable and within expected ranges. The kept positive fractions were approximately 0.426 (train), 0.442 (validation), and 0.467 (test). This confirms that quality filtering and deduplication-aware candidate restriction did not introduce severe class-ratio distortion at split level.

### 2.3 Stain-processing behavior: Trial 1 vs Trial 2

Visual inspection of the training patches indicated two recurrent stain appearances: (i) a highly saturated profile with vivid fuchsia-pink and bright purple tones, and (ii) a softer profile with lighter pink regions and relatively darker purple content. Trial 1 used a single reference aligned with the first (high-vibrancy) profile. Trial 2 was designed to represent the second profile more explicitly by using multiple fixed references and cluster-based routing.

Numerically, Trial 2 used a four-cluster routing model in a standardized seven-dimensional feature space \([p_{\text{tissue}},\mu_R,\mu_G,\mu_B,p_{\text{blue-dominant}},p_{\text{pink}},\bar{S}]\). One cluster was designated as a merge cluster, and the remaining three clusters were assigned to three fixed reference patches by minimum-distance matching in the same scaled feature space. For merge-cluster patches, the final reference was selected by minimum L1 distance between patch mean RGB and reference mean RGB. This produced the reported routing map (merge cluster \(=0\), remaining clusters mapped as \(1\to2\), \(2\to1\), \(3\to0\)).

Trial 1 used a single blue-dominance threshold, while Trial 2 used split-specific thresholds. In Trial 2, split-wise blue-dominance thresholds were 0.0537 (train), 0.0604 (validation), and 0.0980 (test), indicating measurable inter-split stain-profile differences that were explicitly modeled.

For Trial 2, normalizer usage rates were also informative. On train data, approximately 13.0% of patches used Macenko as final choice, 24.8% used Reinhard, and 62.2% used luminosity-only output. Validation and test showed similar distributions (luminosity-only near 63.8%). In addition, percentile-based purple-tail replacement affected 7,858 train patches, 1,051 validation patches, and 1,070 test patches. These values show that the safety/fallback pathway is not a rare edge case in Trial 2; it is a material part of the effective preprocessing transformation.

The per-split composition of normalization pathways in Trial 2 is shown in **Figure 5**.
The selected Trial 1 and Trial 2 reference patches are shown in **Figure 7**.

## 3. Baseline CNN results (completed experiment)

The baseline CNN experiment completed end-to-end and produced final validation/test metrics, training logs, and confusion matrices. Using the best validation-selected checkpoint, results were:

| Metric | Validation | Test |
|---|---:|---:|
| ROC-AUC | 0.9341 | 0.8768 |
| Accuracy | 0.8611 | 0.8017 |
| N | 32,768 | 32,768 |

Confusion matrices (rows = true class, columns = predicted class) were:

Validation:
\[
\begin{bmatrix}
15173 & 1226 \\
3325 & 13044
\end{bmatrix}
\]

Test:
\[
\begin{bmatrix}
15224 & 1167 \\
5332 & 11045
\end{bmatrix}
\]

These baseline confusion matrices are presented visually in **Figure 2**.

From these test counts, precision for the positive class is \(\frac{11045}{11045+1167}\approx 0.904\), while recall is \(\frac{11045}{11045+5332}\approx 0.674\). Thus, the baseline shows strong positive predictive value but a substantial false-negative burden on test data, which is a clinically relevant limitation to address in later model iterations.

Training dynamics were also stable and interpretable. Training accuracy increased monotonically from 0.8370 (epoch 1) to 0.9717 (epoch 10), while validation accuracy peaked around epoch 4 (0.8610) and then fluctuated below that level for several epochs. This pattern is consistent with progressive overfitting after early-to-mid training, despite continued optimization of training loss.

The full train/validation loss and accuracy trajectories are shown in **Figure 3**.

## 4. Trial 1 vs Trial 2 preprocessing comparison using matched summary-color metrics

To compare preprocessing variants without rerunning training, both trials were evaluated on the same sampled train indices using identical summary-color definitions. The mean L1 distance between each patch mean RGB vector and fixed reference vectors was computed for both trials:

| Metric (same sampled patches) | Trial 1 | Trial 2 |
|---|---:|---:|
| Mean L1 to Trial 1 reference mean RGB | 0.3413 | 0.4178 |
| Mean L1 to Trial 2 reference mean RGB | 0.5302 | 0.5422 |

Color-spread statistics of patch mean RGB showed:

| Statistic | Trial 1 | Trial 2 |
|---|---:|---:|
| std(R) | 0.0925 | 0.1035 |
| std(G) | 0.1285 | 0.1324 |
| std(B) | 0.0658 | 0.1134 |

The paired same-index mean L1 distance between Trial 1 and Trial 2 patch mean RGB values was 0.2413, indicating a substantial transformation difference between the two preprocessing variants for the same underlying images.

Interpretationally, these numbers indicate that Trial 1 enforces tighter convergence toward a single color target, whereas Trial 2 preserves a broader color distribution (especially in the blue channel). This is expected from the design difference (single-reference harmonization versus multi-reference routing with substantial luminosity fallback). However, these statistics alone do not determine task-level superiority; they quantify color-space behavior, not classification utility.

The matched-sample quantitative comparison between Trial 1 and Trial 2 is visualized in **Figure 4**, and qualitative before/after examples from Trial 2 are shown in **Figure 6**.

## 5. Interim interpretation and limitations

The current evidence supports three robust interim claims. First, the data curation stage is effective and controlled: removals are concentrated in low-information patches, and split-level class balance remains acceptable. Second, the baseline CNN provides a strong initial benchmark with high validation AUC and reasonable test AUC, but with a recall deficit that motivates stronger representation learning. Third, Trial 2 (multi-reference stain normalization) is demonstrably different from Trial 1 (single-reference stain normalization) in measurable color-space behavior, and therefore represents a meaningful experimental factor rather than a trivial implementation variation.

The main limitation is that the effect of preprocessing choice on downstream model performance has not yet been measured with a completed Virchow2 training comparison under identical evaluation conditions. Consequently, conclusions about "better preprocessing" must remain conditional at interim stage: we can characterize transformation behavior confidently, but final efficacy must be established through model-level comparisons once compute resources permit.
