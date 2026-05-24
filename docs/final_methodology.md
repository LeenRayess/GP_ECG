# METHODOLOGY

The methodology covers patch-level binary tumor classification on hematoxylin and eosin (H&E) patches from PatchCamelyon (PCam) and CAMELYON17 (WILDS). Patches were screened for within-dataset duplicates, cross-dataset overlap, and image quality; those that failed were removed, and the remainder was filtered and stain-normalized with single-reference Macenko after a short benchmark of common stain methods. A convolutional neural network baseline and a Virchow2 classifier were trained, the latter with a frozen pretrained encoder and an optimized linear head; hyperparameters, the saved checkpoint, and a calibration temperature were chosen from validation data only.

Each fitted model was scored on the held-out test split of its training dataset and then on the other dataset without further training, using one decision threshold and report layout. Discrimination, calibration, deterministic test-time probabilities for each patch, and a small structured qualitative review were combined with bootstrap intervals for the main metrics, paired permutation tests for planned model comparisons on the same test material, and a fixed ordering when several hypotheses were reported together.

The overall workflow is shown in Figure 1.

# 1. Study design

The study is retrospective and uses only the PCam and CAMELYON17 (WILDS) patch data. Models are trained on one dataset, scored on its held-out test split, and then scored on the other dataset with the same weights and the same reporting rules, and the same sequence is run with the training and test roles exchanged so that transfer is measured in both directions. Hyperparameters, the saved checkpoint, and the calibration temperature are chosen from the training dataset’s validation split alone; test labels from neither dataset are used for those choices, and labels from the external dataset are used only after training and calibration are finished.

The primary endpoint is discriminative performance on the dataset that was not used for training, together with the drop from performance on the held-out test split of the training dataset (absolute differences in the main metrics, and relative change from the in-domain score when that helps interpretation). On test data, discrimination was summarized primarily by ROC-AUC and PR-AUC; transfer degradation was reported on those ranking metrics and on threshold-based metrics (accuracy, recall, F1) at a fixed probability cut-off of 0.5 after calibration. The secondary endpoints are stability of probability calibration when the test domain changes and whether a small structured qualitative review matches the patterns seen in the tables.

# 2. Datasets

PatchCamelyon (PCam) and CAMELYON17 as packaged in WILDS were the only data sources. Both address the same clinical problem class: detection of lymph node metastasis in hematoxylin and eosin (H&E) sections from the sentinel or axillary lymph node assessment pathway in breast cancer, where the task is to find metastatic tumor in nodal tissue on H&E rather than to classify a primary breast tumor on its own. PCam was built by extracting fixed \(96 \times 96\) RGB patches from whole-slide images released with CAMELYON16. CAMELYON17 follows the CAMELYON17 challenge lineage; in the WILDS release, patches are \(96 \times 96\) H&E crops from lymph node slides collected at several Dutch hospitals, with hospital identifiers supplied as metadata so that inter-site differences in staining and acquisition are explicit while the label remains patch-level metastasis detection in the same nodal setting. For both benchmarks, the public definition of a positive patch is aligned: positivity requires tumor tissue in the central \(32 \times 32\) pixel region of each \(96 \times 96\) field, with the outer rim providing context only. PCam was used as a large benchmark for development and in-domain testing; CAMELYON17 was used to stress external performance under hospital-related domain shift. The transfer layout was fixed and mirrored as in Table 2.1. In-domain testing was defined as evaluation on the held-out test split of the dataset on which the model was trained; external-domain testing was defined as evaluation on the other dataset with weights frozen after training and calibration on the source domain.

| Direction | Development set (train + validation) | In-domain test set | External-domain test set |
|---|---|---|---|
| PCam to CAMELYON17 | PCam | PCam test split | CAMELYON17 |
| CAMELYON17 to PCam | CAMELYON17 | CAMELYON17 test split | PCam |

## 2.1 Descriptive statistics and sample sizes

Before any model was trained, split-level descriptive statistics were computed so that class balance and coarse appearance could be compared without fitted parameters. Each quantity was evaluated per patch and then summarized within train, validation, and test for PCam and for CAMELYON17. For every split, 4,000 patches were drawn at random with a fixed sampling design; the same sample size and the same descriptor set were used in both datasets. The descriptor list is given in Table 2.2.

| Category | Descriptors |
|---|---|
| Class composition | Number of positives, number of negatives, positive fraction |
| Channel and grayscale appearance | Mean and standard deviation of R, G, B, and grayscale intensity |
| Color and tissue proxies | Saturation-based tissue-content proxy, black-pixel ratio, blue-dominance ratio, pink-proxy ratio |
| Intensity bounds | Patch-level minimum and maximum intensity |
| Split-level aggregation | Mean, standard deviation, minimum, and maximum across sampled patches |



# 3. Preprocessing

The preprocessing stage was meant to eliminate leakage and reduce non-biological variation while keeping the patch definition aligned with the public benchmarks. Train, validation, and test were handled under fixed rules, PCam was deduplicated by pixel hashing, CAMELYON17 and PCam were checked for exact cross-dataset overlap, and hyperparameter search together with temperature scaling were restricted to source validation so that test material never steered those choices. After those integrity steps, patches that failed simple image-quality gates (solid-color, high-black, or low-tissue behaviour) were dropped so that stain normalization and training did not spend capacity on empty or artifact-dominated tiles. Stain normalization was then applied to align H&E appearance across scanners and sites while leaving diagnostically relevant structure in place, followed by linear rescaling of stored uint8 RGB to \([0,1]\). Patches stayed on disk at native \(96 \times 96\); for the Virchow2 backbone, bicubic upsampling to \(224 \times 224\) was applied only when tensors were formed for the model. 

## 3.1 Split policy and integrity

Train, validation, and test were kept disjoint within each source. Hyperparameter search and temperature fitting were limited to the source validation split only, and neither in-domain nor external test labels were used for those decisions. Duplicate patches within PCam, hash-based overlap checks between PCam and CAMELYON17, and eligibility for quality-based exclusion were handled under fixed rules before stain normalization. These control rules are summarized in Table 3.1.

| Control area | Rule applied |
|---|---|
| Split isolation | Train, validation, and test splits were treated as disjoint sets within each dataset. |
| Model selection boundary | Hyperparameter tuning and calibration were restricted to source-domain validation data. |
| Test-set protection | In-domain and external-domain test sets were not used for model selection. |
| Duplicate control (within PCam) | Exact duplicate patches were screened by SHA-256 hashing; one index per duplicate group was retained. |
| Cross-dataset overlap screening | Exact hash matching was run between CAMELYON17 and PCam to identify potential pixel-level overlap. |
| Quality-control exclusions | Low-quality patches were removed using predefined criteria (solid-color, high-black, and low-tissue filters). |


## 3.2 Quality filtering

Quality filtering was applied after the split and integrity steps to exclude patches with limited histologic usefulness before stain normalization and model training. The objective was to reduce non-informative inputs (for example, low-texture regions, very dark artifacts, and tissue-poor patches) that can bias optimization and inflate variance without adding diagnostic evidence. In practical terms, this step prioritizes tissue-bearing patches with interpretable morphology and consistent color structure. It does not attempt lesion-level semantic filtering; instead, it provides a conservative image-quality gate so that downstream learning focuses on biologically meaningful content.

For each patch \(I \in [0,1]^{H \times W \times 3}\), grayscale intensity was defined as:
\[
Y = 0.299R + 0.587G + 0.114B.
\]

Pixel-wise saturation was defined as:
\[
S = \frac{\max(R,G,B)-\min(R,G,B)}{\max(R,G,B)+\epsilon},
\]
where \(\epsilon\) is a small constant to avoid division by zero.

The final tissue proxy was:
\[
\text{TissueRatio} = \frac{1}{HW}\sum \mathbf{1}(S > 0.12).
\]

A patch was removed if its grayscale variability was below the solid-color threshold (\(\sigma_Y < 0.04\)), if at least half of its pixels were very dark (\(\frac{1}{HW}\sum \mathbf{1}(Y < 0.05) \geq 0.5\)), or if the tissue proxy indicated too little stained tissue (\(\text{TissueRatio} < 0.35\)). These thresholds were fixed before final runs and applied consistently across splits.

## 3.3 Stain normalization

Stain normalization was implemented to reduce color variability unrelated to tissue biology, particularly variability introduced by differences in staining conditions and acquisition environments. The goal was not to force identical visual appearance across all patches, but to improve color consistency while preserving morphology that is relevant for diagnosis.

A bench mark of the classical and adaptive stain normalization techniques was conducted in a controlled subset-based setting to compare stain-handling strategies under matched data, architecture, and optimization conditions. The subsets were created from PCam with class-balanced sampling targets before quality filtering, with train target size 40,000 patches, validation target size 8,000 patches, and test target size 16,000 patches. A shared quality filter was then applied, and the same filtered indices were reused across stain-method variants to ensure a fair method comparison.

The stain normalization techniques included in the benchmark were:
- Macenko (classical single-reference),
- Reinhard (classical single-reference),
- Vahadane (classical single-reference),
- adaptive single-reference routing,
- adaptive multi-reference routing,
- adaptive multi-reference with training-time augmentation.

**Single-reference** methods map every patch to one fixed reference look. The reference came from quality-filtered PCam train tiles with a stricter tissue rule (\(T_i \ge 0.50\)). Among candidates, the final reference was the tile with smallest weighted distance between its RGB means and tissue ratio and the target profile. That distance is calculated as:
\[
d_i=\sqrt{
w_R(\mu_{R,i}-\mu_R^*)^2+
w_G(\mu_{G,i}-\mu_G^*)^2+
w_B(\mu_{B,i}-\mu_B^*)^2+
w_T(T_i-T^*)^2
},
\]
where \(\mu_{R,i}, \mu_{G,i}, \mu_{B,i}\) and \(T_i\) describe the candidate and \((\mu_R^*,\mu_G^*,\mu_B^*,T^*)\) the target. Fixed weights were \(w_R = w_B = 1.5\) and \(w_G = w_T = 1.0\), emphasizing R and B over G and tissue fraction. Targets were set for a typical pink–blue H&E balance (R at the 62nd percentile and B at the 38th percentile of the candidate pool). The chosen `reference_train_index` was written to `stain_reference.json` and reused for all benchmark runs and both datasets.

**Multi-reference** methods cluster candidates in stain-feature space, pick one central reference per cluster, fit a normalizer for each reference, and assign each patch to its nearest cluster so it is normalized with the matching “reference pack.” **Adaptive routing** picks a path per patch using blue dominance,
\[
\text{BlueDom}(I)=\frac{1}{HW}\sum \mathbf{1}(B>R).
\]
Per split, \(\tau_{\text{blue}}\) was the 25th percentile of BlueDom on kept patches: tiles with \(\text{BlueDom}(I) < \tau_{\text{blue}}\) tried Reinhard first, others tried Macenko first; the other method was used if the first failed, and luminosity-only output if both failed. Extra post-normalization checks could also send failing tiles to luminosity-only. Adaptive runs additionally applied a “purple-tail” safeguard: per split, cutoffs from the 2nd percentile of mean red and of pink-ratio sent tiles below either cutoff to luminosity-only. The adaptive multi-reference + augmentation condition kept this routing and added train-only color and intensity jitter (no geometric warps): gain in \([0.9, 1.1]\), bias in \([-0.03, 0.03]\), saturation scale in \([0.9, 1.1]\), then clip to \([0,1]\); validation and test were not augmented.

Alignment was checked by how much the Euclidean distance between the patch mean RGB \(\mu(I)=[\mu_R,\mu_G,\mu_B]\) and the reference mean \(\mu_{\text{ref}}\) dropped after normalization, together with split-level preprocessing reports to catch unstable color artifacts.

All stain variants used the same baseline CNN and training: four conv blocks (32, 64, 128, 256 channels), global average pooling, a 128-unit dropout head, binary cross-entropy with logits, and Adam with shared hyperparameters. Scores were reported with the same metric set (ROC-AUC, accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, and TP/TN/FP/FN). Macenko was chosen for the main experiments because it was the most stable under the cross-domain preprocessing protocol, so final PCam and CAMELYON17 runs used Macenko in the benchmark-style single-reference configuration with that fixed reference.

## 3.4 Value normalization and resizing

After quality and stain processing, pixel values were normalized to:
\[
I_{norm} = \frac{I}{255}
\]
That convention was applied before model input and metric computation so all scores used the same intensity scale.

Patch storage remained at \(96 \times 96\). For architectures requiring \(224 \times 224\) input, interpolation was applied as:
\[
I_{224} = \text{Resize}_{\text{bicubic}}(I_{96}).
\]
Bicubic was chosen over bilinear after side-by-side inspection of representative preprocessed validation patches upsampled to \(224 \times 224\). Bicubic crops were clearly sharper at nuclear membranes and chromatin texture; bilinear crops looked slightly smoother. No separate classifier benchmark was run for the resize kernels—the difference was obvious on visual review, and informal timing checks showed no meaningful batch-time penalty for bicubic. Bicubic was therefore adopted for all Virchow2 training and evaluation. Stored patches remain \(96 \times 96\) crops; upsampling does not restore true high-resolution structure from the source scanner but reduces smoothing before the frozen encoder relative to bilinear interpolation.

# 4. Models and training

Two model families were used under a harmonized training and evaluation framework: a conventional CNN baseline and a Virchow2-based classifier. The CNN baseline is a compact convolutional network trained end-to-end from random initialization on \(96 \times 96\) patches—it serves as a conventional, dataset-specific reference without a pretrained pathology backbone. Virchow2 is a large pathology foundation model pretrained on broad whole-slide histopathology data; here its encoder was kept frozen and only a small linear head was trained on top of its patch embeddings (after bicubic upsampling to \(224 \times 224\)). This dual-model design supports both methodological benchmarking and transfer-focused analysis under matched preprocessing and split policies. The experiments run during this study are organized into labeled conditions (codes C1–C8 in Table 4.1). Each condition states where the model trains, which preprocessing and stain arm apply for that row, which architecture is used, and which splits are scored. Within a comparison family, the intent is that only one major factor differs across runs (typically the training domain, the stain policy, or the backbone). Conditions C1–C4 form the main Virchow2 block: in-domain and external tests in both transfer directions under the adopted Macenko study pipeline (quality filter, Macenko normalization, values in \([0,1]\)). Conditions C5–C6 run the stain-method benchmark on controlled PCam subsets with the shared baseline CNN—three classical single-reference arms under C5 and three adaptive arms under C6, all on the same filtered benchmark indices. Condition C7 is the shallow CNN transfer reference in four training arms (PCam or CAMELYON17 \(\times\) raw uint8 or study-pipeline tensors), each scored on in-domain and external test with the same mirrored layout as C1–C4. Condition C8 is the structured qualitative error review (fixed checklist, bucket sampling) on Virchow2 outputs for C1–C4 only. Stain ablations reuse identical benchmark patch indices across methods; domain-shift rows follow the mirrored train and test layout fixed in Chapter 2.

| Condition ID | Condition purpose | Stain policy | Model | Train dataset | Test dataset(s) |
|---|---|---|---|---|---|
| C1 | In-domain PCam baseline evaluation | Macenko benchmark-style | Virchow2 + linear head | PCam | PCam test |
| C2 | External transfer PCam to CAMELYON17 | Macenko benchmark-style | Virchow2 + linear head | PCam | CAMELYON17 |
| C3 | In-domain CAMELYON17 evaluation | Macenko benchmark-style | Virchow2 + linear head | CAMELYON17 | CAMELYON17 test |
| C4 | External transfer CAMELYON17 to PCam | Macenko benchmark-style | Virchow2 + linear head | CAMELYON17 | PCam test |
| C5 | Stain-method benchmark (classical single-reference) | Macenko / Reinhard / Vahadane | Baseline CNN | PCam benchmark subset | PCam benchmark test subset |
| C6 | Stain-method benchmark (adaptive variants) | Adaptive single-ref / adaptive multi-ref / adaptive multi-ref + aug | Baseline CNN | PCam benchmark subset | PCam benchmark test subset |
| C7 | Shallow CNN transfer reference (raw and study-pipeline) | Raw uint8 **or** study pipeline (QC + Macenko + \([0,1]\)) | Baseline CNN | PCam **or** CAMELYON17 (four arms) | In-domain + external test per arm |
| C8 | Structured qualitative error review (checklist) | Study pipeline (Macenko) | Virchow2 (C1–C4) | Per condition (C1–C4) | Sampled error buckets on full test splits |

## 4.1 CNN baseline

The CNN baseline served as the conventional reference architecture for both controlled stain-method benchmarking and non-foundation transfer comparison. Its role in the study was to provide a lower-complexity model family against which the behavior of Virchow2-based transfer could be interpreted under matched preprocessing and evaluation conditions. The architecture consisted of four convolutional stages with channel progression \(32 \rightarrow 64 \rightarrow 128 \rightarrow 256\), followed by global average pooling and a dense head with 128 hidden units, dropout regularization, and a final binary logit output. Nonlinear activation and downsampling were applied through the feature-extraction pipeline to progressively increase representational abstraction while controlling spatial resolution and parameter growth. All CNN parameters were trainable, and each run started from random initialization. Optimization used binary cross-entropy with logits and Adam under fixed settings within each comparison family. This configuration allowed stain-handling effects to be compared without confounding from architecture changes or inconsistent optimization rules.

## 4.3 Virchow2 classifier

The primary model family for cross-domain analysis was a Virchow2-based binary classifier. In this setup, each input patch was first mapped by the pretrained Virchow2 encoder into a high-level pathology representation, and this representation was then passed to a linear classification head that produced the tumor-versus-non-tumor logit. The trainable-parameter policy followed a frozen-feature protocol across both transfer directions. Encoder weights were kept fixed, and only the linear head was optimized during supervised training. This design made the adaptation stage computationally tractable, reduced instability risk during optimization, and limited overfitting pressure in cross-domain settings where distribution mismatch is expected. The Virchow2 head was trained with the same loss formulation, optimizer family, and validation logic used in the shared study framework, so that differences between model families could be interpreted as representation and transfer effects rather than differences in development procedure. Keeping the encoder frozen also separated preprocessing-domain effects from deeper representation reconfiguration.

## 4.4 Training and model selection

To ensure model-family comparisons remained interpretable, training was harmonized across CNN and Virchow2 conditions by fixing the development protocol within each experiment family. For the full-data transfer experiments, the conventional CNN baseline and the Virchow2 linear head were each trained for ten epochs so the number of passes over the training indices matched; the CNN still applies a full-network gradient step each iteration whereas Virchow2 updates only the head on frozen encoder features, so per-step compute and effective capacity are not identical even though the optimization horizon is aligned. Non-target settings (including the full preprocessing order from split integrity through resizing, the optimization framework, and the reporting template) were kept consistent so that observed differences could be attributed to the intended ablation factor. Model selection and hyperparameter control remained validation-bound at the source domain, with no target-domain information used during development. Reproducibility was supported through explicit seed control and run-level artifact logging of configurations and outputs. Checkpointing followed a validation-driven policy. For each run, model states were monitored during training and the checkpoint achieving the best source-domain validation score under a predefined rule was retained as the final reporting model. The selection rules differed by experiment family so that each matched how discrimination was optimized in that block: the shallow CNN on full-transfer arms and on stain-benchmark subsets used the epoch with highest validation ROC-AUC on raw sigmoid probabilities; Virchow2 used the epoch with highest validation accuracy on the source validation split; the stain-method benchmark used validation ROC-AUC for checkpointing and test ROC-AUC on the held-out benchmark split to choose Macenko for full-corpus runs. Held-out source-domain test evaluation and external-domain test evaluation were executed only after checkpoint selection had been finalized. This ordering prevented leakage of test information into model development and maintained consistency across in-domain and transfer comparisons.

# 5. Outcomes
Each model was evaluated in-domain on the source test split and externally on the other dataset without retraining or target-specific tuning. PCam-trained models were scored on PCam test and on CAMELYON17; CAMELYON17-trained models were scored on CAMELYON17 test and on PCam test, giving a bidirectional transfer view under the same protocol.

Transfer degradation compares each metric’s in-domain and external-domain value. For metrics where higher is better (for example ROC-AUC, PR-AUC, accuracy, and F1), \(M_{\text{in}}\) and \(M_{\text{ext}}\) are denoted as the in-domain and external-domain values, respectively, and the transfer degradation is computed as
\[
\Delta M_{\text{abs}} = M_{\text{in}} - M_{\text{ext}},
\]
\[
\Delta M_{\text{rel}} = \frac{M_{\text{in}} - M_{\text{ext}}}{M_{\text{in}}+\epsilon},
\]
where \(\epsilon\) is a small positive constant for numerical stability. For metrics where lower is better (for example log loss and Brier score), transfer degradation was summarized by the external minus in-domain change,
\[
\Delta L_{\text{abs}} = L_{\text{ext}} - L_{\text{in}}.
\]
The same sign convention links discrimination and reliability summaries.

Discrimination used ROC-AUC and PR-AUC; threshold-based reporting used accuracy, precision, recall (sensitivity), specificity, F1, confusion-matrix counts (TP, TN, FP, FN), and MCC for imbalance-aware correlation. Probability quality used Brier score, log loss, and expected calibration error (ECE).

## 5.2 Calibration

Binary decisions used a fixed probability cut-off of \(0.5\) on the reported (calibrated) positive probability \(\hat{p}\), with no per-domain retuning of the cut-off. The resulting class label is
\[
\hat{y}=\mathbf{1}(\hat{p}\ge 0.5),
\]
with \(\hat{y}\in\{0,1\}\) and \(\mathbf{1}(\cdot)\) the indicator function. The same rule was applied on source test and external test for every condition so transfer comparisons are not confounded by post hoc threshold choice.

Temperature scaling maps each pre-calibration logit \(z(x)\) through a single scalar \(T>0\) learned on source-domain validation only. Calibrated probabilities are
\[
\hat{p}_{T}(y=1\mid x)=\sigma\!\left(\frac{z(x)}{T}\right),
\]
with \(\sigma\) the logistic sigmoid; that \(T\) was then applied unchanged to source test and external test. External labels and logits were not used to fit or adjust \(T\), in line with the split rules in Section 3.1. Brier score, log loss, and expected calibration error (ECE) summarize how well those probabilities match observed outcomes under this mapping.

## 5.3 Deterministic test scores and error subsets

Main tables and figures use one deterministic forward pass per patch (head dropout off), followed by the sigmoid and the same temperature mapping when calibrated probabilities are reported. From the resulting \(\hat{p}(x)\), case-level summaries use confidence \(c(x)=\max(\hat{p}(x),1-\hat{p}(x))\) and predictive entropy
\[
H(x)=-\hat{p}(x)\log\hat{p}(x)-\left(1-\hat{p}(x)\right)\log\left(1-\hat{p}(x)\right).
\]
High \(H(x)\) with low \(c(x)\) is treated only as a descriptive flag for ambiguous single-pass predictions, not as a full uncertainty model.

Failure sets were fixed before inspection. With ground truth \(y\) and predicted class \(\hat{y}\) from Subsection 5.2, misclassified patches satisfy \(\hat{y}\neq y\). Two reliability-focused subsets on errors are
\[
\mathcal{E}_{\text{conf}}=\left\{x:\hat{y}(x)\neq y(x),\,c(x)\ge \tau_c\right\},
\qquad
\mathcal{E}_{\text{ent}}=\left\{x:\hat{y}(x)\neq y(x),\,H(x)\ge \tau_H\right\},
\]
with \(\tau_c=0.9\) and \(\tau_H\) the 90th percentile of \(H(x)\) within each evaluation split, so confident wrong calls can be separated from wrong calls under flatter probabilities. The same cut-offs, calibration rule, and definitions were used in both transfer directions (PCam to CAMELYON17 and CAMELYON17 to PCam).

# 6. Qualitative analysis

A structured visual review was done on the same deterministic test outputs summarized in Chapter 5, to relate recurring patch appearance and quality-related cues to the transfer and calibration results. It supplements the tables rather than replacing them. Buckets and sampling rules were fixed beforehand so reviewed material stays comparable across transfer directions and model conditions.

Buckets use the same \(y\), \(\hat{y}\), \(c(x)\), \(H(x)\), and \(\hat{p}(x)\) as in Subsection 5.3. With \(\tau_H\) the 90th percentile of \(H(x)\) within each evaluation split, the primary sets are
\[
\mathcal{B}_{\mathrm{FP}}=\{x:\hat{y}(x)=1,\;y(x)=0\},\quad
\mathcal{B}_{\mathrm{FN}}=\{x:\hat{y}(x)=0,\;y(x)=1\},
\]
\[
\mathcal{B}_{\mathrm{HE}}=\{x:\hat{y}(x)\neq y(x),\;H(x)\ge \tau_H\},\quad
\mathcal{B}_{\mathrm{HC}}=\{x:\hat{y}(x)=y(x),\;H(x)\ge \tau_H\},
\]
and high-confidence errors are tracked separately as
\[
\mathcal{B}_{\mathrm{CE}}=\{x:\hat{y}(x)\neq y(x),\;c(x)\ge 0.9\}.
\]

For each transfer direction the review targeted 40 cases in total—up to 10 each from false positives, false negatives, high-entropy mistakes, and high-entropy correct calls—with random draws within a bucket using a fixed seed; when the high-confidence error bucket was included, up to 10 additional cases were added (50 cap per direction). Buckets smaller than the target contributed all available patches and the shortfall was noted in reporting. Tabulated qualitative prevalences in the results chapter refer to specific buckets only (for example false negatives and high-entropy errors at \(k/10\) per condition); they must not be read as if every exported case or every bucket contributed to those tables.

Each case was stored in a single evidence package: raw patch, preprocessed patch, label, prediction, \(\hat{p}\), \(c\), \(H\), bucket, transfer direction, and model condition, so visuals and numeric context were read together. Review used a fixed checklist with Present, Absent, or Unclear for tissue scarcity, artifact burden, borderline morphology, small-focus lesion pattern, color or stain atypia, and patch-context limits, plus a short free-text line on the main suspected driver of error or uncertainty. The same rubric and the same gallery-to-spreadsheet row order were used in every bucket and direction to limit reviewer drift.

Within each bucket \(b\) and checklist pattern \(r\), prevalence was summarized by \(\hat{p}_{r,b}=k_{r,b}/n_b\), where \(k_{r,b}\) counts reviews marked Present for \(r\) and \(n_b\) is the number of cases reviewed in \(b\). Those counts were read next to small representative panels when figures were prepared. Qualitative counts support interpretation of failure modes and domain-sensitive errors alongside Chapter 5; they are not treated as free-standing proof of causality, and claims stay tied to what was actually seen in the sampled set.

# 7. Statistical analysis

Case-level bootstrap with \(B=2000\) replicates was used to attach 95% percentile intervals to the point metrics in Chapter 5, and paired permutation tests were used when two methods were compared on the same test patches. Primary versus secondary claims and multiplicity rules were fixed before the results were read, to keep confirmatory statements from drifting with post hoc choices.

Primary inference targeted the main Virchow2 pipeline: within each transfer direction, the drop from in-domain to external performance, and across directions, asymmetry in those drops. Secondary material (for example model-family contrasts or qualitative prevalence summaries) was read as supportive unless it still cleared multiplicity control. On a test split of size \(n\), each scalar metric \(M\) was summarized by a point estimate on the observed cases.

\[
\hat{M}=M(\{y_i,\hat{p}_i\}_{i=1}^{n}).
\]

Bootstrap replicate \(b\) drew a case-level resample with replacement and recomputed the metric on that replicate.

\[
\hat{M}^{(b)}=M(\{y_i^{*(b)},\hat{p}_i^{*(b)}\}_{i=1}^{n}).
\]

The 95% interval was the 2.5th and 97.5th percentiles of the bootstrap distribution \(\{\hat{M}^{(1)},\ldots,\hat{M}^{(B)}\}\).

\[
\mathrm{CI}_{95\%}(M)=\left[Q_{0.025}(\hat{M}^{(1:B)}),\;Q_{0.975}(\hat{M}^{(1:B)})\right].
\]

The same bootstrap machinery was applied to discrimination, threshold-based, and reliability metrics. Transfer drops followed Section 5.1, using bootstrap point estimates \(\hat{M}_{\text{in}}\) and \(\hat{M}_{\text{ext}}\) within each replicate.

\[
\Delta M_{\text{abs}}=\hat{M}_{\text{in}}-\hat{M}_{\text{ext}}.
\]

\[
\Delta M_{\text{rel}}=\frac{\hat{M}_{\text{in}}-\hat{M}_{\text{ext}}}{\hat{M}_{\text{in}}+\epsilon}.
\]

Intervals for those drop quantities came from the paired bootstrap draws of in-domain and external estimates within each replicate. Head-to-head model comparisons on a shared test set used paired permutation tests on case-level score differences \(d_i\) (for example squared-error pieces for Brier or log-loss pieces for NLL). The tested null was

\[
H_0:\mathbb{E}[d_i]=0.
\]

Permutation \(p\)-values used 10,000 random sign flips of \(\{d_i\}\) so cases stayed matched. Where a metric did not admit a stable per-case split, reporting relied on bootstrap interval overlap and effect size, and that limitation was stated in the results. Confirmatory \(p\)-values were limited to the primary claim family and adjusted across those tests by Benjamini–Hochberg at \(q=0.05\). Other analyses kept unadjusted intervals and \(p\)-values and were labeled exploratory. Tables emphasized effect size and interval width, with adjusted or unadjusted \(p\)-values by tier and a short robustness interpretation rather than \(p\)-values alone. For qualitative prevalence summaries (Section 6), sample prevalence was written

\[
\hat{p}=\frac{k}{n}
\]

with intervals when \(n\) allowed it; otherwise raw counts were emphasized.

# 8. Reproducibility and limitations

This section states what must be held fixed so that the numerical results, comparisons, and inferential conclusions of this study can be reproduced or independently verified. Only factors that directly determine reported outcomes are retained.

This section records methodological limitations that bound how strongly conclusions may be drawn, and the controls applied to reduce associated bias or misinterpretation.

## 8.1 Reproducibility

Reported metrics depend on the exact curated data used after duplicate handling, cross-dataset overlap screening, quality filtering, stain normalization, value scaling, and resizing policy defined in Section 3. Split boundaries (train, validation, test) and the rule that development uses source-domain validation only must be preserved. Changing any of these steps changes the empirical results.

Reported test performance corresponds to the model weights obtained under the architecture and training protocol in Section 4, using the hyperparameters and optimizer settings recorded for each condition. The checkpoint used for final evaluation is the one selected by the source-domain validation rule documented in Section 4.4 (validation ROC-AUC for the shallow CNN and stain benchmark; validation accuracy for Virchow2). Reported numbers therefore depend on this selection rule; a different rule would yield different reported test metrics even with the same raw training history.

All discrimination and reliability metrics are computed under the fixed threshold, calibration fitting boundary, and deterministic test-time evaluation described in Section 5. Reproduction requires applying the same logits or probabilities, the same temperature scaling parameter fitted on source validation only, and the same metric definitions, with inference run as a single forward pass per patch (no stochastic passes at test time). Altering threshold choice, refitting calibration on test data, or changing the evaluation code path would change the reported scores.

## 8.2 Outputs and inference

Bootstrap intervals, permutation tests, and multiplicity handling follow Section 7 with fixed replicate counts and seeds as stated there. Qualitative case sampling follows Section 6 with the predefined buckets and sampling seed. These choices affect interval widths, p-values, and which examples appear in figures; they must match the documented plan when reproducing supplementary analyses.

To verify results without ambiguity, each experimental condition retains a configuration record (data handling, model family, hyperparameters, and evaluation settings), the selected model weights used at evaluation time, and exported predictions or logits sufficient to recompute metrics and Section 7 analyses. Statistical post-processing can be rerun from those frozen predictions alone.

Repeated GPU training may exhibit small run-to-run variation due to nondeterministic kernels and floating-point order; bitwise-identical loss curves are not asserted. Reproducibility of the study claims is therefore anchored on identical data, protocol, selection rule, and evaluation pipeline, with frozen predictions used whenever exact recomputation of metrics and inference is required.

Optimization trajectories may vary slightly across hardware and software stacks even when hyperparameters are fixed; conclusions were therefore tied to the documented evaluation pipeline and to frozen predictions where recomputation is required (Sections 8.1 and 8.2). That framing limits over-interpretation of small numerical differences between repeated training runs while preserving strict reproducibility of metrics once weights and inputs are fixed.

## 8.3 Limitations

Both datasets provide patch-level binary labels rather than full-slide diagnoses or spatially refined lesion maps. Performance therefore reflects patch classification under the benchmark definition of positivity, not exhaustive metastasis detection on whole lymph-node sections. Mitigation consisted of using publicly standardized benchmarks with documented label semantics, harmonizing preprocessing across datasets so that comparisons reflect domain shift rather than inconsistent patch handling, and interpreting metrics as transfer behavior on the stated task rather than as validated clinical deployment performance.

Transfer between PCam and CAMELYON17 reflects joint variation in staining appearance, acquisition context, center composition, and sampling structure, not stain variability alone. Mitigation included explicit split and integrity handling within preprocessing (Section 3), with documented stain normalization in the same chapter, mirrored bidirectional transfer to expose asymmetry rather than a single convenient direction, and retaining a conventional CNN reference under matched settings so that conclusions are not attributed solely to the foundation-model pathway.

Virchow2 was used under a frozen-encoder protocol with a trainable linear head. That choice stabilizes optimization and isolates transfer effects from deep fine-tuning, but it also limits the extent to which the encoder can adapt to the target domain. Mitigation was to report results under this fixed protocol consistently across domains and to interpret transfer gaps as arising under light adaptation unless explicitly extended elsewhere.

Native patch tensors are \(96 \times 96\) before bicubic upsampling to \(224 \times 224\) for Virchow2. This does not recover scanner-native micron-per-pixel detail; conclusions about resize kernels apply to interpolation choice on fixed low-resolution crops, not to restoration of whole-slide resolution.

Binary reporting used a fixed probability threshold of 0.5 without clinical cost-weighting. Real deployments would tune sensitivity–specificity trade-offs to institutional priorities. Mitigation included reporting ROC-AUC and PR-AUC alongside threshold-based metrics, documenting the threshold rule in Section 5, and treating calibration and reliability analyses as complementary views on probability quality rather than as final clinical thresholds.

Test-time scores were obtained with one deterministic forward pass per patch, so spread across repeated stochastic draws of the head was not estimated and should not be read as a full uncertainty decomposition. Mitigation consisted of pairing those scores with calibration metrics and qualitative review (Sections 5 and 6) and avoiding language that treats entropy or confidence flags as clinical risk estimates on their own.

Several metrics and comparisons were examined. Mitigation followed Section 7: a predefined primary claim family, Benjamini–Hochberg control at \(q=0.05\) for primary p-values, exploratory labeling for secondary analyses, and emphasis on effect sizes and intervals rather than isolated significance statements.
