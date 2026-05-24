# 1.5 Aims and Objectives

Copy-paste source for the thesis Introduction (§1.5). Section numbering assumes 1.4 Problem Statement immediately precedes this block. The literature review (Chapter 2) is also in `docs/literature_review.md`.

---

## 1.5.1 Aims

This project aims to build and evaluate a reproducible framework for patch-level lymph-node metastasis detection on H&E images, using PatchCamelyon (PCam) and CAMELYON17 (WILDS) under official train, validation, and test splits. It will compare a pathology foundation model (Virchow2) with a conventional shallow CNN under the same cross-domain protocol, including bidirectional transfer: models trained on one benchmark will be tested on that benchmark’s held-out split and on the other benchmark without retraining, so that in-domain and external performance can be compared in both directions.

A central aim is to determine whether quality-aware preprocessing remains necessary for reliable performance when a modern foundation model is used, or whether large-scale pretraining largely absorbs stain- and centre-related variation between datasets. The work will also assess outcomes beyond in-dataset accuracy—discrimination under domain shift, probability calibration, and structured review of typical failure modes—so that conclusions inform validation practice for cross-centre patch classifiers rather than resting on benchmark scores alone.

---

## 1.5.2 Objectives

To achieve these aims, the project will:

- Define a clinically relevant patch-level task on H&E lymph-node tissue (metastasis present versus absent in the central region) and adopt PCam and CAMELYON17 (WILDS) with their official splits, documenting label meaning, patch size, and in-domain versus external-domain evaluation.

- Establish a unified, traceable preprocessing pipeline with fixed project standards, applicable across both datasets.

- Implement data-integrity controls including exact deduplication on PCam, quality screening on both corpora, cross-dataset overlap checks, and auditable manifests so that performance estimates are not inflated by duplicates or leakage.

- Characterize both datasets through exploratory and audit workflows (class balance, tissue and colour statistics, quality-filter impact, split-wise summaries) to support methodological transparency in the report.

- Compare stain-handling strategies under matched conditions, select a primary normalizer for full-corpus preprocessing, and document the rationale using prespecified benchmark criteria.

- Train and evaluate a conventional CNN baseline on PCam and CAMELYON17, including raw public patches and study-pipeline inputs, to quantify preprocessing effects and transfer behaviour before and alongside foundation-model evaluation.

- Develop Virchow2-based classifiers with a frozen encoder and trainable head, consistent input preparation, and reproducible experiment artefacts (weights, logs, and predictions).

- Conduct bidirectional cross-domain evaluation by training on PCam and testing on CAMELYON17, and training on CAMELYON17 and testing on PCam, reporting in-domain and external-domain performance and transfer degradation in both directions.

- Assess predictive reliability using calibration-focused metrics and post-hoc temperature scaling fitted on validation only, separating ranking performance (ROC-AUC and PR-AUC) from probability quality.

- Stratify test errors using predictive confidence and entropy on deterministic outputs, alongside standard false positives and false negatives.

- Perform structured qualitative error analysis using predefined buckets, a human review checklist, and exported review packages for case-based interpretation.

- Document reproducibility and experimental governance through versioned scripts, run manifests, naming conventions for experimental conditions, and explicit separation of corpus curation from per-patch preprocessing so that the pipeline can be replicated or extended.

In sum, this study will determine whether quality-aware preprocessing together with a pathology foundation model yields more reliable cross-corpus metastasis detection than a conventional CNN under staining and scanner differences, and whether calibrated probabilities remain trustworthy when the test domain changes.


---

# 2. Literature Review

(Same text as `docs/literature_review.md`; copy from that file for Word.)

