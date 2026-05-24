# 1.5 Aims and Objectives

Copy-paste source for the thesis Introduction (§1.5). Section numbering assumes 1.4 Problem Statement immediately precedes this block. The literature review (Chapter 2) is also in `docs/literature_review.md`.

---

## 1.5.1 Aims

This project aims to build and evaluate a reproducible patch-level framework for lymph-node metastasis on H&E images, using PatchCamelyon (PCam) and CAMELYON17 (WILDS) under their official splits. The main aim is strong external performance when training on PCam and testing on held-out CAMELYON17 test—multi-hospital nodal tissue that stresses stain and acquisition differences relative to PCam. Integrity checks, quality screening, stain normalization (after a prespecified benchmark), and validation-bound training and calibration are aligned with that forward target, while keeping both corpora on a common pipeline for fair comparison.

Virchow2 (frozen encoder, trainable head) is compared to a shallow CNN on identical tensors and reporting rules. A prespecified mirror—train on CAMELYON17, test on PCam—is run to describe transfer asymmetry (for example when ranking stays high but threshold or calibration behaviour shifts); it informs interpretation and is not treated as a second bar that must be met for the PCam→CAMELYON17 claim. Additional aims are to quantify how much preprocessing still matters beside a frozen foundation encoder, and to report calibration, uncertainty, and structured qualitative review next to discrimination.

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

- Treat PCam train → CAMELYON17 test as the primary external criterion (with in-domain PCam test as reference), reporting discrimination, calibration, and errors on the multi-hospital split.

- Run the mirror (CAMELYON17 train → PCam test) to document asymmetry and ranking versus threshold/calibration effects; use it for interpretation, not as a symmetric requirement for the forward claim.

- Assess predictive reliability using calibration-focused metrics and post-hoc temperature scaling fitted on validation only, separating ranking performance (ROC-AUC and PR-AUC) from probability quality.

- Stratify test errors using predictive confidence and entropy on deterministic outputs, alongside standard false positives and false negatives.

- Perform structured qualitative error analysis using predefined buckets, a human review checklist, and exported review packages for case-based interpretation.

- Document reproducibility and experimental governance through versioned scripts, run manifests, naming conventions for experimental conditions, and explicit separation of corpus curation from per-patch preprocessing so that the pipeline can be replicated or extended.

In sum, the study tests whether a PCam-trained classifier (integrity-aware pipeline; Virchow2 or CNN) performs strongly on external CAMELYON17 test, how large the CNN–Virchow2 gap is on that path, and what the mirror adds about asymmetric transfer. It also asks whether quality-aware preprocessing remains necessary when using a frozen pathology foundation encoder (Virchow2), compared with the same pipeline for a conventional CNN. The artefacts are meant to support later work on more corpora or sites and on clinical-style operating points, without claiming deployment from patch benchmarks alone.
