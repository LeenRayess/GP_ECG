# Methodology Section Blueprint and Gap Analysis

This document is a writing guide for the graduation report methodology chapter and a status audit against the planned Q1-style study design.

Scope note: this is about methodology structure and completeness, not final results interpretation.

## 1) What the methodology section should contain (academic structure)

Use this order so the Results chapter can mirror it one-to-one.

1. Problem framing and study design
2. Datasets and domain definition
3. Data integrity and leakage prevention
4. Preprocessing pipeline
5. Experimental conditions and ablations
6. Model architectures and training protocol
7. Evaluation protocol and metrics
8. Reliability and uncertainty protocol
9. Qualitative error analysis protocol
10. Statistical analysis plan
11. Reproducibility and implementation details
12. Planned limitations and risk controls

## 2) Recommended subsection-by-subsection methodology template

## 2.1 Study Design and Research Questions
- State the study as bidirectional domain transfer under real-world domain shift.
- Define the two transfer directions clearly:
  - PCam -> CAMELYON17
  - CAMELYON17 -> PCam
- Declare primary endpoint(s) and secondary endpoint(s):
  - Primary: external-domain discrimination and transfer drop.
  - Secondary: calibration, uncertainty behavior, qualitative failure modes.

Status: **Partially done** (well defined in `docs/q1_plan.md`, but needs final frozen endpoint wording in report text).

## 2.2 Datasets and Domain Definition
- Describe PCam and CAMELYON17/WILDS separately:
  - source, label meaning, split design, known caveats.
- Define what counts as in-domain vs out-of-domain in each phase.
- Provide final sample accounting tables (before/after filtering).

Status: **Partially done** (PCam exploration exists; CAMELYON17/WILDS equivalent dataset characterization is not yet complete in documentation).

## 2.3 Data Integrity and Leakage Prevention
- Report duplicate/near-duplicate policy.
- Report cross-dataset overlap checks and exclusion logic.
- Explicitly state constraints: no target-test information in preprocessing parameter fitting.
- Provide an auditable exclusion table with reason codes.

Status: **Partially done** (you implemented overlap/leakage-aware tooling, but final consolidated audit table is still missing from report docs).

## 2.4 Preprocessing Pipeline (Unified Across Datasets)
- Present the pipeline as fixed ordered stages:
  - optional quality filtering
  - stain handling / normalization
  - resize / resolution standardization
  - value scaling and output format
- Document parameterization and provenance:
  - reference patch policy
  - threshold policy
  - fallback behavior
  - generated artifacts (manifest, reports)
- Include an explicit "what is shared across datasets" and "what is dataset-specific" subsection.

Status: **Mostly done** (pipeline is implemented and documented; should be tightened into one concise final-method text).

## 2.5 Experimental Conditions and Ablation Logic
- Define each condition exactly and keep non-target settings fixed.
- Clarify whether the current manuscript includes:
  - baseline preprocessing
  - Macenko-based normalization
  - additional stain methods (Vahadane/hybrid) or future work only
- Add a table: condition name, stain policy, model, train dataset, test dataset.

Status: **Partially done** (current code/docs strongly cover baseline + Macenko workflow; multi-method stain ablation in Q1 plan is not fully executed yet).

## 2.6 Model Architectures and Training Protocol
- Separate model families:
  - conventional CNN baseline(s)
  - Virchow2-based classifier
- For each: architecture summary, trainable parameters policy, optimizer/loss/schedule, checkpoint rule, seed policy.
- Explain why frozen-feature strategy was chosen (if used) and what remains for fine-tuning experiments.

Status: **Mostly done** for Virchow2; **partially done** overall if CNN baseline coverage is incomplete for both transfer phases.

## 2.7 Evaluation Protocol and Metrics
- Define:
  - in-domain evaluation
  - external-domain evaluation
  - transfer degradation (absolute and relative)
- List core metrics and threshold policy:
  - ROC-AUC, PR-AUC, accuracy, precision, recall, F1, confusion matrix
  - Brier, log loss, ECE
- Specify where calibration fitting is allowed (validation only).

Status: **Mostly done** in code and partial runs; complete reporting awaits finishing ongoing CAMELYON17-related runs.

## 2.8 Reliability and Uncertainty Protocol
- Define uncertainty method(s), e.g. MC dropout setup and sampling count.
- Define calibration procedure, temperature scaling split policy.
- Define failure criteria (e.g., confident errors on OOD samples).

Status: **Partially done** (infrastructure exists; final comparative analysis across both transfer directions pending).

## 2.9 Qualitative Error Analysis Protocol
- Predefine case buckets:
  - false positives, false negatives
  - high-uncertainty true/false predictions
  - preprocessing-sensitive examples
- Predefine evidence package per case:
  - raw patch, preprocessed patch, prediction/probability, uncertainty, metadata.

Status: **Missing / early** (planned in Q1 doc, not yet clearly executed as a complete analysis protocol).

## 2.10 Statistical Analysis Plan
- Include confidence intervals (bootstrap) for key metrics.
- Predefine paired comparisons for major claims.
- State multiplicity handling or claim hierarchy (primary vs secondary claims).

Status: **Mostly missing** (important for publication-grade methodology).

## 2.11 Reproducibility and Experiment Management
- Freeze environment + dependencies + hardware notes.
- Provide run manifests and artifact map.
- Document storage paths and naming conventions for experiments.
- Record runtime caveats (e.g., cloud I/O behavior and safeguards).

Status: **Partially done** (artifacts and manifests exist; project-wide reproducibility table/checklist still needed in docs).

## 2.12 Limitations and Risk Controls (Methodology-side)
- Explicitly acknowledge:
  - patch-level labeling limits
  - domain mismatch not purely stain-driven
  - cloud runtime/IO-induced variance
- Link each limitation to a mitigation measure used.

Status: **Partially done** (strongly present in planning narrative, but not yet formalized in final method writeup).

## 3) Critique of current project status (against your goal)

This is the key candid assessment given your current repo and ongoing Colab runs.

1. Strongest area today: **engineering methodology implementation**
- Unified preprocessing and reporting artifacts are in place.
- Virchow2 train/eval path exists and has already yielded strong PCam test evidence.

2. Main gap for report readiness: **symmetry of methodological evidence**
- You investigated PCam data quality/EDA deeply.
- Equivalent CAMELYON17/WILDS data characterization is weaker in current docs and should be brought to similar depth for methodological balance.

3. Main gap for publication readiness: **statistical rigor layer**
- You have rich metrics, but not yet a complete predefined statistical comparison plan with confidence intervals across all key claims.

4. Main gap for narrative coherence: **single canonical methodology text**
- Information is currently distributed across multiple docs and notebooks.
- You need one final method chapter that is concise, locked, and citation-friendly.

5. Ongoing-run dependency risk
- Since cross-dataset training/testing runs are still in progress, do not overcommit claims in methodology text that assume those outcomes.
- Write methodology in protocol form now; reserve claim strength for Results/Discussion after run completion.

## 4) What appears missing (from Q1 plan and beyond)

## 4.1 Missing from current execution relative to `q1_plan.md`
- Full stain-ablation set beyond current baseline/Macenko-focused pipeline (if intended for this paper version).
- Complete bidirectional run matrix finalized and logged (still ongoing).
- Formal explainability-driven error analysis section with predefined selection protocol.
- Formal statistical testing/CI framework integrated into final reporting.

## 4.2 Missing but important beyond the Q1 plan
- A "dataset shift characterization" subsection quantifying source-target distribution differences before modeling (color/texture/stat summaries).
- A "methodological decision log" table (decision, rationale, alternatives rejected, risk) to strengthen transparency for reviewers.
- A strict "claim-evidence map" linking each final claim to specific experiment IDs/figures/tables.
- Explicit separation of "confirmatory analyses" vs "exploratory analyses" to avoid reviewer concerns about post hoc conclusions.

## 5) Concrete tasks to close the methodology gap quickly

1. Create CAMELYON17/WILDS dataset characterization notebook/doc matching PCam depth:
- class distribution, patch/tissue stats, stain/color stats, quality-filter impact, split-wise summaries.

2. Build one experiment matrix table (the backbone of both Methodology and Results):
- columns: experiment_id, preprocess condition, model, train_domain, test_domain, seed, status, artifacts path.

3. Add statistical plan section before results are finalized:
- bootstrap CI settings, primary comparisons, secondary comparisons.

4. Define qualitative review protocol now (before seeing full results):
- sample-selection rules and annotation template.

5. Consolidate all method details into one locked chapter draft:
- then keep only minor edits after runs finish.

## 6) Methodology -> Results mirror map (for clean report flow)

Use this exact mapping:

- Method 2.1 Study design -> Results 3.1 Experiment completion map
- Method 2.2 Datasets -> Results 3.2 Dataset/split summary tables
- Method 2.3 Leakage controls -> Results 3.3 Integrity check outcomes
- Method 2.4 Preprocessing -> Results 3.4 Preprocessing diagnostics/impact
- Method 2.5 Ablations -> Results 3.5 Condition-wise comparisons
- Method 2.6 Models/training -> Results 3.6 Convergence and model performance
- Method 2.7 Metrics -> Results 3.7 In-domain/OOD metrics and transfer drop
- Method 2.8 Reliability/uncertainty -> Results 3.8 Calibration and uncertainty behavior
- Method 2.9 Qualitative protocol -> Results 3.9 Case-based error analysis
- Method 2.10 Statistics -> Results 3.10 CIs and significance-aware interpretation

## 7) Suggested claim discipline for writing now

When drafting methodology now:
- Use protocol language ("we defined", "we evaluated", "we pre-specified").
- Avoid result-implying language ("this improved", "this outperformed") until ongoing runs finish.
- Mark pending analyses as "predefined and currently running" where relevant.

---

Prepared as a report-writing guide tied to current repository status and Q1 plan coverage.
