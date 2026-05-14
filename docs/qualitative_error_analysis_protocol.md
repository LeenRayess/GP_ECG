# Qualitative Error Analysis Protocol (Post-Run)

This document defines a lightweight, publication-ready qualitative error analysis workflow to run after training and testing are complete.

It is designed to be feasible under limited time and compute, while still producing defensible evidence for the methodology and results sections.

## 1. Purpose

The goal is to understand *why* the model succeeds or fails on representative cases, especially under domain shift. The protocol focuses on structured case review rather than anecdotal examples.

## 2. Inputs Required (After Runs Finish)

Prepare these artifacts for each evaluated condition:

- patch identifiers (sample index or unique ID),
- ground-truth label,
- predicted probability and predicted class,
- correctness flag (correct/incorrect),
- uncertainty value(s), at minimum MC-std if available,
- domain metadata (train-source domain, test domain, split),
- access to raw and preprocessed patch images.

If MC uncertainty is unavailable for one condition, run the same qualitative buckets using confidence only and state that limitation explicitly.

## 3. Case Buckets (Predefined)

Use the same bucket definitions across all model/domain conditions.

1. **False Positives (FP):** \(\hat{y}=1, y=0\)
2. **False Negatives (FN):** \(\hat{y}=0, y=1\)
3. **High-uncertainty errors:** \(\hat{y}\neq y\) and uncertainty in top decile of the evaluated split
4. **High-uncertainty correct:** \(\hat{y}=y\) and uncertainty in top decile

Optional (if time allows):

5. **High-confidence errors:** incorrect predictions with confidence \(\ge 0.9\)

## 4. Sampling Plan (Low-Effort, Strong Coverage)

Target **40 total reviewed cases per comparison direction** (four core buckets), sampled as:

- 10 FP
- 10 FN
- 10 high-uncertainty errors
- 10 high-uncertainty correct

If the optional high-confidence error bucket is included, add **10** more (50 total per direction).

If a bucket has fewer than target size, take all available and document shortfall.

Use fixed random seed for random sampling within each bucket (recommended: seed = 42).

## 5. Evidence Package Per Case

For each selected case, store:

- raw patch image,
- preprocessed patch image,
- alignment note: preprocessed `test_x.h5` rows follow the evaluation manifest; the matching row in the **original** raw `test_x.h5` is `manifest.json` → `test.kept_indices[prep_row]` (not necessarily the same integer as `prep_row` when quality filtering removed patches),
- true label and predicted label,
- predicted probability,
- uncertainty statistic(s),
- bucket assignment,
- domain direction (PCam->CAMELYON17 or CAMELYON17->PCam),
- model condition ID (e.g., C2/C4/C7).

## 6. Human Review Checklist (Per Case)

For each case, mark each item as Present / Absent / Unclear:

1. Tissue scarcity (limited informative tissue)
2. Artifact burden (blur, compression, stain artifact, dark region)
3. Borderline morphology (ambiguous visual pattern)
4. Small-focus lesion pattern (tiny focal signal likely easy to miss)
5. Color/stain atypia relative to typical in-domain appearance
6. Patch-context limitation (insufficient context at patch scale)

Add one short free-text note (1-2 lines) explaining the dominant suspected failure or uncertainty driver.

## 7. Reviewer Consistency Rule

To reduce subjective drift:

- export files list cases in the same order as `gallery.html` (bucket sections alphabetically; `review_order` runs 1..N in that sequence); randomize in Excel/Sheets if you prefer a different review sequence,
- use the same checklist wording throughout,
- avoid changing definitions mid-review,
- if uncertain, mark "Unclear" rather than forcing a category.

If two reviewers are available, dual-review a 15-20 case subset and report agreement percentage for checklist items.

## 8. Summary Tables to Produce

After review, create:

1. **Bucket composition table**
   - bucket name, sampled \(n\), available \(n\), shortfall (if any)

2. **Pattern frequency table**
   - checklist pattern vs frequency within each bucket and domain direction

3. **Representative-case table**
   - 2-3 exemplar cases per key pattern with concise interpretation notes

## 9. Minimal Statistical Reporting

For each checklist pattern, report proportion:
\[
\hat{p}=\frac{k}{n},
\]
where \(k\) is count of cases marked Present and \(n\) is bucket sample size.

For transparency, include 95% Wilson interval (optional but recommended):
\[
\text{CI}_{95\%}^{\text{Wilson}}(\hat{p}, n).
\]

If formal CI computation is skipped, report raw counts clearly (e.g., 4/10).

## 10. Predefined Interpretation Boundaries

To avoid overclaiming:

- treat qualitative findings as explanatory support, not causal proof,
- do not generalize beyond reviewed buckets without quantitative backing,
- separate observations (what was seen) from hypotheses (why it happened).

## 11. Suggested Folder Layout (When You Execute)

Recommended output location:

`reports/qualitative_error_analysis/<run_id>/`

Include:

- `selected_cases.csv`
- `review_labels.csv`
- `summary_tables.csv`
- `representative_cases.md`
- `figures/` (raw/preprocessed side-by-side examples)

## 12. Fill-In Template (Ready to Use)

### 12.1 Run metadata

- Run ID:
- Date:
- Reviewer(s):
- Compared condition(s):
- Domain direction:

### 12.2 Sampling summary

- FP sampled:
- FN sampled:
- High-uncertainty errors sampled:
- High-uncertainty correct sampled:
- Total reviewed:

### 12.3 Key observations (3-6 bullets)

- 
- 
- 

### 12.4 Limits noted during review

- 
- 

## 13. Time Budget (Realistic)

With 80 cases and checklist-only annotation:

- case preparation/export: 10-25 min
- human review: 60-90 min
- summary tables and notes: 30-45 min

Total typical effort: ~2 to 3 hours.

