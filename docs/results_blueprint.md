# Results Section Blueprint and Gap Analysis

This document is a writing guide for the graduation report **Results** chapter and a status audit against **`docs/final_methodology.md`**.

Scope note: the Results chapter should report **empirical outcomes for essentially every methodological step** that produced numbers, tables, plots, or audits—not only the main classifier scores (C1–C4). Where a step is purely definitional, Results still states **observed counts or checks** (e.g. how many patches were removed and why). Use the same blueprint **voice** as `docs/methodology_section_blueprint_and_gap_analysis.md`: bullets, explicit **Status:** lines, gap sections, and a mirror map.

---

## 1) What the results section should contain (pipeline-faithful order)

Order Results so a reader can walk **Methodology → evidence** in the **same sequence** as the methodology chapter (with a short **reporting conventions** block up front, and **model-centric outcomes** after preprocessing is fully evidenced).

1. **Opening overview** — Results reports **observed** outcomes for each pipeline stage under the frozen protocol in Methodology; points to Figure 1 only if needed (do not duplicate the methodology schematic unless you add a results-only figure).
2. **Reporting conventions** — threshold, calibration, deterministic inference, bootstrap *B*, primary vs secondary / exploratory claims (pointers to Methodology Chapters 5–7; no equation re-derivation).
3. **Splits and integrity (Methodology §3.1, Table 3.1)** — outcomes of split isolation checks; **PCam deduplication** (how many groups / patches removed or merged); **cross-dataset exact overlap** (how many hashes flagged, disposition); confirm no test leakage in numbers (e.g. validation-only usage is reflected in artefact dates or counts, if you report that).
4. **Descriptive dataset statistics (Methodology Chapter 2, §2.1)** — if not fully absorbed into Chapter 2, Results repeats or extends **split-level summaries** actually used in the thesis (sampled descriptors, class balance confirmation); otherwise one paragraph + “as in Table 2.x” with any **post-QC** update if different from pre-QC sampling.
5. **Quality filtering (Methodology §3.2)** — **before/after patch counts** per split and dataset; exclusion counts **by rule** (solid-color, high-black, low-tissue) if available from reports; optional distribution shift plots for a sampled subset **before vs after** QC.
6. **Stain normalization (Methodology §3.3)** — **full benchmark outcomes** on the controlled PCam subsets: CNN metrics per stain method / adaptive variant (C5–C6 family); any **distance-to-reference** or preprocessing-report summaries used to justify stability; **Macenko vs alternatives** in numbers (this supports the methodological choice with evidence, not only C5–C6 in the experiment table sense).
7. **Value normalization and resizing (Methodology §3.4)** — **bicubic vs bilinear** comparison outcomes (same benchmark subsets, same CNN, same metrics as methodology); numeric summary **and** note of informal visual checks; confirm **final resize policy** used downstream matches reported winner.
8. **Final cohort sizes after preprocessing** — reconciled table **per dataset × split** (train / val / test) aligned with Methodology Table 2.3 if unchanged, or **updated** totals if QC/stain steps changed counts after that table was frozen; explicit **“final analysed N”** for PCam and CAMELYON17.
9. **Experiment / condition matrix (Methodology Chapter 4, Table 4.1)** — artefact traceability: **C1–C8** rows with checkpoint paths, logit export paths, status (complete / pending); links preprocessing **subset** runs vs **full-data** runs where both exist.
10. **Model performance: Virchow2 (C1–C4)** — in-domain and external discrimination + calibration rows; transfer-gap summary.
11. **Model performance: CNN baseline (C7)** — same structure as far as completed.
12. **Stain ablation as experiment conditions (C5–C6)** — if not fully folded into §1.6, repeat the **condition-indexed** table rows here for the thesis experiment inventory (or cross-reference §1.6 and avoid duplicate tables—**one** canonical table is enough if clearly signposted).
13. **Calibration and reliability (Methodology Chapter 5)** — Brier, log loss, ECE for each reported test surface; domain-shift commentary.
14. **Statistical inference (Methodology Chapter 7)** — bootstrap CIs, permutation tests, multiplicity / claim tier labelling.
15. **Qualitative synthesis (Methodology Chapter 6, C8)** — bucket prevalences, panels, counts.
16. **Chapter synthesis** — ties preprocessing evidence + model evidence to primary/secondary endpoints; no new numbers.

---

## 2) Recommended subsection-by-subsection results template

Below, **§** is placeholder numbering for your final Results document. Rename to match thesis chapter numbering.

### 2.1 Opening overview

- One paragraph: Results = **empirical trace** of the methodology pipeline + **condition-wise** model outcomes.
- Second paragraph: read order follows **Methodology Chapters 2–7** (data → preprocessing evidence → models → metrics → qualitative → statistics).

**Status:** **To draft**.

### 2.2 Reporting conventions and claim hierarchy

- Pointers only: Methodology **Sections 5.2, 5.3, Chapter 7**; bootstrap *B*; primary vs secondary vs exploratory.

**Status:** **To draft**.

### 2.3 Splits, deduplication, and cross-dataset overlap (Methodology §3.1)

- **PCam:** duplicate group counts, patches removed or representative retention rule (one index per group kept—show resulting delta).
- **PCam ↔ CAMELYON17:** exact-hash overlap count; confirm exclusion or handling policy applied.
- Optional: small table mirroring **Table 3.1** with a **“observed outcome”** column (numbers, not just rules).

**Status:** **Partially done** (tooling and some JSON/CSV reports exist under `reports/`; consolidated prose + one thesis table may be missing).

### 2.4 Descriptive statistics and sample sizes (Methodology Chapter 2, §2.1)

- If Chapter 2 already holds all descriptive tables, Results adds only **post-QC** or **post-stain** updates if they differ from pre-processing sampling.
- Otherwise: reproduce or cite **Table 2.2-style** summaries for the **final analysed cohort**.

**Status:** **Partially done** (depends whether final counts changed after methodology tables were fixed).

### 2.5 Quality filtering outcomes (Methodology §3.2)

- **Per split, per dataset:** patches before QC → after QC; counts removed by each gate (or aggregate if only aggregate exists).
- If reports give **sampled distributions** of tissue proxy / black fraction before vs after, include one figure or cite report path.

**Status:** **Partially done** (audit / filter logic exists; thesis-ready tables may need assembly from `reports/` or notebooks).

### 2.6 Stain normalization: benchmark and diagnostics (Methodology §3.3)

- **Benchmark CNN metrics** for Macenko / Reinhard / Vahadane / adaptive variants on the **same** PCam subset indices (train/val/test as defined in methodology).
- Any **colour-distance** or preprocessing-report statistics used to argue stability (before vs after, or across methods).
- **Selection outcome:** Macenko chosen—show the **numeric** basis (e.g. best trade-off on agreed primary benchmark metric, or consistency row) as reported in results, not only prose in methodology.

**Status:** **Variable** (subset benchmarks exist; full write-up in thesis may be partial).

### 2.7 Resizing: bicubic vs bilinear (Methodology §3.4)

- Side-by-side **quantitative** comparison (same subsets, same CNN, same metrics); state winner and magnitudes (even if small).
- One sentence on **informal visual** assessment if that was part of the decision.

**Status:** **Partially done** (e.g. `notebooks/temp_bilinear_vs_bicubic.ipynb`—promote to cited thesis result or extract summary table).

### 2.8 Final analysed patch counts (post full preprocessing)

- Single **master table**: dataset × split × **final N** positives/negatives after **all** steps through resize policy; footnote reconciliation with Methodology Table 2.3 if identical or if updated.

**Status:** **To assemble** (source from final manifests / training dataloaders).

### 2.9 Condition and artefact index (Methodology Chapter 4)

- **C1–C8** table: purpose (short), artefact locations, completion status.
- Distinguish **preprocessing benchmark subsets** from **full training runs** if both appear in the repo.

**Status:** **Partially done**.

### 2.10 Main classification performance: Virchow2 (C1–C4)

- As before: full metric rows + transfer gaps.

**Status:** **Pending / in progress** (eval notebooks).

### 2.11 CNN baseline (C7)

- In-domain + external for both training directions where available.

**Status:** **Partially done**.

### 2.12 Stain conditions on benchmark inventory (C5–C6)

- If §2.6 already contains the full benchmark narrative, **either** shorten this to “experiment conditions as registered in Table 4.1” **or** use this subsection for **headline comparison to final Macenko full-data** (only if defined and not redundant).

**Status:** **Editorial choice** (avoid triple-reporting the same numbers).

### 2.13 Calibration and reliability outcomes

- Brier, log loss, ECE tables for each reported evaluation surface.

**Status:** **Pending**.

### 2.14 Statistical inference outcomes

- CIs, permutation *p*, BH tier labels.

**Status:** **Mostly missing** in consolidated thesis form.

### 2.15 Qualitative synthesis (C8)

- Bucket prevalences + figure references.

**Status:** **Early**.

### 2.16 Chapter synthesis

- Links **§2.3–2.8** (pipeline evidence) to **§2.10+** (model evidence) and states endpoint answers.

**Status:** **To draft** last.

---

## 3) Critique of current project status (results-readiness)

1. **Strong:** model metric **definitions** and reporting template are clear in methodology.
2. **Weak:** **pipeline results** are often still in **notebooks / `reports/` JSON** rather than one **Results-facing** narrative—examiner may not connect preprocessing evidence to final Ns.
3. **Weak:** **bicubic vs bilinear** and **stain benchmark** risk being “methods folklore” unless they appear as **numbered Results** with tables.
4. **Risk:** **duplicate counts** if Table 2.3 (methodology) and post-QC totals diverge—Results must **reconcile** explicitly.
5. **Risk:** **scope creep**—use **one canonical table** per topic (benchmark stains; bicubic; final Ns) and cross-reference elsewhere.

---

## 4) What appears missing (relative to “every step has results”)

### 4.1 Missing or scattered empirical packages

- Consolidated **QC exclusion** table per split.
- Consolidated **overlap + dedup** outcome table.
- **Final cohort** table after **all** preprocessing (not only after QC).
- Thesis-ready **bicubic vs bilinear** summary extracted from notebook experiments.
- **Stain benchmark** summary table aligned to methodology’s listed methods (not only the final Macenko paragraph).

### 4.2 Missing for narrative coherence

- A **single diagram or table** “pipeline stage → artefact path → subsection in Results” (optional one-pager in appendix).
- Explicit **“not reported”** list for any planned diagnostic that failed or was skipped (reviewer trust).

---

## 5) Concrete tasks to close the gap (pipeline-first)

1. Inventory **all existing outputs** under `reports/`, `notebooks/`, and run folders → map each to a **§2.x** subsection above.
2. Build **§2.8 master counts** from final training manifests (source of truth for models).
3. Extract **§2.7** numbers from the bicubic comparison notebook into a CSV + one figure for the thesis.
4. Extract **§2.6** stain-benchmark tables from benchmark logs / notebooks (same metric columns as methodology).
5. Build **§2.3–2.5** tables from integrity + QC audit scripts (automate if possible).
6. Only after **§2.3–2.9** are stable, lock **§2.10–2.14** model and inference tables (so Discussion does not fight preprocessing Ns).

---

## 6) Expanded methodology → results mirror map

| Methodology location | What Results must show (empirical) |
|---|---|
| **Ch 1** Study design | Minimal: how reported metrics map to **primary/secondary endpoints** (can be a short table); avoid repeating ethics/design prose. |
| **Ch 2** Datasets + §2.1 descriptors | Final or updated **descriptive** summaries if cohort changed after methodology tables; else explicit “unchanged since Table 2.x”. |
| **Ch 3.1** Split / integrity / Table 3.1 | **Dedup**, **overlap**, split integrity **counts** (and exclusions if any). |
| **Ch 3.2** Quality filtering | **Removals by rule**, before/after **Ns** per split. |
| **Ch 3.3** Stain normalization + benchmark | **Benchmark metrics**, distance or stability diagnostics, **evidence for Macenko choice**. |
| **Ch 3.4** Value norm + resize | **Bicubic vs bilinear** quantitative outcome (+ visual note). |
| **Ch 4** Conditions C1–C8 + training | **Artefact index**, optional **training curves** or selection criterion values if reported. |
| **Ch 5** Outcomes | **Classification + reliability metrics** for each completed condition. |
| **Ch 6** Qualitative | **Bucket prevalences**, panels. |
| **Ch 7** Statistical analysis | **CIs, tests, multiplicity**. |
| **Ch 8** Reproducibility | Not duplicated; at most **one sentence** in Results synthesis pointing forward to Discussion/limitations. |

---

## 7) Suggested claim discipline while drafting Results

- **Past tense** for measured outcomes (“removed”, “achieved”, “observed”).
- **Present tense** only when pointing to methodology definitions.
- Every **subsection §2.3–2.8** should contain **at least one number or table** (no empty “we did X” without evidence).
- Do not **upgrade** a preprocessing diagnostic to a **clinical claim**—keep interpretation modest; save deployment claims for Discussion.
- If methodology and results **counts diverge**, fix methodology **or** explain the delta in Results **first** before publishing model tables.

---

Prepared as a **pipeline-complete** results blueprint aligned with **`docs/final_methodology.md`** and the style of **`docs/methodology_section_blueprint_and_gap_analysis.md`**.
