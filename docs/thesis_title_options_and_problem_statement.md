# Thesis title options (same style as current)

Current style: **[Main line: problem + scale]** + **colon** + **three pillars** (methods / representation / evaluation), formal tone.

---

**Option A (evaluation-forward, keeps “domain shift”)**  
Robust Patch-Level Metastasis Detection Under Domain Shift: Preprocessing, Foundation-Model Representations, and Multi-Hospital External Validation from PCam to CAMELYON17

**Option B (shorter evaluation clause)**  
Robust Patch-Level Metastasis Detection Under Domain Shift: Preprocessing, Foundation-Model Representations, and PCam-to-CAMELYON17 External Transfer with a Prespecified Mirror

**Option C (stress “asymmetry” instead of “bidirectional”)**  
Robust Patch-Level Metastasis Detection Under Domain Shift: Preprocessing, Foundation-Model Representations, and Asymmetric Cross-Benchmark Transfer with External CAMELYON17 Testing

**Option D (integrity + foundation, slightly more methods-led)**  
Robust Patch-Level Metastasis Detection Under Domain Shift: Integrity-Aware Preprocessing, Virchow2 Representations, and External Validation on CAMELYON17 with Mirror-Augmented Analysis

**Option E (closest length to original subtitle)**  
Robust Patch-Level Metastasis Detection Under Domain Shift: Preprocessing, Pathology Foundation Models, and Forward External Validation with Prespecified Transfer Asymmetry

**Option F (if you want “generalization” instead of “transfer” once)**  
Robust Patch-Level Metastasis Detection Under Domain Shift: Preprocessing, Foundation-Model Representations, and Generalization to Multi-Hospital CAMELYON17 from PCam Development Data

---

Pick one pillar to shorten if the cover page line must be one line in Word; **B** and **E** tend to fit layouts that constrained the old title.

---

# §1.4 Problem statement (revised for the pivot)

Copy-paste for Introduction §1.4 Problem Statement (adjust cross-references to your chapter numbering if needed).

Histopathological assessment of sentinel lymph nodes on hematoxylin and eosin (H&E) sections remains indispensable for breast cancer staging, yet substantive inter-observer disagreement persists in routine nodal review, and the work is labor-intensive, which sustains demand for scalable computational assistance. Patch-level machine learning offers a plausible path toward faster, more standardized screening, but its practical utility is constrained by domain shift: systems developed under one staining protocol, scanner family, or institutional pipeline commonly lose performance when applied under another, which undermines naive claims of generalisation from a single development environment. A further unresolved tension accompanies large pathology foundation models used with a frozen encoder and shallow trainable head, because the incremental benefit of careful preprocessing and stain harmonization relative to representation already absorbed during pretraining has not been clearly separated, in this setting, from the role of an end-to-end convolutional baseline trained on the same input distribution.

These issues become sharper on public patch benchmarks such as PatchCamelyon (PCam) and CAMELYON17 (WILDS). Transfer is not symmetric in practice. A model trained on PCam may still rank well on multi-hospital CAMELYON17 test while a mirrored setup, with training on CAMELYON17 and testing on PCam, can show different threshold behaviour and calibration when labels and centre mixes differ. Treating both directions as the same kind of proof is misleading if the main clinical stress is whether PCam-trained weights work on harder external nodal tissue. There is also a reporting problem. Summaries that lean on in-dataset accuracy or on a single headline score hide poor probability behaviour under shift. They say little about uncertainty in a useful form, and they rarely connect errors to tissue appearance in a structured way. What is missing is routine joint reporting of discrimination, calibration, uncertainty summaries, and qualitative error review. What is missing, too, is a clear habit of foregrounding PCam-trained models on held-out multi-hospital CAMELYON17 test while using a prespecified mirror only to show where ranking, thresholds, and calibration diverge, not to restate the main external question as if it were the same problem in reverse.

