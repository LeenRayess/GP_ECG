# Plan: Clinical priors for the Virchow model

**Goal:** Use histopathological knowledge so the model looks at the right features (nuclei, stroma, tissue vs background) and behaves more like a pathologist. We add **one intervention at a time**, compare to the previous run, and keep a clear baseline for ablations.

**When to use this:** After baseline Virchow training on preprocessed PCam is done and you have a reference (val/test metrics and saved model).

---

## Baseline (reference)

- **Setup:** Virchow2 frozen backbone + linear head on preprocessed PCam (96→224, ImageNet norm). No clinical priors.
- **Action:** Train to completion; record val accuracy, test accuracy, and save `model_best.pt` and `metrics.json`.
- **This is the comparison point** for every intervention below.

---

## Interventions (try one by one)

Each bullet is **one** intervention. Implement it, train, compare to the previous run (or baseline), then move to the next. Do **not** stack multiple new ideas in a single run until you have evaluated each alone.

### 1. Stain channels (H&E as explicit inputs)

- **Clinical idea:** Pathologists reason over hematoxylin (nuclei) and eosin (cytoplasm/stroma) separately. The model should have the same “channels.”
- **Implementation (conceptual):** For each patch, compute stain decomposition (e.g. Macenko or simple OD matrix) to get H and E channels. Feed the model: **RGB + H + E** (e.g. 5 or 6 channels), or a separate small branch that processes H and E. Backbone may need to accept more channels (e.g. first conv 5→64 or 6→64) or we fuse H/E with RGB before the backbone.
- **Compare:** Val/test accuracy vs baseline. Optional: check if predictions rely less on a single channel (e.g. channel-ablation or saliency).

### 2. Tissue / nucleus attention prior

- **Clinical idea:** Decisions should be driven by tissue and nuclei, not background. Encourage the model to “look at” tissue/nucleus-rich regions.
- **Implementation (conceptual):** Use an existing tissue mask or nucleus-density map (e.g. from preprocessing: saturation + local variance, or a simple H-channel threshold). Options: (A) **Auxiliary task:** predict tissue % or a low-res tissue map from the backbone features; (B) **Attention weighting:** weight the loss so errors on patches (or spatial positions) with more tissue/nuclei count more; (C) **Extra input:** give the model a tissue or nucleus map as an extra channel so it can attend to it.
- **Compare:** Val/test accuracy vs baseline. Optional: visualize attention/CAM and check overlap with tissue/nucleus regions.

### 3. Abstain on low-tissue / high uncertainty

- **Clinical idea:** Pathologists don’t call a patch “negative” when there’s almost no tissue; the model should be allowed to say “uninformative.”
- **Implementation (conceptual):** (A) **Threshold on tissue %:** at inference, if patch tissue % (from our preprocessing) is below a threshold, output “abstain” (or a third class) instead of 0/1; (B) **Confidence threshold:** if max probability is below a value (e.g. 0.7), output “abstain”; (C) **Explicit head:** add a small “is this patch diagnostically usable?” head and gate the main prediction.
- **Compare:** Val/test accuracy **on non-abstained patches** vs baseline; also report abstain rate and how often abstains are correct when we force a label (to see if abstains are sensible).

### 4. Don’t decide on a single channel (regularizer)

- **Clinical idea:** The decision should use both nuclear (H) and cytoplasmic/stromal (E) information, not a single channel.
- **Implementation (conceptual):** Add a regularizer that penalizes the model if the gradient-based saliency or attention is concentrated on only one of R, G, B (or only H or only E if we have stain channels). E.g. encourage entropy over channel-wise importance, or penalize if 90% of attribution is on one channel.
- **Compare:** Val/test accuracy vs baseline; optionally report channel-attribution balance before/after.

### 5. (Optional) Explainability alignment

- **Clinical idea:** The model should “look where the pathologist would look.”
- **Implementation (conceptual):** If we later obtain a small set of pathologist annotations (e.g. “this region is metastasis” or point annotations), add a loss term that rewards overlap between gradient-based saliency (or CAM) and those annotations. Only feasible when we have such labels.
- **Compare:** Val/test accuracy; spatial overlap (e.g. Dice) between model attention and expert regions.

---

## Order of experiments (recommended)

| Step | What to do | Compare to |
|------|------------|------------|
| 0 | Train baseline Virchow (frozen) on preprocessed PCam; record metrics | — |
| 1 | Add **stain channels (H&E)** as input; train; record metrics | Baseline (step 0) |
| 2 | Add **tissue/nucleus attention** (one of: auxiliary task, loss weighting, or extra channel); train | Step 1 (or baseline if step 1 is skipped) |
| 3 | Add **abstain** on low-tissue or low-confidence; evaluate | Baseline or best so far |
| 4 | Add **single-channel regularizer** if we have stain channels | Step 1 or 2 |
| 5 | **Explainability alignment** only if we get pathologist annotations | — |

You can reorder (e.g. do abstain before tissue attention) depending on what’s easiest to implement first; the important part is **one intervention per run** and a clear comparison target.

---

## What to record per run

- Val accuracy, test accuracy (and loss if useful).
- Any new hyperparameters (e.g. abstain threshold, regularizer weight).
- Short note: “Run X: baseline + stain channels; val acc 0.XX, test acc 0.XX.”
- Optionally: one or two attention/saliency figures per intervention to check that the model is looking at sensible regions.

---

## Where this lives in the repo

- **Baseline training:** `scripts/train_virchow_preprocessed.py`, outputs in `experiments/virchow2_preprocessed/`.
- **Future scripts:** e.g. `scripts/train_virchow_preprocessed_stain_channels.py` (or a flag `--stain-channels`) so we keep one script per variant or one script with switches. Decide when implementing.
- **This plan:** `docs/clinical_priors_plan.md` — return here after baseline is done to pick the first intervention and implement it.
