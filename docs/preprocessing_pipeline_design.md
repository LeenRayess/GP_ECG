# Generalizable preprocessing pipeline — design

**Goal:** The **model + preprocessing pipeline** is generalizable. We define a **chosen standard** once and bring every image (training or deployment, any dataset, any stain) to that standard. The pipeline is **adaptive** so it can handle new scanners, labs, and datasets without retraining.

---

## 1. Chosen standard

We fix three things **once** and reuse them everywhere:

| Dimension | Chosen standard | Purpose |
|-----------|-----------------|---------|
| **Stain appearance** | One reference patch | Any stain is mapped to the same H&E look the model was trained on. |
| **Resolution / size** | Reference resolution + fixed input size (e.g. 96×96 or 224×224) | Any image becomes a fixed grid the model expects. |
| **Value range** | [0, 1] (divide by 255), optional outlier clip | Pixel statistics are consistent across scanners/datasets. |

Training and deployment use the **same** pipeline and standard.

---

## 2. Pipeline order

```
Input image
  → [Optional] Quality filter (exclude patches that fail our rules)
  → Stain normalization (map to reference stain)
  → Resolution/size standardization
  → Value normalization
  → Output: image in our standard space
```

Each step is **adaptive** where needed (per-image stain, resolution); the **target** is always our fixed standard.

---

## 3. Generalizability: what works on any patch vs what is fixed

So that **any patch from any dataset** can be fed in and **output to our standard**, the pipeline is designed as follows.

### Fully general (any patch, any dataset)

- **Value normalization:** Divide by 255 → [0,1], optional percentile clip. No dataset-specific parameters; works on any RGB image.
- **Pipeline structure:** The same sequence (quality → stain → resolution → value) and the same **logic** (e.g. Macenko vs Reinhard by blue_dom_pct, tissue % = saturation OR variance + Otsu on edge cases) apply to any input. The **code path** is the same for PCam, CAMELYON17, or a new lab’s data.
- **Quality rules:** Solid-colour (low gray std), high black, low tissue (combined mix) are defined from image statistics only—no labels or scanner IDs. Any patch can be scored and excluded or kept by the same rules.

### Our fixed standard (same for everyone)

These are **chosen once** and reused so that every image is brought to the **same** target. They do not depend on which dataset the patch came from.

- **Stain reference:** One reference patch (and its fitted Macenko/Reinhard parameters). Every input is normalized **to that look**. So “any stain” → “our standard stain.”
- **Reference resolution and input size:** e.g. 0.5 µm/px (or dataset documented value) and 96×96 or 224×224. Every image is scaled and resized to that resolution and size. Bicubic for 96→224.
- **Value range:** [0, 1]. Same for all.

### General logic with fixed or tunable parameters

- **Stain:** The **rule** (blue_dom_pct vs a threshold → Reinhard vs Macenko) is general. The **threshold** can be (a) computed per dataset (e.g. 25th percentile of blue_dom_pct on a sample of that dataset), so the rule adapts to the new data, or (b) fixed from our reference dataset and reused everywhere so behaviour is identical. For “any patch from any dataset” we can fix the threshold so the pipeline is fully portable.
- **Quality thresholds** (e.g. gray_std &lt; 0.04, black ≥ 0.5, tissue % &lt; 0.35): Intended as **general defaults** for H&E-like images. They are not fit to PCam labels. If a new dataset has very different statistics (e.g. much paler slides), these can be tuned without changing the pipeline logic.
- **Resolution:** If metadata or dataset documentation exists, we use it. Otherwise we use content-based estimation (scale-invariant) or a default. The **method** is general; the **reference resolution** (our target µm/px) is part of our fixed standard.

### Assumptions (limit of “any dataset”)

- **H&E-like stain:** Macenko and our reference are for two-stain H&E. Non-H&E (e.g. IHC, other stains) would need a different reference and possibly different methods.
- **RGB digital images:** Pipeline expects RGB patches. Same idea can be extended to other modalities with appropriate normalization steps.
- **Histology context:** Tissue % and nucleus-based resolution estimation assume histology (tissue vs background, nuclei). The **logic** (saturation, variance, Otsu) is still applicable to other imaging if “tissue” is reinterpreted.

**Summary:** Any patch from any (H&E-like) dataset can be passed through the pipeline; the **output** is always in **our** standard (our reference stain, our resolution/size, [0,1]). What is general is the **procedure** and the **logic**; what is fixed is the **target** we impose so everything “looks like our standards.”

---

## 4. Stain normalization

**Idea:** Estimate stain from the current image and map it to a **reference**. Same math for every image; only the source changes.

**Why we need it:** Different labs and scanners give different shades of purple and pink. Stain normalization recolors each image so it matches our **reference** H&E look. The model then sees a consistent “colour world.”

**Why not greyscale?** In H&E, colour carries meaning (purple = hematoxylin/nuclei, pink = eosin/cytoplasm). Greyscale would lose that; we keep colour and standardise the shade.

### 4.1 Methods we use

- **Macenko (default):** Separates the two stain colours in optical-density space and remaps them to the reference. Works well on typical H&E with enough tissue. Can fail on very pale slides, almost no tissue, or degenerate stain (empty tissue mask, singular matrix).
- **Reinhard (fallback):** Matches mean and standard deviation of each channel (in LAB) to the reference. Does not use a tissue mask. More robust when Macenko fails (e.g. no tissue, solid colour).

Other methods (Vahadane, SPCN) exist; we use only Macenko + Reinhard.

### 4.2 Decision rule: Macenko vs Reinhard

We choose **per patch**:

- **blue_dom_pct** = fraction of pixels where blue > red (simple proxy for “how blue-dominated”).
- Compute the **25th percentile** of blue_dom_pct on a sample of the dataset → **threshold**.
- If **blue_dom_pct < threshold** → use **Reinhard** (patch is H-heavy / low eosin; Macenko often fails or gives poor results).
- Else → use **Macenko**.

If **both** Macenko and Reinhard fail (e.g. empty tissue mask, zero variance), we use **luminosity-standardized image only** (no stain norm) and **record the patch index** for later (e.g. to check overlap with low-tissue / solid-colour removal).

### 4.3 Reference patch

- **Luminosity standardization** is applied to the reference (and to each source patch) before fitting/transforming Macenko, so brightness is comparable.
- **Selection:** From patches that pass quality filters (not solid-colour, not high black, tissue % ≥ 0.5), we pick **one** reference by **weighted distance** to a target: we bias toward “pink” H&E by targeting **62nd percentile R** and **38th percentile B** (on the candidate set). Weights emphasize R and B over G and tissue % so the chosen patch matches the desired stain balance. The chosen index (e.g. from PCam train) is saved (e.g. `stain_reference.json`) and reused for the full dataset and deployment.

### 4.4 Robustness

- **Fallback order:** Try chosen method (Macenko or Reinhard by rule) → on exception (e.g. `TissueMaskException`, `LinAlgError`), try the other → if both fail, use luminosity-only and record index.
- **Contrast stretch** is used only for **display** of normalized images; the stored/used standard is the normalizer output (optionally after value normalization to [0,1]).

---

## 5. Resolution and size

**Idea:** Every image is brought to a **reference resolution** and then to a **fixed input size** (e.g. 96×96 for baseline, 224×224 for Virchow2).

**Why we can’t infer resolution from pixels alone:** Physical resolution (µm per pixel) comes from acquisition (scanner, objective), not from pixel values. We need **metadata** (e.g. MPP), **documented** dataset resolution, or a **content-based estimate**.

### 5.1 Resolution order

1. **Metadata** (e.g. MPP) if present → use it to scale to reference resolution.
2. **Documented dataset resolution** (e.g. PCam README: 2.43 µm/px) → use it; no nucleus estimation needed for that dataset.
3. **Content-based estimate** when metadata and documentation are missing: estimate µm/px from structure size (e.g. median nucleus diameter in pixels + prior ~7 µm), with **scale-invariant** blob filtering (e.g. percentile of blob areas), then sanity bounds. See notebook `test_nucleus_resolution_estimation.ipynb`.
4. **Default** if estimate is invalid or missing.

### 5.2 Resizing 96→224 (Virchow2)

- We **upscale** 96→224 so the backbone sees its expected input size. We use **bicubic** interpolation (chosen over bilinear for sharper edges). Field of view and biological content are unchanged; we do not gain real detail.

---

## 6. Value normalization

- **Standard:** `x = x / 255` → [0, 1].
- **Optional:** Clip pixel intensities by 1st–99th percentile (per image or per channel) before scaling, to reduce impact of artifacts and saturation. The “extreme values” we clip are **pixel intensities** (raw R/G/B), not dataset-level statistics.

---

## 7. Quality filter (optional)

Applied **before** stain/size/value. We **exclude** patches that fail the rules below; we do **not** remove by high white ratio (only solid-colour, high black, and low tissue). This is not slide-artifact detection (folds, pen marks, bubbles)—those are a separate, optional step; here we only exclude by content/quality.

**Order:**

1. **Solid-colour:** Remove patches with very low grayscale standard deviation (e.g. &lt; 0.04 on [0,1]). These have almost no texture.
2. **High black:** Remove patches with fraction of (nearly) black pixels ≥ 0.5.
3. **Low tissue:** Remove patches with **tissue % final &lt; 0.35**.

### 7.1 Tissue % (what we use)

We use a **combined, opacity-invariant** measure:

- **Saturation:** Fraction of pixels with HSV saturation &gt; threshold (e.g. 0.12). Catches coloured tissue vs white/grey background.
- **Local variance:** Fraction of pixels with local variance (e.g. 11×11 window) &gt; threshold (e.g. 0.003). Catches texture vs flat background.
- **Combined:** A pixel counts as “tissue” if **either** condition holds. **Tissue %** = fraction of such pixels.
- **Edge cases:** When this combined % is in a middle range (e.g. 0.12–0.45), we also compute **contrast-normalized Otsu** (5th–95th percentile stretch, then Otsu; tissue = fraction below threshold) and take **max**(combined, Otsu-based) so borderline patches are not undercounted.

**Threshold:** Remove if **tissue % final &lt; 0.35**. This is a general default (not tuned to PCam labels).

**Generality:** Solid-colour and high-black rules are general. The tissue logic (saturation OR variance, Otsu on edge cases) and the 0.35 threshold are intended as a general default for H&E-like data; the same pipeline can be applied to other datasets.

**Blur:** Flagging or excluding very blurry patches (e.g. Laplacian variance below a percentile) is **optional** and can be added later. **Slide artifacts** (folds, pen marks, bubbles): detecting and masking or excluding those is a separate, optional step; not part of the current pipeline.

---

## 8. What we include (checklist)

| Step | Include? | Notes |
|------|----------|--------|
| Stain normalization (Macenko / Reinhard by blue_dom_pct, fallback to luminosity-only) | **Yes** | Single reference, luminosity std, indices saved when both fail |
| Resolution/size to reference µm/px + fixed size | **Yes** | Metadata → documented → content-based → default; bicubic 96→224 |
| Value normalization [0,1], optional clip | **Yes** | Same rule for all |
| Quality filter (solid-colour, high black, low tissue) | **Optional** | Order and thresholds as above; no high-white removal |
| Deduplication | **Separate** | Not part of per-image standardization |

---

## 9. Implementation outline

1. **Chosen standard (saved):**
   - One reference patch index and fitted normalizers (or reference stain parameters) for stain.
   - Reference resolution and input size(s) (96, 224).
   - Value rule: [0, 1], optional percentile clip.

2. **Per image:**
   - Optional quality filter → exclude or keep.
   - Stain: luminosity-standardize, then Macenko or Reinhard by blue_dom_pct; on double failure use luminosity-only and record index.
   - Resolution/size: scale to reference resolution, then to 96 or 224 (bicubic when upscaling 96→224).
   - Value: divide by 255, optional clip.

3. **Training and deployment:** Same pipeline and config for PCam and for new data (e.g. CAMELYON17).

4. **Artifacts:** Save reference index, normalization summary, and (when applicable) indices of patches that received luminosity-only so they can be compared with quality-filter exclusions (e.g. low-tissue / solid-colour).

This gives a **generalizable preprocessing pipeline** that brings any dataset to the chosen standard and keeps **model + preprocessing** as the deployable unit.
