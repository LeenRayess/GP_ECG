# Generalizable preprocessing pipeline — design for “any dataset, any stain”

**Goal (supervisor):** The **model + preprocessing pipeline** should be generalizable. We do not train on raw, unstandardized images; we define a **chosen standard** once and bring every image (training or deployment, any dataset, any stain) to that standard. The pipeline must be **adaptive** so it can handle new scanners, labs, and datasets without retraining.

---

## 1. What “chosen standard” means

We fix three things **once** and reuse them everywhere:

| Dimension | Chosen standard | Purpose |
|-----------|-----------------|---------|
| **Stain appearance** | Reference stain (one or a few representative images) | So “any stain” is mapped to the same H&E look the model was trained on. |
| **Resolution / size** | Reference resolution (e.g. 0.5 µm/px) + fixed input size (e.g. 96×96 or 224×224) | So “any image” (different magnification/size) becomes a fixed grid the model expects. |
| **Value range** | e.g. [0, 1] or fixed mean/std | So pixel statistics are consistent (no scanner A vs B intensity drift). |

Training: all data (e.g. PCam) is preprocessed to this standard.  
Deployment: any new image (e.g. CAMELYON17, new lab) is preprocessed with the **same** standard. No retraining; the pipeline is **portable**.

---

## 2. Pipeline steps (modular, in order)

A single pipeline that works for any input image:

```
Input image (any size, any stain, any scanner)
    → [Optional] Quality / artifact check (mask or exclude)
    → Stain normalization (map to reference stain)     ← “any stain”
    → Resolution/size standardization                ← “any image”
    → Value normalization (to chosen range)          ← “any dataset”
    → Output: image in our standard space, ready for the model
```

Each step is **adaptive**: it uses the image (and optionally metadata) to decide how to transform it, but the **target** is always our fixed standard.

---

## 3. Step 1: Stain normalization (“any stain”)

**Idea:** Estimate the stain (e.g. H&E) from the current image and map it to a **reference stain**. Same math for every image; only the source changes.

**Why we need it (in plain terms):** Histology slides are stained with dyes (e.g. hematoxylin = purple/blue, eosin = pink). Different hospitals, scanners, and protocols give different shades—one lab’s “purple” might be darker or more blue than another’s. The model would see the same tissue as “different” just because of color. Stain normalization asks: “What are the two main dye colours in this image, and how much of each?” Then it **recolors** the image so that those dyes look like our **reference** (e.g. one chosen “gold standard” slide). After that, every image is in the same “colour world,” and the model only has to learn tissue appearance, not scanner differences.

**Why not just turn them all into greyscale?**  
In H&E, **colour carries meaning**. Purple/blue = hematoxylin (nuclei, some structures); pink/red = eosin (cytoplasm, stroma). Pathologists—and good models—use that distinction to tell cell types and tissue apart. If we convert to greyscale, different colours can become similar grey levels (e.g. dark purple and dark pink both look dark grey), so we **lose the “which stain is this?” information**. Stain normalization keeps that information (we still have two “channels” of meaning) and only makes the *shade* of purple and pink consistent. So: greyscale would simplify the pipeline but throw away useful, biology-relevant signal; for tasks like metastasis detection we usually want to keep colour and just standardise it.

---

### 3.1 Methods explained (layman-friendly)

#### **Macenko**

- **In one sentence:** It figures out the two main stain colours (e.g. purple and pink) in the image and remaps them so they match a reference image.
- **How to think about it:** The method treats the image as a mix of two “ingredients” (the two dyes). It uses maths (optical density + PCA) to separate “how much purple” and “how much pink” at each pixel, then says: “In our reference, this amount of purple and pink looks like *this* RGB colour.” It rewrites every pixel using that rule. So you get stain-specific correction: the *type* of colour (purple vs pink) is preserved, but the *shade* is aligned to the reference.
- **When it works well:** Typical H&E slides with enough tissue and clear staining.
- **When it can fail:** Very pale or washed-out slides, almost no tissue, or weird staining (e.g. mostly one colour). Then the “two ingredients” are hard to estimate and the maths can break.

#### **Reinhard**

- **In one sentence:** It matches the overall brightness and colour spread of the image to the reference, like applying a single “photo filter” to the whole image.
- **How to think about it:** Instead of separating two dyes, it looks at the whole image in a colour space (LAB: lightness and two colour axes). It computes “average brightness” and “how spread out the colours are” for both your image and the reference, then shifts and scales your image so those two numbers match. So the image gets one global adjustment: brighter/darker and more/less saturated to match the reference. It does *not* explicitly model “this is hematoxylin, this is eosin”—it just makes the overall look similar.
- **When it works well:** Robust; works even when there’s little structure or when Macenko fails (e.g. very faint stain, few pixels). Good fallback.
- **Trade-off:** Less “stain-specific.” Two slides might end up with similar overall colour stats but still have different purple/pink balance. For strict H&E matching, Macenko or Vahadane are usually better; for “something reasonable when others fail,” Reinhard is ideal.

#### **Vahadane**

- **In one sentence:** Like Macenko, it separates the two stains and remaps to a reference, but uses a different mathematical recipe (sparse NMF in optical density) that often preserves fine detail and structure a bit better.
- **How to think about it:** Same goal as Macenko—find the two dye components and map them to the reference—but the way it decomposes “this pixel = how much dye A + how much dye B” is different (non-negative matrix factorisation, with sparsity). In practice that can give cleaner separation and less noise in the normalised image, so textures and small structures can look a bit more natural.
- **When it works well:** When you want high-quality, stain-specific normalization and are willing to spend a bit more compute.
- **Trade-off:** Slower and more parameters than Macenko; still needs enough tissue and reasonable staining. Not always available in every library (e.g. staintools has it).

#### **Structure-preserving colour normalization (SPCN)**

- **In one sentence:** A more advanced method that tries to preserve both the *structure* (edges, textures) and the *colour* of the tissue when matching to the reference.
- **How to think about it:** Some methods can slightly blur or alter fine detail when changing colours. SPCN is designed to change the colour appearance while keeping structures (e.g. cell boundaries, nuclei) as intact as possible. So you get reference-like colours without losing detail.
- **When it works well:** When you need the highest visual quality and have implementation available.
- **Trade-off:** More complex and less commonly used in off-the-shelf pipelines; more of an option for later or for comparison.

---

### 3.2 Methods (technical summary)

| Method | How it works (technical) | Pros | Cons |
|--------|---------------------------|------|------|
| **Macenko** | OD space, PCA on stain vectors, fit to reference | Widely used, good for H&E | Can fail on very faint/atypical stains; needs enough tissue |
| **Reinhard** | Match mean and std of LAB to reference | Simple, robust to lack of structure | Less “stain-specific”; more global color transfer |
| **Vahadane** (staintools) | Sparse NMF in OD space | Often better structure preservation | Slower, more parameters |
| **Structure-preserving (SPCN)** | Combines structure + color | High quality | More complex |

**Recommendation:** Start with **Macenko** (you already use it) as the default; add **Reinhard** as a **fallback** when Macenko fails (e.g. too little tissue, degenerate stain matrix). Optionally support **Vahadane** for quality comparison. All are **adaptive**: they take “this image” and “reference” and produce “this image in reference stain.”

### 3.3 Making it adaptive to any stain

- **Input:** Any RGB image (patch or tile).
- **Reference:** Fixed set of reference image(s) from our “chosen standard” (e.g. from PCam or a public standard slide). We **fit the normalizer once** on the reference and **save the fitted normalizer** (or the reference stain matrix/concentration stats).
- **Per image:**  
  - Convert to optical density (OD).  
  - Estimate stain matrix (and concentrations) from this image.  
  - Map to reference stain (using the saved reference).  
  - Convert back to RGB.  
- So: **any** H&E (different lab, scanner, protocol) is mapped to the **same** reference H&E. No need to retrain when the stain changes.

### 3.4 Robustness (important for “any image”)

- **Failure cases:** Very pale stain, almost no tissue, heavy artifacts, non-H&E.  
- **Fallbacks:**  
  - If stain estimation fails (e.g. matrix singular, too few pixels): use **Reinhard** (global color match) or **per-image percentile/contrast stretch** so we still feed a consistent value range.  
  - Optional: **quality gate** before stain norm (e.g. “skip stain norm if tissue % &lt; X or blur &gt; Y”) and flag for exclusion or use fallback.  
- **Reference choice:** Use one or several **good** patches (enough tissue, clear stain). Optionally average stain matrices from multiple reference patches to get a **robust reference** (innovation angle). See below for how to pick them.

### 3.5 Picking reference patch(es): how and based on what

Stain normalization (Macenko, Vahadane, Reinhard) maps every image to a **reference** stain. The reference defines the “target” look (purple and pink shades). Picking a good reference is the first step.

**What we need from a reference patch**

1. **Enough tissue** — The patch must have a sufficient amount of stained tissue (not mostly white background or empty). Methods like Macenko estimate two stain vectors from the pixel distribution in optical density (OD) space; if most pixels are background or there is too little variation, the stain matrix is **degenerate or unstable** (e.g. singular, or both vectors point in the same direction). So we exclude patches that are solid-color, very low tissue %, or mostly white/black.
2. **Representative staining** — The reference should look like “typical” H&E: visible purple/blue (hematoxylin) and pink/red (eosin). We do **not** want an outlier: e.g. an unusually pale slide, a heavily faded one, or a slide with dominant single stain. Normalizing everyone to an outlier would push the whole dataset toward that look. So the reference should be **in the middle** of the staining distribution (or from a chosen “gold standard” slide).
3. **Technically valid** — For Macenko/Vahadane we need a patch on which the method **succeeds** (non-singular stain matrix, reasonable stain vectors). So after candidate selection we can **try** fitting the normalizer on each candidate and keep one that doesn’t fail.

**How to pick (practical steps)**

- **Step 1 — Filter by quality:** From your dataset (e.g. PCam train), exclude patches that we would anyway drop: solid-color (very low gray std), very high black/white ratio, very low tissue % (e.g. below our combined threshold 0.35). What remains are patches with enough content and variation.
- **Step 2 — Avoid extremes:** Optionally exclude patches that are **stain outliers**. For example, compute a simple stain proxy per patch (e.g. mean R, G, B, or mean H and E from a simple color-deconvolution), then drop patches in the bottom/top 5–10% of those stats so the reference isn’t the palest or darkest. Alternatively, pick a patch whose stain stats are near the **median** of the dataset.
- **Step 3 — Choose one or a few:**  
  - **Single reference:** Pick one patch (e.g. the one at the median of tissue % and median intensity, or the first valid patch after filtering). Fit Macenko (and save the reference stain matrix) on that patch. Simple and reproducible.  
  - **Multiple references (robust):** Pick a small set of patches (e.g. 5–20) from different slides or different parts of the dataset. Fit the normalizer on each, then **average** the reference stain matrices (or take the median per matrix element), or fit once on the **concatenation** of all reference patches. The result is a single “average” reference that is less sensitive to one bad patch.
- **Step 4 — Validate:** Run Macenko (and Reinhard if used) on the chosen reference patch; ensure it doesn’t fail. Optionally apply the fitted normalizer to a few other patches and visually check that normalized images look plausible (no strong color cast or collapse).

**Criteria summary**

| Criterion | Why |
|-----------|-----|
| High enough tissue % (e.g. above 0.35) | So there are enough stained pixels to estimate stain vectors; avoids degenerate matrices. |
| Not solid-color (gray std above threshold) | Same reason; need variation. |
| Not mostly white or black | White/black don’t carry stain information. |
| Representative / median-like stain | So we normalize toward “typical” H&E, not an outlier. |
| Macenko (or chosen method) succeeds on it | So the reference is technically usable. |

**For PCam specifically:** Use the same quality filters we defined (solid-color, high black, low tissue combined). From the remaining patches, pick one (or several) with tissue % in the middle range and, if available, median-like RGB or H/E stats; or pick at random among the filtered set and validate that Macenko fits. Save the chosen reference patch index (or the fitted normalizer / reference matrix) so the pipeline is reproducible.

---

## 4. Step 2: Resolution and size (“any image”)

**Idea:** We define a **reference resolution** (e.g. 0.5 µm per pixel) and **input size** (e.g. 96×96 for baseline, 224×224 for Virchow2). Every image is brought to that resolution and then to that size.

**Why we can’t “compute” the original resolution from the image alone:**  
Physical resolution (e.g. “one pixel = 0.5 µm”) is **not stored in the pixel values**. A 96×96 patch is just 9,216 numbers; it doesn’t say whether that patch represents 96 µm × 96 µm of tissue or 24 µm × 24 µm. The same pixel grid could come from a low-magnification scan (each pixel = more tissue) or a high-magnification scan (each pixel = less tissue). That scale is determined by **how the slide was acquired** (scanner, objective lens, zoom)—i.e. by metadata (MPP, magnification), not by the image content. So we can’t derive “microns per pixel” from the pixels themselves; we need either metadata or a documented assumption (e.g. “this dataset was scanned at 0.5 µm/px”), or an **estimate from content** (below).

### 4.1 When metadata exists (e.g. MPP, magnification)

- If the image has **resolution metadata** (µm/px or MPP):  
  - Compute scale factor = (image resolution) / (reference resolution).  
  - Resize image so that 1 pixel = reference resolution (e.g. 0.5 µm/px).  
  - Then crop or pad to exact **input size** (e.g. 96×96 or 224×224).  
- Result: “Any image” from any scanner (with metadata) is mapped to the same **biological scale** and same **pixel grid**.

### 4.1b Documented dataset resolution (e.g. PCam) — no nucleus estimation needed

When the **dataset documents** resolution (e.g. in a README or paper), use that value; **no nucleus-based estimation** is needed.

**PCam:** The PCam README (in `pcam-master/README.md`) states that source slides were acquired with a **40× objective (0.243 µm/px)** and that patches are **undersampled at 10×** to increase field of view. So PCam’s effective resolution is **0.243 × 10 = 2.43 µm/px**. The meta CSV files only describe which Camelyon16 slide each patch came from; they do not add per-patch resolution. For PCam we therefore use **2.43 µm/px** as the documented resolution.

**Pipeline order:** Use **metadata (e.g. MPP) if present** → else **documented dataset resolution (e.g. README)** → else **estimate from content (nucleus)** → else **fixed default**.

### 4.2 When metadata and documentation are missing: estimate from content, then fallback

When there is no per-image metadata **and** no dataset-level documentation (e.g. README) for resolution, we **can** estimate resolution from the image content (e.g. nucleus size) instead of assuming a single default.

**How it works (layman):**  
We use a known physical size in tissue—e.g. “average nucleus diameter is roughly 7 µm” for many cell types—and measure that same thing in the image in **pixels**. If we detect nuclei and find their typical size is, say, 14 pixels, then we get: 7 µm ≈ 14 px → **~0.5 µm/px**. We then use that µm/px to resize the patch to our reference resolution (e.g. 0.5 µm/px) and to the target size (96×96 or 224×224).

**What we need:**  
- A way to get “size in pixels” of a known structure (usually nuclei). This does **not** have to be a heavy detector like YOLO. Options from simple to more involved:
  - **Simple (prefer first):** Separate the hematoxylin (purple) channel (e.g. with stain separation or a red/blue channel), threshold to get “nucleus-like” regions, then run **blob detection** or connected components; measure the diameter (or area) of each blob and take the median. Libraries like OpenCV (`cv2.connectedComponentsWithStats`) or scikit-image (`skimage.measure.regionprops`) can do this. No deep learning.
  - **Medium:** Classic **circle/ellipse detection** (e.g. Hough for circles) on the hematoxylin channel, or a small **segmentation U-Net** trained only to segment nuclei (many public models exist).
  - **Heavy (optional):** Full object-detection (e.g. YOLO) or instance segmentation (e.g. HoVer-Net, Cellpose) if we already use them for other tasks. Overkill for “median nucleus size in pixels” unless we want maximum robustness.
- Then: median (or mean) diameter in pixels + a **prior** for physical size (e.g. 7 µm) → µm/px.  
- A **prior** for the physical size (e.g. 7 µm for nucleus diameter; can be a constant or tissue-specific).  
- Optional: **sanity bounds** on µm/px (e.g. 0.1–2 µm/px) so we reject crazy estimates (e.g. from empty or artifact patches).

**When it works well:**  
Patches with several visible nuclei and reasonable staining. Same idea can be applied per image, or once per dataset (e.g. sample N patches, estimate µm/px from each, take median) so all images in that dataset get one estimated resolution.

**When it fails / fallback:**  
- Too few nuclei, or detection fails (e.g. blur, no tissue, non-H&E).  
- Estimate outside sanity bounds (e.g. 0.05 µm/px or 5 µm/px → likely wrong).  
- Mixed content (e.g. lymphocytes vs tumour cells have different sizes; prior may not match).  

In those cases we **fall back** to a **dataset-level or global default** (e.g. “assume 0.5 µm/px”) and optionally log a warning. So the pipeline is: **metadata if present → else content-based estimate (with sanity check) → else fixed default.**

**Recommendation:**  
For datasets like PCam, use the documented resolution (README: 2.43 µm/px) and skip nucleus estimation. When both metadata and dataset documentation are missing: try to estimate µm/px from content (nucleus size or similar); if the estimate is plausible, use it; otherwise use a documented default. That way we don’t rely on metadata only, and we still have a safe fallback.

**Generalizable vs dataset-specific settings:** A nucleus-based estimator can be tuned per dataset (e.g. "best" channel and area bounds that match a known resolution). For a **general pipeline** that works on **any dataset without metadata**, the following should be generalizable: (1) **channel** options (e.g. inverted green, blue, purple for H&E), (2) **threshold** type (Otsu or percentile), (3) **prior** (e.g. 7 µm nucleus diameter), (4) **scale-invariant** blob filtering. In contrast, **fixed pixel area bounds** (e.g. min_area=3, max_area=40) are **resolution-dependent**: they match one scale (e.g. PCam at 2.43 µm/px) and will fail on others. For "any dataset," use a **scale-invariant** filter instead: e.g. keep blobs whose area is between the 5th and 95th percentile of all blob areas in the patch. The notebook `test_nucleus_resolution_estimation.ipynb` implements both: PCam-optimized (fixed area) for validation, and a general (percentile-of-area) variant for deployment on unknown datasets.

### 4.3 Single pipeline for multiple input sizes

- **Option A:** One standard resolution + one input size (e.g. 0.5 µm/px, 96×96). All models (baseline CNN, Virchow2) get the same resolution; for Virchow2 we **upscale** 96→224 in a fixed way (bicubic) so the “content” is still at 0.5 µm/px.  
- **Option B:** Two standards (e.g. 96×96 for baseline, 224×224 for Virchow2) both at the same reference resolution; pipeline has an output size parameter.  
- Recommendation: **one reference resolution**, **one or two output sizes** (96, 224), both produced by the same pipeline so “any image” is standardized the same way.

### 4.4 Resizing 96→224 for Virchow2: how it affects the data

Virchow2 (and most ImageNet-style backbones) expect a fixed input size, usually **224×224**. PCam patches are **96×96**. So we **upscale** 96→224 before feeding the model. We use **bicubic** interpolation (chosen over bilinear for slightly sharper edges and fine structure). Here's how that affects the data and what to keep in mind.

**What actually changes**

- **Field of view:** Unchanged. The same tissue region (e.g. 96×96 at 0.5 µm/px ≈ 48 µm × 48 µm) is still represented; we only change the number of pixels.
- **True resolution:** Unchanged. We do not gain real detail. Upscaling (**bicubic**, our chosen method) **interpolates** between existing pixels; it does not add new information. Biologically we still have "one patch at 0.5 µm/px," just drawn on a 224×224 grid.
- **What the model sees:** A smoother, larger version of the same patch. Edges and textures can look slightly softer depending on the interpolation method. The model's receptive field and patch grid (e.g. 14×14 tokens for a ViT) now operate on this upscaled image, so spatial structure is preserved but at a different pixel density.

**Why we resize (and don't feed 96×96 directly)**

- The backbone was pretrained on 224×224 (or similar) images. Its first layers and patch embedding assume that spatial scale. Feeding 96×96 would change the effective receptive field and the meaning of each patch token; performance and transfer are typically better when we match the expected input size and then let the head adapt to the task.
- So the resize is mainly for **architectural compatibility**, not for increasing resolution.

**Practical impact**

- **Information:** No new biological information; same content, same diagnostic label. The head trained on top of the backbone learns from embeddings of this upscaled view.
- **Possible downsides:** Slight blur or interpolation artifacts; in principle the model could be sensitive to them, but in practice this setup is standard and works well.
- **Consistency:** Use **bicubic** resize and the same normalization (e.g. ImageNet mean/std if the backbone was pretrained that way) for training and inference so the distribution the backbone sees is stable.

**Summary**

Resizing 96→224 does **not** change the physical scale or the amount of real detail in the patch; it only changes the **representation** to match the model's expected input size. The data (field of view, label, biological meaning) are the same; we use a fixed **bicubic** interpolation step so that Virchow2 can be used as a fixed feature extractor or fine-tuned on PCam without changing the backbone's input interface.

---

## 5. Step 3: Value normalization (“any dataset”)

**Idea:** After stain and size, we put pixel values in a **fixed range** so different scanners/datasets don’t shift the distribution.

### 5.1 Simple and portable

- **Option 1:** `x = x / 255.0` → [0, 1]. No dataset-specific stats; works for any image.  
- **Option 2:** Reference mean and std (e.g. computed once on our reference set). Then `x = (x - mean) / std`. Same mean/std for training and deployment.  
- **Option 3:** Per-image rescale to [0,1] (e.g. percentile clip 1–99 then min-max). More adaptive to outliers but can change relative contrast between images; use only as fallback or for robustness studies.

**Recommendation:** Standard = **[0, 1]** (divide by 255) for simplicity and portability. Optionally add **reference mean/std** (Option 2) as an alternative standard and compare.



## 6. Optional: quality and artifacts

To make the pipeline robust to “any” real-world dataset:

- **Before** or **after** stain/size/value:  
  - **Blur / focus:** Flag or exclude very blurry patches (e.g. Laplacian variance below threshold).  
  - **Tissue content:** Flag or exclude patches with very little tissue. **Pipeline order:** (1) Remove **solid-color** patches first. (2) Then apply **high black** (pure black patches). (3) **Low tissue** is determined by our **chosen mix** (saturation OR local variance, with contrast-normalized Otsu on edge cases); we **settled on threshold 0.35** (see below).  
  - **Artifacts:** Folds, pen marks, bubbles — optionally detect and mask or exclude.  
- These steps are **adaptive** (per image) and **optional** (can be turned on for training and deployment so both see the same rules).

### How we currently determine tissue content

- **Method:** For each patch we compute **grayscale** (mean of R,G,B), run **Otsu’s method** on that grayscale image to get a single threshold, then define **tissue %** = fraction of pixels **darker** than that threshold. So we assume: “dark” = tissue, “bright” = background (typical for H&E).
- **Why it can fail:** Otsu only separates **two intensity groups**. If the patch is **pale overall** (low opacity / faint stain), almost all pixels are bright and there may be no clear dark peak—so the “tissue” (dark) fraction can be tiny or the split can be arbitrary. Conversely, a patch that is mostly tissue but lightly stained can be classified as “mostly background” because we’re using **intensity**, not “is there structure or color here?” So the current measure is **sensitive to opacity** (stain strength / overall brightness).

### Detecting tissue content regardless of opacity — options

Goal: estimate “how much of this patch is tissue” in a way that doesn’t depend on how pale or dark the stain is. Below are options that are more **opacity-invariant**.

1. **Saturation / color-based (HSV)**  
   Background is often **white or light gray** (no hue, low saturation). Tissue in H&E has **color** (purple, pink). So: convert patch to HSV (or similar); **tissue %** = fraction of pixels with **saturation above a threshold** (e.g. S &gt; 0.1 or 0.15). Pale tissue still has some saturation; white background has none. Pros: simple, no training; cons: threshold may need tuning, and very pale or washed-out tissue can have low saturation.

2. **Local variance / texture**  
   Tissue has **texture** (nuclei, cytoplasm, stroma); background is **flat**. For each pixel (or each small block), compute **local variance** (e.g. in a 5×5 or 11×11 window). **Tissue %** = fraction of the patch where local variance is above a threshold (or above a percentile of the patch’s own variance distribution). This depends on **structure**, not global brightness, so it’s largely **opacity-invariant**. Pros: robust to pale/dark; cons: window size and threshold need choosing; very blurry tissue may have low variance.

3. **Edge density**  
   Tissue has **edges** (cell boundaries, nuclei); background doesn’t. Compute an edge map (e.g. Sobel or Laplacian magnitude), then **tissue %** = fraction of pixels (or of area) where edge strength is above a threshold, or “fraction of the patch covered by regions with enough edges.” Again this is about **structure**, not absolute intensity. Pros: opacity-invariant; cons: sensitive to blur and noise; thresholds need tuning.

4. **Contrast-normalized Otsu**  
   Keep Otsu but run it on a **normalized** version of the patch so the split is based on **relative** dark vs bright inside the patch, not absolute values. For example: stretch the patch’s intensity to a fixed range (e.g. 5th–95th percentile → [0, 1]), then run Otsu. So “dark” and “bright” are defined relative to the patch’s own range. Pros: minimal change to current pipeline; can help when the main issue is global brightness. Cons: if the whole patch is nearly one shade (very pale or very dark), stretching may not create a meaningful bimodal histogram.

5. **Stain deconvolution (H&E)**  
   Use **color deconvolution** to get hematoxylin and eosin “amount” per pixel. Background has **negligible stain**; tissue has at least one stain present. **Tissue %** = fraction of pixels where (e.g.) max(H, E) or H+E is above a low threshold. This measures “amount of stain” rather than raw intensity, so it can be more stable across opacity. Pros: well-suited to H&E; cons: needs a stain matrix and is H&E-specific.

### Which method is best? Can we use a mix?

**Best single method (opacity-invariant, general):** **Local variance / texture.** It doesn’t rely on color or stain type—only on “is there structure?” So it works across stain strength, pale/dark slides, and even non-H&E if you only need “content vs empty.” Drawback: very blurry or out-of-focus tissue can have low variance.

**Best single method for H&E (if you can use a library):** **Stain deconvolution.** It directly measures “amount of stain” and is the most interpretable and robust to opacity. Cost: dependency (e.g. `staintools`) and H&E-specific.

**Practical choice: use a mix.** Combining two or more metrics is usually better than one:

- **Recommended mix:** **Saturation + local variance.**  
  - Define a pixel as “tissue” if **either** saturation is above a low threshold **or** local variance is above a threshold (or above a percentile of that patch’s variance).  
  - **Tissue %** = fraction of pixels that satisfy at least one condition.  
  - Why: pale but textured tissue is caught by variance; colored but flat regions (e.g. stain blobs) by saturation; white background fails both. You get opacity-invariant behaviour without stain deconvolution.

- **Optional third ingredient:** **Contrast-normalized Otsu** as a fallback or tie-breaker when both saturation and variance are borderline (e.g. use Otsu-based tissue % only for patches where the saturation+variance estimate is in an uncertain range).

- **Implementation note:** For the mix, compute per patch: `tissue_pct_sat` (fraction of pixels with S > threshold), `tissue_pct_var` (fraction of pixels with local variance > threshold or above patch percentile). Then e.g. **tissue_pct = max(tissue_pct_sat, tissue_pct_var)** or **tissue_pct = fraction of pixels where (saturated OR high_variance)**. The “OR” version is slightly more conservative (avoids double-counting the same pixel in two different ways) and is the one recommended above.


### Chosen method and threshold (what we use)

We **use the mix** and have **settled on threshold 0.35** for low-tissue removal.

**Mix of methods:**
(1) **Saturation** — fraction of pixels with HSV saturation above a low threshold (e.g. S > 0.12); catches colored (tissue) vs white/gray (background).
(2) **Local variance** — fraction of pixels whose local variance (e.g. 11×11 window) is above a threshold (e.g. 0.003); catches texture (tissue) vs flat (background).
(3) **Combined:** a pixel counts as "tissue" if **either** condition holds; **tissue % final** = fraction of such pixels in the patch.
(4) **Edge cases:** when this combined % falls in a middle range (e.g. 0.12–0.45), we also compute **contrast-normalized Otsu** (5th–95th percentile stretch, then Otsu; tissue = fraction below threshold) and take the **max** of combined and Otsu-norm, so borderline patches are not undercounted.

**Threshold:** We **remove** a patch if **tissue % final < 0.35**. This value was chosen to include more borderline low-tissue patches (relaxed from 0.20); it is a general default, not fit to PCam labels. See "Does this work for other datasets?" below.

### Otsu and “adaptiveness” (plain language)

- **What Otsu does:** It looks at the **histogram** of brightness in the patch (how many pixels are dark, how many medium, how many bright) and finds a **single cutoff** that best splits the image into “two groups”—e.g. “dark stuff” vs “bright stuff.” It doesn’t use a fixed rule like “everything darker than 0.5 is tissue”; it asks “where does *this* image naturally divide?” So a pale slide and a dark slide each get their own cutoff. That’s what we mean by **adaptive**: the rule adjusts to each image.
- **Why that helps when it works:** In H&E, tissue is usually **darker** (purple/pink) and background **brighter** (white/light). Otsu finds that split for that patch. We then say “tissue” = the darker side and set **tissue %** = fraction of pixels on that side. So a low-saturation but textured patch can still have high tissue % (lots of “darker” pixels), while a solid white or solid purple patch either has no real split or gets filtered earlier as solid-color. **Limitation:** when the whole patch is pale (low opacity), there may be no clear dark peak, so tissue % becomes unreliable—hence the alternatives above.

### Does this work for other datasets, or is it PCam-specific?

- **Solid-color filter (low gray std):** **General.** Any digital image can have “almost no variation” (solid or near-solid). The threshold (e.g. std &lt; 0.04 on [0,1]) is a **generic** “no texture” criterion; it doesn’t assume PCam resolution or stain. It works for other histology datasets, other modalities, and other image sizes as long as the same rule is applied.
- **High white / high black ratio:** **General.** “Fraction of pixels very bright or very dark” is a simple, dataset-agnostic rule. The exact thresholds (e.g. 0.75) can be tuned per use case but the *idea* (exclude mostly white or mostly black patches) applies to any dataset.
- **Otsu-based tissue %:** **General in method, H&E-biased in interpretation.** The **algorithm** (Otsu threshold on grayscale, then “tissue % = fraction on the dark side”) is not PCam-specific—it runs on any image. The **meaning** “dark = tissue, bright = background” is a **convention** that fits H&E (and many bright-field histology slides). For other H&E or similar datasets it transfers directly. For other modalities (e.g. fluorescence, IHC with different contrast), “tissue” might need a different rule (e.g. “above threshold” instead of “below,” or a different channel). So: **same pipeline step works on other datasets**; only the *interpretation* of which side is “tissue” may need to be checked for non–H&E-like images. We are **not** tying the logic to PCam patch size, scanner, or labels—no PCam-specific parameters.

- **Low-tissue threshold (e.g. 0.35):** **General as a default.** The value (e.g. “remove if tissue % final &lt; 0.35”) is a **semantic** choice: “how little tissue is too little?” It was **not** fit to PCam labels or task performance—we only relaxed from 0.20 to 0.35 to include more borderline cases. So it is **not** a PCam-tuned parameter. The same cutoff can be used as a **reasonable default** for other H&E (or similar) datasets: patches with &lt; 35% tissue-like pixels are excluded. If a new dataset has a very different distribution (e.g. much paler slides), you can tune within a range (e.g. 0.25–0.40) without changing the pipeline; the **logic** (saturation OR variance, Otsu on edge cases) remains general.

---

## 7. What we include in the preprocessing (checklist)

| Step | Include? | Adaptive? | Purpose |
|------|----------|------------|---------|
| Stain normalization (Macenko + Reinhard fallback) | **Yes** | Yes (per-image stain → fixed reference) | Any stain → our standard stain |
| Resolution/size to reference µm/px + fixed size | **Yes** | Yes (use metadata or default) | Any image → same scale and grid |
| Value normalization ([0,1] or reference mean/std) | **Yes** | No (same rule for all) | Any dataset → same value range |
| Quality/artifact checks (blur, tissue %, optional) | Optional | Yes (per image) | Robustness on any dataset |
| Deduplication / anomaly exclusion | Optional (separate) | N/A | Clean training set; not “per-image” standardization |

---

## 8. Is there a way to make it adaptive to “anything”?

- **Any stain:** Yes — stain normalization estimates stain from each image and maps to our reference; no retraining.  
- **Any image size/resolution:** Yes — resolution step (with metadata or default) + resize/crop/pad to fixed size.  
- **Any dataset:** Yes — same reference stain, same reference resolution/size, same value range; new datasets are just new inputs through the same pipeline.  
- **Truly “any” modality (e.g. IHC, different stain types):** Only within the same stain model (e.g. H&E). For other stain types you’d need a different reference or method (research frontier).

**Limits:**  
- Pipeline assumes **H&E-like** (two-stain) for Macenko/Vahadane; Reinhard is more agnostic.  
- Reference must be **fixed and saved** (e.g. in a config or serialized normalizer) so deployment uses the same standard as training.

---

## 9. Innovation angles (for supervisor discussion)

1. **Robust reference:** Use multiple reference patches (e.g. from different slides) and average stain matrices or pick the median; reduces sensitivity to a single bad reference.  
2. **Automatic fallback:** Try Macenko → if it fails (numerical or quality check), fall back to Reinhard or to “no stain norm + value norm only” and flag the sample.  
3. **One pipeline, multiple targets:** Same code path for PCam and CAMELYON17; only config (paths, optional resolution metadata) changes. Publish the reference and config so others can bring their data to the same standard.  
4. **Resolution-aware deployment:** When WSIs have MPP, compute patches at exact reference resolution so train (PCam at 0.5 µm/px) and test (CAMELYON17 at 0.5 µm/px) are aligned.  
5. **Validation:** Report train/val/test metrics with and without the full pipeline; on external data (e.g. CAMELYON17), report with “our preprocessing” vs “no preprocessing” to show the gain of standardization.

---

## 10. Implementation outline

1. **Define and save “chosen standard”:**
   - Reference image(s) for stain (e.g. one or more patches from PCam or public standard).
   - Reference resolution (e.g. 0.5 µm/px) and input size(s) (96, 224).
   - Value rule: [0, 1] (and optionally reference mean/std).

2. **Implement pipeline (per image):**
   - Stain: fit normalizer on reference once; for each image, apply transform (Macenko with Reinhard fallback).
   - Resolution/size: scale by (image_res / ref_res), then crop/pad to 96 or 224.
   - Value: divide by 255 (and optional clip).

3. **Integrate with training:**  
   - Training data (e.g. PCam) is preprocessed through this pipeline before or inside the data loader.  
   - Same pipeline (same reference, same config) is used at inference for any new dataset.

4. **Document and version:**  
   - Store reference patches or fitted normalizer and config (resolution, sizes) in the repo or artifact store so the pipeline is reproducible and portable.

This gives you a **generalizable preprocessing pipeline** that brings any dataset to your chosen standard and keeps the **model + preprocessing** together as the deployable, generalizable unit.
