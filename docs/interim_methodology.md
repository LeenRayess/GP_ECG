# Methodology

## 1. Dataset

This work uses the PatchCamelyon (PCam) benchmark, which is a patch-level binary classification dataset constructed from CAMELYON lymph-node whole-slide images. Each sample is an RGB patch of size \(96\times96\) and is labeled according to whether metastatic tissue is present in the central region of the patch. In formal terms, for each patch \(X_i\in\mathbb{R}^{96\times96\times3}\), the target \(y_i\in\{0,1\}\) indicates absence (\(0\)) or presence (\(1\)) of metastasis. The dataset is provided in a fixed train/validation/test split through HDF5 containers.

Although PCam is curated, two practical issues remain relevant for robust model development. First, histopathology patches can contain low-information or artifactual content (e.g., near-uniform backgrounds, dark artifacts, low tissue occupancy), and these samples can bias optimization and inflate apparent class imbalance effects during training. Second, in our own data audit we observed exact pixel-level duplicate patches within candidate sets; therefore, we performed a dedicated deduplication stage before quality filtering. Public PCam documentation describes patch extraction and sampling strategy but does not explicitly specify the mechanism that yields exact duplicates, so we treat deduplication as an empirical data-quality safeguard rather than attributing it to a single confirmed source. For this reason, preprocessing was treated as a first-class component of the experimental design rather than a cosmetic step.

## 2. Data Handling and Preprocessing

### 2.1 Motivation and high-level structure

The preprocessing pipeline was designed to improve training signal quality while preserving biologically meaningful morphology and stain variation. The pipeline consists of candidate restriction by deduplication, quality filtering, stain normalization with safety fallbacks, and final value normalization. Outputs are stored in a standardized format that preserves the exact subset of accepted patches and the exact decisions taken during preprocessing, enabling reproducibility and non-destructive comparison across preprocessing variants.

Let \(X\) denote a patch and \(u\) index pixels. For channel \(c\in\{R,G,B\}\), the patch mean is
\[
\mu_c(X)=\frac{1}{HW}\sum_u X_{u,c}.
\]
When input is uint8, patches are mapped to \([0,1]\) by
\[
X^{(01)}=\mathrm{clip}\!\left(\frac{X}{255},0,1\right),
\]
and left unchanged (with clipping) if already in \([0,1]\).

### 2.2 Deduplication-aware candidate restriction

Before quality filtering, we optionally restrict processing to a deduplicated candidate subset. In this work, this was motivated by empirically detected exact duplicates (pixel-identical patches), not only by visual similarity. When exact duplicates are present, stochastic gradient descent repeatedly sees the same observation under different sample indices, reducing effective sample diversity and potentially biasing optimization dynamics and uncertainty estimates.

Operationally, for each split \(s\), the full index set \(\mathcal{I}_s\) is replaced by a retained candidate subset \(\mathcal{I}_s^{\text{dedup}}\subseteq\mathcal{I}_s\). All subsequent filtering and normalization are performed only on this subset. This does not alter labels directly; it alters which examples are admitted to downstream preprocessing and training.

### 2.3 Quality filtering

Quality filtering removes patches that are unlikely to provide meaningful histologic signal. A patch is rejected if it violates any of three tests: low grayscale variance (solid-color), excessive black occupancy, or low tissue occupancy.

We define grayscale intensity
\[
G_u=\frac{X_{u,R}+X_{u,G}+X_{u,B}}{3}.
\]
The solid-color criterion rejects patches with
\[
\mathrm{std}(G)<0.04.
\]
This threshold removes near-uniform patches where gradients are dominated by noise rather than morphology.

For black-content rejection, we compute
\[
r_{\text{black}}=\frac{\#\{(u,c):X_{u,c}\le 0.1\}}{3HW},
\]
and reject when \(r_{\text{black}}\ge 0.5\). Intuitively, this removes patches that are mostly dark background or acquisition artifacts.

For tissue occupancy, we combine chromatic and local-texture cues. Pixel-wise saturation is estimated as
\[
S_u=\frac{\max_c X_{u,c}-\min_c X_{u,c}}{\max_c X_{u,c}},
\]
with safe handling near zero denominators. Local grayscale variance is estimated using a windowed second-moment identity:
\[
V_u=\mathbb{E}[G^2]_u-\mathbb{E}[G]_u^2,
\]
where expectations are local uniform-filter averages over an \(11\times11\) neighborhood. We then define binary masks
\[
M_S(u)=\mathbf{1}[S_u>0.12],\qquad M_V(u)=\mathbf{1}[V_u>0.003],
\]
and a combined tissue fraction
\[
p_T=\frac{1}{HW}\sum_u (M_S(u)\lor M_V(u)).
\]

For borderline cases (\(0.12\le p_T\le0.45\)), we additionally apply Otsu thresholding on contrast-normalized grayscale to avoid false rejection of faint but real tissue. With
\[
\tilde{G}_u=\mathrm{clip}\!\left(\frac{G_u-p_5}{\max(p_{95}-p_5,\epsilon)},0,1\right),
\]
Otsu chooses
\[
t^*=\arg\max_t\left[w_0(t)w_1(t)\big(\mu_0(t)-\mu_1(t)\big)^2\right],
\]
and yields a secondary tissue estimate
\[
p_{\text{otsu}}=\frac{1}{HW}\sum_u \mathbf{1}[\tilde{G}_u<t^*].
\]
The final tissue estimate is
\[
p_{\text{final}}=
\begin{cases}
\max(p_T,p_{\text{otsu}}), & 0.12\le p_T\le0.45,\\
p_T, & \text{otherwise}.
\end{cases}
\]
A patch is rejected if \(p_{\text{final}}<0.35\).

### 2.4 Stain normalization: routing, fallback, and guardrails

After quality filtering, each retained patch undergoes stain normalization. We use Macenko and Reinhard normalizers with a data-driven routing rule based on blue dominance:
\[
b(X)=\frac{1}{HW}\sum_u \mathbf{1}[X_{u,B}>X_{u,R}].
\]
For each split, a threshold \(\tau_b\) is estimated as the 25th percentile of \(b(X)\) over sampled retained patches:
\[
\tau_b=Q_{0.25}(\{b(X_i)\}).
\]
If \(b(X)<\tau_b\), Reinhard is attempted first; otherwise Macenko is attempted first. If the primary method fails numerically, the alternative method is attempted. If both fail, luminosity-only standardization is used.

To prevent implausible outputs from entering the final dataset, we apply post-normalization guardrails. For normalized patch \(\hat{X}\), we compute
\[
\mu_R(\hat{X}),\quad
p_{\text{pink}}(\hat{X})=\frac{1}{HW}\sum_u\mathbf{1}[\hat{X}_{u,R}>\hat{X}_{u,B}],\quad
\bar{S}(\hat{X})=\frac{1}{HW}\sum_u S_u.
\]
If any condition
\[
\mu_R<0.08 \;\lor\; p_{\text{pink}}<0.01 \;\lor\; \bar{S}<0.03
\]
is true, the patch is replaced by luminosity-only output. This mechanism is intended as a conservative safety net for rare but severe normalization failures.

We also apply a split-wise purple-tail replacement: after normalization, we compute patch-level \(\mu_R\) and \(p_{\text{pink}}\), estimate 2nd-percentile thresholds
\[
t_R=Q_{0.02}(\{\mu_R\}),\qquad t_P=Q_{0.02}(\{p_{\text{pink}}\}),
\]
and replace any patch satisfying
\[
\mu_R\le t_R \;\lor\; p_{\text{pink}}\le t_P
\]
with luminosity-only output. This removes extreme low-red/low-pink outliers that otherwise remain visually implausible.

### 2.5 Multi-reference stain mode

In addition to single-reference normalization, we implemented a multi-reference mode to reduce dependence on one stain target. Three fixed train references are used, and patch routing is learned by \(k\)-means (\(k=4\)) on a seven-dimensional handcrafted feature vector:
\[
f(X)=\big[p_{\text{final}},\mu_R,\mu_G,\mu_B,b(X),p_{\text{pink}}(X),\bar{S}(X)\big].
\]
Features are z-scored
\[
z=\frac{f-\mu_f}{\sigma_f},
\]
then clustered via
\[
\min_{\{c_i\},\{\nu_j\}} \sum_i \|z_i-\nu_{c_i}\|_2^2.
\]

One cluster is identified as a legacy merge cluster via proximity to an anchor patch in feature space. Non-merge clusters are assigned to the three references by minimum-cost matching. For merge-cluster samples, the final reference is chosen by nearest mean RGB:
\[
j^*=\arg\min_{j\in\{1,2,3\}}\|\mu(X)-\mu(r_j)\|_1.
\]
This allows one ambiguous cluster to be distributed adaptively rather than forced to a single fixed target.

### 2.6 Output format and reproducibility

All final outputs are clipped to \([0,1]\):
\[
\hat{X}\leftarrow\mathrm{clip}(\hat{X},0,1),
\]
stored as float32 tensors in HDF5 format. Alongside the patch tensors, we store metadata that records filtering counts, thresholds, and normalization mode usage. Additional per-patch arrays record the chosen normalization path and any fallback decisions, enabling exact auditing of each preprocessing run.

## 3. Baseline CNN Training (Completed)

The baseline model is trained as a reference supervised pipeline. Inputs are normalized to \([0,1]\) at batch time by \(X/255\). The architecture is a compact convolutional network with increasing channel depth (32, 64, 128, 256), interleaved pooling, global average pooling, a 128-unit dense layer, dropout (\(p=0.5\)), and a sigmoid output neuron for binary prediction.

The model output is
\[
\hat{p}=\sigma(z)=\frac{1}{1+e^{-z}},
\]
and training minimizes binary cross-entropy
\[
\mathcal{L}_{\text{BCE}}=-\frac{1}{N}\sum_{i=1}^N\Big(y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\Big).
\]
Optimization uses Adam with learning rate \(10^{-3}\), batch size \(64\), and \(10\) epochs, with best-checkpoint selection by validation accuracy. Evaluation includes ROC-AUC, accuracy, and confusion matrices. With threshold \(\hat{y}=\mathbf{1}[\hat{p}\ge0.5]\), we compute standard confusion-derived metrics:
\[
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN},\quad
F1=\frac{2PR}{P+R}.
\]

## 4. Virchow2 Transfer Pipeline (Implemented; full experiment pending)

We use a transfer-learning pipeline with a frozen Virchow2 backbone and a trainable linear head. Because Virchow2 expects 224-pixel inputs, each preprocessed patch is resized from \(96\times96\) to \(224\times224\) using bicubic interpolation, then normalized using ImageNet channel statistics:
\[
X'_c=\frac{X_c-\mu_c}{\sigma_c},\qquad
\mu=(0.485,0.456,0.406),\ \sigma=(0.229,0.224,0.225).
\]

Let \(T\) denote output tokens from the backbone. The embedding used for classification concatenates class token and mean patch-token representation:
\[
e=\big[t_{\text{cls}} \,\|\, \overline{t}_{\text{patch}}\big]\in\mathbb{R}^{2560},
\]
followed by a linear head
\[
z=w^\top e+b.
\]
Training uses BCE-with-logits on \(z\), equivalent to sigmoid plus BCE but with improved numerical stability:
\[
\mathcal{L}=-\frac{1}{N}\sum_i\left[y_i\log\sigma(z_i)+(1-y_i)\log\big(1-\sigma(z_i)\big)\right].
\]

Only head parameters are updated; backbone gradients are disabled throughout. Default settings are 10 epochs, batch size 64, and head learning rate \(10^{-3}\). Checkpointing stores epoch state, optimizer state, and best validation accuracy for resumable training.

At the time of this interim report, the baseline CNN study is complete, while full comparative Virchow2 experiments across preprocessing variants remain pending due to current GPU memory and runtime constraints. The entire training pipeline is available for reruns and supports checkpoint-resume behavior; the remaining limitation is computational capacity rather than methodology design.
