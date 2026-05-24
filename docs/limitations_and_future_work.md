# Limitations and future work

This section states what the evidence cannot claim by design, what bounded the scope of optional experiments, and which directions follow once more compute or data become available. It does not repeat the Results and Discussion narrative.

## Limitations

### What the study can and cannot claim

The work uses retrospective public patch benchmarks, not prospectively collected clinical material or multi-reader studies. Findings are therefore about discrimination, calibration, and error behaviour on fixed test splits under published label definitions, not about regulatory approval, workflow integration, or patient outcomes.

Both PatchCamelyon and CAMELYON17 (WILDS) release 96×96 RGB tiles with a patch-level binary label determined by tumour in a central 32×32 pixel region; the outer field is context only. That task is well defined for benchmarking but is not a complete model of whole-slide search behaviour, ITC or micrometastasis classes, or how hospitals set operating points from prevalence and cost.

Bootstrap percentile intervals and paired permutation tests quantify uncertainty on the same held-out tensors they are computed from; they do not replace an independent prospectively collected test cohort.

The structured qualitative review uses small, prespecified exports from error and high-entropy buckets. It is illustrative and checklist-driven; it is not a census of every test patch and does not estimate full-test prevalence of morphological patterns.

### Practical scope bound by compute and runtime

The experimental programme was executed under finite GPU capacity, wall-clock limits on full-corpus training and evaluation, and the need to keep a single traceable pipeline end-to-end. In practice, that meant prioritising a prespecified preprocessing path (including the stain benchmark that justified Macenko for main runs), two model families on the same tensors, the bidirectional transfer layout, calibration fitting on source validation only, and the planned statistical and qualitative add-ons.

Within that envelope, many reasonable extensions (additional normalizers on full data, alternative foundation checkpoints, partial encoder fine-tuning, multi-seed variance, exhaustive threshold sweeps, or per-hospital disaggregation of CAMELYON17) would each multiply training and inference passes. They were not omitted for lack of scientific interest, but because they would have exceeded the time and hardware budget available for this thesis while still doing justice to the core comparison the examiners were asked to judge.

### Reproducibility and environment

Exact numeric reproduction depends on library versions, GPU nondeterminism, and the frozen public splits. The repository records scripts, manifests, and settings; reruns should agree within small numerical tolerances rather than demanding bit-identical logits across machines unless seeds and hardware are fully pinned.

---

## Future work

The items below are natural extensions of the same benchmarks and protocols. Several were already identified during design; they are listed here as forward work rather than as shortcomings, because extending them was deferred by the compute and runtime envelope described above.

### Preprocessing, stain space, and ablations

After the prespecified stain-handling benchmark on class-balanced subsets, main corpora used single-reference Macenko. A larger compute budget would allow full-corpus or multi-seed reruns with other classical or learned normalizers under identical train–test rules, or ablations that vary quality thresholds, to see whether the difficult CAMELYON17-trained to external PCam cell responds more to colour alignment than to representation capacity alone.

### Architecture, fine-tuning depth, and model zoo

Virchow2 was studied in a frozen-encoder, trainable-head configuration suited to limited adaptation time. Partial fine-tuning of top transformer blocks, alternative heads, or additional foundation checkpoints (and distilled variants) are obvious next steps; each adds hyperparameter and checkpoint-selection work and multiplies GPU hours per condition.

### Operating points, thresholds, and decision costs

All threshold summaries in the main tables use probability 0.5 after source-validation temperature scaling, for comparability across arms. Future work should report cost-sensitive cut-offs, sensitivity at fixed false-positive rates, or prevalence-aware objectives alongside ROC–AUC and PR–AUC, especially where ROC–AUC stays high while recall at 0.5 is low.

### Calibration when the deployment domain shifts

Post-hoc temperature scaling on the source validation split improved probability metrics in the reported runs but did not by itself fix recall at the fixed 0.5 cut on the hardest transfer cell. With more experimentation budget, target-aware calibration (small held-out sets from the target site, lightweight adapters, or carefully validated test-time procedures) could be compared under the same leakage rules.

### Site structure and external validity

CAMELYON17 provides hospital metadata; secondary analyses could stratify metrics by site or run leave-one-hospital-out protocols when runtimes allow. Prospective multi-centre data remain the gold standard for deployment claims.

### Slide-level aggregation and richer labels

Patch benchmarks enabled clean bidirectional transfer; clinical use is often slide-level. MIL-style aggregation, explicit ITC or size strata where labels exist, and joint models with segmentation are natural when annotations and compute support them.

### Throughput, compression, and monitoring

Reporting latency, memory footprint, and compressed or distilled models would matter for laboratory deployment, alongside simple monitoring when stain or scanner drift is suspected.

---

Placement note (for the thesis): insert this block after the Discussion chapter and before the Conclusion chapter; point Methodology limitations at this chapter in one cross-reference if overlapping text is removed there.
