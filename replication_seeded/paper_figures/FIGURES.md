# Paper Figures

All 12 figures from the paper. The TikZ/pgfplots source files (`.tex`) are not included in this repo since they require a full LaTeX installation to render. Instead, the pre-computed PNG result plots — which feed those figures — are committed under `artifacts/results/` and linked below.

---

## Quick Reference

| Fig | Paper location | Task(s) | Result PNGs |
|-----|---------------|---------|-------------|
| Fig 1 — TDI curves + Jacobian plane | Main §6 | T04 | [tdi_results.json](../artifacts/results/04_vision_transformer_pmh/interp/tdi_results.json) |
| Fig 2 — Three objectives, three geometries | Main §3 | Theory | (diagram — no data file) |
| Fig A1 — Cross-task performance | Appendix A | T01–T07 | [T02 accuracy_bars.png](../artifacts/results/02_graph_classification/evals/accuracy_bars.png) |
| Fig A2 — Layer-wise linear probes | Appendix A | T04 | [linear_probe_results.json](../artifacts/results/04_vision_transformer_pmh/interp/linear_probe_results.json) |
| Fig A3 — Cross-modality TDI | Appendix A | T04, T07, T08 | [semantic_tdi_plot.png](../artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_plot.png) |
| Fig A4 — BERT SST-2 blind spot | Appendix A | T08 | [semantic_tdi_results.json](../artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json) |
| Fig A5 — Cross-domain mechanistic evidence | Appendix A | T01, T02, T07 | [embedding_stability.json](../artifacts/results/01_image_classification/evals/embedding_stability.json) |
| Fig A6 — Re-ID per-shift accuracy | Appendix A | T06 | [robust_rank1.png](../artifacts/results/06_reid_pmh/eval_out_robust/robust_rank1.png) |
| Fig A7 — T-alignment proof sketch | Appendix A | Theory | (diagram — no data file) |
| Fig A8 — FGSM robustness + BERT scale | Appendix A | T04, T08 | [fgsm_results.json](../artifacts/results/04_vision_transformer_pmh/adversarial/fgsm_results.json) |
| Fig A9 — Graph robustness generalisation | Appendix A | T02 | [comparison_curves.png](../artifacts/results/02_graph_classification/evals/comparison_curves.png) · [accuracy_under_edge_removal.png](../artifacts/results/02_graph_classification/evals/accuracy_under_edge_removal.png) |
| Fig A10 — BERT fine-tuning blind-spot hierarchy | Appendix A | T08 | [semantic_tdi_results.json](../artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json) |

---

## Figure Descriptions and Data

### Fig 1 — TDI curves and Frobenius / TDI plane

**Paper §6 (mechanistic core).**

- **Left panel:** TDI vs noise level σ for B0, VAT, PGD-2/255, PGD-4/255, E1 no PMH, E1 (PMH). PMH achieves lowest TDI at every σ; PGD starts at TDI 1.336 on clean inputs — worse than ERM (1.093), confirming Corollary 4.
- **Right panel:** Jacobian Frobenius norm (log x-axis) vs clean-input TDI. PGD-4/255 achieves the lowest Frobenius (2.91) but worst TDI (1.336): the *output-patching zone*. PMH lands in the *geometric-repair zone* (Frobenius 8.08, TDI 0.904).

Numbers source: [`artifacts/results/04_vision_transformer_pmh/interp/tdi_results.json`](../artifacts/results/04_vision_transformer_pmh/interp/tdi_results.json) and [`interp/jacobian_norm_results.json`](../artifacts/results/04_vision_transformer_pmh/interp/jacobian_norm_results.json)

| Model | TDI@0 | TDI@0.10 | Jac. Fro |
|-------|:-----:|:--------:|:--------:|
| B0 | 1.093 | 1.584 | 34.58 |
| VAT | 1.276 | 1.524 | 5.01 |
| PGD-2/255 | 1.319 | 1.450 | 3.42 |
| PGD-4/255 | 1.336 | 1.394 | 2.91 |
| E1 no PMH | 1.074 | 1.262 | 13.09 |
| **E1 (PMH)** | **0.904** | **1.050** | **8.08** |

---

### Fig 2 — Three objectives, three geometries

**Paper §3 (theory).** Three diagrams illustrating how ERM / VAT / PMH each shape the local Jacobian geometry. No data file — purely conceptual.

---

### Fig A1 — Cross-task performance and TDI degradation slopes

**Source:** all 7 vision tasks (T01–T07).

- **Left:** Per-task normalised robustness (best = 100) for B0 / VAT / E1.  PMH leads 5 of 7; B0 leads T05 (mean PCK@0.05); VAT leads T06 (avg-shift Rank-1).
- **Right:** Mean ΔTDI/Δσ slope per method (lower = better). PMH slope 3.20; E1 no PMH 3.40; B0/VAT degrade fastest (6.48 / 5.09).

See task-specific plots in each `artifacts/results/<task>/evals/` directory.

---

### Fig A2 — Layer-wise linear-probe analysis (T04 ViT)

Source: [`artifacts/results/04_vision_transformer_pmh/interp/linear_probe_results.json`](../artifacts/results/04_vision_transformer_pmh/interp/linear_probe_results.json)

| Model | L6 clean | L6 @ σ=0.10 | Retention |
|-------|:--------:|:-----------:|:---------:|
| B0 | 68.20% | 52.40% | 0.768 |
| VAT | 78.75% | 67.60% | 0.858 |
| E1 no PMH | 78.65% | 73.70% | **0.937** |
| E1 (PMH) | 79.55% | 72.90% | 0.916 |

---

### Fig A3 — Cross-modality TDI summary

Sources: T04 TDI, T07 stage distances, T08 semantic TDI.

Demonstrates architecture-independence: PMH repairs geometry on image (ViT), medical (ResNet-50), and language (BERT) encoders.

- [`artifacts/results/04_vision_transformer_pmh/interp/tdi_results.json`](../artifacts/results/04_vision_transformer_pmh/interp/tdi_results.json)
- [`artifacts/results/07_chestxray_pmh/interp_resnet/stage_distances.json`](../artifacts/results/07_chestxray_pmh/interp_resnet/stage_distances.json)
- [`artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json`](../artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json)

---

### Fig A4 — BERT SST-2 geometric blind spot

Source: [`artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json`](../artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json)
Plot: [`semantic_tdi_plot.png`](../artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_plot.png)

- **Left:** TDI vs σ on token embeddings — PMH sits 28.7% below baseline.
- **Right:** Synonym-paraphrase drift: PMH 1.01 vs ERM 4.38 (−76.9% CLS displacement).

---

### Fig A5 — Cross-domain mechanistic evidence

Three panels:
- **(a)** Embedding drift @ σ=0.10 on T01 (CIFAR-10) and T02 (graph) — source: [`01_image_classification/evals/embedding_stability.json`](../artifacts/results/01_image_classification/evals/embedding_stability.json) · [`02_graph_classification/evals/embedding_drift.json`](../artifacts/results/02_graph_classification/evals/embedding_drift.json)
- **(b)** Stage-wise drift on T07 ResNet — source: [`07_chestxray_pmh/interp_resnet/stage_distances.json`](../artifacts/results/07_chestxray_pmh/interp_resnet/stage_distances.json) · plot: [`resnet_stage_clean_noisy_distance.png`](../artifacts/results/07_chestxray_pmh/interp_resnet/resnet_stage_clean_noisy_distance.png)
- **(c)** Saliency cosine similarity on T07 — source: [`07_chestxray_pmh/saliency_stability/saliency_stability_results.json`](../artifacts/results/07_chestxray_pmh/saliency_stability/saliency_stability_results.json) · plot: [`saliency_stability.png`](../artifacts/results/07_chestxray_pmh/saliency_stability/saliency_stability.png)

---

### Fig A6 — Task 06 Re-ID per-shift accuracy

Source: [`artifacts/results/06_reid_pmh/eval_out_robust/compare_results_robust.json`](../artifacts/results/06_reid_pmh/eval_out_robust/compare_results_robust.json)
Plots: [`robust_rank1.png`](../artifacts/results/06_reid_pmh/eval_out_robust/robust_rank1.png) · [`robust_mAP.png`](../artifacts/results/06_reid_pmh/eval_out_robust/robust_mAP.png)

PMH (E1) reaches **63.74%** average Rank-1 across all shifts vs B0's 43.02% — exceeding the paper on every single shift condition.

---

### Fig A7 — T-alignment proof sketch

Geometric proof sketch for Theorem 1 and how the PMH loss term bounds the Frobenius norm. No data source — conceptual diagram.

---

### Fig A8 — FGSM robustness + BERT scale universality

Source: [`adversarial/fgsm_results.json`](../artifacts/results/04_vision_transformer_pmh/adversarial/fgsm_results.json) · [`semantic_tdi/semantic_tdi_results.json`](../artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json)
Plot: [`fgsm_accuracy.png`](../artifacts/results/04_vision_transformer_pmh/adversarial/fgsm_accuracy.png)

BERT blind-spot ratio at all three scales:

| Model | Params | Blind-spot ratio |
|-------|:------:|:----------------:|
| DistilBERT | 66 M | 0.860 |
| BERT-base | 110 M | 0.765 |
| BERT-large | 340 M | 0.742 |

All < 1.0 → blind spot is present at every scale; PMH closes it.

---

### Fig A9 — T02 graph robustness generalisation

Sources: [`eval_E1.json`](../artifacts/results/02_graph_classification/evals/eval_E1.json) · [`comparison_curves.png`](../artifacts/results/02_graph_classification/evals/comparison_curves.png) · [`accuracy_under_edge_removal.png`](../artifacts/results/02_graph_classification/evals/accuracy_under_edge_removal.png) · [`accuracy_under_feature_dropout.png`](../artifacts/results/02_graph_classification/evals/accuracy_under_feature_dropout.png)

PMH was trained with Gaussian node-feature noise but generalises to unseen perturbation types:

| Condition | B0 | VAT | E1 (PMH) | PMH gain |
|-----------|:--:|:---:|:--------:|:--------:|
| Clean | 77.68% | 75.00% | 79.46% | +1.8 pp |
| Edge drop 30% | 60.18% | 53.75% | **76.43%** | **+16.3 pp** |
| Feature drop 30% | 33.04% | 31.96% | **71.61%** | **+38.6 pp** |
| Worst case | 31.61% | 31.61% | **63.75%** | **+32.1 pp** |

---

### Fig A10 — BERT fine-tuning blind-spot hierarchy

Source: [`semantic_tdi_results.json`](../artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json)

ERM fine-tuning amplifies nuisance encoding; PMH repairs it:

| Condition | Paraphrase drift | Blind-spot ratio |
|-----------|:----------------:|:----------------:|
| Pretrained BERT | 0.0244 | 0.765 |
| ERM fine-tuned | 0.0375 | 0.681 |
| **PMH fine-tuned** | **0.0033** | **0.633** |

ERM fine-tuning **worsens** drift +54% over pretrained. PMH **repairs** drift 11× relative to ERM.
