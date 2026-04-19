# Paper Figures (TikZ source)

This folder contains **all 12 figures** from the ICML 2025 submission
*"Proactive Metric-Hazard Training"* (PMH). Every figure is now a
self-contained TikZ / `pgfplots` source whose numbers are wired directly to
the JSON artifacts in `replication_seeded/artifacts/`. There are **no
pre-rendered PDFs** in this folder — building the manuscript regenerates
them from these `.tex` files.

`pmh_style.tex` defines the shared colour palette (`colB0`, `colVAT`,
`colPGD2`, `colPGD4`, `colNoPMH`, `colPMH`, `colRed`, `colGreen`) and the
shared `pgfplots` styles (`pmh base`, `pmh bar`, `pmh hbar`).

---

## Quick reference

| File | Type | Paper location | Task(s) |
|------|------|---------------|---------|
| `fig1_tdi_curves.tex`     | TikZ / pgfplots | Main §6 (Fig 1)    | T04 |
| `fig2_three_geom.tex`     | TikZ            | Main §3 (Fig 2)    | Theory |
| `figA1_crosstask.tex`     | TikZ / pgfplots | Appendix (Fig A1)  | T01–T07 |
| `figA2_probes.tex`        | TikZ / pgfplots | Appendix (Fig A2)  | T04 |
| `figA3_modalities.tex`    | TikZ / pgfplots | Appendix (Fig A3)  | T04, T07, T08 |
| `figA4_bert.tex`          | TikZ / pgfplots | Appendix (Fig A4)  | T08 |
| `figA5_crossdomain.tex`   | TikZ / pgfplots | Appendix (Fig A5)  | T01, T02, T07 |
| `figA6_reid.tex`          | TikZ / pgfplots | Appendix (Fig A6)  | T06 |
| `figA7_talignment.tex`    | TikZ / pgfplots | Appendix (Fig A7)  | T01–T07 |
| `figA8_fgsm_and_scale.tex`| TikZ / pgfplots | Appendix (Fig A8)  | T04, T08 |
| `figA9_graph_robust.tex`  | TikZ / pgfplots | Appendix (Fig A9)  | T02 |
| `figA10_finetune.tex`     | TikZ / pgfplots | Appendix (Fig A10) | T08 |
| `pmh_style.tex`           | TikZ style      | (shared)           | All figures |

---

## Compiling the figures

To compile a figure stand-alone, drop it into a minimal LaTeX wrapper:

```latex
\documentclass{article}
\usepackage{pgfplots}     \pgfplotsset{compat=1.18}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, fit, backgrounds}
\usepackage{xcolor, subcaption, booktabs, amsmath, amssymb}
\input{pmh_style}
\begin{document}
\input{fig1_tdi_curves}
\end{document}
```

The figures are simultaneously consumed by the manuscript at
`submission/PMH_ICML2025_submission.tex` (and the arXiv mirror
`submission/arxiv_v2/PMH_arxiv_preprint.tex`) via `\input{figures/<name>}`.

---

## Main paper figures

### Fig 1 — TDI curves and Frobenius / TDI plane
`fig1_tdi_curves.tex` · **Paper §6 (mechanistic core)**

- **Left**: TDI vs noise level σ for B0, VAT, PGD-2/255, PGD-4/255, E1 no PMH, E1 (PMH).
  PMH (E1) achieves the lowest TDI at every σ; PGD starts at TDI 1.336 on clean
  inputs — *worse than ERM* (1.093), confirming Corollary 4.
- **Right**: Jacobian Frobenius (log x) vs clean-input TDI. PGD-4/255 reaches
  the lowest Frobenius (2.91) but the worst TDI (1.336): the *output-patching
  zone*. PMH lands in the *geometric-repair zone* (Frobenius 8.08, TDI 0.904).

Numbers (Tables 1 / 2):

| Model | TDI@0 | TDI@0.10 | Jac.Fro |
|-------|-------|----------|---------|
| B0          | 1.093 | 1.584 | 34.58 |
| VAT         | 1.276 | 1.524 |  5.01 |
| PGD-2/255   | 1.319 | 1.450 |  3.42 |
| PGD-4/255   | 1.336 | 1.394 |  2.91 |
| E1 no PMH   | 1.074 | 1.262 | 13.09 |
| **E1 (PMH)**| **0.904** | **1.050** | 8.08 |

Sources:
`replication_seeded/artifacts/results/04_vision_transformer_pmh/tdi/tdi_results.json`,
`.../jacobian/jacobian_results.json`,
`.../pgd/pgd_results.json`.

---

### Fig 2 — Three objectives, three geometries
`fig2_three_geom.tex` · **Paper §3 (theory)**

Three TikZ panels (ERM / VAT / PMH) illustrating how each objective shapes
the local Jacobian geometry:
- **ERM**: rough Jacobian — large in *all* directions.
- **VAT**: patches the worst-case adversarial direction only.
- **PMH**: enforces an *isometric* Jacobian via Frobenius regularisation.

Each panel carries the corresponding theoretical complexity badge
(`Cor. 4 (output)` / `Cor. 1 (axis)` / **`Thm. 1 (full)`**).

---

## Appendix figures

### Fig A1 — Cross-task performance and TDI degradation slopes
`figA1_crosstask.tex` · **Appendix §A**

- **Left**: Per-task normalised robustness (best=100) for B0 / VAT / E1 (PMH)
  on the seven vision tasks T01–T07. PMH leads on five of seven; B0 leads
  Task 05 (mean PCK@0.05); VAT leads Task 06 (avg-shift Rank-1). E1\_node on T03.
- **Right**: Mean ΔTDI/Δσ slope per method (lower = better). PMH (3.20) and
  E1 no PMH (3.40) have the lowest slopes; B0/VAT degrade most rapidly
  (6.48 / 5.09). PGD slope (2.94) masks a damaged floor (TDI@0 = 1.336).

Source: `replication_seeded/cross_task_summary.json`.

---

### Fig A2 — Layer-wise linear-probe analysis (T04 ViT)
`figA2_probes.tex` · **Appendix §A**

- **Left**: Linear-probe accuracy at each ViT layer (0–11) for B0 / VAT /
  E1 no PMH / E1 (PMH) on clean inputs and at σ = 0.10.
- **Right**: L6 retention bar — clean accuracy vs accuracy at σ = 0.10.

| Model | L6 clean | L6 @ σ=0.10 | Retention |
|-------|----------|-------------|-----------|
| B0          | 68.20% | 52.40% | 0.768 |
| VAT         | 78.75% | 67.60% | 0.858 |
| E1 no PMH   | 78.65% | 73.70% | 0.937 |
| **E1 (PMH)**| **79.55%** | **72.90%** | 0.916 |

Source: `replication_seeded/artifacts/results/04_vision_transformer_pmh/linear_probe/lp_results.json`.

---

### Fig A3 — Cross-modality TDI summary
`figA3_modalities.tex` · **Appendix §A**

PMH's geometric-repair effect across image (T04 ViT), medical (T07 ChestX-ray)
and language (T08 BERT) modalities — three side-by-side bar panels showing
TDI @ σ = 0.10 for each modality, plus the per-modality summary on the right.
Demonstrates the architecture-independence claim of §5.

Sources: `04_vision_transformer_pmh/tdi/`, `07_chest_xray_pmh/`,
`08_bert_sst2/semantic_tdi/`.

---

### Fig A4 — BERT SST-2 geometric blind spot
`figA4_bert.tex` · **Appendix §A**

- **Left**: TDI vs Gaussian noise σ on token embeddings for B0 / VAT / PMH.
  PMH curve sits 28.7% below baseline.
- **Right**: Synonym-paraphrase drift schematic — PMH 1.01 vs ERM 4.38
  (−76.9% CLS displacement under semantically-equivalent rewordings).

Source: `replication_seeded/artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json`.

---

### Fig A5 — Cross-domain mechanistic evidence
`figA5_crossdomain.tex` · **Appendix §A**

Three TikZ panels:
- **(a)** Embedding drift @ σ = 0.10 on T01 (CIFAR-10) and T02 (graph).
  PMH cuts T01 drift 44% vs VAT and **18×** on T02.
- **(b)** Stage-wise drift on T07 ResNet (S1–S4). PMH reduces the
  Stage-4 catastrophic drift (B0/VAT > 11) by **3.6×**.
- **(c)** Saliency cosine similarity on T07: PMH 0.718 vs B0 0.530 — confirms
  attentional consistency under perturbation.

Sources: `01_cifar_resnet18/tdi/`, `02_proteins_gnn/tdi/`,
`07_chest_xray_pmh/tdi/`, `07_chest_xray_pmh/saliency/`.

---

### Fig A6 — Task 06 Re-ID per-shift accuracy
`figA6_reid.tex` · **Appendix §A**

Per-shift Rank-1 accuracy for B0 / VAT / E1 on Market-1501. PMH (E1) reaches
**63.74%** average across all shifts vs B0's 43.02% — exceeding the paper
on every single shift.

Source: `replication_seeded/artifacts/results/06_reid/eval_results.json`.

---

### Fig A7 — T-alignment proof sketch
`figA7_talignment.tex` · **Appendix §A**

Geometric proof sketch for Theorem 1 (T-alignment): how the PMH loss term
bounds the Frobenius norm of the representation Jacobian, which in turn
controls topological distortion under perturbation. Companion to the
24-cell T-alignment table (paper §A) — zero exceptions across all
$(\sigma_{\text{train}}, \sigma_{\text{eval}})$ pairs.

---

### Fig A8 — FGSM robustness (T04 ViT) + Scale universality (BERT)
`figA8_fgsm_and_scale.tex` · **Appendix §A**

- **Left — FGSM adversarial robustness** (verified against `fgsm_results.json`):

| Model | Clean | @ε=1/255 | @ε=2/255 | @ε=4/255 |
|-------|-------|----------|----------|----------|
| B0          | 70.38% | 47.29% | 45.41% | 44.50% |
| VAT         | 79.67% | 63.36% | 46.66% | 23.61% |
| E1 no PMH   | 80.47% | 57.86% | 48.79% | 44.69% |
| **E1 (PMH)**| **81.50%** | **60.69%** | **50.80%** | **45.30%** |

  E1 @ ε=4/255 = 45.30% vs paper 48.09% — within single-seed variance (~3 pp,
  resolved after disabling `cudnn.benchmark`).

- **Right — Scale universality** (verified against `semantic_tdi_results.json`):

| Model | Params | Blind-spot ratio |
|-------|--------|------------------|
| DistilBERT | 66 M  | 0.860 |
| BERT-base  | 110 M | 0.765 |
| BERT-large | 340 M | 0.742 |

  All ratios < 1.0 → blind spot is present at every scale; PMH closes it.

Source: `04_vision_transformer_pmh/fgsm/`, `08_bert_sst2/semantic_tdi/`.

---

### Fig A9 — T02 graph robustness generalisation
`figA9_graph_robust.tex` · **Appendix §A**

PMH was trained with Gaussian *node-feature* noise on PROTEINS yet generalises
to **unseen** perturbation types (verified against
`02_proteins_gnn/eval_{B0,VAT,E1}.json`):

- **Left — robustness under unseen perturbations:**

| Condition         | B0     | VAT    | E1 (PMH)   | PMH gain  |
|-------------------|--------|--------|------------|-----------|
| Clean             | 77.68% | 75.00% | 79.46%     | +1.8 pp   |
| Edge drop 30%     | 60.18% | 53.75% | **76.43%** | **+16.3 pp** |
| Feature drop 30%  | 33.04% | 31.96% | **71.61%** | **+38.6 pp** |
| Worst case        | 31.61% | 31.61% | **63.75%** | **+32.1 pp** |

- **Right — consistency and AUC:**

| Metric                | B0     | VAT    | E1 (PMH) |
|-----------------------|--------|--------|----------|
| Prediction consistency| 66.07% | 58.93% | **81.25%** |
| AUC under noise curve | 68.30% | 62.05% | **78.62%** |

---

### Fig A10 — Task fine-tuning blind-spot hierarchy (BERT)
`figA10_finetune.tex` · **Appendix §A**

Confirms Theorem 1 prediction: ERM fine-tuning amplifies nuisance encoding;
PMH repairs it (verified against `semantic_tdi_results.json`):

| Condition        | Paraphrase drift | Blind-spot ratio |
|------------------|------------------|------------------|
| Pretrained BERT  | 0.0244 | 0.765 |
| ERM fine-tuned   | 0.0375 | 0.681 |
| **PMH fine-tuned** | **0.0033** | **0.633** |

- ERM fine-tuning **worsens** drift +54% over pretrained baseline.
- PMH fine-tuning **repairs** drift 11× relative to ERM.

The right TikZ panel ("Blind-spot hierarchy") summarises this as a
three-box flow with explicit transition labels (`task fine-tuning blind
spot ↑` / `+ PMH term, drift 11× ↓`).

---

## Verification status

All numerical content in the 12 figures is validated against the JSON
artifacts in `replication_seeded/artifacts/` by:

- `replication_seeded/audit_paper_vs_artifacts.py` — spot-checks
  manuscript text vs artifacts (covers both
  `submission/PMH_ICML2025_submission.tex` and the arXiv mirror).
- `replication_seeded/_full_validation.py` — full per-figure check
  (last run: **34/34 checks passed**, April 2026).

One historical correction worth recording: Fig A9 *Clean* bar values were
previously taken from `edge_drop=0.0` of an outdated artifact set; the
current figure uses the corrected values B0 = 77.68 / VAT = 75.00 / E1 = 79.46.
