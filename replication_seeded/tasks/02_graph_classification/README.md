# Task 02 — PROTEINS Graph Classification (GNN)

**Paper section:** §6.2 / Table 3 (cross-task headline) / Table A1 (embedding drift)  
**New figure:** Fig A9 (`paper_figures/figA9_graph_robust.tex`)

---

## What this task does

Trains a 4-layer GIN (Graph Isomorphism Network) on the PROTEINS dataset under
B0, VAT, and PMH (E1), then evaluates:

1. **AUC under Gaussian noise** — AUC-ROC at σ = 0.05 / 0.10 / 0.15 / 0.20
   (node-feature Gaussian noise)
2. **Robustness to unseen perturbation types** — edge removal (30%) and
   feature dropout (30%); *PMH never saw these during training*
3. **Prediction consistency** — fraction of test graphs where prediction is
   unchanged after perturbation
4. **Embedding drift** — normalised L2 displacement of graph-level embeddings

## Key results (replication vs. paper)

### AUC @σ=0.10 (paper Table 3 headline)

| Model | AUC @0.10 (paper) | Δ |
|-------|------------------|---|
| B0 | 73.75% (57.86%) | +15.89pp |
| VAT | 66.79% (67.50%) | −0.71pp |
| **E1 (PMH)** | **77.86%** (78.04%) | −0.18pp |

### Robustness to unseen perturbations (Fig A9 — NEW, not in original paper)

| Condition | B0 | VAT | E1 (PMH) | PMH gain |
|-----------|----|-----|---------|---------|
| Clean | 77.68% | 75.00% | 79.46% | — |
| Edge drop 30% | 60.18% | 53.75% | **76.43%** | +16.3pp |
| Feature drop 30% | 33.04% | 31.96% | **71.61%** | **+38.6pp** |
| Worst case | 31.61% | 31.61% | **63.75%** | +32.1pp |
| Prediction consistency | 66.07% | 58.93% | **81.25%** | +15.2pp |

> **Key insight**: PMH was trained only on Gaussian noise but generalises to
> edge removal (+16.3pp) and feature dropout (+38.6pp). This confirms PMH
> performs global geometric repair, not perturbation matching.

### Embedding drift @σ=0.10 (paper Table A1)

| Model | Drift (paper) | Δ |
|-------|--------------|---|
| B0 | 0.375 (0.825) | −0.450 |
| VAT | 0.304 (0.574) | −0.270 |
| E1 | **0.021** (0.044) | −0.023 |

PMH reduces graph embedding drift by **94%** vs. B0.

## How to run

```bash
# Train
python run_task.py --task 02 --run B0
python run_task.py --task 02 --run VAT
python run_task.py --task 02 --run E1

# Evaluate (AUC + unseen perturbations + consistency)
python tasks/02_graph_classification/eval.py

# Embedding drift
python tasks/02_graph_classification/embedding_stability.py
```

## Artifacts produced

```
artifacts/
  models/02_graph_classification/PROTEINS/{B0,VAT,E1}/best.pt
  results/02_graph_classification/evals/
    eval_{B0,VAT,E1}.json       — AUC, edge/feature drop, consistency
    embedding_drift.json        — drift at σ = 0.05/0.10/0.15/0.20
```

## Hyperparameters (PMH / E1)

| Parameter | Value |
|-----------|-------|
| GNN layers | 4 |
| Hidden dim | 128 |
| `noise_sigma` | 0.10 |
| `pmh_weight` | 0.5 |
| `epochs` | 200 |
| `seed` | 42 |

## Paper claim

> "PMH achieves 94% reduction in graph embedding drift and generalises to
> unseen perturbation types: +38.6pp on feature dropout, +16.3pp on edge
> removal — never seen during training. Prediction consistency: 81.3% vs. B0 66.1%."
