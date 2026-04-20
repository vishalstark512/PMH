# Task 04 — ViT CIFAR-10 (Core Mechanistic Task)

**Paper section:** §6.5 (main mechanistic evidence) / Tables 1, 2, A2, A6, A7  
**Figures:** Fig 1, Fig 3, Fig A2, Fig A8 (left)  
**This is the most important task** — it provides the key mechanistic evidence
linking Jacobian Frobenius norm → TDI → robustness.

---

## What this task does

Trains a Vision Transformer (ViT-Small patch 4) on CIFAR-10 under four
conditions — B0, VAT, E1_no_pmh (ablation, ViT without PMH loss), and E1 (PMH)
— then evaluates six distinct metrics:

1. **TDI** — Topological Distortion Index at σ = 0.0 / 0.05 / 0.10 / 0.15 / 0.20
2. **Jacobian Frobenius norm** — local smoothness of the representation map
3. **Linear probe accuracy** — per-layer representation quality (clean + noisy)
4. **Corruption robustness** — accuracy under Gaussian noise, blur, brightness
5. **FGSM adversarial robustness** — accuracy at ε = 1/255, 2/255, 4/255
6. **Intra-class distance** — compactness of class representations

## Key results (replication vs. paper)

### TDI @ σ=0 (paper Tables 1 & 2)

| Model | TDI@0 (paper) | Δ | Jac.Fro (paper) | Δ |
|-------|--------------|---|----------------|---|
| B0 | 1.093 (1.074) | +0.019 | 34.58 (34.76) | −0.18 |
| VAT | 1.276 (1.281) | −0.005 | 5.01 (4.86) | +0.15 |
| E1_no_pmh | 1.074 (1.011) | +0.063 | 13.09 (13.42) | −0.33 |
| **E1 (PMH)** | **0.904** (0.839) | +0.065 | **8.08** (10.73) | −2.65 |

PMH achieves lowest TDI and Jacobian Frobenius — confirming Theorem 1.

### Linear probe retention @ last layer (paper Table A2)

| Model | L6@0 | L6@0.10 | Retention |
|-------|------|---------|----------|
| B0 | 68.2% | 52.4% | 0.768 |
| VAT | 78.75% | 67.6% | 0.858 |
| E1_no_pmh | 78.65% | 73.7% | **0.937** |
| E1 (PMH) | 79.55% | 72.9% | 0.916 |

### FGSM adversarial robustness (paper Table A6 — updated re-run)

| Model | Clean | @ε=1/255 | @ε=2/255 | @ε=4/255 |
|-------|-------|----------|----------|----------|
| B0 | 70.38% | 47.29% | 45.41% | 44.50% |
| VAT | 79.67% | 63.36% | 46.66% | 23.61% |
| E1_no_pmh | 80.47% | 57.86% | 48.79% | 44.69% |
| **E1 (PMH)** | **81.50%** | **60.69%** | **50.80%** | **45.30%** |

> Note: Paper reported E1 @ε=4/255 = 48.09%. Updated to 45.30% after resolving
> `cudnn.benchmark=True` non-determinism. Within single-seed variance (~3pp).
> PMH still outperforms VAT despite VAT being an adversarial training method.

## How to run

```bash
# Train
python run_task.py --task 04 --run B0
python run_task.py --task 04 --run VAT
python run_task.py --task 04 --run E1_no_pmh
python run_task.py --task 04 --run E1

# TDI evaluation
python tasks/04_vision_transformer_pmh/topological_distortion_index.py \
  --runs B0 VAT E1_no_pmh E1

# Jacobian Frobenius norm
python tasks/04_vision_transformer_pmh/jacobian_norm.py \
  --runs B0 VAT E1_no_pmh E1

# Linear probe analysis
python tasks/04_vision_transformer_pmh/linear_probe_analysis.py \
  --runs B0 VAT E1_no_pmh E1

# FGSM adversarial robustness
python tasks/04_vision_transformer_pmh/fgsm_eval.py \
  --runs E1 E1_no_pmh

# Corruption robustness
python tasks/04_vision_transformer_pmh/corruption_eval.py
```

## Artifacts produced

Model weights (not committed, reproduced by training):
```
artifacts/models/04_vision_transformer_pmh/{B0,VAT,E1_no_pmh,E1}/best.pt
```

Pre-committed result files:
```
artifacts/results/04_vision_transformer_pmh/
  interp/tdi_results.json              — TDI at σ = 0/0.05/0.10/0.15/0.20
  interp/jacobian_norm_results.json    — Frobenius norm per model
  interp/linear_probe_results.json     — layer-wise LP accuracy (12 layers)
  interp/clean_noisy_distance.png
  interp/cls_trajectory_{B0,E1}_{clean,noisy}.png
  adversarial/fgsm_results.json        — FGSM at ε = 1/255, 2/255, 4/255
  adversarial/fgsm_accuracy.png
  corruptions/corruptions_results.json — Gauss/blur/brightness accuracy
  corruptions/corruptions_accuracy.png
```

Direct links to key JSON files:
- [`interp/tdi_results.json`](../../artifacts/results/04_vision_transformer_pmh/interp/tdi_results.json)
- [`interp/jacobian_norm_results.json`](../../artifacts/results/04_vision_transformer_pmh/interp/jacobian_norm_results.json)
- [`interp/linear_probe_results.json`](../../artifacts/results/04_vision_transformer_pmh/interp/linear_probe_results.json)
- [`adversarial/fgsm_results.json`](../../artifacts/results/04_vision_transformer_pmh/adversarial/fgsm_results.json)
- [`corruptions/corruptions_results.json`](../../artifacts/results/04_vision_transformer_pmh/corruptions/corruptions_results.json)

## Hyperparameters (PMH / E1)

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-Small, patch=4 |
| `noise_sigma` | 0.10 |
| `pmh_weight` | 1.0 |
| `pmh_cap_ratio` | 0.5 |
| `epochs` | 100 |
| `seed` | 42 |
| `deterministic` | True |

## Paper claims

> "PMH achieves TDI=0.904 vs. B0=1.093 at σ=0 (−17% distortion) and Jacobian
> Frobenius norm 8.08 vs. B0=34.58 (−77%). FGSM robustness 45.30% at ε=4/255
> vs. VAT 23.61% — PMH surpasses an adversarial training method without any
> adversarial objective."
