# Task 01 — CIFAR-10 Image Classification (ResNet-18)

**Paper section:** §6.2 / Table 3 (cross-task headline) / Table A1 (embedding drift)  
**Figure:** Fig 4 (cross-domain results, embedding drift panel)

---

## What this task does

Trains a ResNet-18 on CIFAR-10 under three objectives — ERM baseline (B0),
Virtual Adversarial Training (VAT), and PMH (E1) — then evaluates:

1. **Clean accuracy** — standard test-set accuracy
2. **Corruption robustness** — accuracy under Gaussian noise at σ = 0.05 / 0.10 / 0.15 / 0.20
3. **Embedding drift** — mean normalised L2 distance between clean and noisy
   penultimate-layer embeddings (Table A1; lower = more stable representations)

## Key results (replication vs. paper)

| Model | Clean | Robust @σ=0.10 (paper) | Δ |
|-------|-------|------------------------|---|
| B0 (ERM) | 87.41% | 40.04% (40.39%) | −0.35pp |
| VAT | 94.36% | 65.17% (70.59%) | −5.42pp |
| **E1 (PMH)** | **93.43%** | **80.38%** (81.36%) | **−0.98pp** |

### Embedding drift @σ=0.10 (paper Table A1)

| Model | Drift (paper) | Δ |
|-------|--------------|---|
| B0 | 0.692 (0.691) | +0.001 |
| VAT | 0.699 (0.663) | +0.036 |
| E1 | **0.385** (0.373) | +0.012 |

PMH reduces embedding drift by **44%** relative to B0.

## How to run

```bash
# Train all three models (from replication_seeded/)
python run_task.py --task 01 --run B0
python run_task.py --task 01 --run VAT
python run_task.py --task 01 --run E1

# Evaluate corruption robustness
python tasks/01_image_classification/eval.py

# Evaluate embedding drift
python tasks/01_image_classification/embedding_stability.py
```

## Artifacts produced

Model weights (not committed, reproduced by training):
```
artifacts/models/01_image_classification/cifar10/{B0,VAT,E1}/best.pt
```

Pre-committed result files:
```
artifacts/results/01_image_classification/evals/
  embedding_stability.json    — drift at σ = 0.05/0.10/0.15/0.20 for B0/VAT/E1
```

→ [`artifacts/results/01_image_classification/evals/embedding_stability.json`](../../artifacts/results/01_image_classification/evals/embedding_stability.json)

## Hyperparameters (PMH / E1)

| Parameter | Value |
|-----------|-------|
| `noise_sigma` | 0.10 |
| `pmh_max_weight` | 1.0 |
| `pmh_cap_ratio` | 0.5 |
| `warmup_epochs` | 5 |
| `epochs` | 100 |
| `seed` | 42 |

## Paper claim

> "PMH reduces embedding drift at σ=0.10 by 44% on CIFAR-10 while achieving
> 80.38% robustness accuracy vs. B0's 40.04% — a +40pp gain with no
> corruption-specific training."
