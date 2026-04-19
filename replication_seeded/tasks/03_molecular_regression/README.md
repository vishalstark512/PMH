# Task 03 — QM9 Molecular Property Regression (GNN)

**Paper section:** §6.2 / Table 3 (cross-task headline) / Table A11 (noise MAE curve)

---

## What this task does

Trains a GNN on the QM9 dataset to predict molecular properties (HOMO-LUMO gap
target) under B0, VAT, and PMH (E1), then evaluates:

1. **Clean MAE** — mean absolute error on the clean test set
2. **MAE under position noise** — MAE at σ = 0.0 / 0.005 / 0.01 / 0.02 / 0.05
   / 0.10 / 0.20 (Gaussian noise added to atomic positions)

## Key results (replication vs. paper)

### Clean MAE (paper Table 3)

| Model | Clean MAE (paper) | Δ |
|-------|------------------|---|
| B0 | 23.664 (25.06) | −1.396 |
| VAT | 26.886 (28.51) | −1.624 |
| **E1 (PMH)** | **22.020** (23.62) | **−1.600** |

Replication **beats** the paper on all three — PMH achieves 22.02 MAE vs.
23.62 reported.

### MAE at noise levels (paper Table A11)

| σ | B0 repl (paper) | E1 repl (paper) |
|---|-----------------|-----------------|
| 0.0 | 61.9 (64.72) | 45.34 (44.18) |
| 0.10 | 65.23 (70.18) | 50.52 (47.46) |
| 0.20 | 72.9 (79.0) | 60.94 (54.64) |

> Note: Table A11 in the paper uses a different target / normalisation
> (position noise). Our E1 uses the node-only config (E1_node per §4.2),
> which achieves better clean MAE than the paper's position-noise E1.

## How to run

```bash
# Train
python run_task.py --task 03 --run B0
python run_task.py --task 03 --run VAT
python run_task.py --task 03 --run E1

# Evaluate (clean MAE + noise MAE curve)
python tasks/03_molecular_regression/eval.py
```

## Artifacts produced

```
artifacts/
  models/03_molecular_regression/QM9/{B0,VAT,E1}/best.pt
  results/03_molecular_regression/
    eval_{B0,VAT,E1}.json       — clean_mae, noisy_mae, noise curve
```

## Hyperparameters (PMH / E1)

| Parameter | Value |
|-----------|-------|
| `noise_sigma` | 0.05 |
| `pmh_weight` | 0.3 |
| `epochs` | 300 |
| `seed` | 42 |
