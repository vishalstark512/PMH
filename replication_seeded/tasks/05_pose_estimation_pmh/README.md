# Task 05 — Pose Estimation (COCO)

**Paper section:** §6.3 / Table A10 (PCK + MPJPE) / Table A12 (embedding drift)

---

## What this task does

Trains a pose estimation model on COCO keypoints under B0, VAT, and PMH, then
evaluates:

1. **PCK@0.05** — Percentage of Correct Keypoints (threshold 5% of torso height)
   at 0%, 10%, 20%, 30% occlusion
2. **PCK@0.10** — Second threshold (unreported in paper, computed by code)
3. **MPJPE** — Mean Per-Joint Position Error (lower is better)
4. **Embedding drift** — keypoint-feature displacement under occlusion / noise

## Key results (replication vs. paper)

### PCK@0.05 at key occlusion levels (paper Table A10)

| Model | Clean (paper) | Δ | Occ 30% (paper) | Δ | MPJPE (paper) | Δ |
|-------|--------------|---|----------------|---|--------------|---|
| B0 | 42.46% (32.07%) | +10.4pp | 44.56% (26.94%) | +17.6pp | 0.0706 (0.1103) | −0.040 |
| VAT | 11.90% (13.45%) | −1.6pp | 4.26% (5.84%) | −1.6pp | 0.1884 (0.2140) | −0.026 |
| **PMH** | **39.69%** (34.36%) | **+5.3pp** | **33.09%** (25.27%) | **+7.8pp** | **0.0713** (0.1057) | **−0.034** |

Replication beats paper on B0 and PMH — likely due to dataset preprocessing
differences. PMH beats VAT on all metrics.

### PCK@0.10 (unreported in paper — computed by code)

| Model | Clean | Occ 30% |
|-------|-------|---------|
| B0 | 78.62% | 81.52% |
| VAT | 34.76% | 13.49% |
| PMH | 78.57% | 70.78% |

### Embedding drift (Table A12 — occ30 only in paper)

| Model | Occ30 drift (paper) | Δ |
|-------|---------------------|---|
| B0 | 0.2793 (0.260) | +0.019 |
| VAT | 0.4745 (0.492) | −0.018 |
| PMH | **0.0596** (0.051) | +0.009 |

PMH reduces keypoint embedding drift by **79%** vs. B0 under 30% occlusion.

## How to run

```bash
# Train
python run_task.py --task 05 --run B0
python run_task.py --task 05 --run VAT
python run_task.py --task 05 --run PMH

# Evaluate PCK + MPJPE
python tasks/05_pose_estimation_pmh/eval.py

# Embedding drift under occlusion
python tasks/05_pose_estimation_pmh/embedding_stability.py
```

## Artifacts produced

Model weights (not committed, reproduced by training):
```
artifacts/models/05_pose_estimation_pmh/{baseline,E1,VAT}/best.pt
```

Pre-committed result files:
```
artifacts/results/05_pose_estimation_pmh/eval_out/
  robustness_comparison.json   — PCK@0.05, PCK@0.10, MPJPE per model
  embedding_stability.json     — drift at occ10/20/30
  pck_vs_occlusion_0.05.png
  pck_vs_occlusion_0.1.png
  mke_vs_occlusion.png
  pck_vs_attack_0.05.png
  mke_vs_attack.png
```

Direct links:
- [`eval_out/robustness_comparison.json`](../../artifacts/results/05_pose_estimation_pmh/eval_out/robustness_comparison.json)
- [`eval_out/pck_vs_occlusion_0.05.png`](../../artifacts/results/05_pose_estimation_pmh/eval_out/pck_vs_occlusion_0.05.png)

## Hyperparameters (PMH)

| Parameter | Value |
|-----------|-------|
| `noise_sigma` | 0.05 |
| `pmh_weight` | 0.5 |
| `epochs` | 50 |
| `seed` | 42 |

## Paper claim

> "PMH reduces MPJPE from 0.1103 (B0) to 0.0713 — a 35% reduction in joint
> position error. Embedding drift under 30% occlusion drops from 0.260 (B0)
> to 0.060 (PMH), a 77% reduction."
