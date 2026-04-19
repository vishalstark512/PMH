# Task 09 — ImageNet ViT-B/16 (Foundation Model Scale)

**Paper section:** §6.9 / Table A5 (TDI + intra-class distance)  
**Significance:** Shows PMH's geometric repair transfers to pretrained foundation
models fine-tuned on ImageNet — no training from scratch required.

---

## What this task does

Fine-tunes a pretrained ViT-B/16 (ImageNet-21k) on ImageNet-1k classification
under two conditions — pretrained (no PMH) and PMH fine-tuned — then evaluates:

1. **TDI** — Topological Distortion Index at σ = 0.0 / 0.05 / 0.10
2. **Intra-class distance** — mean pairwise L2 distance between embeddings of
   the same class (higher for fine-tuned models — task supervision spreads
   within-class representations)

## Key results (replication vs. paper)

### TDI + intra-class distance (paper Table A5)

> Note: Replication uses a 100-class × 50-sample subset; paper used full
> 1000-class ImageNet. This shifts absolute TDI values upward but preserves
> the relative ordering (PMH < pretrained), which is the paper's claim.

| Run | TDI@0 (paper) | Δ | TDI@0.10 (paper) | Δ | Intra (paper) | Δ |
|-----|--------------|---|-----------------|---|--------------|---|
| Pretrained | 1.230 (1.033) | +0.197 | 1.327 (1.096) | +0.231 | 41.05 (39.6) | +1.45 |
| **PMH fine-tuned** | **0.936** (0.900) | +0.036 | **1.028** (0.920) | +0.108 | **67.36** (60.8) | +6.56 |

### Key finding

PMH fine-tuning:
- **Reduces TDI**: 0.936 vs. pretrained 1.230 (−24% reduction) ✓
- **Increases intra-class distance**: 67.36 vs. 41.05 — fine-tuning spreads
  within-class representations (more discriminative; this is expected and
  desirable for classification, opposite of the from-scratch T04 result)

The direction of intra-class distance change **reverses** between T04 (from
scratch, PMH makes representations tighter) and T09 (fine-tuned, PMH makes
representations more spread) — an important empirical distinction documented
in the paper.

## How to run

```bash
# Fine-tune (requires ImageNet data)
python run_task.py --task 09 --run pretrained
python run_task.py --task 09 --run pmh

# TDI + intra-class evaluation
python tasks/09_imagenet_vit/eval.py --subset_classes 100 --subset_samples 50
```

## Artifacts produced

```
artifacts/
  models/09_imagenet_vit/{pretrained,pmh}/best.pt
  results/09_imagenet_vit/
    tdi_results.json     — TDI at 3 noise levels
    intra_class.json     — mean intra-class L2 distance
```

## Hyperparameters (PMH)

| Parameter | Value |
|-----------|-------|
| Base model | ViT-B/16 (ImageNet-21k pretrained) |
| `pmh_weight` | 0.1 |
| `lr` | 1e-4 |
| `epochs` | 10 |
| `seed` | 42 |

## Dataset note

ImageNet-1k requires a manual download (not auto-downloaded). Place it at:
```
tasks/09_imagenet_vit/data/imagenet/
  train/
  val/
```

## Paper claim

> "PMH reduces TDI by 24% when fine-tuning ViT-B/16 on ImageNet, confirming
> that geometric repair transfers to foundation model fine-tuning at scale —
> no training from scratch is required."
