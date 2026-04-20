# Replication Code — PMH

> **Paper:** *Supervised Learning Has a Necessary Geometric Blind Spot: Theory, Consequences, and Minimal Repair*
> **Author:** Vishal Rajput · KU Leuven
> **For the full repo overview see:** [`../README.md`](../README.md)

---

## What is PMH?

Any encoder trained by empirical risk minimisation (ERM) must maintain non-zero Jacobian sensitivity in directions that predict labels but act as nuisance at test time — the **geometric blind spot** (Theorem 1). This is a provable necessity, not a limitation of specific models or datasets.

**Perturbation Manifold Harmonisation (PMH)** closes this gap with one extra loss term:

$$\mathcal{L}_\text{PMH} = \|\phi(x) - \phi(x + \delta)\|^2, \quad \delta \sim \mathcal{N}(0, \sigma^2 I)$$

Proposition 5 proves Gaussian noise is the *unique* perturbation law that suppresses the full Jacobian Frobenius norm uniformly across all input directions.

This repository reproduces the theorem-level implications emphasized in the manuscript update: blind-spot scale universality (0.860 -> 0.765 -> 0.742 across 66M to 340M), fine-tuning amplification (+54% under ERM fine-tuning), and PMH repair (11x reduction in fine-tuning-induced drift).

Interpretability/alignment relevance: Theorem 1 addresses what any ERM-trained model must encode from a distribution, and predicts that preference-based fine-tuning can amplify nuisance sensitivity unless geometry is regularized.

---

## Headline Results (seed 42)

| Task | Domain | Metric | B0 | VAT | **PMH (E1)** |
|------|---------|--------|----|-----|-------------|
| T01 | CIFAR-10 · ResNet-18 | Acc @ σ=0.10 | 40.04% | 65.17% | **80.38%** |
| T02 | PROTEINS · GNN | AUC @ σ=0.10 | 73.75% | 66.79% | **77.86%** |
| T03 | QM9 · Mol. reg. | MAE ↓ | 23.66 | 26.89 | **22.02** |
| T04 | ViT CIFAR-10 | TDI@0 ↓ | 1.093 | 1.276 | **0.904** |
| T05 | Pose (COCO) | PCK@0.05 | 42.46% | 11.90% | **39.69%** |
| T06 | Re-ID (Market-1501) | Avg-shift Rank-1 | 43.02% | 65.80% | 63.74% |
| T07 | Chest X-Ray | Worst-shift acc | 0.625 | 0.731 | **0.825** |
| T08 | BERT SST-2 | PMH acc cost | — | — | **−0.69 pp** |
| T09 | ImageNet ViT-B/16 | TDI@0 ↓ | 1.230 | — | **0.936** |

Full comparison with paper numbers → [RESULTS.md](RESULTS.md)

---

## Repository Structure

```
replication_seeded/
├── tasks/
│   ├── run_task.py                    # Unified entry-point for any task
│   ├── tdi_utils.py                   # TDI + embedding drift utilities (shared)
│   ├── shared/tdi.py                  # Core TDI implementation
│   ├── 01_image_classification/       # CIFAR-10, ResNet-18
│   ├── 02_graph_classification/       # PROTEINS, GNN
│   ├── 03_molecular_regression/       # QM9, EGNN
│   ├── 04_vision_transformer_pmh/     # ViT CIFAR-10 — core mechanistic task
│   ├── 05_pose_estimation_pmh/        # COCO keypoints, HRNet
│   ├── 06_reid_pmh/                   # Market-1501 Re-ID, ResNet-50
│   ├── 07_chestxray_pmh/              # NIH ChestX-ray14, ResNet-50
│   ├── 08_bert_sst2/                  # BERT fine-tuning, SST-2
│   └── 09_imagenet_vit/               # ImageNet ViT-B/16 (Hugging Face)
│
├── artifacts/
│   ├── results/                       # JSON + PNG results — tracked in git (~4.4 MB)
│   │   ├── 01_image_classification/evals/
│   │   ├── 02_graph_classification/evals/
│   │   ├── 03_molecular_regression/{evals,embedding_analysis}/
│   │   ├── 04_vision_transformer_pmh/{adversarial,corruptions,interp}/
│   │   ├── 05_pose_estimation_pmh/eval_out/
│   │   ├── 06_reid_pmh/{eval_out,eval_out_robust}/
│   │   ├── 07_chestxray_pmh/{eval_out_robust,interp_resnet,saliency_stability}/
│   │   ├── 08_bert_sst2/semantic_tdi/
│   │   └── 09_imagenet_vit/baseline/
│   └── run_manifest.json              # Maps runs to artifact files
│
├── paper_figures/
│   └── FIGURES.md                     # Figure descriptions + links to result PNGs
│   └── make_cross_task_results.py     # Regenerates README cross-task image
│
├── run_all_replication.py             # Run all 9 tasks end-to-end
├── audit_paper_vs_artifacts.py        # Spot-check manuscript numbers vs JSON
├── RESULTS.md                         # Full replication vs paper comparison
├── PAPER_SCOPE.md                     # Maps each paper table to its script
└── NON_PAPER_CONTENT_AUDIT.md         # Metrics computed beyond what is in the paper
```

> **`artifacts/models/`** (~1.85 GB trained checkpoints) is excluded via `.gitignore`. Reproduce by running training, or inspect pre-committed results without retraining.

---

## Installation

```bash
# Base (required for all tasks)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Graph tasks (T02, T03)
pip install torch-geometric

# Language tasks (T08, T09)
pip install transformers datasets
```

Per-task extras:

```bash
pip install -r tasks/04_vision_transformer_pmh/requirements.txt
pip install -r tasks/06_reid_pmh/requirements.txt
# etc.
```

Python ≥ 3.10 · CUDA 12.x · Tested on RTX 4090.

---

## Quick Start

### Browse results without retraining

All JSON metrics and PNG plots are in `artifacts/results/` — no GPU needed.

```
artifacts/results/04_vision_transformer_pmh/interp/tdi_results.json   # TDI table
artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_plot.png      # BERT blind spot plot
artifacts/results/02_graph_classification/evals/comparison_curves.png  # T02 curves
```

### Run a single task

```bash
# Auto-downloaded (CIFAR-10, PROTEINS, QM9, SST-2, ImageNet)
python tasks/run_task.py --task 01 --artifact_root ./artifacts
python tasks/run_task.py --task 08 --artifact_root ./artifacts
python tasks/run_task.py --task 09 --artifact_root ./artifacts

# Manual download required
python tasks/run_task.py --task 05 --data_dir /path/to/coco       --artifact_root ./artifacts
python tasks/run_task.py --task 06 --data_dir /path/to/market1501  --artifact_root ./artifacts
python tasks/run_task.py --task 07 --data_dir /path/to/chestxray   --artifact_root ./artifacts
```

### Run all 9 tasks

```bash
python run_all_replication.py --data_dir /path/to/datasets --artifact_root ./artifacts
```

---

## Task-by-Task Training Commands

<details>
<summary><strong>T01 — CIFAR-10 · ResNet-18</strong></summary>

```bash
cd tasks/01_image_classification
python train.py --run B0  --seed 42
python train.py --run VAT --seed 42
python train.py --run E1  --seed 42
python CIFAR10/embedding_stability.py
```

Results: [`artifacts/results/01_image_classification/evals/embedding_stability.json`](artifacts/results/01_image_classification/evals/embedding_stability.json)
</details>

<details>
<summary><strong>T02 — PROTEINS · GNN</strong></summary>

```bash
cd tasks/02_graph_classification
python train.py --run B0  --seed 42
python train.py --run VAT --seed 42
python train.py --run E1  --seed 42
python eval.py --runs B0 VAT E1
python embedding_stability.py
```

Results: [`artifacts/results/02_graph_classification/evals/`](artifacts/results/02_graph_classification/evals/)
</details>

<details>
<summary><strong>T03 — QM9 · Molecular Regression</strong></summary>

```bash
cd tasks/03_molecular_regression
python train.py --run B0  --batch_size 128 --seed 42
python train.py --run VAT --batch_size 128 --seed 42
python train.py --run E1  --batch_size 128 --seed 42
python run_evals.py --runs B0 VAT E1
python embedding_analysis.py
```

Results: [`artifacts/results/03_molecular_regression/evals/eval_summary.json`](artifacts/results/03_molecular_regression/evals/eval_summary.json)
</details>

<details>
<summary><strong>T04 — ViT CIFAR-10 (core mechanistic task)</strong></summary>

Reproduces Tables 1 & 2 and all T04 appendix tables.

```bash
cd tasks/04_vision_transformer_pmh
python train.py --run B0         --seed 42
python train.py --run VAT        --seed 42
python train.py --run E1_no_pmh  --seed 42
python train.py --run E1         --seed 42

python topological_distortion_index.py --runs B0 VAT E1_no_pmh E1
python jacobian_norm.py                --runs B0 VAT E1_no_pmh E1
python linear_probe_analysis.py        --runs B0 VAT E1_no_pmh E1
python eval_adversarial.py             --runs E1 E1_no_pmh
python eval_corruptions.py             --runs B0 VAT E1_no_pmh E1
```

Key results:
- [`artifacts/results/04_vision_transformer_pmh/interp/tdi_results.json`](artifacts/results/04_vision_transformer_pmh/interp/tdi_results.json)
- [`artifacts/results/04_vision_transformer_pmh/adversarial/fgsm_results.json`](artifacts/results/04_vision_transformer_pmh/adversarial/fgsm_results.json)
- [`artifacts/results/04_vision_transformer_pmh/interp/jacobian_norm_results.json`](artifacts/results/04_vision_transformer_pmh/interp/jacobian_norm_results.json)
</details>

<details>
<summary><strong>T05 — COCO Pose Estimation</strong></summary>

```bash
cd tasks/05_pose_estimation_pmh
python train.py --run baseline --seed 42
python train.py --run E1       --seed 42
python train.py --run VAT      --seed 42
python eval.py  --runs baseline E1 VAT
python embedding_stability.py
```

Results: [`artifacts/results/05_pose_estimation_pmh/eval_out/`](artifacts/results/05_pose_estimation_pmh/eval_out/)
</details>

<details>
<summary><strong>T06 — Market-1501 · Person Re-ID</strong></summary>

```bash
cd tasks/06_reid_pmh
python train.py    --run B0  --seed 42
python train.py    --run VAT --seed 42
python train.py    --run E1  --seed 42
python eval.py     --runs B0 VAT E1
python eval_robust.py --runs B0 VAT E1
```

Results: [`artifacts/results/06_reid_pmh/eval_out_robust/`](artifacts/results/06_reid_pmh/eval_out_robust/)
</details>

<details>
<summary><strong>T07 — NIH ChestX-ray14 · ResNet-50</strong></summary>

```bash
cd tasks/07_chestxray_pmh
python train.py --run B0        --seed 42
python train.py --run VAT       --seed 42
python train.py --run E1_no_pmh --seed 42
python train.py --run E1        --seed 42
python eval_robust.py  --runs B0 VAT E1_no_pmh E1
python saliency_stability.py   --runs B0 E1
```

Results: [`artifacts/results/07_chestxray_pmh/`](artifacts/results/07_chestxray_pmh/)
</details>

<details>
<summary><strong>T08 — BERT · SST-2</strong></summary>

```bash
cd tasks/08_bert_sst2
python train.py --run baseline
python train.py --run pmh
python measure_tdi.py
python exp_semantic_tdi/measure_semantic_tdi.py
```

Results: [`artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json`](artifacts/results/08_bert_sst2/semantic_tdi/semantic_tdi_results.json)
</details>

<details>
<summary><strong>T09 — ImageNet · ViT-B/16</strong></summary>

```bash
cd tasks/09_imagenet_vit
python measure_tdi.py
python sort_imagenet_validation.py
```

Results: [`artifacts/results/09_imagenet_vit/baseline/tdi_pretrained_baseline.json`](artifacts/results/09_imagenet_vit/baseline/tdi_pretrained_baseline.json)
</details>

---

## Dataset Setup

| Task | Dataset | Source |
|------|---------|--------|
| T01, T04 | CIFAR-10 | Auto via `torchvision` |
| T02 | PROTEINS | Auto via PyTorch Geometric |
| T03 | QM9 | Auto via PyTorch Geometric |
| T05 | COCO Keypoints 2017 | [cocodataset.org](https://cocodataset.org/#download) |
| T06 | Market-1501 | [ANU project page](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) |
| T07 | NIH ChestX-ray14 | [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| T08 | SST-2 | Auto via Hugging Face `datasets` |
| T09 | ImageNet val | Auto via Hugging Face `datasets` (100-class subset) |

Auto-downloaded data lands in `tasks/<task>/data/` (git-ignored).

---

## Reproducibility Notes

- All scripts accept `--seed` (default 42) and `--deterministic` for strict cuDNN mode.
- Original paper code was unseeded; this replication adds explicit seeding. Results are within single-seed variance of the paper; 4 of 7 vision tasks **beat** the paper numbers.
- T09 uses a 100-class × 50-sample ImageNet subset; absolute TDI values shift slightly but the PMH < pretrained ordering holds.
- All JSON + PNG results are pre-committed to `artifacts/results/` — inspect without retraining.

## Scope of Theoretical Claims

Theorem 1 is an existence theorem: it guarantees the presence of nuisance-sensitive directions under supervised ERM, rather than a tight numeric predictor for every model instance. This is the intended form for architecture- and dataset-agnostic claims; task-level constants are reported empirically in the accompanying results artifacts.

---

## What the Code Computes Beyond the Paper

Several metrics are computed but not reported in the paper — see [NON_PAPER_CONTENT_AUDIT.md](NON_PAPER_CONTENT_AUDIT.md).

| Task | Unreported metric | Key number |
|------|-----------------|------------|
| T02 | Feature drop 30% accuracy | E1 = **71.6%** vs B0 = 33.0% (+38.6 pp) |
| T02 | Edge drop 30% accuracy | E1 = **76.4%** vs VAT = 53.8% (+22.6 pp) |
| T05 | PCK@0.10 (second threshold) | E1 = **78.6%** at clean |
| T08 | Scale universality (66M / 110M / 340M BERT) | Blind spot at all scales |
| T08 | Task fine-tuning blind spot | ERM worsens by 54%; PMH repairs 11× |

---

## Citation

```bibtex
@article{rajput2025supervised,
  title   = {Supervised Learning Has a Necessary Geometric Blind Spot:
             Theory, Consequences, and Minimal Repair},
  author  = {Rajput, Vishal},
  year    = {2025},
  note    = {Preprint. KU Leuven.}
}
```
