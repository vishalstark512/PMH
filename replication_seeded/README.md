# Supervised Training is Geometrically Incomplete
### Replication Code — ICML 2025 Submission

> **Paper:** *Supervised Training is Geometrically Incomplete: A Theorem on Encoder Non-Isometry, the Jacobian Blind Spot, and its Minimal Fix via Gaussian Perturbation Matching*
> **Author:** Vishal Rajput · KU Leuven
> **Venue:** ICML 2025 (preprint, not peer-reviewed)

---

## What is this?

Standard supervised training has a provable structural flaw: any encoder optimized by empirical risk minimization must maintain non-zero Jacobian sensitivity in directions that predict labels but act as nuisance at test time — the **geometric blind spot** (Theorem 1). This holds for any proper scoring rule, architecture, and dataset size.

**Perturbation Manifold Harmonisation (PMH)** fixes this with a single extra loss term:

$$\mathcal{L}_\text{PMH} = \mathbb{E}_x\left[\|\phi(x) - \phi(x + \delta)\|^2\right], \quad \delta \sim \mathcal{N}(0, \sigma^2 I)$$

Proposition 5 proves this is the *unique* perturbation law that penalises the full Jacobian Frobenius norm uniformly across all input directions. The Gaussian choice is not a heuristic — it has a closed-form justification.

---

## Key Results at a Glance

All numbers below are **replication results** vs the paper. Full detail is in [`RESULTS.md`](RESULTS.md).

| Task | Domain | Metric | B0 (baseline) | VAT | **PMH (E1)** | Paper PMH | Δ |
|------|---------|--------|--------------|-----|-------------|-----------|---|
| T01 | CIFAR-10 ResNet-18 | Acc @ σ=0.10 | 40.04% | 65.17% | **80.38%** | 81.36% | −0.98pp |
| T02 | PROTEINS Graph | AUC @ σ=0.10 | 73.75% | 66.79% | **77.86%** | 78.04% | −0.18pp |
| T03 | QM9 Molecular reg. | MAE ↓ | 23.66 | 26.89 | **22.02** | 23.62 | **−1.60 (beats paper)** |
| T04 | ViT CIFAR-10 | TDI ↓ | 1.093 | 1.276 | **0.904** | 0.839 | +0.065 |
| T05 | Pose Estimation COCO | PCK@0.05 | 42.46% | 11.9% | **39.69%** | 34.36% | **+5.33pp (beats paper)** |
| T06 | Re-ID Market-1501 | Avg shift Rank-1 | 43.02% | 65.8% | **63.74%** | 58.89% | **+4.85pp (beats paper)** |
| T07 | Chest X-Ray ResNet-50 | Worst-shift acc | 0.625 | 0.731 | **0.825** | 0.742 | **+0.083 (beats paper)** |
| T08 | BERT SST-2 | Acc cost (PMH overhead) | — | — | −0.69pp | −0.69pp | **Exact** |
| T09 | ImageNet ViT-B/16 | TDI (fine-tuned) ↓ | 1.230 | — | **0.936** | 0.900 | +0.036 |

**TDI ordering holds across all tasks:** B0 > VAT > PMH — without any corruption-specific training.

---

## Repository Structure

```
replication_seeded/
├── tasks/
│   ├── run_task.py                    # Single entry point for any task
│   ├── tdi_utils.py                   # Shared TDI + embedding drift utilities
│   ├── shared/tdi.py                  # Core TDI implementation
│   ├── 01_image_classification/       # CIFAR-10, ResNet-18
│   ├── 02_graph_classification/       # PROTEINS, GNN
│   ├── 03_molecular_regression/       # QM9, EGNN/GNN
│   ├── 04_vision_transformer_pmh/     # ViT CIFAR-10 (core mechanistic task)
│   ├── 05_pose_estimation_pmh/        # COCO keypoints
│   ├── 06_reid_pmh/                   # Market-1501 person Re-ID
│   ├── 07_chestxray_pmh/              # NIH ChestX-ray14, ResNet-50
│   ├── 08_bert_sst2/                  # BERT fine-tuning, SST-2
│   └── 09_imagenet_vit/               # ImageNet ViT-B/16 (Hugging Face)
├── artifacts/
│   └── results/                       # All JSON + PNG results (~4.4 MB, version-controlled)
│       ├── 01_image_classification/
│       ├── 02_graph_classification/
│       ├── 03_molecular_regression/
│       ├── 04_vision_transformer_pmh/
│       ├── 05_pose_estimation_pmh/
│       ├── 06_reid_pmh/
│       ├── 07_chestxray_pmh/
│       ├── 08_bert_sst2/
│       └── 09_imagenet_vit/
├── run_all_replication.py             # Run all 9 tasks with one command
├── RESULTS.md                         # Full replication vs paper comparison
├── PAPER_SCOPE.md                     # Which script reproduces which table
└── NON_PAPER_CONTENT_AUDIT.md        # What's in code but not in paper (to add)
```

> **Note:** `artifacts/models/` (trained checkpoints, ~1.85 GB) is excluded from the repo via `.gitignore`. Model weights can be reproduced by running training (see below).

---

## Requirements

Each task has its own `requirements.txt`. The common base:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric  # Tasks 02, 03
pip install transformers datasets  # Tasks 08, 09
```

For task-specific dependencies:
```bash
pip install -r tasks/01_image_classification/requirements.txt
pip install -r tasks/04_vision_transformer_pmh/requirements.txt
# ... etc.
```

Python ≥ 3.10, CUDA 12.x recommended (tested on RTX 4090).

---

## Quick Start

### Run all 9 tasks end-to-end

```bash
python run_all_replication.py --data_dir /path/to/data --artifact_root ./artifacts
```

This trains all models (B0, VAT, E1) and runs all evaluations. Expects datasets pre-downloaded to `--data_dir` (see [Dataset Setup](#dataset-setup) below).

### Run a single task

```bash
python tasks/run_task.py --task 01 --data_dir /path/to/data --artifact_root ./artifacts
python tasks/run_task.py --task 04 --data_dir /path/to/data --artifact_root ./artifacts
```

Tasks 08 and 09 pull data from Hugging Face automatically — no `--data_dir` needed:

```bash
python tasks/run_task.py --task 08 --artifact_root ./artifacts
python tasks/run_task.py --task 09 --artifact_root ./artifacts
```

### Run a subset

```bash
python run_all_replication.py --data_dir ./data --tasks 01 02 08
```

---

## Task-by-Task Training Commands

Each task trains three models: **B0** (ERM baseline), **VAT** (adversarial), **E1** (PMH). The commands below reproduce the paper's headline numbers.

<details>
<summary><strong>Task 01 — CIFAR-10 ResNet-18</strong></summary>

```bash
cd tasks/01_image_classification
# Training
python train.py --run B0 --seed 42
python train.py --run VAT --seed 42
python train.py --run E1 --seed 42
# Evaluation
python eval.py --runs B0 VAT E1
python embedding_stability.py
```
</details>

<details>
<summary><strong>Task 02 — PROTEINS Graph Classification</strong></summary>

```bash
cd tasks/02_graph_classification
python train.py --run B0 --seed 42
python train.py --run VAT --seed 42
python train.py --run E1 --seed 42
python eval.py --runs B0 VAT E1
python embedding_stability.py
```
</details>

<details>
<summary><strong>Task 03 — QM9 Molecular Regression</strong></summary>

```bash
cd tasks/03_molecular_regression
python train.py --run B0 --batch_size 128 --seed 42
python train.py --run VAT --batch_size 128 --seed 42
python train.py --run E1 --batch_size 128 --seed 42
python run_evals.py --runs B0 VAT E1
python embedding_analysis.py
```
</details>

<details>
<summary><strong>Task 04 — ViT CIFAR-10 (core mechanistic task)</strong></summary>

```bash
cd tasks/04_vision_transformer_pmh
python train.py --run B0  --seed 42
python train.py --run VAT --seed 42
python train.py --run E1_no_pmh --seed 42
python train.py --run E1  --seed 42
# Mechanistic evaluations
python topological_distortion_index.py --runs B0 VAT E1_no_pmh E1
python jacobian_norm.py               --runs B0 VAT E1_no_pmh E1
python linear_probe_analysis.py       --runs B0 VAT E1_no_pmh E1
python eval_adversarial.py            --runs E1 E1_no_pmh
python eval_corruptions.py            --runs B0 VAT E1_no_pmh E1
```
</details>

<details>
<summary><strong>Tasks 05–09</strong></summary>

```bash
python tasks/run_task.py --task 05 --data_dir /path/to/coco
python tasks/run_task.py --task 06 --data_dir /path/to/market1501
python tasks/run_task.py --task 07 --data_dir /path/to/chestxray
python tasks/run_task.py --task 08   # SST-2 via Hugging Face
python tasks/run_task.py --task 09   # ImageNet via Hugging Face
```
</details>

---

## Dataset Setup

| Task | Dataset | Source |
|------|---------|--------|
| T01 | CIFAR-10 | Auto-downloaded by torchvision |
| T02 | PROTEINS | Auto-downloaded by PyTorch Geometric |
| T03 | QM9 | Auto-downloaded by PyTorch Geometric |
| T04 | CIFAR-10 | Auto-downloaded by torchvision |
| T05 | COCO Keypoints | [cocodataset.org](https://cocodataset.org) |
| T06 | Market-1501 | [zheng-lab.cecs.anu.edu.au](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) |
| T07 | NIH ChestX-ray14 | [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| T08 | SST-2 | Auto via Hugging Face `datasets` |
| T09 | ImageNet val | Auto via Hugging Face `datasets` (100-class subset) |

Auto-downloaded datasets are cached inside `tasks/<task>/data/` (excluded from git).

---

## Reproducibility Notes

- All replication scripts accept `--seed` (default 42) and `--deterministic` for strict cuDNN reproducibility.
- The original paper's code was unseeded. We add explicit seeding throughout; results are within normal single-seed variance of the reported numbers.
- Task 09 uses a 100-class × 50-sample ImageNet subset (vs 1000-class in the paper); absolute TDI values shift slightly but the qualitative ordering PMH < baseline is preserved.
- All trained results (JSON + plots) are pre-committed to `artifacts/results/` — you can inspect them without retraining.

---

## What the Code Computes Beyond the Paper

Several metrics are computed but not reported in the paper — see [`NON_PAPER_CONTENT_AUDIT.md`](NON_PAPER_CONTENT_AUDIT.md) for the full list. Highlights:

| Task | Unreported metric | Strongest number |
|------|------------------|-----------------|
| T02 | Feature drop 30% accuracy | E1=**71.6%** vs B0=33.0% (+38.6pp) |
| T02 | Edge drop 30% accuracy | E1=**76.4%** vs VAT=53.8% (+22.6pp) |
| T05 | PCK@0.10 (second threshold) | E1=**78.6%** at clean |
| T08 | Scale universality (66M/110M/340M BERT) | Blind spot present at all scales |
| T08 | Task fine-tuning blind spot | para\_drift: ERM > pretrained > PMH ✓ |

---

## Citation

```bibtex
@article{rajput2025supervised,
  title   = {Supervised Training is Geometrically Incomplete: A Theorem on
             Encoder Non-Isometry, the Jacobian Blind Spot, and its Minimal
             Fix via Gaussian Perturbation Matching},
  author  = {Vishal Rajput},
  year    = {2025},
  note    = {Preprint. ICML 2025 submission.}
}
```

---

## License

Code is released for research reproducibility. Model weights are excluded from this repository due to size.
