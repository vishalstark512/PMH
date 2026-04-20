# Numbers the replication tree should recover

Source: preprint. Exact floats can vary slightly with hardware unless `--deterministic` is used; headline metrics are the contract.

## Cross-task headline table (`tab:crosstask`)

| Task | Domain | Primary metric | B0 | VAT | PMH (E1 or noted) |
|------|--------|----------------|-----|-----|-------------------|
| 01 | CIFAR-10 cls. | acc @ σ=0.1 | 40.39 | 70.59 | **81.36** |
| 02 | Graph cls. | acc @ σ=0.1 | 57.86 | 67.50 | **78.04** |
| 03 | Mol. reg. | MAE ↓ (E1_node = node-feature PMH) | 25.06 | 28.51 | **23.62** |
| 04 | ViT cls. | L6 probe (table) | 54.75 | 64.30 | **76.15** |
| 05 | Pose | PCK@0.05 | 28.5 | 8.7 | **28.6** |
| 06 | Re-ID | avg-shift rank-1 | 26.81 | 57.21 | **58.89** |
| 07 | Chest X-ray | worst-shift acc | 0.625 | 0.686 | **0.742** |

**Task 03 note:** The paper’s “E1_node” is the QM9 run with **PMH on node features** (not position-only). The repo matches this via `tasks/run_all_tasks.py` Task03 flags: `--node_noise 0.1`, `--noise_std 0.15`, `--pmh_max_weight 1.0`, etc., and stores checkpoints under `runs/QM9/E1/` (name “E1” in code, “E1_node” in prose).

## Task 03 appendix (`tab:a11_mol`) — MAE vs position noise σ (Å)

Rows: σ ∈ {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}. Columns: B0, VAT, E1 **with the original position-noise training** (table caption). For the **clean headline 23.62 / 25.06 / 28.51** use **`run_evals.py`** with **`--node_noise 0.1`** so eval matches the Chemistry-style setup (same as `run_all_tasks` `eval_robust`).

Published table (position-noise sweep, mean-over-elements style in TeX):

| σ | B0 | VAT | E1 |
|---|-----|-----|-----|
| 0.000 | 64.72 | **35.12** | 44.18 |
| 0.005 | 64.96 | **34.99** | 44.44 |
| 0.010 | 65.30 | **35.21** | 44.16 |
| 0.020 | 65.20 | **35.50** | 44.53 |
| 0.050 | 66.32 | **36.50** | 44.83 |
| 0.100 | 70.18 | **40.47** | 47.46 |
| 0.200 | 79.00 | **51.20** | 54.64 |

## Task 03 embedding drift (`tab:a12_drift`)

| Perturbation | B0 | VAT | E1 |
|--------------|-----|-----|-----|
| Gaussian σ=0.1 | 1.168 | 0.548 | **0.547** |

Replicate with `run_evals.py` / `evaluate_with_drift` at σ=0.1 and the same `node_noise` as training eval.

## This folder (`replication_seeded/`) — paper-only scope

Only scripts needed to reproduce reported paper results are kept.

- **Task 01 (`tab:crosstask` T01):** `tasks/01_image_classification/CIFAR10/{train.py,model.py}`.
- **Task 02 (`tab:crosstask` T02):** `tasks/02_graph_classification/{train.py,eval.py,data.py,model.py}`.
- **Task 03 (`tab:crosstask` T03 + `tab:a11_mol` + `tab:a12_drift`):** `tasks/03_molecular_regression/{train.py,eval.py,data.py,model.py,perturb.py,run_evals.py,embedding_analysis.py}`.
- **Task 04 (`tab:crosstask` T04 + mechanistic/appendix tables):** `tasks/04_vision_transformer_pmh/{train.py,interp.py,topological_distortion_index.py,jacobian_norm.py,linear_probe_analysis.py,eval_corruptions.py,eval_adversarial.py,check_bound_tightness.py,nuisance_subspace_bound.py,retrain_with_L_tracking.py,train_multiscale_pmh.py,model.py}`.
- **Task 05 (`tab:crosstask` T05 + `tab:a10_pose` + `tab:a12_drift`):** `tasks/05_pose_estimation_pmh/{data.py,train.py,eval.py,embedding_stability.py,model.py,geometry.py,pmh_loss.py,train_both_gpus.py}`.
- **Task 06 (`tab:crosstask` T06 + `tab:a8_reid` + `tab:a12_drift`):** `tasks/06_reid_pmh/{data.py,train.py,eval.py,eval_robust.py,model.py}`.
- **Task 07 (`tab:crosstask` T07 + `tab:a9_xray` + `tab:a12_drift`):** `tasks/07_chestxray_pmh/{data.py,train.py,eval.py,eval_robust.py,model.py,interp_resnet.py,saliency_stability.py,compare_embed_only.py}`.
- **Task 08 (BERT SST-2, `tab:a3_bert`, Figure A4):** `tasks/08_bert_sst2/{train.py,measure_tdi.py,exp_semantic_tdi/measure_semantic_tdi.py}`.
- **Task 09 (ImageNet ViT-B/16, `tab:a5_imagenet`):** `tasks/09_imagenet_vit/{measure_tdi.py,clean_hf_imagenet_cache.py,sort_imagenet_validation.py}`.
- **Shared utility for Task 08/09:** `tasks/shared/tdi.py`.

Reproducibility note: replicated train/eval scripts include `--seed` and `--deterministic` where applicable.

Operational note: run each task with one command via `tasks/run_task.py --task <id> --data_dir <path>`. Outputs are standardized under each task:
- models/checkpoints: `tasks/<task_name>/artifacts/models/`
- metrics/plots/json: `tasks/<task_name>/artifacts/results/`
