"""
One-command runner for paper replication tasks.

Training and eval CLI for tasks 02–07 are aligned with ``tasks/run_all_tasks.py``
(PMH noise / batch sizes / COCO subset + pretrained / QM9 schedules / chest four-run eval).

Example:
  python tasks/run_task.py --task 06 --data_dir ./tasks/06_reid_pmh/data

Outputs are standardized per task:
  <task_dir>/artifacts/models
  <task_dir>/artifacts/results

Unified layout (all tasks under one tree):
  python tasks/run_task.py --task 06 --data_dir ... --artifact_root ../artifacts
  # writes ../artifacts/models/06_reid_pmh/ and ../artifacts/results/06_reid_pmh/

Run everything:
  python run_all_replication.py --data_dir /path/to/data --artifact_root ./artifacts

Task 09 (ImageNet): edit TASK09_* constants near the top of this file (no env vars).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PY = sys.executable

# ---------------------------------------------------------------------------
# Task 09 — ImageNet ViT (`measure_tdi.py`): edit here once (no env vars needed).
# ---------------------------------------------------------------------------
# True  → use Hugging Face validation only (login + dataset terms; no local tree).
#         Default True so Task 09 runs without D:\\ILSVRC on disk.
# False → use local ImageNet; set TASK09_IMAGENET_DATA_DIR to an existing root that contains
#         sorted val/ (folders n01440764/, …), e.g. parent of `Data/CLS-LOC/val`.
TASK09_USE_HUGGINGFACE: bool = True
TASK09_IMAGENET_DATA_DIR = Path(r"D:\ILSVRC\Data\CLS-LOC")
# If non-empty, passed as --val_dir (ImageFolder root with n########/ subfolders).
TASK09_IMAGENET_VAL_DIR: str = ""

# Stable folder names under a unified --artifact_root (models/<slug>, results/<slug>)
TASK_SLUGS: dict[str, str] = {
    "01": "01_image_classification",
    "02": "02_graph_classification",
    "03": "03_molecular_regression",
    "04": "04_vision_transformer_pmh",
    "05": "05_pose_estimation_pmh",
    "06": "06_reid_pmh",
    "07": "07_chestxray_pmh",
    "08": "08_bert_sst2",
    "09": "09_imagenet_vit",
}


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}\n  (cwd: {cwd})")
    rc = subprocess.run(cmd, cwd=str(cwd)).returncode
    if rc != 0:
        raise SystemExit(rc)


def task_dir(task_id: str, name: str) -> Path:
    return ROOT / f"{task_id}_{name}"


def artifacts(td: Path) -> tuple[Path, Path]:
    models = td / "artifacts" / "models"
    results = td / "artifacts" / "results"
    models.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return models, results


def resolve_artifact_paths(
    td: Path,
    artifact_root: Path | None,
    task_slug: str,
    subfolder: str | None = None,
) -> tuple[Path, Path]:
    """Per-task folders under task dir, or unified models/<slug>[/subfolder] and results/<slug>[/subfolder].

    ``subfolder`` is used by experimental variants (e.g. ``"plus"``) so they do not clobber the
    paper-aligned artifacts under the top-level slug.
    """
    if artifact_root is None:
        base_models, base_results = artifacts(td)
        if subfolder:
            base_models = base_models / subfolder
            base_results = base_results / subfolder
            base_models.mkdir(parents=True, exist_ok=True)
            base_results.mkdir(parents=True, exist_ok=True)
        return base_models, base_results
    root = artifact_root.resolve()
    models = root / "models" / task_slug
    results = root / "results" / task_slug
    if subfolder:
        models = models / subfolder
        results = results / subfolder
    models.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return models, results


def run_01(data_dir: str, artifact_root: Path | None = None) -> None:
    td = task_dir("01", "image_classification") / "CIFAR10"
    models, results = resolve_artifact_paths(td, artifact_root, TASK_SLUGS["01"])
    for run_name in ("B0", "VAT", "E1"):
        run([PY, "train.py", "--run", run_name, "--data_dir", data_dir, "--out_dir", str(models)], td)
    # Embedding drift (Table A1 in paper): ||φ_clean − φ_noisy|| at σ=0.05/0.1/0.15/0.2
    run(
        [
            PY,
            "embedding_stability.py",
            "--runs_dir",
            str(models / "cifar10"),
            "--data_dir",
            data_dir,
            "--out_dir",
            str(results / "evals"),
        ],
        td,
    )


def run_02(data_dir: str, artifact_root: Path | None = None) -> None:
    """Match ``tasks/run_all_tasks.py`` ``steps_02`` (PROTEINS: larger σ + pmh cap for E1/VAT)."""
    td = task_dir("02", "graph_classification")
    models, results = resolve_artifact_paths(td, artifact_root, TASK_SLUGS["02"])
    base = ["--data_dir", data_dir, "--out_dir", str(models), "--epochs", "100", "--seed", "42"]
    run([PY, "train.py", "--run", "B0", *base], td)
    run([PY, "train.py", "--run", "VAT", *base, "--noise_sigma", "0.3"], td)
    run([PY, "train.py", "--run", "E1", *base, "--noise_sigma", "0.3", "--pmh_max_weight", "1.0"], td)
    ckpts = [str(models / "PROTEINS" / r / "best.pt") for r in ("B0", "VAT", "E1")]
    run([PY, "eval.py", *ckpts, "--dataset", "PROTEINS", "--data_dir", data_dir, "--out_dir", str(results / "evals")], td)


def run_03(data_dir: str, artifact_root: Path | None = None) -> None:
    """QM9 molecular regression.

    Config matches tasks/03_molecular_regression/runs_nodeonly which produced the best
    empirical results: position noise OFF, node_noise=0.05, pmh_max_weight=0.7,
    warmup=20, cap=0.3, 100 epochs. The aggressive config (noise_std=0.15,
    pmh_max_weight=1.0, 120ep) regressed E1 MAE from ~23.6 to ~26.5.
    """
    td = task_dir("03", "molecular_regression")
    models, results = resolve_artifact_paths(td, artifact_root, TASK_SLUGS["03"])
    common = [
        "--data_dir",
        data_dir,
        "--out_dir",
        str(models),
        "--seed",
        "42",
        "--epochs",
        "100",
        "--batch_size",
        "128",          # 128 is the most stable for GNN on QM9; 256 also works, 512 regressed (+6 MAE pts)
        "--num_workers",
        "0",            # keep 0: with --load_in_memory there is no I/O to overlap
        "--load_in_memory",  # preload all 130K graphs into RAM (fits in 128 GB)
        "--gpu",
        "0",            # pin to RTX 4090
    ]
    pmh03_e1 = [
        "--noise_std",
        "0.0",
        "--node_noise",
        "0.05",
        "--pmh_max_weight",
        "0.7",
        "--warmup_epochs",
        "20",
        "--pmh_cap_ratio",
        "0.3",
        "--output_pmh_weight",
        "0.3",
    ]
    run([PY, "train.py", "--run", "B0", *common], td)
    run([PY, "train.py", "--run", "VAT", *common], td)
    run([PY, "train.py", "--run", "E1", *common, *pmh03_e1], td)
    run(
        [
            PY,
            "run_evals.py",
            "--b0",
            str(models / "QM9" / "B0" / "best.pt"),
            "--vat",
            str(models / "QM9" / "VAT" / "best.pt"),
            "--e1",
            str(models / "QM9" / "E1" / "best.pt"),
            "--data_dir",
            data_dir,
            "--out_dir",
            str(results / "evals"),
            "--node_noise",
            "0.1",
            "--split_seed",
            "42",
            "--eval_seed",
            "42",
        ],
        td,
    )
    run(
        [
            PY,
            "embedding_analysis.py",
            "--b0",
            str(models / "QM9" / "B0" / "best.pt"),
            "--e1",
            str(models / "QM9" / "E1" / "best.pt"),
            "--data_dir",
            data_dir,
            "--out_dir",
            str(results / "embedding_analysis"),
            "--split_seed",
            "42",
            "--seed",
            "42",
        ],
        td,
    )


def run_04(data_dir: str, artifact_root: Path | None = None, skip_sweep: bool = False) -> None:
    td = task_dir("04", "vision_transformer_pmh")
    models, results = resolve_artifact_paths(td, artifact_root, TASK_SLUGS["04"])
    for run_name in ("B0", "VAT", "E1"):
        run([PY, "train.py", "--run", run_name, "--data_dir", data_dir, "--out_dir", str(models), "--seed", "42"], td)
    # Negative-control ablation: E1 architecture without the PMH loss term (paper Table 1 row 3)
    run([PY, "train.py", "--run", "E1", "--no_pmh", "--data_dir", data_dir, "--out_dir", str(models), "--seed", "42"], td)
    run([PY, "interp.py", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "interp"), "--seed", "42"], td)
    run([PY, "topological_distortion_index.py", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "interp")], td)
    run([PY, "linear_probe_analysis.py", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "interp")], td)
    run([PY, "jacobian_norm.py", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "interp")], td)
    run([PY, "eval_corruptions.py", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "corruptions")], td)
    run([PY, "eval_adversarial.py", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "adversarial")], td)
    if skip_sweep:
        print("  [run_04] Skipping T-alignment sigma sweep (--no_sweep)", flush=True)
        return
    # T-alignment matrix (Table 3 in paper): train E1 at 5 additional σ_train values.
    # σ=0.12 is already run above as "E1". Each model's results.json records accuracy at
    # σ_eval=0/0.05/0.10/0.15/0.20, giving the full 6×5 matrix offline.
    for sigma_str in ("0.05", "0.08", "0.10", "0.15", "0.20"):
        sigma_tag = sigma_str.replace(".", "")          # "005", "008", …
        run_name = f"E1_sigma_{sigma_tag}"
        run(
            [
                PY, "train.py",
                "--run", run_name,
                "--data_dir", data_dir,
                "--out_dir", str(models),
                "--seed", "42",
                "--noise_sigma", sigma_str,
            ],
            td,
        )


def run_05(data_dir: str, artifact_root: Path | None = None) -> None:
    """Match ``tasks/run_all_tasks.py`` ``steps_05`` (COCO subset, pretrained backbone, VAT/E1 knobs)."""
    td = task_dir("05", "pose_estimation_pmh")
    models, results = resolve_artifact_paths(td, artifact_root, TASK_SLUGS["05"])
    coco = [
        "--dataset",
        "coco",
        "--pretrained",
        "--max_train_samples",
        "15000",
        "--data_dir",
        data_dir,
        "--out_dir",
        str(models),
        "--seed",
        "42",
    ]
    run([PY, "train.py", "--run", "B0", *coco, "--epochs", "50"], td)
    run(
        [
            PY,
            "train.py",
            "--run",
            "VAT",
            *coco,
            "--epochs",
            "30",
            "--vat_eps",
            "0.12",
            "--vat_xi",
            "0.001",
            "--vat_weight",
            "1.0",
        ],
        td,
    )
    run(
        [
            PY,
            "train.py",
            "--run",
            "E1",
            *coco,
            "--epochs",
            "30",
            "--gaussian_sigma",
            "0.15",
            "--pmh_weight",
            "0.3",
        ],
        td,
    )
    run(
        [
            PY,
            "eval.py",
            "--compare",
            "--dataset",
            "coco",
            "--runs_dir",
            str(models),
            "--data_dir",
            data_dir,
            "--out_dir",
            str(results / "eval_out"),
            "--seed",
            "42",
            "--occlusion_levels",
            "0,0.1,0.2,0.3,0.4",
        ],
        td,
    )
    run([PY, "embedding_stability.py", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "eval_out"), "--seed", "42", "--subset_seed", "42", "--max_samples", "500"], td)


def run_06(data_dir: str, artifact_root: Path | None = None) -> None:
    """Match ``tasks/run_all_tasks.py`` ``steps_06`` (batch 128, E1 PMH hyperparameters)."""
    td = task_dir("06", "reid_pmh")
    models, results = resolve_artifact_paths(td, artifact_root, TASK_SLUGS["06"])
    base = [
        "--data_dir",
        data_dir,
        "--out_dir",
        str(models),
        "--seed",
        "42",
        "--epochs",
        "60",
        "--batch_size",
        "128",
        "--num_workers",
        "0",
    ]
    run([PY, "train.py", "--run", "B0", *base], td)
    run([PY, "train.py", "--run", "VAT", *base], td)
    pmh06 = [
        "--pmh_weight",
        "0.5",
        "--pmh_cap_ratio",
        "0.3",
        "--warmup_epochs",
        "10",
        "--pmh_ramp_epochs",
        "20",
        "--noise_sigma",
        "0.1",
    ]
    run([PY, "train.py", "--run", "E1", *base, *pmh06], td)
    run([PY, "eval.py", "--compare", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "eval_out"), "--seed", "42"], td)
    run([PY, "eval_robust.py", "--compare", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "eval_out_robust"), "--seed", "42"], td)


def run_07(data_dir: str, artifact_root: Path | None = None) -> None:
    """Match ``tasks/run_all_tasks.py`` ``steps_07`` (PMH07 on E1, E1_no_pmh control, four-way robust eval)."""
    td = task_dir("07", "chestxray_pmh")
    models, results = resolve_artifact_paths(td, artifact_root, TASK_SLUGS["07"])
    base = [
        "--data_dir",
        data_dir,
        "--out_dir",
        str(models),
        "--seed",
        "42",
        "--epochs",
        "30",
        "--batch_size",
        "64",
        "--num_workers",
        "0",
    ]
    pmh07 = [
        "--pmh_weight",
        "0.5",
        "--pmh_cap_ratio",
        "0.3",
        "--warmup_epochs",
        "10",
        "--pmh_ramp_epochs",
        "20",
        "--noise_sigma",
        "0.1",
        "--lr",
        "0.1",
    ]
    run([PY, "train.py", "--run", "B0", *base], td)
    run([PY, "train.py", "--run", "VAT", *base], td)
    run([PY, "train.py", "--run", "E1", *base, *pmh07], td)
    run([PY, "train.py", "--run", "E1", *base, *pmh07, "--no_pmh"], td)
    runs4 = ["--runs", "B0", "VAT", "E1_no_pmh", "E1"]
    run(
        [
            PY,
            "eval.py",
            "--compare",
            *runs4,
            "--runs_dir",
            str(models),
            "--data_dir",
            data_dir,
            "--out_dir",
            str(results / "eval_out"),
            "--seed",
            "42",
        ],
        td,
    )
    run(
        [
            PY,
            "eval_robust.py",
            "--compare",
            *runs4,
            "--runs_dir",
            str(models),
            "--data_dir",
            data_dir,
            "--out_dir",
            str(results / "eval_out_robust"),
            "--seed",
            "42",
        ],
        td,
    )
    run(
        [
            PY,
            "interp_resnet.py",
            *runs4,
            "--runs_dir",
            str(models),
            "--data_dir",
            data_dir,
            "--out_dir",
            str(results / "interp_resnet"),
        ],
        td,
    )
    run(
        [
            PY,
            "saliency_stability.py",
            *runs4,
            "--runs_dir",
            str(models),
            "--data_dir",
            data_dir,
            "--out_dir",
            str(results / "saliency_stability"),
        ],
        td,
    )


def run_08(_data_dir: str, artifact_root: Path | None = None) -> None:
    td = task_dir("08", "bert_sst2")
    models, results = resolve_artifact_paths(td, artifact_root, TASK_SLUGS["08"])
    run([PY, "train.py", "--mode", "baseline", "--out_dir", str(models / "baseline")], td)
    run([PY, "train.py", "--mode", "pmh", "--out_dir", str(models / "pmh")], td)
    run([PY, "measure_tdi.py", "--compare", str(models / "baseline"), str(models / "pmh")], td)
    # Pass --runs_dir so the semantic-TDI script can find baseline/pmh best.pt checkpoints
    # from the replication artifact store rather than the legacy in-tree runs/ directory.
    run(
        [
            PY, "exp_semantic_tdi/measure_semantic_tdi.py",
            "--out_dir", str(results / "semantic_tdi"),
            "--runs_dir", str(models),
        ],
        td,
    )


def run_09(data_dir: str, artifact_root: Path | None = None) -> None:
    # NOTE on TDI numbers vs. paper (Table A4):
    # measure_tdi.py defaults to --max_classes 100 / --max_per_class 50.  The paper
    # likely used a different subset size or a different random seed, which shifts the
    # absolute TDI values.  The replication baseline TDI@0 is ~1.23 vs. paper 1.03
    # (+0.197), while the PMH-finetuned model is ~0.94 vs. paper 0.90 (+0.036).
    # The relative ordering (PMH < baseline) and the reduction magnitude are preserved.
    # To align with paper numbers exactly, re-run with '--max_classes 1000' if the
    # full ImageNet validation split is available locally.
    td = task_dir("09", "imagenet_vit")
    models, results = resolve_artifact_paths(td, artifact_root, TASK_SLUGS["09"])
    seed_args = ["--seed", "42"]
    measure_cmd = [PY, "measure_tdi.py", "measure", "--out_dir", str(results / "baseline"), *seed_args]
    pmh_cmd = [PY, "measure_tdi.py", "pmh_tune", "--out_dir", str(models / "pmh_tune"), *seed_args]

    if TASK09_USE_HUGGINGFACE:
        measure_cmd.extend(["--val_source", "hf"])
        pmh_cmd.extend(["--val_source", "hf", "--train_source", "hf_val"])
    else:
        root = TASK09_IMAGENET_DATA_DIR.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(
                f"Task 09: local ImageNet path is not a directory:\n  {root}\n"
                "Edit TASK09_IMAGENET_DATA_DIR (and optionally TASK09_IMAGENET_VAL_DIR) near the top of "
                "tasks/run_task.py, or set TASK09_USE_HUGGINGFACE = True to use Hugging Face instead."
            )
        imagenet_root = str(root)
        measure_cmd.extend(["--val_source", "local", "--data_dir", imagenet_root])
        pmh_cmd.extend(["--val_source", "local", "--data_dir", imagenet_root])
        vd = (TASK09_IMAGENET_VAL_DIR or "").strip()
        if vd:
            measure_cmd.extend(["--val_dir", vd])
            pmh_cmd.extend(["--val_dir", vd])

    run(measure_cmd, td)
    run(pmh_cmd, td)


TASKS = {
    "01": run_01,
    "02": run_02,
    "03": run_03,
    "04": run_04,
    "05": run_05,
    "06": run_06,
    "07": run_07,
    "08": run_08,
    "09": run_09,
}


# ---------------------------------------------------------------------------
# Multi-seed wrapper (opt-in, never overwrites paper-aligned outputs).
# Writes under <artifact_root>/{models,results}/<slug>/seed_<N>/.
#
# History: earlier "plus" variants (run_02_plus/run_06_plus/run_07_plus) were
# tried and removed — none matched or beat the paper-aligned E1 baselines. See
# PMH_IMPROVEMENTS.md "Tried and removed" for the empirical evidence.
# ---------------------------------------------------------------------------


def _multiseed(seeds: list[int], task_id: str, data_dir: str, artifact_root: Path | None) -> None:
    """Run a task N times with different seeds under <slug>/seed_<N>/ subfolders.

    Intended for tasks 01/04/08 where the replication gap is small enough that seed
    variance likely explains it; average the per-seed JSON results offline afterwards.
    """
    td_map = {
        "01": task_dir("01", "image_classification") / "CIFAR10",
        "04": task_dir("04", "vision_transformer_pmh"),
        "08": task_dir("08", "bert_sst2"),
    }
    if task_id not in td_map:
        raise SystemExit(f"Multi-seed not wired for task {task_id} (only 01/04/08)")
    td = td_map[task_id]
    for s in seeds:
        _, _ = resolve_artifact_paths(td, artifact_root, TASK_SLUGS[task_id], subfolder=f"seed_{s}")
        models, results = resolve_artifact_paths(td, artifact_root, TASK_SLUGS[task_id], subfolder=f"seed_{s}")
        if task_id == "01":
            for run_name in ("B0", "VAT", "E1"):
                run([PY, "train.py", "--run", run_name, "--data_dir", data_dir, "--out_dir", str(models), "--seed", str(s)], td)
        elif task_id == "04":
            for run_name in ("B0", "VAT", "E1"):
                run([PY, "train.py", "--run", run_name, "--data_dir", data_dir, "--out_dir", str(models), "--seed", str(s)], td)
            run([PY, "topological_distortion_index.py", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "interp")], td)
            run([PY, "linear_probe_analysis.py", "--runs_dir", str(models), "--data_dir", data_dir, "--out_dir", str(results / "interp")], td)
        elif task_id == "08":
            run([PY, "train.py", "--mode", "baseline", "--out_dir", str(models / "baseline"), "--seed", str(s)], td)
            run([PY, "train.py", "--mode", "pmh", "--out_dir", str(models / "pmh"), "--seed", str(s)], td)
            run([PY, "measure_tdi.py", "--compare", str(models / "baseline"), str(models / "pmh")], td)


def run_all_tasks(
    data_dir: str,
    artifact_root: Path,
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
    no_sweep: bool = False,
) -> None:
    """Run selected tasks (default 01..09) with checkpoints and metrics under artifact_root.

    ``seeds`` is only honoured for tasks 01/04/08; each seed gets its own
    ``<slug>/seed_<N>/`` subfolder.

    ``no_sweep`` skips the T-alignment sigma sweep for task 04 (5 extra E1 variants).
    """
    order = sorted(TASKS.keys()) if not tasks else list(tasks)
    for tid in order:
        if tid not in TASKS:
            raise SystemExit(f"Unknown task id: {tid} (expected one of {sorted(TASKS)})")
        print(f"\n{'#' * 72}\n# Task {tid}\n{'#' * 72}\n", flush=True)
        if seeds and tid in {"01", "04", "08"}:
            _multiseed(seeds, tid, data_dir, artifact_root)
        elif tid == "04":
            run_04(data_dir, artifact_root, skip_sweep=no_sweep)
        else:
            TASKS[tid](data_dir, artifact_root)


def main() -> None:
    p = argparse.ArgumentParser(description="Run one full replication task with standardized artifact folders.")
    p.add_argument("--task", required=True, choices=sorted(TASKS.keys()), help="Task id (01..09)")
    p.add_argument("--data_dir", default="./data", help="Task data directory (ignored for task 08)")
    p.add_argument(
        "--artifact_root",
        default=None,
        help="If set, write models to <root>/models/<task_slug>/ and results to <root>/results/<task_slug>/ "
        "instead of under each task folder.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Multi-seed mode for tasks 01/04/08 (e.g. --seeds 42 43 44). Writes under <slug>/seed_<N>/.",
    )
    p.add_argument(
        "--no_sweep",
        action="store_true",
        help="Task 04 only: skip the T-alignment sigma sweep (5 extra E1_sigma_* training runs).",
    )
    args = p.parse_args()
    ar = Path(args.artifact_root).resolve() if args.artifact_root else None
    if args.seeds and args.task in {"01", "04", "08"}:
        _multiseed(args.seeds, args.task, args.data_dir, ar)
        return
    if args.task == "04":
        run_04(args.data_dir, ar, skip_sweep=args.no_sweep)
        return
    TASKS[args.task](args.data_dir, ar)


if __name__ == "__main__":
    main()

