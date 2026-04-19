"""
Robust Re-ID evaluation: task metrics and embedding stability under perturbations.

This script sits between the pure-benchmark eval (Market-1501 rank-1/mAP on clean
data) and the MOT experiments that directly stress PMH under heavy corruptions.

For a given checkpoint (or B0/B1/E1 from runs/), we:
- Evaluate standard Market-1501 metrics on clean query/gallery.
- Re-evaluate after applying controlled perturbations to QUERY images only
  (gallery stays clean), for several corruption types/strengths.
- For each perturbation, also measure embedding-level stability:
    mean ||phi_clean - phi_pert||_2 across all queries.

Output:
- Pretty-printed table to stdout.
- JSON file with full results (per run, per perturbation).
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch


def set_global_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False

from model import get_model
from data import (
    get_eval_loaders,
    find_market1501_root,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from eval import compute_rank1_map

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _plot_robust_results(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
) -> None:
    """
    all_results: run_name -> pert_name -> {rank1, mAP, embedding_mean, ...}
    Saves: robust_rank1.png, robust_mAP.png, robust_embedding_stability.png
    """
    if not _HAS_MPL or not all_results:
        return
    pert_names = list(next(iter(all_results.values())).keys())
    run_names = list(all_results.keys())
    n_pert = len(pert_names)
    n_runs = len(run_names)
    x = np.arange(n_pert)
    width = 0.8 / max(n_runs, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_runs, 1)))

    # 1) Rank-1 by perturbation
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, run in enumerate(run_names):
        vals = [all_results[run][p]["rank1"] for p in pert_names]
        offset = (i - (n_runs - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=run, color=colors[i])
    ax.set_ylabel("Rank-1 (%)", fontsize=11)
    ax.set_title("Robust Re-ID: Rank-1 under perturbations (query only)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pert_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "robust_rank1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) mAP by perturbation
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, run in enumerate(run_names):
        vals = [all_results[run][p]["mAP"] for p in pert_names]
        offset = (i - (n_runs - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=run, color=colors[i])
    ax.set_ylabel("mAP (%)", fontsize=11)
    ax.set_title("Robust Re-ID: mAP under perturbations (query only)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pert_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "robust_mAP.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) Embedding stability ||Δφ|| (lower = more stable)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, run in enumerate(run_names):
        vals = [all_results[run][p]["embedding_mean"] for p in pert_names]
        offset = (i - (n_runs - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=run, color=colors[i])
    ax.set_ylabel("Mean ||φ_clean − φ_pert||₂", fontsize=11)
    ax.set_title("Embedding stability under perturbations (lower = more stable)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pert_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "robust_embedding_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plots saved to {out_dir}: robust_rank1.png, robust_mAP.png, robust_embedding_stability.png")


def _denorm(img: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalized tensor(s) back to [0, 1] RGB."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img * std + mean


def _renorm(img: torch.Tensor) -> torch.Tensor:
    """Convert [0, 1] RGB tensor(s) back to ImageNet-normalized."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = img.clamp(0.0, 1.0)
    return (img - mean) / std


def perturb_gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise in pixel space: denorm → noise → clamp → renorm."""
    if sigma <= 0:
        return img
    x = _denorm(img)
    noise = torch.randn_like(x) * sigma
    x = torch.clamp(x + noise, 0.0, 1.0)
    x = _renorm(x)
    return x if img.dim() == x.dim() else x.squeeze(0)


def perturb_brightness(img: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale brightness in pixel space, then renormalize."""
    x = _denorm(img)
    x = torch.clamp(x * factor, 0.0, 1.0)
    x = _renorm(x)
    return x if img.dim() == x.dim() else x.squeeze(0)


def perturb_contrast(img: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust contrast in pixel space."""
    x = _denorm(img)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = torch.clamp((x - mean) * factor + mean, 0.0, 1.0)
    x = _renorm(x)
    return x if img.dim() == x.dim() else x.squeeze(0)


def perturb_blur(img: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Simple average blur in pixel space."""
    x = _denorm(img)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    x = _renorm(x)
    return x if img.dim() == x.dim() else x.squeeze(0)


def perturb_occlusion(img: torch.Tensor, ratio: float) -> torch.Tensor:
    """Random rectangular occlusion covering ~ratio of height/width."""
    if ratio <= 0:
        return img
    x = _denorm(img)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    _, _, h, w = x.shape
    oh, ow = int(h * ratio), int(w * ratio)
    if oh > 0 and ow > 0:
        top = torch.randint(0, max(1, h - oh), (1,)).item()
        left = torch.randint(0, max(1, w - ow), (1,)).item()
        x[:, :, top : top + oh, left : left + ow] = 0.0
    x = _renorm(x)
    return x if img.dim() == x.dim() else x.squeeze(0)


def perturb_identity(img: torch.Tensor) -> torch.Tensor:
    """No-op perturbation (clean baseline)."""
    return img


def extract_embeddings_with_perturb(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    perturb_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract L2-normalized embeddings for all samples in loader, optionally
    applying a perturbation to the images before the forward pass.
    Returns (embeddings, pids, cams) as numpy arrays.
    """
    model.eval()
    embs, pids, cams = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                img, pid, cam, _ = batch
            else:
                img, pid, cam = batch[0], batch[1], batch[2]
            img = img.to(device, non_blocking=True)
            img = perturb_fn(img)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                emb = model.get_embedding(img, normalize=True)
            embs.append(emb.cpu().numpy())
            pids.append(pid.numpy() if torch.is_tensor(pid) else np.asarray(pid))
            cams.append(cam.numpy() if torch.is_tensor(cam) else np.asarray(cam))
    if not embs:
        return np.zeros((0, 1)), np.array([]), np.array([])
    return (
        np.vstack(embs),
        np.concatenate(pids),
        np.concatenate(cams) if cams else np.array([]),
    )


def evaluate_model_with_perturbations(
    model: torch.nn.Module,
    query_loader: torch.utils.data.DataLoader,
    gallery_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    exclude_same_cam: bool,
    perturbations: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
) -> Dict[str, Dict[str, float]]:
    """
    For a single model, evaluate clean + perturbed queries against clean gallery.

    Returns:
        Dict[pert_name] -> {
            'rank1', 'mAP', 'rank5', 'rank10', 'embedding_mean'
        }
    """
    # First, extract clean query/gallery embeddings once
    model.eval()
    # Clean queries
    q_clean_emb, q_pid, q_cam = extract_embeddings_with_perturb(
        model, query_loader, device, use_amp, perturb_identity
    )
    # Clean gallery
    g_clean_emb, g_pid, g_cam = extract_embeddings_with_perturb(
        model, gallery_loader, device, use_amp, perturb_identity
    )

    results: Dict[str, Dict[str, float]] = {}

    for name, fn in perturbations.items():
        if name == "clean":
            q_emb = q_clean_emb
            emb_mean = 0.0
        else:
            q_emb, q_pid_p, q_cam_p = extract_embeddings_with_perturb(
                model, query_loader, device, use_amp, fn
            )
            # Sanity: align lengths for stability computation
            n = min(len(q_clean_emb), len(q_emb))
            if n > 0:
                diff = q_clean_emb[:n] - q_emb[:n]
                # L2 distance per sample
                d = np.linalg.norm(diff, axis=1)
                emb_mean = float(d.mean())
            else:
                emb_mean = 0.0
            # Use the pid/cam from perturbed pass (should be identical)
            q_pid, q_cam = q_pid_p, q_cam_p

        r1, mAP, r5, r10 = compute_rank1_map(
            q_emb,
            q_pid,
            q_cam,
            g_clean_emb,
            g_pid,
            g_cam,
            exclude_same_cam=exclude_same_cam,
            return_rank5_10=True,
        )
        results[name] = {
            "rank1": float(r1),
            "mAP": float(mAP),
            "rank5": float(r5),
            "rank10": float(r10),
            "embedding_mean": float(emb_mean),
        }

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint to evaluate")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Compare B0, B1, E1 from runs/ under perturbations",
    )
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="eval_out_robust")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument(
        "--no_exclude_same_cam",
        action="store_true",
        help="Disable Market-1501 same-camera exclusion",
    )
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for query/gallery")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (occlusion positions, Gaussian noise in eval, workers)")
    p.add_argument("--deterministic", action="store_true", help="cudnn deterministic mode")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed, deterministic=args.deterministic)
    use_amp = not args.no_amp and device.type == "cuda"

    root = find_market1501_root(args.data_dir)
    query_loader, gallery_loader, query_ds, gallery_ds = get_eval_loaders(
        root, num_workers=args.num_workers, seed=args.seed
    )
    if len(query_ds) == 0 or len(gallery_ds) == 0:
        raise RuntimeError(
            f"Empty eval set: query={len(query_ds)}, gallery={len(gallery_ds)}. "
            f"Ensure {root}/query/ and {root}/bound_box_test/ contain images."
        )

    def num_classes_from_ckpt(ckpt_path: Path) -> int:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "classifier.weight" in sd:
            return sd["classifier.weight"].shape[0]
        return 751  # Market-1501 default

    exclude_same_cam = not args.no_exclude_same_cam

    # Define perturbation suite (queries only; gallery stays clean)
    perturbations: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "clean": perturb_identity,
        "gaussian_0.05": lambda x: perturb_gaussian_noise(x, 0.05),
        "gaussian_0.10": lambda x: perturb_gaussian_noise(x, 0.10),
        "brightness_0.5": lambda x: perturb_brightness(x, 0.5),
        "brightness_1.5": lambda x: perturb_brightness(x, 1.5),
        "occlusion_0.2": lambda x: perturb_occlusion(x, 0.2),
        "contrast_0.8": lambda x: perturb_contrast(x, 0.8),
        "contrast_1.2": lambda x: perturb_contrast(x, 1.2),
        "blur_3": lambda x: perturb_blur(x, 3),
    }

    os.makedirs(args.out_dir, exist_ok=True)

    if args.compare:
        runs = ["B0", "VAT", "E1"]
        all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for run in runs:
            ckpt = Path(args.runs_dir) / run / "best.pt"
            if not ckpt.exists():
                print(f"Skip {run}: {ckpt} not found")
                continue
            num_classes = num_classes_from_ckpt(ckpt)
            model = get_model(num_classes, embed_dim=args.embed_dim, pretrained=False)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model = model.to(device)
            print(f"\n=== {run}: robust Re-ID evaluation ===")
            run_results = evaluate_model_with_perturbations(
                model,
                query_loader,
                gallery_loader,
                device,
                use_amp,
                exclude_same_cam=exclude_same_cam,
                perturbations=perturbations,
            )
            all_results[run] = run_results
            for name, r in run_results.items():
                print(
                    f"  {run} / {name:12s}  "
                    f"rank-1={r['rank1']:.2f}%  mAP={r['mAP']:.2f}%  "
                    f"rank-5={r['rank5']:.2f}%  rank-10={r['rank10']:.2f}%  "
                    f"||Δφ||={r['embedding_mean']:.4f}"
                )

        # Shift-suite summary: average and worst-case drops from clean.
        summary = {}
        for run, rr in all_results.items():
            clean_r1 = rr.get("clean", {}).get("rank1", 0.0)
            clean_map = rr.get("clean", {}).get("mAP", 0.0)
            pert_names = [k for k in rr.keys() if k != "clean"]
            if pert_names:
                avg_r1 = float(np.mean([rr[p]["rank1"] for p in pert_names]))
                avg_map = float(np.mean([rr[p]["mAP"] for p in pert_names]))
                worst_r1 = float(min(rr[p]["rank1"] for p in pert_names))
                worst_map = float(min(rr[p]["mAP"] for p in pert_names))
            else:
                avg_r1 = clean_r1
                avg_map = clean_map
                worst_r1 = clean_r1
                worst_map = clean_map
            summary[run] = {
                "clean_rank1": clean_r1,
                "clean_mAP": clean_map,
                "avg_shift_rank1": avg_r1,
                "avg_shift_mAP": avg_map,
                "worst_shift_rank1": worst_r1,
                "worst_shift_mAP": worst_map,
                "drop_rank1_worst": float(clean_r1 - worst_r1),
                "drop_mAP_worst": float(clean_map - worst_map),
            }

        out_path = Path(args.out_dir) / "compare_results_robust.json"
        payload = dict(all_results)
        payload["__summary__"] = summary
        payload["seed"] = args.seed
        payload["deterministic"] = bool(args.deterministic)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved robust comparison to {out_path}")
        _plot_robust_results(all_results, Path(args.out_dir))
        return

    # Single checkpoint mode
    if not args.ckpt or not os.path.isfile(args.ckpt):
        print(
            "Usage:\n"
            "  python eval_robust.py --ckpt runs/E1/best.pt --data_dir ./data\n"
            "  python eval_robust.py --compare --data_dir ./data\n"
        )
        return

    ckpt_path = Path(args.ckpt)
    num_classes = num_classes_from_ckpt(ckpt_path)
    model = get_model(num_classes, embed_dim=args.embed_dim, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)

    results = evaluate_model_with_perturbations(
        model,
        query_loader,
        gallery_loader,
        device,
        use_amp,
        exclude_same_cam=exclude_same_cam,
        perturbations=perturbations,
    )

    print(f"\n=== Robust Re-ID for {ckpt_path} ===")
    for name, r in results.items():
        print(
            f"  {name:12s}  "
            f"rank-1={r['rank1']:.2f}%  mAP={r['mAP']:.2f}%  "
            f"rank-5={r['rank5']:.2f}%  rank-10={r['rank10']:.2f}%  "
            f"||Δφ||={r['embedding_mean']:.4f}"
        )

    out_single = Path(args.out_dir) / "eval_robust_single.json"
    with open(out_single, "w") as f:
        json.dump(
            {
                "ckpt": str(ckpt_path),
                "seed": args.seed,
                "deterministic": bool(args.deterministic),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved robust results to {out_single}")
    # Single run as one "run" for plotting
    run_label = ckpt_path.stem or "model"
    _plot_robust_results({run_label: results}, Path(args.out_dir))


if __name__ == "__main__":
    main()

