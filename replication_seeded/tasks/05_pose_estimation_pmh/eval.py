"""
Evaluate pose checkpoint(s): 3D (MPJPE, PA-MPJPE, violation rate) or 2D COCO (PCK, MKE).
Single checkpoint or --compare baseline vs E1 across occlusion levels for full robustness.
With --out_dir, saves JSON and robustness plots (PCK / MKE vs occlusion).

Replication: --seed for global RNG (attacks / subsampling consistency with train when matched).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from model import get_model
from data import (
    get_pose_dataloader,
    get_coco_pose_dataloader,
    ensure_coco_pose,
    apply_random_occlusion,
    apply_eval_attack,
    DummyPoseDataset,
)
from geometry import mpjpe, pampjpe, geometric_violation_rate

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def pck_2d(pred, gt, vis, threshold=0.05):
    """Percentage of Correct Keypoints (2D). pred, gt: (N, 17, 3) with x,y in [0,1]. vis: (N, 17)."""
    mask = (vis >= 1).float()
    diff = (pred[:, :, :2] - gt[:, :, :2]).norm(dim=-1)
    correct = ((diff <= threshold).float() * mask).sum()
    total = mask.sum().clamp(min=1)
    return (correct / total).item()


def mke_2d(pred, gt, vis):
    """Mean keypoint error (L2 in normalized coords) over visible keypoints."""
    mask = (vis >= 1).float().unsqueeze(-1)
    diff = (pred[:, :, :2] - gt[:, :, :2]) * mask
    n = mask.sum().clamp(min=1)
    return (diff.norm(dim=-1).sum() / n).item()


def evaluate(model, loader, device, occlusion_ratio=0.0, gaussian_sigma=0.0, use_amp=False):
    """Run model on loader; optionally apply eval attack (occlusion and/or Gaussian)."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for images, pose_gt in loader:
            images = images.to(device, non_blocking=True)
            pose_gt = pose_gt.to(device, non_blocking=True)
            images = apply_eval_attack(images, occlusion_ratio=occlusion_ratio, gaussian_sigma=gaussian_sigma, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                pose_pred = model(images)
            preds.append(pose_pred.cpu())
            gts.append(pose_gt.cpu())
    pred = torch.cat(preds, dim=0)
    gt = torch.cat(gts, dim=0)
    return pred, gt


def load_model_and_eval(ckpt_path, loader, device, occlusion_ratio, gaussian_sigma, use_amp, use_coco):
    """Load one checkpoint, run evaluation at given attack (occ, gaussian). Returns (pred, gt) for COCO else metrics dict for 3D."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = get_model(backbone="resnet18", pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    pred, gt = evaluate(model, loader, device, occlusion_ratio=occlusion_ratio, gaussian_sigma=gaussian_sigma, use_amp=use_amp)
    if use_coco:
        return pred, gt
    return {
        "ckpt": str(ckpt_path),
        "occlusion_ratio": occlusion_ratio,
        "mpjpe_mm": round(mpjpe(pred, gt).item(), 2),
        "pampjpe_mm": round(pampjpe(pred, gt).item(), 2),
        "geometric_violation_rate": round(geometric_violation_rate(pred, gt).item(), 4),
    }


def _build_attack_specs(occlusion_levels, gaussian_sigmas, combined_attacks):
    """Build list of (attack_name, occlusion_ratio, gaussian_sigma) from CLI args."""
    specs = []
    for occ in occlusion_levels:
        if occ == 0:
            specs.append(("clean", 0.0, 0.0))
        else:
            specs.append((f"occ_{occ:.2f}".replace(".", "_"), float(occ), 0.0))
    for s in gaussian_sigmas:
        if s > 0:
            specs.append((f"gauss_{s:.3f}".replace(".", "_"), 0.0, float(s)))
    for occ, sig in combined_attacks:
        name = f"occ_{occ:.2f}_gauss_{sig:.3f}".replace(".", "_")
        specs.append((name, float(occ), float(sig)))
    return specs


def _save_robustness_plots(out_dir, results, occlusion_levels, pck_thresholds, names=None):
    """Save PCK and MKE vs occlusion plots. results: if names is None, single model {occ: {pck@t, mke}}; else {name: {occ: {...}}}."""
    if not _HAS_MATPLOTLIB:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    occ = sorted(occlusion_levels)

    if names is None:
        # Single model: results[occ] = {pck@0.05, pck@0.10, mke} (keys may vary)
        for th in pck_thresholds:
            key = f"pck@{th}" if f"pck@{th}" in next(iter(results.values()), {}) else "pck"
            y = [results.get(o, {}).get(key, results.get(o, {}).get("pck", 0)) * 100 for o in occ]
            plt.figure()
            plt.plot(occ, y, "o-", linewidth=2, markersize=8)
            plt.xlabel("Occlusion ratio")
            plt.ylabel(f"PCK@{th} (%)")
            plt.title("Robustness: PCK vs occlusion")
            plt.grid(True, alpha=0.3)
            plt.xticks(occ)
            plt.tight_layout()
            plt.savefig(out_dir / f"pck_vs_occlusion_{th}.png", dpi=150)
            plt.close()
        mke_vals = [results.get(o, {}).get("mke", 0) for o in occ]
        plt.figure()
        plt.plot(occ, mke_vals, "o-", linewidth=2, markersize=8)
        plt.xlabel("Occlusion ratio")
        plt.ylabel("MKE")
        plt.title("Robustness: MKE vs occlusion")
        plt.grid(True, alpha=0.3)
        plt.xticks(occ)
        plt.tight_layout()
        plt.savefig(out_dir / "mke_vs_occlusion.png", dpi=150)
        plt.close()
        return

    # Compare: names = ["baseline", "E1"]
    for th in pck_thresholds:
        key = f"pck@{th}"
        plt.figure()
        for name in names:
            if name not in results:
                continue
            y = [results[name].get(o, {}).get(key, 0) * 100 for o in occ]
            plt.plot(occ, y, "o-", linewidth=2, markersize=8, label=name)
        plt.xlabel("Occlusion ratio")
        plt.ylabel(f"PCK@{th} (%)")
        plt.title("Robustness: baseline vs E1")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(occ)
        plt.tight_layout()
        plt.savefig(out_dir / f"pck_vs_occlusion_{th}.png", dpi=150)
        plt.close()
    plt.figure()
    for name in names:
        if name not in results:
            continue
        mke_vals = [results[name].get(o, {}).get("mke", 0) for o in occ]
        plt.plot(occ, mke_vals, "o-", linewidth=2, markersize=8, label=name)
    plt.xlabel("Occlusion ratio")
    plt.ylabel("MKE")
    plt.title("Robustness: baseline vs E1 (MKE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(occ)
    plt.tight_layout()
    plt.savefig(out_dir / "mke_vs_occlusion.png", dpi=150)
    plt.close()
    print(f"Saved plots to {out_dir}")


def _save_robustness_bar_charts(out_dir, results, attack_specs, pck_thresholds, names=None):
    """Bar chart: x=attack names, y=metric, grouped by method. One figure per metric."""
    if not _HAS_MATPLOTLIB or not names:
        return
    out_dir = Path(out_dir)
    attack_names = [a[0] for a in attack_specs]
    n_attacks = len(attack_names)
    n_methods = len(names)
    width = 0.8 / max(n_methods, 1)
    x = range(n_attacks)
    for th in pck_thresholds:
        key = f"pck@{th}"
        plt.figure(figsize=(max(8, n_attacks * 0.8), 5))
        for i, name in enumerate(names):
            if name not in results:
                continue
            vals = [results[name].get(an, {}).get(key, 0) * 100 for an in attack_names]
            offset = (i - n_methods / 2 + 0.5) * width
            plt.bar([xi + offset for xi in x], vals, width=width, label=name)
        plt.xlabel("Attack")
        plt.ylabel(f"PCK@{th} (%)")
        plt.title("Robustness: PCK vs attack (all attacks)")
        plt.xticks(x, attack_names, rotation=45, ha="right")
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out_dir / f"pck_vs_attack_{th}.png", dpi=150)
        plt.close()
    plt.figure(figsize=(max(8, n_attacks * 0.8), 5))
    for i, name in enumerate(names):
        if name not in results:
            continue
        vals = [results[name].get(an, {}).get("mke", 0) for an in attack_names]
        offset = (i - n_methods / 2 + 0.5) * width
        plt.bar([xi + offset for xi in x], vals, width=width, label=name)
    plt.xlabel("Attack")
    plt.ylabel("MKE")
    plt.title("Robustness: MKE vs attack (all attacks)")
    plt.xticks(x, attack_names, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "mke_vs_attack.png", dpi=150)
    plt.close()
    print(f"Saved bar charts to {out_dir}")


def run_robustness_comparison(args, loader, device, use_amp):
    """Compare baseline, VAT, E1 across multiple attacks (occlusion, Gaussian, combined). COCO 2D only."""
    occlusion_levels = [float(x) for x in args.occlusion_levels.replace(",", " ").split()]
    gaussian_sigmas = [float(x) for x in args.gaussian_sigmas.replace(",", " ").split()]
    combined_attacks = []
    if (args.combined_attacks or "").strip():
        for pair in args.combined_attacks.strip().split():
            parts = [float(x) for x in pair.split(",")]
            if len(parts) == 2:
                combined_attacks.append((parts[0], parts[1]))
    attack_specs = _build_attack_specs(occlusion_levels, gaussian_sigmas, combined_attacks)
    pck_thresholds = [float(x) for x in args.pck_thresholds.replace(",", " ").split()]
    run_dir = Path(args.runs_dir)
    # Include baseline, VAT, E1 — only those whose best.pt exists
    candidates = [("baseline", run_dir / "baseline" / "best.pt"), ("VAT", run_dir / "VAT" / "best.pt"), ("E1", run_dir / "E1" / "best.pt")]
    checkpoints = []
    names = []
    for name, ckpt in candidates:
        if ckpt.exists():
            checkpoints.append(ckpt)
            names.append(name)
    if not checkpoints:
        print("No checkpoints found (baseline/best.pt, VAT/best.pt, E1/best.pt). Run training first.")
        return {}
    results = {}
    print("\n--- Robustness comparison (COCO val, multiple attacks) ---")
    print(f"  Attack specs: {[a[0] for a in attack_specs]}")
    print(f"  PCK thresholds: {pck_thresholds}")
    print(f"  Checkpoints: {[str(c) for c in checkpoints]}\n")
    for name, ckpt in zip(names, checkpoints):
        results[name] = {}
        for attack_name, occ, gauss in attack_specs:
            try:
                pred, gt = load_model_and_eval(ckpt, loader, device, occ, gauss, use_amp, use_coco=True)
            except FileNotFoundError as e:
                print(f"  Skip {name} @ {attack_name}: {e}")
                continue
            vis = gt[:, :, 2]
            results[name][attack_name] = {"mke": mke_2d(pred, gt, vis)}
            for th in pck_thresholds:
                results[name][attack_name][f"pck@{th}"] = pck_2d(pred, gt, vis, threshold=th)
    # Table: rows = (method, attack), columns = PCK@..., MKE
    print("Results:")
    print("-" * (12 * (2 + len(pck_thresholds)) + 20))
    header = ["Method", "Attack"] + [f"PCK@{t}" for t in pck_thresholds] + ["MKE"]
    print("  ".join(f"{h:>10}" for h in header))
    print("-" * (12 * (2 + len(pck_thresholds)) + 20))
    for name in names:
        if name not in results:
            continue
        for attack_name, _, _ in attack_specs:
            if attack_name not in results[name]:
                continue
            row = [name, attack_name]
            for th in pck_thresholds:
                row.append(f"{results[name][attack_name][f'pck@{th}']*100:.2f}%")
            row.append(f"{results[name][attack_name]['mke']:.4f}")
            print("  ".join(f"{x:>10}" for x in row))
    print()
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "robustness_comparison.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {out_path}")
        occ_only = [(an, occ, g) for an, occ, g in attack_specs if g == 0]
        occ_levels_plot = sorted(set(a[1] for a in occ_only))
        if occ_levels_plot:
            results_occ = {n: {occ: results[n].get(an, {}) for an, occ, _ in occ_only} for n in names if n in results}
            _save_robustness_plots(out_dir, results_occ, occ_levels_plot, pck_thresholds, names=names)
        _save_robustness_bar_charts(out_dir, results, attack_specs, pck_thresholds, names=names)
    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate pose checkpoint(s). Use --compare for baseline vs E1 robustness.")
    p.add_argument("--ckpt", type=str, default=None, help="Single checkpoint (ignored if --compare)")
    p.add_argument("--compare", action="store_true",
                   help="Compare baseline and E1: eval both at multiple occlusion levels, print table.")
    p.add_argument("--runs_dir", type=str, default="runs",
                   help="Root dir for runs (e.g. runs/baseline/best.pt) when using --compare")
    p.add_argument("--occlusion_levels", type=str, default="0,0.1,0.2,0.3,0.4",
                   help="Comma-separated occlusion ratios for robustness (e.g. 0,0.2,0.4)")
    p.add_argument("--gaussian_sigmas", type=str, default="0",
                   help="Comma-separated Gaussian noise sigmas (e.g. 0,0.05,0.1); add gaussian-only attacks")
    p.add_argument("--combined_attacks", type=str, default="",
                   help="Space-separated 'occ,sigma' pairs for combined occlusion+Gaussian (e.g. '0.2,0.05 0.3,0.05')")
    p.add_argument("--pck_thresholds", type=str, default="0.05,0.10",
                   help="Comma-separated PCK thresholds (e.g. 0.05,0.1)")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--occlusion", type=float, default=0.0, help="Test occlusion (single-ckpt mode)")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--dataset", type=str, default=None, choices=["dummy", "coco"])
    p.add_argument("--pck_threshold", type=float, default=0.05, help="PCK threshold (single-ckpt mode)")
    p.add_argument("--image_size", type=int, default=256,
                   help="Input size (must match train; train default 256)")
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--subset_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42, help="Global RNG for reproducible eval attacks")
    p.add_argument("--imagenet_norm", action="store_true",
                   help="Use ImageNet norm (use when model was trained with --pretrained)")
    p.add_argument("--no_imagenet_norm", action="store_true",
                   help="No ImageNet norm (use when model was trained without --pretrained)")
    p.add_argument("--sweep_occlusion", action="store_true",
                   help="Single-ckpt: eval at each --occlusion_levels, print table and save plots if --out_dir")
    args = p.parse_args()
    set_global_seed(args.seed)

    use_coco = args.dataset == "coco"
    use_dummy = args.dummy or args.dataset == "dummy"

    # Match train preprocessing: train uses image_size=256, imagenet_norm=args.pretrained.
    # Default COCO eval to imagenet_norm=True so it matches typical train (--pretrained).
    if use_coco:
        imagenet_norm = args.imagenet_norm if args.imagenet_norm else (False if args.no_imagenet_norm else True)
    else:
        imagenet_norm = args.imagenet_norm

    if not args.compare and not args.ckpt:
        print("Provide --ckpt or use --compare to evaluate baseline vs E1.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    if use_coco:
        print("COCO eval:", flush=True)
        ensure_coco_pose(args.data_dir)
        nw = 0 if sys.platform == "win32" else args.num_workers
        loader = get_coco_pose_dataloader(
            args.data_dir, batch_size=args.batch_size, num_workers=nw,
            image_size=args.image_size, max_samples=args.max_val_samples, subset_seed=args.subset_seed,
            shuffle=False, imagenet_norm=imagenet_norm, split="val", loader_seed=None,
        )
        print(f"  Preprocessing: image_size={args.image_size} imagenet_norm={imagenet_norm} (match train)", flush=True)
        if args.max_val_samples is not None:
            print(f"  Subset: {args.max_val_samples} val samples (random, seed={args.subset_seed}).", flush=True)
    elif use_dummy:
        loader = torch.utils.data.DataLoader(
            DummyPoseDataset(num_samples=200, image_size=256),
            batch_size=args.batch_size, shuffle=False,
        )
    else:
        loader = get_pose_dataloader(args.data_dir, split="test", batch_size=args.batch_size)
        if loader is None:
            print("No test data. Use --dataset coco or --dummy.")
            return

    # --- Comparison mode: baseline vs E1, multiple occlusion levels ---
    if use_coco and args.compare:
        run_robustness_comparison(args, loader, device, use_amp)
        return

    # --- Single-checkpoint mode ---
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    if use_coco:
        pck_thresholds = [float(x) for x in args.pck_thresholds.replace(",", " ").split()]
        if args.sweep_occlusion:
            occlusion_levels = [float(x) for x in args.occlusion_levels.replace(",", " ").split()]
            gaussian_sigmas = [float(x) for x in (args.gaussian_sigmas or "0").replace(",", " ").split()]
            combined_attacks = []
            if (getattr(args, "combined_attacks", None) or "").strip():
                for pair in args.combined_attacks.strip().split():
                    parts = [float(x) for x in pair.split(",")]
                    if len(parts) == 2:
                        combined_attacks.append((parts[0], parts[1]))
            attack_specs = _build_attack_specs(occlusion_levels, gaussian_sigmas, combined_attacks)
            results = {"ckpt": str(ckpt_path), "dataset": "coco", "per_attack": {}}
            for attack_name, occ, gauss in attack_specs:
                pred, gt = load_model_and_eval(ckpt_path, loader, device, occ, gauss, use_amp, use_coco=True)
                vis = gt[:, :, 2]
                results["per_attack"][attack_name] = {"mke": round(mke_2d(pred, gt, vis), 4)}
                for th in pck_thresholds:
                    results["per_attack"][attack_name][f"pck@{th}"] = round(pck_2d(pred, gt, vis, threshold=th), 4)
            print("Results (sweep attacks):")
            header = ["Attack"] + [f"PCK@{t}" for t in pck_thresholds] + ["MKE"]
            print("  ".join(f"{h:>10}" for h in header))
            for attack_name, _, _ in attack_specs:
                if attack_name not in results["per_attack"]:
                    continue
                r = results["per_attack"][attack_name]
                row = [attack_name] + [f"{r[f'pck@{t}']*100:.2f}%" for t in pck_thresholds] + [f"{r['mke']:.4f}"]
                print("  ".join(f"{x:>10}" for x in row))
            if args.out_dir:
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "eval_results.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                occ_only = [a for a in attack_specs if a[2] == 0]
                if occ_only:
                    occ_levels = sorted(set(a[1] for a in occ_only))
                    per_occ = {a[1]: results["per_attack"][a[0]] for a in occ_only}
                    _save_robustness_plots(out_dir, per_occ, occ_levels, pck_thresholds, names=None)
                print(f"Wrote {out_dir / 'eval_results.json'} and plots to {out_dir}")
        else:
            pred, gt = load_model_and_eval(ckpt_path, loader, device, args.occlusion, 0.0, use_amp, use_coco=True)
            vis = gt[:, :, 2]
            pck = pck_2d(pred, gt, vis, threshold=args.pck_threshold)
            mke = mke_2d(pred, gt, vis)
            results = {
                "ckpt": str(ckpt_path),
                "dataset": "coco",
                "occlusion_ratio": args.occlusion,
                "pck": round(pck, 4),
                "pck_threshold": args.pck_threshold,
                "mke": round(mke, 4),
            }
            print(f"PCK@{args.pck_threshold}: {pck*100:.2f}%  MKE: {mke:.4f}")
    else:
        results = load_model_and_eval(ckpt_path, loader, device, args.occlusion, 0.0, use_amp, use_coco=False)
        print(f"MPJPE: {results['mpjpe_mm']} mm  PA-MPJPE: {results['pampjpe_mm']} mm  Violations: {results['geometric_violation_rate']*100:.2f}%")

    if args.out_dir and not (use_coco and args.sweep_occlusion):
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {out_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
