"""
Embedding stability for Pose: mean ||φ_clean − φ_pert|| under occlusion and Gaussian noise.
Internal-state metric for PMH: lower drift = more consistent internal state.

Usage:
  python embedding_stability.py --data_dir ./data --out_dir runs/eval_out
  python embedding_stability.py --runs_dir runs --data_dir ./data

Output: embedding_stability.json with per-run, per-perturbation embedding_mean (drift).

Replication: --seed (global + eval), --subset_seed (COCO val subset).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add tasks dir for tdi_utils (tasks/tdi_utils.py)
_tasks_dir = Path(__file__).resolve().parents[1]
if str(_tasks_dir) not in sys.path:
    sys.path.insert(0, str(_tasks_dir))
from tdi_utils import embedding_drift

from model import get_model
from data import get_coco_pose_dataloader, ensure_coco_pose, apply_eval_attack, DummyPoseDataset

SEED = 42  # default; overridden by --seed in main()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
OCCLUSION_LEVELS = [0.1, 0.2, 0.3]
GAUSSIAN_SIGMAS = [0.05, 0.1]


@torch.no_grad()
def extract_embeddings(model, loader, device, occlusion_ratio=0.0, gaussian_sigma=0.0, seed=42, use_amp=False):
    """Extract pooled backbone features. Optionally apply eval attack with fixed seed."""
    model.eval()
    embs = []
    if seed is not None:
        torch.manual_seed(seed)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        if occlusion_ratio > 0 or gaussian_sigma > 0:
            images = apply_eval_attack(images, occlusion_ratio=occlusion_ratio, gaussian_sigma=gaussian_sigma, device=device)
        feats = model.get_features(images, return_all=False)
        # Pool: (B, C, h, w) -> (B, C)
        feats = feats.mean(dim=(2, 3)).cpu().numpy()
        embs.append(feats)
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 512))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="runs/eval_out")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42, help="Global RNG and extract_embeddings attack seed")
    p.add_argument("--subset_seed", type=int, default=42, help="COCO val subset shuffle (max_samples)")
    args = p.parse_args()
    set_global_seed(args.seed)

    ensure_coco_pose(args.data_dir)
    loader = get_coco_pose_dataloader(
            args.data_dir, batch_size=args.batch_size, num_workers=0,
            image_size=256, max_samples=args.max_samples, subset_seed=args.subset_seed,
            shuffle=False, imagenet_norm=True, split="val", loader_seed=None,
        )
    if loader is None:
        print("No COCO pose data; skipping embedding stability.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    runs_dir = Path(args.runs_dir)
    run_names = ["baseline", "VAT", "E1"]
    ckpt_paths = {r: runs_dir / r / "best.pt" for r in run_names}

    results = {"runs": {}, "perturbations": ["clean"] + [f"occ_{o}" for o in OCCLUSION_LEVELS] + [f"gauss_{s}" for s in GAUSSIAN_SIGMAS]}

    for run_name in run_names:
        ckpt = ckpt_paths.get(run_name)
        if ckpt is None or not ckpt.exists():
            print(f"Skip {run_name}: no checkpoint at {ckpt}")
            continue

        model = get_model(backbone="resnet18", pretrained=False).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

        embs_clean = extract_embeddings(model, loader, device, 0.0, 0.0, seed=args.seed, use_amp=use_amp)
        results["runs"][run_name] = {"embedding_drift": {"clean": 0.0}, "embedding_mean": {"clean": 0.0}}

        for occ in OCCLUSION_LEVELS:
            embs_pert = extract_embeddings(model, loader, device, occlusion_ratio=occ, gaussian_sigma=0.0, seed=args.seed, use_amp=use_amp)
            drift = embedding_drift(embs_clean, embs_pert)
            key = f"occ_{occ}"
            results["runs"][run_name]["embedding_drift"][key] = round(drift, 6)
            results["runs"][run_name]["embedding_mean"][key] = round(drift, 6)

        for sig in GAUSSIAN_SIGMAS:
            embs_pert = extract_embeddings(model, loader, device, occlusion_ratio=0.0, gaussian_sigma=sig, seed=args.seed, use_amp=use_amp)
            drift = embedding_drift(embs_clean, embs_pert)
            key = f"gauss_{sig}"
            results["runs"][run_name]["embedding_drift"][key] = round(drift, 6)
            results["runs"][run_name]["embedding_mean"][key] = round(drift, 6)

        drift_occ02 = results["runs"][run_name]["embedding_drift"].get("occ_0.2", 0)
        print(f"{run_name}: drift @ occ_0.2 = {drift_occ02}")

    out_path = out_dir / "embedding_stability.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
