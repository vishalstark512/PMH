"""
Embedding stability for CIFAR10: mean ||φ_clean − φ_noisy|| under Gaussian noise.
Internal-state metric for PMH: lower drift = more consistent internal state.

Usage:
  python embedding_stability.py runs/cifar10/B0/best.pt runs/cifar10/VAT/best.pt runs/cifar10/E1/best.pt
  python embedding_stability.py --out_dir runs/evals --runs_dir runs/cifar10

Output: embedding_stability.json with per-run, per-sigma embedding_mean (drift).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Add tasks dir for tdi_utils (tasks/tdi_utils.py)
_tasks_dir = Path(__file__).resolve().parents[2]
if str(_tasks_dir) not in sys.path:
    sys.path.insert(0, str(_tasks_dir))
from tdi_utils import embedding_drift

from model import get_model

CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DEFAULT_SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2]


def get_loader(data_dir, batch_size=256, max_samples=None):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = CIFAR10(root=data_dir, train=False, download=True, transform=t)
    if max_samples:
        from torch.utils.data import Subset
        ds = Subset(ds, np.arange(min(max_samples, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def extract_embeddings(model, loader, device, noise_sigma=0.0, seed=42):
    """Extract pre-fc features (embeddings) for test set. Optionally add Gaussian noise to input."""
    model.eval()
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    embs, labels = [], []
    if seed is not None:
        torch.manual_seed(seed)
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        if noise_sigma > 0:
            x = images * std + mean
            x = (x + noise_sigma * torch.randn_like(x, device=device)).clamp(0, 1)
            images = (x - mean) / std
        feats = model.get_features(images, return_all=False)
        feats = feats.cpu().numpy()
        embs.append(feats)
        labels.append(y.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="*", help="Checkpoint paths (e.g. runs/cifar10/B0/best.pt)")
    p.add_argument("--runs_dir", type=str, default="runs/cifar10",
                   help="If no checkpoints, use runs_dir/B0/best.pt etc.")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="runs/evals")
    p.add_argument("--noise_sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    if not args.checkpoint:
        runs_dir = Path(args.runs_dir)
        for r in ["B0", "VAT", "E1"]:
            ckpt = runs_dir / r / "best.pt"
            if ckpt.exists():
                args.checkpoint.append(str(ckpt))
        if not args.checkpoint:
            print("No checkpoints found. Specify paths or --runs_dir.")
            return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_loader(args.data_dir, batch_size=args.batch_size, max_samples=args.max_samples)

    results = {"runs": {}, "noise_sigmas": args.noise_sigmas}

    for ckpt_path in args.checkpoint:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"Skip (not found): {ckpt_path}")
            continue
        run_name = ckpt_path.parent.name if ckpt_path.parent.name in ("B0", "VAT", "E1") else ckpt_path.stem

        model = get_model("resnet18", num_classes=10).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

        embs_clean, _ = extract_embeddings(model, loader, device, noise_sigma=0.0)
        results["runs"][run_name] = {"embedding_drift": {}, "embedding_mean": {}}
        for sigma in args.noise_sigmas:
            if sigma == 0:
                results["runs"][run_name]["embedding_drift"]["0.0"] = 0.0
                results["runs"][run_name]["embedding_mean"]["0.0"] = 0.0
                continue
            embs_noisy, _ = extract_embeddings(model, loader, device, noise_sigma=sigma)
            drift = embedding_drift(embs_clean, embs_noisy)
            results["runs"][run_name]["embedding_drift"][str(sigma)] = round(drift, 6)
            results["runs"][run_name]["embedding_mean"][str(sigma)] = round(drift, 6)
        print(f"{run_name}: drift @ σ=0.1 = {results['runs'][run_name]['embedding_drift'].get('0.1', 'N/A')}")

    out_path = out_dir / "embedding_stability.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
