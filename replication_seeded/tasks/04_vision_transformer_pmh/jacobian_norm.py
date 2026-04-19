"""
Estimate clean-input Jacobian Frobenius norm for ViT on CIFAR-10.

We estimate ||J||_F where J = d logits(x) / d x (logits dim = 10, x dim = 3*32*32),
using Hutchinson:

  E_v || J^T v ||^2 = ||J||_F^2,  v ~ N(0, I_C)

So we sample v and compute gradients of <logits, v> w.r.t. inputs.

Usage:
  python jacobian_norm.py --runs_dir runs --data_dir ./data --runs B0 VAT E1 --n_batches 50 --n_probes 2

Outputs:
  runs/interp/jacobian_norm_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import get_model

CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def resolve_ckpt(runs_dir, run_name):
    """Support both runs/<name>/best.pt and runs/<name>/<algo>/best.pt layouts."""
    base = Path(runs_dir) / run_name
    candidates = [
        base / "best.pt",
        base / "PGD" / "best.pt",
        base / "VAT" / "best.pt",
        base / "E1" / "best.pt",
        base / "B0" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def get_loader(data_dir, batch_size=128, num_workers=4, max_samples=None):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = CIFAR10(root=data_dir, train=False, download=True, transform=t)
    if max_samples:
        ds = Subset(ds, np.arange(min(max_samples, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def _accuracy(model, loader, device, use_amp=True):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total if total else 0.0


def estimate_jacobian_fro(model, loader, device, n_batches=50, n_probes=2, use_amp=True):
    """
    Returns dict with mean Fro norm estimate over batches and some diagnostics.
    """
    model.eval()
    use_amp = use_amp and device.type == "cuda"
    vals = []
    for bi, (x, _y) in enumerate(loader):
        if bi >= n_batches:
            break
        x = x.to(device, non_blocking=True)
        x = x.detach().requires_grad_(True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)  # (B, C)
        B, C = logits.shape
        # Hutchinson probes over class dimension
        batch_est = 0.0
        for _ in range(n_probes):
            v = torch.randn((B, C), device=device, dtype=logits.dtype)
            s = (logits * v).sum()
            grad = torch.autograd.grad(s, x, retain_graph=True, create_graph=False)[0]
            # ||J^T v||^2 for each sample, then mean
            jt_v_sq = grad.reshape(B, -1).pow(2).sum(dim=1).mean()
            batch_est += jt_v_sq
        batch_est = batch_est / float(n_probes)
        # batch_est approximates ||J||_F^2
        vals.append(float(torch.sqrt(batch_est).detach().cpu()))
        # free graph
        del logits
    out = {
        "n_batches": int(min(n_batches, len(vals))),
        "n_probes": int(n_probes),
        "jacobian_fro_mean": float(np.mean(vals)) if vals else None,
        "jacobian_fro_std": float(np.std(vals)) if vals else None,
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default=None, help="Default: <runs_dir>/interp")
    p.add_argument("--runs", type=str, nargs="+", default=["B0", "VAT", "E1_no_pmh", "E1"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--n_batches", type=int, default=50)
    p.add_argument("--n_probes", type=int, default=2)
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir or (Path(args.runs_dir) / "interp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    loader = get_loader(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, max_samples=args.max_samples)

    results = {"runs": {}, "settings": {"max_samples": args.max_samples, "n_batches": args.n_batches, "n_probes": args.n_probes}}
    for run in args.runs:
        ckpt = resolve_ckpt(args.runs_dir, run)
        if not ckpt.exists():
            print(f"Skip {run}: {ckpt} not found")
            continue
        model = get_model(num_classes=10).to(device)
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        acc = _accuracy(model, loader, device, use_amp=use_amp)
        est = estimate_jacobian_fro(model, loader, device, n_batches=args.n_batches, n_probes=args.n_probes, use_amp=use_amp)
        results["runs"][run] = {"clean_acc_on_subset": acc, **est}
        print(f"  {run}: J_fro={est['jacobian_fro_mean']:.4g}±{est['jacobian_fro_std']:.4g}  acc={acc:.2f}%")

    out_path = out_dir / "jacobian_norm_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

