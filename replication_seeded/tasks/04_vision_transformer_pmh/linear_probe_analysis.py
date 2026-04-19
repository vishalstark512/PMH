"""
Linear probe analysis: per-layer probe accuracy under noise (B0, VAT, E1).
Trains a linear classifier on each layer's CLS token; evaluates on clean and noisy test.
Use case: show E1 preserves discriminative structure deeper (higher probe acc at mid layers under noise).

Usage:
  python linear_probe_analysis.py --runs_dir runs --data_dir ./data --runs B0 VAT E1
  python linear_probe_analysis.py --runs_dir runs --noise_sigmas 0 0.05 0.1 0.15 0.2 --max_train 5000

Outputs: runs/interp/linear_probe_results.json, runs/interp/linear_probe.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


def get_loaders(data_dir, batch_size=256, num_workers=4, max_train=None, max_test=None):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_ds = CIFAR10(root=data_dir, train=True, download=True, transform=t)
    test_ds = CIFAR10(root=data_dir, train=False, download=True, transform=t)
    if max_train:
        idx = np.random.RandomState(42).choice(len(train_ds), min(max_train, len(train_ds)), replace=False)
        train_ds = Subset(train_ds, idx)
    if max_test:
        test_ds = Subset(test_ds, np.arange(min(max_test, len(test_ds))))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def extract_features(model, loader, device, noise_sigma=0.0, mean=None, std=None):
    """Return list of (N, embed_dim) per layer, and labels (N,)."""
    model.eval()
    if mean is None:
        mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    if std is None:
        std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    feats_by_layer = None
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if noise_sigma > 0:
                x = images * std + mean
                x = (x + noise_sigma * torch.randn_like(x, device=device)).clamp(0, 1)
                images = (x - mean) / std
            features = model.get_features(images, return_all=True)
            if feats_by_layer is None:
                feats_by_layer = [[] for _ in range(len(features))]
            for i, f in enumerate(features):
                feats_by_layer[i].append(f.cpu().numpy())
            all_labels.append(labels.numpy())
    feats_by_layer = [np.concatenate(f, axis=0) for f in feats_by_layer]
    labels = np.concatenate(all_labels, axis=0)
    return feats_by_layer, labels


def train_probe(X_train, y_train, X_test, y_test, device, max_epochs=50):
    """Train linear classifier; return test accuracy."""
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)
    D = X_train.shape[1]
    C = int(y_train.max().item()) + 1
    probe = nn.Linear(D, C).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for _ in range(max_epochs):
        probe.train()
        opt.zero_grad(set_to_none=True)
        logits = probe(X_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(X_test).argmax(dim=1)
        acc = (pred == y_test).float().mean().item()
    return acc * 100.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--runs", type=str, nargs="+", default=["B0", "VAT", "E1"], help="Run names (must have best.pt)")
    p.add_argument("--noise_sigmas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15, 0.2])
    p.add_argument("--max_train", type=int, default=5000, help="Cap train samples for probe fitting")
    p.add_argument("--max_test", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", type=str, default=None, help="Default: <runs_dir>/interp")
    p.add_argument("--plot", action="store_true", help="Save linear_probe.png")
    args = p.parse_args()

    out_dir = Path(args.out_dir or Path(args.runs_dir) / "interp")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers, args.max_train, args.max_test)
    model_tmp = get_model(num_classes=10)
    num_layers = len(model_tmp.blocks)

    results = {"runs": args.runs, "noise_sigmas": args.noise_sigmas, "num_layers": num_layers,
               "probe_acc": {run: {str(sig): [0.0] * num_layers for sig in args.noise_sigmas} for run in args.runs}}

    for run in args.runs:
        ckpt = resolve_ckpt(args.runs_dir, run)
        if not ckpt.exists():
            print(f"  Skip {run}: no {ckpt}")
            continue
        model = get_model(num_classes=10).to(device)
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"  Run {run}: extracting features...")
        for sigma in args.noise_sigmas:
            feats_train, y_train = extract_features(model, train_loader, device, noise_sigma=sigma, mean=mean, std=std)
            feats_test, y_test = extract_features(model, test_loader, device, noise_sigma=sigma, mean=mean, std=std)
            for layer in range(num_layers):
                acc = train_probe(feats_train[layer], y_train, feats_test[layer], y_test, device)
                results["probe_acc"][run][str(sigma)][layer] = round(acc, 2)
                print(f"    sigma={sigma} layer={layer} probe_acc={acc:.2f}%")

    out_json = out_dir / "linear_probe_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_json}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            for run in args.runs:
                for sigma in args.noise_sigmas:
                    accs = results["probe_acc"][run][str(sigma)]
                    ax.plot(range(num_layers), accs, label=f"{run} σ={sigma}", marker="o", markersize=3)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Probe accuracy (%)")
            ax.legend(ncol=2, fontsize=8)
            ax.set_title("Linear probe accuracy vs layer (noise)")
            fig.tight_layout()
            fig.savefig(out_dir / "linear_probe.png", dpi=150)
            plt.close()
            print(f"Saved {out_dir / 'linear_probe.png'}")
        except Exception as e:
            print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
