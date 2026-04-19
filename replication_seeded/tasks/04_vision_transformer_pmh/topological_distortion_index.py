"""
Topological Distortion Index (TDI) for ViT: intra-class / inter-class latent distance under noise.
Supports Theorem 1 (topological blindness): B0/VAT TDI explodes with σ; E1 stays bounded.

Usage:
  python topological_distortion_index.py --runs_dir runs --data_dir ./data --runs B0 VAT E1
  python topological_distortion_index.py --runs_dir runs --noise_sigmas 0 0.05 0.1 0.15 0.2 --max_samples 2000

Outputs: runs/interp/tdi_results.json, optional runs/interp/tdi.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
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


def get_loader(data_dir, batch_size=128, max_samples=None):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = CIFAR10(root=data_dir, train=False, download=True, transform=t)
    if max_samples:
        ds = Subset(ds, np.arange(min(max_samples, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


def extract_embeddings(model, loader, device, noise_sigma=0.0):
    """Return (N, D) embeddings and (N,) labels."""
    model.eval()
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    embs, labels = [], []
    with torch.no_grad():
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


def evaluate_accuracy(model, loader, device, noise_sigma=0.0):
    """Return accuracy (%) for model on loader with optional input noise."""
    model.eval()
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    correct, total = 0, 0
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            if noise_sigma > 0:
                x = images * std + mean
                x = (x + noise_sigma * torch.randn_like(x, device=device)).clamp(0, 1)
                images = (x - mean) / std
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred.cpu() == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total else 0.0


def compute_tdi(embeddings, labels, num_classes=10):
    """
    TDI = intra_class_mean / inter_class_mean (in latent space).
    Higher TDI = more distortion (same-class points scattered).
    """
    embs = np.asarray(embeddings, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.int64)
    intra_dists = []
    for c in range(num_classes):
        mask = labs == c
        if mask.sum() < 2:
            continue
        X = embs[mask]
        # mean pairwise distance within class (upper triangle)
        n = X.shape[0]
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        triu = np.triu_indices(n, k=1)
        intra_dists.extend(d[triu].tolist())
    intra_mean = np.mean(intra_dists) if intra_dists else 0.0

    # inter-class: mean distance between class centroids (or sample one pair per class pair)
    centroids = []
    for c in range(num_classes):
        mask = labs == c
        if mask.sum() == 0:
            continue
        centroids.append(embs[mask].mean(axis=0))
    centroids = np.stack(centroids)
    n_c = centroids.shape[0]
    inter_dists = []
    for i in range(n_c):
        for j in range(i + 1, n_c):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    inter_mean = np.mean(inter_dists) if inter_dists else 1.0

    if inter_mean <= 0:
        return float(intra_mean), 0.0
    tdi = intra_mean / inter_mean
    return float(intra_mean), float(tdi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--runs", type=str, nargs="+", default=["B0", "VAT", "E1"])
    p.add_argument("--noise_sigmas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15, 0.2])
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir or Path(args.runs_dir) / "interp")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_loader(args.data_dir, args.batch_size, args.max_samples)

    results = {"runs": args.runs, "noise_sigmas": args.noise_sigmas, "tdi": {}, "intra_mean": {}, "accuracy": {}}

    for run in args.runs:
        ckpt = resolve_ckpt(args.runs_dir, run)
        if not ckpt.exists():
            print(f"  Skip {run}: no {ckpt}")
            results["tdi"][run] = {str(s): None for s in args.noise_sigmas}
            results["intra_mean"][run] = {str(s): None for s in args.noise_sigmas}
            results["accuracy"][run] = {str(s): None for s in args.noise_sigmas}
            continue
        model = get_model(num_classes=10).to(device)
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        results["tdi"][run] = {}
        results["intra_mean"][run] = {}
        results["accuracy"][run] = {}
        for sigma in args.noise_sigmas:
            emb, lab = extract_embeddings(model, loader, device, noise_sigma=sigma)
            intra_mean, tdi = compute_tdi(emb, lab)
            acc = evaluate_accuracy(model, loader, device, noise_sigma=sigma)
            results["intra_mean"][run][str(sigma)] = round(intra_mean, 6)
            results["tdi"][run][str(sigma)] = round(tdi, 6)
            results["accuracy"][run][str(sigma)] = round(acc, 2)
            print(f"  {run}  sigma={sigma}  acc={acc:.2f}%  intra_mean={intra_mean:.4f}  TDI={tdi:.4f}")

    out_json = out_dir / "tdi_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_json}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # Single panel: TDI vs sigma
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            for run in args.runs:
                if run not in results["tdi"] or all(v is None for v in results["tdi"][run].values()):
                    continue
                sigmas = args.noise_sigmas
                tdis = [results["tdi"][run].get(str(s)) for s in sigmas]
                tdis = [t for t in tdis if t is not None]
                sigmas_plot = sigmas[:len(tdis)]
                ax.plot(sigmas_plot, tdis, label=run, marker="o", markersize=5)
            ax.set_xlabel("Noise σ")
            ax.set_ylabel("TDI (intra / inter)")
            ax.legend()
            ax.set_title("Topological Distortion Index vs noise")
            fig.tight_layout()
            fig.savefig(out_dir / "tdi.png", dpi=150)
            plt.close()
            print(f"Saved {out_dir / 'tdi.png'}")

            # Dissociation figure (Gap 1): accuracy vs σ and TDI vs σ — B0/VAT can have good acc but exploding TDI
            has_acc = "accuracy" in results and any(
                results["accuracy"].get(r) and any(v is not None for v in results["accuracy"][r].values())
                for r in args.runs
            )
            if has_acc:
                fig2, (ax_acc, ax_tdi) = plt.subplots(1, 2, figsize=(10, 4))
                for run in args.runs:
                    if run not in results["accuracy"] or all(v is None for v in results["accuracy"][run].values()):
                        continue
                    sigmas = args.noise_sigmas
                    accs = [results["accuracy"][run].get(str(s)) for s in sigmas]
                    accs = [a for a in accs if a is not None]
                    sigmas_a = sigmas[:len(accs)]
                    ax_acc.plot(sigmas_a, accs, label=run, marker="o", markersize=5)
                    tdis = [results["tdi"][run].get(str(s)) for s in sigmas]
                    tdis = [t for t in tdis if t is not None]
                    sigmas_t = sigmas[:len(tdis)]
                    ax_tdi.plot(sigmas_t, tdis, label=run, marker="s", markersize=5)
                ax_acc.set_xlabel("Noise σ")
                ax_acc.set_ylabel("Accuracy (%)")
                ax_acc.set_title("Output accuracy vs noise")
                ax_acc.legend()
                ax_acc.grid(alpha=0.3)
                ax_tdi.set_xlabel("Noise σ")
                ax_tdi.set_ylabel("TDI (intra / inter)")
                ax_tdi.set_title("Topological distortion vs noise")
                ax_tdi.legend()
                ax_tdi.grid(alpha=0.3)
                fig2.suptitle("Dissociation: accuracy ≠ topology (B0/VAT: good acc, exploding TDI; E1: bounded TDI)", fontsize=11)
                fig2.tight_layout()
                fig2.savefig(out_dir / "tdi_dissociation.png", dpi=150)
                plt.close(fig2)
                print(f"Saved {out_dir / 'tdi_dissociation.png'}")
        except Exception as e:
            print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
