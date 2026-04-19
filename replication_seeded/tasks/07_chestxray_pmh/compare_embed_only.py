"""
Mechanistic experiment 2: Compare full E1 (multi-level PMH) vs E1_embed_only (embedding-only PMH).
Computes stage-wise clean–noisy distance and clean accuracy for B0, E1, E1_embed_only.
Run after training E1_embed_only (python train.py --pmh_embed_only --run E1 ...).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from model import get_model
from data import get_eval_loaders, IMAGENET_MEAN, IMAGENET_STD

# Reuse stage distance logic from interp_resnet
def _denorm(img, device):
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=device).view(1, 3, 1, 1)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img * std + mean

def _renorm(img, device):
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=device).view(1, 3, 1, 1)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = img.clamp(0.0, 1.0)
    return (img - mean) / std

def add_noise(images, sigma, device):
    x = _denorm(images, device)
    x = torch.clamp(x + torch.randn_like(x, device=device) * sigma, 0.0, 1.0)
    return _renorm(x, device)

@torch.no_grad()
def stage_clean_noisy_distance(model, loader, device, noise_sigma=0.08, use_amp=True, max_batches=None):
    model.eval()
    n_stages = 4
    sum_dists = [0.0] * n_stages
    count = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        img, _ = batch[0], batch[1]
        img = img.to(device, non_blocking=True)
        img_noisy = add_noise(img, noise_sigma, device)
        B = img.size(0)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            fc = model.get_stage_features(img)
            fn = model.get_stage_features(img_noisy)
        for s in range(n_stages):
            d = (fc[s] - fn[s]).float().norm(dim=1).mean().item()
            sum_dists[s] += d * B
        count += B
    return [sum_dists[s] / count for s in range(n_stages)] if count else [0.0] * n_stages

@torch.no_grad()
def eval_clean_accuracy(model, loader, device, use_amp=True, max_batches=None):
    model.eval()
    correct, total = 0, 0
    for bi, (img, labels) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        img = img.to(device, non_blocking=True)
        labels = labels.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(img)
        probs = torch.sigmoid(torch.clamp(logits, -50, 50))
        pred = (probs >= 0.5).float()
        correct += (pred == labels).all(dim=1).sum().item()
        total += img.size(0)
    return 100.0 * correct / total if total else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--out_dir", type=str, default=None, help="Default: runs_dir/compare_embed_only")
    p.add_argument("--noise_sigma", type=float, default=0.08)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--no_download", action="store_true")
    p.add_argument("--dataset", type=str, default="pneumonia", choices=["pneumonia", "nih"])
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir or str(Path(args.runs_dir) / "compare_embed_only"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    _, test_loader, _, _, num_classes = get_eval_loaders(
        args.data_dir, batch_size=args.batch_size,
        auto_download=not args.no_download, dataset=args.dataset,
    )

    runs = ["B0", "E1_no_pmh", "E1", "E1_embed_only"]
    results = {}
    for run in runs:
        ckpt = Path(args.runs_dir) / run / "best.pt"
        if not ckpt.exists():
            print(f"Skip {run}: {ckpt} not found")
            continue
        model = get_model(num_classes=num_classes, pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        stage_d = stage_clean_noisy_distance(model, test_loader, device, args.noise_sigma, use_amp, args.max_batches)
        acc = eval_clean_accuracy(model, test_loader, device, use_amp, args.max_batches)
        results[run] = {"stage_dist": [round(x, 4) for x in stage_d], "clean_acc": round(acc, 4)}
        print(f"  {run}: clean_acc={acc:.2f}%  stage_dist={[f'{x:.4f}' for x in stage_d]}")

    with open(out_dir / "compare_embed_only_results.json", "w", encoding="utf-8") as f:
        json.dump({"noise_sigma": args.noise_sigma, "runs": results}, f, indent=2)
    print(f"Saved {out_dir / 'compare_embed_only_results.json'}")

    if _HAS_MPL and results:
        runs = list(results.keys())
        stage_labels = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
        x = np.arange(4)
        width = 0.8 / len(runs)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for i, run in enumerate(runs):
            dist = results[run]["stage_dist"]
            ax1.bar(x + (i - (len(runs) - 1) / 2) * width, dist, width, label=run)
        ax1.set_ylabel("Clean–noisy distance")
        ax1.set_xticks(x)
        ax1.set_xticklabels(stage_labels)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)
        ax1.set_title("Stage-wise stability")
        accs = [results[run]["clean_acc"] for run in runs]
        ax2.bar(runs, accs, color=["#1f77b4", "#888888", "#2ca02c", "#d62728"][: len(runs)])
        ax2.set_ylabel("Clean accuracy %")
        ax2.set_title("Clean accuracy")
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "compare_embed_only.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / 'compare_embed_only.png'}")


if __name__ == "__main__":
    main()
