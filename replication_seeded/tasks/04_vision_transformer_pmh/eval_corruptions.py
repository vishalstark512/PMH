"""
Corruption/perturbation robustness for ViT on CIFAR-10.

Evaluates B0, E1_no_pmh, and E1 on a small suite of distortions:
- Gaussian noise (σ in {0.05, 0.10})
- Gaussian blur
- Brightness and contrast shifts

Outputs a JSON summary and prints a compact table.
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
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import get_model


CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_base_loader(data_dir, batch_size=256, num_workers=4):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = CIFAR10(root=data_dir, train=False, download=True, transform=t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _denorm(x, device):
    mean = torch.tensor(CIFAR10_MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    return x * std + mean


def _renorm(x, device):
    mean = torch.tensor(CIFAR10_MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    x = x.clamp(0.0, 1.0)
    return (x - mean) / std


def perturb_gaussian(x, sigma, device):
    if sigma <= 0:
        return x
    x01 = _denorm(x, device)
    x01 = x01 + sigma * torch.randn_like(x01, device=device)
    return _renorm(x01, device)


def perturb_blur(x, kernel_size, device):
    # x in normalized space; blur in [0,1] then renorm
    x01 = _denorm(x, device)
    x_blur = torch.nn.functional.avg_pool2d(x01, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return _renorm(x_blur, device)


def perturb_brightness(x, factor, device):
    x01 = _denorm(x, device)
    x01 = x01 * factor
    return _renorm(x01, device)


def perturb_contrast(x, factor, device):
    x01 = _denorm(x, device)
    mean = x01.mean(dim=(2, 3), keepdim=True)
    x01 = (x01 - mean) * factor + mean
    return _renorm(x01, device)


@torch.no_grad()
def eval_corruptions(model, loader, device, perturb_fns):
    model.eval()
    results = {name: 0.0 for name in perturb_fns}
    counts = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)
        counts += B

        for name, fn in perturb_fns.items():
            x_pert = fn(images)
            logits = model(x_pert)
            pred = logits.argmax(dim=1)
            results[name] += (pred == labels).sum().item()

    if counts == 0:
        return {name: 0.0 for name in results}
    return {name: 100.0 * correct / counts for name, correct in results.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--out_dir", type=str, default=None, help="Default: runs_dir/corruptions")
    p.add_argument("--runs", type=str, nargs="+", default=["B0", "VAT", "E1_no_pmh", "E1"], help="Run names to compare (include VAT for PMH comparison)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_batches", type=int, default=None, help="(Unused for now; full test set)")
    args = p.parse_args()

    out_dir = Path(args.out_dir or str(Path(args.runs_dir) / "corruptions"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_base_loader(args.data_dir, args.batch_size)

    perturb_fns = {
        "clean": lambda x: x,
        "gaussian_0.05": lambda x: perturb_gaussian(x, 0.05, device),
        "gaussian_0.10": lambda x: perturb_gaussian(x, 0.10, device),
        "blur_3": lambda x: perturb_blur(x, 3, device),
        "brightness_0.7": lambda x: perturb_brightness(x, 0.7, device),
        "brightness_1.3": lambda x: perturb_brightness(x, 1.3, device),
        "contrast_0.7": lambda x: perturb_contrast(x, 0.7, device),
        "contrast_1.3": lambda x: perturb_contrast(x, 1.3, device),
    }

    all_results = {}
    for run in args.runs:
        ckpt = Path(args.runs_dir) / run / "best.pt"
        if not ckpt.exists():
            print(f"Skip {run}: {ckpt} not found")
            continue
        model = get_model(num_classes=10).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        print(f"Evaluating corruptions for {run} …")
        res = eval_corruptions(model, loader, device, perturb_fns)
        all_results[run] = res
        print("  " + ", ".join(f"{k}={v:.2f}%" for k, v in res.items()))

    with open(out_dir / "corruptions_results.json", "w", encoding="utf-8") as f:
        json.dump({"runs": all_results, "perturbations": list(perturb_fns.keys())}, f, indent=2)
    print(f"Saved {out_dir / 'corruptions_results.json'}")

    if _HAS_MPL and all_results:
        pert_names = list(perturb_fns.keys())
        runs = list(all_results.keys())
        x = np.arange(len(pert_names))
        width = 0.8 / len(runs)
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, run in enumerate(runs):
            vals = [all_results[run].get(p, 0) for p in pert_names]
            offset = (i - (len(runs) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=run)
        ax.set_ylabel("Accuracy %")
        ax.set_xticks(x)
        ax.set_xticklabels(pert_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "corruptions_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / 'corruptions_accuracy.png'}")


if __name__ == "__main__":
    main()

