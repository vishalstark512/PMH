"""
Adversarial robustness: FGSM evaluation for ViT (B0 vs E1).
Produces a table of clean and robust accuracy for paper.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import get_model

CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_loader(data_dir, batch_size=256, num_workers=4):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = CIFAR10(root=data_dir, train=False, download=True, transform=t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def fgsm_step(model, x_norm, labels, eps_pixel, mean, std, device, use_amp=True):
    """FGSM in pixel space [0,1]: x_01_adv = x_01 + eps_pixel * sign(grad_x01 L), then clamp and re-normalize."""
    x_norm = x_norm.to(device).detach()
    x_01 = (x_norm * std + mean).detach().requires_grad_(True)
    labels = labels.to(device)
    x_norm_in = (x_01 - mean) / std
    with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
        logits = model(x_norm_in)
        loss = F.cross_entropy(logits, labels)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        x_01_adv = (x_01 + eps_pixel * x_01.grad.sign()).clamp(0.0, 1.0)
    x_adv_norm = (x_01_adv - mean) / std
    return x_adv_norm.detach()


def evaluate_fgsm(model, loader, device, eps_list, use_amp=True, max_batches=None):
    """eps_list: list of eps in pixel space [0,1] (e.g. 2/255)."""
    model.eval()
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    results = {"clean": 0.0}
    for eps_pixel in eps_list:
        results[f"fgsm_{eps_pixel:.4f}"] = 0.0
    total = 0
    correct_clean = 0
    correct_adv = {k: 0 for k in results if k != "clean"}

    for bi, (images, labels) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(images)
            pred = logits.argmax(dim=1)
            correct_clean += (pred == labels).sum().item()
        total += B

        for eps_pixel in eps_list:
            x_adv = fgsm_step(model, images, labels, eps_pixel, mean, std, device, use_amp)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    logits_adv = model(x_adv)
                pred_adv = logits_adv.argmax(dim=1)
                correct_adv[f"fgsm_{eps_pixel:.4f}"] += (pred_adv == labels).sum().item()

    results["clean"] = 100.0 * correct_clean / total
    for k in correct_adv:
        results[k] = 100.0 * correct_adv[k] / total
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--out_dir", type=str, default=None, help="Default: runs_dir/adversarial")
    p.add_argument("--runs", type=str, nargs="+", default=["B0", "VAT", "E1_no_pmh", "E1"], help="Run names to compare (VAT baseline for PMH comparison)")
    p.add_argument("--eps", type=float, nargs="+", default=[1/255, 2/255, 4/255], help="FGSM eps in pixel [0,1]")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir or str(Path(args.runs_dir) / "adversarial"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    loader = get_loader(args.data_dir, args.batch_size)

    all_results = {}
    for run in args.runs:
        ckpt = Path(args.runs_dir) / run / "best.pt"
        if not ckpt.exists():
            print(f"Skip {run}: {ckpt} not found")
            continue
        model = get_model(num_classes=10).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        res = evaluate_fgsm(model, loader, device, args.eps, use_amp=use_amp, max_batches=args.max_batches)
        all_results[run] = res
        print(f"  {run}: clean={res['clean']:.2f}%  " + "  ".join(f"{k}={res[k]:.2f}%" for k in res if k != "clean"))

    with open(out_dir / "fgsm_results.json", "w") as f:
        json.dump({"eps_pixel": args.eps, "runs": all_results}, f, indent=2)
    print(f"Saved {out_dir / 'fgsm_results.json'}")

    if _HAS_MPL and all_results:
        first_keys = list(list(all_results.values())[0].keys())
        keys = ["clean"] + sorted([k for k in first_keys if k != "clean"])
        runs = list(all_results.keys())
        x = np.arange(len(keys))
        width = 0.8 / len(runs)
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, run in enumerate(runs):
            vals = [all_results[run].get(k, 0) for k in keys]
            offset = (i - (len(runs) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=run)
        ax.set_ylabel("Accuracy %")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "fgsm_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / 'fgsm_accuracy.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
