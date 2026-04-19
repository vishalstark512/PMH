"""
ResNet mechanistic interp: stage-wise clean–noisy distance.
Shows that E1 (PMH) has smaller representation drift in later stages, matching ViT findings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import get_model
from data import get_eval_loaders, IMAGENET_MEAN, IMAGENET_STD

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


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
    """Compute mean L2 distance (clean vs noisy) per ResNet stage. Returns list of 4 floats."""
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


def main():
    p = argparse.ArgumentParser(description="ResNet stage-wise clean–noisy distance (mech interp)")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--out_dir", type=str, default=None, help="Default: runs_dir/interp_resnet")
    p.add_argument("--noise_sigma", type=float, default=0.08)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_batches", type=int, default=None, help="Cap for quick run")
    p.add_argument("--no_download", action="store_true")
    p.add_argument("--dataset", type=str, default="pneumonia", choices=["pneumonia", "nih"])
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--runs", type=str, nargs="+", default=["B0", "VAT", "E1"], help="Run names (VAT baseline for PMH comparison)")
    args = p.parse_args()

    out_dir = Path(args.out_dir or str(Path(args.runs_dir) / "interp_resnet"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    _, test_loader, _, _, num_classes = get_eval_loaders(
        args.data_dir, batch_size=args.batch_size,
        auto_download=not args.no_download, dataset=args.dataset,
    )

    def load_run(run):
        ckpt = Path(args.runs_dir) / run / "best.pt"
        if not ckpt.exists():
            return None, None
        model = get_model(num_classes=num_classes, pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        return run, model

    results = {}
    for run in args.runs:
        name, model = load_run(run)
        if model is None:
            print(f"Skip {run}: checkpoint not found")
            continue
        dists = stage_clean_noisy_distance(
            model, test_loader, device,
            noise_sigma=args.noise_sigma, use_amp=use_amp, max_batches=args.max_batches,
        )
        results[run] = dists
        print(f"  {run}: stage distances = {[f'{d:.4f}' for d in dists]}")

    with open(out_dir / "stage_distances.json", "w") as f:
        json.dump({"noise_sigma": args.noise_sigma, "runs": results}, f, indent=2)

    if _HAS_MPL and results:
        fig, ax = plt.subplots(figsize=(6, 4))
        stages = np.arange(1, 5)
        colors = {"B0": "steelblue", "VAT": "orange", "E1_no_pmh": "gray", "E1": "green"}
        for run, dists in results.items():
            ax.plot(stages, dists, "o-", label=run, color=colors.get(run, "gray"), lw=2)
        ax.set_xlabel("ResNet stage")
        ax.set_ylabel("Mean L2 dist(clean, noisy)")
        ax.set_title("Stage-wise representation stability\n(lower = more stable; E1 expected lower in later stages)")
        ax.set_xticks(stages)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "resnet_stage_clean_noisy_distance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / 'resnet_stage_clean_noisy_distance.png'}")

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
