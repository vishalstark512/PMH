"""
Mechanistic experiment 6: Saliency stability (B0 vs E1).
Computes input-gradient saliency for clean vs perturbed images; compares stability
(how similar saliency maps are under perturbation) between B0 and E1.
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

from model import get_model
from data import get_eval_loaders, IMAGENET_MEAN, IMAGENET_STD


def _denorm(img, device):
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=device).view(1, 3, 1, 1)
    return img * std + mean


def _renorm(img, device):
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=device).view(1, 3, 1, 1)
    img = img.clamp(0.0, 1.0)
    return (img - mean) / std


def perturb_gaussian(img, sigma, device):
    x = _denorm(img, device)
    x = x + sigma * torch.randn_like(x, device=device)
    x = x.clamp(0.0, 1.0)
    return _renorm(x, device)


def saliency_map(model, x, target_logits, device):
    """
    Gradient of (sum of target logits) w.r.t. input x.
    x: (B, C, H, W). Returns (B, 1, H, W) saliency magnitude per pixel (L2 over channels).
    Must run with grad enabled (no @torch.no_grad()).
    """
    x_in = x.detach().requires_grad_(True)
    logits = model(x_in)
    if target_logits is None:
        target = logits.max(dim=1, keepdim=True)[0].sum()
    else:
        target = (logits * target_logits).sum()
    grad_out = torch.autograd.grad(outputs=target, inputs=x_in, retain_graph=False, create_graph=False)[0]
    sal = (grad_out ** 2).sum(dim=1, keepdim=True).sqrt().detach()
    return sal


def flatten_normalize(sal):
    """Flatten (B,H,W) to (B, -1), L2 normalize per sample."""
    b = sal.size(0)
    s = sal.flatten(1)
    s = F.normalize(s, p=2, dim=1)
    return s


def stability_score(sal_clean, sal_pert):
    """Mean cosine similarity between clean and perturbed saliency (higher = more stable)."""
    a = flatten_normalize(sal_clean.squeeze(1))
    b = flatten_normalize(sal_pert.squeeze(1))
    return (a * b).sum(dim=1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--noise_sigma", type=float, default=0.08)
    ap.add_argument("--max_batches", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--dataset", type=str, default="pneumonia", choices=["pneumonia", "nih"])
    ap.add_argument("--runs", type=str, nargs="+", default=["B0", "VAT", "E1"], help="Run names to compare")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or str(Path(args.runs_dir) / "saliency_stability"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _, _, _ = get_eval_loaders(
        args.data_dir, batch_size=args.batch_size, max_test_samples=args.max_batches * args.batch_size,
        auto_download=True, dataset=args.dataset,
    )

    results = {}
    for run in args.runs:
        ckpt = Path(args.runs_dir) / run / "best.pt"
        if not ckpt.exists():
            print(f"Skip {run}: {ckpt} not found")
            continue
        num_classes = 2 if args.dataset == "pneumonia" else 14
        model = get_model(num_classes=num_classes, embed_dim=args.embed_dim, pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        scores = []
        n_b = 0
        for images, _ in test_loader:
            if n_b >= args.max_batches:
                break
            images = images.to(device)
            perturbed = perturb_gaussian(images, args.noise_sigma, device)

            with torch.no_grad():
                logits_clean = model(images)
                pred = logits_clean.argmax(dim=1)

            sal_clean = saliency_map(model, images, None, device)
            sal_pert = saliency_map(model, perturbed, None, device)
            sc = stability_score(sal_clean, sal_pert)
            scores.append(sc)
            n_b += 1

        mean_stability = float(np.mean(scores))
        results[run] = {"saliency_stability_cos_sim": round(mean_stability, 4), "n_batches": n_b}
        print(f"  {run}: saliency stability (cos sim clean vs perturbed) = {mean_stability:.4f}")

    with open(out_dir / "saliency_stability_results.json", "w", encoding="utf-8") as f:
        json.dump({"noise_sigma": args.noise_sigma, "runs": results}, f, indent=2)
    print(f"Saved {out_dir / 'saliency_stability_results.json'}")

    if _HAS_MPL and results:
        runs = list(results.keys())
        vals = [results[r]["saliency_stability_cos_sim"] for r in runs]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(runs, vals, color=["#1f77b4", "#ff7f0e"][: len(runs)])
        ax.set_ylabel("Saliency stability (cos sim clean vs perturbed)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Mechanistic 6: Saliency stability")
        plt.tight_layout()
        fig.savefig(out_dir / "saliency_stability.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / 'saliency_stability.png'}")


if __name__ == "__main__":
    main()
