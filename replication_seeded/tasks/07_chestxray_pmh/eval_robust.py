"""
Robust chest X-ray evaluation: accuracy and AUC under perturbations, plus embedding stability.
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch

from model import get_model
from data import get_eval_loaders, IMAGENET_MEAN, IMAGENET_STD
from eval import evaluate

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def set_global_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False


def _denorm(img):
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img * std + mean


def _renorm(img):
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    img = img.clamp(0.0, 1.0)
    return (img - mean) / std


def perturb_gaussian(img, sigma):
    if sigma <= 0:
        return img
    x = _denorm(img)
    x = torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)
    x = _renorm(x)
    return x.squeeze(0) if img.dim() == 3 else x


def perturb_intensity(img, factor):
    x = _denorm(img)
    x = torch.clamp(x * factor, 0.0, 1.0)
    x = _renorm(x)
    return x.squeeze(0) if img.dim() == 3 else x


def perturb_identity(img):
    return img


def perturb_rotate(img, degrees):
    """Small rotation around center, in pixel space."""
    x = _denorm(img)
    # affine_grid expects angle in radians via cos/sin; use torchvision-like small rotation via grid_sample
    theta = torch.zeros(img.size(0), 2, 3, device=img.device, dtype=img.dtype)
    rad = degrees * 3.1415926535 / 180.0
    c = torch.cos(torch.tensor(rad, device=img.device, dtype=img.dtype))
    s = torch.sin(torch.tensor(rad, device=img.device, dtype=img.dtype))
    theta[:, 0, 0] = c
    theta[:, 0, 1] = -s
    theta[:, 1, 0] = s
    theta[:, 1, 1] = c
    grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
    x_rot = torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode="border")
    return _renorm(x_rot)


def perturb_zoom(img, scale):
    """Simple center zoom in/out using affine grid."""
    x = _denorm(img)
    theta = torch.zeros(img.size(0), 2, 3, device=img.device, dtype=img.dtype)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
    x_zoom = torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode="border")
    return _renorm(x_zoom)


def perturb_contrast_jitter(img, factor):
    x = _denorm(img)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * factor + mean
    return _renorm(x)


def perturb_gamma(img, gamma):
    x = _denorm(img).clamp(0.0, 1.0)
    x = x.pow(gamma)
    return _renorm(x)


def perturb_blur(img, k=3):
    x = _denorm(img)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return _renorm(x)


def evaluate_with_perturbation(model, loader, device, use_amp, perturb_fn, return_embeddings=False):
    """Run model on loader with images passed through perturb_fn. Return metrics and optionally embeddings."""
    model.eval()
    all_logits = []
    all_labels = []
    all_embs = [] if return_embeddings else None
    with torch.no_grad():
        for batch in loader:
            img, labels = batch[0], batch[1]
            img = img.to(device, non_blocking=True)
            img = perturb_fn(img)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                if return_embeddings:
                    logits, emb = model(img, return_embedding=True)
                    all_embs.append(emb.cpu().numpy())
                else:
                    logits = model(img)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    # Stable sigmoid: clip to avoid exp overflow
    x = np.clip(logits, -20, 20)
    probs = 1.0 / (1.0 + np.exp(-x))
    pred = (probs >= 0.5).astype(np.float32)
    accuracy = np.mean(pred == labels)
    try:
        from sklearn.metrics import roc_auc_score
        aucs = []
        for j in range(labels.shape[1]):
            if np.unique(labels[:, j]).size > 1:
                aucs.append(roc_auc_score(labels[:, j], probs[:, j]))
            else:
                aucs.append(0.5)
        auc_macro = np.mean(aucs)
    except ImportError:
        auc_macro = 0.0
    out = {"accuracy": float(accuracy), "auc_macro": float(auc_macro)}
    if return_embeddings and all_embs:
        out["embeddings"] = np.vstack(all_embs)
    return out


def run_robust_eval(model, test_loader, device, use_amp, perturbations):
    """For each perturbation, compute accuracy, AUC, and (vs clean) embedding_mean."""
    # Clean pass (and get clean embeddings for stability)
    clean_out = evaluate_with_perturbation(model, test_loader, device, use_amp, perturb_identity, return_embeddings=True)
    emb_clean = clean_out.pop("embeddings", None)
    results = {"clean": {"accuracy": clean_out["accuracy"], "auc_macro": clean_out["auc_macro"], "embedding_mean": 0.0}}

    for name, fn in perturbations.items():
        if name == "clean":
            continue
        out = evaluate_with_perturbation(model, test_loader, device, use_amp, fn, return_embeddings=True)
        emb_pert = out.pop("embeddings", None)
        emb_mean = 0.0
        if emb_clean is not None and emb_pert is not None:
            n = min(len(emb_clean), len(emb_pert))
            d = np.linalg.norm(emb_clean[:n] - emb_pert[:n], axis=1)
            emb_mean = float(d.mean())
        results[name] = {
            "accuracy": out["accuracy"],
            "auc_macro": out["auc_macro"],
            "embedding_mean": emb_mean,
        }
    return results


def plot_results(all_results, out_dir):
    if not _HAS_MPL or not all_results:
        return
    pert_names = list(next(iter(all_results.values())).keys())
    run_names = list(all_results.keys())
    n_pert = len(pert_names)
    n_runs = len(run_names)
    x = np.arange(n_pert)
    width = 0.8 / max(n_runs, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_runs, 1)))

    for metric, ylabel, fname in [
        ("accuracy", "Accuracy", "robust_accuracy.png"),
        ("auc_macro", "AUC-ROC (macro)", "robust_auc.png"),
        ("embedding_mean", "Mean ||φ_clean − φ_pert||₂", "robust_embedding_stability.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, run in enumerate(run_names):
            vals = [all_results[run][p].get(metric, 0) for p in pert_names]
            offset = (i - (n_runs - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=run, color=colors[i])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(pert_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        if metric == "embedding_mean":
            ax.set_title("Embedding stability (lower = more stable)", fontsize=12)
        else:
            ax.set_title(f"Robust evaluation: {metric}", fontsize=12)
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Plots saved to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--runs", type=str, nargs="+", default=["B0", "VAT", "E1"], help="Run names to compare (include VAT for PMH comparison)")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="eval_out_robust")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--no_download", action="store_true", help="Do not auto-download data if missing")
    p.add_argument("--dataset", type=str, default="pneumonia", choices=["pneumonia", "nih"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed, deterministic=args.deterministic)
    use_amp = not args.no_amp and device.type == "cuda"

    _, test_loader, _, _, _ = get_eval_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, max_test_samples=args.max_test_samples,
        seed=args.seed,
        auto_download=not args.no_download, dataset=args.dataset,
    )

    perturbations = {
        "clean": perturb_identity,
        "gaussian_0.05": lambda x: perturb_gaussian(x, 0.05),
        "gaussian_0.10": lambda x: perturb_gaussian(x, 0.10),
        "intensity_0.7": lambda x: perturb_intensity(x, 0.7),
        "intensity_1.3": lambda x: perturb_intensity(x, 1.3),
        "gamma_0.8": lambda x: perturb_gamma(x, 0.8),
        "gamma_1.2": lambda x: perturb_gamma(x, 1.2),
        "rotate_5": lambda x: perturb_rotate(x, 5.0),
        "rotate_10": lambda x: perturb_rotate(x, 10.0),
        "zoom_1.1": lambda x: perturb_zoom(x, 1.1),
        "zoom_0.9": lambda x: perturb_zoom(x, 0.9),
        "contrast_0.8": lambda x: perturb_contrast_jitter(x, 0.8),
        "contrast_1.2": lambda x: perturb_contrast_jitter(x, 1.2),
        "blur_3": lambda x: perturb_blur(x, 3),
    }

    def num_classes_from_ckpt(path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if "classifier.weight" in sd:
            return sd["classifier.weight"].shape[0]
        return 14

    os.makedirs(args.out_dir, exist_ok=True)

    if args.compare:
        all_results = {}
        for run in args.runs:
            ckpt = Path(args.runs_dir) / run / "best.pt"
            if not ckpt.exists():
                print(f"Skip {run}: {ckpt} not found")
                continue
            n = num_classes_from_ckpt(ckpt)
            model = get_model(n, embed_dim=args.embed_dim, pretrained=False).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            results = run_robust_eval(model, test_loader, device, use_amp, perturbations)
            all_results[run] = results
            for name, r in results.items():
                print(f"  {run} / {name:12s}  acc={r['accuracy']:.4f}  auc={r['auc_macro']:.4f}  ||Δφ||={r['embedding_mean']:.4f}")
        # Shift-suite summary: average and worst-case drops from clean.
        summary = {}
        for run, rr in all_results.items():
            clean_acc = rr.get("clean", {}).get("accuracy", 0.0)
            clean_auc = rr.get("clean", {}).get("auc_macro", 0.0)
            pert_names = [k for k in rr.keys() if k != "clean"]
            if pert_names:
                avg_acc = float(np.mean([rr[p]["accuracy"] for p in pert_names]))
                avg_auc = float(np.mean([rr[p]["auc_macro"] for p in pert_names]))
                worst_acc = float(min(rr[p]["accuracy"] for p in pert_names))
                worst_auc = float(min(rr[p]["auc_macro"] for p in pert_names))
            else:
                avg_acc = clean_acc
                avg_auc = clean_auc
                worst_acc = clean_acc
                worst_auc = clean_auc
            summary[run] = {
                "clean_accuracy": clean_acc,
                "clean_auc_macro": clean_auc,
                "avg_shift_accuracy": avg_acc,
                "avg_shift_auc_macro": avg_auc,
                "worst_shift_accuracy": worst_acc,
                "worst_shift_auc_macro": worst_auc,
                "drop_accuracy_worst": float(clean_acc - worst_acc),
                "drop_auc_worst": float(clean_auc - worst_auc),
            }
        payload = dict(all_results)
        payload["__summary__"] = summary
        payload["seed"] = args.seed
        payload["deterministic"] = bool(args.deterministic)
        with open(Path(args.out_dir) / "compare_results_robust.json", "w") as f:
            json.dump(payload, f, indent=2)
        plot_results(all_results, Path(args.out_dir))
        return

    if not args.ckpt or not os.path.isfile(args.ckpt):
        print("Usage: python eval_robust.py --ckpt runs/E1/best.pt  OR  python eval_robust.py --compare")
        return

    n = num_classes_from_ckpt(args.ckpt)
    model = get_model(n, embed_dim=args.embed_dim, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    results = run_robust_eval(model, test_loader, device, use_amp, perturbations)
    print(f"Robust results for {args.ckpt}:")
    for name, r in results.items():
        print(f"  {name:12s}  acc={r['accuracy']:.4f}  auc={r['auc_macro']:.4f}  ||Δφ||={r['embedding_mean']:.4f}")
    with open(Path(args.out_dir) / "eval_robust_single.json", "w") as f:
        json.dump(
            {"ckpt": args.ckpt, "seed": args.seed, "deterministic": bool(args.deterministic), "results": results},
            f,
            indent=2,
        )
    plot_results({Path(args.ckpt).stem: results}, Path(args.out_dir))


if __name__ == "__main__":
    main()
