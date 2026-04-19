"""
Chest X-ray evaluation: accuracy (threshold 0.5) and AUC-ROC (per-label + macro).
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from model import get_model
from data import get_eval_loaders, ensure_nih_chestxray


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


def evaluate(model, loader, device, use_amp=True):
    """Compute accuracy and AUC-ROC (macro over 14 labels). Returns dict."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            img, labels = batch[0], batch[1]
            img = img.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(img)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    # Stable sigmoid: clip to avoid exp overflow
    x = np.clip(logits, -20, 20)
    probs = 1.0 / (1.0 + np.exp(-x))
    pred = (probs >= 0.5).astype(np.float32)
    # Accuracy: fraction of correct labels (multi-label)
    accuracy = np.mean(pred == labels)
    # AUC-ROC per label then macro
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
    return {"accuracy": float(accuracy), "auc_macro": float(auc_macro)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--compare", action="store_true", help="Compare runs from runs_dir")
    p.add_argument("--runs", type=str, nargs="+", default=["B0", "VAT", "E1"], help="Run names to compare (include VAT for PMH comparison)")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="eval_out")
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

    val_loader, test_loader, _, _, num_classes = get_eval_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, max_test_samples=args.max_test_samples,
        seed=args.seed,
        auto_download=not args.no_download, dataset=args.dataset,
    )

    def num_classes_from_ckpt(path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if "classifier.weight" in sd:
            return sd["classifier.weight"].shape[0]
        return 14

    if args.compare:
        results = {}
        for run in args.runs:
            ckpt = Path(args.runs_dir) / run / "best.pt"
            if not ckpt.exists():
                print(f"Skip {run}: {ckpt} not found")
                continue
            n = num_classes_from_ckpt(ckpt)
            model = get_model(n, embed_dim=args.embed_dim, pretrained=False).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            metrics = evaluate(model, test_loader, device, use_amp)
            results[run] = metrics
            print(f"  {run}: accuracy={metrics['accuracy']:.4f}  AUC (macro)={metrics['auc_macro']:.4f}")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(Path(args.out_dir) / "compare_results.json", "w") as f:
            payload = {**results, "seed": args.seed, "deterministic": bool(args.deterministic)}
            json.dump(payload, f, indent=2)
        print(f"Saved {args.out_dir}/compare_results.json")
        return

    if not args.ckpt or not os.path.isfile(args.ckpt):
        print("Usage: python eval.py --ckpt runs/E1/best.pt  OR  python eval.py --compare")
        return

    num_classes = num_classes_from_ckpt(args.ckpt)
    model = get_model(num_classes, embed_dim=args.embed_dim, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    metrics = evaluate(model, test_loader, device, use_amp)
    print(f"accuracy={metrics['accuracy']:.4f}  auc_macro={metrics['auc_macro']:.4f}")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(Path(args.out_dir) / "eval_results.json", "w") as f:
        json.dump({"ckpt": args.ckpt, "seed": args.seed, "deterministic": bool(args.deterministic), **metrics}, f, indent=2)


if __name__ == "__main__":
    main()
