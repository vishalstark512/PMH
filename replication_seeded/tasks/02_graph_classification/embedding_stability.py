"""
Embedding drift for PROTEINS graph classification:
  mean normalized ||phi_clean(G) - phi_noisy(G)||_2  under node feature noise.

This replicates the Table A1 embedding drift values for Task 02.
Drift is computed on graph-level embeddings (global mean pool output before classifier).

Usage:
  python embedding_stability.py --runs_dir ../../artifacts/models/02_graph_classification \
                                --out_dir  ../../artifacts/results/02_graph_classification/evals \
                                --data_dir ./data

Output: <out_dir>/embedding_drift.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add tasks dir for tdi_utils
_tasks_dir = Path(__file__).resolve().parents[1]
if str(_tasks_dir) not in sys.path:
    sys.path.insert(0, str(_tasks_dir))
from tdi_utils import embedding_drift

from data import get_loaders
from model import get_model

DEFAULT_SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20]
HIDDEN = 128
NUM_LAYERS = 4
DATASET = "PROTEINS"
EVAL_SEED = 12345


def perturb_node_features(data, noise_std, device, seed):
    """Add Gaussian noise to node features; uses per-batch deterministic seed for reproducibility."""
    data = data.clone().to(device)
    if noise_std <= 0:
        return data
    x = data.x
    if x.dtype in (torch.float, torch.float32, torch.float16):
        rng = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=rng)
        data.x = x + noise_std * noise
    else:
        # Discrete features: dropout proportional to sigma (cap 50%)
        drop_rate = min(0.5, 2.0 * noise_std)
        rng = torch.Generator(device=device).manual_seed(seed)
        mask = (torch.rand(x.shape, device=device, generator=rng) > drop_rate).float()
        data.x = (x.float() * mask).to(x.dtype)
    return data


@torch.no_grad()
def extract_graph_embeddings(model, loader, device, noise_sigma=0.0, seed=EVAL_SEED):
    """Extract global-mean-pool graph embeddings for every test graph."""
    model.eval()
    all_embs = []
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if noise_sigma > 0:
            batch_seed = seed + batch_idx * 997
            data = perturb_node_features(data, noise_sigma, device, batch_seed)
        _, _, graph_emb = model(data.x, data.edge_index, data.batch, return_embeddings=True)
        all_embs.append(graph_emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str,
                   default="../../artifacts/models/02_graph_classification",
                   help="Root of model directories (each sub-folder PROTEINS/<run>/best.pt)")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str,
                   default="../../artifacts/results/02_graph_classification/evals")
    p.add_argument("--runs", type=str, nargs="+", default=["B0", "VAT", "E1"])
    p.add_argument("--noise_sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    p.add_argument("--split_seed", type=int, default=42,
                   help="Must match the split seed used during training")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader, info = get_loaders(
        DATASET,
        root=args.data_dir,
        batch_size=args.batch_size,
        seed=args.split_seed,
    )
    num_node_features = info["num_node_features"]
    num_classes = info["num_classes"]
    print(f"Dataset: {DATASET}  node_features={num_node_features}  classes={num_classes}")

    results = {"runs": {}, "noise_sigmas": args.noise_sigmas, "dataset": DATASET}

    for run_name in args.runs:
        ckpt = runs_dir / DATASET / run_name / "best.pt"
        if not ckpt.exists():
            print(f"  Skip {run_name}: checkpoint not found at {ckpt}")
            continue

        model = get_model(
            num_node_features, num_classes,
            hidden=args.hidden, num_layers=args.num_layers,
        ).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"\n{run_name}: loaded {ckpt}")

        embs_clean = extract_graph_embeddings(model, test_loader, device, noise_sigma=0.0)
        results["runs"][run_name] = {"embedding_drift": {}, "embedding_mean": {}}

        for sigma in args.noise_sigmas:
            if sigma == 0.0:
                results["runs"][run_name]["embedding_drift"]["0.0"] = 0.0
                results["runs"][run_name]["embedding_mean"]["0.0"] = 0.0
                continue
            embs_noisy = extract_graph_embeddings(model, test_loader, device,
                                                  noise_sigma=sigma)
            drift = embedding_drift(embs_clean, embs_noisy)
            sigma_key = str(round(sigma, 4))
            results["runs"][run_name]["embedding_drift"][sigma_key] = round(drift, 6)
            results["runs"][run_name]["embedding_mean"][sigma_key] = round(drift, 6)
            print(f"  sigma={sigma:.2f}  drift={drift:.4f}")

    out_path = out_dir / "embedding_drift.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Quick summary
    print("\n--- Embedding drift summary ---")
    for run_name, rv in results["runs"].items():
        dr = rv["embedding_drift"]
        vals = "  ".join(
            f"sigma={s:.2f}: {dr.get(str(round(s, 4)), 0):.4f}"
            for s in args.noise_sigmas if s > 0
        )
        print(f"  {run_name}: {vals}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
