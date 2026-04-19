"""
Evaluation for molecular regression: MAE, MSE, and embedding drift on test set.
Supports multiple noise levels; can compute graph-embedding drift (clean vs noisy) like binding.py.
"""
import argparse
import json
from pathlib import Path

import torch

from data import get_loaders, NUM_TARGETS
from model import get_model
from perturb import add_measurement_noise, MEASUREMENT_NOISE_STD


def _norm_tensors(norm_params, device):
    """Get mean, std as 1D tensors on device (from list or tensor)."""
    mean = norm_params.get("mean", 0.0)
    std = norm_params.get("std", 1.0)
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=torch.float32, device=device)
    else:
        mean = mean.to(device).float()
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=torch.float32, device=device)
    else:
        std = std.to(device).float()
    return mean, std

# Default noise levels for robustness eval (position noise σ in Å)
# Include 0.1–0.2 so MAE can increase with noise; 0.05 Å alone is ~3% of bond length so effect is tiny.
DEFAULT_NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def evaluate(model, loader, device, noise_std=0.0, node_noise_std=0.0, norm_params=None, seed=None):
    """
    Compute MAE and MSE in original target units. Multi-task: pred [B,19], y [B,19];
    norm_params mean/std [19]. Denormalize then average over all elements.
    seed: for reproducible noise when noise_std or node_noise_std > 0.
    """
    model.eval()
    norm_params = norm_params or {}
    total_ae = 0.0
    total_se = 0.0
    n = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            mean, std = _norm_tensors(norm_params, data.x.device)
            if noise_std > 0 or node_noise_std > 0:
                gen = None
                if seed is not None:
                    gen = torch.Generator(device=data.x.device).manual_seed(seed + batch_idx * 997)
                data = add_measurement_noise(
                    data, noise_std=noise_std, node_noise_std=node_noise_std, device=device, generator=gen
                )
            pred_norm = model(data.x, data.pos, data.edge_index, data.batch)
            pred = pred_norm * std + mean
            y = data.y.view(pred.shape[0], -1).float()[:, : pred.shape[1]]
            total_ae += (pred - y).abs().sum().item()
            total_se += ((pred - y) ** 2).sum().item()
            n += y.numel()
    n = max(n, 1)
    return total_ae / n, total_se / n


def evaluate_with_drift(
    model, loader, device, noise_std=0.0, node_noise_std=0.0, norm_params=None, seed=None, n_replicates=1
):
    """
    Compute MAE, RMSE (denorm), and embedding drift ‖emb_noisy - emb_clean‖.
    If seed is not None, noise is deterministic so B0 and E1 see the same perturbed inputs.
    With n_replicates=1 (default), one pred per sample per σ so MAE(σ) increases with σ.
    """
    model.eval()
    norm_params = norm_params or {}
    all_preds = []
    all_targets = []
    all_drifts = []
    n_rep = 1 if (noise_std == 0 and node_noise_std == 0) else max(1, int(n_replicates))
    with torch.no_grad():
        for replicate in range(n_rep):
            rep_offset = replicate * 10000 if seed is not None else 0
            for batch_idx, data in enumerate(loader):
                data = data.to(device)
                mean, std = _norm_tensors(norm_params, data.x.device)
                pred_clean, _, graph_clean = model(
                    data.x, data.pos, data.edge_index, data.batch, return_embeddings=True
                )
                if noise_std > 0 or node_noise_std > 0:
                    generator = None
                    if seed is not None:
                        generator = torch.Generator(device=data.x.device).manual_seed(
                            seed + rep_offset + batch_idx + int(noise_std * 1e6)
                        )
                    data_noisy = add_measurement_noise(
                        data, noise_std=noise_std, node_noise_std=node_noise_std, device=device, generator=generator
                    )
                    pred_noisy, _, graph_noisy = model(
                        data_noisy.x,
                        data_noisy.pos,
                        data_noisy.edge_index,
                        data_noisy.batch,
                        return_embeddings=True,
                    )
                else:
                    pred_noisy, graph_noisy = pred_clean, graph_clean
                pred_denorm = pred_noisy * std + mean
                all_preds.append(pred_denorm.cpu())
                y = data.y.view(pred_clean.shape[0], -1).float()[:, : pred_clean.shape[1]]
                all_targets.append(y.cpu())
                drift = (graph_noisy - graph_clean).norm(dim=1).mean()
                all_drifts.append(drift.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_drifts = torch.stack(all_drifts)
    mae = (all_preds - all_targets).abs().mean().item()
    rmse = ((all_preds - all_targets) ** 2).mean().sqrt().item()
    emb_drift = all_drifts.mean().item()
    # Per-target MAE for plotting per-property (list of 19)
    num_t = all_preds.shape[1]
    mae_per_target = [
        (all_preds[:, t] - all_targets[:, t]).abs().mean().item()
        for t in range(num_t)
    ]
    # Binding-style: scalar MAE = average of per-target MAEs (not mean over all elements)
    mae_avg = sum(mae_per_target) / len(mae_per_target) if mae_per_target else mae
    return {
        "mae": mae,
        "mae_avg": mae_avg,
        "rmse": rmse,
        "emb_drift": emb_drift,
        "mae_per_target": mae_per_target,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128, help="Must match training (128 = Chemistry/binding)")
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Dataset split seed; must match train.py --seed",
    )
    p.add_argument(
        "--noise_levels",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.005, 0.01],
        help="Position noise std (Å) for robustness eval",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, info = get_loaders(
        root=args.data_dir,
        subset=args.subset,
        batch_size=args.batch_size,
        seed=args.split_seed,
    )
    num_node_features = info["num_node_features"]
    num_targets = info["num_targets"]

    model = get_model(
        num_node_features,
        num_targets=num_targets,
        hidden=args.hidden,
        num_layers=args.num_layers,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    norm_params = {"mean": info["target_mean"], "std": info["target_std"]}
    print(f"Checkpoint: {args.checkpoint}  Multi-task {num_targets} targets")
    print("Noise(Å)\tMAE\t\tMSE")
    print("-" * 40)
    for sigma in args.noise_levels:
        mae, mse = evaluate(
            model, test_loader, device, noise_std=sigma, node_noise_std=0.0, norm_params=norm_params
        )
        print(f"{sigma:.4f}\t\t{mae:.6f}\t{mse:.8f}")


if __name__ == "__main__":
    main()
