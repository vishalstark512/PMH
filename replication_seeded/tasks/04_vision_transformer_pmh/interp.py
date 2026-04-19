"""
Mechanistic Interpretability Analysis: ViT + PMH (B0 / B1 / E1)
================================================================
Analyses:
  1. CLS Token Trajectory  — PCA of per-layer CLS embeddings, coloured by class
  2. Logit Lens            — per-layer class confidence on clean vs noisy inputs
  3. Attention Maps        — per-head attention rollout, B0 vs E1
  4. Residual Contributions — per-block L2 norm of delta added to CLS stream
  5. Robustness Commitment  — layer at which model first "commits" to correct class
                             comparing B0/B1/E1 on noisy inputs

Usage:
    python interp.py --runs_dir runs --data_dir ./data --noise_sigma 0.2 --n_samples 512
    python interp.py --runs_dir runs --data_dir ./data --runs B0 E1   # subset of runs

Outputs (saved to <runs_dir>/interp/):
    cls_trajectory_<run>.png
    logit_lens_<run>_clean.png  /  logit_lens_<run>_noisy.png
    attn_maps_<run>_layer<L>.png
    residual_norms.png
    commitment_layer.png
    summary_table.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

# ── import your model ───────────────────────────────────────────────────────
from model import get_model

# ── constants ───────────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4565, 0.4067)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CLASSES      = ["airplane","automobile","bird","cat","deer",
                "dog","frog","horse","ship","truck"]
try:
    PALETTE = plt.colormaps.get_cmap("tab10")
except AttributeError:
    PALETTE = plt.cm.get_cmap("tab10", 10)

# ── normalisation helpers ────────────────────────────────────────────────────
def norm_tensors(device):
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1,3,1,1)
    std  = torch.tensor(CIFAR10_STD,  device=device).view(1,3,1,1)
    return mean, std

def add_noise(images, sigma, mean, std, device, generator=None):
    """Add pixel-space Gaussian noise then renormalise."""
    x = images * std + mean
    if generator is None:
        noise = torch.randn_like(x)
    else:
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    x = (x + sigma * noise).clamp(0, 1)
    return (x - mean) / std

# ── extended model methods (monkey-patched onto loaded model) ────────────────
def _get_attention_maps(model, x):
    """Return list[depth] of (B, heads, N, N) attention weight tensors."""
    B = x.shape[0]
    x = model.patch_embed(x)
    cls = model.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + model.pos_embed
    attn_maps = []
    for blk in model.blocks:
        normed = blk.norm1(x)
        _, w = blk.attn(normed, normed, normed,
                        need_weights=True, average_attn_weights=False)
        attn_maps.append(w.detach().cpu())   # (B, heads, N, N)
        x = blk(x)
    return attn_maps

def _logit_lens(model, x):
    """Return list[depth] of (B, num_classes) logits from each layer's CLS."""
    B = x.shape[0]
    x = model.patch_embed(x)
    cls = model.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + model.pos_embed
    lens = []
    for blk in model.blocks:
        x = blk(x)
        logits = model.head(model.norm(x[:, 0]))
        lens.append(logits.detach().cpu())
    return lens

def _cls_trajectory(model, x):
    """Return list[depth] of (B, embed_dim) CLS vectors (pre-norm)."""
    B = x.shape[0]
    x = model.patch_embed(x)
    cls = model.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + model.pos_embed
    traj = []
    for blk in model.blocks:
        x = blk(x)
        traj.append(x[:, 0].detach().cpu())
    return traj

def _residual_contributions(model, x):
    """Return list[depth] of (B, embed_dim) deltas each block adds to CLS."""
    B = x.shape[0]
    x = model.patch_embed(x)
    cls = model.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + model.pos_embed
    deltas, prev = [], x[:, 0].clone()
    for blk in model.blocks:
        x = blk(x)
        deltas.append((x[:, 0] - prev).detach().cpu())
        prev = x[:, 0].clone()
    return deltas

# ── data loader ──────────────────────────────────────────────────────────────
def get_loader(data_dir, n_samples, batch_size=128, seed=42, num_workers=0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds  = CIFAR10(root=data_dir, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(len(ds), generator=g)[:n_samples].tolist()
    sub = Subset(ds, idx)
    kw = dict(batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    if num_workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(sub, **kw)

# ── load model ───────────────────────────────────────────────────────────────
def load_model(run, runs_dir, device):
    ckpt = Path(runs_dir) / run / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = get_model(num_classes=10).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ── collect batch data ────────────────────────────────────────────────────────
@torch.no_grad()
def collect(model, loader, device, noise_sigma=0.0, noise_seed=42):
    mean, std = norm_tensors(device)
    all_traj, all_lens, all_deltas, all_labels = [], [], [], []
    all_attn = None
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        if noise_sigma > 0:
            gen = torch.Generator(device=device).manual_seed(int(noise_seed) + batch_idx * 997)
            images = add_noise(images, noise_sigma, mean, std, device, generator=gen)
        traj   = _cls_trajectory(model, images)         # list[L] of (B, D)
        lens   = _logit_lens(model, images)             # list[L] of (B, C)
        deltas = _residual_contributions(model, images) # list[L] of (B, D)
        attn   = _get_attention_maps(model, images)     # list[L] of (B,H,N,N)

        all_traj.append([t.numpy() for t in traj])
        all_lens.append([l.numpy() for l in lens])
        all_deltas.append([d.numpy() for d in deltas])
        all_labels.append(labels.numpy())

        if all_attn is None:
            all_attn = [[a.numpy()] for a in attn]
        else:
            for li, a in enumerate(attn):
                all_attn[li].append(a.numpy())

    depth = len(traj)
    # concatenate along batch dimension
    traj_cat   = [np.concatenate([b[l] for b in all_traj],   axis=0) for l in range(depth)]
    lens_cat   = [np.concatenate([b[l] for b in all_lens],   axis=0) for l in range(depth)]
    deltas_cat = [np.concatenate([b[l] for b in all_deltas], axis=0) for l in range(depth)]
    attn_cat   = [np.concatenate(all_attn[l],                axis=0) for l in range(depth)]
    labels_cat = np.concatenate(all_labels, axis=0)
    return traj_cat, lens_cat, deltas_cat, attn_cat, labels_cat

# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — CLS Trajectory (PCA)
# ════════════════════════════════════════════════════════════════════════════
def plot_cls_trajectory(traj, labels, run, out_dir, noise_label="clean"):
    from sklearn.decomposition import PCA
    depth = len(traj)
    fig, axes = plt.subplots(2, depth // 2, figsize=(3 * depth // 2, 7))
    axes = axes.flatten()
    all_data = np.concatenate(traj, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_data)
    for li, (ax, vecs) in enumerate(zip(axes, traj)):
        proj = pca.transform(vecs)
        for c in range(10):
            mask = labels == c
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=[PALETTE(c)], s=6, alpha=0.5, label=CLASSES[c] if li == 0 else "")
        ax.set_title(f"Layer {li+1}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0],[0], marker='o', color='w',
                markerfacecolor=PALETTE(c), markersize=7, label=CLASSES[c]) for c in range(10)]
    fig.legend(handles=handles, ncol=5, loc="lower center",
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"CLS Trajectory — {run} ({noise_label})\n"
                 f"PCA of CLS token at each block", fontsize=11, y=1.01)
    plt.tight_layout()
    p = out_dir / f"cls_trajectory_{run}_{noise_label}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Logit Lens
# ════════════════════════════════════════════════════════════════════════════
def plot_logit_lens(lens, labels, run, out_dir, noise_label="clean"):
    depth = len(lens)
    # per-layer: mean softmax confidence on correct class
    correct_conf = []
    top1_acc     = []
    for l_logits in lens:
        probs = torch.softmax(torch.tensor(l_logits), dim=-1).numpy()
        conf  = probs[np.arange(len(labels)), labels].mean()
        acc   = (l_logits.argmax(axis=1) == labels).mean() * 100
        correct_conf.append(conf)
        top1_acc.append(acc)

    layers = np.arange(1, depth + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(layers, correct_conf, "o-", color="steelblue", lw=2)
    ax1.set_xlabel("Transformer Layer"); ax1.set_ylabel("Mean P(correct class)")
    ax1.set_title("Logit Lens — Correct Class Confidence")
    ax1.set_ylim(0, 1); ax1.set_xticks(layers); ax1.grid(alpha=0.3)

    ax2.plot(layers, top1_acc, "s-", color="darkorange", lw=2)
    ax2.set_xlabel("Transformer Layer"); ax2.set_ylabel("Top-1 Accuracy (%)")
    ax2.set_title("Logit Lens — Top-1 Accuracy per Layer")
    ax2.set_ylim(0, 100); ax2.set_xticks(layers); ax2.grid(alpha=0.3)

    fig.suptitle(f"Logit Lens — {run} ({noise_label})", fontsize=12)
    plt.tight_layout()
    p = out_dir / f"logit_lens_{run}_{noise_label}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")
    return correct_conf, top1_acc

# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Attention Maps (CLS→patches, averaged over samples per class)
# ════════════════════════════════════════════════════════════════════════════
def plot_attention_maps(attn, labels, run, out_dir, layers_to_plot=None, n_patch=8):
    """
    attn: list[depth] of (N, heads, seq, seq)
    Plot CLS-to-patch attention for each head at selected layers.
    Averaged over all samples (one row per head, one column per selected layer).
    """
    depth = len(attn)
    if layers_to_plot is None:
        # pick first, middle, last
        layers_to_plot = sorted({0, depth//2, depth-1})

    num_heads = attn[0].shape[1]
    n_cols    = len(layers_to_plot)

    fig, axes = plt.subplots(num_heads, n_cols,
                             figsize=(3.2 * n_cols, 3 * num_heads))
    if num_heads == 1:
        axes = axes[np.newaxis, :]

    for col_i, layer_i in enumerate(layers_to_plot):
        w = attn[layer_i]              # (N, heads, seq, seq)
        cls_to_patch = w[:, :, 0, 1:] # (N, heads, n_patches)
        mean_map = cls_to_patch.mean(axis=0)  # (heads, n_patches)
        for h in range(num_heads):
            ax  = axes[h, col_i]
            img = mean_map[h].reshape(n_patch, n_patch)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img, cmap="hot", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if col_i == 0:
                ax.set_ylabel(f"Head {h+1}", fontsize=9)
            if h == 0:
                ax.set_title(f"Layer {layer_i+1}", fontsize=9)

    fig.suptitle(f"CLS→Patch Attention Maps — {run}\n"
                 f"(mean over all samples)", fontsize=11, y=1.01)
    plt.tight_layout()
    p = out_dir / f"attn_maps_{run}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Residual Contributions
# ════════════════════════════════════════════════════════════════════════════
def plot_residual_norms(all_deltas_dict, out_dir):
    """
    all_deltas_dict: {run: list[depth] of (N, D)}
    One line per run showing mean L2 norm of per-block delta.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"B0": "steelblue", "B1": "darkorange", "E1": "green"}
    for run, deltas in all_deltas_dict.items():
        norms = [np.linalg.norm(d, axis=1).mean() for d in deltas]
        layers = np.arange(1, len(norms) + 1)
        ax.plot(layers, norms, "o-", label=run,
                color=colors.get(run, "gray"), lw=2)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean L2 norm of block delta")
    ax.set_title("Residual Stream Contributions per Block\n"
                 "(how much each layer 'writes' to the CLS stream)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = out_dir / "residual_norms.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Commitment Layer (noisy inputs)
# ════════════════════════════════════════════════════════════════════════════
def commitment_layer(lens, labels, threshold=0.5):
    """
    For each sample, find the first layer where softmax P(correct) >= threshold.
    Returns array of shape (N,) with layer indices (0-based), -1 if never reached.
    """
    N = len(labels)
    depth = len(lens)
    commit = np.full(N, -1, dtype=int)
    for li, l_logits in enumerate(lens):
        probs = torch.softmax(torch.tensor(l_logits), dim=-1).numpy()
        conf  = probs[np.arange(N), labels]
        newly = (commit == -1) & (conf >= threshold)
        commit[newly] = li
    return commit

def plot_commitment(commit_dict, out_dir, threshold=0.5):
    """
    commit_dict: {run: commit_array}  from noisy inputs.
    Bar chart: mean commitment layer per run.
    Also histogram per run.
    """
    runs = list(commit_dict.keys())
    depths = [len(v) for v in commit_dict.values()]  # unused but illustrative
    colors = {"B0": "steelblue", "B1": "darkorange", "E1": "green"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # mean commitment layer
    means = []
    for run in runs:
        c = commit_dict[run]
        valid = c[c >= 0]
        means.append(valid.mean() if len(valid) else float("nan"))
    bars = ax1.bar(runs, means, color=[colors.get(r, "gray") for r in runs],
                   edgecolor="black", width=0.5)
    ax1.bar_label(bars, fmt="%.2f", padding=3)
    ax1.set_ylabel("Mean layer index (0-based)")
    ax1.set_title(f"Mean Commitment Layer on Noisy Input\n(threshold P(correct) ≥ {threshold})")
    ax1.set_ylim(0, len(commit_dict[runs[0]]) + 0.5)
    ax1.grid(axis="y", alpha=0.3)

    # histogram overlay
    depth = max(c.max() for c in commit_dict.values() if c.max() >= 0) + 1
    bins  = np.arange(-0.5, depth + 0.5, 1)
    for run in runs:
        c = commit_dict[run]
        ax2.hist(c[c >= 0], bins=bins, alpha=0.5, label=run,
                 color=colors.get(run, "gray"), density=True)
    ax2.set_xlabel("Commitment layer"); ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Commitment Layers (noisy)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Commitment Layer Analysis — B0 / B1 / E1 on Noisy Inputs", fontsize=12)
    plt.tight_layout()
    p = out_dir / "commitment_layer.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Logit Lens comparison across runs (single figure)
# ════════════════════════════════════════════════════════════════════════════
def plot_logit_lens_comparison(lens_data, out_dir, noise_label="noisy"):
    """
    lens_data: {run: (correct_conf_list, top1_acc_list)}
    Overlaid lines for all runs on same axes.
    """
    colors = {"B0": "steelblue", "B1": "darkorange", "E1": "green"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for run, (conf, acc) in lens_data.items():
        layers = np.arange(1, len(conf) + 1)
        ax1.plot(layers, conf, "o-", label=run, color=colors.get(run,"gray"), lw=2)
        ax2.plot(layers, acc,  "s-", label=run, color=colors.get(run,"gray"), lw=2)
    ax1.set_xlabel("Layer"); ax1.set_ylabel("Mean P(correct)")
    ax1.set_title("Correct Class Confidence per Layer"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Layer"); ax2.set_ylabel("Top-1 Acc (%)")
    ax2.set_title("Top-1 Accuracy per Layer"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(f"Logit Lens Comparison — {noise_label} inputs", fontsize=12)
    plt.tight_layout()
    p = out_dir / f"logit_lens_comparison_{noise_label}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Clean vs Noisy CLS distance per layer
# ════════════════════════════════════════════════════════════════════════════
def plot_clean_noisy_distance(traj_clean_dict, traj_noisy_dict, out_dir):
    """
    For each run, plot mean L2 distance between clean and noisy CLS vectors at each layer.
    E1 (PMH-trained) should show smaller distances, especially in later layers.
    """
    colors = {"B0": "steelblue", "B1": "darkorange", "E1": "green"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for run in traj_clean_dict:
        tc = traj_clean_dict[run]
        tn = traj_noisy_dict[run]
        dists = [np.linalg.norm(tc[l] - tn[l], axis=1).mean() for l in range(len(tc))]
        layers = np.arange(1, len(dists) + 1)
        ax.plot(layers, dists, "o-", label=run, color=colors.get(run,"gray"), lw=2)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean L2 dist(clean CLS, noisy CLS)")
    ax.set_title("Clean vs Noisy CLS Distance per Layer\n"
                 "(lower = more robust representation)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = out_dir / "clean_noisy_distance.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════
def save_summary(summary_rows, out_dir):
    p = out_dir / "summary_table.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  saved {p}")
    # pretty-print
    header = list(summary_rows[0].keys())
    col_w  = [max(len(h), max(len(str(r[h])) for r in summary_rows)) for h in header]
    fmt    = "  ".join(f"{{:<{w}}}" for w in col_w)
    print("\n" + fmt.format(*header))
    print("  ".join("-" * w for w in col_w))
    for row in summary_rows:
        print(fmt.format(*[str(row[h]) for h in header]))

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir",     type=str, default="runs")
    parser.add_argument("--out_dir",      type=str, default=None, help="Output dir (default: runs_dir/interp)")
    parser.add_argument("--data_dir",     type=str, default="./data")
    parser.add_argument("--runs",         nargs="+", default=["B0","B1","E1"])
    parser.add_argument("--noise_sigma",  type=float, default=0.2)
    parser.add_argument("--n_samples",    type=int, default=512)
    parser.add_argument("--batch_size",   type=int, default=128)
    parser.add_argument("--threshold",    type=float, default=0.5,
                        help="Confidence threshold for commitment layer analysis")
    parser.add_argument("--no_cuda",      action="store_true")
    parser.add_argument("--seed",         type=int, default=42, help="Subset indices + noisy eval in collect()")
    parser.add_argument("--num_workers",  type=int, default=0, help="DataLoader workers for interp subset")
    args = parser.parse_args()

    device  = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_dir) / "interp"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Mechanistic Interpretability  |  device={device}")
    print(f"  runs={args.runs}  noise_sigma={args.noise_sigma}  n={args.n_samples}  seed={args.seed}")
    print(f"  output → {out_dir}")
    print(f"{'='*60}\n")

    loader = get_loader(args.data_dir, args.n_samples, args.batch_size, seed=args.seed, num_workers=args.num_workers)

    # ── per-run data collection ──────────────────────────────────────────────
    traj_clean_dict  = {}
    traj_noisy_dict  = {}
    lens_clean_dict  = {}   # {run: (conf_list, acc_list)}
    lens_noisy_dict  = {}
    deltas_dict      = {}
    commit_dict      = {}
    summary_rows     = []

    for run in args.runs:
        print(f"── {run} ──────────────────────────────────────")
        try:
            model = load_model(run, args.runs_dir, device)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  collecting clean data …")
        traj_c, lens_c, deltas_c, attn_c, labels = collect(
            model, loader, device, noise_sigma=0.0, noise_seed=args.seed)

        print(f"  collecting noisy data (σ={args.noise_sigma}) …")
        traj_n, lens_n, deltas_n, attn_n, _ = collect(
            model, loader, device, noise_sigma=args.noise_sigma, noise_seed=args.seed)

        traj_clean_dict[run] = traj_c
        traj_noisy_dict[run] = traj_n
        deltas_dict[run]     = deltas_c

        # ── plot 1: CLS trajectory ─────────────────────────────────────────
        try:
            from sklearn.decomposition import PCA
            plot_cls_trajectory(traj_c, labels, run, out_dir, "clean")
            plot_cls_trajectory(traj_n, labels, run, out_dir, "noisy")
        except ImportError:
            print("  (sklearn not found — skipping CLS trajectory PCA)")

        # ── plot 2: logit lens ─────────────────────────────────────────────
        cc, ca = plot_logit_lens(lens_c, labels, run, out_dir, "clean")
        nc, na = plot_logit_lens(lens_n, labels, run, out_dir, "noisy")
        lens_clean_dict[run] = (cc, ca)
        lens_noisy_dict[run] = (nc, na)

        # ── plot 3: attention maps ─────────────────────────────────────────
        plot_attention_maps(attn_c, labels, run, out_dir)

        # ── commitment layer ───────────────────────────────────────────────
        commit = commitment_layer(lens_n, labels, threshold=args.threshold)
        commit_dict[run] = commit
        mean_commit = commit[commit >= 0].mean() if (commit >= 0).any() else float("nan")

        # summary stats
        final_clean_conf = cc[-1]
        final_noisy_conf = nc[-1]
        final_clean_acc  = ca[-1]
        final_noisy_acc  = na[-1]
        summary_rows.append({
            "run":                run,
            "final_clean_conf":   f"{final_clean_conf:.3f}",
            "final_noisy_conf":   f"{final_noisy_conf:.3f}",
            "final_clean_acc":    f"{final_clean_acc:.1f}",
            "final_noisy_acc":    f"{final_noisy_acc:.1f}",
            "mean_commit_layer":  f"{mean_commit:.2f}",
        })

    # ── cross-run plots ──────────────────────────────────────────────────────
    if len(deltas_dict) > 1:
        plot_residual_norms(deltas_dict, out_dir)

    if len(commit_dict) > 1:
        plot_commitment(commit_dict, out_dir, threshold=args.threshold)

    if len(lens_clean_dict) > 1:
        plot_logit_lens_comparison(lens_clean_dict, out_dir, noise_label="clean")
        plot_logit_lens_comparison(lens_noisy_dict, out_dir, noise_label="noisy")

    if len(traj_clean_dict) > 1:
        plot_clean_noisy_distance(traj_clean_dict, traj_noisy_dict, out_dir)

    if summary_rows:
        save_summary(summary_rows, out_dir)

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()