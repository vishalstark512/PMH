"""
exp_semantic_tdi/measure_semantic_tdi.py

Semantic Blind Spot experiment — inference-only, no training required.

Tests two predictions derived from the PMH Theorem (Theorem 1):

  PREDICTION 1 — Task fine-tuning creates a semantic blind spot
    ERM on SST-2 forces BERT to exploit surface features correlated with
    sentiment. Under semantic-preserving surface changes (paraphrases), the
    encoder should drift MORE in the ERM-fine-tuned model than in the
    pretrained baseline.  PMH regularises against exactly this drift.

    Models tested:
      bert-pretrained  : bert-base-uncased, no task fine-tuning
      bert-sst2-erm    : bert-base-uncased fine-tuned on SST-2 (cross-entropy)
      bert-sst2-pmh    : bert-base-uncased fine-tuned on SST-2 + PMH loss

    Expected ordering: ERM_drift > pretrained_drift > PMH_drift

  PREDICTION 2 — Scale amplifies the blind spot
    Larger models discover higher-order nuisance correlations that smaller
    models cannot represent.  Pre-trained (no task fine-tuning) models of
    increasing size should show increasing semantic drift.

    Models tested: distilbert-base-uncased (66M) → bert-base (110M) → bert-large (340M)
    Expected ordering: drift(large) > drift(base) > drift(distil)

Dataset
-------
  PAWS "labeled_final" (test split, ~8k pairs).
  label=1 → gold paraphrase pair (same meaning, different surface form)
  label=0 → non-paraphrase pair (different meaning, similar surface form)
  PAWS is adversarially constructed: label-0 pairs LOOK syntactically similar
  but differ semantically, making it a hard test of semantic sensitivity.

Metric
------
  semantic_cosine_drift  = mean (1 - cosine_similarity(CLS(s1), CLS(s2)))
                           for all label=1 paraphrase pairs.

  nonpara_cosine_drift   = same for label=0 non-paraphrase pairs.

  blind_spot_ratio       = semantic_cosine_drift / nonpara_cosine_drift
    → 1.0 : model is equally confused by paraphrases and non-paraphrases
    → 0.0 : model keeps paraphrases perfectly close, separates non-paraphrases

  The blind_spot_ratio is the primary Semantic TDI metric.  A good semantic
  encoder has ratio → 0; an ERM-distorted encoder has ratio → 1.

Usage
-----
  # Full experiment (both predictions) — runs in ~10 min on CPU, ~2 min on GPU
  python measure_semantic_tdi.py

  # Task fine-tuning only (uses existing checkpoints in runs/)
  python measure_semantic_tdi.py --exp task

  # Scale only (no checkpoints needed)
  python measure_semantic_tdi.py --exp scale

  # Quick smoke test (~2 min CPU)
  python measure_semantic_tdi.py --quick

Outputs
-------
  results/semantic_tdi_results.json  — full metrics for all models
  results/semantic_tdi_plot.png      — two-panel bar chart
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_EXP02 = _HERE.parent                          # .../exp02_bert_sst2/
_SHARED = _EXP02.parent.parent / "experiments" / "shared"
# Fallback: try direct sibling of experiments/
if not _SHARED.exists():
    _SHARED = _EXP02.parent / "shared"
sys.path.insert(0, str(_SHARED))

try:
    from tdi import compute_tdi  # noqa: E402
except ImportError:
    # If shared tdi not found, provide a local fallback
    def compute_tdi(embeddings, labels, num_classes=None, max_per_class=400, rng_seed=42):
        embs = np.asarray(embeddings, dtype=np.float64)
        labs = np.asarray(labels, dtype=np.int64)
        if num_classes is None:
            num_classes = int(labs.max()) + 1
        rng = np.random.default_rng(rng_seed)
        intra_dists, centroids = [], []
        for c in range(num_classes):
            mask = labs == c
            if mask.sum() == 0:
                continue
            X = embs[mask]
            n = X.shape[0]
            if n > max_per_class:
                X = X[rng.choice(n, max_per_class, replace=False)]
                n = max_per_class
            centroids.append(X.mean(axis=0))
            if n >= 2:
                d = np.linalg.norm(X[:, None] - X[None], axis=2)
                i, j = np.triu_indices(n, k=1)
                intra_dists.extend(d[i, j].tolist())
        intra_mean = float(np.mean(intra_dists)) if intra_dists else 0.0
        cents = np.stack(centroids)
        nc = cents.shape[0]
        inter = [float(np.linalg.norm(cents[i] - cents[j]))
                 for i in range(nc) for j in range(i + 1, nc)]
        inter_mean = float(np.mean(inter)) if inter else 1.0
        return intra_mean, intra_mean / inter_mean if inter_mean > 0 else 0.0

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        BertForSequenceClassification,
        BertTokenizerFast,
    )
except ImportError:
    sys.exit("pip install transformers>=4.35")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets>=2.14")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_task_models(runs_dir: Path | None = None) -> list[dict]:
    """Build TASK_MODELS list with checkpoint paths resolved from runs_dir.

    runs_dir should contain subdirectories 'baseline/' and 'pmh/' each with
    a 'best.pt' file.  Falls back to the legacy in-tree 'runs/' directory when
    runs_dir is None (original codebase layout).
    """
    base = runs_dir if runs_dir is not None else (_EXP02 / "runs")
    return [
        {
            "name": "bert-pretrained",
            "label": "BERT\n(pretrained)",
            "hf_id": "bert-base-uncased",
            "ckpt": None,
            "color": "#4C72B0",
        },
        {
            "name": "bert-sst2-erm",
            "label": "BERT\n(SST-2 ERM)",
            "hf_id": "bert-base-uncased",
            "ckpt": str(base / "baseline" / "best.pt"),
            "color": "#DD8452",
        },
        {
            "name": "bert-sst2-pmh",
            "label": "BERT\n(SST-2 PMH)",
            "hf_id": "bert-base-uncased",
            "ckpt": str(base / "pmh" / "best.pt"),
            "color": "#55A868",
        },
    ]


TASK_MODELS: list[dict] = _build_task_models()

SCALE_MODELS: list[dict] = [
    {
        "name": "distilbert-66M",
        "label": "DistilBERT\n(66 M)",
        "hf_id": "distilbert-base-uncased",
        "ckpt": None,
        "color": "#C44E52",
    },
    {
        "name": "bert-base-110M",
        "label": "BERT-base\n(110 M)",
        "hf_id": "bert-base-uncased",
        "ckpt": None,
        "color": "#8172B2",
    },
    {
        "name": "bert-large-340M",
        "label": "BERT-large\n(340 M)",
        "hf_id": "bert-large-uncased",
        "ckpt": None,
        "color": "#937860",
    },
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_encoder(cfg: dict, device: torch.device):
    """
    Return (encoder, tokenizer).
    encoder always exposes .last_hidden_state in its output (BertModel API).
    If cfg["ckpt"] is set, load fine-tuned BertForSequenceClassification
    weights then expose the inner .bert module as the encoder.
    """
    hf_id = cfg["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    if cfg["ckpt"] is not None:
        ckpt_path = Path(cfg["ckpt"])
        if not ckpt_path.exists():
            print(f"  [WARN] Checkpoint not found: {ckpt_path}. Skipping {cfg['name']}.")
            return None, None
        clf = BertForSequenceClassification.from_pretrained(hf_id, num_labels=2)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        clf.load_state_dict(state, strict=True)
        encoder = clf.bert          # BertModel — has last_hidden_state
        print(f"  Loaded fine-tuned checkpoint: {ckpt_path.name}")
    else:
        encoder = AutoModel.from_pretrained(hf_id)
        print(f"  Loaded pretrained: {hf_id}")

    encoder = encoder.to(device).eval()
    return encoder, tokenizer


# ---------------------------------------------------------------------------
# CLS extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_cls_embeddings(
    encoder,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    """Return (N, D) CLS embeddings.  Works for BERT and DistilBERT."""
    all_cls: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = encoder(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        all_cls.append(cls)
    return np.concatenate(all_cls, axis=0)


# ---------------------------------------------------------------------------
# PAWS dataset loading
# ---------------------------------------------------------------------------

def load_paws_pairs(
    max_per_label: int = 1000,
    split: str = "test",
) -> tuple[list[str], list[str], list[int]]:
    """
    Return (s1_list, s2_list, labels) from PAWS labeled_final.
    Balances paraphrase (label=1) and non-paraphrase (label=0) counts.
    """
    print(f"Loading PAWS labeled_final ({split} split, max {max_per_label} per label)...")
    ds = load_dataset("paws", "labeled_final", split=split)

    s1_para, s2_para = [], []
    s1_nonpara, s2_nonpara = [], []

    for ex in ds:
        if ex["label"] == 1 and len(s1_para) < max_per_label:
            s1_para.append(ex["sentence1"])
            s2_para.append(ex["sentence2"])
        elif ex["label"] == 0 and len(s1_nonpara) < max_per_label:
            s1_nonpara.append(ex["sentence1"])
            s2_nonpara.append(ex["sentence2"])

    s1 = s1_para + s1_nonpara
    s2 = s2_para + s2_nonpara
    labels = [1] * len(s1_para) + [0] * len(s1_nonpara)

    print(f"  Paraphrase pairs   (label=1): {len(s1_para)}")
    print(f"  Non-paraphrase pairs (label=0): {len(s1_nonpara)}")
    return s1, s2, labels


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def measure_semantic_tdi(
    encoder,
    tokenizer,
    s1_list: list[str],
    s2_list: list[str],
    labels: list[int],
    device: torch.device,
    batch_size: int = 64,
    desc: str = "",
) -> dict:
    """
    Compute Semantic TDI metrics for one model.

    Returns a dict with:
      para_drift        — mean cosine drift for paraphrase pairs
      nonpara_drift     — mean cosine drift for non-paraphrase pairs
      blind_spot_ratio  — para_drift / nonpara_drift  (PRIMARY METRIC)
      tdi_paws_labels   — TDI using compute_tdi on [CLS(s1), CLS(s2)] with PAWS labels
      para_l2_drift     — mean L2 drift for paraphrase pairs (raw, unnormalized)
      nonpara_l2_drift  — same for non-paraphrase
    """
    all_s = s1_list + s2_list
    all_labels = labels + labels  # each s1 and s2 gets the pair's label

    print(f"  Extracting CLS embeddings for {len(all_s)} sentences {desc}...")
    embs = get_cls_embeddings(
        encoder, tokenizer, all_s, device, batch_size=batch_size
    )  # (2N, D)

    n = len(labels)
    embs_s1 = embs[:n]       # (N, D)
    embs_s2 = embs[n:]       # (N, D)
    labs = np.array(labels)  # (N,)

    # Cosine similarity between each (s1, s2) pair
    t1 = torch.from_numpy(embs_s1).float()
    t2 = torch.from_numpy(embs_s2).float()
    cos_sim = F.cosine_similarity(t1, t2, dim=1).numpy()  # (N,)
    cos_drift = 1.0 - cos_sim                              # (N,)

    # L2 drift
    l2_drift = np.linalg.norm(embs_s1 - embs_s2, axis=1)  # (N,)

    para_mask = labs == 1
    nonpara_mask = labs == 0

    para_cos = float(np.mean(cos_drift[para_mask]))
    nonpara_cos = float(np.mean(cos_drift[nonpara_mask]))
    blind_spot_ratio = para_cos / nonpara_cos if nonpara_cos > 0 else float("nan")

    para_l2 = float(np.mean(l2_drift[para_mask]))
    nonpara_l2 = float(np.mean(l2_drift[nonpara_mask]))

    # TDI on pooled embeddings with PAWS labels as class labels
    # (how well does the CLS space separate paraphrase-class from non-paraphrase-class?)
    _, paws_tdi = compute_tdi(embs, np.array(all_labels), num_classes=2)

    return {
        "para_drift": round(para_cos, 6),
        "nonpara_drift": round(nonpara_cos, 6),
        "blind_spot_ratio": round(blind_spot_ratio, 6),
        "tdi_paws_labels": round(paws_tdi, 6),
        "para_l2_drift": round(para_l2, 4),
        "nonpara_l2_drift": round(nonpara_l2, 4),
        "n_para": int(para_mask.sum()),
        "n_nonpara": int(nonpara_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

def run_experiment(
    model_list: list[dict],
    s1: list[str],
    s2: list[str],
    labels: list[int],
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    results = []
    for cfg in model_list:
        print(f"\n[{cfg['name']}]")
        encoder, tokenizer = load_encoder(cfg, device)
        if encoder is None:
            results.append({"name": cfg["name"], "label": cfg["label"], "skipped": True})
            continue

        metrics = measure_semantic_tdi(
            encoder, tokenizer, s1, s2, labels, device,
            batch_size=batch_size,
            desc=f"({cfg['name']})",
        )
        entry = {"name": cfg["name"], "label": cfg["label"], **metrics}
        results.append(entry)

        print(
            f"  para_drift={metrics['para_drift']:.4f}  "
            f"nonpara_drift={metrics['nonpara_drift']:.4f}  "
            f"blind_spot_ratio={metrics['blind_spot_ratio']:.4f}  "
            f"tdi={metrics['tdi_paws_labels']:.4f}"
        )

        # Free GPU memory between models
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    task_results: list[dict],
    scale_results: list[dict],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not found. Skipping plot. pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Semantic Blind Spot: How Much Does [CLS] Drift Under Paraphrase?\n"
        "blind_spot_ratio = cosine_drift(paraphrase) / cosine_drift(non-paraphrase)\n"
        "Higher = more semantically blind  |  PMH Theorem prediction: ERM > Pretrained > PMH",
        fontsize=10,
    )

    def _bar_panel(ax, results, title, model_list):
        valid = [r for r in results if not r.get("skipped")]
        if not valid:
            ax.text(0.5, 0.5, "No results", ha="center", va="center")
            ax.set_title(title)
            return

        names = [r["label"] for r in valid]
        ratios = [r["blind_spot_ratio"] for r in valid]
        colors = [next(m["color"] for m in model_list if m["name"] == r["name"])
                  for r in valid]

        x = range(len(names))
        bars = ax.bar(x, ratios, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

        # Value labels on top of bars
        for bar, val in zip(bars, ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("blind_spot_ratio (↑ worse)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(ratios) * 1.25 + 0.01)
        ax.axhline(y=min(ratios), color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Secondary panel: raw cosine drifts
        ax2 = ax.twinx()
        ax2.set_ylabel("cosine drift", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
        para_drifts = [r["para_drift"] for r in valid]
        nonpara_drifts = [r["nonpara_drift"] for r in valid]
        xi = list(x)
        ax2.plot(xi, para_drifts, "o--", color="#2196F3", alpha=0.7,
                 markersize=5, label="paraphrase drift")
        ax2.plot(xi, nonpara_drifts, "s--", color="#FF5722", alpha=0.7,
                 markersize=5, label="non-para drift")
        ax2.legend(fontsize=7, loc="upper right")

    _bar_panel(
        axes[0], task_results,
        "Prediction 1: Task Fine-tuning Creates Blind Spot\n"
        "(ERM vs PMH vs Pretrained — same architecture)",
        TASK_MODELS,
    )
    _bar_panel(
        axes[1], scale_results,
        "Prediction 2: Scale Amplifies the Blind Spot\n"
        "(Pretrained models, increasing capacity)",
        SCALE_MODELS,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure Semantic TDI (blind spot ratio) on BERT-family models"
    )
    p.add_argument(
        "--exp", choices=["task", "scale", "all"], default="all",
        help="Which experiment to run (default: all).",
    )
    p.add_argument(
        "--max_pairs", type=int, default=1000,
        help="Max PAWS pairs per label (default: 1000).",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Smoke test: 150 pairs per label, smaller batch.",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--out_dir", type=str,
        default=str(_HERE / "results"),
        help="Directory for output JSON and plot.",
    )
    p.add_argument(
        "--runs_dir", type=str, default=None,
        help=(
            "Directory that contains 'baseline/' and 'pmh/' subdirectories "
            "with best.pt checkpoints for the task fine-tuning experiment. "
            "Overrides the default in-tree 'runs/' path so that replication "
            "artifacts stored outside the source tree can be used."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_pairs = 150 if args.quick else args.max_pairs
    batch_size = 32 if args.quick else args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  max_pairs_per_label: {max_pairs}")

    # Resolve task-model checkpoint paths (supports --runs_dir for replication layout)
    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    task_models = _build_task_models(runs_dir)
    if runs_dir is not None:
        print(f"Using runs_dir: {runs_dir}")

    s1, s2, labels = load_paws_pairs(max_per_label=max_pairs, split="test")

    task_results: list[dict] = []
    scale_results: list[dict] = []

    if args.exp in ("task", "all"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Task fine-tuning creates semantic blind spot")
        print("=" * 60)
        task_results = run_experiment(task_models, s1, s2, labels, device, batch_size)

    if args.exp in ("scale", "all"):
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Scale amplifies the blind spot")
        print("=" * 60)
        scale_results = run_experiment(SCALE_MODELS, s1, s2, labels, device, batch_size)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'para_drift':>12} {'nonpara_drift':>14} {'blind_spot_ratio':>18}")
    print("-" * 70)
    for r in task_results + scale_results:
        if r.get("skipped"):
            print(f"  {r['name']:<20}  {'(skipped)':>44}")
        else:
            print(
                f"  {r['name']:<20}  {r['para_drift']:>12.4f}"
                f"  {r['nonpara_drift']:>14.4f}  {r['blind_spot_ratio']:>18.4f}"
            )

    # Save JSON
    out_json = out_dir / "semantic_tdi_results.json"
    payload = {
        "device": str(device),
        "max_pairs_per_label": max_pairs,
        "n_total_pairs": len(labels),
        "paws_split": "test",
        "task_finetuning_experiment": task_results,
        "scale_experiment": scale_results,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Plot
    plot_results(task_results, scale_results, out_dir / "semantic_tdi_plot.png")


if __name__ == "__main__":
    main()
