"""
Experiment 2 — BERT-base-uncased fine-tuned on SST-2 sentiment classification.

Modes
-----
  baseline  Standard cross-entropy fine-tuning.
  pmh       CE + PMH loss: align [CLS] representations of clean vs.
            Gaussian-noised input embeddings. Formalises Perturbation B
            (embedding-space Gaussian, σ_train) in NLP.

The PMH loss here is the direct NLP analogue of Task 04: we treat the BERT
input-embedding space as the "pixel space" and regularise the encoder to
produce identical [CLS] representations for clean and noisy token sequences.

Usage
-----
  # Baseline (3-epoch SST-2 fine-tune, ~20 min on RTX 4090)
  python train.py --mode baseline --out_dir runs/baseline

  # PMH (adds one extra BERT forward pass per step; ~35 min)
  python train.py --mode pmh --noise_sigma 0.05 --pmh_weight 0.3 --out_dir runs/pmh

Outputs: <out_dir>/best.pt, <out_dir>/results.json, <out_dir>/tokenizer/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from transformers import (
        BertForSequenceClassification,
        BertTokenizerFast,
        get_linear_schedule_with_warmup,
    )
except ImportError:
    sys.exit("pip install transformers>=4.35")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets>=2.14")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_loaders(
    tokenizer: BertTokenizerFast,
    batch_size: int = 32,
    max_length: int = 128,
) -> tuple[DataLoader, DataLoader]:
    ds = load_dataset("glue", "sst2")

    def tokenise(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tokenise, batched=True, remove_columns=["sentence", "idx"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    kw = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(ds["validation"], batch_size=batch_size * 2, shuffle=False, **kw)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: BertForSequenceClassification,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    mode: str,
    noise_sigma: float,
    pmh_weight: float,
    pmh_cap_ratio: float,
    use_amp: bool,
) -> dict:
    model.train()
    total_ce = total_pmh = total_loss = n_steps = 0
    t0 = time.perf_counter()

    for batch in tqdm(loader, desc="  train", leave=False, unit="batch"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                ce, ploss, loss = _forward(
                    model, input_ids, attention_mask, labels, device,
                    mode, noise_sigma, pmh_weight, pmh_cap_ratio,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            ce, ploss, loss = _forward(
                model, input_ids, attention_mask, labels, device,
                mode, noise_sigma, pmh_weight, pmh_cap_ratio,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_ce += ce
        total_pmh += ploss
        total_loss += loss.item()
        n_steps += 1

    return {
        "ce_loss": round(total_ce / max(n_steps, 1), 5),
        "pmh_loss": round(total_pmh / max(n_steps, 1), 5),
        "loss": round(total_loss / max(n_steps, 1), 5),
        "epoch_time_s": round(time.perf_counter() - t0, 1),
    }


def _forward(
    model: BertForSequenceClassification,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    mode: str,
    noise_sigma: float,
    pmh_weight: float,
    pmh_cap_ratio: float,
) -> tuple[float, float, torch.Tensor]:
    """Single training step; returns (ce_scalar, pmh_scalar, loss_tensor)."""

    if mode == "baseline":
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss.item(), 0.0, out.loss

    # PMH mode — share the embedding computation between clean and noisy passes.
    # model.bert.embeddings() produces the combined (token + position + segment)
    # embedding tensor, matching exactly what the encoder receives internally.
    embeds = model.bert.embeddings(input_ids=input_ids)  # (B, L, 768)

    # Clean pass: CE loss on original embeddings
    out_clean = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
    )
    ce_loss = out_clean.loss

    # Noisy pass: Perturbation B — Gaussian noise on embedding space
    noisy_embeds = embeds + noise_sigma * torch.randn_like(embeds)
    out_noisy = model(
        inputs_embeds=noisy_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # PMH loss: align [CLS] representations (last hidden layer, position 0)
    cls_clean = out_clean.hidden_states[-1][:, 0, :]   # (B, 768)
    cls_noisy = out_noisy.hidden_states[-1][:, 0, :]   # (B, 768)
    pmh_loss = (
        (F.normalize(cls_clean, dim=1) - F.normalize(cls_noisy, dim=1))
        .pow(2).sum(dim=1).mean()
    )

    # Cap ratio: identical to Task 04 — prevents PMH from overwhelming CE
    cap = pmh_cap_ratio * ce_loss.item()
    eff_w = min(pmh_weight, cap / (pmh_loss.item() + 1e-8))
    loss = ce_loss + eff_w * pmh_loss

    return ce_loss.item(), pmh_loss.item(), loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: BertForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune BERT on SST-2 (baseline or PMH)")
    p.add_argument("--mode", choices=["baseline", "pmh"], default="baseline")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_length", type=int, default=128)
    # PMH-specific
    p.add_argument(
        "--noise_sigma", type=float, default=0.05,
        help="Std of Gaussian noise on input embeddings (Perturbation B).",
    )
    p.add_argument("--pmh_weight", type=float, default=0.3)
    p.add_argument("--pmh_cap_ratio", type=float, default=0.30)
    # I/O
    p.add_argument("--out_dir", default="runs/baseline")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  AMP: {use_amp}  Mode: {args.mode}")
    if args.mode == "pmh":
        print(f"  σ_noise={args.noise_sigma}  λ_pmh={args.pmh_weight}  cap={args.pmh_cap_ratio}")

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model = model.to(device)

    train_loader, val_loader = build_loaders(tokenizer, args.batch_size, args.max_length)
    print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_acc = 0.0
    history: list[dict] = []
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            args.mode, args.noise_sigma, args.pmh_weight, args.pmh_cap_ratio, use_amp,
        )
        val_acc = evaluate(model, val_loader, device, use_amp)
        stats["val_acc"] = round(val_acc, 4)
        history.append(stats)
        print(
            f"  ce={stats['ce_loss']:.4f}  pmh={stats['pmh_loss']:.4f}"
            f"  val_acc={val_acc:.2f}%  t={stats['epoch_time_s']:.0f}s"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")
            tokenizer.save_pretrained(out_dir / "tokenizer")
            print(f"  ✓ New best: {best_val_acc:.2f}%")

    train_time = time.perf_counter() - t_start

    results = {
        "mode": args.mode,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "best_val_acc": round(best_val_acc, 4),
        "train_time_s": round(train_time, 1),
        "noise_sigma": args.noise_sigma if args.mode == "pmh" else None,
        "pmh_weight": args.pmh_weight if args.mode == "pmh" else None,
        "pmh_cap_ratio": args.pmh_cap_ratio if args.mode == "pmh" else None,
        "history": history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest val acc: {best_val_acc:.2f}%  Total time: {train_time:.1f}s")
    print(f"Saved checkpoint + results to {out_dir}")


if __name__ == "__main__":
    main()
