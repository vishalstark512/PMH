"""
Experiment 2 — Measure TDI on BERT [CLS] representations (SST-2).

Two perturbation strategies, both measuring the geometric blind-spot signature:

  Perturbation B (embedding-space Gaussian, exact NLP analogue of Task 04)
      Add Gaussian noise directly to BERT input embeddings before the encoder.
      Mirrors the pixel-space noise used in all vision tasks.
      Measured at σ ∈ {0.0, 0.01, 0.05, 0.10}.

  Perturbation A (semantic-preserving surface substitution)
      Replace ~20% of content words with WordNet synonyms to generate N
      paraphrases per sentence. These are NUISANCE perturbations: they alter
      surface form but not semantic content. TDI measured on the distribution
      of [CLS] drifts across paraphrase variants.
      Prediction: TDI_A(baseline) >> TDI_A(pmh), because PMH trains the
      encoder to ignore embedding-level variance — which synonym substitution
      primarily changes.

Usage
-----
  # Measure a single run
  python measure_tdi.py --ckpt_dir runs/baseline

  # Compare baseline vs PMH (prints summary table)
  python measure_tdi.py --compare runs/baseline runs/pmh

  # Skip synonym TDI (faster, no NLTK needed)
  python measure_tdi.py --ckpt_dir runs/pmh --no_pert_a

Outputs: <ckpt_dir>/tdi_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Shared TDI utility — resolve relative to this file
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "shared"))
from tdi import compute_tdi  # noqa: E402

try:
    from transformers import BertForSequenceClassification, BertTokenizerFast
except ImportError:
    sys.exit("pip install transformers>=4.35")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("pip install datasets>=2.14")


# ---------------------------------------------------------------------------
# NLTK synonym substitution (Perturbation A)
# ---------------------------------------------------------------------------

def _ensure_nltk() -> bool:
    try:
        import nltk
        for resource in [("corpora", "wordnet"), ("tokenizers", "punkt"),
                         ("tokenizers", "punkt_tab")]:
            try:
                nltk.data.find(f"{resource[0]}/{resource[1]}")
            except LookupError:
                nltk.download(resource[1], quiet=True)
        return True
    except ImportError:
        return False


def _synonym(word: str, rng: random.Random) -> str:
    """Return a random WordNet synonym for word, or word itself if none."""
    from nltk.corpus import wordnet  # type: ignore
    candidates: set[str] = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower() and name.isascii():
                candidates.add(name)
    return rng.choice(sorted(candidates)) if candidates else word


def synonym_paraphrase(
    text: str,
    n_variants: int = 10,
    aug_ratio: float = 0.20,
    seed: int = 0,
) -> list[str]:
    """
    Generate n_variants paraphrases by synonym-substituting aug_ratio of tokens.
    Only substitutes alphabetic tokens (skips punctuation/numbers).
    """
    try:
        from nltk.tokenize import word_tokenize  # type: ignore
    except ImportError:
        return [text] * n_variants

    rng = random.Random(seed)
    tokens = word_tokenize(text)
    variants: list[str] = []
    for _ in range(n_variants):
        out = []
        for tok in tokens:
            if tok.isalpha() and rng.random() < aug_ratio:
                out.append(_synonym(tok, rng))
            else:
                out.append(tok)
        variants.append(" ".join(out))
    return variants


# ---------------------------------------------------------------------------
# Model + data loading
# ---------------------------------------------------------------------------

def load_model(ckpt_dir: Path, device: torch.device) -> tuple:
    """Load fine-tuned BERT from ckpt_dir/best.pt; tokenizer from ckpt_dir/tokenizer/."""
    tok_path = ckpt_dir / "tokenizer"
    if not tok_path.exists():
        tok_path = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(str(tok_path))
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    ckpt = ckpt_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}. Run train.py first.")
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, tokenizer


def get_validation_texts(max_samples: int = 872) -> tuple[list[str], list[int]]:
    """Return (texts, labels) from SST-2 validation split."""
    ds = load_dataset("glue", "sst2")["validation"]
    ds = ds.select(range(min(max_samples, len(ds))))
    return list(ds["sentence"]), list(ds["label"])


# ---------------------------------------------------------------------------
# Perturbation B: embedding-space Gaussian
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_cls_embeddings(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizerFast,
    texts: list[str],
    device: torch.device,
    noise_sigma: float = 0.0,
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    """
    Return (N, 768) [CLS] hidden states (last layer) for texts.
    If noise_sigma > 0, adds Gaussian noise to BERT input embeddings before
    the encoder — exact Perturbation B.
    """
    all_cls: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        if noise_sigma > 0:
            embeds = model.bert.embeddings(input_ids=input_ids)
            embeds = embeds + noise_sigma * torch.randn_like(embeds)
            out = model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        cls = out.hidden_states[-1][:, 0, :].cpu().numpy()  # (B, 768)
        all_cls.append(cls)
    return np.concatenate(all_cls, axis=0)


# ---------------------------------------------------------------------------
# Perturbation A: synonym-based paraphrase TDI
# ---------------------------------------------------------------------------

def compute_paraphrase_tdi(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizerFast,
    texts: list[str],
    labels: list[int],
    device: torch.device,
    n_variants: int = 10,
    aug_ratio: float = 0.20,
    max_samples: int = 200,
    batch_size: int = 64,
    max_length: int = 128,
) -> dict:
    """
    For each of max_samples sentences, generate n_variants synonym paraphrases.
    Extract [CLS] for all variants and compute:
      - drift_mean: mean L2 distance between clean [CLS] and paraphrase [CLS]
      - tdi_pert_a: TDI computed on {paraphrase_cls} with class labels expanded.

    Returns dict with drift_mean, tdi_pert_a, intra_mean_pert_a.
    """
    n = min(max_samples, len(texts))
    texts = texts[:n]
    labs = labels[:n]

    all_cls_clean: list[np.ndarray] = []
    all_cls_para: list[np.ndarray] = []
    all_labs_para: list[int] = []

    for idx, (text, lab) in enumerate(
        tqdm(zip(texts, labs), total=n, desc="  Pert-A paraphrase TDI", leave=False)
    ):
        variants = synonym_paraphrase(text, n_variants=n_variants, aug_ratio=aug_ratio, seed=idx)
        all_texts = [text] + variants  # first is clean
        cls = extract_cls_embeddings(
            model, tokenizer, all_texts, device,
            noise_sigma=0.0, batch_size=batch_size, max_length=max_length,
        )
        all_cls_clean.append(cls[0])
        all_cls_para.append(cls[1:])           # (n_variants, 768)
        all_labs_para.extend([lab] * n_variants)

    # Mean drift: how far does [CLS] move under synonym paraphrasing?
    drifts = []
    for clean_v, para_mat in zip(all_cls_clean, all_cls_para):
        for para_v in para_mat:
            drifts.append(float(np.linalg.norm(para_v - clean_v)))
    drift_mean = float(np.mean(drifts))

    # TDI computed on all paraphrase variants
    para_embs = np.concatenate(all_cls_para, axis=0)  # (n * n_variants, 768)
    para_labs = np.array(all_labs_para)
    intra_mean, tdi = compute_tdi(para_embs, para_labs, num_classes=2)

    return {
        "drift_mean": round(drift_mean, 6),
        "tdi_pert_a": round(tdi, 6),
        "intra_mean_pert_a": round(intra_mean, 6),
    }


# ---------------------------------------------------------------------------
# Main measurement logic
# ---------------------------------------------------------------------------

def measure_run(
    ckpt_dir: Path,
    sigmas: list[float],
    run_pert_a: bool,
    n_variants: int,
    max_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    print(f"\n--- Measuring: {ckpt_dir.name} ---")
    model, tokenizer = load_model(ckpt_dir, device)

    texts, labels = get_validation_texts(max_samples)
    labs_arr = np.array(labels)

    results: dict = {
        "run": ckpt_dir.name,
        "n_samples": len(texts),
        "perturbation_b": {},
        "perturbation_a": None,
    }

    # Perturbation B
    for sigma in sigmas:
        print(f"  Pert-B  σ={sigma}", end="  ", flush=True)
        cls = extract_cls_embeddings(
            model, tokenizer, texts, device,
            noise_sigma=sigma, batch_size=batch_size,
        )
        intra, tdi = compute_tdi(cls, labs_arr, num_classes=2)

        # Accuracy under noise
        correct = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_labs = torch.tensor(labels[i : i + batch_size], device=device)
            enc = tokenizer(batch_texts, padding="max_length", truncation=True,
                            max_length=128, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                if sigma > 0:
                    embeds = model.bert.embeddings(input_ids=input_ids)
                    embeds = embeds + sigma * torch.randn_like(embeds)
                    logits = model(inputs_embeds=embeds, attention_mask=attention_mask).logits
                else:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            correct += (logits.argmax(dim=1) == batch_labs).sum().item()
        acc = round(100.0 * correct / len(texts), 2)

        results["perturbation_b"][str(sigma)] = {
            "tdi": round(tdi, 6),
            "intra_mean": round(intra, 6),
            "accuracy": acc,
        }
        print(f"TDI={tdi:.4f}  acc={acc:.2f}%")

    # Perturbation A
    if run_pert_a:
        has_nltk = _ensure_nltk()
        if has_nltk:
            pert_a = compute_paraphrase_tdi(
                model, tokenizer, texts, labels, device,
                n_variants=n_variants, max_samples=min(200, len(texts)),
                batch_size=batch_size,
            )
            results["perturbation_a"] = pert_a
            print(
                f"  Pert-A  drift_mean={pert_a['drift_mean']:.4f}"
                f"  TDI={pert_a['tdi_pert_a']:.4f}"
            )
        else:
            print("  Pert-A: skipped (NLTK not available)")

    out_path = ckpt_dir / "tdi_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")
    return results


def print_comparison(results_list: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("COMPARISON TABLE — Perturbation B (Embedding-Space Gaussian TDI)")
    print("=" * 70)
    # Collect all sigmas
    all_sigmas = sorted(
        {s for r in results_list for s in r["perturbation_b"]},
        key=float,
    )
    header = f"{'Run':<20}" + "".join(f"  σ={s:<6}" for s in all_sigmas)
    print(header)
    print("-" * len(header))
    for r in results_list:
        row = f"{r['run']:<20}"
        for s in all_sigmas:
            tdi = r["perturbation_b"].get(s, {}).get("tdi", "—")
            row += f"  {tdi:<8.4f}" if isinstance(tdi, float) else f"  {'—':<8}"
        print(row)

    print("\nCOMPARISON TABLE — Perturbation A (Synonym Paraphrase Drift)")
    print("-" * 50)
    for r in results_list:
        pa = r.get("perturbation_a")
        if pa:
            print(
                f"  {r['run']:<20}  drift={pa['drift_mean']:.4f}"
                f"  TDI_A={pa['tdi_pert_a']:.4f}"
            )
        else:
            print(f"  {r['run']:<20}  (not computed)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure BERT [CLS] TDI on SST-2")
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="Single run to measure.")
    p.add_argument("--compare", type=str, nargs="+", default=None,
                   help="Two or more run dirs to measure and compare.")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.10],
                   help="Gaussian σ values for Perturbation B.")
    p.add_argument("--no_pert_a", action="store_true",
                   help="Skip synonym paraphrase TDI (faster).")
    p.add_argument("--n_variants", type=int, default=10,
                   help="Paraphrase variants per sentence for Perturbation A.")
    p.add_argument("--max_samples", type=int, default=872,
                   help="Max validation sentences to use.")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    targets: list[Path] = []
    if args.compare:
        targets = [Path(d) for d in args.compare]
    elif args.ckpt_dir:
        targets = [Path(args.ckpt_dir)]
    else:
        sys.exit("Provide --ckpt_dir <path> or --compare <path1> <path2>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results: list[dict] = []
    for d in targets:
        r = measure_run(
            d, args.sigmas,
            run_pert_a=not args.no_pert_a,
            n_variants=args.n_variants,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device=device,
        )
        all_results.append(r)

    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
