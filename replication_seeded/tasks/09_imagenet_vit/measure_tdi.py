"""
Experiment 3 — TDI on pretrained ViT-B/16 (ImageNet-1K scale).

All scripts and default outputs for this experiment live under
``experiments/exp03_imagenet_vit/``. Run commands from that directory (or use
``python run.py exp03 …`` from ``experiments/``, which sets cwd here).
``--out_dir`` paths are relative to the current working directory.

Measures the geometric blind-spot signature on the canonical large-scale
vision model: vit_base_patch16_224 (pretrained on ImageNet-1K, ~85.8% top-1).

Two modes
---------
  measure     Load pretrained ViT-B/16, extract penultimate features under
              pixel-space Gaussian noise at σ ∈ {0.0, 0.01, 0.05, 0.10},
              compute TDI. No training required.

  pmh_tune    Fine-tune the last N transformer blocks + head with PMH loss
              (CE + alignment of [CLS] under noisy images); default 24 epochs
              (linear warmup + cosine decay) on ImageNet training data, then
              re-measure TDI.

  corrupt     Top-1 accuracy on ImageNet-C (local tree: <root>/<corruption>/<1-5>/).
              Uses the same class subset and seed as clean val (--max_classes /
              --max_per_class / --seed) for apples-to-apples baseline vs PMH.

Data options
------------
  --data_dir     Parent of val/ or ILSVRC2012_img_val/ (auto-detects common layouts).
                 Do **not** point this at a mixed multi-task folder (e.g. same root as
                 CIFAR + QM9 + COCO): a sibling ``val2017/`` is COCO, not ImageNet val.
  --val_dir      Force validation ImageFolder (class subfolders).
  --val_source   local (default) | hf — Hugging Face validation if local tree is missing.
  --train_dir    Force train root (pmh_tune only).
  --max_classes  Subsample N classes (default 100).

Usage
-----
  # Measure TDI (auto-detects val/, ILSVRC2012_img_val/, or use --val_dir explicitly)
  python measure_tdi.py measure --data_dir D:/imagenet --out_dir results/baseline
  python measure_tdi.py measure --val_dir D:/imagenet/ILSVRC2012_img_val --out_dir results

  # PMH fine-tune + re-measure (requires train/ split; ~2× wall time vs old 10-ep default)
  python measure_tdi.py pmh_tune --data_dir D:/imagenet --out_dir results/pmh

  # ImageNet-C robustness (needs ImageFolder val + imagenet-c checkout)
  python measure_tdi.py corrupt --data_dir D:/imagenet --imagenet_c_root D:/imagenet-c --out_dir results/corrupt
  python measure_tdi.py corrupt --val_dir D:/imagenet/val --imagenet_c_root D:/imagenet-c --checkpoint results/pmh/pmh_best.pt

  # Hugging Face: **validation parquet shards only** (~few GB), not full train (~140GB+)
  python measure_tdi.py measure --val_source hf --out_dir results
  python clean_hf_imagenet_cache.py --delete   # reclaim space after a mistaken full download

  # Sort flat val JPEGs into ImageFolder (one-time)
  python sort_imagenet_validation.py --flat_dir .../ILSVRC2012_img_val --out_dir .../val --devkit .../ILSVRC2012_devkit_t12

Outputs: <out_dir>/tdi_*.json, summary.json. For pmh_tune: baseline_kway.pt (K-way
  init before training), pmh_best.pt (lowest train CE checkpoint).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Shared TDI utility
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "shared"))
from tdi import compute_tdi  # noqa: E402

try:
    import timm
except ImportError:
    sys.exit("pip install timm>=0.9")


# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _count_class_subdirs(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


_WNID_DIR = re.compile(r"^n\d{8}$", re.IGNORECASE)


def _imagenet_synset_subdir_count(root: Path) -> int:
    """Count immediate subfolders named like ImageNet wnids (n01440764, …)."""
    if not root.is_dir():
        return 0
    return sum(1 for p in root.iterdir() if p.is_dir() and _WNID_DIR.match(p.name))


def _looks_like_imagenet_imagefolder(root: Path) -> bool:
    """
    True if ``root`` looks like sorted ImageNet train/val (wnid class folders).

    A generic project folder (e.g. QM9/, TU/, annotations/) has ≥2 subdirs but is
    **not** ImageNet; requiring synset-shaped names avoids mis-detecting --data_dir.
    """
    n_wnid = _imagenet_synset_subdir_count(root)
    total = _count_class_subdirs(root)
    if total < 2:
        return False
    # Full val has 1000 classes; allow smaller trees if every class dir is wnid-shaped.
    if n_wnid >= 8:
        return True
    if n_wnid >= 2 and n_wnid == total:
        return True
    return False


def _env_val_dir() -> str | None:
    v = os.environ.get("PMH_IMAGENET_VAL", "").strip()
    return v or None


def _folder_diag(path: Path, limit: int = 24) -> str:
    """Short listing of immediate children for error messages."""
    if not path.is_dir():
        return f"  (not a directory or missing: {path})"
    try:
        names = sorted(p.name for p in path.iterdir())
    except OSError as e:
        return f"  (cannot read: {e})"
    if not names:
        return "  (empty directory)"
    head = names[:limit]
    more = f" … (+{len(names) - limit} more)" if len(names) > limit else ""
    return "  Contents: " + ", ".join(head) + more


def _discover_val_imagefolders(data_dir: Path, max_depth: int = 6) -> list[Path]:
    """
    Breadth-first search under ``data_dir`` for subdirectories named ``val`` (any case)
    that look like ImageFolder (e.g. ImageNet sorted validation).
    """
    found: list[Path] = []
    if not data_dir.is_dir():
        return found
    q: deque[tuple[Path, int]] = deque([(data_dir.resolve(), 0)])
    while q:
        d, depth = q.popleft()
        if depth > max_depth:
            continue
        try:
            for child in d.iterdir():
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if child.name.casefold() == "val" and _looks_like_imagenet_imagefolder(child):
                    found.append(child.resolve())
                q.append((child, depth + 1))
        except OSError:
            continue
    return found


def _explain_val_path_failure(requested: Path, data_dir: Path) -> str:
    """Why ``requested`` is not a usable val ImageFolder; suggest fixes."""
    req = requested.expanduser()
    lines: list[str] = [
        "The path you gave for validation is not usable as an ImageFolder root.",
        "",
        f"  Requested: {req}",
    ]
    if req.exists():
        if req.is_file():
            lines.append("  → This path exists but is a FILE, not a folder.")
        elif req.is_dir():
            lines.append(
                f"  → Folder exists but has only {_count_class_subdirs(req)} class-like "
                "subfolders (need ≥2)."
            )
    else:
        lines.append("  → This path does NOT exist (wrong path, drive letter, or parent folder name).")

    lines.extend(["", "  Path from leaf to drive (exists?):"])
    cur = req
    for _ in range(12):
        if cur == cur.parent:
            break
        if not cur.exists():
            lines.append(f"    [missing]  {cur}")
        elif cur.is_file():
            lines.append(f"    [file]     {cur}")
        else:
            lines.append(f"    [dir {_count_class_subdirs(cur)} subfolders]  {cur}")
        cur = cur.parent

    # Deepest ancestor that exists as a directory (to show what's really there)
    anchor = req
    while anchor != anchor.parent and not (anchor.exists() and anchor.is_dir()):
        anchor = anchor.parent
    if anchor.exists() and anchor.is_dir():
        lines.extend(["", f"  Listing: {anchor}", _folder_diag(anchor)])

    if data_dir.is_dir():
        discovered = _discover_val_imagefolders(data_dir)
        lines.append("")
        if discovered:
            lines.append("  Auto-discovered val/ ImageFolder(s) under --data_dir:")
            for d in discovered[:5]:
                lines.append(f"    Try: --val_dir \"{d}\"")
            if len(discovered) > 5:
                lines.append(f"    … and {len(discovered) - 5} more.")
        else:
            lines.append(
                f"  No folder named 'val' with class subfolders found under:\n    {data_dir}"
            )
            lines.append(_folder_diag(data_dir))

    lines.extend([
        "",
        "  Common causes:",
        "    • Dataset still zipped, or only train/ downloaded.",
        "    • Validation not sorted into n01440764/ … subfolders (flat JPEG folder).",
        "    • Different layout, e.g. ILSVRC2012_img_val on another drive — use Explorer to copy the real path.",
        "",
        "  PowerShell — find directories named val:",
        f'    Get-ChildItem -Path "{data_dir}" -Recurse -Directory -Filter val -ErrorAction SilentlyContinue | Select-Object -First 5 FullName',
    ])
    return "\n".join(lines)


def resolve_imagenet_val_root(data_dir: Path, explicit_val_dir: str | None) -> Path:
    """
    Find the validation ImageFolder root under ``data_dir`` or use ``explicit_val_dir``.

    Also checks env ``PMH_IMAGENET_VAL`` (full path to val ImageFolder).

    Tries, in order:
      1. ``explicit_val_dir`` if set
      2. ``PMH_IMAGENET_VAL`` env var
      3. Common layouts under ``data_dir`` (val/, ILSVRC devkit CLS-LOC/val/, …)
      4. ``data_dir`` itself if it already looks like sorted ImageNet (n######## class folders)
    """
    data_dir = data_dir.expanduser().resolve()
    tried: list[str] = []

    env_val = explicit_val_dir or _env_val_dir()
    if env_val:
        p = Path(env_val).expanduser().resolve()
        tried.append(str(p))
        if p.is_dir() and _looks_like_imagenet_imagefolder(p):
            return p
        raise FileNotFoundError(_explain_val_path_failure(p, data_dir))

    if not data_dir.exists():
        raise FileNotFoundError(
            "ImageNet --data_dir path does not exist on this machine.\n\n"
            f"  Given: {data_dir}\n\n"
            "  Fix (pick one):\n"
            "    • Use the real path to your ImageNet root in --data_dir, or\n"
            "    • Point --val_dir at the sorted validation ImageFolder (contains n01440764/, …), e.g.\n"
            '        --val_dir "E:\\data\\ILSVRC\\Data\\CLS-LOC\\val"\n'
            "      (then --data_dir can be omitted if --val_dir is set).\n"
            "  PowerShell — search for val:\n"
            '    Get-ChildItem -Path E:\\,D:\\ -Recurse -Directory -Filter val -ErrorAction SilentlyContinue 2>$null | Select-Object -First 10 FullName'
        )
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet --data_dir is not a directory:\n  {data_dir}"
        )

    candidates = [
        data_dir / "val",
        data_dir / "ILSVRC2012_img_val",
        data_dir / "validation",
        data_dir / "Data" / "CLS-LOC" / "val",
        data_dir / "CLS-LOC" / "val",
        data_dir / "ILSVRC" / "Data" / "CLS-LOC" / "val",
        data_dir / "imagenet" / "val",
        data_dir,
    ]
    for c in candidates:
        tried.append(str(c))
        if c.is_dir() and _looks_like_imagenet_imagefolder(c):
            return c.resolve()

    diag = _folder_diag(data_dir)
    hints: list[str] = []
    coco_val = data_dir / "val2017"
    if coco_val.is_dir():
        hints.append(
            "Found `val2017/` under --data_dir: that is almost always **COCO** (flat JPEGs), "
            "not ImageNet. ImageNet needs `val/n01440764/*.JPEG` (or `Data/CLS-LOC/val/...`)."
        )
    val_dir_candidate = data_dir / "val"
    if val_dir_candidate.is_dir() and not _looks_like_imagenet_imagefolder(val_dir_candidate):
        hints.append(
            "`val/` exists but is not sorted ImageNet (no `n########/` class folders). "
            "Use the real CLS-LOC val root, or `--val_source hf`."
        )
    hint_block = ("\n\n  " + "\n  ".join(hints)) if hints else ""

    msg = (
        "Cannot find ImageNet validation data in sorted ImageFolder layout (class dirs n01440764/, …).\n\n"
        f"  --data_dir: {data_dir}\n"
        f"{diag}\n\n"
        "  Checked (in order):\n"
        + "\n".join(f"    - {t}" for t in tried)
        + "\n\n"
        "  Fix (pick one):\n"
        "    • Use a dedicated ImageNet root (not a mixed multi-task data folder), or\n"
        "    • Pass the sorted val folder explicitly:\n"
        '        --val_dir "D:\\path\\to\\folder_with_n01440764_etc"\n'
        "    • Or set environment variable PMH_IMAGENET_VAL to that same path.\n"
        "    • Or use Hugging Face validation only (no local ImageNet tree):\n"
        "        --val_source hf\n"
        "    • Ensure val images are sorted into one subfolder per class (not one flat folder).\n\n"
        "  PowerShell — list what is under data_dir:\n"
        f'    Get-ChildItem "{data_dir}"\n\n'
        "  Devkit layout after unpack sometimes looks like:\n"
        "    <root>\\Data\\CLS-LOC\\val\\n01440764\\*.JPEG\n"
        f"{hint_block}"
    )
    raise FileNotFoundError(msg)


def resolve_imagenet_train_root(data_dir: Path, explicit_train_dir: str | None) -> Path:
    """Train root for pmh_tune (same idea as validation)."""
    data_dir = data_dir.expanduser().resolve()
    tried: list[str] = []

    if explicit_train_dir:
        p = Path(explicit_train_dir).expanduser().resolve()
        tried.append(str(p))
        if not p.is_dir():
            raise FileNotFoundError(f"--train_dir is not a directory:\n  {p}")
        if not _looks_like_imagenet_imagefolder(p):
            raise FileNotFoundError(
                f"--train_dir does not look like sorted ImageNet (need many n######## class folders):\n  {p}"
            )
        return p

    env_tr = os.environ.get("PMH_IMAGENET_TRAIN", "").strip()
    if env_tr:
        p = Path(env_tr).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"PMH_IMAGENET_TRAIN is not a directory:\n  {p}")
        if not _looks_like_imagenet_imagefolder(p):
            raise FileNotFoundError(
                f"PMH_IMAGENET_TRAIN does not look like sorted ImageNet:\n  {p}"
            )
        return p

    candidates = [
        data_dir / "train",
        data_dir / "ILSVRC2012_img_train",
        data_dir / "Data" / "CLS-LOC" / "train",
        data_dir / "CLS-LOC" / "train",
        data_dir / "ILSVRC" / "Data" / "CLS-LOC" / "train",
        data_dir / "imagenet" / "train",
    ]
    for c in candidates:
        tried.append(str(c))
        if c.is_dir() and _looks_like_imagenet_imagefolder(c):
            return c.resolve()

    # If user only has train at repo root as ImageFolder (uncommon)
    tried.append(str(data_dir))
    if data_dir.is_dir() and _looks_like_imagenet_imagefolder(data_dir):
        return data_dir.resolve()

    msg = (
        "Cannot find ImageNet training data for pmh_tune.\n\n"
        f"  --data_dir: {data_dir}\n\n"
        "  Checked:\n"
        + "\n".join(f"    - {t}" for t in tried)
        + "\n\n"
        "  Pass --train_dir to your train ImageFolder root (class subfolders), or place "
        "data under `<data_dir>/train` or `<data_dir>/ILSVRC2012_img_train`.\n"
    )
    raise FileNotFoundError(msg)


def _default_num_workers() -> int:
    return 0 if platform.system() == "Windows" else 4


def _subsample_dataset(
    ds: datasets.ImageFolder,
    max_classes: int,
    max_per_class: int,
    seed: int = 42,
) -> tuple[Subset, dict[int, int]]:
    """
    Select max_classes random classes, up to max_per_class samples each.
    Returns (Subset, old_class_idx -> new_class_idx mapping).
    """
    rng = np.random.default_rng(seed)
    all_classes = list(range(len(ds.classes)))
    selected = sorted(rng.choice(all_classes, min(max_classes, len(all_classes)), replace=False).tolist())
    class_map = {old: new for new, old in enumerate(selected)}

    indices: list[int] = []
    for old_idx in selected:
        class_samples = [i for i, (_, c) in enumerate(ds.samples) if c == old_idx]
        n = min(max_per_class, len(class_samples))
        chosen = rng.choice(class_samples, n, replace=False).tolist()
        indices.extend(chosen)

    return Subset(ds, indices), class_map


def _build_loader(
    split_root: Path,
    max_classes: int,
    max_per_class: int,
    batch_size: int,
    train: bool = False,
    num_workers: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, dict[int, int]]:
    transform = _TRAIN_TRANSFORM if train else _EVAL_TRANSFORM
    split_root = split_root.resolve()
    ds = datasets.ImageFolder(str(split_root), transform=transform)
    subset, class_map = _subsample_dataset(ds, max_classes, max_per_class, seed=seed)
    nw = _default_num_workers() if num_workers is None else num_workers
    kw: dict = dict(batch_size=batch_size, shuffle=train, num_workers=nw, pin_memory=torch.cuda.is_available())
    if nw > 0:
        kw["persistent_workers"] = True
    loader = DataLoader(subset, **kw)
    return loader, class_map


class _HFImageNetValDataset(Dataset):
    """Hugging Face ``ILSVRC/imagenet-1k`` validation rows as (tensor, label)."""

    def __init__(self, hf_split, transform):
        self.hf = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, idx: int):
        ex = self.hf[idx]
        img = ex["image"]
        if getattr(img, "mode", None) is not None and img.mode != "RGB":
            img = img.convert("RGB")
        lab = ex.get("label")
        if lab is None:
            lab = ex["labels"]
        return self.transform(img), int(lab)


def _hf_token_resolve() -> bool | str:
    env_tok = os.environ.get("HF_TOKEN", "").strip()
    return env_tok if env_tok else True


def _hf_imagenet_validation_parquet_urls(repo_id: str, hf_token: bool | str) -> list[str]:
    """
    List HTTPS URLs for ``validation-*.parquet`` shards only (no train shards).

    ``load_dataset("ILSVRC/imagenet-1k", split="validation")`` triggers a full
    ``download_and_prepare`` that fetches all 294 train parquet files (~140GB).
    Loading the parquet builder with explicit validation URLs avoids that.
    """
    from huggingface_hub import HfApi

    tok = None if hf_token is True else hf_token
    api = HfApi(token=tok)
    rev = "main"
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=rev)
    val_paths = sorted(
        f
        for f in files
        if f.endswith(".parquet") and Path(f).name.startswith("validation-")
    )
    if not val_paths:
        return []

    base = f"https://huggingface.co/datasets/{repo_id}/resolve/{rev}/"
    return [base + p.replace("\\", "/") for p in val_paths]


def _load_hf_imagenet_validation_dataset():
    """
    Download / open HF ImageNet-1k **validation** parquet shards only (~few GB, not train).
    Returns the ``datasets.Dataset`` (validation split).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "HF validation requires: pip install datasets\n"
            "Then: huggingface-cli login\n"
            "And accept terms for ILSVRC/imagenet-1k on the Hugging Face website."
        ) from e

    hf_token = _hf_token_resolve()
    hf_val = None
    last_err: Exception | None = None

    for repo_id in ("ILSVRC/imagenet-1k", "imagenet-1k"):
        try:
            urls = _hf_imagenet_validation_parquet_urls(repo_id, hf_token)
            if not urls:
                last_err = RuntimeError(f"No validation-*.parquet files listed for {repo_id}")
                continue
            print(f"  Loading validation only: {len(urls)} parquet shard(s) from {repo_id} (~few GB, not full train)")
            hf_val = load_dataset(
                "parquet",
                data_files={"validation": urls},
                split="validation",
                token=hf_token,
            )
            print(f"  Rows: {len(hf_val)}")
            break
        except Exception as e:
            last_err = e
            hf_val = None
            continue

    if hf_val is None:
        err = last_err
        gated_hint = ""
        es = str(err or "").lower()
        if "gated" in es or "authenticated" in es or "401" in es or "403" in es:
            gated_hint = (
                "\n"
                "  Gated dataset — do **all** of this (order matters):\n"
                "    A. Log in: huggingface-cli login  (or HF_TOKEN=hf_...)\n"
                "    B. Accept terms: https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
                "    C. Re-run.\n\n"
            )
        raise RuntimeError(
            "Could not load ImageNet-1k **validation** from Hugging Face (validation shards only).\n"
            "  1. pip install datasets huggingface_hub\n"
            "  2. huggingface-cli login   (or set HF_TOKEN)\n"
            "  3. Accept dataset terms on the Hub\n"
            f"{gated_hint}"
            f"  Underlying error: {err}"
        ) from err

    return hf_val


def _hf_labels_array(hf_val) -> np.ndarray:
    lab_col = "label" if "label" in hf_val.column_names else "labels"
    return np.asarray(hf_val[lab_col], dtype=np.int64)


def _hf_pick_classes(labels: np.ndarray, max_classes: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    unique_cls = np.unique(labels)
    k = min(max_classes, len(unique_cls))
    return sorted(rng.choice(unique_cls, size=k, replace=False).tolist())


def _build_hf_val_loader(
    max_classes: int,
    max_per_class: int,
    batch_size: int,
    seed: int = 42,
    num_workers: int | None = None,
) -> tuple[DataLoader, dict[int, int]]:
    """
    Load ImageNet-1K validation from the Hugging Face Hub (no local ``val/`` tree).

    Only **validation** parquet shards are downloaded (on the order of a few GB), not train.

    Requires: ``pip install datasets``, ``huggingface-cli login``, and accepting the
    dataset terms on https://huggingface.co/datasets/ILSVRC/imagenet-1k
    """
    hf_val = _load_hf_imagenet_validation_dataset()
    labels = _hf_labels_array(hf_val)
    n = len(labels)
    if n < 1:
        raise RuntimeError("Hugging Face validation split is empty.")

    selected = _hf_pick_classes(labels, max_classes, seed)
    class_map = {old: new for new, old in enumerate(selected)}
    selected_set = set(selected)
    counts = {c: 0 for c in selected}
    indices: list[int] = []
    for i in range(n):
        lab = int(labels[i])
        if lab not in selected_set:
            continue
        if counts[lab] >= max_per_class:
            continue
        indices.append(i)
        counts[lab] += 1

    base = _HFImageNetValDataset(hf_val, _EVAL_TRANSFORM)
    subset = Subset(base, indices)
    # HF datasets often misbehave with multi-process DataLoader; keep single-threaded.
    nw = 0 if num_workers is None else num_workers
    kw: dict = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    if nw > 0:
        kw["persistent_workers"] = True
    loader = DataLoader(subset, **kw)
    return loader, class_map


def build_hf_pmh_train_eval_loaders(
    max_classes: int,
    max_per_class: int,
    hf_train_per_class: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[int, int], str]:
    """
    PMH fine-tune using **only** the downloaded HF validation split: first
    ``hf_train_per_class`` images per class (file order) for training, remainder
    of the per-class cap (up to ``max_per_class``) for **held-out** TDI.

    No ImageNet train tarball (~150GB) required.
    """
    if hf_train_per_class < 1:
        raise ValueError("hf_train_per_class must be >= 1")
    if hf_train_per_class >= max_per_class:
        raise ValueError(
            "Need max_per_class > hf_train_per_class so some images stay held out for TDI "
            f"(got max_per_class={max_per_class}, hf_train_per_class={hf_train_per_class})."
        )

    hf_val = _load_hf_imagenet_validation_dataset()
    labels = _hf_labels_array(hf_val)
    n = len(labels)
    if n < 1:
        raise RuntimeError("Hugging Face validation split is empty.")

    selected = _hf_pick_classes(labels, max_classes, seed)
    class_map = {old: new for new, old in enumerate(selected)}
    selected_set = set(selected)

    train_indices: list[int] = []
    eval_indices: list[int] = []
    for c in selected:
        idxs = [i for i in range(n) if int(labels[i]) == c]
        idxs = idxs[:max_per_class]
        if len(idxs) <= hf_train_per_class:
            n_tr = max(1, len(idxs) // 2)
        else:
            n_tr = hf_train_per_class
        train_indices.extend(idxs[:n_tr])
        eval_indices.extend(idxs[n_tr:])

    tr = Subset(_HFImageNetValDataset(hf_val, _TRAIN_TRANSFORM), train_indices)
    ev = Subset(_HFImageNetValDataset(hf_val, _EVAL_TRANSFORM), eval_indices)
    kw = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    train_ld = DataLoader(tr, batch_size=batch_size, shuffle=True, **kw)
    eval_ld = DataLoader(ev, batch_size=batch_size, shuffle=False, **kw)
    desc = (
        "HF ILSVRC val only: PMH train "
        f"{len(train_indices)} imgs (~{hf_train_per_class}/class); "
        f"TDI held-out {len(eval_indices)} imgs (rest of ≤{max_per_class}/class)"
    )
    print(f"  {desc}")
    return train_ld, eval_ld, class_map, desc


# ---------------------------------------------------------------------------
# HF *training* split — in-memory subset for full train/val separation
# ---------------------------------------------------------------------------

class _HFImageNetTrainDataset(Dataset):
    """Wraps a list of PIL images + remapped integer labels collected from HF train split."""

    def __init__(self, images: list, labels: list[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if not hasattr(img, "convert"):
            from PIL import Image as _PIL
            img = _PIL.fromarray(img)
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


_FREQ_CACHE = _HERE / "results" / "hf_train_class_freqs.json"


def _get_hf_train_class_frequencies(cache_path: Path = _FREQ_CACHE) -> dict[int, int]:
    """
    Read the label column only from all HF training parquet shards and return
    per-class image counts.  Parquet is columnar so only the tiny label column
    (~4 bytes × N rows per shard) is transferred — roughly 5 MB total even for
    the full 1.28 M-image training set.  Result is cached to disk.
    """
    if cache_path.is_file():
        with open(cache_path) as f:
            return {int(k): v for k, v in json.load(f).items()}

    try:
        import pyarrow.parquet as pq  # noqa: F401 — just for the error message
    except ImportError as exc:
        raise RuntimeError("pip install pyarrow  (needed for label-column scan)") from exc

    from huggingface_hub import HfApi, HfFileSystem

    hf_token = _hf_token_resolve()
    tok = None if hf_token is True else hf_token
    api = HfApi(token=tok)
    repo_id = "ILSVRC/imagenet-1k"
    files = list(api.list_repo_files(repo_id, repo_type="dataset"))
    # Training parquet files are named "train-NNNNN-of-NNNNN.parquet".
    train_paths = sorted(
        f for f in files
        if f.endswith(".parquet") and "train" in Path(f).name and "validation" not in Path(f).name
    )
    if not train_paths:
        # Fallback: any non-validation, non-test parquet
        train_paths = sorted(
            f for f in files
            if f.endswith(".parquet") and "validation" not in f and "test" not in f
        )
    if not train_paths:
        raise RuntimeError(
            f"No training parquet shards found for {repo_id}.\n"
            f"Parquet files seen: {[f for f in files if f.endswith('.parquet')][:10]}"
        )

    import pyarrow.parquet as pq

    # HfFileSystem handles auth and supports HTTP range reads so pyarrow only
    # fetches the parquet footer + label column bytes, not the embedded image data.
    fs = HfFileSystem(token=tok)
    freqs: dict[int, int] = {}

    print(f"  Reading label column from {len(train_paths)} training shards via HfFileSystem…")
    for path in tqdm(train_paths, desc="freq scan", unit="shard"):
        hf_path = f"datasets/{repo_id}/{path}"
        with fs.open(hf_path, "rb") as fobj:
            table = pq.read_table(fobj, columns=["label"])
        for lab in table["label"].to_pylist():
            freqs[int(lab)] = freqs.get(int(lab), 0) + 1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(freqs, f)
    print(f"  Cached class frequencies → {cache_path}")
    return freqs


def _pick_most_least_classes(freqs: dict[int, int], n_most: int, n_least: int) -> list[int]:
    """Return n_least lowest-frequency + n_most highest-frequency class IDs (sorted)."""
    ordered = sorted(freqs.items(), key=lambda x: x[1])
    least = [c for c, _ in ordered[:n_least]]
    most  = [c for c, _ in ordered[-n_most:]]
    return sorted(set(least + most))


def _collect_hf_train_subset(
    selected_classes: list[int],
    per_class: int,
    seed: int,
) -> tuple[list, list[int]]:
    """
    Stream the HF training split and collect up to *per_class* images for each
    class in *selected_classes*.  Downloads parquet shards on-demand (one at a
    time); stops as soon as every class has enough samples.

    Returns ``(pil_images, remapped_labels)`` already shuffled.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    hf_token = _hf_token_resolve()
    token = None if hf_token is True else hf_token

    selected_set = set(selected_classes)
    class_map = {old: new for new, old in enumerate(selected_classes)}
    buffers: dict[int, list] = {c: [] for c in selected_classes}
    done: set[int] = set()

    n_need = len(selected_classes) * per_class
    print(
        f"  Streaming HF train split: {len(selected_classes)} classes × {per_class}/class"
        f" = {n_need:,} images target"
    )
    print("  (parquet shards downloaded on-demand; stops early once target is met)")

    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, token=token)
    collected = 0
    for item in ds:
        label = item["label"]
        if label not in selected_set or label in done:
            continue
        buffers[label].append(item["image"])
        collected += 1
        if len(buffers[label]) >= per_class:
            done.add(label)
        if len(done) == len(selected_classes):
            break

    short = {c: len(v) for c, v in buffers.items() if len(v) < per_class}
    if short:
        print(f"  Note: {len(short)} classes have <{per_class} images in HF train split.")

    all_imgs: list = []
    all_labs: list[int] = []
    for c in selected_classes:
        all_imgs.extend(buffers[c])
        all_labs.extend([class_map[c]] * len(buffers[c]))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_imgs)).tolist()
    all_imgs = [all_imgs[i] for i in perm]
    all_labs = [all_labs[i] for i in perm]

    print(f"  Collected {len(all_imgs):,} training images from HF train split.")
    return all_imgs, all_labs


def build_hf_full_split_loaders(
    n_classes: int,
    train_per_class: int,
    test_per_class: int,
    batch_size: int,
    seed: int,
    class_mode: str = "random",
) -> tuple[DataLoader, DataLoader, dict[int, int], str]:
    """
    Build train + test loaders with **no data leakage**:

    * **Train**: HF *training* split, *train_per_class* images/class (streamed).
    * **Test**:  HF *validation* split, *test_per_class* images/class (cached).
    * **Class selection**: ``'random'`` picks *n_classes* uniformly; ``'most_least'``
      picks the 50 most + 50 least frequent classes from the training distribution
      (requires a one-time label-column scan, result cached to disk).

    Typical usage::

        train_ld, test_ld, class_map, desc = build_hf_full_split_loaders(
            n_classes=100, train_per_class=450, test_per_class=50,
            batch_size=128, seed=42, class_mode='most_least',
        )
    """
    # --- class selection ---
    hf_val = _load_hf_imagenet_validation_dataset()
    labels_val = _hf_labels_array(hf_val)

    if class_mode == "most_least":
        freqs = _get_hf_train_class_frequencies()
        n_half = n_classes // 2
        selected = _pick_most_least_classes(freqs, n_most=n_half, n_least=n_classes - n_half)
        print(f"  Class mode=most_least: {n_classes - n_half} least-freq + {n_half} most-freq classes selected.")
    else:
        selected = _hf_pick_classes(labels_val, n_classes, seed)
        print(f"  Class mode=random: {len(selected)} classes selected (seed={seed}).")

    class_map = {old: new for new, old in enumerate(selected)}
    selected_set = set(selected)

    # --- test loader from HF val split ---
    test_indices: list[int] = []
    counts: dict[int, int] = {c: 0 for c in selected}
    for i, raw_lab in enumerate(labels_val.tolist()):
        lab = int(raw_lab)
        if lab not in selected_set or counts[lab] >= test_per_class:
            continue
        test_indices.append(i)
        counts[lab] += 1
    test_ds = Subset(_HFImageNetValDataset(hf_val, _EVAL_TRANSFORM), test_indices)
    kw = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    # --- train loader from HF training split ---
    train_imgs, train_labels_remap = _collect_hf_train_subset(selected, train_per_class, seed)
    train_ds = _HFImageNetTrainDataset(train_imgs, train_labels_remap, _TRAIN_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)

    desc = (
        f"HF full split ({class_mode}): {len(selected)} classes | "
        f"train {len(train_imgs):,} imgs ({train_per_class}/class, HF train) | "
        f"test {len(test_indices):,} imgs ({test_per_class}/class, HF val)"
    )
    print(f"  {desc}")
    return train_loader, test_loader, class_map, desc


def build_validation_loader(
    val_source: str,
    val_root: Path | None,
    max_classes: int,
    max_per_class: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, dict[int, int], str]:
    """Return (val_loader, class_map, human-readable data description)."""
    if val_source == "hf":
        loader, cmap = _build_hf_val_loader(max_classes, max_per_class, batch_size, seed=seed)
        return loader, cmap, "Hugging Face ILSVRC/imagenet-1k validation split"
    if val_root is None:
        raise ValueError("val_root is required when val_source=local")
    loader, cmap = _build_loader(
        val_root, max_classes, max_per_class, batch_size, train=False, seed=seed
    )
    return loader, cmap, str(val_root.resolve())


# ---------------------------------------------------------------------------
# ViT model helpers
# ---------------------------------------------------------------------------

def load_vit(pretrained: bool = True, num_classes: int = 0) -> nn.Module:
    """
    Load ViT-B/16 from timm.
    num_classes=0  → feature extractor (returns 768-dim CLS vector).
    num_classes=1000 → full classifier.
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def _add_pixel_noise(
    images: torch.Tensor,
    noise_sigma: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Add Gaussian noise in pixel space (denorm → add noise → clamp → renorm)."""
    x = images * std + mean           # [0, 1] space
    x = (x + noise_sigma * torch.randn_like(x)).clamp(0, 1)
    return (x - mean) / std






# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    class_map: dict[int, int],
    device: torch.device,
    noise_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (N, 768) features and (N,) remapped labels.
    model should return features (num_classes=0) or we use forward_features.
    """
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    model.eval()

    all_feats: list[np.ndarray] = []
    all_labs: list[int] = []

    for images, targets in tqdm(loader, desc=f"  Extract σ={noise_sigma}", leave=False):
        images = images.to(device, non_blocking=True)
        if noise_sigma > 0:
            images = _add_pixel_noise(images, noise_sigma, mean, std)

        # Use forward_features to get (B, seq, D), then CLS token
        feats = model.forward_features(images)  # (B, 197, 768) for ViT-B/16
        cls = feats[:, 0, :].cpu().numpy()      # (B, 768) — CLS token

        all_feats.append(cls)
        remapped = [class_map[t.item()] for t in targets]
        all_labs.extend(remapped)

    return np.concatenate(all_feats, axis=0), np.array(all_labs)


# ---------------------------------------------------------------------------
# PMH fine-tuning
# ---------------------------------------------------------------------------

def pmh_finetune_with_loader(
    train_loader: DataLoader,
    class_map: dict[int, int],
    out_dir: Path,
    epochs: int,
    lr: float,
    noise_sigma: float,
    pmh_weight: float,
    pmh_cap_ratio: float,
    freeze_blocks: int,
    device: torch.device,
    use_amp: bool,
    warmup_epochs: int = 0,
    train_labels_remapped: bool = False,
) -> nn.Module:
    """PMH fine-tune from a training ``DataLoader`` and ImageNet label remap.

    If ``train_labels_remapped`` is True, batch targets are already in ``0..num_classes-1``
    (e.g. ``build_hf_full_split_loaders`` / in-memory HF train subset). Otherwise targets are
    raw ImageNet class IDs and are remapped with ``class_map`` (local ImageFolder, HF val rows).
    """
    wtxt = f", warmup={warmup_epochs}" if warmup_epochs else ""
    print(f"\nPMH fine-tune: {epochs} epochs{wtxt}, σ={noise_sigma}, freeze first {freeze_blocks}/12 blocks")

    num_cls = len(class_map)
    print(f"  Training on {num_cls} classes (K-way head; targets remapped 0..{num_cls - 1})")

    model = load_vit(pretrained=True, num_classes=num_cls).to(device)

    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks[freeze_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.norm.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.05,
    )
    warmup_epochs = max(0, min(warmup_epochs, max(epochs - 1, 0)))
    cos_t = max(epochs - warmup_epochs, 1)
    if warmup_epochs > 0:
        w = LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs
        )
        c = CosineAnnealingLR(optimizer, T_max=cos_t)
        scheduler = SequentialLR(
            optimizer, schedulers=[w, c], milestones=[warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    best_ckpt = out_dir / "pmh_best.pt"
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_ce = total_pmh = n_steps = 0
        t0 = time.perf_counter()

        for images, targets in tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            if train_labels_remapped:
                targets = targets.to(device, non_blocking=True, dtype=torch.long)
            else:
                targets = torch.tensor(
                    [class_map[t.item()] for t in targets], device=device
                )

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    ce, pmh_loss, loss = _pmh_step(
                        model, images, targets, mean, std,
                        noise_sigma, pmh_weight, pmh_cap_ratio,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                ce, pmh_loss, loss = _pmh_step(
                    model, images, targets, mean, std,
                    noise_sigma, pmh_weight, pmh_cap_ratio,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_ce += ce
            total_pmh += pmh_loss
            n_steps += 1

        scheduler.step()
        avg_ce = total_ce / max(n_steps, 1)
        avg_pmh = total_pmh / max(n_steps, 1)
        t_ep = time.perf_counter() - t0
        print(f"  Epoch {epoch}  ce={avg_ce:.4f}  pmh={avg_pmh:.4f}  t={t_ep:.0f}s")

        if avg_ce < best_loss:
            best_loss = avg_ce
            torch.save(model.state_dict(), best_ckpt)

    print(f"  PMH fine-tune complete. Best ckpt: {best_ckpt}")
    state = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)

    feat_model = load_vit(pretrained=False, num_classes=0).to(device)
    feat_state = {k: v for k, v in model.state_dict().items() if not k.startswith("head.")}
    feat_model.load_state_dict(feat_state, strict=False)
    return feat_model




def pmh_finetune(
    train_root: Path,
    out_dir: Path,
    max_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_sigma: float,
    pmh_weight: float,
    pmh_cap_ratio: float,
    freeze_blocks: int,
    device: torch.device,
    use_amp: bool,
    seed: int = 42,
    warmup_epochs: int = 0,
) -> nn.Module:
    """
    Fine-tune pretrained ViT-B/16 with PMH loss for `epochs` epochs.
    Only unfreezes the last (12 - freeze_blocks) transformer blocks + norm + head.
    """
    train_loader, class_map = _build_loader(
        train_root, max_classes, max_per_class=500,
        batch_size=batch_size, train=True, seed=seed,
    )
    return pmh_finetune_with_loader(
        train_loader,
        class_map,
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        noise_sigma=noise_sigma,
        pmh_weight=pmh_weight,
        pmh_cap_ratio=pmh_cap_ratio,
        freeze_blocks=freeze_blocks,
        device=device,
        use_amp=use_amp,
        warmup_epochs=warmup_epochs,
    )


def _pmh_step(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    noise_sigma: float,
    pmh_weight: float,
    pmh_cap_ratio: float,
) -> tuple[float, float, torch.Tensor]:
    """Single PMH training step. Returns (ce_scalar, pmh_scalar, loss_tensor)."""
    # Clean features + logits
    feats_clean = model.forward_features(images)   # (B, 197, 768)
    cls_clean = feats_clean[:, 0, :]               # (B, 768)
    logits = model.head(model.fc_norm(cls_clean)) if hasattr(model, "fc_norm") else model.head(cls_clean)
    ce = F.cross_entropy(logits, targets)

    # Noisy features (pixel-space perturbation, consistent with Task 04)
    noisy_images = _add_pixel_noise(images, noise_sigma, mean, std)
    feats_noisy = model.forward_features(noisy_images)
    cls_noisy = feats_noisy[:, 0, :]

    pmh = (
        (F.normalize(cls_clean, dim=1) - F.normalize(cls_noisy, dim=1))
        .pow(2).sum(dim=1).mean()
    )

    cap = pmh_cap_ratio * ce.item()
    eff_w = min(pmh_weight, cap / (pmh.item() + 1e-8))
    loss = ce + eff_w * pmh
    return ce.item(), pmh.item(), loss


# ---------------------------------------------------------------------------
# Top-level measure function
# ---------------------------------------------------------------------------

def run_measure(
    model: nn.Module,
    val_loader: DataLoader,
    class_map: dict[int, int],
    val_data_summary: str,
    out_dir: Path,
    sigmas: list[float],
    max_classes: int,
    max_per_class: int,
    run_name: str,
    device: torch.device,
) -> dict:
    model = model.to(device).eval()
    n_classes = len(class_map)
    print(f"  Validation data: {val_data_summary}")
    print(f"  Validation loader: {len(val_loader)} batches, {n_classes} classes")

    results: dict = {
        "run": run_name,
        "model": "vit_base_patch16_224",
        "val_data": val_data_summary,
        "n_classes": n_classes,
        "max_per_class": max_per_class,
        "tdi": {},
        "intra_mean": {},
        "accuracy": {},
    }

    for sigma in sigmas:
        feats, labs = extract_features(model, val_loader, class_map, device, sigma)

        # Accuracy (only meaningful for classifier model; feature extractor has no head)
        # We report it as None for feature-extractor mode
        results["tdi"][str(sigma)] = None
        results["intra_mean"][str(sigma)] = None
        results["accuracy"][str(sigma)] = None

        intra, tdi = compute_tdi(feats, labs, num_classes=n_classes, max_per_class=max_per_class)
        results["tdi"][str(sigma)] = round(tdi, 6)
        results["intra_mean"][str(sigma)] = round(intra, 6)
        print(f"  σ={sigma:<5}  TDI={tdi:.4f}  intra_mean={intra:.4f}")

    out_path = out_dir / f"tdi_{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 3: ViT-B/16 TDI on ImageNet")
    p.add_argument("mode", choices=["measure", "pmh_tune"],
                   help="measure: pretrained TDI only. pmh_tune: fine-tune + re-measure TDI.")
    p.add_argument("--data_dir", default=None,
                   help="ImageNet root (parent of val/ or ILSVRC2012_img_val/).")
    p.add_argument("--val_dir", default=None,
                   help="Override: validation ImageFolder with class subfolders. "
                   "corrupt mode: if omitted with no --data_dir, env PMH_IMAGENET_VAL or IMAGENET_VAL.")
    p.add_argument("--train_dir", default=None,
                   help="Override: training ImageFolder (pmh_tune).")
    p.add_argument("--out_dir", default="results",
                   help="Directory for outputs.")
    # Scale controls
    p.add_argument("--max_classes", type=int, default=100,
                   help="Subsample N classes. 100 is representative; 1000 for full ImageNet TDI.")
    p.add_argument("--max_per_class", type=int, default=50,
                   help="Max validation samples per class for TDI.")
    p.add_argument("--batch_size", type=int, default=128)
    # Noise levels
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.10])
    # PMH fine-tune options
    p.add_argument("--epochs", type=int, default=24,
                   help="pmh_tune: fine-tune epochs (default 24; cosine decay after warmup).")
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="pmh_tune: linear LR warmup before cosine (0 = cosine only).",
    )
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--noise_sigma", type=float, default=0.10,
                   help="σ used during PMH fine-tuning.")
    p.add_argument("--pmh_weight", type=float, default=0.3)
    p.add_argument("--pmh_cap_ratio", type=float, default=0.30)
    p.add_argument("--freeze_blocks", type=int, default=8,
                   help="Freeze first N of 12 ViT-B/16 blocks. Train last (12-N).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--val_source",
        choices=["local", "hf"],
        default="local",
        help="local: ImageFolder on disk (needs sorted val/). "
        "hf: Hugging Face ILSVRC/imagenet-1k validation (login + accept terms; no local val/).",
    )
    p.add_argument(
        "--train_source",
        choices=["local", "hf_val", "hf_train"],
        default="local",
        help=(
            "pmh_tune only. "
            "local: ImageNet train/ on disk. "
            "hf_val: use a subset of the HF validation split for training (no 150 GB tarball). "
            "hf_train: stream HF training split (--hf_train_per_class images/class) and use "
            "HF val split (--max_per_class images/class) as the held-out test set. "
            "This gives a clean train/test separation with no data leakage."
        ),
    )
    p.add_argument(
        "--hf_train_per_class",
        type=int,
        default=30,
        help=(
            "hf_val mode: images per class used for PMH training (rest of ≤max_per_class held out). "
            "hf_train mode: images per class streamed from HF training split (default 450 → 45k/100cls)."
        ),
    )
    p.add_argument(
        "--class_mode",
        choices=["random", "most_least"],
        default="random",
        help=(
            "hf_train mode only. "
            "random: pick max_classes uniformly at random (default). "
            "most_least: pick half from the most-frequent and half from the least-frequent "
            "training classes (requires a one-time label-column scan of training parquets, "
            "result cached to results/hf_train_class_freqs.json)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_source == "local":
        if args.data_dir is None and args.val_dir is None and args.mode != "corrupt":
            sys.exit(
                "Error: for --val_source local, provide --data_dir and/or --val_dir.\n"
                "Or use --val_source hf to use Hugging Face validation (no disk layout needed)."
            )
    if args.train_source == "hf_val" and args.mode == "measure":
        sys.exit("--train_source hf_val applies only to --mode pmh_tune.")

    if args.mode == "pmh_tune" and args.train_source == "local":
        if args.data_dir is None and args.train_dir is None:
            sys.exit(
                "Error: pmh_tune with --train_source local needs --data_dir or --train_dir.\n"
                "To use only downloaded HF validation (~few GB), use:\n"
                "  --train_source hf_val --val_source hf"
            )

    if args.mode == "pmh_tune" and args.train_source == "hf_val" and args.val_source != "hf":
        sys.exit("Error: --train_source hf_val requires --val_source hf.")
    if args.mode == "pmh_tune" and args.train_source == "hf_train" and args.val_source not in ("hf", "local"):
        sys.exit("Error: --train_source hf_train is compatible with --val_source hf (default) or local.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    base_dir: Path | None = None
    if args.data_dir:
        base_dir = Path(args.data_dir).expanduser().resolve()
    elif args.train_dir:
        base_dir = Path(args.train_dir).expanduser().resolve().parent
    elif args.val_dir:
        base_dir = Path(args.val_dir).expanduser().resolve().parent
    val_root: Path | None = None
    if args.val_source == "local":
        if base_dir is None:
            sys.exit("Internal error: local val but base_dir unset (need --data_dir and/or --val_dir).")
        val_root = resolve_imagenet_val_root(base_dir, args.val_dir)

    print(f"Device: {device}  Mode: {args.mode}  Classes: {args.max_classes}")
    print(f"Validation source: {args.val_source}")
    if args.mode == "pmh_tune":
        print(f"Train source (PMH): {args.train_source}")

    use_hf_val_split   = args.mode == "pmh_tune" and args.train_source == "hf_val"
    use_hf_train_split = args.mode == "pmh_tune" and args.train_source == "hf_train"
    train_ld_hf: DataLoader | None = None

    if use_hf_train_split:
        hf_train_per = args.hf_train_per_class
        test_per = args.max_per_class  # images/class from HF val split (test set)
        train_ld_hf, val_loader, class_map, val_desc = build_hf_full_split_loaders(
            n_classes=args.max_classes,
            train_per_class=hf_train_per,
            test_per_class=test_per,
            batch_size=args.batch_size,
            seed=args.seed,
            class_mode=getattr(args, "class_mode", "random"),
        )
    elif use_hf_val_split:
        train_ld_hf, val_loader, class_map, val_desc = build_hf_pmh_train_eval_loaders(
            args.max_classes,
            args.max_per_class,
            args.hf_train_per_class,
            args.batch_size,
            args.seed,
        )
    else:
        val_loader, class_map, val_desc = build_validation_loader(
            args.val_source,
            val_root,
            args.max_classes,
            args.max_per_class,
            args.batch_size,
            args.seed,
        )

    all_results: list[dict] = []

    # -----------------------------------------------------------------------
    # Step 1: Measure pretrained baseline TDI (always)
    # -----------------------------------------------------------------------
    print("\n[Step 1] Pretrained ViT-B/16 TDI (no PMH)")
    feat_model = load_vit(pretrained=True, num_classes=0).to(device)
    r_baseline = run_measure(
        feat_model, val_loader, class_map, val_desc, out_dir, args.sigmas,
        args.max_classes, args.max_per_class,
        run_name="pretrained_baseline", device=device,
    )
    all_results.append(r_baseline)

    # -----------------------------------------------------------------------
    # Step 2: PMH fine-tune + re-measure (optional)
    # -----------------------------------------------------------------------
    if args.mode == "pmh_tune":
        use_hf_loader = use_hf_val_split or use_hf_train_split
        print("\n[Step 2] PMH fine-tuning")
        if use_hf_loader:
            assert train_ld_hf is not None
            pmh_feat_model = pmh_finetune_with_loader(
                train_ld_hf,
                class_map,
                out_dir=out_dir,
                epochs=args.epochs,
                lr=args.lr,
                noise_sigma=args.noise_sigma,
                pmh_weight=args.pmh_weight,
                pmh_cap_ratio=args.pmh_cap_ratio,
                freeze_blocks=args.freeze_blocks,
                device=device,
                use_amp=use_amp,
                warmup_epochs=args.warmup_epochs,
                train_labels_remapped=use_hf_train_split,
            )
        else:
            if base_dir is None:
                sys.exit("pmh_tune + local train: set --data_dir or --train_dir.")
            train_root = resolve_imagenet_train_root(base_dir, args.train_dir)
            print(f"  Training ImageFolder: {train_root}")
            pmh_feat_model = pmh_finetune(
                train_root=train_root,
                out_dir=out_dir,
                max_classes=args.max_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                noise_sigma=args.noise_sigma,
                pmh_weight=args.pmh_weight,
                pmh_cap_ratio=args.pmh_cap_ratio,
                freeze_blocks=args.freeze_blocks,
                device=device,
                use_amp=use_amp,
                seed=args.seed,
                warmup_epochs=args.warmup_epochs,
            )
        print("\n[Step 3] Re-measure TDI after PMH fine-tuning")
        if use_hf_loader:
            r_pmh = run_measure(
                pmh_feat_model, val_loader, class_map, val_desc, out_dir, args.sigmas,
                args.max_classes, args.max_per_class,
                run_name="pmh_finetuned", device=device,
            )
        else:
            val_loader2, class_map2, val_desc2 = build_validation_loader(
                args.val_source,
                val_root,
                args.max_classes,
                args.max_per_class,
                args.batch_size,
                args.seed,
            )
            r_pmh = run_measure(
                pmh_feat_model, val_loader2, class_map2, val_desc2, out_dir, args.sigmas,
                args.max_classes, args.max_per_class,
                run_name="pmh_finetuned", device=device,
            )
        all_results.append(r_pmh)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY — TDI vs Gaussian noise σ (pixel space)")
    print("=" * 65)
    all_sigmas = [str(s) for s in args.sigmas]
    header = f"{'Run':<25}" + "".join(f"  σ={s:<6}" for s in all_sigmas)
    print(header)
    print("-" * len(header))
    for r in all_results:
        row = f"{r['run']:<25}"
        for s in all_sigmas:
            v = r["tdi"].get(s)
            row += f"  {v:<8.4f}" if v is not None else f"  {'—':<8}"
        print(row)
    print()

    # Save combined summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
