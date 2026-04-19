"""
Chest X-ray (NIH ChestX-ray14) data loaders for classification + PMH.
Multi-label; two-view (clean / perturbed) for E1 with noise and intensity.
Auto-downloads from Kaggle if data not found (requires kaggle API + credentials).
"""
import os
import random
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Kaggle dataset slugs
KAGGLE_NIH_DATASET = "nih-chest-xrays/data"
KAGGLE_PNEUMONIA_DATASET = "paultimothymooney/chest-xray-pneumonia"  # ~1–2 GB, 2 classes (NORMAL / PNEUMONIA)

# ImageNet norm for pretrained backbone
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# NIH ChestX-ray14: 14 pathology labels (order matters for multi-hot)
NIH14_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def _seed_worker(worker_id, base_seed):
    """Picklable worker init for DataLoader (Windows spawn)."""
    s = int(base_seed) + int(worker_id)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _parse_nih_labels(finding_str):
    """Parse 'Finding Labels' string (pipe-separated) into 14-dim multi-hot."""
    if pd.isna(finding_str) or not str(finding_str).strip():
        return np.zeros(len(NIH14_LABELS), dtype=np.float32)
    parts = [p.strip() for p in str(finding_str).split("|")]
    out = np.zeros(len(NIH14_LABELS), dtype=np.float32)
    for p in parts:
        if p in NIH14_LABELS:
            out[NIH14_LABELS.index(p)] = 1.0
    return out


def find_nih_root(data_dir):
    """Locate NIH data: data_dir or data_dir/NIH_ChestXray14 or data_dir/chestxray, or any subdir with CSV."""
    data_dir = Path(data_dir)
    for candidate in [data_dir, data_dir / "NIH_ChestXray14", data_dir / "chestxray", data_dir / "ChestX-ray14"]:
        if candidate.is_dir():
            csv_candidates = list(candidate.glob("**/Data_Entry_2017.csv")) + list(candidate.glob("**/*Entry*.csv"))
            if csv_candidates:
                return str(csv_candidates[0].parent)
            if (candidate / "Data_Entry_2017.csv").exists():
                return str(candidate)
    # Search any subdir for CSV (e.g. after Kaggle extract)
    for sub in data_dir.rglob("Data_Entry*.csv"):
        return str(sub.parent)
    return str(data_dir)


def _has_nih_data(data_dir):
    """True if we find a CSV and can load at least one sample."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return False
    root = find_nih_root(data_dir)
    try:
        samples, _ = load_nih_csv(root, max_samples=1, seed=42)
        return len(samples) > 0
    except Exception:
        return False


def _download_nih_via_kaggle(data_dir):
    """Download NIH Chest X-rays from Kaggle and extract; unzip images_*.zip into images/."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            "NIH data not found and Kaggle API is required to auto-download.\n\n"
            "1. Install:  pip install kaggle\n"
            "2. Get credentials: https://www.kaggle.com/settings → Create New API Token (downloads kaggle.json)\n"
            "3. Place kaggle.json in:\n"
            "   - Windows:  %USERPROFILE%\\.kaggle\\kaggle.json\n"
            "   - Linux/Mac: ~/.kaggle/kaggle.json\n"
            "4. Run this script again.\n\n"
            "Alternatively, download the dataset manually from Kaggle or NIH Box and set --data_dir to the folder containing Data_Entry_2017.csv and an 'images' folder."
        ) from e
    print("  Downloading NIH Chest X-rays from Kaggle (this may take a while)...", flush=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_NIH_DATASET, path=str(data_dir), unzip=True)
    # Find where CSV landed (Kaggle often creates a subfolder)
    csv_path = None
    for p in data_dir.rglob("Data_Entry*.csv"):
        csv_path = p
        break
    if csv_path is None:
        raise FileNotFoundError(f"Download completed but Data_Entry_2017.csv not found under {data_dir}")
    extract_root = csv_path.parent
    # Unzip images_*.zip into extract_root/images/
    images_dir = extract_root / "images"
    images_dir.mkdir(exist_ok=True)
    for z in extract_root.glob("images_*.zip"):
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(images_dir)
        print(f"  Extracted {z.name}", flush=True)
    print(f"  NIH data ready at {extract_root}", flush=True)
    return str(extract_root)


def ensure_nih_chestxray(data_dir, auto_download=True):
    """If NIH data is present under data_dir, return its root. Otherwise download from Kaggle (if auto_download) or raise."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if _has_nih_data(data_dir):
        return find_nih_root(data_dir)
    if not auto_download:
        raise FileNotFoundError(
            f"NIH Chest X-ray data not found under {data_dir.absolute()}.\n"
            "Download manually from https://www.kaggle.com/datasets/nih-chest-xrays/data "
            "or https://nihcc.app.box.com/v/ChestXray-NIHCC, then set --data_dir to the folder "
            "containing Data_Entry_2017.csv and an 'images' folder with the X-rays.\n"
            "Or install Kaggle API (pip install kaggle), add credentials, and run without --no_download."
        )
    print("NIH Chest X-ray data not found. Auto-downloading...", flush=True)
    _download_nih_via_kaggle(data_dir)
    return find_nih_root(data_dir)


# ---------------------------------------------------------------------------
# Pneumonia dataset (small: ~1–2 GB, 2 classes) — default for this task
# ---------------------------------------------------------------------------

PNEUMONIA_CLASSES = ["NORMAL", "PNEUMONIA"]  # 0 = NORMAL, 1 = PNEUMONIA


def find_pneumonia_root(data_dir):
    """Locate Pneumonia data: folder with chest_xray/train/NORMAL and train/PNEUMONIA (or train/val/test)."""
    data_dir = Path(data_dir)
    for candidate in [data_dir, data_dir / "chest_xray", data_dir / "chest-xray-pneumonia"]:
        if not candidate.is_dir():
            continue
        train_dir = candidate / "train" if (candidate / "train").is_dir() else candidate
        if (train_dir / "NORMAL").is_dir() and (train_dir / "PNEUMONIA").is_dir():
            return str(candidate)
    for sub in data_dir.rglob("NORMAL"):
        if sub.is_dir() and (sub.parent / "PNEUMONIA").is_dir():
            return str(sub.parent.parent)  # train's parent
    return str(data_dir)


def _collect_pneumonia_samples(root, split):
    """split in ('train','val','test'). Returns list of (path, label_vec) with label_vec shape (2,)."""
    root = Path(root)
    base = root / "chest_xray" if (root / "chest_xray").is_dir() else root
    split_dir = base / split if (base / split).is_dir() else base
    samples = []
    for class_idx, name in enumerate(PNEUMONIA_CLASSES):
        class_dir = split_dir / name
        if not class_dir.is_dir():
            continue
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            for p in class_dir.glob(ext):
                label = np.zeros(2, dtype=np.float32)
                label[class_idx] = 1.0
                samples.append((str(p), label))
    return samples


def load_pneumonia_splits(root, max_train=None, seed=42):
    """Returns (train_s, val_s, test_s), num_classes=2. Uses val if present else 10% of train."""
    root = Path(root)
    train_s = _collect_pneumonia_samples(root, "train")
    val_s = _collect_pneumonia_samples(root, "val")
    test_s = _collect_pneumonia_samples(root, "test")
    if not train_s:
        return [], [], [], 2
    if max_train and len(train_s) > max_train:
        rng = np.random.default_rng(seed)
        train_s = [train_s[i] for i in rng.choice(len(train_s), size=max_train, replace=False)]
    if not val_s and train_s:
        rng = np.random.default_rng(seed)
        n_val = max(1, int(0.1 * len(train_s)))
        idx = rng.choice(len(train_s), size=n_val, replace=False)
        val_s = [train_s[i] for i in idx]
        train_s = [train_s[i] for i in range(len(train_s)) if i not in idx]
    if not test_s:
        test_s = val_s
    return train_s, val_s, test_s, 2


def _has_pneumonia_data(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return False
    root = find_pneumonia_root(data_dir)
    train_s, _, _, _ = load_pneumonia_splits(root, max_train=1)
    return len(train_s) > 0


def _download_pneumonia_via_kagglehub():
    """Download Chest X-Ray Pneumonia via kagglehub (~1–2 GB). Returns path to dataset root."""
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError(
            "Pneumonia data not found. Auto-download needs kagglehub, which is not installed.\n\n"
            "  pip install kagglehub\n\n"
            "Then run training again. Or use --no_download and place the dataset manually under data_dir."
        ) from e
    print("Downloading Chest X-Ray Pneumonia via kagglehub (~1–2 GB)...", flush=True)
    path = kagglehub.dataset_download(KAGGLE_PNEUMONIA_DATASET)
    root = find_pneumonia_root(Path(path))
    if not _has_pneumonia_data(Path(path)):
        raise FileNotFoundError(f"Download completed but Pneumonia train/NORMAL,PNEUMONIA not found under {path}")
    print(f"Pneumonia data ready at {root}", flush=True)
    return root


def ensure_pneumonia(data_dir, auto_download=True):
    """If Pneumonia data is present under data_dir, return root. Else download via kagglehub (if auto_download) or raise."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if _has_pneumonia_data(data_dir):
        return find_pneumonia_root(data_dir)
    if not auto_download:
        raise FileNotFoundError(
            f"Pneumonia chest X-ray data not found under {data_dir.absolute()}.\n"
            "Download from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia "
            "or run without --no_download (requires pip install kagglehub)."
        )
    print("Pneumonia chest X-ray data not found. Auto-downloading...", flush=True)
    return _download_pneumonia_via_kagglehub()


def load_nih_csv(root, max_samples=None, seed=42):
    """
    Load NIH ChestX-ray14 style CSV. Expects 'Image Index' and 'Finding Labels' (or similar).
    Returns list of (image_path, labels_vec) and num_classes=14.
    """
    root = Path(root)
    csv_path = root / "Data_Entry_2017.csv"
    if not csv_path.exists():
        csv_candidates = list(root.glob("**/*Entry*.csv")) + list(root.glob("**/*.csv"))
        csv_path = csv_candidates[0] if csv_candidates else None
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            f"NIH CSV not found under {root}. Expect Data_Entry_2017.csv with columns 'Image Index', 'Finding Labels'. "
            "Download from https://nihcc.app.box.com/v/ChestXray-NIHCC or Kaggle NIH Chest X-rays."
        )
    df = pd.read_csv(csv_path)
    # Accept various column names
    img_col = None
    for c in ["Image Index", "Image_Index", "ImageIndex", "image"]:
        if c in df.columns:
            img_col = c
            break
    label_col = None
    for c in ["Finding Labels", "Finding_Labels", "FindingLabels", "labels"]:
        if c in df.columns:
            label_col = c
            break
    if img_col is None or label_col is None:
        raise ValueError(f"CSV must have image and label columns. Found: {list(df.columns)}")
    samples = []
    # Image folder: images/ or ./
    img_dir = root / "images"
    if not img_dir.is_dir():
        img_dir = root
    for _, row in df.iterrows():
        fname = row[img_col]
        if pd.isna(fname):
            continue
        fname = str(fname).strip()
        path = None
        for p in [img_dir / fname, img_dir / (Path(fname).stem + ".png"), img_dir / (Path(fname).stem + ".jpg")]:
            if p.exists():
                path = p
                break
        if path is None:
            # try inside images/ if we were at root
            if img_dir == root and (root / "images").is_dir():
                for p in [(root / "images") / fname, (root / "images") / (Path(fname).stem + ".png")]:
                    if p.exists():
                        path = p
                        break
        if path is None:
            continue
        labels = _parse_nih_labels(row[label_col])
        samples.append((str(path), labels))
    if max_samples is not None and len(samples) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in idx]
    return samples, 14


def train_val_test_split(samples, val_ratio=0.1, test_ratio=0.2, seed=42):
    """Random split by image (not by patient) for reproducibility."""
    rng = np.random.default_rng(seed)
    n = len(samples)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
    train_s = [samples[i] for i in idx[:n_train]]
    val_s = [samples[i] for i in idx[n_train : n_train + n_val]]
    test_s = [samples[i] for i in idx[n_train + n_val :]]
    return train_s, val_s, test_s


class ChestXrayDataset(Dataset):
    """Single-view: (image, labels). Optional two-view for E1: (img_clean, img_perturb, labels)."""

    def __init__(
        self,
        samples,
        transform=None,
        transform_perturb=None,
        two_view=False,
        noise_sigma=0.0,
        intensity_scale_range=(0.9, 1.1),
        img_size=(224, 224),
        load_in_memory=False,
    ):
        self.samples = samples
        self.transform = transform
        self.transform_perturb = transform_perturb
        self.two_view = two_view
        self.noise_sigma = noise_sigma
        self.intensity_scale_range = intensity_scale_range
        self.img_size = img_size
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        if load_in_memory:
            self._images = [Image.open(path).convert("RGB") for path, _ in samples]
        else:
            self._images = None

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")  # grayscale -> repeat to 3 ch
        return img

    def _get_image(self, idx):
        """Return PIL image for sample index (from RAM if load_in_memory, else from disk)."""
        if self._images is not None:
            return self._images[idx]
        path, _ = self.samples[idx]
        return self._load_image(path)

    def _perturb(self, x):
        """Pixel-space: denorm -> noise + intensity -> clamp -> renorm."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        mean = self.mean.to(x.device).to(x.dtype)
        std = self.std.to(x.device).to(x.dtype)
        x = x * std + mean
        if self.noise_sigma > 0:
            x = x + self.noise_sigma * torch.randn_like(x, device=x.device)
        if self.intensity_scale_range:
            lo, hi = self.intensity_scale_range
            scale = lo + (hi - lo) * torch.rand(1, device=x.device).item()
            x = x * scale
        x = x.clamp(0.0, 1.0)
        x = (x - mean) / std
        return x.squeeze(0) if x.dim() == 4 and x.size(0) == 1 else x

    def __getitem__(self, idx):
        path, labels = self.samples[idx]
        labels = torch.from_numpy(labels).float()
        img = self._get_image(idx)
        if self.two_view and self.transform_perturb is not None and (self.noise_sigma > 0 or self.intensity_scale_range):
            img_clean = self.transform(img) if self.transform else img
            img_perturb = self.transform_perturb(img) if self.transform_perturb else self.transform(img)
            img_perturb = self._perturb(img_perturb)
            return img_clean, img_perturb, labels
        if self.transform:
            img = self.transform(img)
        return img, labels


def get_transforms(img_size=(224, 224), train=True, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    base = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if train:
        base = [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    return transforms.Compose(base)


def get_train_loader(
    data_dir,
    run,
    batch_size=32,
    num_workers=4,
    img_size=(224, 224),
    two_view=False,
    noise_sigma=0.08,
    intensity_scale_range=(0.85, 1.15),
    max_samples=None,
    val_ratio=0.1,
    test_ratio=0.2,
    seed=42,
    auto_download=True,
    dataset="pneumonia",
    load_in_memory=False,
):
    """dataset: 'pneumonia' (default, ~1–2 GB, 2 classes) or 'nih' (full NIH, ~45 GB, 14 classes).
    load_in_memory: preload all train images into RAM (fits in 128GB; no disk I/O during training)."""
    if dataset == "pneumonia":
        root = ensure_pneumonia(data_dir, auto_download=auto_download)
        train_s, val_s, test_s, num_classes = load_pneumonia_splits(root, max_train=max_samples, seed=seed)
        if not train_s:
            raise FileNotFoundError(f"No Pneumonia samples under {root}. Check train/NORMAL and train/PNEUMONIA.")
    else:
        root = ensure_nih_chestxray(data_dir, auto_download=auto_download)
        samples, num_classes = load_nih_csv(root, max_samples=max_samples, seed=seed)
        if not samples:
            raise FileNotFoundError(f"No samples under {root}. Check CSV and images.")
        train_s, val_s, test_s = train_val_test_split(samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    tf = get_transforms(img_size=img_size, train=True)
    tf_perturb = get_transforms(img_size=img_size, train=True)
    dataset_ds = ChestXrayDataset(
        train_s,
        transform=tf,
        transform_perturb=tf_perturb if two_view else None,
        two_view=two_view,
        noise_sigma=noise_sigma if run in ("E1", "E1_embed_only", "E1_no_pmh") else 0.0,
        intensity_scale_range=intensity_scale_range if run in ("E1", "E1_embed_only", "E1_no_pmh") else None,
        img_size=img_size,
        load_in_memory=load_in_memory,
    )
    dl_kw = dict(
        dataset=dataset_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if seed is not None:
        dl_kw["generator"] = torch.Generator().manual_seed(int(seed))
    if num_workers > 0 and seed is not None:
        dl_kw["persistent_workers"] = True
        dl_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed))
    loader = torch.utils.data.DataLoader(**dl_kw)
    return loader, num_classes, val_s, test_s


def get_eval_loaders(data_dir, batch_size=64, num_workers=4, img_size=(224, 224), max_test_samples=None, seed=42, auto_download=True, dataset="pneumonia"):
    """dataset: 'pneumonia' (default) or 'nih'."""
    if dataset == "pneumonia":
        root = ensure_pneumonia(data_dir, auto_download=auto_download)
        _, val_s, test_s, num_classes = load_pneumonia_splits(root, max_train=None, seed=seed)
        if not test_s and val_s:
            test_s = val_s
    else:
        root = ensure_nih_chestxray(data_dir, auto_download=auto_download)
        samples, num_classes = load_nih_csv(root, max_samples=None, seed=seed)
        if not samples:
            raise FileNotFoundError(f"No samples under {root}.")
        _, val_s, test_s = train_val_test_split(samples, seed=seed)
    if max_test_samples and len(test_s) > max_test_samples:
        rng = np.random.default_rng(seed + 1)
        test_s = [test_s[i] for i in rng.choice(len(test_s), size=max_test_samples, replace=False)]
    tf = get_transforms(img_size=img_size, train=False)
    val_ds = ChestXrayDataset(val_s or test_s, transform=tf)
    test_ds = ChestXrayDataset(test_s, transform=tf)
    eval_kw = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if num_workers > 0:
        eval_kw["persistent_workers"] = True
    if num_workers > 0 and seed is not None:
        eval_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed))
    val_loader = torch.utils.data.DataLoader(val_ds, **eval_kw)
    test_loader = torch.utils.data.DataLoader(test_ds, **eval_kw)
    return val_loader, test_loader, val_ds, test_ds, num_classes
