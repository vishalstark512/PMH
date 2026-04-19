"""
Re-ID data: Market-1501 (and compatible) train / query / gallery.
Builds two views (clean, perturbed) for E1; optional occlusion and noise.
Auto-downloads Market-1501 from Google Drive if missing.
"""
import os
import random
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet norm for pretrained backbone
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Market-1501: Google Drive file id (public)
MARKET1501_GOOGLE_DRIVE_ID = "0B8-rUzbwVRk0c054eEozWG9COHM"
MARKET1501_ZIP_NAME = "Market1501.zip"

# Zip may use "bounding_box_train" / "bounding_box_test"; we accept both
TRAIN_DIR_ALIASES = ("bound_box_train", "bounding_box_train")
TEST_DIR_ALIASES = ("bound_box_test", "bounding_box_test")


def _seed_worker(worker_id, base_seed):
    """Picklable worker init for DataLoader (Windows spawn)."""
    s = int(base_seed) + int(worker_id)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _parse_market1501_name(filename):
    """Parse Market-1501 filename: 0002_c1s1_001051_00.jpg -> pid=2, cam='c1s1'."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 1:
        try:
            pid = int(parts[0])
            cam = parts[1] if len(parts) > 1 else ""
            return pid, cam
        except ValueError:
            pass
    return None, ""


def _resolve_split_dir(root, logical_name):
    """Return the actual folder name under root for train/test/query (handles bounding_box_* vs bound_box_*)."""
    root = Path(root)
    if logical_name == "query":
        return "query" if (root / "query").is_dir() else None
    if logical_name == "bound_box_train":
        for name in TRAIN_DIR_ALIASES:
            if (root / name).is_dir():
                return name
        return None
    if logical_name == "bound_box_test":
        for name in TEST_DIR_ALIASES:
            if (root / name).is_dir():
                return name
        return None
    return logical_name if (root / logical_name).is_dir() else None


def collect_market1501_split(root, split_dir):
    """
    Collect (path, pid, cam) for a split. split_dir in ('bound_box_train', 'query', 'bound_box_test').
    Accepts bounding_box_train / bounding_box_test if present. Returns list of (rel_path, pid, cam) and set of unique pids.
    """
    root = Path(root)
    actual_dir = _resolve_split_dir(root, split_dir)
    if actual_dir is None:
        return [], set()
    folder = root / actual_dir

    samples = []
    pids = set()
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        pid, cam = _parse_market1501_name(p.name)
        if pid is None:
            continue
        rel = os.path.relpath(p, root)
        samples.append((rel, pid, cam))
        pids.add(pid)

    return samples, pids


def _has_market1501_data(root):
    """True if root contains train, query, and test splits (under any accepted folder name)."""
    root = Path(root)
    train_ok = any((root / d).is_dir() for d in TRAIN_DIR_ALIASES)
    test_ok = any((root / d).is_dir() for d in TEST_DIR_ALIASES)
    query_ok = (root / "query").is_dir()
    return train_ok and test_ok and query_ok


def _download_market1501(data_dir):
    """Download Market1501.zip from Google Drive and extract. Returns path to extracted root."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / MARKET1501_ZIP_NAME

    if zip_path.exists():
        print(f"  Using existing {zip_path}", flush=True)
    else:
        try:
            import gdown
        except ImportError:
            raise RuntimeError(
                "Market-1501 not found and gdown is required to auto-download. "
                "Install with: pip install gdown\n"
                "Then run again, or download manually from "
                "https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view "
                f"and place the extracted folder in {data_dir.absolute()}"
            ) from None
        print(f"  Downloading Market-1501 from Google Drive to {zip_path} ...", flush=True)
        gdown.download(
            id=MARKET1501_GOOGLE_DRIVE_ID,
            output=str(zip_path),
            quiet=False,
        )
        if not zip_path.exists():
            raise RuntimeError("Download failed; file not created.")

    # Extract: zip may contain "Market1501" or "Market-1501-v15.09.15" or files at root
    print(f"  Extracting {zip_path} ...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Find root of the dataset (folder that will contain query, bounding_box_*, etc.)
        top_dirs = {n.split("/")[0] if "/" in n else n.split("\\")[0] for n in names}
        extract_root = data_dir
        for candidate in ("Market1501", "Market-1501-v15.09.15", "market1501"):
            if candidate in top_dirs:
                extract_root = data_dir / candidate
                break
        zf.extractall(data_dir)

    # If we extracted into a subdir, that's our root
    if extract_root != data_dir and extract_root.is_dir():
        return str(extract_root)
    # Else content might be directly in data_dir
    if _has_market1501_data(data_dir):
        return str(data_dir)
    # Or inside a single subdir
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    for d in subdirs:
        if _has_market1501_data(d):
            return str(d)
    raise RuntimeError(
        f"Extracted {zip_path} but could not find query/ and train/test folders under {data_dir.absolute()}"
    )


def ensure_market1501(data_dir):
    """
    If Market-1501 data is present under data_dir, return its root path.
    Otherwise download and extract from Google Drive, then return the root path.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Already have data?
    market = data_dir / "Market1501"
    if market.is_dir() and _has_market1501_data(market):
        return str(market)
    if _has_market1501_data(data_dir):
        return str(data_dir)
    for sub in data_dir.iterdir():
        if sub.is_dir() and _has_market1501_data(sub):
            return str(sub)

    # Auto-download
    print("Market-1501 not found. Auto-downloading ...", flush=True)
    root = _download_market1501(data_dir)
    print(f"Market-1501 ready at {root}", flush=True)
    return root


def find_market1501_root(data_dir):
    """If data_dir/Market1501 exists (or has bound_box_train), return that; else return data_dir."""
    data_dir = Path(data_dir)
    market = data_dir / "Market1501"
    if market.is_dir() and _has_market1501_data(market):
        return str(market)
    if _has_market1501_data(data_dir):
        return str(data_dir)
    for sub in data_dir.iterdir():
        if sub.is_dir() and _has_market1501_data(sub):
            return str(sub)
    return str(data_dir)


class Market1501Train(Dataset):
    """
    Training set: (image, pid). pid is 0..num_pids-1 (remapped).
    For E1 two_view=True:
      - use_identity_pairs=True (recommended): return (img1, img2, pid) with two *different* images of same person.
      - use_identity_pairs=False: return (img_clean, img_perturbed, pid) from same image with noise/occlusion.
    """

    def __init__(
        self,
        root,
        transform_clean=None,
        transform_perturb=None,
        two_view=False,
        use_identity_pairs=True,
        same_image_frac=0.0,
        noise_sigma=0.1,
        occlusion_ratio=0.0,
        img_size=(256, 128),
        load_in_memory=False,
    ):
        self.root = Path(root)
        self.transform_clean = transform_clean
        self.transform_perturb = transform_perturb
        self.two_view = two_view
        self.use_identity_pairs = use_identity_pairs and two_view
        self.same_image_frac = max(0.0, min(1.0, same_image_frac))
        self.noise_sigma = noise_sigma
        self.occlusion_ratio = occlusion_ratio
        self.img_size = img_size

        samples, pids = collect_market1501_split(root, "bound_box_train")
        self.samples = samples
        self.pids_sorted = sorted(pids)
        self.pid2idx = {p: i for i, p in enumerate(self.pids_sorted)}
        self.num_classes = len(self.pids_sorted)
        # pid_raw -> list of indices into self.samples (for identity-level pairs)
        self.pid_to_indices = {}
        for i, (_, pid_raw, _) in enumerate(self.samples):
            self.pid_to_indices.setdefault(pid_raw, []).append(i)
        if load_in_memory:
            # Preload all train images into RAM (Market-1501 ~32k images fits in 128GB)
            self._images = [Image.open(self.root / rel_path).convert("RGB") for rel_path, _, _ in self.samples]
        else:
            self._images = None
        if not self.samples:
            raise FileNotFoundError(
                f"No training images in {root}/bound_box_train.\n"
                "Expect Market-1501 layout:\n"
                "  bound_box_train/ (or bounding_box_train/), query/, bound_box_test/\n"
                "Run with --data_dir <dir> to auto-download (requires gdown), or download from:\n"
                "  https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view"
            )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, rel_path):
        path = self.root / rel_path
        img = Image.open(path).convert("RGB")
        return img

    def _get_image(self, idx):
        """Return PIL image for sample index (from RAM if load_in_memory, else from disk)."""
        if self._images is not None:
            return self._images[idx]
        return self._load_image(self.samples[idx][0])

    def _perturb_image(self, img_tensor, noise_sigma=0.0, occlusion_ratio=0.0):
        """Apply Gaussian noise (in pixel space: denorm → noise → clamp → renorm) and/or occlusion."""
        x = img_tensor.clone()
        if noise_sigma > 0:
            mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(3, 1, 1)
            if x.dim() == 4:
                mean, std = mean.unsqueeze(0), std.unsqueeze(0)
            x = x * std + mean
            x = x + noise_sigma * torch.randn_like(x, device=x.device)
            x = x.clamp(0.0, 1.0)
            x = (x - mean) / std
        if occlusion_ratio > 0 and x.dim() >= 3:
            # Random rectangular occlusion
            c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
            oh, ow = int(h * occlusion_ratio), int(w * occlusion_ratio)
            if oh > 0 and ow > 0:
                top = random.randint(0, max(0, h - oh))
                left = random.randint(0, max(0, w - ow))
                if x.dim() == 3:
                    x[..., top : top + oh, left : left + ow] = 0.0
                else:
                    x[..., top : top + oh, left : left + ow] = 0.0
        return x

    def __getitem__(self, idx):
        rel_path, pid_raw, _ = self.samples[idx]
        pid = self.pid2idx[pid_raw]

        if self.two_view and self.use_identity_pairs:
            # With probability same_image_frac, use same-image two-views (aligns with robustness eval: gaussian/occlusion).
            # Otherwise identity-level: two different images of same person.
            use_same_image = self.same_image_frac > 0 and random.random() < self.same_image_frac
            if use_same_image:
                img = self._get_image(idx)
                img_clean = self.transform_clean(img) if self.transform_clean else img
                img_perturb = self.transform_perturb(img) if self.transform_perturb else img
                if self.noise_sigma > 0 or self.occlusion_ratio > 0:
                    img_perturb = self._perturb_image(
                        img_perturb,
                        noise_sigma=self.noise_sigma,
                        occlusion_ratio=self.occlusion_ratio,
                    )
                return img_clean, img_perturb, pid
            # Identity-level PMH: two different images of same person; second view gets Gaussian noise + occlusion (like other PMH)
            indices_same_pid = self.pid_to_indices[pid_raw]
            if len(indices_same_pid) >= 2:
                idx2 = random.choice([i for i in indices_same_pid if i != idx])
                img1 = self._get_image(idx)
                img2 = self._get_image(idx2)
                t = self.transform_clean if self.transform_clean else (lambda x: x)
                view1 = t(img1)
                view2 = t(img2)
                if self.noise_sigma > 0 or self.occlusion_ratio > 0:
                    view2 = self._perturb_image(
                        view2,
                        noise_sigma=self.noise_sigma,
                        occlusion_ratio=self.occlusion_ratio,
                    )
                return view1, view2, pid
            # Fallback: only one image for this identity → same-image two-views
            img = self._get_image(idx)
            img_clean = self.transform_clean(img) if self.transform_clean else img
            img_perturb = self.transform_perturb(img) if self.transform_perturb else img
            if self.noise_sigma > 0 or self.occlusion_ratio > 0:
                img_perturb = self._perturb_image(
                    img_perturb,
                    noise_sigma=self.noise_sigma,
                    occlusion_ratio=self.occlusion_ratio,
                )
            return img_clean, img_perturb, pid
        elif self.two_view:
            # Same-image two-views (noise/occlusion on one view)
            img = self._get_image(idx)
            img_clean = self.transform_clean(img) if self.transform_clean else img
            img_perturb = self.transform_perturb(img) if self.transform_perturb else img
            if self.noise_sigma > 0 or self.occlusion_ratio > 0:
                img_perturb = self._perturb_image(
                    img_perturb,
                    noise_sigma=self.noise_sigma,
                    occlusion_ratio=self.occlusion_ratio,
                )
            return img_clean, img_perturb, pid
        else:
            img = self._get_image(idx)
            if self.transform_clean:
                img = self.transform_clean(img)
            return img, pid


class Market1501Eval(Dataset):
    """Query or gallery: (image, pid, cam). Used for eval; no two-view."""

    def __init__(self, root, split_dir, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples, _ = collect_market1501_split(root, split_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, pid, cam = self.samples[idx]
        path = self.root / rel_path
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, pid, cam, rel_path


def get_train_transforms(run, img_size=(256, 128), mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """B0: minimal. B1/E1: resize + flip + color jitter. E1 perturb view can use stronger aug."""
    base = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if run == "B0":
        return transforms.Compose(base), transforms.Compose(base)

    train_tf = [
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    # Perturbed view for E1: same but we may add noise/occlusion in tensor space
    perturb_tf = [
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(train_tf), transforms.Compose(perturb_tf)


def get_eval_transform(img_size=(256, 128), mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class DummyReIDDataset(Dataset):
    """Synthetic (image, pid) or (img_clean, img_perturb, pid) for testing without Market-1501."""

    def __init__(self, num_samples=500, num_classes=32, img_size=(256, 128), two_view=False):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.two_view = two_view
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        self.norm = lambda x: (x - mean) / std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        h, w = self.img_size
        img = torch.rand(3, h, w)
        img = self.norm(img)
        pid = idx % self.num_classes
        if self.two_view:
            other = img + 0.1 * torch.randn_like(img)
            return img, other, pid
        return img, pid


def get_train_loader(
    root,
    run,
    batch_size=32,
    num_workers=4,
    img_size=(256, 128),
    two_view=False,
    use_identity_pairs=True,
    same_image_frac=0.0,
    noise_sigma=0.1,
    occlusion_ratio=0.0,
    dummy=False,
    load_in_memory=False,
    seed=None,
):
    if dummy:
        dataset = DummyReIDDataset(
            num_samples=500, num_classes=32, img_size=tuple(img_size), two_view=two_view
        )
        dl_kw = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        if seed is not None:
            dl_kw["generator"] = torch.Generator().manual_seed(int(seed))
        loader = torch.utils.data.DataLoader(**dl_kw)
        return loader, dataset.num_classes

    root = ensure_market1501(root)
    tf_clean, tf_perturb = get_train_transforms(run, img_size=img_size)
    dataset = Market1501Train(
        root,
        transform_clean=tf_clean,
        transform_perturb=tf_perturb if two_view else None,
        two_view=two_view,
        use_identity_pairs=use_identity_pairs if run == "E1" else False,
        same_image_frac=same_image_frac if run == "E1" else 0.0,
        noise_sigma=noise_sigma if run == "E1" else 0.0,
        occlusion_ratio=occlusion_ratio if run == "E1" else 0.0,
        img_size=img_size,
        load_in_memory=load_in_memory,
    )

    # persistent_workers avoids respawning workers each epoch (faster with num_workers > 0)
    persistent = num_workers > 0
    dl_kw = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
    )
    if seed is not None:
        dl_kw["generator"] = torch.Generator().manual_seed(int(seed))
    if num_workers > 0 and seed is not None:
        dl_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed))
    loader = torch.utils.data.DataLoader(**dl_kw)
    return loader, dataset.num_classes


def get_eval_loaders(root, batch_size=64, num_workers=4, img_size=(256, 128), seed=None):
    """Returns (query_loader, gallery_loader, query_dataset, gallery_dataset)."""
    root = ensure_market1501(root)
    tf = get_eval_transform(img_size=img_size)
    query_ds = Market1501Eval(root, "query", transform=tf)
    gallery_ds = Market1501Eval(root, "bound_box_test", transform=tf)
    eval_kw = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if num_workers > 0:
        eval_kw["persistent_workers"] = True
    if num_workers > 0 and seed is not None:
        eval_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(seed))
    query_loader = torch.utils.data.DataLoader(query_ds, **eval_kw)
    gallery_loader = torch.utils.data.DataLoader(gallery_ds, **eval_kw)
    return query_loader, gallery_loader, query_ds, gallery_ds
