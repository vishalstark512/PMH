"""
Pose dataset and occlusion augmentation.
- COCO 2D pose: auto-download val2017 + keypoints (--dataset coco).
- Dummy / 3D: use --dummy or Human3.6M when available.
"""
import json
import random
import sys
from functools import partial
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

NUM_JOINTS = 17


def _seed_worker(worker_id, base_seed):
    s = int(base_seed) + int(worker_id)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ImageNet mean/std for pretrained backbone
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# COCO val2017: small, auto-downloadable
COCO_ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGES_URL = "https://images.cocodataset.org/zips/val2017.zip"


def _download_file(url, dest, desc="Downloading"):
    """Download url to dest; show progress. Retries COCO host with HTTP if HTTPS cert fails."""
    candidates = [url]
    if "images.cocodataset.org" in url and url.startswith("https://"):
        # Some environments have SSL interception / hostname mismatch on cocodataset.
        # Retry plain HTTP as a practical fallback.
        candidates.append("http://" + url[len("https://"):])

    last_err = None
    for i, candidate in enumerate(candidates, start=1):
        try:
            req = urllib.request.Request(candidate, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                chunk_size = 1 << 20
                downloaded = 0
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total and total > 0:
                            pct = 100 * downloaded / total
                            sys.stdout.write(
                                f"\r  {desc} [{i}/{len(candidates)}]: "
                                f"{downloaded/1e6:.1f} / {total/1e6:.1f} MB ({pct:.0f}%)"
                            )
                            sys.stdout.flush()
                if total:
                    print()
            return
        except Exception as e:
            last_err = e
            if dest.exists():
                dest.unlink(missing_ok=True)
            print(f"  Download attempt failed ({candidate}): {e}", flush=True)

    raise RuntimeError(
        f"Download failed after {len(candidates)} attempt(s): {last_err}. "
        "If you have COCO data already, put annotations in "
        "data_dir/annotations/person_keypoints_val2017.json and images in data_dir/val2017/ "
        "then run again (no download will be attempted)."
    ) from last_err


def _find_coco_split_paths(data_dir, split="val"):
    """Return (images_dir, ann_path) for COCO train2017 or val2017 if found; else (None, None). split in ('train','val')."""
    data_dir = Path(data_dir).resolve()
    if not data_dir.exists():
        return None, None
    ann_name = f"person_keypoints_{split}2017.json"
    dir_name = f"{split}2017"
    ann_candidates = [
        data_dir / "annotations" / ann_name,
        data_dir / ann_name,
        data_dir / "annotations_trainval2017" / ann_name,
        data_dir / "coco" / "annotations" / ann_name,
    ]
    img_candidates = [
        data_dir / dir_name,
        data_dir / "images" / dir_name,
        data_dir / "coco" / dir_name,
    ]
    for ann_path in ann_candidates:
        if not ann_path.exists():
            continue
        for img_dir in img_candidates:
            if img_dir.exists() and img_dir.is_dir() and any(img_dir.iterdir()):
                return str(img_dir), str(ann_path)
    for ann_path in ann_candidates:
        if not ann_path.exists():
            continue
        try:
            with open(ann_path, "r") as f:
                data = json.load(f)
            if not data.get("images"):
                continue
            first_file = data["images"][0].get("file_name", "000000000139.jpg")
            for img_dir in [data_dir / dir_name, data_dir / "coco" / dir_name, data_dir]:
                if img_dir.exists() and (img_dir / first_file).exists():
                    return str(img_dir), str(ann_path)
        except (json.JSONDecodeError, KeyError):
            pass
    return None, None


def _find_coco_pose_paths(data_dir):
    """Return (images_dir, ann_path) if COCO val2017 + keypoints found; else (None, None)."""
    return _find_coco_split_paths(data_dir, split="val")


def ensure_coco_pose(data_dir):
    """Use existing COCO val2017 + keypoints if found; else download. Returns (images_dir, ann_path) for VAL."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    images_dir, ann_path = _find_coco_pose_paths(data_dir)
    if images_dir and ann_path:
        print(f"Using existing COCO val: images={images_dir}, annotations={ann_path}", flush=True)
        return images_dir, ann_path

    # Download only if not found
    ann_zip = data_dir / "annotations_trainval2017.zip"
    val_zip = data_dir / "val2017.zip"
    ann_dir = data_dir / "annotations"
    val_dir = data_dir / "val2017"
    ann_file = ann_dir / "person_keypoints_val2017.json"

    if not ann_dir.exists() or not ann_file.exists():
        if not ann_zip.exists():
            print("Downloading COCO annotations (person keypoints)...", flush=True)
            _download_file(COCO_ANNOTATIONS_URL, ann_zip, "annotations")
        print("Extracting annotations...", flush=True)
        with zipfile.ZipFile(ann_zip, "r") as z:
            z.extractall(data_dir)
        ann_zip.unlink(missing_ok=True)

    if not val_dir.exists() or not any(val_dir.iterdir()):
        if not val_zip.exists():
            print("Downloading COCO val2017 images (~1 GB)...", flush=True)
            _download_file(COCO_VAL_IMAGES_URL, val_zip, "val2017")
        print("Extracting val2017...", flush=True)
        with zipfile.ZipFile(val_zip, "r") as z:
            z.extractall(data_dir)
        val_zip.unlink(missing_ok=True)

    return str(val_dir), str(ann_file)


def ensure_coco_train(data_dir):
    """Return (images_dir, ann_path) for COCO train2017 if present; else None. No download (train not in standard zip)."""
    data_dir = Path(data_dir)
    images_dir, ann_path = _find_coco_split_paths(data_dir, split="train")
    return (images_dir, ann_path) if (images_dir and ann_path) else None


class CocoPose2DDataset(torch.utils.data.Dataset):
    """
    COCO person keypoints (17 keypoints). Returns (image, keypoints).
    use_bbox_crop=True: crop around person bbox then resize (person fills frame) -> much better PCK.
    use_bbox_crop=False: full image resized to image_size (person often tiny) -> weak performance.
    """
    def __init__(self, images_dir, ann_path, image_size=256, max_samples=None, subset_seed=42, imagenet_norm=False, cache_images=False, use_bbox_crop=True):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.imagenet_norm = imagenet_norm
        self.cache_images = cache_images
        self.use_bbox_crop = use_bbox_crop
        self._cache = {}
        with open(ann_path, "r") as f:
            data = json.load(f)
        self.images = {im["id"]: im for im in data["images"]}
        anns = [a for a in data["annotations"] if a.get("num_keypoints", 0) >= 5]
        if max_samples and max_samples < len(anns):
            rng = random.Random(subset_seed)
            indices = list(range(len(anns)))
            rng.shuffle(indices)
            anns = [anns[i] for i in indices[:max_samples]]
        elif max_samples:
            anns = anns[:max_samples]
        self.anns = anns
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1) if imagenet_norm else None
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1) if imagenet_norm else None

    def __len__(self):
        return len(self.anns)

    def _load_one(self, idx):
        ann = self.anns[idx]
        im_id = ann["image_id"]
        im = self.images[im_id]
        path = self.images_dir / im["file_name"]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        kp = ann["keypoints"]
        keypoints = torch.tensor(kp, dtype=torch.float32).view(17, 3)

        if self.use_bbox_crop and "bbox" in ann:
            x, y, bw, bh = ann["bbox"]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + bw))
            y2 = min(h, int(y + bh))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            size = max(x2 - x1, y2 - y1) * 1.25
            x1 = max(0, int(cx - size / 2))
            y1 = max(0, int(cy - size / 2))
            x2 = min(w, int(cx + size / 2))
            y2 = min(h, int(cy + size / 2))
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < 8 or crop_h < 8:
                crop_w = max(crop_w, 8)
                crop_h = max(crop_h, 8)
            img = img.crop((x1, y1, x2, y2))
            keypoints[:, 0] = (keypoints[:, 0] - x1) / crop_w
            keypoints[:, 1] = (keypoints[:, 1] - y1) / crop_h
            keypoints[:, 0] = keypoints[:, 0].clamp(0.0, 1.0)
            keypoints[:, 1] = keypoints[:, 1].clamp(0.0, 1.0)
        else:
            keypoints[:, 0] /= w
            keypoints[:, 1] /= h

        if self.image_size and (img.width != self.image_size or img.height != self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        if self.imagenet_norm:
            img_t = (img_t - self._mean) / self._std
        return img_t, keypoints

    def __getitem__(self, idx):
        if self.cache_images and idx in self._cache:
            return self._cache[idx]
        img_t, keypoints = self._load_one(idx)
        if self.cache_images:
            self._cache[idx] = (img_t, keypoints)
        return img_t, keypoints


def apply_random_occlusion(image, mask_ratio=0.3, device=None):
    """Random rectangular occlusion. image: (B, C, H, W). mask_ratio: approximate fraction to zero out."""
    B, C, H, W = image.shape
    device = device or image.device
    out = image.clone()
    n = int(H * W * mask_ratio + 0.5)
    for b in range(B):
        # Random top-left and size so ~mask_ratio of area is covered
        h_len = max(1, int(H * (mask_ratio ** 0.5)) + torch.randint(-2, 3, (1,), device=device).item())
        w_len = max(1, min(W, (n + h_len - 1) // h_len))
        h_len = min(H, max(1, (n + w_len - 1) // w_len))
        top = torch.randint(0, max(1, H - h_len + 1), (1,), device=device).item()
        left = torch.randint(0, max(1, W - w_len + 1), (1,), device=device).item()
        out[b, :, top : top + h_len, left : left + w_len] = 0
    return out


def perturb_input(images, occlusion_ratio=0.3, gaussian_sigma=0.05, device=None):
    """
    Perturbed view for PMH: occlusion + Gaussian noise (simultaneous).
    Occlusion removes spatial regions; Gaussian corrupts remaining pixels.
    """
    device = device or images.device
    out = apply_random_occlusion(images, mask_ratio=occlusion_ratio, device=device)
    out = out + torch.randn_like(out, device=device) * gaussian_sigma
    # Keep valid range: [0,1] for raw pixels; skip clamp if ImageNet-normalized (values outside [0,1])
    if images.min() >= 0 and images.max() <= 1:
        out = out.clamp(0.0, 1.0)
    return out


def apply_eval_attack(images, occlusion_ratio=0.0, gaussian_sigma=0.0, device=None):
    """
    Apply a single attack for evaluation: none (clean), occlusion only, Gaussian only, or both.
    Returns perturbed images; (0, 0) returns images unchanged.
    """
    device = device or images.device
    if occlusion_ratio <= 0 and gaussian_sigma <= 0:
        return images
    if occlusion_ratio > 0 and gaussian_sigma <= 0:
        return apply_random_occlusion(images, mask_ratio=occlusion_ratio, device=device)
    if occlusion_ratio <= 0 and gaussian_sigma > 0:
        out = images + torch.randn_like(images, device=device) * gaussian_sigma
        if images.min() >= 0 and images.max() <= 1:
            out = out.clamp(0.0, 1.0)
        return out
    # Both
    out = apply_random_occlusion(images, mask_ratio=occlusion_ratio, device=device)
    out = out + torch.randn_like(out, device=device) * gaussian_sigma
    if images.min() >= 0 and images.max() <= 1:
        out = out.clamp(0.0, 1.0)
    return out


def get_coco_pose_dataloader(
    data_dir,
    batch_size=32,
    num_workers=0,
    image_size=256,
    max_samples=None,
    subset_seed=42,
    shuffle=True,
    imagenet_norm=False,
    cache_images=False,
    use_bbox_crop=True,
    split="train",
    loader_seed=None,
):
    """
    Build DataLoader for COCO 2D pose.
    split='train': use train2017 if present, else val2017 (and warn).
    split='val': use val2017 (for validation/eval).
    max_samples: if set, take a random subset (seed=subset_seed) for reproducibility.
    loader_seed: if set and shuffle=True, fixes batch order (DataLoader generator).
    """
    if split == "val":
        images_dir, ann_path = ensure_coco_pose(data_dir)
        split_label = "VAL"
    else:
        train_paths = ensure_coco_train(data_dir)
        if train_paths:
            images_dir, ann_path = train_paths
            split_label = "TRAIN"
        else:
            images_dir, ann_path = ensure_coco_pose(data_dir)
            split_label = "TRAIN (val2017 fallback)"
            print("Warning: No train2017 found. Training on val2017; eval will be on same data (not held-out).", flush=True)
    dataset = CocoPose2DDataset(
        images_dir, ann_path, image_size=image_size, max_samples=max_samples, subset_seed=subset_seed,
        imagenet_norm=imagenet_norm, cache_images=cache_images, use_bbox_crop=use_bbox_crop,
    )
    n = len(dataset)
    print(f"  [{split_label}] {images_dir} -> {n} samples", flush=True)
    loader_kw = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kw["prefetch_factor"] = 2
    if shuffle and loader_seed is not None:
        loader_kw["generator"] = torch.Generator().manual_seed(int(loader_seed))
    if num_workers > 0 and loader_seed is not None:
        loader_kw["worker_init_fn"] = partial(_seed_worker, base_seed=int(loader_seed))
    return torch.utils.data.DataLoader(dataset, **loader_kw)


def get_pose_dataloader(data_dir, split="train", batch_size=32, num_workers=0, occlusion_ratio=0.0):
    """
    Placeholder for 3D datasets (Human3.6M / 3DPW). For COCO 2D use get_coco_pose_dataloader().
    When implemented: return DataLoader that yields (image, pose_3d). image: (B, 3, H, W), pose_3d: (B, 17, 3) in mm.
    """
    data_dir = Path(data_dir)
    if not (data_dir / "train").exists() and not (data_dir / "train.h5").exists():
        return None
    return None


class DummyPoseDataset(torch.utils.data.Dataset):
    """Minimal placeholder so train/eval can be imported. Replace with real dataset."""
    def __init__(self, num_samples=100, image_size=256):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(3, self.image_size, self.image_size)
        pose = torch.randn(17, 3) * 500  # dummy 3D in mm
        return image, pose
