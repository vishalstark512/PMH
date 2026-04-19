"""
Turn **flat** ILSVRC2012 validation images into an ImageFolder tree::

    <out_dir>/n01440764/ILSVRC2012_val_00030958.JPEG

Required devkit files (ILSVRC2012_devkit_t12)::

    data/ILSVRC2012_validation_ground_truth.txt   — one integer per line (1..1000)
    data/ILSVRC2012_synset_words.txt              — 1000 lines, first token = nXXXXXXXX

Usage::

    python sort_imagenet_validation.py ^
      --flat_dir D:\\imagenet\\ILSVRC2012_img_val ^
      --out_dir D:\\imagenet\\Data\\CLS-LOC\\val ^
      --devkit D:\\imagenet\\ILSVRC2012_devkit_t12

Then run TDI::

    python measure_tdi.py measure --val_dir D:\\imagenet\\Data\\CLS-LOC\\val ...
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _read_synsets(devkit: Path) -> list[str]:
    syn_path = devkit / "data" / "ILSVRC2012_synset_words.txt"
    if not syn_path.is_file():
        raise FileNotFoundError(f"Missing {syn_path}")
    synsets: list[str] = []
    with open(syn_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            synsets.append(line.split()[0])
    if len(synsets) != 1000:
        raise ValueError(f"Expected 1000 synsets in {syn_path}, got {len(synsets)}")
    return synsets


def _read_ground_truth(devkit: Path) -> list[int]:
    gt_path = devkit / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing {gt_path}")
    out: list[int] = []
    with open(gt_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(int(line))
    if len(out) != 50000:
        raise ValueError(f"Expected 50000 lines in {gt_path}, got {len(out)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Sort flat ImageNet val into class folders")
    p.add_argument("--flat_dir", required=True, help="Folder with ILSVRC2012_val_*.JPEG")
    p.add_argument("--out_dir", required=True, help="Output root (ImageFolder), e.g. .../val")
    p.add_argument("--devkit", required=True, help="Path to ILSVRC2012_devkit_t12")
    p.add_argument("--dry_run", action="store_true", help="Print moves only")
    args = p.parse_args()

    flat = Path(args.flat_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    devkit = Path(args.devkit).expanduser().resolve()

    if not flat.is_dir():
        raise FileNotFoundError(f"flat_dir not a directory: {flat}")

    synsets = _read_synsets(devkit)
    gt = _read_ground_truth(devkit)

    n_done = 0
    n_skip = 0
    for i, cls_1 in enumerate(gt, start=1):
        if cls_1 < 1 or cls_1 > 1000:
            raise ValueError(f"Bad class id {cls_1} at line {i}")
        folder = synsets[cls_1 - 1]
        stem = f"ILSVRC2012_val_{i:08d}"
        src = flat / f"{stem}.JPEG"
        if not src.is_file():
            src = flat / f"{stem}.jpeg"
        if not src.is_file():
            n_skip += 1
            continue
        dest_dir = out_root / folder
        dest = dest_dir / src.name
        if args.dry_run:
            print(f"would move {src} -> {dest}")
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        n_done += 1

    print(f"Done: moved {n_done} images into {out_root}")
    if n_skip:
        print(f"Skipped {n_skip} (source file missing in {flat})")


if __name__ == "__main__":
    main()
