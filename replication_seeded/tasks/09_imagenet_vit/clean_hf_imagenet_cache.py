"""
Remove Hugging Face caches for ImageNet-1k (e.g. after a full train download).

Safe to run if you only want to reclaim disk space; you will re-download
**validation-only** shards next time you run ``measure_tdi.py --val_source hf``
(after the code fix that loads parquet by URL).

Windows default cache root::

    %USERPROFILE%\\.cache\\huggingface\\hub

Usage::

    python clean_hf_imagenet_cache.py              # dry-run (lists only)
    python clean_hf_imagenet_cache.py --delete
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def cache_roots() -> list[Path]:
    home = Path.home()
    roots = [
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "huggingface" / "datasets",
    ]
    return [p for p in roots if p.is_dir()]


def candidate_dirs() -> list[Path]:
    out: list[Path] = []
    for root in cache_roots():
        if not root.is_dir():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            name = p.name.lower()
            if "imagenet" in name or "ilsvrc" in name:
                out.append(p)
    return sorted(set(out))


def main() -> None:
    p = argparse.ArgumentParser(description="Remove HF ImageNet-related cache folders")
    p.add_argument("--delete", action="store_true", help="Actually delete (default is dry-run)")
    args = p.parse_args()

    found = candidate_dirs()
    if not found:
        print("No matching cache directories under:")
        for r in cache_roots():
            print(f"  {r}")
        print("(Nothing to remove, or cache is elsewhere.)")
        return

    total = 0
    for d in found:
        try:
            sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        except OSError:
            sz = 0
        total += sz
        gb = sz / (1024**3)
        print(f"  {d}  (~{gb:.2f} GiB)")

    print(f"\nTotal ~{total / (1024**3):.2f} GiB across {len(found)} folder(s).")

    if not args.delete:
        print("\nDry-run only. Re-run with --delete to remove these folders.")
        return

    for d in found:
        print(f"Deleting {d} ...")
        shutil.rmtree(d, ignore_errors=True)
    print("Done. Partial train parquet files under hub downloads may also exist;")
    print("if disk is still full, inspect:")
    print(f"  {Path.home() / '.cache' / 'huggingface' / 'hub'}")


if __name__ == "__main__":
    main()
