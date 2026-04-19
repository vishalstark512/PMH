"""
Launch baseline on GPU 0 and E1 on GPU 1 in parallel (two processes).
Use when you have 2 GPUs and want to train both runs at once instead of DataParallel on one run.

  python train_both_gpus.py --dataset coco --data_dir ./data --pretrained --max_train_samples 10000

Pass through any train.py args (epochs, batch_size, etc.). Each run uses one GPU.
"""
import subprocess
import sys
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    train_py = script_dir / "train.py"
    argv = [sys.executable, str(train_py)] + sys.argv[1:]

    # Build baseline argv: add --run baseline --gpu 0
    base_argv = argv + ["--run", "baseline", "--gpu", "0"]
    e1_argv = argv + ["--run", "E1", "--gpu", "1"]

    print("Launching baseline on GPU 0 and E1 on GPU 1...", flush=True)
    p_baseline = subprocess.Popen(base_argv, cwd=script_dir, stdout=sys.stdout, stderr=sys.stderr)
    p_e1 = subprocess.Popen(e1_argv, cwd=script_dir, stdout=sys.stdout, stderr=sys.stderr)
    # Output will interleave; for clean logs run in two terminals instead:
    #   CUDA_VISIBLE_DEVICES=0 python train.py --run baseline ...
    #   CUDA_VISIBLE_DEVICES=1 python train.py --run E1 ...
    p_baseline.wait()
    p_e1.wait()
    sys.exit(0 if (p_baseline.returncode == 0 and p_e1.returncode == 0) else 1)


if __name__ == "__main__":
    main()
