"""How few labeled frames does the RF-DETR "Boost" detector need?

Trains on N random frames of the S7 training split (car crops, same augmentation, no flip) and
scores on the same 143 held-out frames as S7. Compare with DINOv3 matching (no training) and with
the full 574-frame model.

    python spikes/s6_boost_curve.py --n 15 40
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

from rfdetr import RFDETRSmall

from s7_train import AUG

ROOT = Path("data/example")
FULL = ROOT / "s7_ds"


def subset(n: int, seed: int) -> Path:
    out = ROOT / f"s6_ds_{n}"
    shutil.rmtree(out, ignore_errors=True)
    for split in ("train", "valid"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
    picks = random.Random(seed).sample(sorted((FULL / "train/images").glob("*.jpg")), n)
    for split, files in (("train", picks), ("valid", sorted((FULL / "valid/images").glob("*.jpg")))):
        for f in files:
            shutil.copy(f, out / split / "images" / f.name)
            shutil.copy(FULL / split / "labels" / (f.stem + ".txt"), out / split / "labels" / (f.stem + ".txt"))
    shutil.copy(FULL / "data.yaml", out / "data.yaml")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[15, 40])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for n in args.n:
        ds = subset(n, args.seed)
        run = ROOT / "s6_runs" / f"n{n}"
        epochs = max(30, 2400 // n)              # roughly the same number of training steps per size
        RFDETRSmall().train(dataset_dir=str(ds), output_dir=str(run), epochs=epochs, resolution=640,
                            batch_size=4, grad_accum_steps=1, lr=1e-4, aug_config=AUG, eval_interval=10,
                            checkpoint_interval=1000, tensorboard=False, progress_bar=None, num_workers=2)
        rows = [r for r in csv.DictReader(open(run / "metrics.csv")) if r.get("val/mAP_50")]
        best = max(rows, key=lambda r: float(r["val/mAP_50"]))
        print(f"N={n}: best held-out mAP50 {float(best['val/mAP_50']):.3f}, mAP50-95 {float(best['val/mAP_50_95']):.3f}, "
              f"F1 {float(best['val/F1']):.3f}, recall {float(best['val/recall']):.3f}, "
              f"precision {float(best['val/precision']):.3f} (epoch {int(best['epoch']) + 1}/{epochs})", flush=True)


if __name__ == "__main__":
    main()
