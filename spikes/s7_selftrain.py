"""S7 step 5: one self-training round on the station-2 videos.

prepare: the baseline detector labels every 5th frame of each target video (car crop, conf >= 0.6).
         Grille, skid and roof rack -- classes the source labels in ~93-100% of frames -- are carried
         over from a neighbouring frame (<= 3 samples away) when missed while the car stands still.
         Lamps, logo, sunroof and plate are never filled: the source labels them only part of the time.
train:   fine-tune from the baseline on white train crops + these pseudo-labeled crops; the white
         held-out frames stay the check that nothing regressed.

    python spikes/s7_selftrain.py prepare --ckpt data/example/s7_runs/baseline/checkpoint_best_total.pth
    python spikes/s7_selftrain.py train --ckpt data/example/s7_runs/baseline/checkpoint_best_total.pth
"""
import argparse
import shutil
from pathlib import Path

import av
import numpy as np
from rfdetr import RFDETRSmall

from engine.parent import ParentFinder, crop_box
from s7_train import AUG

ROOT = Path("data/example")
BASE, OUT = ROOT / "s7_ds", ROOT / "s7_ds_selftrain"
VIDEOS = ["black_car_front.mp4", "green_car_front.mp4"]
FILL = {"fr_bumper_grill", "front_bumper_skid", "roof_rack"}


def prepare(args):
    names = (ROOT / "dataset/classes.txt").read_text().split()
    fill_ids = {names.index(n) for n in FILL}
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(BASE, OUT)                       # white train + held-out, data.yaml
    model = RFDETRSmall(pretrain_weights=args.ckpt, resolution=640)
    finder = ParentFinder("car")
    added = filled = 0
    for vid in VIDEOS:
        samples = []                                  # (name, crop image, car box, detections)
        car = None
        with av.open(str(ROOT / "videos" / vid)) as c:
            for i, fr in enumerate(c.decode(video=0)):
                if i % 5:
                    continue
                img = fr.to_image()
                if car is None or len(samples) % 4 == 0:
                    box = finder.find(img)
                    car = crop_box(box, *img.size) if box else None
                if car is None:
                    continue
                crop = img.crop(car)
                det = model.predict(crop, threshold=args.conf)
                samples.append((f"{Path(vid).stem}_f{i:06d}", crop, car, det))
        for k, (name, crop, car, det) in enumerate(samples):
            boxes = [(int(c), *xyxy) for xyxy, c in zip(det.xyxy.tolist(), det.class_id)]
            have = {b[0] for b in boxes}
            for cid in fill_ids - have:
                for d in (1, -1, 2, -2, 3, -3):       # nearest neighbour with the part, car not moved
                    j = k + d
                    if 0 <= j < len(samples) and np.abs(np.subtract(samples[j][2], car)).max() <= 20:
                        nb = samples[j][3]
                        hit = nb.xyxy[nb.class_id == cid]
                        if len(hit):
                            boxes.append((cid, *hit[0].tolist())); filled += 1
                            break
            W, H = crop.size
            lines = [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                     for c, x1, y1, x2, y2 in boxes]
            crop.save(OUT / "train/images" / f"{name}.jpg", quality=95)
            (OUT / "train/labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            added += 1
        print(f"{vid}: {len(samples)} pseudo-labeled frames", flush=True)
    print(f"added {added} frames, carried over {filled} grille/skid/roof-rack boxes")


def train(args):
    model = RFDETRSmall(pretrain_weights=args.ckpt, resolution=640)
    model.train(dataset_dir=str(OUT), output_dir=args.out, epochs=args.epochs, resolution=640,
                batch_size=4, grad_accum_steps=4, lr=5e-5, aug_config=AUG, checkpoint_interval=4,
                tensorboard=False, progress_bar="tqdm", num_workers=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="data/example/s7_runs/selftrain")
    args = ap.parse_args()
    (prepare if args.stage == "prepare" else train)(args)


if __name__ == "__main__":
    main()
