"""S7 step 1: build the training set from the labeled White frames.

Each labeled frame is cropped to the car (SAM 3 text prompt), labels are moved into crop
coordinates, and 20% of frames are held out as whole blocks of consecutive frames (so the
test set is not near-duplicates of training frames).

    python spikes/s7_prepare.py
"""
import json
import shutil
from pathlib import Path

from PIL import Image

from engine.parent import ParentFinder, crop_box

ROOT = Path("data/example")
OUT = ROOT / "s7_ds"
N_BLOCKS, HELD_OUT = 10, {3, 8}          # 20% of frames, from two different phases of the video


def to_crop(line: str, crop, W: int, H: int) -> str | None:
    c, cx, cy, w, h = line.split()
    x1, y1 = (float(cx) - float(w) / 2) * W, (float(cy) - float(h) / 2) * H
    x2, y2 = x1 + float(w) * W, y1 + float(h) * H
    X1, Y1, X2, Y2 = crop
    x1, y1, x2, y2 = max(x1, X1), max(y1, Y1), min(x2, X2), min(y2, Y2)
    if x2 <= x1 or y2 <= y1:
        return None
    cw, ch = X2 - X1, Y2 - Y1
    return f"{c} {((x1 + x2) / 2 - X1) / cw:.6f} {((y1 + y2) / 2 - Y1) / ch:.6f} {(x2 - x1) / cw:.6f} {(y2 - y1) / ch:.6f}"


def main():
    names = (ROOT / "dataset/classes.txt").read_text().split()
    labels = sorted((ROOT / "dataset/labels").glob("*.txt"))
    shutil.rmtree(OUT, ignore_errors=True)
    for split in ("train", "valid"):
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)

    finder = ParentFinder("car")
    crops, dropped, per_block = {}, 0, len(labels) / N_BLOCKS
    for k, f in enumerate(labels):
        img = Image.open(ROOT / "dataset/images" / (f.stem + ".jpg")).convert("RGB")
        box = finder.find(img)
        crop = crop_box(box, *img.size) if box else (0, 0, *img.size)
        crops[f.stem] = crop
        split = "valid" if int(k // per_block) in HELD_OUT else "train"
        rows = [to_crop(l, crop, *img.size) for l in f.read_text().splitlines() if l.strip()]
        dropped += sum(r is None for r in rows)
        img.crop(crop).save(OUT / split / "images" / (f.stem + ".jpg"), quality=95)
        (OUT / split / "labels" / f.name).write_text("\n".join(r for r in rows if r) + ("\n" if any(rows) else ""))
        if k % 100 == 0:
            print(f"{k}/{len(labels)}", flush=True)

    (OUT / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\n"
        f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    (OUT / "crops.json").write_text(json.dumps(crops))
    counts = {s: len(list((OUT / s / "images").glob("*.jpg"))) for s in ("train", "valid")}
    print(f"done: {counts}, labels fully outside the car crop: {dropped}")


if __name__ == "__main__":
    main()
