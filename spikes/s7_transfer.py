"""S7 step 4: label the other videos automatically, in the same format as the source dataset.

For each video: every Nth frame -> find the car (SAM 3) -> detect parts inside the crop
(trained RF-DETR) -> boxes back in full-frame coordinates -> YOLO label file. Output mirrors
the source dataset (images/, labels/, classes.txt, data.yaml) plus a preview video and a
per-class summary.

    python spikes/s7_transfer.py --ckpt data/example/s7_runs/baseline/checkpoint_best_total.pth --threshold 0.4
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
from rfdetr import RFDETRSmall

from engine.parent import ParentFinder, crop_box

ROOT = Path("data/example")
VIDEOS = ["black_car_front.mp4", "green_car_front.mp4"]


def yolo_line(c: int, x1, y1, x2, y2, W: int, H: int) -> str:
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
    return f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def label_video(video: Path, out: Path, model, finder, names, threshold: float, every: int, car_every: int):
    stem = video.stem.split("_front")[0]
    for d in ("images", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "dataset/classes.txt", out / "classes.txt")
    (out / "data.yaml").write_text("path: .\ntrain: images\nval: images\n\n"
                                   f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    counts, per_frame, car, k = Counter(), [], None, 0
    with av.open(str(video)) as src, av.open(str(out / "preview.mp4"), "w") as dst:
        s = src.streams.video[0]
        rate = max(1, round(float(s.average_rate) / every))
        vs = dst.add_stream("libx264", rate=rate)
        vs.width, vs.height, vs.pix_fmt = 960, 540, "yuv420p"
        for i, fr in enumerate(src.decode(video=0)):
            if i % every:
                continue
            img = fr.to_image()
            W, H = img.size
            if car is None or k % car_every == 0:           # the car barely moves between samples
                box = finder.find(img)
                car = crop_box(box, W, H) if box else (0, 0, W, H)
            k += 1
            det = model.predict(img.crop(car), threshold=threshold)
            ox, oy = car[0], car[1]
            lines = [yolo_line(int(c), x1 + ox, y1 + oy, x2 + ox, y2 + oy, W, H)
                     for (x1, y1, x2, y2), c in zip(det.xyxy, det.class_id)]
            name = f"{stem}_f{i:06d}"
            img.save(out / "images" / f"{name}.jpg", quality=95)
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            counts.update(int(c) for c in det.class_id)
            per_frame.append(len(lines))
            view = img.copy()
            draw = ImageDraw.Draw(view)
            for (x1, y1, x2, y2), c, conf in zip(det.xyxy, det.class_id, det.confidence):
                draw.rectangle([x1 + ox, y1 + oy, x2 + ox, y2 + oy], outline=(255, 50, 50), width=4)
                draw.text((x1 + ox, y1 + oy - 14), f"{names[c]} {conf:.2f}", fill=(255, 255, 0))
            for pkt in vs.encode(av.VideoFrame.from_image(view.resize((960, 540)))):
                dst.mux(pkt)
        for pkt in vs.encode():
            dst.mux(pkt)
    summary = {"frames": len(per_frame), "boxes": int(sum(per_frame)),
               "boxes_per_frame_median": float(np.median(per_frame)) if per_frame else 0,
               "empty_frames": int(sum(n == 0 for n in per_frame)),
               "boxes_per_class": {names[c]: n for c, n in sorted(counts.items())}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--every", type=int, default=5, help="label every Nth frame (source dataset used 5)")
    ap.add_argument("--car-every", type=int, default=4, help="re-find the car every N labeled frames")
    ap.add_argument("--out", default="data/example/auto_labels")
    args = ap.parse_args()
    names = (ROOT / "dataset/classes.txt").read_text().split()
    model = RFDETRSmall(pretrain_weights=args.ckpt, resolution=args.resolution)
    finder = ParentFinder("car")
    for v in VIDEOS:
        out = Path(args.out) / Path(v).stem
        shutil.rmtree(out, ignore_errors=True)
        print(v, json.dumps(label_video(ROOT / "videos" / v, out, model, finder, names,
                                        args.threshold, args.every, args.car_every)), flush=True)


if __name__ == "__main__":
    main()
