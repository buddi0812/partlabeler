"""S4: Laya Vision as a quality gate on the RTX 3060.

Crops of a tracked object (SAM 3 masks) and random background crops from the sample clip
go through the quality questions. Checks: speed per crop, GPU memory, and whether the
answers separate object crops from background crops.

    python spikes/s4_laya.py
"""
import json, random, statistics, time
from pathlib import Path

import numpy as np
import torch
import laya
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from engine.models import SAM3, fetch
from s1_tracker import SAMPLE, read_frames

OUT = Path(__file__).parent / "out"
LAYA = {"repo": "thaitea/laya-vision", "revision": "8b318c99d7ad3ce19c24369263463882eada9d1e"}
KINDS = ["a person", "a single object", "background such as a wall, floor or empty surface", "a blurred or unclear region"]
QUESTIONS = {
    "what": {"type": "choice", "instructions": "What does this image crop mainly show?", "criteria": KINDS},
    "whole": {"type": "noul", "instructions": "Is the main thing in this crop fully visible, not cut off at the edges?"},
    "blur": {"type": "noul", "instructions": "Is this image blurry?"},
}


def object_boxes(frames, point):
    """Track the clicked object through `frames` with SAM 3; return one padded box per frame."""
    weights = fetch(SAM3)
    model = Sam3TrackerVideoModel.from_pretrained(weights, dtype=torch.bfloat16).to("cuda")
    proc = Sam3TrackerVideoProcessor.from_pretrained(weights)
    sess = proc.init_video_session(video=frames, inference_device="cuda", dtype=torch.bfloat16)
    proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=1,
                                         input_points=[[[point]]], input_labels=[[[1]]])
    boxes = {}
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(sess, start_frame_idx=0):
            m = proc.post_process_masks([out.pred_masks], original_sizes=[[sess.video_height, sess.video_width]])[0][0, 0]
            ys, xs = np.nonzero(m.cpu().numpy())
            if len(xs):
                boxes[out.frame_idx] = pad((xs.min(), ys.min(), xs.max(), ys.max()), 1.3, sess.video_width, sess.video_height)
    del model, sess
    torch.cuda.empty_cache()
    return boxes


def pad(b, f, w, h):
    cx, cy, bw, bh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * f, (b[3] - b[1]) * f
    return (max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)), min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2)))


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def main():
    random.seed(0)
    frames = read_frames(SAMPLE, 12)
    boxes = object_boxes(frames, [210.0, 350.0])
    w, h = frames[0].size
    obj, bg = [], []
    for i, b in boxes.items():
        obj.append(frames[i].crop(b))
        bw, bh = b[2] - b[0], b[3] - b[1]
        for _ in range(50):  # a same-sized box elsewhere in the frame
            x, y = random.randint(0, max(0, w - bw)), random.randint(0, max(0, h - bh))
            cand = (x, y, x + bw, y + bh)
            if not overlaps(cand, b):
                bg.append(frames[i].crop(cand))
                break

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = time.perf_counter()
    agent = laya.load_vlm(LAYA["repo"], revision=LAYA["revision"], device="cuda")
    load_s = time.perf_counter() - t
    for img in obj[:2]:  # warm-up
        agent.predict({"image": img}, QUESTIONS)

    rows, times = [], []
    for kind, crops in (("object", obj), ("background", bg)):
        for img in crops:
            t = time.perf_counter()
            r = agent.predict({"image": img}, QUESTIONS)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
            rows.append((kind, r["answers"]))
    first = rows[0][1]

    def p_choice(ans, label):
        probs = ans.get("probabilities") or {}
        return float(probs.get(label, 0.0)) if isinstance(probs, dict) else 0.0

    def mean(kind, f):
        vals = [f(a) for k, a in rows if k == kind]
        return round(statistics.mean(vals), 3) if vals else None

    res = {
        "object_crops": len(obj), "background_crops": len(bg),
        "load_s": round(load_s, 1),
        "ms_per_crop_median": round(statistics.median(times), 1),
        "ms_per_crop_p90": round(sorted(times)[int(0.9 * (len(times) - 1))], 1),
        "vram_mb": round((torch.cuda.max_memory_allocated() - base) / 2**20),
        "p_background_on_object": mean("object", lambda a: p_choice(a["what"], KINDS[2])),
        "p_background_on_background": mean("background", lambda a: p_choice(a["what"], KINDS[2])),
        "top_answer_object": statistics.mode(a["what"].get("choice") for k, a in rows if k == "object"),
        "top_answer_background": statistics.mode(a["what"].get("choice") for k, a in rows if k == "background"),
        "example_answers": first,
    }
    res["pass"] = bool(res["ms_per_crop_median"] < 100 and
                       res["p_background_on_background"] > res["p_background_on_object"])
    print(json.dumps(res, indent=2, default=str))
    OUT.mkdir(exist_ok=True)
    (OUT / "s4").mkdir(exist_ok=True)
    for i, img in enumerate(obj[:3] + bg[:3]):
        img.save(OUT / "s4" / f"{'obj' if i < 3 else 'bg'}_{i % 3}.jpg")
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps({"spike": "S4", **{k: v for k, v in res.items() if k != "example_answers"}}, default=str) + "\n")


if __name__ == "__main__":
    main()
