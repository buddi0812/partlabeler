"""S2: carry what the tracker learned into the next chunk of a long video.

Tracks one object through the clip in a single session (the reference result), then again
split into two chunks where chunk B starts a fresh session seeded with masks from chunk A:
  overlap : chunk B begins at A's last frame, seeded with that frame's mask
  refs    : overlap, plus A's first and middle frames prepended as extra reference masks
Passes if mean mask IoU against the single session is >= 0.9 on chunk B's frames.

    python spikes/s2_chunks.py
"""
import argparse, gc, json, time
from pathlib import Path

import numpy as np
import torch
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from engine.models import SAM3, fetch
from s1_tracker import SAMPLE, read_frames

OUT = Path(__file__).parent / "out"


def track(model, proc, frames, seeds, dev, dtype, prompt=None):
    """Return {local_frame_idx: HxW bool mask}. `seeds` maps local idx -> mask; `prompt` is a
    point prompt on frame 0 when there are no seeds."""
    sess = proc.init_video_session(video=frames, inference_device=dev, dtype=dtype)
    if prompt:
        proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=1,
                                             input_points=[[[prompt]]], input_labels=[[[1]]])
    for idx, m in seeds.items():
        proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=idx, obj_ids=1,
                                             input_masks=m.astype(np.float32))
    masks = {}
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(sess, start_frame_idx=0):
            m = proc.post_process_masks([out.pred_masks], original_sizes=[[sess.video_height, sess.video_width]])[0]
            masks[out.frame_idx] = m[0, 0].cpu().numpy().astype(bool)
    del sess
    gc.collect()
    torch.cuda.empty_cache()
    return masks


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=SAMPLE)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--point", default="210,350")
    args = ap.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    frames = read_frames(args.video, args.frames)
    n, half = len(frames), len(frames) // 2
    weights = fetch(SAM3)
    model = Sam3TrackerVideoModel.from_pretrained(weights, dtype=dtype).to(dev)
    proc = Sam3TrackerVideoProcessor.from_pretrained(weights)
    point = [float(v) for v in args.point.split(",")]

    t = time.perf_counter()
    single = track(model, proc, frames, {}, dev, dtype, prompt=point)
    a = track(model, proc, frames[:half], {}, dev, dtype, prompt=point)
    last = half - 1

    # overlap: chunk B = frames[last:], local 0 == global `last`
    b_overlap = track(model, proc, frames[last:], {0: a[last]}, dev, dtype)
    # refs: [frame 0, frame half//2] + frames[last:]; global g -> local g - last + 2
    refs = [0, half // 2]
    b_refs = track(model, proc, [frames[i] for i in refs] + frames[last:],
                   {0: a[0], 1: a[half // 2], 2: a[last]}, dev, dtype)

    chunk_b = range(half, n)
    res = {
        "frames": n, "chunk_size": half,
        "chunk_a_iou": round(np.mean([iou(a[g], single[g]) for g in range(half)]), 3),
        "overlap_iou": round(np.mean([iou(b_overlap[g - last], single[g]) for g in chunk_b]), 3),
        "refs_iou": round(np.mean([iou(b_refs[g - last + 2], single[g]) for g in chunk_b]), 3),
        "overlap_min_iou": round(min(iou(b_overlap[g - last], single[g]) for g in chunk_b), 3),
        "refs_min_iou": round(min(iou(b_refs[g - last + 2], single[g]) for g in chunk_b), 3),
        "seconds": round(time.perf_counter() - t, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    res = {k: float(v) if isinstance(v, np.floating) else v for k, v in res.items()}
    res["pass"] = max(res["overlap_iou"], res["refs_iou"]) >= 0.9
    print(json.dumps(res, indent=2))
    OUT.mkdir(exist_ok=True)
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps({"spike": "S2", **res}) + "\n")


if __name__ == "__main__":
    main()
