"""S1: SAM 3 tracker on this GPU.

Checks: reference prompt -> propagate a chunk -> correction click mid-chunk -> re-propagate
-> full teardown. Records peak VRAM, speed, and whether VRAM returns to baseline (issue #305).

    python spikes/s1_tracker.py                       # HF sample clip, point prompt
    python spikes/s1_tracker.py --video my.mp4 --box 100,120,340,400
"""
import argparse, gc, json, time
from pathlib import Path

import torch
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from engine.frames import SAMPLE, read_frames
from engine.models import SAM3, fetch

OUT = Path(__file__).parent / "out"


def vram_mb() -> float:
    return torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=SAMPLE)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--point", default="210,350", help="x,y positive click on frame 0")
    ap.add_argument("--box", default=None, help="x1,y1,x2,y2 on frame 0 (overrides --point)")
    ap.add_argument("--state-on-gpu", action="store_true", help="keep frames + tracking state on the GPU")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    store = dev if args.state_on_gpu else "cpu"
    res = {"device": dev, "gpu": torch.cuda.get_device_name() if dev == "cuda" else None, "dtype": args.dtype,
           "state_device": store}

    frames = read_frames(args.video, args.frames)
    res["frames"], res["size"] = len(frames), frames[0].size

    base = vram_mb()
    t = time.perf_counter()
    weights = fetch(SAM3)
    model = Sam3TrackerVideoModel.from_pretrained(weights, dtype=dtype).to(dev)
    proc = Sam3TrackerVideoProcessor.from_pretrained(weights)
    res["load_s"] = round(time.perf_counter() - t, 1)
    res["vram_model_mb"] = round(vram_mb() - base)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sess = proc.init_video_session(video=frames, inference_device=dev, inference_state_device=store,
                                   video_storage_device=store, dtype=dtype)
    prompt = dict(input_boxes=[[[float(v) for v in args.box.split(",")]]]) if args.box else \
        dict(input_points=[[[[float(v) for v in args.point.split(",")]]]], input_labels=[[[1]]])
    proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=1, **prompt)

    def propagate(start=0):
        t0, areas = time.perf_counter(), {}
        with torch.inference_mode():
            for out in model.propagate_in_video_iterator(sess, start_frame_idx=start):
                m = proc.post_process_masks([out.pred_masks], original_sizes=[[sess.video_height, sess.video_width]])[0]
                areas[out.frame_idx] = int(m[0].sum())
        return len(areas) / (time.perf_counter() - t0), areas

    fps, areas = propagate()
    res["fps_first_pass"] = round(fps, 2)
    res["frames_with_mask"] = sum(a > 0 for a in areas.values())

    mid = len(frames) // 2  # correction click, then re-track from there
    if args.box:
        b = [float(v) for v in args.box.split(",")]
        x, y = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    else:
        x, y = map(float, args.point.split(","))
    proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=mid, obj_ids=1,
                                         input_points=[[[[x, y]]]], input_labels=[[[1]]], clear_old_inputs=False)
    fps2, _ = propagate(start=mid)
    res["fps_after_correction"] = round(fps2, 2)
    if dev == "cuda":
        res["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)

    del sess
    gc.collect()
    torch.cuda.empty_cache()
    res["vram_after_teardown_mb"] = round(vram_mb() - base)  # should be ~= vram_model_mb
    res["leak_mb"] = res["vram_after_teardown_mb"] - res["vram_model_mb"]
    res["pass"] = res["frames_with_mask"] > 0.9 * len(frames) and res.get("peak_vram_mb", 0) < 10_240 and res["leak_mb"] < 256

    print(json.dumps(res, indent=2))
    OUT.mkdir(exist_ok=True)
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps({"spike": "S1", **res}) + "\n")


if __name__ == "__main__":
    main()
