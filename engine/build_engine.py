"""Build a TensorRT engine from an exported model, on the machine that will run it, and check it.

Standalone on purpose: `partlabeler export-model` copies this file next to the ONNX model, model.json and a check/
folder (held-out frames with the boxes and scores the trained model gives them), so that folder can be copied to the
production machine as it is. A TensorRT engine only runs on the GPU model and TensorRT version it was built with, so
build it there:

    python build_engine.py                  # <model>.engine in FP16, then the check
    python build_engine.py --fp32           # full precision (if the FP16 check fails)
    python build_engine.py --boxes-only     # without the outline (mask) output: faster, the same boxes
    python build_engine.py --check-only     # check an engine built before
    python build_engine.py --trtexec /usr/src/tensorrt/bin/trtexec

Speed without changing the model: FP16 (the check below proves the boxes stay the trained model's), static input
shape and batch (model.json "batch": 2 runs two cameras' frames in one go), and the builder's highest optimization
level (5: it tries more kernels, so the build takes longer, the precision is the same) when this TensorRT has it.
INT8 is left out on purpose: it needs calibration and can cost accuracy. TensorRT 10 and older build FP16 from the
FP32 model with --fp16; TensorRT 11 and newer take precision from the graph, so they build from the FP16 model
(<model>.fp16.onnx, FP32 inputs and outputs kept). A bundle may carry only the FP16 model (half the size): TensorRT 10
then builds it strongly typed too, and --fp32 needs the FP32 model copied in. The build log (build_log.txt) ends with
trtexec's timing of the engine on this GPU. --boxes-only (outline models) prunes the mask output first (with the
`onnx` package): the boxes and scores are the same, the engine skips making every query's outline (RF-DETR seg-small
at 960 px on an RTX 3060: 25 ms instead of 31 ms); use it when only the parts' boxes are needed.

Needs TensorRT's trtexec (found on PATH, in $TENSORRT_ROOT/bin, /usr/src/tensorrt/bin or C:\\Program Files\\NVIDIA
GPU Computing Toolkit\\TensorRT*\\bin); the check also needs numpy and OpenCV. The check runs every check/ frame through
the engine (with trtexec) and compares with check/expected.json, using the preparation and decoding model.json
describes, i.e. what production code has to do: it fails when a part is lost or added, a box moves more than 4 px, or
(outline models) a part's pixels overlap the trained model's by less than 85% (IoU).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_BOX_PX = 4.0
MIN_OUTLINE_IOU = 0.85          # outline models: each part's pixels against the trained model's


def find_trtexec(given=None):
    names = ("trtexec.exe", "trtexec") if os.name == "nt" else ("trtexec",)
    places = [given] if given else []
    places += [shutil.which("trtexec")]
    places += [str(Path(os.environ["TENSORRT_ROOT"]) / "bin" / n) for n in names if os.environ.get("TENSORRT_ROOT")]
    places += ["/usr/src/tensorrt/bin/trtexec", "/usr/local/tensorrt/bin/trtexec"]
    places += sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT*\bin\trtexec.exe"), reverse=True)
    return next((p for p in places if p and Path(p).is_file()), None)


def prepare(rgb, card):
    """One RGB frame (H x W x 3 uint8) as the model's input: [1, 3, S, S] float32."""
    import cv2
    import numpy as np
    s = card["resolution"]
    x = cv2.resize(rgb.astype(np.float32) / 255, (s, s), interpolation=cv2.INTER_LINEAR)   # half-pixel, no antialias
    x = (x - np.array(card["input"]["mean"], np.float32)) / np.array(card["input"]["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)


def decode(dets, labels, width, height, threshold):
    """(boxes x1, y1, x2, y2 in frame pixels, scores, class ids, query ids) from one frame's dets [Q, 4] and labels
    [Q, C + 1]."""
    import numpy as np
    scores = 1 / (1 + np.exp(-labels[:, :-1].clip(-88, 88)))          # sigmoid per class; last column: background
    C = scores.shape[1]
    top = np.argsort(-scores.ravel(), kind="stable")[:len(dets)]
    top = top[scores.ravel()[top] > threshold]
    q, cls = top // C, top % C
    cx, cy, w, h = dets[q].T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * [width, height, width, height]
    return xyxy.clip(0, [width, height, width, height]), scores.ravel()[top], cls, q


def outlines(masks, q, width, height):
    """Part pixels [K, height, width] of the kept pairs: their query planes of masks [Q, Hm, Wm] resized to the
    frame (bilinear, half-pixel centres), > 0."""
    import cv2
    import numpy as np
    if not len(q):
        return np.zeros((0, height, width), bool)
    return np.stack([cv2.resize(masks[i].astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) > 0
                     for i in q])


def read_outputs(path):
    """trtexec --exportOutput: [{"name", "dimensions": "1x300x4", "values": [...]}, ...] -> {name: array}."""
    import numpy as np
    out = {}
    for t in json.loads(Path(path).read_text()):
        out[t["name"]] = np.array(t["values"], np.float32).reshape([int(d) for d in t["dimensions"].split("x")])
    return out


def trt_major(usage):
    """TensorRT's major version from trtexec's banner ("[TensorRT v110300]" = 11.3.0, "[TensorRT v8601]" = 8.6.1)."""
    import re
    m = re.search(r"TensorRT v(\d+)", usage)
    n = int(m.group(1)) if m else 0
    return n // 10000 if n >= 100000 else n // 1000


def build(trtexec, card, engine, fp32, boxes_only=False):
    usage = subprocess.run([trtexec, "--help"], capture_output=True, text=True, errors="replace")
    usage = usage.stdout + usage.stderr
    onnx, flags = HERE / card["model"], []
    fp16_only = not onnx.is_file()                          # a small bundle: the FP16 model alone
    if fp32 and fp16_only:
        sys.exit(f"--fp32 needs {onnx.name} (the FP32 model) in this folder")
    if not fp32 and (trt_major(usage) >= 11 or fp16_only):  # strongly typed: precision from the graph
        if not card.get("model_fp16") or not (HERE / card["model_fp16"]).is_file():
            sys.exit("this TensorRT needs the FP16 model (export again with onnxconverter-common installed), or --fp32")
        onnx = HERE / card["model_fp16"]
        if trt_major(usage) < 11:
            flags.append("--stronglyTyped" if "stronglyTyped" in usage else "--fp16")
    elif not fp32:
        flags.append("--fp16")
    if boxes_only:                                          # the same graph up to the boxes and scores
        try:
            import onnx as ox
        except ImportError:
            sys.exit("--boxes-only needs the onnx package: pip install onnx")
        pruned = onnx.with_name(onnx.stem + ".boxes.onnx")
        ox.utils.extract_model(str(onnx), str(pruned), [card["input"]["name"]], ["dets", "labels"])
        onnx = pruned
    if "builderOptimizationLevel" in usage:
        flags.append("--builderOptimizationLevel=5")
    cmd = [trtexec, f"--onnx={onnx}", f"--saveEngine={engine}"] + flags
    print(" ".join(map(str, cmd)), "\n(building: several minutes at the highest optimization level)", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    log = HERE / "build_log.txt"
    log.write_text(r.stdout + r.stderr)
    if r.returncode:
        sys.exit(f"trtexec failed; see {log}")
    for line in r.stdout.splitlines():                   # trtexec's timing of the new engine on this GPU
        if any(k in line for k in ("Throughput:", "Latency:", "GPU Compute Time:", "TensorRT version")):
            print(line.split("] ", 1)[-1])
    print(f"engine: {engine} (log: {log})")


def check(trtexec, engine, card):
    import cv2
    import numpy as np
    expected = json.loads((HERE / "check" / "expected.json").read_text())
    found = missing = extra = 0
    worst_box = worst_score = 0.0
    ious = []
    n = card.get("batch", 1)                                   # frames per run: different frames run together
    names = list(expected)
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(0, len(names), n):
            chunk = names[k:k + n]
            rgbs = [cv2.cvtColor(cv2.imread(str(HERE / "check" / name)), cv2.COLOR_BGR2RGB) for name in chunk]
            inp, out = Path(tmp) / "input.bin", Path(tmp) / "out.json"
            np.concatenate([prepare(rgb, card) for rgb in rgbs + rgbs[-1:] * (n - len(rgbs))]).tofile(inp)
            # relative names: --loadInputs is "name:file", and a Windows path's "C:" would split it
            r = subprocess.run([trtexec, f"--loadEngine={Path(engine).resolve()}",
                                f"--loadInputs={card['input']['name']}:{inp.name}", f"--exportOutput={out.name}",
                                "--iterations=1", "--warmUp=0", "--duration=0", "--avgRuns=1"],
                               capture_output=True, text=True, errors="replace", cwd=tmp)
            if r.returncode:
                sys.exit(f"trtexec could not run the engine:\n{(r.stdout + r.stderr)[-2000:]}")
            o = read_outputs(out)
            for j, (name, rgb) in enumerate(zip(chunk, rgbs)):
                parts = expected[name]
                H, W = rgb.shape[:2]
                xyxy, scores, cls, q = decode(o["dets"][j], o["labels"][j], W, H, card["threshold"])
                px = outlines(o["masks"][j], q, W, H) if "masks" in o else [None] * len(q)
                got = {}
                for b, p, c, m in zip(xyxy, scores, cls, px):                # surest first: one box per class
                    got.setdefault(card["classes"][c], (b, p, m))
                for part, want in parts.items():
                    if part not in got:
                        missing += 1
                        print(f"  {name}: {part} lost")
                        continue
                    found += 1
                    worst_box = max(worst_box, float(np.abs(got[part][0] - np.array(want["box"])).max()))
                    worst_score = max(worst_score, abs(float(got[part][1]) - want["score"]))
                    if want.get("mask") and got[part][2] is not None:   # outlines: pixel overlap with the trained model
                        t = cv2.imread(str(HERE / "check" / want["mask"]), cv2.IMREAD_GRAYSCALE) > 0
                        a = got[part][2]
                        ious.append(float((a & t).sum() / max(1, (a | t).sum())))
                for part in set(got) - set(parts):
                    extra += 1
                    print(f"  {name}: {part} added (score {float(got[part][1]):.3f})")
    print(f"check on {len(expected)} frames: {found} parts found, {missing} lost, {extra} added; largest box "
          f"difference {worst_box:.2f} px, largest score difference {worst_score:.3f}"
          + (f"; outline overlap (IoU) lowest {min(ious):.3f}, mean {np.mean(ious):.3f}" if ious else ""))
    return missing == 0 and extra == 0 and worst_box <= MAX_BOX_PX and (not ious or min(ious) >= MIN_OUTLINE_IOU)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fp32", action="store_true", help="build in full precision instead of FP16")
    ap.add_argument("--boxes-only", action="store_true", help="without the outline output (faster, same boxes)")
    ap.add_argument("--check-only", action="store_true", help="only check an engine built before")
    ap.add_argument("--trtexec", help="path to trtexec")
    a = ap.parse_args()
    card = json.loads((HERE / "model.json").read_text())
    kind = (".boxes" if a.boxes_only else "") + (".fp32" if a.fp32 else "")
    engine = (HERE / card["model"]).with_suffix(kind + ".engine")
    trtexec = find_trtexec(a.trtexec)
    if not trtexec:
        sys.exit("trtexec not found: install TensorRT, or pass --trtexec /path/to/trtexec")
    if not a.check_only:
        build(trtexec, card, engine, a.fp32, a.boxes_only)
    if not (HERE / "check" / "expected.json").exists():
        return
    try:
        ok = check(trtexec, engine, card)
    except ImportError:
        sys.exit("the check needs numpy and OpenCV: pip install numpy opencv-python")
    if not ok:
        sys.exit("the engine's results differ from the trained model's" +
                 ("" if a.fp32 else ": try python build_engine.py --fp32"))
    print("OK: the engine gives the trained model's results")


if __name__ == "__main__":
    main()
