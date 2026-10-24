"""RF-DETR detector: fine-tune on a YOLO dataset, load a checkpoint, predict boxes.

Used by Teach & Transfer (and later by Boost). Augmentation is colour-heavy so a model taught on one
colour of a product still finds the parts on other colours, with small rotation and shear for a camera that
sits a little differently. Horizontal flip is added only when the
class names have no left/right pairs: a mirror would swap their meaning.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

from engine import hw
from engine.project import IMAGE_EXTS, mirrored_pairs, read_classes

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",      # boxes
         "seg-nano": "RFDETRSegNano", "seg-small": "RFDETRSegSmall", "seg-medium": "RFDETRSegMedium"}   # + outlines
BLOCK = 32                     # patch 16 x 2 windows for all box sizes: resolution must be a multiple
SEG_BLOCK = {"seg-nano": 12, "seg-small": 24, "seg-medium": 24}   # patch 12 x 1 or 2 windows
EFFECTIVE_BATCH = 16           # batch x grad_accum, as in S7 (4 x 4)
_GB_PER_IMAGE = {"nano": 0.7, "small": 1.1, "medium": 1.6,   # training memory at 640 px (S7: small, batch 4 = 4.4 GB)
                 "seg-nano": 1.0, "seg-small": 1.6, "seg-medium": 2.2}   # ponytail: guesses; OOM halves the batch
_GB_BASE = 2.0
# Measured directly (docs/VRAM_CALIBRATION.md, tests/test_vram_calibration.py), real training steps, RTX 3060,
# torch 2.14+cu130, 320 px, batch 1/2/4/8, torch.cuda.max_memory_reserved(): least-squares peak = base + per *
# batch, +10% on the slope as a safety margin (the fit is a touch optimistic at the high end of the tested
# range). Sizes not here fall back to the older, rougher _GB_PER_IMAGE/_GB_BASE guess.
_VRAM_CAL_RESOLUTION = 320
_VRAM_CAL = {"nano": {"base": 0.60, "per_batch": 0.40}, "small": {"base": 0.65, "per_batch": 0.48}}

STRONG = {
    "HueSaturationValue": {"hue_shift_limit": 180, "sat_shift_limit": 50, "val_shift_limit": 40, "p": 0.7},
    "RandomBrightnessContrast": {"brightness_limit": 0.3, "contrast_limit": 0.3, "p": 0.6},
    "ToGray": {"p": 0.1},
    "GaussianBlur": {"blur_limit": 3, "p": 0.2},
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.2},
    "RandomGamma": {"gamma_limit": (80, 120), "p": 0.3},    # exposure
    "Affine": {"scale": (0.85, 1.15), "translate_percent": (-0.05, 0.05), "rotate": (-7, 7), "shear": (-5, 5), "p": 0.5},
    "CLAHE": {"clip_limit": (2, 4), "tile_grid_size": (8, 8), "p": 0.4},          # local contrast: shine, uneven light
    # a hand or a tool over part of the picture: 1-3 holes of 10-25 % of the training image (the work area) each; a
    # part a hole hides entirely loses its label, one it hides in part keeps its visible part (train() makes rfdetr
    # pass the boxes through it)
    "CoarseDropout": {"num_holes_range": (1, 3), "hole_height_range": (0.1, 0.25), "hole_width_range": (0.1, 0.25), "p": 0.4},
}
MEDIUM = {
    "HueSaturationValue": {"hue_shift_limit": 30, "sat_shift_limit": 30, "val_shift_limit": 30, "p": 0.5},
    "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5},
    "GaussianBlur": {"blur_limit": 3, "p": 0.1},
    "Affine": {"scale": (0.9, 1.1), "translate_percent": (-0.03, 0.03), "p": 0.3},
}
# A worn camera (helmet, chest, head strap; Hawk-AI's "Worn" mounting): STRONG's colours and exposure, plus the
# movement such a camera sees -- head turns and tilts, motion blur, the picture's angle changing, a compressed
# wireless stream. Egocentric work blurs with random direction and kernel up to ~7-9 px and varies the viewpoint
# (e.g. arXiv 1606.07256, 2112.07642); the numbers are ours, to be checked against the first worn station's score.
WORN = {
    **{k: STRONG[k] for k in ("HueSaturationValue", "RandomBrightnessContrast", "ToGray", "GaussNoise", "RandomGamma",
                              "CLAHE", "CoarseDropout")},
    "Affine": {"scale": (0.7, 1.3), "translate_percent": (-0.1, 0.1), "rotate": (-20, 20), "shear": (-10, 10), "p": 0.7},
    "Perspective": {"scale": (0.02, 0.08), "p": 0.4},                 # the picture's angle: a head tilting
    "MotionBlur": {"blur_limit": (3, 9), "p": 0.35},                  # a head turning mid-frame
    "GaussianBlur": {"blur_limit": (3, 5), "p": 0.25},
    "ImageCompression": {"compression_type": "jpeg", "quality_range": (60, 95), "p": 0.3},   # a wireless stream
}
AUGS = ("strong", "medium", "off", "worn")
# Training (2026-10-24, research's list for Hawk-AI): longer, with more patience before an early stop; the pretrained
# encoder moves more slowly than the new head (rfdetr's own default for it, 1.5e-4, is above the head's 1e-4).
EPOCHS, PATIENCE, LR, LR_ENCODER = 80, 15, 1e-4, 5e-5
MOUNTS = {"fixed": "strong", "worn": "worn", "off": "off"}    # Hawk-AI's "how the camera is mounted" -> a preset


def boxes_through_holes() -> None:
    """rfdetr passes the boxes only through the transforms it lists as moving pixels; CoarseDropout isn't listed, so a
    part a hole hid entirely kept its label (a positive with nothing to see). Listed, albumentations drops that box and
    trims one partly hidden to what is left (rfdetr's documented way to add one)."""
    from rfdetr.datasets import transforms
    transforms.GEOMETRIC_TRANSFORMS.add("CoarseDropout")


def aug_config(classes, strength: str = "strong") -> dict:
    """Albumentations config for rfdetr. rfdetr's own default includes a flip, so one is always passed;
    "off" is {} (no augmentation, no flip)."""
    if strength not in AUGS:
        raise ValueError(f"unknown augmentation {strength!r}; choose one of {', '.join(AUGS)}")
    if strength == "off":
        return {}
    aug = dict({"strong": STRONG, "medium": MEDIUM, "worn": WORN}[strength])
    if not mirrored_pairs(classes):
        aug["HorizontalFlip"] = {"p": 0.5}
    return aug


def is_seg(size: str) -> bool:
    """A model that also outlines each part (trained on polygons, predicts masks)."""
    return size.startswith("seg-")


def check_resolution(resolution: int, size: str = "small") -> None:
    block = SEG_BLOCK.get(size, BLOCK)
    if resolution <= 0 or resolution % block:
        raise ValueError(f"resolution {resolution} must be a positive multiple of {block} for {size} "
                         f"(e.g. {block * 20}, {block * 30}, {block * 40})")


VRAM_MARGIN_GB = 1.0          # headroom left below free (not total) GPU memory, for anything else that grows a bit


def _vram_needed_gb(size: str, resolution: int, batch: int = 1) -> float:
    """This computer's real peak GPU memory to train size/resolution at batch (fits_gpu(), batch_plan(),
    vram_shortfall_message() all go through here) - the direct 320 px measurement (_VRAM_CAL) scaled to
    resolution by the same squared rule as the base/pixel-count already used for images, when this size was
    measured; else the older, rougher per-image guess."""
    cal = _VRAM_CAL.get(size)
    if cal:
        return cal["base"] + batch * cal["per_batch"] * (resolution / _VRAM_CAL_RESOLUTION) ** 2
    return _GB_BASE + 2 * batch * _GB_PER_IMAGE[size] * (resolution / 640) ** 2


def batch_plan(size: str, resolution: int, free_gb: float | None, n_train: int | None = None) -> tuple[int, int]:
    """(batch, grad_accum) keeping batch x accum = 16: the largest power-of-two batch whose real measured peak
    (_vram_needed_gb) fits in free_gb of GPU memory minus VRAM_MARGIN_GB of headroom, at most the number of
    training images. free_gb None (CPU): batch 2."""
    budget = None if free_gb is None else max(0.0, free_gb - VRAM_MARGIN_GB)
    batch = 2 if free_gb is None else 1
    while budget is not None and batch < EFFECTIVE_BATCH and _vram_needed_gb(size, resolution, batch * 2) <= budget:
        batch *= 2
    if n_train:
        batch = max(1, min(batch, n_train))
    return batch, max(1, EFFECTIVE_BATCH // batch)


def fits_gpu(size: str, resolution: int, free_gb: float) -> bool:
    """Whether even batch 1 fits in free_gb of GPU memory, minus the same VRAM_MARGIN_GB headroom batch_plan()
    budgets with - the preflight (train()) only refuses outright when this says no."""
    return _vram_needed_gb(size, resolution, batch=1) <= max(0.0, free_gb - VRAM_MARGIN_GB)


def vram_shortfall_message(size: str, resolution: int, free_gb: float) -> str:
    """Plain language for when this computer's free GPU memory can't train size/resolution right now - never a
    raw CUDA error (Windows' WDDM driver model raises as vague a "CUDA error: unknown error" as a clean
    out-of-memory one when another process, e.g. a box's own assistant model, already holds most of the card;
    see _gpu_resource_error())."""
    from engine.accounts import box_managed
    needed = _vram_needed_gb(size, resolution, batch=1) + VRAM_MARGIN_GB     # fits_gpu's real threshold
    why = (" Ask Hawk's model and the live inspection also use it." if box_managed()
           else " Close other programs that use the graphics card and try again.")
    return f"The graphics card doesn't have enough free memory to train now (needs about {needed:.1f} GB, {free_gb:.1f} GB free).{why}"


def _gpu_resource_error(e: BaseException) -> bool:
    """A clean out-of-memory error, or the vague WDDM "CUDA error: unknown error" that means the same thing
    under Windows' driver model when the card is shared with another process."""
    if out_of_memory(e):
        return True
    while e is not None:
        s = str(e).lower()
        if "cuda error" in s and "unknown error" in s:
            return True
        e = e.__cause__ or e.__context__
    return False


def _workers() -> int:
    """DataLoader worker processes. Windows starts workers by re-importing the main script, which fails when
    the caller has none on disk (python - < script): then load data in the main process.

    Linux workers are forked, not spawned - and by the time Teach builds the DataLoader, the RF-DETR
    model has already been moved onto the GPU in this process, so CUDA already has a context open.
    Forking a process with an open CUDA context routinely deadlocks the children (confirmed directly:
    a real Teach run on a Linux cloud GPU hung with every worker sitting at 0% CPU indefinitely,
    while the exact same code path is fine on Windows, where multiprocessing uses spawn instead of
    fork - so this had never been exercised before). 0 workers loads data in the main process, same
    as the Windows fallback above; the real fix is spawning workers before CUDA initializes (or with
    multiprocessing_context="spawn"), which needs its own verification before relying on it here."""
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    if main and not Path(main).exists():
        return 0
    return 2 if sys.platform == "win32" else 0


def _model_class(size: str):
    if size not in SIZES:
        raise ValueError(f"unknown model size {size!r}; choose one of {', '.join(SIZES)}")
    import rfdetr
    return getattr(rfdetr, SIZES[size])


def _hooks(epochs: int, progress, should_stop):
    """PyTorch Lightning callback: progress every few batches, in the person's words (time left, never epochs or
    scores: those are in the report), and a clean stop."""
    import time
    from pytorch_lightning import Callback
    t0 = time.time()

    class Hooks(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if should_stop and should_stop():
                trainer.should_stop = True
            nb = max(1, int(trainer.num_training_batches))
            if progress and (batch_idx % 5 == 0 or batch_idx + 1 == nb):
                done, total = trainer.current_epoch * nb + batch_idx + 1, epochs * nb
                left = (time.time() - t0) * (total - done) / done / 60
                progress(done, total, "Training…" + (f" about {max(1, round(left))} min left" if done / total > 0.02 else ""))
    return Hooks()


def best_checkpoint(out_dir) -> Path | None:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "last_ema.pth"):
        if (Path(out_dir) / name).exists():
            return Path(out_dir) / name
    return None


def train(dataset_dir, out_dir, size: str = "small", epochs: int = EPOCHS, resolution: int = 640, aug: str = "strong",
          resume=None, progress=None, should_stop=None) -> Path | None:
    """Fine-tune RF-DETR on a YOLO dataset (train/ and valid/ splits + data.yaml). Batch size follows the GPU
    memory free right now; with too little free it trains on the CPU - except when this computer is embedded
    in another product (box_managed(), e.g. a Hawk-AI box: CPU training there would be too slow to be a real
    option), where it refuses up front with vram_shortfall_message() instead. If the GPU still runs out of
    memory (the estimate comes from an RTX 3060; a T4 without bf16 needed about 3x as much) or hits the same
    thing under a vaguer name (_gpu_resource_error(): WDDM's "CUDA error: unknown error"), the batch is halved
    and training starts again, with grad_accum doubled so the effective batch stays 16 - until batch 1 still
    doesn't fit, where it gives the same plain message rather than the raw error. Returns the best checkpoint,
    or None when stopped early by should_stop."""
    check_resolution(resolution, size)
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    classes = read_classes(dataset_dir / "data.yaml")
    n_train = sum(p.suffix.lower() in IMAGE_EXTS for p in (dataset_dir / "train" / "images").iterdir())
    device, free = hw.device(), hw.free_memory_gb()
    if free is not None and not fits_gpu(size, resolution, free):
        from engine.accounts import box_managed
        if box_managed():                      # CPU training is too slow to be a real option on a box
            raise RuntimeError(vram_shortfall_message(size, resolution, free))
        if progress:
            progress(0, epochs, "Not enough graphics card memory free: training on the processor (slow)")
        device, free = "cpu", None
    batch, accum = batch_plan(size, resolution, free, n_train)
    if device == "cuda" and os.environ.get("PARTLABELER_BATCH"):   # e.g. a big cloud GPU: start high, halve on OOM
        batch = max(1, min(int(os.environ["PARTLABELER_BATCH"]), n_train))
        accum = max(1, EFFECTIVE_BATCH // batch)
    import rfdetr.training as rt
    boxes_through_holes()
    hooks, build = _hooks(epochs, progress, should_stop), rt.build_trainer

    def build_with_hooks(*a, **k):
        trainer = build(*a, **k)
        trainer.callbacks.append(hooks)
        return trainer

    while True:
        model = _model_class(size)(device=device)
        rt.build_trainer = build_with_hooks      # rfdetr's train() takes no extra callbacks
        try:
            model.train(dataset_dir=str(dataset_dir), output_dir=str(out_dir), epochs=epochs, resolution=resolution,
                        batch_size=batch, grad_accum_steps=accum, lr=LR, lr_encoder=LR_ENCODER,
                        aug_config=aug_config(classes, aug), early_stopping=True, early_stopping_patience=PATIENCE,
                        checkpoint_interval=max(2, epochs),
                        tensorboard=False, progress_bar=None, num_workers=_workers(), device=device,
                        resume=str(resume) if resume else None,
                        # held-out check every N epochs (the last one always); outline models score outlines at
                        # the mask head's size there: full-size scoring took longer than the training epoch itself
                        eval_interval=int(os.environ.get("PARTLABELER_EVAL_EVERY", 1)),
                        **({"eval_masks_head_resolution": True} if is_seg(size) else {}))
            break
        except Exception as e:
            if device != "cuda" or not _gpu_resource_error(e):
                raise
            if batch == 1:
                raise RuntimeError(vram_shortfall_message(size, resolution, hw.free_memory_gb() or 0.0)) from e
            batch, accum = batch // 2, accum * 2
            if progress:
                progress(0, epochs, "The graphics card ran out of memory: starting again in smaller steps")
        finally:
            rt.build_trainer = build
            del model
            hw.free_gpu_memory()
    if should_stop and should_stop():
        return None
    return best_checkpoint(out_dir)


def out_of_memory(e: BaseException) -> bool:
    """A CUDA out-of-memory error, also when a training framework wrapped it in another exception."""
    while e is not None:
        if "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError":
            return True
        e = e.__cause__ or e.__context__
    return False


def load(checkpoint, size: str = "small", resolution: int = 640):
    """A trained RF-DETR ready for predict()."""
    check_resolution(resolution, size)
    return _model_class(size)(pretrain_weights=str(checkpoint), resolution=resolution, device=hw.device())


def predict(model, images, threshold: float = 0.5, batch: int = 8):
    """supervision Detections per image (boxes in that image's pixels); one Detections for a single image."""
    single = not isinstance(images, (list, tuple))
    images = [images] if single else list(images)
    out = []
    for i in range(0, len(images), batch):
        r = model.predict(images[i:i + batch], threshold=threshold, include_source_image=False)
        out.extend(r if isinstance(r, list) else [r])
    return out[0] if single else out


# ---- ONNX for production (ONNX Runtime, or a TensorRT engine built on the target machine) ----------------------
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def onnx_input(image, size: int):
    """The ONNX input for one RGB image: [1, 3, size, size] float32, stretched (no letterbox) with bilinear
    half-pixel resizing and no antialias, scaled to 0..1, normalised with ImageNet mean and std (as predict())."""
    import numpy as np
    import torch
    t = torch.from_numpy(np.asarray(image.convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None]
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False, antialias=False)
    return ((t - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)).numpy()


def onnx_decode(dets, labels, width: int, height: int, threshold: float):
    """(xyxy pixel boxes, scores, class ids, query ids) from one image's outputs: dets [Q, 4] cx, cy, w, h (0..1),
    labels [Q, C + 1] logits whose last column is background. Each (query, class) pair is scored on its own
    (sigmoid); the Q best pairs are kept when their score is above the threshold."""
    import numpy as np
    scores = 1 / (1 + np.exp(-labels[:, :-1].clip(-88, 88)))
    C = scores.shape[1]
    top = np.argsort(-scores.ravel(), kind="stable")[:len(dets)]
    top = top[scores.ravel()[top] > threshold]
    q, cls = top // C, top % C
    cx, cy, w, h = dets[q].T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * [width, height, width, height]
    return xyxy.clip(0, [width, height, width, height]), scores.ravel()[top], cls, q


def onnx_masks(masks, q, width: int, height: int):
    """Part pixels [K, height, width] of the kept pairs (outline models): each pair's query plane of masks
    [Q, Hm, Wm] (logits at a quarter of the input size) resized to the frame, bilinear with half-pixel centres, > 0."""
    import cv2
    import numpy as np
    if not len(q):
        return np.zeros((0, height, width), bool)
    return np.stack([cv2.resize(masks[i].astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) > 0
                     for i in q])


def crop_card(task_cameras: dict | None) -> dict | None:
    """model.json's "crop" field from a Teach run's settings.json "task_cameras" ({task name: {"camera_id",
    "crop"}}): one rectangle per camera, shares 0..1, written only when every task of that camera used the
    same one (or none did); None when no task carried a camera_id at all. Raises ValueError, naming the
    camera, when its tasks were cut differently (some cropped, some not, or to different rectangles) -
    PartLabeler's "their call" refusal for a mixed-crop run (hawk-ai-internal §13.3)."""
    if not task_cameras:
        return None
    by_cam = defaultdict(set)
    for info in task_cameras.values():
        cid = info.get("camera_id")
        if cid:
            by_cam[cid].add(tuple(info["crop"]) if info.get("crop") else None)
    crop = {}
    for cid, vals in by_cam.items():
        if len(vals) > 1:
            raise ValueError(f"camera {cid!r} was cut differently across its training tasks (some cropped, some "
                             "not, or to different rectangles): crop every task of a camera the same way before "
                             "exporting, or leave that camera's tasks all uncropped")
        v = next(iter(vals))
        if v is not None:
            crop[cid] = list(v)
    return crop or None


def export_onnx(run_dir, out_dir=None, check_frames: int = 40, bundle_frames: int = 20, batch: int = 1) -> dict:
    """A Teach run's model as ONNX, with model.json (classes, threshold, input/output contract) and a parity check:
    onnx_input/onnx_decode through ONNX Runtime against predict() on held-out frames. The folder is a bundle for the
    production machine: build_engine.py builds and checks a TensorRT engine there against check/ (a few held-out
    frames, every class if possible, with the trained model's surest box and score per class). batch: a fixed batch
    size (e.g. 2 for two cameras seen at the same moment); the parity check then runs different frames together."""
    import json
    import shutil
    import numpy as np
    from PIL import Image
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("The tools that prepare a model for production are missing: update PartLabeler, or run "
                           "its installer again") from None     # the teach extra's rfdetr[onnx] + onnxconverter-common
    ort.disable_telemetry_events()                   # official Windows builds send usage telemetry (ETW) by default
    run_dir = Path(run_dir)
    s = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    out_dir = Path(out_dir or run_dir / "export")
    model = load(run_dir / s["checkpoint"], s["size"], s["resolution"])
    path = Path(model.export(output_dir=str(out_dir), format="onnx", verbose=False, batch_size=batch))
    fp16 = path.with_suffix(".fp16.onnx")                   # TensorRT 11+ takes precision from the graph
    try:
        from rfdetr.export._tensorrt.exporter import fp16_source_graph
        with fp16_source_graph(str(path)) as cast:          # fp16 inside, FP32 inputs and outputs kept
            shutil.copyfile(cast, fp16)
    except ImportError:                                     # onnxconverter-common missing: FP32 route only
        fp16.unlink(missing_ok=True)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    valid, src = run_dir / "dataset" / "valid" / "images", run_dir / "source"
    if valid.is_dir():
        files = sorted(p for p in valid.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:                                                   # trained elsewhere: the labeled frames of its source
        labeled = lambda p: (lab := src / "labels" / f"{p.stem}.txt").is_file() and lab.stat().st_size
        files = sorted(p for p in (src / "images").iterdir()
                       if p.suffix.lower() in IMAGE_EXTS and labeled(p)) if (src / "images").is_dir() else []
    files = files[::max(1, len(files) // check_frames)][:check_frames]
    outline = "masks" in [o.name for o in sess.get_outputs()]
    boxes = missing = 0
    worst, mask_ious = 0.0, []
    expected, masks_of = {}, {}                              # masks_of: (file, part) -> the trained model's pixels
    names = [o.name for o in sess.get_outputs()]
    runs = {}                                                # file -> its ONNX outputs, frames run `batch` at a time
    for k in range(0, len(files), batch):
        chunk = files[k:k + batch]
        ims = [Image.open(f).convert("RGB") for f in chunk]
        x = np.concatenate([onnx_input(im, s["resolution"]) for im in ims + ims[-1:] * (batch - len(ims))])
        outs = dict(zip(names, sess.run(None, {"input": x})))
        runs.update({f: {n: v[j] for n, v in outs.items()} for j, f in enumerate(chunk)})
    for f in files:
        im = Image.open(f).convert("RGB")
        ref = predict(model, im, s["threshold"])
        best = {}
        for k, (box, c, p) in enumerate(zip(ref.xyxy, ref.class_id, ref.confidence)):
            if s["classes"][c] not in best or p > best[s["classes"][c]][1]:
                best[s["classes"][c]] = ([round(float(v), 2) for v in box], p, k)
        expected[f] = {n: {"box": b, "score": round(float(p), 4)} for n, (b, p, _) in best.items()}
        out = runs[f]
        xyxy, _, cls, q = onnx_decode(out["dets"], out["labels"], *im.size, s["threshold"])
        got = onnx_masks(out["masks"], q, *im.size) if outline else None
        for n, (b, p, k) in best.items():
            boxes += 1
            same = np.flatnonzero(cls == s["classes"].index(n))
            if not len(same):
                missing += 1
                continue
            j = same[0]                                      # the surest pair of the class
            worst = max(worst, float(np.abs(xyxy[j] - ref.xyxy[k]).max()))
            if outline:
                a, t = got[j], ref.mask[k]
                mask_ious.append(float((a & t).sum() / max(1, (a | t).sum())))
                masks_of[(f, n)] = t
    del model
    hw.free_gpu_memory()
    n, r = len(s["classes"]), s["resolution"]
    Q = {o.name: o.shape for o in sess.get_outputs()}["dets"][1]     # queries: 300 (detect) or 100 (seg-small)
    card = {"model": path.name, "model_fp16": fp16.name if fp16.exists() else None, "classes": s["classes"], "threshold": s["threshold"], "resolution": r,
            "parent": s.get("parent"), "parent_margin": s.get("parent_margin") if s.get("parent") else None,
            "crop": crop_card(s.get("task_cameras")),
            "batch": batch,
            "input": {"name": "input", "shape": [batch, 3, r, r], "dtype": "float32", "layout": "NCHW, RGB",
                      "prepare": (f"the whole frame{' cut to the parent object first' if s.get('parent') else ''}, "
                                  f"stretched to {r}x{r} (no letterbox), bilinear with half-pixel centres and no "
                                  "antialias; pixel / 255; minus mean, divided by std per channel. OpenCV: "
                                  f"cv2.resize(rgb.astype(float32) / 255, ({r}, {r}), interpolation=cv2.INTER_LINEAR)"),
                      "mean": list(MEAN), "std": list(STD)},
            "outputs": {"dets": f"[{batch}, {Q}, 4] box centre x, y, width, height as a share (0..1) of the input",
                        "labels": f"[{batch}, {Q}, {n + 1}] logits; column k is classes[k], the last column is background"}
                       | ({"masks": f"[{batch}, {Q}, {r // 4}, {r // 4}] outline logits per query"} if outline else {}),
            "decode": (f"sigmoid each class column (not softmax); of all {Q} x classes (query, class) pairs keep the "
                       f"{Q} highest, then those above the threshold; box = centre -/+ size / 2, times the frame's "
                       "width and height" + ("; outline = that pair's query plane of masks resized to the frame "
                                             "(bilinear, half-pixel centres), pixels > 0" if outline else "")),
            "outlines": outline,
            "tensorrt": ("on the production machine, in this folder: python build_engine.py (FP16 engine at "
                         "optimization level 5 from the FP16 model, strongly typed (TensorRT 10 uses the FP32 model "
                         "with --fp16 when it is there); then a check on the check/ frames)"),
            "runtime": (f"{'one frame' if batch == 1 else f'{batch} frames of the same moment (one per camera)'} "
                        "per run (a fixed batch: pad a missing frame with a copy of another); keep the input and "
                        "output buffers and the CUDA stream alive between runs, and prepare frames on the GPU"),
            "parity": {"frames": len(files), "boxes": boxes, "missing": missing, "worst_px": round(worst, 3)}
                      | ({"outline_iou_min": round(min(mask_ious), 4), "outline_iou_mean": round(float(np.mean(mask_ious)), 4)}
                         if mask_ious else {})}
    (out_dir / "model.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    check = out_dir / "check"
    shutil.rmtree(check, ignore_errors=True)
    check.mkdir()
    chosen, seen = [], set()
    for _ in range(min(bundle_frames, len(expected))):                 # each next frame: most classes not seen yet
        f = max((f for f in expected if f not in chosen), key=lambda f: (len(set(expected[f]) - seen), len(expected[f])))
        chosen.append(f)
        seen |= set(expected[f])
    for f in chosen:
        shutil.copyfile(f, check / f.name)
        for part, e in expected[f].items():                  # outline models: the trained model's pixels
            if (f, part) in masks_of:
                e["mask"] = f"{f.stem}__{part}.png"
                Image.fromarray(masks_of[(f, part)].astype(np.uint8) * 255).save(check / e["mask"])
    (check / "expected.json").write_text(json.dumps({f.name: expected[f] for f in chosen}, indent=1), encoding="utf-8")
    for tool in ("build_engine.py", "partcheck.py", "lampcheck.py"):   # the production side, standalone
        shutil.copyfile(Path(__file__).with_name(tool), out_dir / tool)
    if (run_dir / "report.md").is_file():                   # the training report travels with the model
        shutil.copyfile(run_dir / "report.md", out_dir / "report.md")
    return card | {"path": str(path)}
