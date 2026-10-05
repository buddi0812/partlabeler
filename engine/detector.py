"""RF-DETR detector: fine-tune on a YOLO dataset, load a checkpoint, predict boxes.

Used by Teach & Transfer (and later by Boost). Augmentation is colour-heavy so a model taught on one
colour of a product still finds the parts on other colours, with small rotation and shear for a camera that
sits a little differently. Horizontal flip is added only when the
class names have no left/right pairs: a mirror would swap their meaning.
"""
import os
import sys
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

STRONG = {
    "HueSaturationValue": {"hue_shift_limit": 180, "sat_shift_limit": 50, "val_shift_limit": 40, "p": 0.7},
    "RandomBrightnessContrast": {"brightness_limit": 0.3, "contrast_limit": 0.3, "p": 0.6},
    "ToGray": {"p": 0.1},
    "GaussianBlur": {"blur_limit": 3, "p": 0.2},
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.2},
    "Affine": {"scale": (0.85, 1.15), "translate_percent": (-0.05, 0.05), "rotate": (-7, 7), "shear": (-5, 5), "p": 0.5},
}
MEDIUM = {
    "HueSaturationValue": {"hue_shift_limit": 30, "sat_shift_limit": 30, "val_shift_limit": 30, "p": 0.5},
    "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5},
    "GaussianBlur": {"blur_limit": 3, "p": 0.1},
    "Affine": {"scale": (0.9, 1.1), "translate_percent": (-0.03, 0.03), "p": 0.3},
}
AUGS = ("strong", "medium", "off")


def aug_config(classes, strength: str = "strong") -> dict:
    """Albumentations config for rfdetr. rfdetr's own default includes a flip, so one is always passed;
    "off" is {} (no augmentation, no flip)."""
    if strength not in AUGS:
        raise ValueError(f"unknown augmentation {strength!r}; choose one of {', '.join(AUGS)}")
    if strength == "off":
        return {}
    aug = dict(STRONG if strength == "strong" else MEDIUM)
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


def batch_plan(size: str, resolution: int, free_gb: float | None, n_train: int | None = None) -> tuple[int, int]:
    """(batch, grad_accum) keeping batch x accum = 16: the largest power-of-two batch that fits in free_gb
    of GPU memory, at most the number of training images. free_gb None (CPU): batch 2."""
    per = _GB_PER_IMAGE[size] * (resolution / 640) ** 2
    batch = 2 if free_gb is None else 1
    while free_gb is not None and batch < EFFECTIVE_BATCH and _GB_BASE + 2 * batch * per <= free_gb:
        batch *= 2
    if n_train:
        batch = max(1, min(batch, n_train))
    return batch, max(1, EFFECTIVE_BATCH // batch)


def fits_gpu(size: str, resolution: int, free_gb: float) -> bool:
    return _GB_BASE + _GB_PER_IMAGE[size] * (resolution / 640) ** 2 <= free_gb


def _workers() -> int:
    """DataLoader worker processes. Windows starts workers by re-importing the main script, which fails when
    the caller has none on disk (python - < script): then load data in the main process."""
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    if main and not Path(main).exists():
        return 0
    return 2 if sys.platform == "win32" else min(8, max(2, (os.cpu_count() or 2) // 2))   # Linux: e.g. a cloud GPU


def _model_class(size: str):
    if size not in SIZES:
        raise ValueError(f"unknown model size {size!r}; choose one of {', '.join(SIZES)}")
    import rfdetr
    return getattr(rfdetr, SIZES[size])


def _hooks(epochs: int, progress, should_stop):
    """PyTorch Lightning callback: progress every few batches (with the last held-out mAP50) and a clean stop."""
    from pytorch_lightning import Callback

    class Hooks(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if should_stop and should_stop():
                trainer.should_stop = True
            nb = max(1, int(trainer.num_training_batches))
            if progress and (batch_idx % 5 == 0 or batch_idx + 1 == nb):
                m = trainer.callback_metrics.get("val/mAP_50")
                progress(trainer.current_epoch * nb + batch_idx + 1, epochs * nb,
                         f"Training epoch {trainer.current_epoch + 1}/{epochs}"
                         + (f" (held-out mAP50 {float(m):.3f})" if m is not None else ""))
    return Hooks()


def best_checkpoint(out_dir) -> Path | None:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "last_ema.pth"):
        if (Path(out_dir) / name).exists():
            return Path(out_dir) / name
    return None


def train(dataset_dir, out_dir, size: str = "small", epochs: int = 30, resolution: int = 640, aug: str = "strong",
          resume=None, progress=None, should_stop=None) -> Path | None:
    """Fine-tune RF-DETR on a YOLO dataset (train/ and valid/ splits + data.yaml). Batch size follows the GPU
    memory free right now; with too little free it trains on the CPU. If the GPU still runs out of memory
    (the estimate comes from an RTX 3060; a T4 without bf16 needed about 3x as much), the batch is halved and
    training starts again, with grad_accum doubled so the effective batch stays 16. Returns the best
    checkpoint, or None when stopped early by should_stop."""
    check_resolution(resolution, size)
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    classes = read_classes(dataset_dir / "data.yaml")
    n_train = sum(p.suffix.lower() in IMAGE_EXTS for p in (dataset_dir / "train" / "images").iterdir())
    device, free = hw.device(), hw.free_memory_gb()
    if free is not None and not fits_gpu(size, resolution, free):
        if progress:
            progress(0, epochs, f"Only {free:.1f} GB free on the GPU: training on the CPU (slow)")
        device, free = "cpu", None
    batch, accum = batch_plan(size, resolution, free, n_train)
    if device == "cuda" and os.environ.get("PARTLABELER_BATCH"):   # e.g. a big cloud GPU: start high, halve on OOM
        batch = max(1, min(int(os.environ["PARTLABELER_BATCH"]), n_train))
        accum = max(1, EFFECTIVE_BATCH // batch)
    import rfdetr.training as rt
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
                        batch_size=batch, grad_accum_steps=accum, lr=1e-4, aug_config=aug_config(classes, aug),
                        early_stopping=True, early_stopping_patience=8, checkpoint_interval=max(2, epochs),
                        tensorboard=False, progress_bar=None, num_workers=_workers(), device=device,
                        resume=str(resume) if resume else None,
                        # held-out check every N epochs (the last one always); outline models score outlines at
                        # the mask head's size there: full-size scoring took longer than the training epoch itself
                        eval_interval=int(os.environ.get("PARTLABELER_EVAL_EVERY", 1)),
                        **({"eval_masks_head_resolution": True} if is_seg(size) else {}))
            break
        except Exception as e:
            if device != "cuda" or batch == 1 or not out_of_memory(e):
                raise
            batch, accum = batch // 2, accum * 2
            if progress:
                progress(0, epochs, f"Out of GPU memory: starting again with batch {batch} (x{accum} accumulation)")
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
        raise RuntimeError('ONNX export needs the ONNX tools: pip install "rfdetr[onnx]==1.11.0" onnxconverter-common') from None
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
