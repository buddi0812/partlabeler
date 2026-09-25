"""RF-DETR detector: fine-tune on a YOLO dataset, load a checkpoint, predict boxes.

Used by Teach & Transfer (and later by Boost). Augmentation is colour-heavy so a model taught on one
colour of a product still finds the parts on other colours. Horizontal flip is added only when the
class names have no left_/right_ pairs: a mirror would swap their meaning.
"""
import sys
from pathlib import Path

from engine import hw
from engine.project import IMAGE_EXTS, mirrored_pairs, read_classes

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}
BLOCK = 32                     # patch 16 x 2 windows for all three sizes: resolution must be a multiple
EFFECTIVE_BATCH = 16           # batch x grad_accum, as in S7 (4 x 4)
_GB_PER_IMAGE = {"nano": 0.7, "small": 1.1, "medium": 1.6}   # training memory at 640 px (S7: small, batch 4 = 4.4 GB)
_GB_BASE = 2.0

STRONG = {
    "HueSaturationValue": {"hue_shift_limit": 180, "sat_shift_limit": 50, "val_shift_limit": 40, "p": 0.7},
    "RandomBrightnessContrast": {"brightness_limit": 0.3, "contrast_limit": 0.3, "p": 0.6},
    "ToGray": {"p": 0.1},
    "GaussianBlur": {"blur_limit": 3, "p": 0.2},
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.2},
    "Affine": {"scale": (0.85, 1.15), "translate_percent": (-0.05, 0.05), "p": 0.4},
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


def check_resolution(resolution: int) -> None:
    if resolution <= 0 or resolution % BLOCK:
        raise ValueError(f"resolution {resolution} must be a positive multiple of {BLOCK} (e.g. 320, 480, 640)")


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
    return 0 if main and not Path(main).exists() else 2


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
    memory free right now; with too little free it trains on the CPU. Returns the best checkpoint, or None
    when stopped early by should_stop."""
    check_resolution(resolution)
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    classes = read_classes(dataset_dir / "data.yaml")
    n_train = sum(p.suffix.lower() in IMAGE_EXTS for p in (dataset_dir / "train" / "images").iterdir())
    device, free = hw.device(), hw.free_memory_gb()
    if free is not None and not fits_gpu(size, resolution, free):
        if progress:
            progress(0, epochs, f"Only {free:.1f} GB free on the GPU: training on the CPU (slow)")
        device, free = "cpu", None
    batch, accum = batch_plan(size, resolution, free, n_train)
    import rfdetr.training as rt
    model = _model_class(size)(device=device)
    hooks, build = _hooks(epochs, progress, should_stop), rt.build_trainer

    def build_with_hooks(*a, **k):
        trainer = build(*a, **k)
        trainer.callbacks.append(hooks)
        return trainer

    rt.build_trainer = build_with_hooks          # rfdetr's train() takes no extra callbacks
    try:
        model.train(dataset_dir=str(dataset_dir), output_dir=str(out_dir), epochs=epochs, resolution=resolution,
                    batch_size=batch, grad_accum_steps=accum, lr=1e-4, aug_config=aug_config(classes, aug),
                    early_stopping=True, early_stopping_patience=8, checkpoint_interval=max(2, epochs),
                    tensorboard=False, progress_bar=None, num_workers=_workers(), device=device,
                    resume=str(resume) if resume else None)
    finally:
        rt.build_trainer = build
        del model
        hw.free_gpu_memory()
    if should_stop and should_stop():
        return None
    return best_checkpoint(out_dir)


def load(checkpoint, size: str = "small", resolution: int = 640):
    """A trained RF-DETR ready for predict()."""
    check_resolution(resolution)
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
