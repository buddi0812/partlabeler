"""Teach for image classes (classify projects; hawk-ai-internal app-015-plan, Setup flow v2): a small head on frozen
DINOv3 features of each confirmed image, proven on held-out images, then exported as one ONNX model (backbone + head)
for production's two-stage check: a detector finds the part, this names its state on the part's crop (cut the way
the training images were cut; Hawk-AI: the part's box plus 25 % on each side).

Frozen features + a linear head: minutes, not hours, and it holds up with the few dozen images a station gives
(the features already tell a seated clip from a missing one; the head only learns which is which here)."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from engine import hw

SIZE = 224                       # the model's input: the crop stretched to SIZE x SIZE
HELD_OUT = 0.2                   # share of each class's images kept back to prove the model
TARGET_ACCURACY = 0.90           # held-out accuracy, and each class's recall, to pass (Teach's recall target)
STEPS = 300


class Classifier(torch.nn.Module):
    """RGB 0..1 [N, 3, SIZE, SIZE] -> class probabilities [N, classes]; normalisation inside, so production only
    resizes and divides by 255."""

    def __init__(self, backbone, mean, std, head):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return F.softmax(self.head(self.backbone((x - self.mean) / self.std)), dim=1)


def _tensor(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)


def _backbone(device):
    from engine.embed import Embedder
    e = Embedder("small", device=device)
    return e.model, e.mean.flatten().tolist(), e.std.flatten().tolist()


def split(labels: list[int], held_out: float = HELD_OUT, seed: int = 0) -> tuple[list[int], list[int]]:
    """(train, held-out) indices: a share of each class held out, at least one when a class has two or more."""
    rng, train, test = random.Random(seed), [], []
    for c in sorted(set(labels)):
        idx = [i for i, x in enumerate(labels) if x == c]
        rng.shuffle(idx)
        k = min(len(idx) - 1, max(1, round(held_out * len(idx)))) if len(idx) > 1 else 0
        test += idx[:k]
        train += idx[k:]
    return sorted(train), sorted(test)


def train(p, run_dir, project: str | None = None, progress=None, should_stop=lambda: False) -> dict:
    """A classify project's confirmed images with their class -> run_dir/{settings,report}.json and run_dir/export/
    (model.onnx + model.json, as GET /api/v1/models lists it). Returns the report."""
    say = progress or (lambda *a, **k: None)
    run_dir = Path(run_dir)
    tags, st = p.tags(), p.statuses()
    items = [k for k, s in enumerate(st) if s == 4 and tags.get(k, {}).get("cls") is not None]
    labels = [tags[k]["cls"] for k in items]
    counts = {c: labels.count(c) for c in set(labels)}
    if len(counts) < 2 or min(counts.values()) < 2:
        raise ValueError("needs at least two classes with two or more confirmed images each "
                         f"(now: {', '.join(f'{p.classes[c]} {n}' for c, n in sorted(counts.items())) or 'none'})")
    device = hw.device()
    backbone, mean, std = _backbone(device)
    tr, te = split(labels)
    say(0, len(items) + STEPS, "Reading the images")
    feats = []
    with torch.inference_mode():
        for n, k in enumerate(items):
            if should_stop():
                return {"status": "stopped"}
            x = _tensor(p.image(k))
            batch = torch.stack([x, x.flip(2)]).to(device)              # the image and its mirror (more to learn from)
            feats.append(backbone((batch - torch.tensor(mean, device=device).view(1, 3, 1, 1))
                                  / torch.tensor(std, device=device).view(1, 3, 1, 1)).float().cpu())
            say(n + 1, len(items) + STEPS, "Reading the images")
    n_cls = len(p.classes)
    head = torch.nn.Linear(feats[0].shape[1], n_cls)
    xs = torch.cat([feats[i] for i in tr])
    ys = torch.tensor([labels[i] for i in tr for _ in (0, 1)])
    w = torch.tensor([1 / max(1, (ys == c).sum().item()) for c in range(n_cls)])     # each class counts the same
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    for step in range(STEPS):
        if should_stop():
            return {"status": "stopped"}
        loss = F.cross_entropy(head(xs), ys, weight=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            say(len(items) + step, len(items) + STEPS, "Learning the classes")
    with torch.no_grad():
        pred = [int(head(feats[i][:1]).argmax()) for i in te]
    truth = [labels[i] for i in te]
    per_class = {p.classes[c]: {"images": counts[c], "held_out": truth.count(c),
                                "recall": (sum(1 for a, b in zip(pred, truth) if b == c and a == c) / truth.count(c))
                                if truth.count(c) else None} for c in sorted(counts)}
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    accuracy = sum(a == b for a, b in zip(pred, truth)) / len(te) if te else None
    balanced = sum(recalls) / len(recalls) if recalls else None
    warnings = [f"classes with no held-out image, so not measured: {n}" for n, v in per_class.items() if v["recall"] is None]
    passed = bool(te) and balanced is not None and balanced >= TARGET_ACCURACY and min(recalls) >= TARGET_ACCURACY
    settings = {"task": "classify", "size": "classify", "classes": p.classes, "resolution": SIZE, "backbone": "dinov3 small",
                "project": project, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    report = {"status": "done", "run_dir": str(run_dir), "task": "classify", "passed": passed,
              "knowledge_score": round(100 * balanced, 1) if balanced is not None else None,
              "targets": {"balanced_accuracy": {"target": TARGET_ACCURACY, "value": balanced,
                                                "pass": balanced is not None and balanced >= TARGET_ACCURACY}},
              "held_out": {"accuracy": accuracy, "balanced_accuracy": balanced, "per_class": per_class},
              "prepare": {"train_frames": len(tr), "held_out_frames": len(te)}, "warnings": warnings, "settings": settings}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    say(len(items) + STEPS, len(items) + STEPS, "Preparing it for the stations")
    model = Classifier(backbone.float().cpu(), mean, std, head.cpu()).eval()
    report["export"] = export(model, p.classes, run_dir / "export", [p.image(items[i]) for i in te[:20]])
    (run_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    hw.free_gpu_memory()
    return report


def export(model: Classifier, classes: list[str], out_dir, check_images) -> dict:
    """model.onnx + model.json (the input/output contract) and a parity check of ONNX Runtime against PyTorch."""
    import onnxruntime as ort
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.onnx"
    torch.onnx.export(model, torch.rand(1, 3, SIZE, SIZE), str(path), input_names=["input"], output_names=["probs"],
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst, agree = 0.0, 0
    with torch.no_grad():
        for img in check_images:
            x = _tensor(img)[None]
            a, b = model(x).numpy(), sess.run(None, {"input": x.numpy()})[0]
            worst, agree = max(worst, float(np.abs(a - b).max())), agree + int(a.argmax() == b.argmax())
    card = {"model": path.name, "task": "classify", "classes": classes, "resolution": SIZE, "parent": None, "crop": None,
            "threshold": None, "batch": 1,
            "input": {"name": "input", "shape": [1, 3, SIZE, SIZE], "dtype": "float32", "layout": "NCHW, RGB",
                      "prepare": (f"the part's crop, cut as the training images were (Hawk-AI: the part's detected box "
                                  f"plus 25 % on each side, clamped to the frame), stretched to {SIZE}x{SIZE} (no "
                                  "letterbox), bilinear; pixel / 255. Mean and std are applied inside the model.")},
            "outputs": {"probs": f"[1, {len(classes)}] probability of each class; column k is classes[k]"},
            "decode": "the class with the highest probability; that probability is how sure the model is",
            "parity": {"frames": len(check_images), "agree": agree, "worst_prob_diff": round(worst, 6)}}
    (out_dir / "model.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    return card["parity"]
