"""SAM 3 concept search on one image: by text or by example boxes drawn on that image.

- Parent object (optional, per project): the thing the parts sit on ("car", "engine block",
  "circuit board"). Searching inside its crop makes different stations, camera positions and
  distances look alike.
- find_like: "box one, find all": every instance that looks like an example box in the same image.
  (SAM 3 exemplars only work within the image they are drawn on.)
"""
import torch
from transformers import Sam3Model, Sam3Processor

from engine import hw
from engine.models import SAM3, fetch


class ConceptFinder:
    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None):
        device = device or hw.device(); dtype = dtype or hw.dtype(device)
        path = fetch(SAM3)
        self.model = Sam3Model.from_pretrained(path, dtype=dtype).to(device).eval()
        self.proc = Sam3Processor.from_pretrained(path)
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def detect(self, image, text: str | None = None, boxes=None, threshold: float = 0.5):
        """[(box [x1, y1, x2, y2], score), ...] for every instance matching the text and/or example boxes."""
        prompt = {}
        if text:
            prompt["text"] = text
        if boxes:
            prompt["input_boxes"] = [[list(map(float, b)) for b in boxes]]
            prompt["input_boxes_labels"] = [[1] * len(boxes)]
        enc = self.proc(images=image, return_tensors="pt", **prompt)
        sizes = enc["original_sizes"].tolist()
        inputs = {k: v.to(self.device, self.dtype if v.is_floating_point() else None) if torch.is_tensor(v) else v
                  for k, v in enc.items()}
        out = self.model(**inputs)
        res = self.proc.post_process_instance_segmentation(out, threshold=threshold, mask_threshold=0.5,
                                                           target_sizes=sizes)[0]
        return [([float(v) for v in b], float(s)) for b, s in zip(res["boxes"].float().cpu(), res["scores"].float().cpu())]

    def largest(self, image, text: str, threshold: float = 0.5):
        """Box of the main instance of `text`: the largest, which in a fixed inspection camera is the
        object being inspected, not one queued behind it; None if nothing is found."""
        found = self.detect(image, text=text, threshold=threshold)
        if not found:
            return None
        return tuple(max(found, key=lambda bs: area(bs[0]))[0])

    def find_like(self, image, example, threshold: float = 0.3, size_range=(1 / 3, 3.0), aspect_range=2.0):
        """Other instances that look like the `example` box in this image, best first. Hits much
        bigger or smaller than the example or with a very different shape are dropped (one example of
        a thin trim piece otherwise matches whole panels), as are hits mostly covering or covered by
        the example or a better hit (other extents of the same part).
        Exemplar scores run low for small parts (a second roof rail: 0.44), hence the 0.3 default."""
        a0, r0 = area(example), aspect(example)
        out = []
        for box, score in sorted(self.detect(image, boxes=[example], threshold=threshold), key=lambda h: -h[1]):
            if overlap(box, example) > 0.5 or any(overlap(box, b) > 0.5 for b, _ in out):
                continue
            if size_range[0] <= area(box) / a0 <= size_range[1] and 1 / aspect_range <= aspect(box) / r0 <= aspect_range:
                out.append((box, score))
        return out


class ParentFinder(ConceptFinder):
    def __init__(self, concept: str, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__(device, dtype)
        self.concept = concept

    def find(self, image, threshold: float = 0.5):
        return self.largest(image, self.concept, threshold)


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def aspect(b) -> float:
    return max(1e-6, b[2] - b[0]) / max(1e-6, b[3] - b[1])


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (area(a) + area(b) - inter + 1e-9)


def crop_box(box, width: int, height: int, margin: float = 0.12):
    """Parent box grown by `margin` of its size on every side, clipped to the image, as ints."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(width, int(x2 + mx)), min(height, int(y2 + my)))


def overlap(a, b) -> float:
    """Intersection over the smaller box: 1.0 when one box sits inside the other."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / (min(area(a), area(b)) + 1e-9)
