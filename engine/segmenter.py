"""Click / box -> mask on one image with SAM 3 (Sam3TrackerModel).

The image is encoded once in `set_image` (the expensive part); each prompt afterwards only
runs the prompt encoder and mask decoder, which keeps interactive clicks fast.
"""
import numpy as np
import torch
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

from engine import hw
from engine.models import SAM3, fetch


class Segmenter:
    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None):
        device = device or hw.device(); dtype = dtype or hw.dtype(device)
        path = fetch(SAM3)
        self.model = Sam3TrackerModel.from_pretrained(path, dtype=dtype).to(device).eval()
        self.proc = Sam3TrackerProcessor.from_pretrained(path)
        self.device, self.dtype = device, dtype
        self.embeddings = self.sizes = None

    @torch.inference_mode()
    def set_image(self, image) -> None:
        inputs = self.proc(images=image, return_tensors="pt").to(device=self.device, dtype=self.dtype)
        self.embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        self.sizes = inputs["original_sizes"].tolist()

    def segment(self, points=(), labels=(), box=None) -> tuple[np.ndarray, float]:
        """points: [[x, y], ...] in image pixels; labels: 1 = part of the object, 0 = not.
        box: [x1, y1, x2, y2] or None. Returns (H x W bool mask, predicted IoU score)."""
        return self._run(points, labels, box, multimask=False)[0]

    def candidates(self, points=(), labels=(), box=None) -> list[tuple[np.ndarray, float]]:
        """The three nested masks SAM proposes for a prompt (e.g. part / sub-part / whole object)."""
        return self._run(points, labels, box, multimask=True)

    def smart_edit(self, old: np.ndarray, stroke, radius: float, erase: bool = False) -> np.ndarray:
        """The smart brush (erase=False) or smart eraser on the image set last: only the pixels a round brush of
        `radius` covers along `stroke` can change, and only where SAM 3 agrees. Brush: SAM outlines what is under
        the stroke together with the part (points deep inside `old`); the footprint inside that outline is added.
        Eraser: SAM outlines what is under the stroke, told the part's inside is not it; the footprint inside that
        outline is removed. Returns the new mask (specks and pinholes cleaned)."""
        from engine import masks as M
        foot = M.footprint(stroke, radius, old.shape)
        along, inside = M.along(stroke), M.inner_points(old & ~foot)
        seen = self.segment(along + inside, [1] * len(along) + [0 if erase else 1] * len(inside))[0]
        change = foot & seen
        return M.clean(old & ~change if erase else old | change)

    @torch.inference_mode()
    def _run(self, points, labels, box, multimask: bool):
        prompt = {}
        if points:
            prompt.update(input_points=[[list(points)]], input_labels=[[list(labels)]])
        if box:
            prompt["input_boxes"] = [[list(box)]]
        enc = self.proc(original_sizes=self.sizes, return_tensors="pt", **prompt)
        inputs = {k: v.to(self.device, self.dtype if v.is_floating_point() else None)
                  for k, v in enc.items() if k != "original_sizes"}
        out = self.model(**inputs, image_embeddings=self.embeddings, multimask_output=multimask)
        masks = self.proc.post_process_masks(out.pred_masks.float().cpu(), self.sizes)[0][0]
        return [(m.numpy().astype(bool), float(s)) for m, s in zip(masks, out.iou_scores[0, 0])]
