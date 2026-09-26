"""Track labeled parts forward or backward through a project's frames with SAM 3 (HF Sam3TrackerVideoModel).

A run starts at one item and covers the next `count` items. Every box on the start item is a
prompt. Each object's most recent earlier manual box is prepended as an extra reference frame,
so what a person drew keeps steering the tracker (S2: reference frames lifted IoU 0.969 -> 0.986).
Runs are split into chunks; each chunk is a fresh, fully released session seeded with the previous
chunk's last boxes (long sessions grow memory without bound).
"""
import gc

import numpy as np
import torch
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from engine import hw
from engine.models import SAM3, fetch

CHUNK = 120          # items per session (halved automatically when the GPU runs out of memory)
MIN_CHUNK = 8
MIN_AREA = 16        # mask pixels; anything smaller counts as "part not visible"


def mask_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_AREA:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def style_offset(drawn, outline):
    """How a person's box sits around SAM's outline box, as fractions of the outline's size."""
    w, h = outline[2] - outline[0], outline[3] - outline[1]
    return [(drawn[0] - outline[0]) / w, (drawn[1] - outline[1]) / h,
            (drawn[2] - outline[2]) / w, (drawn[3] - outline[3]) / h]


def apply_style(outline, offset):
    w, h = outline[2] - outline[0], outline[3] - outline[1]
    return [outline[0] + offset[0] * w, outline[1] + offset[1] * h,
            outline[2] + offset[2] * w, outline[3] + offset[3] * h]


class Tracker:
    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None):
        device = device or hw.device(); dtype = dtype or hw.dtype(device)
        path = fetch(SAM3)
        self.model = Sam3TrackerVideoModel.from_pretrained(path, dtype=dtype).to(device).eval()
        self.proc = Sam3TrackerVideoProcessor.from_pretrained(path)
        self.device, self.dtype, self.chunk = device, dtype, CHUNK

    def _add(self, sess, local: int, boxes: dict) -> None:
        objs = sorted(boxes)
        self.proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=local, obj_ids=objs,
                                                  input_boxes=[[list(map(float, boxes[o])) for o in objs]])

    @torch.inference_mode()
    def track(self, project, start: int, count: int, on_item=None, should_stop=lambda: False,
              direction: int = 1, masks: bool = False) -> int:
        """Track the boxes on item `start` through the next `count` items (direction 1) or the previous
        `count` items (direction -1). Calls on_item(item, {obj: (cls, box or None, score)}) per item, with the
        part's mask as a 4th value when `masks` (outline projects; the box is then the mask's own box);
        returns the items processed. Out of GPU memory, the chunk is halved and retried."""
        seeds = {b["obj"]: b for b in project.boxes(start) if b["source"] != "suggested"}
        if not seeds:
            return 0
        classes = {o: b["cls"] for o, b in seeds.items()}
        refs = {}                                              # item -> {obj: box}: nearest manual box behind start
        for o in seeds:
            behind = [a for a in project.anchors(o) if (a - start) * direction < 0]
            if behind:
                a = behind[-1] if direction > 0 else behind[0]
                refs.setdefault(a, {})[o] = next(b["box"] for b in project.boxes(a) if b["obj"] == o)
        if direction > 0:
            todo = list(range(start + 1, min(len(project.items), start + 1 + count)))
        else:
            todo = list(range(start - 1, max(-1, start - 1 - count), -1))
        cur_item, cur = start, {o: b["box"] for o, b in seeds.items()}
        style = {}                                             # obj -> offset of the drawn box around SAM's outline
        done = pos = 0
        while pos < len(todo) and cur:
            chunk = todo[pos:pos + self.chunk]
            try:
                last, n, stopped = self._chunk(project, refs, cur_item, cur, chunk, classes, style, pos == 0,
                                               on_item, should_stop, masks)
            except torch.OutOfMemoryError:
                hw.free_gpu_memory()
                if self.chunk <= MIN_CHUNK:
                    raise
                self.chunk = max(MIN_CHUNK, self.chunk // 2)   # smaller sessions from now on
                continue
            done += n
            if stopped:
                break
            pos += len(chunk)
            cur_item = chunk[-1]
            cur = {o: r[1] for o, r in last.items() if r[1] is not None}
        return done

    def _chunk(self, project, refs, cur_item, cur, chunk, classes, style, first, on_item, should_stop, keep_masks=False):
        """One fresh session over [reference frames, current frame, chunk...]; the frame list is in
        tracking order, so backward tracking is the same run over reversed frames."""
        ref_items = sorted(refs, key=lambda it: abs(it - cur_item), reverse=True)
        frame_items = ref_items + [cur_item] + chunk
        sess = self.proc.init_video_session(video=[project.image(i) for i in frame_items],
                                            inference_device=self.device, inference_state_device="cpu",
                                            video_storage_device="cpu", dtype=self.dtype)
        last, n, stopped = {}, 0, False
        try:
            for local, it in enumerate(ref_items):
                boxes = {o: b for o, b in refs[it].items() if o in cur}
                if boxes:
                    self._add(sess, local, boxes)
            base = len(ref_items)
            self._add(sess, base, cur)
            for out in self.model.propagate_in_video_iterator(sess, start_frame_idx=0):
                local = out.frame_idx
                if local < base:
                    continue
                masks = self.proc.post_process_masks([out.pred_masks], original_sizes=[[sess.video_height,
                                                                                        sess.video_width]])[0]
                if local == base:                              # the start frame: learn each box's style
                    if first:
                        for k, o in enumerate(sess.obj_ids):
                            outline = mask_box(masks[k, 0].cpu().numpy())
                            if outline and outline[2] > outline[0] and outline[3] > outline[1]:
                                style[o] = style_offset(cur[o], outline)
                    continue
                scores = getattr(out, "object_score_logits", None)
                res = {}
                for k, o in enumerate(sess.obj_ids):
                    m = masks[k, 0].cpu().numpy()
                    box = mask_box(m)
                    score = float(torch.sigmoid(scores[k]).max()) if scores is not None else None
                    if keep_masks:                             # outlines: the mask is the label, its box follows
                        res[o] = (classes[o], box, score, m.astype(bool) if box else None)
                        continue
                    if box and o in style:
                        box = apply_style(box, style[o])
                    res[o] = (classes[o], box, score)
                if on_item:
                    on_item(frame_items[local], res)
                last = res
                n += 1
                if should_stop():
                    stopped = True
                    break
        finally:
            del sess
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return last, n, stopped
