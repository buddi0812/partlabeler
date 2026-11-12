"""Track labeled parts forward or backward through a project's frames: SAM 3 (HF Sam3TrackerVideoModel, Tracker, the
default, described here) or SAM 3.1 multiplex (Tracker31, opt-in, see make_tracker).

A run starts at one item and covers the next `count` items. Every part on the start item is a
prompt: its outline when it has one (segmentation projects: SAM 3 then remembers the exact shape a person
fixed, holes and notches included), else its box. Each part's most recent earlier manual label is prepended
as an extra reference frame (the two nearest), so what a person drew keeps steering the tracker (S2: reference
frames lifted IoU 0.969 -> 0.986); frames a person drew or confirmed further on in the run are prompts where they
are, pulling the part back onto their outline there. Each frame's prompts are turned into the tracker's memory right when they are added
(the model runs on that frame): the session only counts the inputs of the latest call as new, so prompts
on several frames would otherwise be dropped. Runs are split into chunks; each chunk is a fresh, fully
released session seeded with the previous chunk's last outlines or boxes (long sessions grow memory without
bound). S23 (lamp outlines a person confirmed, a synthetic stress clip with a look-alike neighbour and a hand
passing over the part): SAM 3 neither ballooned nor bled, so its outlines are taken as they come (a guard that cut
them was dropped: it cut parts a pole splits in two); prompts at frames a person drew further on lifted the stress
clip's IoU 0.930 -> 0.982. The model shares its image backbone with the other SAM 3 models (hw.sam3).
"""
import gc
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from engine import hw
from engine.models import SAM3, fetch

CHUNK = 120          # items per session (halved automatically when the GPU runs out of memory)
MIN_CHUNK = 8
MIN_AREA = 16        # mask pixels; anything smaller counts as "part not visible"
REFS = 2             # earlier frames a person vouched for, per part, kept as memory (S2: 2 references beat 1)
AREA_TOL = 0.25      # SAM 3.1: an outline this far off its part's recent median area is retried (DAM4SAM uses 20%)
HISTORY = 10         # SAM 3.1: good outlines per part that make that median
RETRY_GAP = 10       # SAM 3.1: frames between two retries of the same part


class Guard:
    """SAM 3.1 only (its outlines can balloon, S19-S22): per part, the median area of its last good outlines. An
    outline more than AREA_TOL off it on the frame after a good one is retried once (Tracker31: memory dropped, a
    recovery prompt from the last good outlines); what the retry gives is kept for its frame, but an outline off the
    median never joins the history, so a balloon never becomes the normal. A part seen again after being hidden is
    not checked (it comes back partly)."""

    def __init__(self):
        self.last, self.areas, self.seen, self.retried = {}, {}, {}, {}

    def ok(self, o, m, step: int) -> bool:
        if not self.areas.get(o) or self.seen.get(o, (None,))[0] != step - 1:   # first sight, or back from hiding
            return True
        med = float(np.median(self.areas[o]))
        return abs(float(m.sum()) - med) <= AREA_TOL * med

    def see(self, o, m, step: int, item: int, fresh: bool = False) -> bool:
        """A part's non-empty outline on a frame; True (and remembered) when it is a good one."""
        good = fresh or self.ok(o, m, step)
        self.seen[o] = (step, item)
        if good:
            if fresh:
                self.areas[o] = deque(maxlen=HISTORY)
            self.last[o] = m
            self.areas.setdefault(o, deque(maxlen=HISTORY)).append(int(m.sum()))
        return good

    def may_retry(self, o, step: int) -> bool:
        if step - self.retried.get(o, -RETRY_GAP) < RETRY_GAP:
            return False
        self.retried[o] = step
        return True


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


def make_tracker():
    """The tracker to use: SAM 3 through Hugging Face (default) or SAM 3.1 multiplex ("sam3.1", faster, see Tracker31),
    set by PARTLABELER_TRACKER or "tracker" in ~/.partlabeler/app.json. SAM 3.1 falls back to SAM 3 if it cannot load."""
    from engine.accounts import app_config
    if (os.environ.get("PARTLABELER_TRACKER") or app_config().get("tracker") or "sam3") != "sam3.1":
        return Tracker()
    try:
        return Tracker31()
    except Exception as e:                                # e.g. no internet for the first download
        print(f"SAM 3.1 could not load ({type(e).__name__}: {e}); tracking with SAM 3", flush=True)
        return Tracker()


class Tracker:
    name = "SAM 3"

    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None):
        device = device or hw.device(); dtype = dtype or hw.dtype(device)
        path = fetch(SAM3)
        self.model = hw.sam3(Sam3TrackerVideoModel, device, dtype)
        self.proc = Sam3TrackerVideoProcessor.from_pretrained(path)
        self.device, self.dtype, self.chunk = device, dtype, CHUNK

    def _add(self, sess, local: int, prompts: dict) -> None:
        """Prompts {obj: mask (H x W bool) or box} on one frame, made into conditioning memory at once."""
        masks = {o: v for o, v in prompts.items() if isinstance(v, np.ndarray)}
        boxes = {o: v for o, v in prompts.items() if not isinstance(v, np.ndarray)}
        if masks:                                          # masks and boxes cannot share one call
            objs = sorted(masks)
            self.proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=local, obj_ids=objs,
                                                      input_masks=[masks[o] for o in objs])
            self.model(inference_session=sess, frame_idx=local)
        if boxes:
            objs = sorted(boxes)
            self.proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=local, obj_ids=objs,
                                                      input_boxes=[[list(map(float, boxes[o])) for o in objs]])
            self.model(inference_session=sess, frame_idx=local)

    @torch.inference_mode()
    def track(self, project, start: int, count: int, on_item=None, should_stop=lambda: False,
              direction: int = 1, masks: bool = False, only=None) -> int:
        """Track the boxes on item `start` (only those of the classes in `only`, when given) through the next
        `count` items (direction 1) or the previous
        `count` items (direction -1). Calls on_item(item, {obj: (cls, box or None, score)}) per item, with the
        part's mask as a 4th value when `masks` (outline projects; the box is then the mask's own box);
        returns the items processed. Out of GPU memory, the chunk is halved and retried."""
        seeds = {b["obj"]: b for b in project.boxes(start) if b["source"] != "suggested" and (not only or b["cls"] in only)}
        if not seeds:
            return 0
        classes = {o: b["cls"] for o, b in seeds.items()}

        def prompt(item, b):                                   # the outline when there is one
            m = project.mask(item, b["obj"]) if masks else None
            return m if m is not None and m.any() else b["box"]

        refs = {}                                              # item -> {obj: prompt}: nearest manual labels behind start
        for o in seeds:
            behind = [a for a in project.anchors(o) if (a - start) * direction < 0]
            for a in (behind[-REFS:] if direction > 0 else behind[:REFS]):
                refs.setdefault(a, {})[o] = prompt(a, next(b for b in project.boxes(a) if b["obj"] == o))
        if direction > 0:
            todo = list(range(start + 1, min(len(project.items), start + 1 + count)))
        else:
            todo = list(range(start - 1, max(-1, start - 1 - count), -1))
        span = set(todo)                                       # frames where a person drew a part further on
        drawn = set(project.db.execute("SELECT item, obj FROM boxes WHERE source='manual'").fetchall())
        ahead = {}                                             # item -> [obj]; their outlines are read chunk by chunk
        for o in seeds:
            for a in project.anchors(o):
                if a in span and (a, o) in drawn:
                    ahead.setdefault(a, []).append(o)

        def at(item, o):
            b = next((x for x in project.boxes(item) if x["obj"] == o), None)
            return prompt(item, b) if b else None

        cur_item, cur = start, {o: prompt(start, b) for o, b in seeds.items()}
        style = {}                                             # obj -> offset of the drawn box around SAM's outline
        done = pos = 0
        while pos < len(todo) and cur:
            chunk = todo[pos:pos + self.chunk]
            try:
                last, n, stopped = self._chunk(project, refs, ahead, at, cur_item, cur, chunk, classes, style,
                                               pos == 0, on_item, should_stop, masks)
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
            cur = {o: (r[3] if masks and len(r) > 3 and r[3] is not None else r[1]) for o, r in last.items() if r[1] is not None}
        return done

    def _chunk(self, project, refs, ahead, at, cur_item, cur, chunk, classes, style, first, on_item, should_stop,
               keep_masks=False):
        """One fresh session over [reference frames, current frame, chunk...]; the frame list is in
        tracking order, so backward tracking is the same run over reversed frames. Returns (last results, frames
        done, stopped)."""
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
            for local, it in enumerate(chunk, base + 1):       # a person's own outline later on: a prompt there
                mine = {o: v for o in ahead.get(it, ()) if o in cur and (v := at(it, o)) is not None}
                if any(isinstance(v, np.ndarray) for v in mine.values()):   # one call per frame: a box asked after
                    mine = {o: v for o, v in mine.items() if isinstance(v, np.ndarray)}   # an outline is not heard
                if mine:
                    self._add(sess, local, mine)
            for out in self.model.propagate_in_video_iterator(sess, start_frame_idx=base):   # references: memory only
                local = out.frame_idx
                masks = self.proc.post_process_masks([out.pred_masks], original_sizes=[[sess.video_height,
                                                                                        sess.video_width]])[0]
                if local == base:                              # the start frame: learn each box's style
                    if first and not keep_masks:
                        for k, o in enumerate(sess.obj_ids):
                            outline = mask_box(masks[k, 0].cpu().numpy())
                            if outline and outline[2] > outline[0] and outline[3] > outline[1] and not isinstance(cur[o], np.ndarray):
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


class Tracker31:
    """SAM 3.1 Object Multiplex (Meta, 2026; code vendored from muggled_sam in engine/vendor), opt-in: every part is
    tracked in one pass, so a frame costs about the same with 2 or 10 parts. Same interface and behaviour as Tracker:
    prompts are the parts on the start frame (outlines, else boxes made into outlines first), the nearest earlier
    frame where a person vouched for every one of them is added as a reference, drawn boxes keep their style. Its
    memory is bounded (the prompt frames and the last 6 frames), so there are no chunks. More than 16 parts at once
    go to SAM 3.
    Measured on an RTX 3060 (S14, S17, S19-S22): 1.97 fps in the app with 4 parts vs 1.14 for SAM 3 (peak 2.5 vs 5.6 GB);
    outlines match SAM 3's except at the edge (IoU 0.998 outside a 3 px band, about 2% smaller). Not the default
    because in one of four lamps tracked through a dim-to-bright switch its outline ballooned onto the background
    (to a third of the frame, score still high) where SAM 3 held on; not a precision effect (same in fp32 and fp16),
    not the image preparation, reference frame or memory length."""
    name = "SAM 3.1"
    SIDE = 1008
    MAX_PARTS = 16

    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None):
        from engine.models import SAM31
        from engine.vendor.muggled_sam.v3p1_sam.make_sam_v3p1 import make_samv3p1_from_state_dict
        device = device or hw.device(); dtype = dtype or hw.dtype(device)
        core = make_samv3p1_from_state_dict(str(fetch(SAM31) / "sam3.1_multiplex.pt"), weights_only=True)
        core.to(device=device, dtype=dtype)
        self.core, self.ctx, self.inter = core, core.get_tracking_context(), core.get_interactive_context()
        self.device, self.dtype, self.fallback = device, dtype, None

    def _encode(self, img):
        """A frame's features, the image prepared exactly as the original model expects."""
        import cv2
        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return self.ctx.encode_image(self.core.image_encoder.prepare_image_like_original(bgr, self.SIDE, True))

    def _full(self, logits, H, W) -> np.ndarray:
        return (F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False)[:, 0] > 0).cpu().numpy()

    def _outlines(self, project, item, objs, enc, H, W, masks: bool) -> dict:
        """{obj: H x W mask} on `item`: its outline, else SAM 3.1's best outline inside its box."""
        out = {}
        for o in objs:
            m = project.mask(item, o) if masks else None
            if m is None or not m.any():
                b = next(x for x in project.boxes(item) if x["obj"] == o)["box"]
                prompts = self.inter.encode_prompts([[(b[0] / W, b[1] / H), (b[2] / W, b[3] / H)]], None, None)
                logits, ious = self.inter.generate_masks(enc, prompts)
                m = self._full(logits[:, int(ious[0].argmax())][:, None], H, W)[0]
            out[o] = m
        return out

    @torch.inference_mode()
    def track(self, project, start: int, count: int, on_item=None, should_stop=lambda: False,
              direction: int = 1, masks: bool = False, only=None) -> int:
        seeds = {b["obj"]: b for b in project.boxes(start) if b["source"] != "suggested" and (not only or b["cls"] in only)}
        if not seeds:
            return 0
        if len(seeds) > self.MAX_PARTS:
            self.fallback = self.fallback or Tracker(self.device, self.dtype)
            return self.fallback.track(project, start, count, on_item, should_stop, direction, masks, only)
        objs = sorted(seeds)
        classes = {o: seeds[o]["cls"] for o in objs}
        img = project.image(start)
        W, H = img.size
        enc = self._encode(img)
        seed = self._outlines(project, start, objs, enc, H, W, masks)
        style = {} if masks else {o: style_offset(seeds[o]["box"], mb) for o in objs
                                  if (mb := mask_box(seed[o])) and mb[2] > mb[0] and mb[3] > mb[1]}
        stack = lambda ms: torch.from_numpy(np.stack([ms[o] for o in objs])).unsqueeze(1).float().to(self.device)
        prompts = deque([self.ctx.encode_prompt_memory_from_mask(enc, stack(seed))])
        vouched = set.intersection(*[{a for a in project.anchors(o) if (a - start) * direction < 0} for o in objs])
        if vouched:                                            # the nearest frame a person checked, with every part
            ref = max(vouched) if direction > 0 else min(vouched)
            ref_enc = self._encode(project.image(ref))
            prompts.appendleft(self.ctx.encode_prompt_memory_from_mask(ref_enc, stack(self._outlines(project, ref, objs, ref_enc, H, W, masks))))
        todo = range(start + 1, min(len(project.items), start + 1 + count)) if direction > 0 else \
            range(start - 1, max(-1, start - 1 - count), -1)
        mems, done, guard, last_enc, base = deque([], maxlen=6), 0, Guard(), enc, list(prompts)
        for o in objs:
            guard.see(o, seed[o], -1, start, fresh=True)
        for step, k in enumerate(todo):
            enc = self._encode(project.image(k))
            for attempt in (0, 1):
                logits, _, ptrs, scores = self.ctx.step_video_masking_multiplex(enc, prompts, mems, num_multiplex_objects=len(objs))
                full, probs = self._full(logits, H, W), torch.sigmoid(scores.float()).cpu().numpy()
                jumped = [o for j, o in enumerate(objs) if probs[j] > 0.5 and full[j].any() and not guard.ok(o, full[j], step)]
                if attempt or not jumped or not any([guard.may_retry(o, step) for o in jumped]):
                    break
                mems.clear()                                   # the balloon must not become memory: start again,
                shown = {o: guard.last[o] if guard.seen[o][0] == step - 1 else np.zeros_like(guard.last[o]) for o in objs}
                prompts = deque(base + [self.ctx.encode_prompt_memory_from_mask(last_enc, stack(shown))])   # seen there only
            if (scores > 0).any():
                mems.append(self.ctx.encode_frame_memory(enc, logits, ptrs, scores))
            res = {}
            for j, o in enumerate(objs):
                if probs[j] > 0.5 and full[j].any():
                    guard.see(o, full[j], step, k)
            last_enc = enc
            for j, o in enumerate(objs):
                box = mask_box(full[j]) if probs[j] > 0.5 else None
                if masks:                                     # outlines: the mask is the label, its box follows
                    res[o] = (classes[o], box, float(probs[j]), full[j] if box else None)
                    continue
                res[o] = (classes[o], apply_style(box, style[o]) if box and o in style else box, float(probs[j]))
            if on_item:
                on_item(k, res)
            done += 1
            if should_stop():
                break
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return done
