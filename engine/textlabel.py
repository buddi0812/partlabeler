"""Annotate by text: SAM 3 finds the objects a few words name ("car", "truck cabin", "wheel") on frames of a project
and adds them as parts to check, each phrase as a class of its own.

    prompts = parse("car, truck cabin; wheel")                # ["car", "truck cabin", "wheel"]
    items = scope_items(project, item, "rest")               # frame | rest | task | project
    annotate(project, items, [(text, cls), ...], finder)     # finder: engine.parent.ConceptFinder

Hits are added like a model's labels (source "imported", score = SAM 3's sureness, so unsure ones are marked to
check, and frames stay unconfirmed until a person confirms them). A hit is skipped when a part of the same class
already covers it (a second run adds nothing twice; a "wheel" inside a "car" box is still found). A part keeps its
id from one frame to the next while its box overlaps the one before, as with tracking, so jumps and gaps show."""
import re
from collections import Counter

from engine.parent import iou, overlap

SCOPES = ("frame", "rest", "task", "project")


def parse(prompts) -> list[tuple[str, str]]:
    """[(phrase, label), ...] from "car, truck cabin; wheel" (commas, semicolons or new lines), a list of phrases,
    or a list of {"text", "label"}; the label is the phrase itself unless given. Repeats are dropped."""
    if isinstance(prompts, str):
        prompts = re.split(r"[,;\n]", prompts)
    out, seen = [], set()
    for p in prompts or []:
        text, label = (p.get("text"), p.get("label")) if isinstance(p, dict) else (p, None)
        text = " ".join(str(text or "").split())
        label = " ".join(str(label or text).split())
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append((text, label))
    return out


def scope_items(project, item: int, scope: str) -> list[int]:
    """The frames of a scope: this one, it and the rest of its task, its whole task, or the whole project."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
    a, b = project.ranges[project.items[item]["task"]]
    return {"frame": [item], "rest": list(range(item, b)), "task": list(range(a, b)),
            "project": list(range(len(project.items)))}[scope]


def annotate(project, items, prompts, finder, threshold: float = 0.5, outlines: bool = False, lock=None,
             progress=None, should_stop=None) -> dict:
    """Find each (phrase, class) of `prompts` on every frame of `items` and add the hits as parts to check.
    outlines: keep SAM 3's mask of each hit (segmentation projects). lock: held around each model call (a model
    shared with other projects). Returns {"frames", "parts", "per_class": {class: count}, "stopped"}."""
    total, frames, parts, stopped = len(items), 0, 0, False
    per_class = Counter()
    texts, classes = [t for t, _ in prompts], {c for _, c in prompts}
    for n, k in enumerate(items, 1):
        if should_stop and should_stop():
            stopped = True
            break
        img = project.image(k)
        if lock:
            with lock:
                hits = finder.detect_texts(img, texts, threshold, masks=outlines)
        else:
            hits = finder.detect_texts(img, texts, threshold, masks=outlines)
        # the frame before in the same task: its parts of these classes lend their ids to the hits that overlap them
        linked = [(b["obj"], b["cls"], b["box"]) for b in project.boxes(k - 1)
                  if b["source"] != "suggested" and b["cls"] in classes] \
            if k > 0 and project.items[k]["task"] == project.items[k - 1]["task"] else []
        here = project.boxes(k)
        have = [(b["cls"], b["box"]) for b in here if b["source"] != "suggested"]
        now, used = [], {b["obj"] for b in here}                 # an id on this frame already is never reused
        with project.db:
            for text, cls in prompts:
                for box, score, mask in sorted(hits.get(text, []), key=lambda h: -h[1]):
                    if any(c == cls and overlap(box, b) >= 0.5 for c, b in have):
                        continue                              # already labeled (or found by an earlier run)
                    match = max((q for q in linked if q[1] == cls and q[0] not in used and iou(q[2], box) >= 0.3),
                                key=lambda q: iou(q[2], box), default=None)
                    obj = match[0] if match else project.new_obj()
                    used.add(obj)
                    project.put(k, obj, cls, box, "imported", round(score, 4),
                                mask=mask if mask is not None and mask.any() else None)
                    have.append((cls, box))
                    now.append((obj, cls, box))
                    per_class[project.classes[cls]] += 1
        parts += len(now)
        frames += 1
        if progress:
            progress(n, total)
    return {"frames": frames, "parts": parts, "per_class": dict(per_class), "stopped": stopped}
