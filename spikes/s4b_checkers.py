"""S4b: can an off-the-shelf decision model check boxes with no training?

For labeled parts on held-out White car crops, each checker sees:
  positive   : the part crop + "The image shows <part>."
  drift      : a crop shifted off the part (like a tracker that slipped) + the same statement
  wrong part : the part crop + a statement about a different part
Reported: how well the checker's probability separates positives from each negative (AUROC,
1.0 = perfect, 0.5 = chance) and time per check. Checkers: openjev (image NLI, zero-shot) and
Laya Vision (noul question, zero-shot).

    python spikes/s4b_checkers.py
"""
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path("data/example/s7_ds/valid")
OPENJEV = ("AlexWortega/openjev", "058a6c24911b46d908fbe23541390f8af3df3e4d", "qwen3.5-0.8b-nli-v2s-long")
LAYA = ("thaitea/laya-vision", "8b318c99d7ad3ce19c24369263463882eada9d1e")
PART = {"left_drl": "a car's LED daytime running light strip", "right_drl": "a car's LED daytime running light strip",
        "left_head_lamp": "a car's headlamp", "right_head_lamp": "a car's headlamp",
        "parking_lamp": "a car's front light bar", "left_mirror_lamp": "the indicator lamp on a car's side mirror",
        "right_mirror_lamp": "the indicator lamp on a car's side mirror", "brand_logo_fr": "the brand logo on a car",
        "fr_bumper_grill": "a car's front grille", "front_license_plate": "a car's front number plate area",
        "front_bumper_skid": "a car's front bumper skid plate", "roof_rack": "a car's roof rail",
        "roof_granish": "a trim piece on a car's roof", "sun_roof": "a car's sunroof"}


def padded(box, W, H, f=1.3, min_side=32):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = max(min_side, (box[2] - box[0]) * f), max(min_side, (box[3] - box[1]) * f)
    return (max(0, cx - w / 2), max(0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2))


def build_cases(names, per_class=10, seed=0):
    rng = random.Random(seed)
    inst = {}
    for f in sorted((ROOT / "labels").glob("*.txt")):
        img = Image.open(ROOT / "images" / (f.stem + ".jpg")).convert("RGB")
        W, H = img.size
        for line in f.read_text().splitlines():
            c, cx, cy, w, h = line.split()
            n = names[int(c)]
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            inst.setdefault(n, []).append((f.stem, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), (W, H)))
    cases = []
    cache = {}
    for n, lst in inst.items():
        for stem, box, (W, H) in rng.sample(lst, min(per_class, len(lst))):
            img = cache.setdefault(stem, Image.open(ROOT / "images" / (stem + ".jpg")).convert("RGB"))
            pos = img.crop(padded(box, W, H))
            bw, bh = box[2] - box[0], box[3] - box[1]
            dx, dy = rng.choice([(1.5, 0), (-1.5, 0), (0, 1.5), (0, -1.5)])
            drift = img.crop(padded((box[0] + dx * bw, box[1] + dy * bh, box[2] + dx * bw, box[3] + dy * bh), W, H))
            other = rng.choice(sorted({v for v in PART.values() if v != PART[n]}))
            cases += [("positive", n, pos, PART[n]), ("drift", n, drift, PART[n]), ("wrong_part", n, pos, other)]
    return cases


class OpenJev:
    def __init__(self, sub: str = OPENJEV[2]):
        from huggingface_hub import snapshot_download
        repo, rev, _ = OPENJEV
        path = Path(snapshot_download(repo, revision=rev, allow_patterns=[f"{sub}/*"])) / sub
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        if not (path / "preprocessor_config.json").exists():     # 4B ships without one; same Qwen3.5 processor
            path_ip = Path(snapshot_download(repo, revision=rev, allow_patterns=[f"{OPENJEV[2]}/*"])) / OPENJEV[2]
        else:
            path_ip = path
        self.ip = AutoImageProcessor.from_pretrained(path_ip)
        self.img_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        probe = self.ip(images=[Image.new("RGB", (320, 240))], return_tensors="pt")
        n = int(probe["image_grid_thw"].prod()) // self.ip.merge_size ** 2
        self.block = "<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>"
        self.template = getattr(self.model.config, "nli_template", None) or "Premise: {premise}\nHypothesis: {hypothesis}"
        self.entail = {v: int(k) for k, v in self.model.config.id2label.items()}["entailment"]

    @torch.inference_mode()
    def score(self, image, part: str) -> float:
        text = self.template.format(premise=self.block + "\nA close-up from a car-factory inspection camera.",
                                    hypothesis=f"The image shows {part}.")
        enc = self.tok(text, add_special_tokens=False, return_tensors="pt")
        vis = self.ip(images=[image.resize((320, 240))], return_tensors="pt")
        batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                 "pixel_values": vis["pixel_values"], "image_grid_thw": vis["image_grid_thw"],
                 "mm_token_type_ids": (enc["input_ids"] == self.img_id).long()}
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**batch).logits.float()[0]
        return float(torch.softmax(logits, -1)[self.entail])


class LayaVision:
    def __init__(self):
        import laya
        self.agent = laya.load_vlm(LAYA[0], revision=LAYA[1], device="cuda")

    def score(self, image, part: str) -> float:
        q = {"q": {"type": "noul", "instructions": f"Does this image show {part}?"}}
        return float(self.agent.predict({"image": image}, q)["answers"]["q"]["noul"])


def evaluate(checker, cases):
    for _, _, img, part in cases[:2]:
        checker.score(img, part)                     # warm-up
    t = time.perf_counter()
    scores = [checker.score(img, part) for _, _, img, part in cases]
    ms = (time.perf_counter() - t) / len(cases) * 1000
    kinds = [k for k, *_ in cases]
    pos = [s for s, k in zip(scores, kinds) if k == "positive"]
    res = {"ms_per_check": round(ms, 1), "median_score_positive": round(statistics.median(pos), 3)}
    for neg in ("drift", "wrong_part"):
        neg_s = [s for s, k in zip(scores, kinds) if k == neg]
        res[f"auroc_vs_{neg}"] = round(roc_auc_score([1] * len(pos) + [0] * len(neg_s), pos + neg_s), 3)
        res[f"median_score_{neg}"] = round(statistics.median(neg_s), 3)
        res[f"accuracy_at_0.5_vs_{neg}"] = round((sum(s >= 0.5 for s in pos) + sum(s < 0.5 for s in neg_s))
                                                / (len(pos) + len(neg_s)), 3)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["openjev", "laya", "all"], default="all")
    ap.add_argument("--openjev-sub", default=OPENJEV[2], help="openjev checkpoint folder, e.g. qwen3.5-4b-nli-v2")
    args = ap.parse_args()
    names = Path("data/example/dataset/classes.txt").read_text().split()
    cases = build_cases(names)
    out = {"cases": len(cases), "parts": len({n for _, n, _, _ in cases})}
    runs = []
    if args.only in ("openjev", "all"):
        runs.append((f"openjev_{args.openjev_sub}", lambda: OpenJev(args.openjev_sub)))
    if args.only in ("laya", "all"):
        runs.append(("laya_vision", LayaVision))
    for name, cls in runs:
        checker = cls()
        out[name] = evaluate(checker, cases)
        print(name, json.dumps(out[name]), flush=True)
        del checker
        torch.cuda.empty_cache()
    Path("spikes/out").mkdir(exist_ok=True)
    Path(f"spikes/out/s4b_checkers_{args.only}_{args.openjev_sub}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
