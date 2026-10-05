"""Production runtime: TensorRT engines built by build_engine.py and part references (lampcheck.py XML) on camera
frames, with the least latency. Standalone on purpose (`partlabeler export-model` copies it next to the model with
lampcheck.py): TensorRT's Python package, PyTorch (GPU buffers and preparation), NumPy and OpenCV.

    eng = Engine("models/front")                 # model.json + the engine build_engine.py made on this machine
    parts = eng([rgb])[0]                        # [Part(name, score, box, query)]: the surest box per class
    cam = Camera(ref, "line_1", functions)       # one camera's lamp functions, judged with the reference
    status = cam.update(t, rgb, parts)           # {function: {"on", "ok", "missing", "faults"}}

    python partcheck.py config.json STATION CAMERA=VIDEO ... [--start S] [--seconds N]   # a demo on recordings

Latency: a frame is copied once into pinned memory, uploaded, resized and normalised on the GPU; the engine and the
decoding (sigmoid, top-k) run on one CUDA stream, captured as one CUDA graph per frame size, and only Q x 7 numbers
per frame come back (Q = 100 queries for seg-small). An engine built for a batch of N (model.json "batch") runs N
cameras' frames of the same moment in one go. Outlines are not needed for the part check (it measures inside the
part's box): Engine.outlines makes them only when asked (display). The part check runs on the CPU, on each found
part's box only.

The part check (lampcheck.py): a steady part is checked on every frame it is found and fails when at least 3 of its
last 5 checks fail; a blinking part (built as sweeping) is checked once per blink, on the frame where it reaches
furthest, and fails when at least 2 of its last 3 blinks fail. A function is ON when its parts are found (blinking:
found within the last `hold` seconds; "together": all its parts found in one frame within `hold`), OK when it is on
and none of its parts fails. For `settle` seconds after a steady function switches on or off, steady parts' faults are
held back (lamps change state then, e.g. a DRL dimming to a parking lamp when the low beam comes on), and so they are
after the object comes into place (a car still rolling to a stop shows its lamps a little smaller). Parts are
judged only while the object is in place: at least half of the found steady parts have their box centre within the
reference's range (widened by `place_tolerance` of the frame), so nothing is judged while a car drives in or out, and
one faulty part (whose box may shift a little) cannot hide itself. Blinking parts move while they sweep, so they do
not count; with only blinking parts found, the frame is judged. Faults are shown only while the object is in place,
and only once they have lasted `persist` seconds (a lamp fading out or a car pulling away never lasts that long; a
real fault lasts the whole test).
"""
import json
import sys
import time
from collections import defaultdict, deque, namedtuple
from pathlib import Path

import numpy as np

try:
    import lampcheck as L                                    # a production bundle: next to this file
except ImportError:
    from engine import lampcheck as L

HERE = Path(__file__).resolve().parent
Part = namedtuple("Part", "name score box query")
PLACE_TOLERANCE = 0.03          # share of the frame a part's centre may be outside its reference range and still count


class Engine:
    """One TensorRT engine with its buffers, stream and CUDA graphs; call it with up to `batch` RGB frames (H x W x 3
    uint8, all of one size) of the same moment."""
    _logger = None                                            # one TensorRT logger for every engine of the process

    def __init__(self, folder, graph: bool = True, engine_file=None, outlines: bool = False):
        import tensorrt as trt
        import torch
        self.torch, self.folder = torch, Path(folder)
        self.card = card = json.loads((self.folder / "model.json").read_text())
        self.classes, self.threshold, self.size = card["classes"], card["threshold"], card["resolution"]
        self.batch = card.get("batch", 1)
        kinds = (".engine", ".fp32.engine") if outlines else (".boxes.engine", ".engine", ".boxes.fp32.engine",
                                                              ".fp32.engine")     # boxes-only engines are faster
        path = Path(engine_file) if engine_file else next(
            (p for p in (self.folder / Path(card["model"]).with_suffix(s).name for s in kinds) if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"no engine in {self.folder}: run build_engine.py there first")
        Engine._logger = Engine._logger or trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(Engine._logger).deserialize_cuda_engine(path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        types = {getattr(trt, k): getattr(torch, k) for k in ("float32", "float16", "int32", "int64", "bool")
                 if hasattr(trt, k)}
        self.buf = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.buf[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)),
                                         dtype=types[self.engine.get_tensor_dtype(name)], device="cuda")
            self.ctx.set_tensor_address(name, self.buf[name].data_ptr())
        self.queries = self.buf["dets"].shape[1]
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(card["input"]["mean"], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(card["input"]["std"], device="cuda").view(1, 3, 1, 1)
        self.graph, self.slots = graph, {}

    def _run(self, frames, out):
        """All the GPU work for one batch, on the current stream: frames [B, H, W, 3] uint8 -> out [B, Q, 7]
        (cx, cy, w, h as shares, score, class, query), the Q surest (query, class) pairs, surest first."""
        torch = self.torch
        x = frames.permute(0, 3, 1, 2).float().div_(255)
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False,
                                            antialias=False)          # as model.json: half-pixel, no antialias
        self.buf["input"].copy_((x - self.mean) / self.std)
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        scores = self.buf["labels"][..., :-1].float().sigmoid()   # sigmoid per class; last column: background
        C = scores.shape[2]
        s, i = scores.flatten(1).topk(self.queries, dim=1)
        q = i // C
        d = torch.gather(self.buf["dets"].float(), 1, q[..., None].expand(-1, -1, 4))
        out.copy_(torch.cat([d, s[..., None], (i % C)[..., None].float(), q[..., None].float()], 2))

    def _slot(self, h, w):
        """Pinned host and device buffers for frames of one size, and the CUDA graph of _run for them."""
        torch = self.torch
        if (h, w) not in self.slots:
            host = torch.empty((self.batch, h, w, 3), dtype=torch.uint8, pin_memory=True)
            dev = torch.empty((self.batch, h, w, 3), dtype=torch.uint8, device="cuda")
            out = torch.empty((self.batch, self.queries, 7), device="cuda")
            back = torch.empty((self.batch, self.queries, 7), pin_memory=True)
            graph = None
            if self.graph:
                with torch.cuda.stream(self.stream):         # TensorRT's first run must be outside a capture
                    self._run(dev, out)
                self.stream.synchronize()
                try:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream, capture_error_mode="thread_local"):
                        self._run(dev, out)
                except Exception as e:                        # ponytail: eager fallback, a few launches slower
                    print(f"CUDA graph not used ({type(e).__name__}: {e})", file=sys.stderr)
                    graph = None
            self.slots[(h, w)] = (host, dev, out, back, graph)
        return self.slots[(h, w)]

    def __call__(self, frames):
        """[[Part, ...] per frame]: the surest part per class above the model's threshold, boxes in frame pixels."""
        if not 1 <= len(frames) <= self.batch:
            raise ValueError(f"this engine runs 1 to {self.batch} frames at a time, not {len(frames)}")
        h, w = frames[0].shape[:2]
        host, dev, out, back, graph = self._slot(h, w)
        hv = host.numpy()
        for j in range(self.batch):                          # a fixed batch: a missing frame is padded with a copy
            hv[j] = frames[min(j, len(frames) - 1)]
        with self.torch.cuda.stream(self.stream):
            dev.copy_(host, non_blocking=True)
            if graph is not None:
                graph.replay()
            else:
                self._run(dev, out)
            back.copy_(out, non_blocking=True)
        self.stream.synchronize()
        return [self._parts(back[j].numpy(), w, h) for j in range(len(frames))]

    def _parts(self, o, w, h):
        parts, seen = [], set()
        for cx, cy, bw, bh, s, c, q in o[o[:, 4] > self.threshold]:
            if int(c) in seen:                                # surest first: one per class
                continue
            seen.add(int(c))
            box = (max(0.0, (cx - bw / 2) * w), max(0.0, (cy - bh / 2) * h), min(w, (cx + bw / 2) * w),
                   min(h, (cy + bh / 2) * h))
            parts.append(Part(self.classes[int(c)], float(s), tuple(float(v) for v in box), int(q)))
        return parts

    def outlines(self, j, parts, w, h):
        """The parts' pixels [K, h, w] bool of frame j of the last call (an engine with outlines, Engine(...,
        outlines=True); a boxes-only engine gives empty ones). For display."""
        torch = self.torch
        if "masks" not in self.buf or not parts:
            return np.zeros((len(parts), h, w), bool)
        q = torch.tensor([p.query for p in parts], device="cuda")
        m = torch.nn.functional.interpolate(self.buf["masks"][j, q][None].float(), size=(h, w), mode="bilinear",
                                            align_corners=False)[0] > 0
        return m.cpu().numpy()


class Camera:
    """The lamp functions one camera sees, judged with a reference group (lampcheck XML). functions:
    [{"name": "DRL", "parts": ["left_drl", "right_drl"], "rule": "all" | "blink" | "together"}]."""

    def __init__(self, ref: dict, group: str, functions, hold: float = 1.0, gap: float = 0.12, check_every: int = 1,
                 settle: float = 1.0, check: bool = True, place_tolerance: float = PLACE_TOLERANCE,
                 persist: float = 0.0):
        if group not in ref:
            raise KeyError(f"no group {group!r} in the reference: {', '.join(ref)}")
        self.ref, self.group, self.functions = ref, group, functions
        self.hold, self.gap, self.check_every, self.settle = hold, gap, check_every, settle
        self.check = check                                   # False: functions on/off only, no part check
        self.place_tolerance, self.persist, self.failing = place_tolerance, persist, {}
        self.changed, self.steady_on = -1e9, None
        parts = ref[group]["parts"]
        self.sweep = {p for p, r in parts.items() if r.get("sweep")}
        self.pairs = [pair for pair in ref[group].get("pairs", {}) if not set(pair) & self.sweep]
        self.votes = L.settings(ref, group)                  # the reference's votes (tuned with it)
        self.results = defaultdict(lambda: deque(maxlen=self.votes["steady_vote"][1]))  # part (or pair): recent checks
        self.blinks = defaultdict(lambda: deque(maxlen=self.votes["sweep_vote"][1]))    # blinking part: recent blinks
        self.open, self.seen, self.together, self.n, self.placed = {}, {}, {}, 0, False

    def update(self, t: float, rgb, parts) -> dict:
        """t: the frame's time in seconds; rgb: the frame (H x W x 3 uint8); parts: the engine's parts of it."""
        found = {p.name: p.box for p in parts}
        self.n += 1
        judge = self.check and self.in_place(found, rgb.shape[1], rgb.shape[0])
        shift = self.shift(found, rgb.shape[1], rgb.shape[0]) if judge else (0.0, 0.0)
        if judge and not self.placed:                        # just came into place: let it settle
            self.changed = t
        for name, box in found.items():
            if judge and name in self.sweep:                   # a blink: keep only its furthest-reaching frame
                long = max(box[2] - box[0], box[3] - box[1])
                b = self.open.setdefault(name, {"peak": -1.0})
                b["last"] = t
                if long > b["peak"]:
                    b.update(peak=long, rgb=rgb, box=box, shift=shift)
            elif judge:
                if t - self.seen.get(name, -1e9) > self.hold:  # lit again after a pause: a new judgement
                    self.results[name].clear()
                if self.n % self.check_every == 0:
                    self.results[name].append(self._check(name, rgb, box, shift))
            self.seen[name] = t
        for name, b in list(self.open.items()):              # a blink ends when its part is gone for `gap` seconds
            if name not in found and t - b["last"] > self.gap:
                self.blinks[name].append(self._check(name, b["rgb"], b["box"], b["shift"]))
                del self.open[name]
        for a, b in self.pairs if judge else ():
            if a in found and b in found and self.n % self.check_every == 0:
                self.results[(a, b)].append(self._pair(a, b, rgb, found))
        for f in self.functions:
            if f["rule"] == "together" and all(p in found for p in f["parts"]):
                self.together[f["name"]] = t
        self.placed = judge
        return self.status(t, found)

    def _check(self, name, rgb, box, shift=(0.0, 0.0)):
        return L.check(self.ref, self.group, name, rgb, box, shift)

    def shift(self, found, w, h) -> tuple:
        """How far the object stands from its usual place (dx, dy as shares of the frame): the median offset of the
        found steady parts from their reference centres (0 when none is found). Picks each part's variant."""
        parts = self.ref[self.group]["parts"]
        offs = [((b[0] + b[2]) / 2 / w - L.centre(parts[n])[0], (b[1] + b[3]) / 2 / h - L.centre(parts[n])[1])
                for n, b in found.items() if n not in self.sweep and n in parts]
        return tuple(float(x) for x in np.median(np.array(offs), axis=0)) if offs else (0.0, 0.0)

    def _pair(self, a, b, rgb, found):
        return L.check_pair(self.ref, self.group, a, rgb, found[a], b, found[b])

    def in_place(self, found, w, h) -> bool:
        """At least half of the found steady parts have their box centre where the reference expects (True when
        only blinking parts are found: their boxes move while they sweep)."""
        parts, tol = self.ref[self.group]["parts"], self.place_tolerance
        inside = [r["cx"][0] - tol <= (b[0] + b[2]) / 2 / w <= r["cx"][1] + tol
                  and r["cy"][0] - tol <= (b[1] + b[3]) / 2 / h <= r["cy"][1] + tol
                  for n, b in found.items() if n not in self.sweep for r in [parts.get(n)] if r]
        return 2 * sum(inside) >= len(inside)

    def verdict(self, key):
        """(ok, reasons) of a part or pair by its vote; ok None while there are too few checks to judge."""
        results = list(self.blinks[key] if key in self.sweep else self.results[key])
        vote = self.votes["sweep_vote" if key in self.sweep else "steady_vote"]
        if len(results) < vote[0]:
            return None, []
        v = L._vote(results, vote)
        return v["ok"], v["reasons"]

    def status(self, t: float, found) -> dict:
        steady_on = tuple(all(p in found for p in f["parts"]) for f in self.functions if f["rule"] == "all")
        if self.steady_on is not None and steady_on != self.steady_on:     # a steady function switched
            self.changed = t
        self.steady_on = steady_on
        settling = t - self.changed < self.settle
        out = {}
        for f in self.functions:
            ps, rule = f["parts"], f["rule"]
            if rule == "all":
                on = all(p in found for p in ps)
            elif rule == "blink":
                on = any(t - self.seen.get(p, -1e9) <= self.hold for p in ps)
            else:
                on = t - self.together.get(f["name"], -1e9) <= self.hold
            keys = list(ps) + [pair for pair in self.pairs if set(pair) <= set(ps)]
            faults = {}
            for k in keys:
                ok, r = self.verdict(k)
                if ok is not False or not self.placed or (settling and k not in self.sweep):
                    self.failing.pop(k, None)
                elif t - self.failing.setdefault(k, t) >= self.persist:     # a fault only once it lasts
                    faults["+".join(k) if isinstance(k, tuple) else k] = r
            out[f["name"]] = {"on": on, "ok": on and not faults, "faults": faults, "in_place": self.placed,
                              "missing": [p for p in ps if p not in found] if rule == "all" else []}
        return out


def load(config_path, station: str, cameras=None):
    """Engines (one per model) and cameras of one station from a config (see the production guide); cameras: only
    these (and their models). A function's "parts_by_station" ({station: [parts]}) replaces its parts there (e.g. a
    part one station's camera cannot see); a camera's own "camera" settings (hold, gap, settle...) override the
    config's."""
    base = Path(config_path).resolve().parent
    cfg = json.loads(Path(config_path).read_text())
    use = {k: c for k, c in cfg["cameras"].items() if cameras is None or k in cameras}
    refs = {k: L.read_xml(base / cfg["references"][k]) for k in {c["reference"] for c in use.values()}}
    engines = {k: Engine(base / cfg["models"][k]) for k in {c["model"] for c in use.values()}}
    functions = {k: [f | {"parts": f.get("parts_by_station", {}).get(station, f["parts"])} for f in fs]
                 for k, fs in cfg["functions"].items()}
    cams = {name: Camera(refs[c["reference"]], c["group"].format(station=station), functions[c["functions"]],
                         **(cfg.get("camera", {}) | c.get("camera", {})), check=c.get("check", True))
            for name, c in use.items()}
    return cfg, engines, cams


def demo(config_path, station, videos, seconds: float = 0, start: float = 0):
    """Recorded videos of one station (CAMERA=VIDEO): cameras sharing an engine run as one batch, frame by frame;
    prints each change of a function's state and the latency of every stage. From `start` s, `seconds` long."""
    import av
    cfg, engines, cams = load(config_path, station, videos)
    by_model = defaultdict(list)
    for name, video in videos.items():
        by_model[cfg["cameras"][name]["model"]].append((name, video))
    timing = defaultdict(list)
    for model, group in by_model.items():
        eng = engines[model]
        for k in range(0, len(group), eng.batch):
            chunk = group[k:k + eng.batch]
            containers = [av.open(v) for _, v in chunk]
            for c in containers:
                c.streams.video[0].thread_type = "AUTO"
            streams = [c.decode(c.streams.video[0]) for c in containers]
            last = {}
            for frames in zip(*streams):
                if float(frames[0].time or 0) < start:          # decoded and skipped: the cameras stay paired
                    continue
                if seconds and float(frames[0].time or 0) > start + seconds:
                    break
                t0 = time.perf_counter()
                rgbs = [f.to_ndarray(format="rgb24") for f in frames]
                t1 = time.perf_counter()
                results = eng(rgbs)
                t2 = time.perf_counter()
                for (name, _), f, rgb, parts in zip(chunk, frames, rgbs, results):
                    st = cams[name].update(float(f.time or 0), rgb, parts)
                    now = {fn: ("OK" if s["ok"] else "FAULT " + json.dumps(s["faults"]) if s["on"] else "off")
                           for fn, s in st.items()}
                    for fn, v in now.items():
                        if last.get((name, fn)) != v:
                            print(f"{float(f.time or 0):7.2f} s  {name:8s} {fn:18s} {v}")
                            last[(name, fn)] = v
                t3 = time.perf_counter()
                for key, v in (("decode", t1 - t0), ("engine", t2 - t1), ("check", t3 - t2)):
                    timing[(model, key)].append(1000 * v)
            for c in containers:
                c.close()
    for (model, key), v in sorted(timing.items()):
        print(f"{model:8s} {key:7s} median {np.median(v):6.2f} ms  p99 {np.percentile(v, 99):6.2f} ms  ({len(v)} runs)")


if __name__ == "__main__":
    args = sys.argv[1:]
    opt = {k: float(args.pop(args.index(f"--{k}") + 1)) if f"--{k}" in args else 0 for k in ("seconds", "start")}
    args = [a for a in args if a not in ("--seconds", "--start")]
    if len(args) < 3:
        sys.exit(__doc__.split("\n\n")[1])
    demo(args[0], args[1], dict(a.split("=", 1) for a in args[2:]), **opt)
