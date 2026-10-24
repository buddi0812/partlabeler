"""The training queue (hawk-ai-internal app-015-plan, Setup flow v2): a project's confirmed frames go to Teach, then
the finished run is exported for production, one run at a time, in the order they were sent. A run waits while the
GPU hasn't room for it ("waiting_for_room", with the numbers), so a box can either let it wait or make room (stop its
assistant) and it starts by itself. The queue lives in the app's default projects folder (_teach/queue.json): a
restart keeps what was queued; a run cut off by the restart is marked failed.

    queued -> [waiting_for_room] -> running -> done | failed | cancelled      (every run ends in one of the three)
"""
import json
import threading
import time
from pathlib import Path

from engine.detector import EPOCHS

H = None                                                    # ui/host_fastapi.py, set by ui/v1.py's mount()
_lock = threading.Lock()
_runs: list[dict] = []
_wake = threading.Event()
_started = False
ACTIVE = ("queued", "waiting_for_room", "running")
# Seconds per training image per epoch on a typical 12 GB card (RTX 3060: ~300 frames x 30 epochs of small at 640 px
# in ~40 min), before this computer has finished runs of its own to measure; outline models are slower. A rough start.
RATE = {"nano": 0.15, "small": 0.25, "medium": 0.4, "classify": 0.02}
SEG_SLOWER = 1.5
CLASSIFY_GB = 1.5                # DINOv3 small reading crops, one at a time (engine/classifier.py)
POLL_S = 10                                                 # how often a waiting run looks for room again
KEEP_DONE = 50


def _file() -> Path:
    return Path(H.state["home"]) / "_teach" / "queue.json"


def _save() -> None:
    f = _file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(_runs[-(KEEP_DONE + 20):], indent=1), encoding="utf-8")


def load() -> None:
    """At start-up: what was queued stays queued; a run the restart cut off failed."""
    try:
        runs = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        runs = []
    cut = False
    for r in runs:
        if r["state"] in ("running", "waiting_for_room"):
            r.update(state="failed", error="PartLabeler restarted during the run") if r["state"] == "running" else r.update(state="queued")
            cut = True
    global _started
    with _lock:
        _runs[:] = runs
        if cut:                                          # said in the file too, not only until the next change
            _save()
        start, _started = not _started, True
    if start:                                            # one worker for the process, whatever starts the app again
        threading.Thread(target=_work, daemon=True, name="train-queue").start()
    _wake.set()


def _position(r: dict) -> int:
    """Runs ahead of it (0: it is next, or running)."""
    active = [x for x in _runs if x["state"] in ACTIVE]
    return active.index(r) if r in active else 0


def view(r: dict) -> dict:
    return {**{k: r.get(k) for k in ("run", "project", "state", "progress", "text", "passed", "knowledge_score",
                                     "export_id", "error", "need_gb", "free_gb", "queued_at", "started_at",
                                     "finished_at", "minutes", "compare")}, "position": _position(r)}


def _rate(size: str, resolution: int) -> tuple[float, str]:
    """(seconds per image per epoch, where the number comes from): this computer's own finished runs of that kind
    when it has any (median), else RATE."""
    kind = "classify" if size == "classify" else size.removeprefix("seg-")
    with _lock:
        own = sorted(r["sec_per_image_epoch"] * (640 / r["resolution"]) ** 2 for r in _runs
                     if r["state"] == "done" and r.get("sec_per_image_epoch") and r["size"] == size)
    if own:
        return own[len(own) // 2] * (resolution / 640) ** 2, "this computer's earlier runs"
    base = RATE.get(kind, RATE["small"]) * (SEG_SLOWER if size.startswith("seg") else 1)
    return base * (1 if size == "classify" else (resolution / 640) ** 2), "a typical 12 GB graphics card"


def estimate(frames: int, size: str, epochs: int | None = None, resolution: int | None = None) -> dict:
    """How long a run should take before it starts (the box's confirm screen): training frames x epochs x rate, as a
    range, plus the export. frames: the confirmed frames (a fifth are kept back to prove the model)."""
    resolution = resolution or (224 if size == "classify" else 624 if size.startswith("seg") else 640)
    epochs = epochs or (1 if size == "classify" else EPOCHS)
    rate, basis = _rate(size, resolution)
    sec = max(1, round(frames * 0.8)) * epochs * rate                               # every epoch
    # it may stop early (no better for PATIENCE epochs): from about a third of the epochs to all of them, + the export
    low, high = (sec, sec) if size == "classify" else (sec * 0.35, sec * 1.3)
    return {"frames": frames, "size": size, "epochs": epochs, "resolution": resolution,
            "minutes": [max(1, round((low + 120) / 60)), max(2, round((high + 120) / 60))], "based_on": basis}


def enqueue(project: str, user, size: str, epochs: int | None = None, resolution: int | None = None,
            frames: int = 0, aug: str = "strong", compare_to: str | None = None) -> dict:
    base = H.safe_name(f"{project}_{time.strftime('%Y%m%d_%H%M%S')}")
    with _lock:
        taken = {x["run"] for x in _runs}
    name = next(n for n in (base if i == 1 else f"{base}_{i}" for i in range(1, 1000))
                if n not in taken and not (H.runs_dir() / n).exists())
    resolution = resolution or (224 if size == "classify" else 624 if size.startswith("seg") else 640)
    r = {"run": name, "project": project, "user": user, "size": size, "epochs": epochs, "resolution": resolution,
         "state": "queued", "progress": 0.0, "text": "Waiting for its turn", "passed": None, "knowledge_score": None,
         "export_id": None, "error": None, "need_gb": None, "free_gb": None, "queued_at": time.time(),
         "started_at": None, "finished_at": None, "frames": frames, "job": None, "cancel": False, "aug": aug,
         "compare_to": compare_to, "compare": None,
         "minutes": estimate(frames, size, epochs, resolution)["minutes"]}
    with _lock:
        _runs.append(r)
        _save()
    _wake.set()
    return view(r)


def get(run: str) -> dict | None:
    with _lock:
        r = next((x for x in _runs if x["run"] == run), None)
        return view(r) if r else None


def queue() -> list[dict]:
    with _lock:
        return [view(r) for r in _runs if r["state"] in ACTIVE] + [view(r) for r in _runs[::-1] if r["state"] not in ACTIVE][:10]


def cancel(run: str) -> dict | None:
    """A queued or waiting run ends now; a running one stops at Teach's next safe point (state "cancelled" then)."""
    with _lock:
        r = next((x for x in _runs if x["run"] == run), None)
        if r is None:
            return None
        if r["state"] in ("queued", "waiting_for_room"):
            r.update(state="cancelled", text="Cancelled", finished_at=time.time())
        elif r["state"] == "running":
            r["cancel"] = True
            if r.get("job") and r["job"] in H.state["jobs"]:
                H.state["jobs"][r["job"]]["stop"] = True
        _save()
        return view(r)


def _set(r: dict, **kw) -> None:
    with _lock:
        r.update(kw)
        _save()


def room(size: str, resolution: int) -> tuple[bool, float, float | None]:
    """(fits, GB it needs at batch 1 with Teach's headroom, GB free now; None without a GPU = it runs on the CPU)."""
    from engine import hw
    from engine.detector import VRAM_MARGIN_GB, _vram_needed_gb
    need = round((CLASSIFY_GB if size == "classify" else _vram_needed_gb(size, resolution, 1)) + VRAM_MARGIN_GB, 2)
    free = hw.free_memory_gb()
    return free is None or free >= need, need, None if free is None else round(free, 2)


def _wait_job(jid: str, r: dict, lo: float, hi: float) -> dict:
    _set(r, job=jid)
    if r["cancel"]:
        H.state["jobs"][jid]["stop"] = True
    while True:
        j = H.state["jobs"][jid]
        if j["total"]:
            _set(r, progress=round(lo + (hi - lo) * min(1.0, j["done"] / j["total"]), 3), text=j["text"])
        if j["finished"]:
            return j
        time.sleep(1)


def _run(r: dict) -> None:
    token = H.USER.set(r["user"])                            # the project's own folder: whoever sent it
    try:
        p = H.project(r["project"])
        if p.task == "classify":                             # its own trainer, which also exports
            from engine import classifier
            j = _wait_job(H.start_job("teach", lambda progress, stop: classifier.train(
                p, H.runs_dir() / r["run"], project=r["project"], progress=progress, should_stop=stop)), r, 0.0, 1.0)
            res = j["result"] or {}
            if r["cancel"] or res.get("status") == "stopped":
                return _set(r, state="cancelled", text="Cancelled", finished_at=time.time())
            if j["error"]:
                return _set(r, state="failed", error=j["error"], finished_at=time.time())
            return _set(r, state="done", progress=1.0, passed=bool(res.get("passed")), export_id=r["run"],
                        knowledge_score=res.get("knowledge_score"), text="Done", finished_at=time.time(),
                        sec_per_image_epoch=(time.time() - r["started_at"]) / max(1, r["frames"]))
        body = {"project": r["project"], "tasks": [t["id"] for t in p.sources], "name": r["run"], "size": r["size"],
                "epochs": r["epochs"], "resolution": r["resolution"], "aug": r.get("aug") or "strong",
                "compare_to": r.get("compare_to")}
        j = _wait_job(H.teach(body)["job"], r, 0.0, 0.9)
        res = j["result"] or {}
        if r["cancel"] or res.get("status") == "stopped":
            return _set(r, state="cancelled", text="Cancelled", finished_at=time.time())
        if j["error"]:
            return _set(r, state="failed", error=j["error"], finished_at=time.time())
        trained = time.time()
        _set(r, passed=bool(res.get("passed", False)), knowledge_score=res.get("knowledge_score"), compare=res.get("compare"),
             text="Preparing it for the stations",
             sec_per_image_epoch=(trained - r["started_at"]) / max(1, round(r["frames"] * 0.8)
                                                                   * ((res.get("training") or {}).get("epochs_run") or r["epochs"] or EPOCHS))
             if r["frames"] else None)
        j = _wait_job(H.export_model({"run": r["run"]})["job"], r, 0.9, 1.0)
        if r["cancel"]:
            return _set(r, state="cancelled", text="Cancelled", finished_at=time.time())
        if j["error"]:
            return _set(r, state="failed", error=j["error"], finished_at=time.time())
        _set(r, state="done", progress=1.0, export_id=r["run"], text="Done", finished_at=time.time())
    except Exception as e:                                   # noqa: BLE001 - the run fails, the queue goes on
        _set(r, state="failed", error=f"{type(e).__name__}: {e}", finished_at=time.time())
    finally:
        H.USER.reset(token)


def _work() -> None:
    while True:
        with _lock:
            r = next((x for x in _runs if x["state"] in ("queued", "waiting_for_room")), None)
        if r is None:
            _wake.wait(POLL_S)
            _wake.clear()
            continue
        fits, need, free = room(r["size"], r["resolution"])
        if not fits:
            _set(r, state="waiting_for_room", need_gb=need, free_gb=free,
                 text=f"Waiting for room on the graphics card: needs {need} GB, {free} GB free")
            _wake.wait(POLL_S)
            _wake.clear()
            continue
        _set(r, state="running", need_gb=need, free_gb=free, started_at=time.time(), text="Starting")
        _run(r)
