"""Cycle recordings: videos the box's cameras already made, browsed by station and camera so a project can be
started straight from them (no need to know where files live on disk). Read-only: PartLabeler never writes here.

Two sources, picked by `~/.partlabeler/app.json`:
- `{"recordings_api": "http://127.0.0.1:<port>", "recordings_token": "..."}`: another local system's recordings
  API (contract: GET <url>/recordings?station_id=&camera_id=&from=&to=&before=&limit= -> {"stations": [...],
  "recordings": [...]}), sent with header `X-Hawk-Token: <recordings_token>` (read fresh from app.json on every
  call, never logged, never put in a URL - that system regenerates it on its own install/repair/update).
  Local only: never called when that URL is not on 127.0.0.1/localhost.
- Otherwise, or when the API cannot be reached: a plain folder, `{"recordings_dir": "D:\\recordings"}` (default
  "recordings" next to the projects folder), laid out as `<dir>/<station>/<camera>/*.mp4` with an optional
  `<dir>/<station>/recordings.json` for ROI and cycle markers:
    {"cameras": {"<camera folder name>": {"roi": [x, y, w, h]}},
     "recordings": {"<file name>": {"cycles": [{"t_start_s": 12.0, "t_end_s": 34.5, "verdict": "pass"}]}}}
"""
import json
import math
import time
from pathlib import Path

import av
import httpx

from engine.accounts import app_config, box_managed, default_home

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_PROBE_CACHE: dict[str, tuple[float, dict]] = {}             # path -> (mtime, probe result)


def _dir() -> Path:
    cfg = app_config()
    return Path(cfg.get("recordings_dir") or default_home().parent / "recordings")


def source() -> dict:
    cfg = app_config()
    api = cfg.get("recordings_api")
    if api and _is_local(api):
        return {"mode": "api", "base_url": api.rstrip("/")}
    return {"mode": "dir", "dir": _dir()}


def _is_local(url: str) -> bool:
    from urllib.parse import urlparse
    return urlparse(url).hostname in ("127.0.0.1", "localhost", "::1")


def valid_crop(crop) -> bool:
    """A camera crop [x, y, w, h], shares 0..1 of its full frame: 4 numbers (not bool - True/False are ints in
    Python but not here, matching hawk), x,y >= 0, w,h > 0, x+w <= 1, y+h <= 1 (1e-6 slack) - the same rule
    hawk-api's model import checks (cross-checked against hawk's own crop.py, 2026-10-09: 0 mismatches over
    200k random crops), so an export built here is never refused there."""
    if not (isinstance(crop, (list, tuple)) and len(crop) == 4):
        return False
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in crop):
        return False
    x, y, w, h = crop
    eps = 1e-6
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= 1 + eps and y + h <= 1 + eps


def crop_rect(crop: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    """A valid_crop() share rectangle as pixels (x1, y1, x2, y2) of a w x h frame: floor the start, ceil the
    end, clipped to the frame, x1/y1 clamped to w-1/h-1 so the cut is never empty - hawk-vision uses this
    identical rule to cut and to map boxes back, so a frame cropped here lines up with what the box crops at
    inference."""
    x, y, cw, ch = crop
    x1 = max(0, min(w - 1, math.floor(x * w)))
    y1 = max(0, min(h - 1, math.floor(y * h)))
    x2 = max(x1 + 1, min(w, math.ceil((x + cw) * w)))
    y2 = max(y1 + 1, min(h, math.ceil((y + ch) * h)))
    return x1, y1, x2, y2


def probe(path: Path) -> dict:
    """Duration, size, fps and codec of a video, cached by path + mtime (re-probed if the file changes)."""
    mtime = path.stat().st_mtime
    cached = _PROBE_CACHE.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        info = {"duration_s": round(float(c.duration / 1_000_000) if c.duration else (float(v.duration * v.time_base) if v.duration else 0), 1),
                "width": v.width, "height": v.height, "fps": round(float(v.average_rate or v.guessed_rate or 0), 2),
                "codec": v.codec_context.name}
    _PROBE_CACHE[str(path)] = (mtime, info)
    return info


def thumbnail(path: Path, cache_dir: Path, at_s: float = 0) -> Path:
    """One frame of the video as a cached JPEG, named by the video's path and change time (stale copies are
    overwritten, old ones left for the OS to reclaim: this is a cache, not state)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{abs(hash((str(path), path.stat().st_mtime, at_s)))}.jpg"
    if not out.exists():
        with av.open(str(path)) as c:
            v = c.streams.video[0]
            if at_s:
                c.seek(int(at_s / v.time_base), stream=v)
            for f in c.decode(v):
                f.to_image().convert("RGB").save(out, "JPEG", quality=85)
                break
    return out


def _dir_stations(d: Path) -> list[dict]:
    if not d.is_dir():
        return []
    stations = []
    for sd in sorted(p for p in d.iterdir() if p.is_dir()):
        meta = {}
        if (sd / "recordings.json").exists():
            try:
                meta = json.loads((sd / "recordings.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        cams = meta.get("cameras", {})
        stations.append({"id": sd.name, "name": sd.name,
                          "cameras": [{"id": cd.name, "name": cd.name, "roi": cams.get(cd.name, {}).get("roi")}
                                      for cd in sorted(p for p in sd.iterdir() if p.is_dir())]})
    return stations


def _dir_recordings(d: Path, station_id: str | None, camera_id: str | None, limit: int,
                     from_: str | None = None, to: str | None = None, before: str | None = None) -> list[dict]:
    out = []
    for sd in ([d / station_id] if station_id else sorted(p for p in d.iterdir() if p.is_dir())) if d.is_dir() else []:
        if not sd.is_dir():
            continue
        meta = {}
        if (sd / "recordings.json").exists():
            try:
                meta = json.loads((sd / "recordings.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        for cd in ([sd / camera_id] if camera_id else sorted(p for p in sd.iterdir() if p.is_dir())) if sd.is_dir() else []:
            if not cd.is_dir():
                continue
            for f in sorted(p for p in cd.iterdir() if p.suffix.lower() in VIDEO_EXTS):
                try:
                    info = probe(f)
                except Exception:
                    continue                                 # unreadable file: skip rather than fail the whole list
                out.append({"id": f"{sd.name}/{cd.name}/{f.name}", "station_id": sd.name, "camera_id": cd.name,
                            "path": str(f), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(f.stat().st_mtime)),
                            "cycles": meta.get("recordings", {}).get(f.name, {}).get("cycles", []), **info})
    if from_:
        out = [r for r in out if r["started_at"] >= from_]
    if to:
        out = [r for r in out if r["started_at"] <= to]
    if before:
        out = [r for r in out if r["started_at"] < before]
    out.sort(key=lambda r: r["started_at"], reverse=True)
    return out[:limit]


def resolve(rid: str) -> Path | None:
    """The local file for a recording id ("station/camera/file"), or None if it doesn't exist or isn't one of
    ours (also rejects an id that tries to escape the recordings folder, e.g. with "..")."""
    try:
        sid, cid, fname = rid.split("/", 2)
    except ValueError:
        return None
    src = source()
    if src["mode"] == "dir":
        d = src["dir"].resolve()
        p = (d / sid / cid / fname).resolve()
        try:
            p.relative_to(d)
        except ValueError:
            return None
        return p if p.is_file() else None
    # a camera can keep up to 144 segments/day: list_all's default limit (100) could miss an older one.
    for r in list_all(sid, cid, limit=500)["recordings"]:
        if r["id"] == rid:
            return Path(r["path"])
    return None


def list_all(station_id: str | None = None, camera_id: str | None = None, limit: int = 100,
             from_: str | None = None, to: str | None = None, before: str | None = None) -> dict:
    """{"stations": [...], "recordings": [...]}, from the API if configured and reachable, else the folder.
    from_/to: recordings overlapping that ISO window. before: recordings that started earlier than that ISO
    timestamp (for paging - pass the previous page's last item's "started_at").

    On a box (box_managed()) an unreachable API never falls back to the folder: the folder's names are this
    computer's own, not the real station registry (a stray one can even be named after camera ids), so
    silently reading it would show a customer internal identifiers that happen to look plausible but are
    wrong. Standalone, the folder fallback stays (genuinely useful there)."""
    src = source()
    if src["mode"] == "api":
        try:
            headers = {}
            token = app_config().get("recordings_token")
            if token:
                headers["X-Hawk-Token"] = token
            with httpx.Client(timeout=5) as c:
                r = c.get(f"{src['base_url']}/recordings",
                          params={k: v for k, v in (("station_id", station_id), ("camera_id", camera_id),
                                                     ("limit", limit), ("from", from_), ("to", to),
                                                     ("before", before)) if v},
                          headers=headers)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError:
            if box_managed():
                raise RuntimeError("Hawk-AI isn't answering right now - try again in a moment.") from None
            # standalone, box's API not up right now: fall back to the folder
    d = src["dir"] if src["mode"] == "dir" else _dir()
    return {"stations": _dir_stations(d), "recordings": _dir_recordings(d, station_id, camera_id, limit, from_, to, before)}
