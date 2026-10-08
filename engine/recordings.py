"""Cycle recordings: videos the box's cameras already made, browsed by station and camera so a project can be
started straight from them (no need to know where files live on disk). Read-only: PartLabeler never writes here.

Two sources, picked by `~/.partlabeler/app.json`:
- `{"recordings_api": "http://127.0.0.1:<port>"}`: another local system's recordings API (contract: GET
  <url>/recordings?station_id=&camera_id=&limit=&before= -> {"stations": [...], "recordings": [...]}).
  Local only: never called when that URL is not on 127.0.0.1/localhost.
- Otherwise, or when the API cannot be reached: a plain folder, `{"recordings_dir": "D:\\recordings"}` (default
  "recordings" next to the projects folder), laid out as `<dir>/<station>/<camera>/*.mp4` with an optional
  `<dir>/<station>/recordings.json` for ROI and cycle markers:
    {"cameras": {"<camera folder name>": {"roi": [x, y, w, h]}},
     "recordings": {"<file name>": {"cycles": [{"t_start_s": 12.0, "t_end_s": 34.5, "verdict": "pass"}]}}}
"""
import json
import time
from pathlib import Path

import av
import httpx

from engine.accounts import app_config, default_home

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


def _dir_recordings(d: Path, station_id: str | None, camera_id: str | None, limit: int) -> list[dict]:
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
    for r in list_all(sid, cid)["recordings"]:
        if r["id"] == rid:
            return Path(r["path"])
    return None


def list_all(station_id: str | None = None, camera_id: str | None = None, limit: int = 100) -> dict:
    """{"stations": [...], "recordings": [...]}, from the API if configured and reachable, else the folder."""
    src = source()
    if src["mode"] == "api":
        try:
            with httpx.Client(timeout=5) as c:
                r = c.get(f"{src['base_url']}/recordings",
                          params={k: v for k, v in (("station_id", station_id), ("camera_id", camera_id), ("limit", limit)) if v})
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError:
            pass                                             # box's API not up right now: fall back to the folder
    d = src["dir"] if src["mode"] == "dir" else _dir()
    return {"stations": _dir_stations(d), "recordings": _dir_recordings(d, station_id, camera_id, limit)}
