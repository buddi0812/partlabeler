"""Notification history: every event worth finding later (created, confirmed, deleted, exported, errors).

One store per projects folder in the web app (persisted, shared by the start screen and every open
project) or one in-memory store per notebook widget. Toasts vanish; these stay until cleared.
A repeated event (e.g. confirming frame after frame) updates the newest entry instead of adding one.
"""
import json
import threading
import time
import uuid
from pathlib import Path

LEVELS = ("info", "success", "warning", "error")


class Notifications:
    def __init__(self, path=None, limit: int = 300, on_change=None):
        self.path, self.limit, self.on_change = (Path(path) if path else None), limit, on_change
        self.items: list[dict] = []
        self.read_upto = 0.0
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.items, self.read_upto = data.get("items", [])[: limit], data.get("read_upto", 0.0)
            except (ValueError, OSError):
                pass

    def add(self, level: str, title: str, detail: str = "", project: str | None = None, action: dict | None = None,
            key: str | None = None, plural: str | None = None, window_s: float = 120) -> dict:
        """Record an event. With `key`, an event of the same key and project within `window_s` of the newest
        entry is merged into it: its count goes up and `plural` (formatted with {n}) becomes the title."""
        level = level if level in LEVELS else "info"
        now = time.time()
        with self._lock:
            last = self.items[0] if self.items else None
            if (key and last and last.get("key") == key and last.get("project") == project
                    and now - last["time"] <= window_s):
                last.update(count=last.get("count", 1) + 1, time=now, detail=detail or last["detail"],
                            action=action or last.get("action"))
                if plural:
                    last["title"] = plural.format(n=last["count"])
                item = last
            else:
                item = {"id": uuid.uuid4().hex[:12], "time": now, "level": level, "title": title, "detail": detail,
                        "project": project, "action": action, "key": key, "count": 1}
                self.items.insert(0, item)
                del self.items[self.limit:]
            self._save()
            unread = self._unread()
        if self.on_change:
            self.on_change({"type": "notify", "item": item, "unread": unread})
        return item

    def snapshot(self) -> dict:
        with self._lock:
            return {"items": list(self.items), "unread": self._unread()}

    def mark_read(self) -> None:
        with self._lock:
            self.read_upto = time.time()
            self._save()
        if self.on_change:
            self.on_change({"type": "notifications", **self.snapshot()})

    def clear(self) -> None:
        with self._lock:
            self.items, self.read_upto = [], time.time()
            self._save()
        if self.on_change:
            self.on_change({"type": "notifications", **self.snapshot()})

    def _unread(self) -> int:
        return sum(1 for n in self.items if n["time"] > self.read_upto)

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"items": self.items, "read_upto": self.read_upto}), encoding="utf-8")
            tmp.replace(self.path)
