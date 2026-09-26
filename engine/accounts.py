"""Local accounts: who is signed in to the app on this computer, and each person's settings.

Stored in ~/.partlabeler/accounts.json (the folder can be moved with PARTLABELER_CONFIG). Passwords are kept as
scrypt hashes and sign-in tokens as SHA-256 hashes, so the file holds no password and no usable token.
An account keeps people's preferences and projects folders apart; it does not encrypt anything: whoever can read
this computer's files can read the projects. Forgotten password: `partlabeler account reset NAME` on this computer.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

ACCENTS = ("teal", "blue", "indigo", "violet", "rose", "orange", "green", "graphite")
EXPORT_FORMATS = ("", "yolo", "coco", "cvat", "voc", "labelstudio", "folders", "csv")
# every preference with its default; set_prefs checks each against its default's type (and the lists below)
DEFAULTS = {
    "theme": "light",          # light | dark | system (follow the computer)
    "accent": "teal",          # colour palette, one of ACCENTS
    "motion": True,            # animations
    "rivet": True,             # show Rivet, the helper robot
    "tips": True,              # Rivet's "Did you know?" bubbles
    "hints": True,             # the first-steps hint over the frame in the annotator
    "toasts": True,            # pop-up messages (off: only problems pop up; the bell keeps everything)
    "update_check": True,      # look for a new version on GitHub
    "track_n": 20,             # frames T and R track
    "brush": 12,               # outline brush size
    "export_format": "",       # the format picked first in export dialogs ("" = the project's first)
    "confirmed_only": False,   # export only confirmed frames, ticked by default
    "every": 5,                # new tasks: keep every Nth video frame
    "quality": 95,             # new tasks: JPEG quality
    "lossless": False,         # new tasks: lossless WebP frames
    "segment_size": 0,         # new tasks: frames per job (0 = one job)
    "projects": "",            # projects folder ("" = the app's default folder)
}
CHOICES = {"theme": ("light", "dark", "system"), "accent": ACCENTS, "export_format": EXPORT_FORMATS}
RANGES = {"track_n": (1, 10000), "brush": (1, 120), "every": (1, 1000), "quality": (5, 100), "segment_size": (0, 1000000)}
USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
MIN_PASSWORD = 8
KEEP_S, SHORT_S = 30 * 86400, 12 * 3600          # "keep me signed in", and a browser session


def config_dir() -> Path:
    return Path(os.environ.get("PARTLABELER_CONFIG") or Path.home() / ".partlabeler")


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32).hex()


def _token_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Accounts:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else config_dir() / "accounts.json"
        self.lock = threading.Lock()
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.data.setdefault("users", {})
        self.data.setdefault("sessions", {})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)                    # never a half-written file

    def _key(self, username: str) -> str | None:
        """The stored name for `username`, matched without case (Ana and ana are one account)."""
        low = str(username or "").strip().lower()
        return next((u for u in self.data["users"] if u.lower() == low), None)

    # ---- accounts ------------------------------------------------------------------------------
    def users(self) -> list[dict]:
        return [{"username": u, "name": d.get("name") or u} for u, d in sorted(self.data["users"].items())]

    def create(self, username: str, password: str, name: str = "") -> dict:
        username = str(username or "").strip()
        if not USERNAME.match(username):
            raise ValueError("A username is 1 to 32 letters, digits, dots, dashes or underscores, starting with a letter or digit")
        if len(password or "") < MIN_PASSWORD:
            raise ValueError(f"Use a password of at least {MIN_PASSWORD} characters")
        with self.lock:
            if self._key(username):
                raise ValueError(f"There is already an account called {username}")
            salt = secrets.token_bytes(16)
            self.data["users"][username] = {"name": " ".join(str(name).split())[:60], "salt": salt.hex(),
                                            "hash": _hash(password, salt), "created": time.time(), "prefs": {}}
            self._save()
        return {"username": username, "name": self.data["users"][username]["name"] or username}

    def verify(self, username: str, password: str) -> str | None:
        """The account's username when the password is right, else None (same work either way)."""
        key = self._key(username)
        user = self.data["users"].get(key) if key else None
        salt = bytes.fromhex(user["salt"]) if user else b"\0" * 16
        ok = hmac.compare_digest(_hash(password or "", salt), user["hash"] if user else "0" * 64)
        return key if user and ok else None

    def set_password(self, username: str, password: str) -> None:
        if len(password or "") < MIN_PASSWORD:
            raise ValueError(f"Use a password of at least {MIN_PASSWORD} characters")
        with self.lock:
            key = self._key(username)
            if not key:
                raise KeyError(f"No account called {username}")
            salt = secrets.token_bytes(16)
            self.data["users"][key].update(salt=salt.hex(), hash=_hash(password, salt))
            self.data["sessions"] = {k: s for k, s in self.data["sessions"].items() if s["user"] != key}
            self._save()

    def rename(self, username: str, name: str) -> dict:
        with self.lock:
            self.data["users"][username]["name"] = " ".join(str(name).split())[:60]
            self._save()
        return self.profile(username)

    def delete(self, username: str) -> None:
        """Remove the account and its sign-ins; its projects stay where they are."""
        with self.lock:
            key = self._key(username)
            if not key:
                raise KeyError(f"No account called {username}")
            del self.data["users"][key]
            self.data["sessions"] = {k: s for k, s in self.data["sessions"].items() if s["user"] != key}
            self._save()

    def profile(self, username: str) -> dict:
        d = self.data["users"][username]
        return {"username": username, "name": d.get("name") or username, "created": d.get("created")}

    # ---- sign-in -------------------------------------------------------------------------------
    def sign_in(self, username: str, keep: bool = False) -> str:
        token, now = secrets.token_urlsafe(32), time.time()
        with self.lock:
            self.data["sessions"] = {k: s for k, s in self.data["sessions"].items() if s["expires"] > now}
            self.data["sessions"][_token_key(token)] = {"user": username, "expires": now + (KEEP_S if keep else SHORT_S)}
            self._save()
        return token

    def user_for(self, token: str | None) -> str | None:
        s = self.data["sessions"].get(_token_key(token)) if token else None
        return s["user"] if s and s["expires"] > time.time() and s["user"] in self.data["users"] else None

    def sign_out(self, token: str | None) -> None:
        with self.lock:
            if token and self.data["sessions"].pop(_token_key(token), None):
                self._save()

    # ---- preferences ---------------------------------------------------------------------------
    def prefs(self, username: str | None) -> dict:
        stored = self.data["users"].get(username, {}).get("prefs", {}) if username else {}
        return {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}

    def set_prefs(self, username: str, changes: dict) -> dict:
        clean = {}
        for k, v in (changes or {}).items():
            if k not in DEFAULTS:
                raise ValueError(f"Unknown setting {k!r}")
            want = type(DEFAULTS[k])
            if want is int and not isinstance(v, bool):
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    raise ValueError(f"{k} must be a whole number") from None
                lo, hi = RANGES[k]
                v = max(lo, min(hi, v))
            elif not isinstance(v, want):
                raise ValueError(f"{k} must be {want.__name__}")
            if k in CHOICES and v not in CHOICES[k]:
                raise ValueError(f"{k} must be one of: {', '.join(c or '(project default)' for c in CHOICES[k])}")
            clean[k] = v.strip() if isinstance(v, str) else v
        with self.lock:
            self.data["users"][username].setdefault("prefs", {}).update(clean)
            self._save()
        return self.prefs(username)
