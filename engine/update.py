"""Update PartLabeler from GitHub, keeping everything the user made.

    partlabeler update [--check]        (Windows: double-click update_windows.bat; or the start screen's Update)

An update replaces only the app's own files, the ones in the GitHub repository. It never touches the projects
folder, data/, .venv, the models (Hugging Face cache) or ~/.partlabeler (Rivet's key). Before updating, every
project's labels (project.json and labels.sqlite, a consistent SQLite copy even while the app has it open) go to
<home>/.partlabeler/backups/<time>/, the last 5 kept, in case a new version's changes to the label database go
wrong. Frames and exports are not copied: updates never change them.

Two kinds of install:
- A git clone: `git merge --ff-only` of GitHub's main. Refused, with nothing changed, when app files were edited
  here or the copy has commits of its own (a developer checkout): those are merged by hand.
- An unzipped download: GitHub's ZIP of main is unpacked over the app. Files the previous version had and the new
  one dropped are removed, known from .partlabeler-version.json (written by each update); nothing else is. If a
  copy fails half-way, the files already replaced are put back.

Python packages are brought up to date only when pyproject.toml changed, the way the installer does it (same
extras, PyTorch pinned). From the start screen that waits for the next start (`finish_pending`, before any heavy
module loads), because Windows cannot replace files the running app has loaded. Colab gets the latest version in
its Step 3 (git pull + install.sh) every session.
"""
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
from importlib import metadata
from pathlib import Path

REPO = "buddi0812/partlabeler"
BRANCH = "main"
APP = Path(__file__).resolve().parent.parent
STAMP = ".partlabeler-version.json"           # zip installs: {"sha", "files"}
PENDING = ".partlabeler-update-pending"       # packages to bring up to date at the next start
PROTECTED = {".git", ".venv", "data", "projects", "graphify-out"}   # never written or removed by an update
KEEP_BACKUPS = 5


def _git(app: Path, *args, check=True) -> str:
    r = subprocess.run(["git", "-C", str(app), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r.stdout.rstrip()                            # not strip(): porcelain lines start with a space


def current(app: Path = APP) -> dict:
    """{"kind": "git" | "zip", "sha": commit or None, "changed": [app files edited here] (git only)}."""
    if (app / ".git").exists() and shutil.which("git"):
        changed = [l[3:] for l in _git(app, "status", "--porcelain", "--untracked-files=no").splitlines()]
        return {"kind": "git", "sha": _git(app, "rev-parse", "HEAD"), "changed": changed}
    try:
        sha = json.loads((app / STAMP).read_text(encoding="utf-8")).get("sha")
    except (OSError, ValueError):
        sha = None
    return {"kind": "zip", "sha": sha, "changed": []}


def _get(url: str, timeout: float = 15):
    import httpx
    r = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"Accept": "application/vnd.github+json"})
    r.raise_for_status()
    return r


def latest(repo: str = REPO, branch: str = BRANCH) -> dict:
    """GitHub's newest commit of the branch: {"sha", "date", "message"}."""
    c = _get(f"https://api.github.com/repos/{repo}/commits/{branch}").json()
    return {"sha": c["sha"], "date": c["commit"]["committer"]["date"], "message": c["commit"]["message"].split("\n")[0]}


def check(app: Path = APP, repo: str = REPO, branch: str = BRANCH) -> dict:
    """Is there a newer version? {"current", "latest", "available", "changes": [first lines], "can_update", "reason"}."""
    cur, new = current(app), latest(repo, branch)
    out = {"current": (cur["sha"] or "")[:7] or "unknown", "latest": new["sha"][:7], "date": new["date"],
           "available": cur["sha"] != new["sha"], "changes": [], "can_update": True, "reason": ""}
    if not out["available"]:
        return out
    if cur["sha"]:
        try:
            cmp = _get(f"https://api.github.com/repos/{repo}/compare/{cur['sha']}...{branch}").json()
            if cmp.get("status") in ("diverged", "behind"):
                out.update(can_update=False, available=False,
                           reason="This copy has changes of its own that are not on GitHub: update it with git by hand.")
            out["changes"] = [c["commit"]["message"].split("\n")[0] for c in cmp.get("commits", [])][-20:][::-1]
        except Exception:                                  # unknown to GitHub: a developer's own commits
            if cur["kind"] == "git":
                out.update(can_update=False, available=False,
                           reason="This copy has commits that are not on GitHub (a developer checkout): update it with git by hand.")
    if cur["changed"]:
        out.update(can_update=False, reason="Some app files were edited on this computer: " + ", ".join(cur["changed"][:5])
                   + ". Undo those edits (or update with git by hand) so nothing is overwritten.")
    return out


def backup(home: Path, keep: int = KEEP_BACKUPS) -> Path | None:
    """Copy every project's project.json and labels.sqlite to <home>/.partlabeler/backups/<time>/<project>/."""
    home = Path(home)
    projects = [f for f in home.iterdir() if (f / "project.json").exists()] if home.exists() else []
    if not projects:
        return None
    root = home / ".partlabeler" / "backups"
    dest = root / time.strftime("%Y%m%d_%H%M%S")
    for f in projects:
        d = dest / f.name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f / "project.json", d / "project.json")
        if (f / "labels.sqlite").exists():
            src, dst = sqlite3.connect(f / "labels.sqlite"), sqlite3.connect(d / "labels.sqlite")
            with dst:
                src.backup(dst)                           # consistent even while the app writes to it
            src.close(); dst.close()
    for old in sorted(p for p in root.iterdir() if p.is_dir())[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return dest


def _protected(rel: str) -> bool:
    return rel.replace("\\", "/").split("/", 1)[0] in PROTECTED


def _unpack(app: Path, data: bytes, sha: str, log) -> list[str]:
    """Put the ZIP's files over the app; returns the files written."""
    with zipfile.ZipFile(io.BytesIO(data)) as z, tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp)
        tops = [p for p in Path(tmp).iterdir()]
        src = tops[0] if len(tops) == 1 and tops[0].is_dir() else Path(tmp)      # GitHub: partlabeler-main/
        files = sorted(str(p.relative_to(src)).replace("\\", "/") for p in src.rglob("*") if p.is_file())
        files = [f for f in files if not _protected(f)]
        try:
            old = set(json.loads((app / STAMP).read_text(encoding="utf-8"))["files"])
        except (OSError, ValueError, KeyError):
            old = set()                                   # first update of an unzipped copy: nothing is removed
        saved = Path(tmp) / "_previous"
        done = []
        try:
            for rel in files:
                target = app / rel
                if target.exists():
                    (saved / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, saved / rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / rel, target)
                done.append(rel)
            for rel in sorted(old - set(files)):
                target = app / rel
                if not _protected(rel) and target.is_file():
                    (saved / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(target), saved / rel)
                    done.append(rel)
        except Exception:
            log("Copying failed; putting the previous files back")
            for rel in done:
                if (saved / rel).exists():
                    (app / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(saved / rel, app / rel)
                elif (app / rel).exists():
                    (app / rel).unlink()                  # a file the new version added
            raise
    (app / STAMP).write_text(json.dumps({"sha": sha, "files": files}, indent=0), encoding="utf-8")
    return files


def update(home: Path, app: Path = APP, repo: str = REPO, branch: str = BRANCH, packages_now: bool = True,
           log=print, download=None) -> dict:
    """Update the app (see the module notes). Returns {"from", "to", "backup", "packages"}; raises with a plain
    reason when it cannot update. `download(url) -> bytes` is for tests."""
    cur = current(app)
    if cur["changed"]:
        raise RuntimeError("Some app files were edited on this computer (" + ", ".join(cur["changed"][:5])
                           + "). Undo those edits, or update with git by hand, so nothing is overwritten.")
    pyproject = _pyproject(app)
    same = {"from": (cur["sha"] or "")[:7], "to": (cur["sha"] or "")[:7], "backup": None, "packages": "unchanged"}
    if cur["kind"] == "git":
        url = repo if "://" in repo or Path(repo).exists() else f"https://github.com/{repo}.git"
        log("Downloading the new version (git)")
        _git(app, "fetch", "--quiet", url, branch)
        new = _git(app, "rev-parse", "FETCH_HEAD")
        if new == cur["sha"]:
            return same
        if subprocess.run(["git", "-C", str(app), "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"]).returncode:
            raise RuntimeError("This copy has commits that are not on GitHub (a developer checkout): update it with git by hand.")
        log("Backing up the labels of every project")
        saved = backup(home)
        log("Updating the app files")
        _git(app, "merge", "--ff-only", "--quiet", "FETCH_HEAD")
    else:
        new = latest(repo, branch)["sha"]
        if new == cur["sha"]:
            return same
        log("Downloading the new version (zip)")
        data = (download or (lambda u: _get(u, timeout=120).content))(f"https://codeload.github.com/{repo}/zip/{new}")
        log("Backing up the labels of every project")
        saved = backup(home)
        log("Updating the app files")
        _unpack(app, data, new, log)
    packages = "unchanged"
    if _pyproject(app) != pyproject:
        if packages_now:
            log("Updating the Python packages")
            packages = "updated" if sync_packages(app, log) else "failed"
        else:
            (app / PENDING).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            packages = "at next start"
    return {"from": (cur["sha"] or "unknown")[:7], "to": new[:7], "backup": str(saved) if saved else None, "packages": packages}


def _pyproject(app: Path) -> str:
    """pyproject.toml with line endings evened out (Windows checkouts have CRLF, GitHub's ZIP has LF)."""
    f = app / "pyproject.toml"
    return f.read_text(encoding="utf-8").replace("\r\n", "\n") if f.exists() else ""


def find_uv() -> str | None:
    """uv, as the installer finds it: on PATH or in the usual install folders."""
    found = shutil.which("uv")
    if found:
        return found
    home, exe = Path.home(), "uv.exe" if os.name == "nt" else "uv"
    cands = [home / ".local" / "bin" / exe, home / ".cargo" / "bin" / exe]
    if os.environ.get("LOCALAPPDATA"):
        cands.append(Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WinGet" / "Links" / exe)
    for base in filter(None, (os.environ.get("APPDATA"), os.environ.get("LOCALAPPDATA"))):
        for py in sorted((Path(base) / "Python").glob("*")) + sorted((Path(base) / "Programs" / "Python").glob("*")):
            cands.append(py / "Scripts" / exe)
    return next((str(c) for c in cands if c.exists()), None)


def sync_packages(app: Path = APP, log=print) -> bool:
    """Install the app's packages as the installer does: same extras (Teach & Transfer, developer tools, if
    present), PyTorch pinned to the installed build so no package can swap it."""
    extras = [e for e, mod in (("teach", "rfdetr"), ("dev", "pytest")) if importlib.util.find_spec(mod)]
    target = "." + (f"[{','.join(extras)}]" if extras else "")
    pins = []
    for p in ("torch", "torchvision"):
        try:
            pins.append(f"{p}=={metadata.version(p).split('+')[0]}")
        except metadata.PackageNotFoundError:
            pass
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(pins) + "\n")
    uv = find_uv()
    cmd = ([uv, "pip", "install", "--python", sys.executable, "-e", target, "--constraints", f.name] if uv
           else [sys.executable, "-m", "pip", "install", "-q", "-e", target, "-c", f.name])
    try:
        ok = subprocess.run(cmd, cwd=app).returncode == 0
    finally:
        os.unlink(f.name)
    if not ok:
        log("Updating the Python packages failed. Run the installer again (install_windows.bat or install.sh): "
            "it is safe to rerun and keeps your projects.")
    return ok


def finish_pending(app: Path = APP, log=print) -> None:
    """At the app's start: packages an update from the start screen left for now (see the module notes)."""
    if (app / PENDING).exists():
        log("Finishing the update: bringing the Python packages up to date…")
        if sync_packages(app, log):
            (app / PENDING).unlink()
