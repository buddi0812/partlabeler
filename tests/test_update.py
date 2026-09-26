"""Updating from GitHub keeps the user's work (engine/update.py): a git clone and an unzipped copy, against local
stand-ins for GitHub (a git repository, ZIP bytes), with projects inside the app folder as run_windows.bat keeps them."""
import io
import json
import shutil
import sqlite3
import subprocess
import zipfile

import pytest

from engine import update as up

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main", *args],
                   cwd=cwd, check=True, capture_output=True)


def commit(repo, files: dict, msg: str):
    for name, text in files.items():
        p = repo / name
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)


def user_project(app):
    """A project with labels, the way the app stores it (data that must survive every update)."""
    p = app / "projects" / "line2"
    p.mkdir(parents=True)
    (p / "project.json").write_text('{"name": "line2"}')
    db = sqlite3.connect(p / "labels.sqlite")
    with db:
        db.execute("CREATE TABLE boxes (item INTEGER, obj INTEGER)")
        db.execute("INSERT INTO boxes VALUES (1, 2)")
    db.close()
    return p


def test_git_clone_updates_and_keeps_projects(tmp_path):
    remote = tmp_path / "github"
    remote.mkdir()
    git(remote, "init", "-q")
    commit(remote, {"pyproject.toml": "v1", "app.py": "one", "old.py": "x", ".gitignore": "projects/\n"}, "v1")
    git(tmp_path, "clone", "-q", str(remote), "app")
    app = tmp_path / "app"
    proj = user_project(app)
    commit(remote, {"app.py": "two", "old.py": None}, "v2")

    res = up.update(app / "projects", app=app, repo=str(remote), packages_now=False, log=lambda t: None)
    assert (app / "app.py").read_text() == "two" and not (app / "old.py").exists() and res["packages"] == "unchanged"
    assert (proj / "project.json").exists() and sqlite3.connect(proj / "labels.sqlite").execute("SELECT * FROM boxes").fetchall() == [(1, 2)]
    backed = sqlite3.connect(next((app / "projects" / ".partlabeler" / "backups").glob("*/line2/labels.sqlite")))
    assert backed.execute("SELECT * FROM boxes").fetchall() == [(1, 2)]

    commit(remote, {"pyproject.toml": "v3"}, "v3: new package")        # new packages wait for the next start
    assert up.update(app / "projects", app=app, repo=str(remote), packages_now=False, log=lambda t: None)["packages"] == "at next start"
    assert (app / up.PENDING).exists()

    (app / "app.py").write_text("edited here")                          # never overwrite a person's edits
    commit(remote, {"app.py": "four"}, "v4")
    with pytest.raises(RuntimeError, match="edited on this computer"):
        up.update(app / "projects", app=app, repo=str(remote), log=lambda t: None)
    assert (app / "app.py").read_text() == "edited here"

    (app / "app.py").write_text("two")                                  # a developer's own commit: refused too
    git(app, "checkout", "-q", "app.py")
    commit(app, {"mine.py": "local work"}, "my own commit")
    with pytest.raises(RuntimeError, match="not on GitHub"):
        up.update(app / "projects", app=app, repo=str(remote), log=lambda t: None)
    assert (app / "mine.py").exists()


def zipped(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(f"partlabeler-main/{name}", text)
    return buf.getvalue()


def test_unzipped_copy_updates_and_keeps_projects(tmp_path, monkeypatch):
    app = tmp_path / "partlabeler-main"
    app.mkdir()
    (app / "pyproject.toml").write_text("v1")
    (app / "app.py").write_text("one")
    (app / "old.py").write_text("x")
    proj = user_project(app)
    (app / "data").mkdir()
    (app / "data" / "mine.txt").write_text("my data")
    versions = iter([{"sha": "a" * 40}, {"sha": "b" * 40}])
    monkeypatch.setattr(up, "latest", lambda repo, branch: next(versions))

    v1 = zipped({"pyproject.toml": "v1", "app.py": "one", "old.py": "x", "ui/page.js": "p1"})
    res = up.update(app / "projects", app=app, packages_now=False, log=lambda t: None, download=lambda url: v1)
    assert res["to"] == "aaaaaaa" and json.loads((app / up.STAMP).read_text())["sha"] == "a" * 40

    v2 = zipped({"pyproject.toml": "v1", "app.py": "two", "ui/page.js": "p2", "data/mine.txt": "overwritten?"})
    res = up.update(app / "projects", app=app, packages_now=False, log=lambda t: None, download=lambda url: v2)
    assert (app / "app.py").read_text() == "two" and (app / "ui/page.js").read_text() == "p2"
    assert not (app / "old.py").exists()                                 # dropped by the new version
    assert (app / "data" / "mine.txt").read_text() == "my data"          # protected folders are never written
    assert (proj / "project.json").exists() and (proj / "labels.sqlite").exists()


def test_a_failed_copy_puts_the_previous_files_back(tmp_path, monkeypatch):
    app = tmp_path / "app"
    app.mkdir()
    (app / "a.py").write_text("old a")
    (app / "z.py").mkdir()                                              # a folder where the new version has a file
    monkeypatch.setattr(up, "latest", lambda repo, branch: {"sha": "c" * 40})
    with pytest.raises(OSError):
        up.update(tmp_path / "none", app=app, log=lambda t: None, download=lambda url: zipped({"a.py": "new a", "b.py": "b", "z.py": "z"}))
    assert (app / "a.py").read_text() == "old a" and not (app / "b.py").exists() and not (app / up.STAMP).exists()


def test_backups_keep_the_last_five(tmp_path, monkeypatch):
    user_project(tmp_path)
    stamps = iter(f"2026010{i}_000000" for i in range(1, 8))
    monkeypatch.setattr(up.time, "strftime", lambda fmt: next(stamps))
    for _ in range(7):
        up.backup(tmp_path / "projects")
    kept = sorted(p.name for p in (tmp_path / "projects" / ".partlabeler" / "backups").iterdir())
    assert kept == [f"2026010{i}_000000" for i in range(3, 8)]
