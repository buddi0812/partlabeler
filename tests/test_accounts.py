"""Accounts and preferences (engine/accounts.py), and moving unfinished tasks between projects as zips."""
import json

import pytest

from engine.accounts import DEFAULTS, Accounts
from engine.project import Project, backup_project, restore_project
from tests.test_project import CLASSES, images  # noqa: F401  (fixture)
from tests.test_tasks import make_video


def test_accounts_passwords_sessions_and_prefs(tmp_path):
    a = Accounts(tmp_path / "accounts.json")
    a.create("Ana", "correct horse", "Ana Lima")
    with pytest.raises(ValueError):
        a.create("ana", "another password")                       # names ignore case
    with pytest.raises(ValueError):
        a.create("bo", "short")
    with pytest.raises(ValueError):
        a.create("../x", "long enough pw")
    assert a.verify("ANA", "correct horse") == "Ana" and a.verify("Ana", "wrong one") is None and a.verify("nobody", "x") is None
    token = a.sign_in("Ana", keep=True)
    stored = (tmp_path / "accounts.json").read_text(encoding="utf-8")
    assert token not in stored and "correct horse" not in stored   # only hashes on disk
    b = Accounts(tmp_path / "accounts.json")                        # survives a restart
    assert b.user_for(token) == "Ana" and b.user_for("made-up") is None and b.user_for(None) is None
    assert b.prefs("Ana") == DEFAULTS
    p = b.set_prefs("Ana", {"theme": "dark", "accent": "violet", "tips": False, "track_n": "50000", "projects": " D:/labels "})
    assert (p["theme"], p["accent"], p["tips"], p["track_n"], p["projects"]) == ("dark", "violet", False, 10000, "D:/labels")
    for bad in ({"theme": "neon"}, {"tips": "no"}, {"nope": 1}, {"quality": "high"}):
        with pytest.raises(ValueError):
            b.set_prefs("Ana", bad)
    b.set_password("Ana", "a new password")                          # signs out everywhere
    assert b.user_for(token) is None and b.verify("Ana", "a new password") == "Ana"
    t2 = b.sign_in("Ana")
    b.sign_out(t2)
    assert b.user_for(t2) is None
    b.delete("Ana")
    assert b.users() == [] and json.loads((tmp_path / "accounts.json").read_text())["users"] == {}


def test_backup_one_task_then_open_it_or_add_it_to_another_project(tmp_path, images):
    p = Project.create(tmp_path / "proj", CLASSES, colors=["#ff0000", "#00ff00", "#0000ff"])
    p.add_source(video=make_video(tmp_path / "a.mp4", n=10), every=5)           # items 0-1
    p.add_source(images=images, subset="train", segment_size=2)               # items 2-4, jobs 2 and 3
    p.put(1, 1, 0, (1, 1, 5, 5))
    p.put(3, 2, 2, (2, 2, 6, 6))
    p.put(4, 3, 1, (3, 3, 7, 7))
    p.set_reviewed(3)
    p.set_job(3, stage="validation")
    info = backup_project(tmp_path / "proj", tmp_path / "task.zip", tasks=[2])
    assert info["items"] == 3 and info["tasks"] == ["photos"]
    with pytest.raises(KeyError):
        backup_project(tmp_path / "proj", tmp_path / "none.zip", tasks=[9])

    alone = Project(restore_project(tmp_path / "task.zip", tmp_path / "home"))  # a project of its own
    assert [t["name"] for t in alone.sources] == ["photos"] and len(alone.items) == 3
    assert alone.boxes(1)[0]["box"] == [2, 2, 6, 6] and alone.is_reviewed(1) and alone.boxes(2)[0]["cls"] == 1
    assert [j["stage"] for j in alone.jobs()] == ["annotation", "validation"]

    q = Project.create(tmp_path / "other", ["grill", "bolt"])                # another order, one label missing
    q.add_source(video=make_video(tmp_path / "b.mp4", n=10), every=5)
    q.put(0, 1, 1, (0, 0, 2, 2))
    res = q.add_tasks_from(tmp_path / "task.zip")
    assert res == {"tasks": ["photos"], "items": 3, "labels_added": ["left_drl", "right_drl"]}
    assert q.classes == ["grill", "bolt", "left_drl", "right_drl"]
    assert len(q.items) == 5 and q.sources[-1]["subset"] == "train"
    assert q.boxes(3)[0]["cls"] == 0 and q.boxes(3)[0]["box"] == [2, 2, 6, 6] and q.is_reviewed(3)   # grill stays grill
    assert q.boxes(4)[0]["cls"] == 3 and q.boxes(0)[0]["obj"] not in {b["obj"] for b in q.boxes(3) + q.boxes(4)}
    assert [(j["task"], j["stage"]) for j in q.jobs()][-2:] == [(2, "annotation"), (2, "validation")]
    assert len({j["id"] for j in q.jobs()}) == 3 and not any(x.name.startswith(".import_") for x in q.folder.iterdir())
    assert q.image(2).size == (80, 60)                                             # pictures moved into the project

    seg = Project.create(tmp_path / "seg", CLASSES, task="segment")
    with pytest.raises(ValueError, match="segmentation"):
        seg.add_tasks_from(tmp_path / "task.zip")
