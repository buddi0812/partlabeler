"""Reference profiles of lit parts and the check against them (engine/lampcheck.py), on synthetic lamp strips."""
import numpy as np

from engine import lampcheck as L
from tests.test_project import video  # noqa: F401  (fixture)

BOX = (40, 40, 160, 60)


def frame(level=255, dead=None, dim=1.0, colour=(255, 255, 255), seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((120, 200, 3), 60, np.uint8) + rng.integers(0, 20, (120, 200, 3), dtype=np.uint8)
    strip = np.array(colour, float) * dim * level / 255
    img[45:55, 45:155] = strip.astype(np.uint8)            # the lit strip inside the box
    if dead:
        img[45:55, dead[0]:dead[1]] = 70                   # a dead section
    outline = np.zeros((120, 200), bool)
    outline[45:55, 45:155] = True
    return img, outline


def reference():
    frames = [frame(seed=s) for s in range(30)]
    return L.build(lambda: ((k, "cam", "drl", img, BOX, outline) for k, (img, outline) in enumerate(frames)))


def test_healthy_passes_and_defects_are_named():
    ref = reference()
    assert L.check(ref, "cam", "drl", frame(seed=99)[0], BOX)["ok"]
    dead = L.check(ref, "cam", "drl", frame(dead=(80, 110), seed=99)[0], BOX)
    assert not dead["ok"] and any(r.startswith("DARK_SEGMENT") for r in dead["reasons"]) and "AREA_LOW" in dead["reasons"]
    dim = L.check(ref, "cam", "drl", frame(dim=0.6, seed=99)[0], BOX)
    assert not dim["ok"]                                    # too dim to count as lit
    amber = L.check(ref, "cam", "drl", frame(colour=(255, 170, 0), seed=99)[0], BOX)
    assert "COLOUR" in amber["reasons"]


def test_a_coloured_part_is_lit_only_in_its_colour():
    frames = [frame(colour=(255, 150, 0), seed=s) for s in range(20)]              # an amber strip
    ref = L.build(lambda: ((k, "cam", "ind", img, BOX, o) for k, (img, o) in enumerate(frames)))
    assert ref["cam"]["parts"]["ind"]["lit_hue"] is not None
    img, _ = frame(colour=(255, 150, 0), seed=99)
    assert L.check(ref, "cam", "ind", img, BOX)["ok"]
    img[45:55, 80:110] = 250                                 # a section bright but white: not the indicator's light
    assert not L.check(ref, "cam", "ind", img, BOX)["ok"]
    assert L.check(ref, "cam", "fog", frame()[0], BOX)["reasons"] == ["UNKNOWN_PART"]


def sweep_frame(reach, dead=None, seed=0):
    """A strip lit from x=45 up to 45 + reach (of 110 px); the box follows the lit part; dead: (x1, x2) never lit."""
    img, _ = frame(seed=seed)
    img[45:55, 45:155] = 70                                 # unlit strip
    end = 45 + reach
    img[45:55, 45:end] = 255
    outline = np.zeros((120, 200), bool)
    outline[45:55, 45:end] = True
    if dead:
        img[45:55, dead[0]:dead[1]] = 70
        outline[45:55, dead[0]:dead[1]] = False
    xs = np.nonzero(outline.any(axis=0))[0]
    box = (int(xs.min()), 45, int(xs.max()) + 1, 55) if len(xs) else None
    return img, box, outline


def test_sweeping_parts_must_reach_their_full_shape():
    blinks = [sweep_frame(r, seed=s) for s in range(12) for r in (15, 40, 70, 110)]
    ref = L.build(lambda: ((k, "cam", "ind", img, box, o) for k, (img, box, o) in enumerate(blinks)), sweep=["ind"])
    assert ref["cam"]["parts"]["ind"]["sweep"] and ref["cam"]["parts"]["ind"]["samples"] == 12   # full frames only
    window = lambda **kw: [sweep_frame(r, seed=99, **kw)[:2] for r in (15, 40, 70, 110)]
    assert L.check_sweep(ref, "cam", "ind", window())["ok"]
    short = L.check_sweep(ref, "cam", "ind", [f for f in window(dead=(128, 155))])        # the far end never lights
    assert "NOT_FULL" in short["reasons"]
    gap = L.check_sweep(ref, "cam", "ind", window(dead=(85, 110)))                        # a dead middle section
    assert any(r.startswith("DARK_SEGMENT") for r in gap["reasons"])
    assert L.check_sweep(ref, "cam", "ind", [])["reasons"] == ["NOT_SEEN"]
    one_bad = [window(), window(dead=(128, 155)), window()]                        # one bad blink: still OK
    assert L.check_blinks(ref, "cam", "ind", one_bad)["ok"]
    two_bad = [window(dead=(128, 155)), window(), window(dead=(128, 155))]         # 2 of 3 fail: not OK
    r = L.check_blinks(ref, "cam", "ind", two_bad)
    assert not r["ok"] and r["failed"] == 2 and "NOT_FULL" in r["reasons"]


def test_xml_round_trip(tmp_path):
    ref = reference()
    L.write_xml(ref, tmp_path / "ref.xml", {"project": "demo"}, {"false_alarms": "0/10"})
    back = L.read_xml(tmp_path / "ref.xml")
    a, b = ref["cam"]["parts"]["drl"], back["cam"]["parts"]["drl"]
    assert a["threshold"] == b["threshold"] and np.allclose(a["template"], b["template"], atol=0.01)
    assert np.allclose(a["lit_px"], b["lit_px"], rtol=1e-3)
    img = frame(dead=(80, 110), seed=5)[0]
    assert L.check(ref, "cam", "drl", img, BOX)["reasons"] == L.check(back, "cam", "drl", img, BOX)["reasons"]


def test_part_reference_command_writes_xml(tmp_path, video):
    from typer.testing import CliRunner
    from engine.cli import app
    from engine.project import Project
    p = Project.create(tmp_path / "proj", ["left_lamp", "right_lamp"], video=video, every=2, task="segment")
    for k in range(8):
        m = np.zeros((48, 64), bool)
        m[10:20, 5:30] = True
        p.put(k, 1, 0, [5, 10, 30, 20], mask=m)
        p.set_reviewed(k)
    p.db.close()
    out = tmp_path / "ref.xml"
    r = CliRunner().invoke(app, ["part-reference", str(tmp_path / "proj"), "--out", str(out)])
    assert r.exit_code == 0, r.output
    ref = L.read_xml(out)
    assert "left_lamp" in ref["all"]["parts"] and ref["all"]["parts"]["left_lamp"]["samples"] == 8
