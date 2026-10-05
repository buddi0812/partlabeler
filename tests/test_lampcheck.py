"""Reference profiles of lit parts and the check against them (engine/lampcheck.py), on synthetic lamp strips."""
from collections import defaultdict

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


def test_a_steady_part_fails_only_when_most_neighbouring_frames_fail():
    frames = [frame(seed=s) for s in range(30)]
    ref = L.build(lambda: ((k, "cam", "drl", img, BOX, o) for k, (img, o) in enumerate(frames)))
    good = [(frame(seed=100 + s)[0], BOX) for s in range(5)]
    dark = frame(dead=(80, 110), seed=99)[0]                  # a dead section
    assert not L.check(ref, "cam", "drl", dark, BOX)["ok"]
    blip = good[:2] + [(dark, BOX)] + good[3:]                # one bad frame: still OK
    assert L.check_frames(ref, "cam", "drl", blip)["ok"]
    r = L.check_frames(ref, "cam", "drl", [good[0], (dark, BOX), (dark, BOX), good[1], (dark, BOX)])
    assert not r["ok"] and r["failed"] == 3


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


def test_tuned_settings_travel_in_the_xml_and_change_the_check(tmp_path):
    ref = reference()
    img = frame(dead=(80, 110), seed=99)[0]
    v = L.measure(img, BOX, ref["cam"]["parts"]["drl"]["threshold"])
    assert L.check_values(ref, "cam", "drl", v) == L.check(ref, "cam", "drl", img, BOX)
    tuned = {"steady_vote": (4, 6), "dark_run": 50, "relative": True}
    L.write_xml(ref, tmp_path / "t.xml", settings=tuned)
    back = L.read_xml(tmp_path / "t.xml")
    s = L.settings(back, "cam")
    assert s["steady_vote"] == (4, 6) and s["dark_run"] == 50 and s["relative"] and s["reach"] == L.REACH
    r = L.check(back, "cam", "drl", img, BOX)
    assert not any(x.startswith(("DARK_SEGMENT", "AREA")) for x in r["reasons"])   # no dark run of 50; no area
    assert L.settings(L.read_xml(tmp_path / "t.xml"), "nope") == L.DEFAULTS
    wide = L.profiles({("cam", "drl"): [L.measure(frame(seed=s)[0], BOX, 200) for s in range(20)]}, {},
                      {("cam", "drl"): 200}, {}, margin=1.0)
    narrow = L.profiles({("cam", "drl"): [L.measure(frame(seed=s)[0], BOX, 200) for s in range(20)]}, {},
                        {("cam", "drl"): 200}, {}, margin=0.0)
    lo, hi = wide["cam"]["parts"]["drl"]["fill"], narrow["cam"]["parts"]["drl"]["fill"]
    assert lo[0] <= hi[0] and lo[1] >= hi[1]
    only = L.profiles({("cam", "drl"): [L.measure(frame(seed=s)[0], BOX, 200) for s in range(20)]}, {},
                      {("cam", "drl"): 200}, {}, margin={"default": 0.0, "fill": 1.0})["cam"]["parts"]["drl"]
    assert only["fill"] == wide["cam"]["parts"]["drl"]["fill"]
    assert only["lit_px"] == narrow["cam"]["parts"]["drl"]["lit_px"]


def test_a_part_with_variants_is_judged_against_the_nearest_variant(tmp_path):
    small = (150, 90, 180, 104)                              # another trim: a small lamp lower down
    def lamp(img, box, dead=None):
        img[box[1] + 2:box[3] - 2, box[0] + 2:box[2] - 2] = 255
        if dead:
            img[box[1]:box[3], dead[0]:dead[1]] = 70
        return img
    vals = defaultdict(list)
    for s in range(30):
        a = lamp(frame(seed=s)[0].copy(), BOX)
        va = L.measure(a, BOX, 200)
        vals[("cam", "ind")].append(va)
        b = lamp(np.full((120, 200, 3), 60, np.uint8), small)
        vb = L.measure(b, small, 200)
        vb["variant"] = 1
        vals[("cam", "ind")].append(vb)
    ref = L.profiles(vals, {}, {("cam", "ind"): 200}, {})
    assert len(ref["cam"]["variants"]["ind"]) == 1
    ok_small = L.check(ref, "cam", "ind", lamp(np.full((120, 200, 3), 60, np.uint8), small), small)
    assert ok_small["ok"] and ok_small["variant"] == 1
    half = L.check(ref, "cam", "ind", lamp(frame(seed=99)[0].copy(), BOX, dead=(100, 160)), BOX)
    assert not half["ok"] and half["variant"] == 0         # a half-dark strip is judged as a strip
    moved = (BOX[0], BOX[1] + 36, BOX[2], BOX[3] + 36)      # the whole object stands lower: still the strip
    img = lamp(np.full((120, 200, 3), 60, np.uint8), moved)
    assert L.check(ref, "cam", "ind", img, moved, shift=(0.0, 0.3))["variant"] == 0
    L.write_xml(ref, tmp_path / "v.xml")
    back = L.read_xml(tmp_path / "v.xml")
    assert len(back["cam"]["variants"]["ind"]) == 1
    assert np.allclose(back["cam"]["variants"]["ind"][0]["cy"], ref["cam"]["variants"]["ind"][0]["cy"])


def test_per_part_margins_and_settings(tmp_path):
    vals = {("cam", p): [L.measure(frame(dead=(155 - s, 155), seed=s)[0], BOX, 200) for s in range(20)]
            for p in ("a", "b")}                             # lamps a little different in length
    ref = L.profiles(vals, {}, {("cam", "a"): 200, ("cam", "b"): 200}, {},
                     margin={"default": 0.0, "a:fill": 1.0, "b:dark_share": 0.0})
    pa, pb = ref["cam"]["parts"]["a"], ref["cam"]["parts"]["b"]
    assert pa["fill"][1] - pa["fill"][0] > pb["fill"][1] - pb["fill"][0]       # only a's fill range widened
    assert pb["dark_share"] == 0.0 and "dark_share" not in pa
    L.write_xml(ref, tmp_path / "p.xml")
    assert L.read_xml(tmp_path / "p.xml")["cam"]["parts"]["b"]["dark_share"] == 0.0
