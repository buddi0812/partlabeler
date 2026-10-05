"""The production runtime's decoding and lamp-function judgement (engine/partcheck.py), without a GPU."""
import numpy as np

from engine import lampcheck as L
from engine.partcheck import Camera, Engine, Part
from tests.test_lampcheck import BOX, frame, sweep_frame


def test_engine_keeps_the_surest_part_per_class_in_frame_pixels():
    eng = Engine.__new__(Engine)
    eng.classes, eng.threshold = ["drl", "ind"], 0.5
    o = np.array([[0.5, 0.5, 0.2, 0.1, 0.9, 1, 7],           # cx, cy, w, h, score, class, query; surest first
                  [0.4, 0.5, 0.2, 0.1, 0.8, 1, 3],           # the same class again: dropped
                  [0.25, 0.5, 0.5, 0.2, 0.7, 0, 2],
                  [0.1, 0.1, 0.1, 0.1, 0.4, 0, 1]], np.float32)   # under the threshold
    parts = eng._parts(o, 200, 100)
    assert [(p.name, p.query) for p in parts] == [("ind", 7), ("drl", 2)]
    assert np.allclose(parts[0].box, (80, 45, 120, 55)) and np.allclose(parts[1].box, (0, 40, 100, 60))


def test_a_steady_function_is_ok_until_most_recent_checks_fail():
    ref = L.build(lambda: ((k, "cam", p, img, BOX, o) for k, (img, o) in enumerate(frame(seed=s) for s in range(30))
                           for p in ("left_drl", "right_drl")), pairs=[("left_drl", "right_drl")])
    cam = Camera(ref, "cam", [{"name": "DRL", "parts": ["left_drl", "right_drl"], "rule": "all"}])
    both = [Part("left_drl", 0.9, BOX, 0), Part("right_drl", 0.9, BOX, 1)]
    T = 1.2                                                  # healthy frames first: past the arrival settle
    for k in range(int(T / 0.04)):
        s = cam.update(k * 0.04, frame(seed=100 + k % 5)[0], both)["DRL"]
    assert s["on"] and s["ok"] and not s["missing"]
    dark = frame(dead=(80, 110), seed=99)[0]
    s = cam.update(T, dark, both)["DRL"]
    assert s["ok"]                                           # one bad frame: still OK
    for k in range(2):
        s = cam.update(T + 0.04 + 0.04 * k, dark, both)["DRL"]
    assert s["on"] and not s["ok"]
    assert any(r.startswith("DARK_SEGMENT") or r == "AREA_LOW" for r in s["faults"]["left_drl"])
    s = cam.update(T + 0.16, frame(seed=7)[0], both[:1])["DRL"]
    assert not s["on"] and s["missing"] == ["right_drl"]
    far = (BOX[0] + 30, BOX[1] + 30, BOX[2] + 30, BOX[3] + 30)   # the object not in place yet: nothing judged
    n = len(cam.results["left_drl"])
    s = cam.update(T + 0.18, dark, [Part("left_drl", 0.9, far, 0), Part("right_drl", 0.9, far, 1)])["DRL"]
    assert not s["in_place"] and len(cam.results["left_drl"]) == n
    s = cam.update(T + 0.20, dark, both)["DRL"]              # back in place and on: faults wait until it settles
    assert s["on"] and s["ok"] and cam.verdict("left_drl")[0] is False
    t = T + 0.20
    while t < T + 0.20 + cam.settle - 0.05:
        t += 0.04
        assert cam.update(t, dark, both)["DRL"]["ok"]
    assert not cam.update(T + 0.24 + cam.settle, dark, both)["DRL"]["ok"]
    slow = Camera(ref, "cam", cam.functions, persist=0.5)    # a fault is shown only once it lasts
    for k in range(int(T / 0.04)):
        slow.update(k * 0.04, frame(seed=100 + k % 5)[0], both)
    oks = [slow.update(T + 0.04 * k, dark, both)["DRL"]["ok"] for k in range(20)]
    assert oks[:14] == [True] * 14 and not oks[-1]            # failing from the 3rd dark frame, shown 0.5 s later


def test_a_blinking_function_is_judged_once_per_blink_on_its_fullest_frame():
    blinks = [sweep_frame(r, seed=s) for s in range(12) for r in (15, 40, 70, 110)]
    ref = L.build(lambda: ((k, "cam", "ind", img, box, o) for k, (img, box, o) in enumerate(blinks)), sweep=["ind"])
    cam = Camera(ref, "cam", [{"name": "Indicator", "parts": ["ind"], "rule": "blink"}])
    t, s = 0.0, None

    def blink(dead=None):
        nonlocal t, s
        for r in (15, 40, 70, 110):                          # lighting up, found in 4 frames
            img, box, _ = sweep_frame(r, dead=dead, seed=99)
            s = cam.update(t, img, [Part("ind", 0.9, box, 0)])["Indicator"]
            t += 0.04
        for _ in range(8):                                   # off: the blink ends after `gap`
            s = cam.update(t, frame(seed=5)[0], [])["Indicator"]
            t += 0.04
    blink(), blink(dead=(128, 155)), blink()
    assert len(cam.blinks["ind"]) == 3 and s["on"] and s["ok"]   # one short blink of three: still OK
    blink(dead=(128, 155))
    assert not s["ok"] and "NOT_FULL" in s["faults"]["ind"]      # two of the last three never reach the end
    off = Camera(ref, "cam", cam.functions, check=False)            # on/off only: no part check
    img, box, _ = sweep_frame(110, dead=(128, 155), seed=99)
    s = off.update(0.0, img, [Part("ind", 0.9, box, 0)])["Indicator"]
    assert s["on"] and s["ok"] and not off.results and not off.open
