"""Transfer's track check (engine/tracks.py) on hand-made detections: (cls, x1, y1, x2, y2, score)."""
from engine.tracks import check, link

CLASSES = ["left_lamp", "right_lamp", "bracket", "bolt"]


def moving(cls, x, n, dx=10, y=100, w=40, score=0.9):
    return [(cls, x + k * dx, y, x + k * dx + w, y + w, score) for k in range(n)]


def frames_of(*tracks, n=6):
    return [[t[k] for t in tracks if k < len(t) and t[k] is not None] for k in range(n)]


def test_a_one_frame_label_flip_is_listed():
    left, right = moving(0, 50, 6), moving(1, 400, 6, dx=-10)
    left[3] = (1, *left[3][1:])                                # frame 3: the left lamp called right
    assert check(frames_of(left, right), CLASSES) == {3: {"label_flip": [["right_lamp", "left_lamp"]], "possible_miss": [], "lone_box": []}}
    assert len(link(frames_of(left, right))) == 2              # still two tracks


def test_short_gaps_are_possible_misses_long_ones_end_the_track():
    t = moving(2, 50, 6)
    t[2] = None
    assert check(frames_of(t), CLASSES) == {2: {"label_flip": [], "possible_miss": ["bracket"], "lone_box": []}}
    t[3] = t[4] = None                                         # 3 frames missing: a new track, nothing listed
    assert check(frames_of(t), CLASSES, max_gap=2) == {}


def test_lone_low_score_boxes_are_listed_confident_ones_are_not():
    track = moving(2, 50, 6)
    ghost = [None, None, (3, 300, 300, 320, 320, 0.4)]
    sure = [None, None, None, None, (3, 500, 50, 520, 70, 0.95)]
    assert check(frames_of(track, ghost, sure), CLASSES) == {2: {"label_flip": [], "possible_miss": [], "lone_box": ["bolt"]}}


def test_nested_parts_of_other_classes_stay_apart():
    bracket = moving(2, 50, 6, w=120)
    bolt = moving(3, 60, 6, w=20)                               # inside the bracket, moving with it
    assert check(frames_of(bracket, bolt), CLASSES) == {} and len(link(frames_of(bracket, bolt))) == 2
