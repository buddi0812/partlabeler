"""Tracking that a person steers (engine/tracker.py): frames where they drew a part further on pull the track back
there, the two nearest earlier ones are memory, and SAM 3.1's guard never lets a balloon become a part's normal size. A scripted stand-in for SAM 3's video session: the part is a
square, and a session that has run long enough balloons it (as drift does)."""
import types

import numpy as np
import torch

from engine import masks as M
from engine import tracker as T
from engine.project import Project
from tests.test_project import CLASSES, video  # noqa: F401  (fixture)

H, W = 48, 64


def square(x, y=10, s=12):
    m = np.zeros((H, W), bool)
    m[y:y + s, x:x + s] = True
    return m


class FakeSession:
    def __init__(self, items):
        self.items, self.obj_ids, self.video_height, self.video_width = items, [], H, W
        self.prompts = {}                                     # local frame -> {obj: prompt}


class FakeProc:
    def __init__(self, log):
        self.log = log

    def init_video_session(self, video, **kw):
        sess = FakeSession(list(self.log["frames"].pop(0)))
        self.log["sessions"].append(sess)
        return sess

    def add_inputs_to_inference_session(self, inference_session, frame_idx, obj_ids, input_masks=None, input_boxes=None):
        for k, o in enumerate(obj_ids):
            if o not in inference_session.obj_ids:
                inference_session.obj_ids.append(o)
            v = input_masks[k] if input_masks is not None else input_boxes[0][k]
            inference_session.prompts.setdefault(frame_idx, {})[o] = v

    def post_process_masks(self, masks, original_sizes):
        return masks


class FakeModel:
    """Frame `item` shows the square at x = 2 + item; a session that has run `drift` frames balloons it, and the
    frame after `speck_at` also gets a detached speck in the corner."""
    def __init__(self, drift=99, speck_at=None, lost=None, hide=None):
        self.drift, self.speck_at, self.lost = drift, speck_at, lost or {}       # lost: {obj: from this item on}
        self.hide = hide or {}                                                   # hide: {obj: (from, back)} back half seen

    def __call__(self, inference_session, frame_idx):
        pass

    def propagate_in_video_iterator(self, sess, start_frame_idx):
        for local in range(start_frame_idx, len(sess.items)):
            item = sess.items[local]
            out = []
            for o in sess.obj_ids:
                given = sess.prompts.get(local, {}).get(o)
                m = given.copy() if isinstance(given, np.ndarray) else square(2 + item)
                if local - start_frame_idx >= self.drift and given is None:
                    m = square(2 + item, y=2, s=40)               # ballooned
                if self.speck_at is not None and item == self.speck_at:
                    m[44:47, 0:3] = True
                if o in self.lost and item >= self.lost[o] and given is None:
                    m[:] = False
                if o in self.hide and given is None:
                    a, b = self.hide[o]
                    if a <= item < b:
                        m[:] = False
                    elif item == b:
                        m = square(2 + item, s=12) & (np.arange(H)[:, None] < 16)   # half of it out from behind
                out.append(m)
            yield types.SimpleNamespace(frame_idx=local, pred_masks=torch.from_numpy(np.stack(out)[:, None]))


def make(model, frames_log):
    t = object.__new__(T.Tracker)
    log = {"frames": frames_log, "sessions": []}
    t.model, t.proc, t.device, t.dtype, t.chunk = model, FakeProc(log), "cpu", torch.float32, T.CHUNK
    return t, log


def run(t, p, start, count, log):
    real_init = t.proc.init_video_session

    def init(video, **kw):                                    # the frame items of each new session, in order
        log["frames"].append(list(t._frames))
        return real_init(video, **kw)
    t.proc.init_video_session = init
    real_chunk = t._chunk

    def chunk(project, refs, ahead, at, cur_item, cur, items, *a):
        t._frames = sorted(refs, key=lambda it: abs(it - cur_item), reverse=True) + [cur_item] + items
        return real_chunk(project, refs, ahead, at, cur_item, cur, items, *a)
    t._chunk = chunk
    got = {}
    n = t.track(p, start, count, lambda item, res: got.__setitem__(item, res), masks=True)
    return n, got


def project(tmp_path, video):
    p = Project.create(tmp_path / "proj", CLASSES, video=video, every=1, task="segment")     # 20 frames of 64 x 48
    p.put(0, 1, 0, (0, 0, 1, 1), mask=square(2))
    return p


def test_the_sam31_guard_never_takes_a_balloon_as_the_normal():
    g = T.Guard()
    g.see(1, square(2), -1, 0, fresh=True)
    for k in range(4):
        assert g.see(1, square(2), k, k + 1)                                  # same size: good
    big = square(2, s=30)
    for k in range(4, 4 + 3 * T.HISTORY):
        assert not g.see(1, big, k, k + 1)                                     # however long it stays
    assert np.array_equal(g.last[1], square(2))
    assert g.may_retry(1, 5) and not g.may_retry(1, 6) and g.may_retry(1, 5 + T.RETRY_GAP)
    assert g.see(1, big, 99, 100, fresh=True) and g.see(1, big, 100, 101)     # a person's size: the normal
    assert g.see(1, square(2, s=4), 200, 201)                                  # back from hiding: not checked


def test_a_frame_a_person_drew_inside_the_run_steers_it(tmp_path, video):
    p = project(tmp_path, video)
    fixed = square(9, y=20, s=20)                                              # drawn by hand on frame 5, larger
    p.put(5, 1, 0, (0, 0, 1, 1), mask=fixed)
    t, log = make(FakeModel(), [])
    n, got = run(t, p, 0, 8, log)
    sess = log["sessions"][0]
    assert n == 8 and np.array_equal(sess.prompts[sess.items.index(5)][1], fixed)


def test_only_frames_where_a_person_drew_that_part_steer(tmp_path, video):
    p = project(tmp_path, video)
    p.put_tracked(4, {1: (0, [2, 10, 14, 22], 0.9, square(6))})               # tracked earlier ...
    p.set_reviewed(4)                                                          # ... and confirmed
    p.put(4, 2, 1, (40, 30, 52, 42))                                           # another part drawn there
    p.put(6, 3, 1, (0, 0, 1, 1), mask=square(30))                              # a person's outline of part 3 ...
    p.put(6, 1, 0, (0, 0, 1, 1), mask=square(8))                               # ... and of part 1 on frame 6
    p.put(6, 4, 1, (40, 30, 52, 42))                                           # and a box: not heard after them
    t, log = make(FakeModel(), [])
    run(t, p, 0, 8, log)
    sess = log["sessions"][0]
    assert 1 not in sess.prompts.get(sess.items.index(4), {})                  # its own confirmed tracking: no prompt
    assert set(sess.prompts[sess.items.index(6)]) == {1}                       # outlines only, of the parts tracked


def test_the_two_nearest_frames_a_person_drew_before_are_memory(tmp_path, video):
    p = project(tmp_path, video)
    for k in (2, 4, 6, 8):                                                     # 8: where tracking starts
        p.put(k, 1, 0, (0, 0, 1, 1), mask=square(2 + k))
    t, log = make(FakeModel(), [])
    run(t, p, 8, 3, log)
    assert log["sessions"][0].items[:3] == [4, 6, 8]


def test_a_part_back_from_behind_a_hand_goes_on(tmp_path, video):
    p = project(tmp_path, video)
    t, log = make(FakeModel(hide={1: (3, 5)}), [])                           # hidden on 3-4, half seen on 5
    n, got = run(t, p, 0, 10, log)
    assert n == 10 and len(log["sessions"]) == 1
    assert got[5][1][1] is not None and all(got[k][1][3].sum() == 144 for k in range(6, 11))


def test_a_part_lost_at_a_chunk_end_is_not_seeded_again(tmp_path, video, monkeypatch):
    monkeypatch.setattr(T, "CHUNK", 3)
    p = project(tmp_path, video)
    p.put(0, 2, 0, (40, 30, 52, 42))                                          # a box with no outline: a box prompt
    t, log = make(FakeModel(lost={2: 2}), [])                                 # part 2 gone from frame 2 on
    n, got = run(t, p, 0, 6, log)
    assert n == 6 and [2 in x.obj_ids for x in log["sessions"]] == [True, False]
    assert got[1][2][1] is not None and all(2 not in got[k] for k in range(4, 7))
