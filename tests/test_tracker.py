"""Tracker prompts (engine/tracker.py) without the model: outlines go in as masks, boxes as boxes, and each
frame's prompts become memory at once (the session only counts the latest call's inputs as new)."""
import numpy as np

from engine.tracker import Tracker


def test_outlines_and_boxes_are_both_used_right_away():
    calls = []

    class Proc:
        def add_inputs_to_inference_session(self, inference_session, frame_idx, obj_ids, **kw):
            calls.append(("add", frame_idx, tuple(obj_ids), next(iter(kw))))

    t = Tracker.__new__(Tracker)                      # no model download: just the prompt logic
    t.proc = Proc()
    t.model = lambda inference_session, frame_idx: calls.append(("run", frame_idx))
    mask = np.zeros((4, 4), bool); mask[1:3, 1:3] = True
    t._add(None, 3, {7: mask, 2: [0, 0, 2, 2], 9: mask})
    assert calls == [("add", 3, (7, 9), "input_masks"), ("run", 3), ("add", 3, (2,), "input_boxes"), ("run", 3)]
