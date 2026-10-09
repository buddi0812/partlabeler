"""model.json's "crop" field (engine.detector.crop_card), built from a Teach run's settings.json "task_cameras"
(hawk-ai-internal §13.3): one rectangle per camera, or a refusal naming the camera when its tasks disagree."""
import pytest

from engine.detector import crop_card
from engine.recordings import valid_crop


def test_no_task_cameras_means_no_crop_field():
    assert crop_card(None) is None
    assert crop_card({}) is None


def test_a_camera_cut_the_same_way_everywhere_gets_one_rectangle():
    task_cameras = {"line1_a": {"camera_id": "cam1", "crop": [0.1, 0.2, 0.5, 0.6]},
                    "line1_b": {"camera_id": "cam1", "crop": [0.1, 0.2, 0.5, 0.6]},
                    "line2_a": {"camera_id": "cam2", "crop": None}}        # cam2: never cropped - fine on its own
    assert crop_card(task_cameras) == {"cam1": [0.1, 0.2, 0.5, 0.6]}


def test_a_camera_cut_differently_refuses_export_by_name():
    task_cameras = {"a": {"camera_id": "cam1", "crop": [0.0, 0.0, 0.5, 0.5]},
                    "b": {"camera_id": "cam1", "crop": [0.1, 0.1, 0.5, 0.5]}}
    with pytest.raises(ValueError, match="cam1"):
        crop_card(task_cameras)


def test_a_camera_partly_cropped_and_partly_whole_also_refuses():
    task_cameras = {"a": {"camera_id": "cam1", "crop": [0.0, 0.0, 0.5, 0.5]},
                    "b": {"camera_id": "cam1", "crop": None}}
    with pytest.raises(ValueError, match="cam1"):
        crop_card(task_cameras)


def test_every_written_crop_passes_hawks_own_validator_with_string_camera_keys():
    """What the peer app module asked for: a round-trip through hawk-api's import rules (x,y>=0, w,h>0,
    x+w<=1, y+h<=1, string camera keys), so a valid PartLabeler export is never refused at import."""
    task_cameras = {"a": {"camera_id": "cam1", "crop": [0.25, 0.1, 0.5, 0.75]},
                    "b": {"camera_id": "cam2", "crop": [0.0, 0.0, 1.0, 1.0]}}
    crop = crop_card(task_cameras)
    assert set(crop) == {"cam1", "cam2"}
    for cid, rect in crop.items():
        assert isinstance(cid, str)
        assert valid_crop(rect)
