"""engine.detector's GPU-memory calibration (docs/VRAM_CALIBRATION.md): the stored fit still predicts what was
really measured, batch_plan() only returns a batch size it actually checked, and the 1 GB margin is real."""
import pytest

from engine import detector

# The measured table from docs/VRAM_CALIBRATION.md (torch.cuda.max_memory_reserved(), real training steps,
# RTX 3060, 320px): size -> {batch: peak_gb}. Update both files together when new real measurements come in.
MEASURED = {
    "nano": {1: 0.955, 2: 1.230, 4: 2.102, 8: 3.469},
    "small": {1: 1.080, 2: 1.480, 4: 2.412, 8: 4.125},
}


@pytest.mark.parametrize("size", sorted(MEASURED))
def test_calibration_predicts_the_measured_table_within_tolerance(size):
    """The stored (base, per_batch) must stay an honest fit of what was really measured - not perfect (there's
    a +10% safety margin baked in on purpose), but never so far off that the constants have silently drifted
    from the table in the README."""
    for batch, measured in MEASURED[size].items():
        predicted = detector._vram_needed_gb(size, detector._VRAM_CAL_RESOLUTION, batch)
        assert measured <= predicted <= measured * 1.25, (size, batch, measured, predicted)


def test_batch_plan_only_returns_a_batch_it_checked_fits():
    """Regression: the pre-calibration batch_plan() checked the CURRENT batch's own need, then doubled
    WITHOUT checking the doubled value's need - so its returned batch was never itself verified (it could be
    up to 2x over budget). Nano at 320px needs about 1.0 GB at batch 1 and 1.4 GB at batch 2 (_VRAM_CAL); a
    budget of 1.35 GB fits batch 1 but not batch 2 - batch_plan must stop at 1, not double past it on the
    strength of batch 1 alone fitting."""
    budget = 1.35
    batch, accum = detector.batch_plan("nano", 320, free_gb=budget + detector.VRAM_MARGIN_GB, n_train=100)
    assert batch == 1 and accum == detector.EFFECTIVE_BATCH
    assert detector._vram_needed_gb("nano", 320, batch) <= budget   # the returned batch's own need was checked
    assert detector._vram_needed_gb("nano", 320, batch * 2) > budget           # confirms batch 2 genuinely didn't fit


def test_batch_plan_leaves_the_margin_free():
    """Exactly enough for batch 2 with no margin must NOT be offered batch 2 - the margin is real headroom,
    not decoration."""
    need_batch_2 = detector._vram_needed_gb("nano", 320, 2)
    batch, _ = detector.batch_plan("nano", 320, free_gb=need_batch_2, n_train=100)         # no margin: refuse 2
    assert batch == 1
    batch, _ = detector.batch_plan("nano", 320, free_gb=need_batch_2 + detector.VRAM_MARGIN_GB, n_train=100)
    assert batch == 2                                                                      # margin included: fine


def test_fits_gpu_also_respects_the_margin():
    needed = detector._vram_needed_gb("nano", 320, batch=1)
    assert not detector.fits_gpu("nano", 320, needed)                        # exactly enough, no margin: refused
    assert detector.fits_gpu("nano", 320, needed + detector.VRAM_MARGIN_GB)   # margin included: fine


def test_uncalibrated_sizes_fall_back_to_the_old_guess():
    """medium/seg-* have no _VRAM_CAL entry yet: the older, rougher formula, unchanged."""
    assert "medium" not in detector._VRAM_CAL
    old = detector._GB_BASE + 2 * 1 * detector._GB_PER_IMAGE["medium"] * (640 / 640) ** 2
    assert detector._vram_needed_gb("medium", 640, batch=1) == old
