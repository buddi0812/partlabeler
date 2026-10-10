# Teach's GPU memory calibration

`engine/detector.py`'s `_vram_needed_gb()` predicts how much GPU memory a Teach run will peak at, for:

- `fits_gpu()` - the preflight `train()` runs before starting at all (refuses with a plain message, or on a
  plain desktop falls back to the CPU, when even batch 1 won't fit);
- `batch_plan()` - picking the largest batch size that fits `free_gb - VRAM_MARGIN_GB` of free GPU memory
  (the margin is headroom for anything else that grows a bit while training runs, e.g. a box's own assistant
  model or live inspection);
- `vram_shortfall_message()` - the plain-language refusal text.

## How the numbers were measured (2026-10-10)

Real training, not a guess: `engine.detector.train()` called directly on a real (tiny) labeled dataset, one
real epoch, resolution 320px, with `PARTLABELER_BATCH` forcing the batch size, reading
`torch.cuda.max_memory_reserved()` straight after. Machine: RTX 3060 12 GB, torch 2.14+cu130, driver 596.49.
The script is not checked in (it was a one-off), but the shape is: for each `(size, batch)` in
`{nano, small} x {1, 2, 4, 8}`, `torch.cuda.reset_peak_memory_stats()`, train one epoch, record
`torch.cuda.max_memory_reserved() / 2**30`.

| size  | batch 1 | batch 2 | batch 4 | batch 8 |
|-------|---------|---------|---------|---------|
| nano  | 0.955   | 1.230   | 2.102   | 3.469   |
| small | 1.080   | 1.480   | 2.412   | 4.125   |

## The fit

Least-squares `peak = base + per_batch * batch` through those four points, per size, at 320px, then +10% on
`per_batch` as a safety margin (the raw fit is a touch optimistic at the high end of the tested range - see
"What this doesn't cover" below):

| size  | base (GB) | per_batch (GB, +10%) |
|-------|-----------|-----------------------|
| nano  | 0.60      | 0.40                   |
| small | 0.65      | 0.48                   |

Stored in `engine.detector._VRAM_CAL` (`_VRAM_CAL_RESOLUTION = 320`). For a resolution other than 320px,
`_vram_needed_gb()` scales `per_batch` by `(resolution / 320) ** 2` (pixel count), keeping `base` fixed (model
weights and CUDA context don't grow with input resolution; only activations do) - this scaling itself is
*not* measured, just the same assumption the old estimate used.

Sizes with no entry in `_VRAM_CAL` (`medium`, `seg-*`) fall back to the older, rougher `_GB_BASE`/`_GB_PER_IMAGE`
guess - add a row here the same way once they're measured.

## What this doesn't cover

- **Only tested at 320px.** The `(resolution / 320) ** 2` scaling to other resolutions is unverified.
- **Only tested up to batch 8.** `nano`'s own four points show the marginal cost per extra batch isn't
  perfectly constant (0.275, 0.436, 0.342 GB for the three steps 1→2→4→8) - close enough for a safe pick in
  the 1-8 range, but extrapolating batch_plan() much further (e.g. toward `EFFECTIVE_BATCH = 16`) is on
  shakier ground than the numbers above suggest. If a real run with a much larger batch disagrees with this
  table, that's new data: add it as a fifth column rather than re-deriving the fit from four points.
- **One machine.** A different GPU/driver/torch build may peak differently for the same batch; this table is
  this RTX 3060's own numbers, the same caveat the pre-calibration guess always had.

## Keeping this current

`tests/test_vram_calibration.py` checks that `_VRAM_CAL`'s stored (base, per_batch) actually predicts the
measured table above within a reasonable tolerance, so a careless edit to the constants doesn't quietly drift
from what was really measured. Add new sizes/resolutions to both the table above and that test's fixture data
when they're measured for real - never by guessing.
