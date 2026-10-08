import os

# Opening a project warms up SAM 3 in the background (engine/api.py Session._warm_up); tests use stand-ins instead.
os.environ.setdefault("PARTLABELER_WARMUP", "0")
