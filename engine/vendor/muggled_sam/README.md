# Vendored: muggled_sam (SAM 3.1 part only)

`v3p1_sam/` is copied unmodified from https://github.com/heyoeyo/muggled_sam at commit
`a004ffc72bde75a8a4d64573e2add06e934f40a4` (Apache License 2.0, see `LICENSE`, (c) heyoeyo).
PartLabeler uses it to track with SAM 3.1 Object Multiplex (`engine/tracker.py`, `Tracker31`), because Hugging Face
transformers does not ship SAM 3.1 yet and the package itself pins torch below 2.14. Replace this folder with a
newer copy to update it; nothing else in PartLabeler imports muggled_sam.
