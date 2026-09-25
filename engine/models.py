"""Where model weights come from: pinned revisions plus integrity checks.

SAM 3 is loaded from jetjodh/sam3, an ungated mirror of the gated facebook/sam3. On
2026-09-25 every file in it was confirmed identical to Meta's repo (git blob ids and LFS
SHA-256). The weights hash is re-checked after download, so a changed mirror never loads
silently. DINOv3 comes from the timm organisation (Hugging Face), licence included.

    python -m engine.models        # download + verify everything
"""
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

SAM3 = {
    "repo_id": "jetjodh/sam3",
    "revision": "1aa50ce07302cb375f85d8084b68a0fb378b8d85",
    # SHA-256 of facebook/sam3 model.safetensors, as published by Meta's repo.
    "sha256": {"model.safetensors": "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"},
}
DINOV3 = {
    "small": {"repo_id": "timm/vit_small_patch16_dinov3.lvd1689m", "revision": "3bf4720a82ec2066db88137180ff1f83a675cef0"},
    "base": {"repo_id": "timm/vit_base_patch16_dinov3.lvd1689m", "revision": "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"},
}

# Weights only as safetensors (no pickle files such as sam3.pt / pytorch_model.bin).
_ALLOW = ["*.json", "*.txt", "*.md", "LICENSE*", "*.safetensors"]
_VERIFIED = Path.home() / ".cache" / "partlabeler" / "verified.json"


def fetch(spec: dict) -> Path:
    """Download (or reuse from cache) a pinned snapshot and verify its checksums."""
    path = Path(snapshot_download(spec["repo_id"], revision=spec["revision"], allow_patterns=_ALLOW))
    for name, expected in spec.get("sha256", {}).items():
        _verify(path / name, expected)
    return path


def _verify(f: Path, expected: str) -> None:
    # Hashing 3+ GB takes a while, so remember files already checked (by size + mtime).
    seen = json.loads(_VERIFIED.read_text()) if _VERIFIED.exists() else {}
    stamp = f"{expected}:{f.stat().st_size}:{f.stat().st_mtime_ns}"
    if seen.get(str(f)) == stamp:
        return
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise RuntimeError(f"{f} does not match the official SHA-256 {expected}; refusing to load it.")
    seen[str(f)] = stamp
    _VERIFIED.parent.mkdir(parents=True, exist_ok=True)
    _VERIFIED.write_text(json.dumps(seen, indent=1))


if __name__ == "__main__":
    for spec in (SAM3, *DINOV3.values()):
        print(spec["repo_id"], "->", fetch(spec))
    print("all models downloaded and verified")
