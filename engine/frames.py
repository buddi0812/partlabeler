"""Video frame reading with PyAV (frame-accurate, no separate FFmpeg install needed)."""
import io
import urllib.request

import av
from PIL import Image

SAMPLE = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"


def read_frames(src: str, n: int) -> list[Image.Image]:
    """First `n` frames of a local video file or URL as RGB PIL images."""
    data = urllib.request.urlopen(src).read() if src.startswith("http") else None
    with av.open(io.BytesIO(data) if data else src) as c:
        frames = []
        for f in c.decode(video=0):
            frames.append(f.to_image())
            if len(frames) == n:
                break
    return frames
