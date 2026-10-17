"""Customer screens never name the AI models inside (a Hawk-AI box rule; credits live in docs/GUIDE.md and the
README). Comments in the page code may name them; the text a person reads may not."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = re.compile(r"SAM ?3|DINO|RF-DETR")


def test_pages_never_show_a_model_name():
    shown = [f"{f.name}:{n}: {line.strip()}" for f in (ROOT / "ui").glob("*.js")
             for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
             if MODELS.search(line.split("//")[0])]
    assert not shown, "model names in page text:\n" + "\n".join(shown)
