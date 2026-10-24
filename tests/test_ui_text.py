"""Customer screens never name the AI models inside (a Hawk-AI box rule; credits live in docs/GUIDE.md and the
README). Comments in the page code may name them; the text a person reads may not."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = re.compile(r"SAM ?3|DINO|RF-DETR")


def test_pages_never_show_a_model_name():
    shown = [f"{f.name}:{n}: {line.strip()}" for f in (ROOT / "ui").glob("*.js")
             for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
             if MODELS.search(line.split("//")[0])]
    assert not shown, "model names in page text:\n" + "\n".join(shown)


# Customers never read technical terms either (a Hawk-AI rule): no model or file-format names, no training jargon.
# Say what happens ("Training… about 20 min left", "How well it learned: 94 %"); the technical side lives only behind
# "Details for support" (a page's <details> of that name; the guide's "Details for support" and credits sections).
TECH = re.compile(r"RF-DETR|DINO|SAM ?3|ONNX|FP16|INT8|TensorRT|\bmAP|(?<![.\w\"'])[Ee]pochs?\b(?![\w\"':=])"
                  r"|[Bb]atch size|[Hh]eld-out")
SPOKEN = {"note", "notify", "say", "progress", "hint", "toast", "RuntimeError", "HTTPException"}


def test_pages_never_show_technical_terms_outside_the_support_details():
    shown = []
    for f in (ROOT / "ui").glob("*.js"):
        support = False
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            support = support or "Details for support" in line
            if not support and TECH.search(line.split("//")[0]):
                shown.append(f"{f.name}:{n}: {line.strip()}")
            if support and "</details>" in line:
                support = False
    assert not shown, "technical terms in page text:\n" + "\n".join(shown)


def test_messages_a_person_reads_have_no_technical_terms():
    """Progress lines, notifications and errors that reach the screen (the training queue's text too)."""
    bad = []
    for f in [ROOT / "ui" / "host_fastapi.py", ROOT / "ui" / "trainq.py", *(ROOT / "engine").glob("*.py")]:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            args = [k.value for k in node.keywords if k.arg in ("text", "detail")]
            if name in SPOKEN:
                args += list(node.args)
            for a in args:
                for c in ast.walk(a):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str) and TECH.search(c.value):
                        bad.append(f"{f.name}:{c.lineno}: {c.value[:90]}")
    assert not bad, "technical terms in what a person reads:\n" + "\n".join(bad)


def test_the_guide_keeps_technical_terms_to_its_support_sections():
    bad, section = [], ""
    for n, line in enumerate((ROOT / "docs" / "GUIDE.md").read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("## "):
            section = line
        if TECH.search(line) and not any(s in section for s in ("Details for support", "Credits and licences")):
            bad.append(f"GUIDE.md:{n} ({section}): {line.strip()[:100]}")
    assert not bad, "technical terms outside the guide's support sections:\n" + "\n".join(bad)
