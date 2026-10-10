"""The annotator's laptop-only gate (ui/responsive.js, hawk-ai-internal: "people need to be doing the
annotations in laptop not mobile"). Pure logic, no DOM: run the real module under plain Node, not a copy of
its logic, so a future edit to the real file is what this test actually checks."""
import json
import shutil
import subprocess

import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="Node is not installed")


def too_small(web: bool, width: int) -> bool:
    module_url = (ROOT / "ui" / "responsive.js").as_uri()
    script = f"""
import {{ tooSmallForAnnotating }} from "{module_url}";
console.log(JSON.stringify(tooSmallForAnnotating({json.dumps(web)}, {width})));
"""
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip())


@pytest.mark.parametrize("width, expected", [(320, True), (768, True), (899, True), (900, False), (1024, False), (1440, False)])
def test_web_app_blocks_below_900px_regardless_of_device(width, expected):
    """A tablet in landscape (typically >= 1024px) passes the same way a laptop does - the rule is width
    only, never a touch/device-type check, so it doesn't matter whether a wide screen has a mouse or a
    finger. Only phone-width (or a deliberately narrowed desktop window) is blocked."""
    assert too_small(True, width) == expected


def test_notebook_context_is_never_blocked_by_width():
    """model.homeUrl is unset for a notebook widget (web=False): narrow Jupyter output cells are a different
    situation, not a phone, and were already usable before this - don't change that."""
    assert too_small(False, 320) is False
    assert too_small(False, 1440) is False
