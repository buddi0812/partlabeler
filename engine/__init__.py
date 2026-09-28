import sys


def utf8_output() -> None:
    """Print UTF-8 (anything unprintable replaced): a Windows console or pipe defaults to cp1252, and RF-DETR's
    training tables (rich, box-drawing characters) then stop Teach with UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):               # not a text stream (e.g. replaced by a test runner)
            pass
