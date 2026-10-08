import sys


def utf8_output() -> None:
    """Print UTF-8 (anything unprintable replaced): a Windows console or pipe defaults to cp1252, and RF-DETR's
    training tables (rich, box-drawing characters) then stop Teach with UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):               # not a text stream (e.g. replaced by a test runner)
            pass
    _no_quick_edit()


def _no_quick_edit() -> None:
    """Windows: a click in the console window starts QuickEdit text selection, and the program then freezes at its
    next print until Esc is pressed (a model load that printed froze the annotator: "Getting the outline model
    ready…" forever, clicks never answered). Switch QuickEdit off for this console; no console: nothing to do."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        con_in = k32.GetStdHandle(-10)                    # STD_INPUT_HANDLE: QuickEdit is part of the input mode
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(con_in, ctypes.byref(mode)):
            k32.SetConsoleMode(con_in, (mode.value | 0x0080) & ~0x0040)   # + EXTENDED_FLAGS, - QUICK_EDIT_MODE
    except Exception:
        pass
