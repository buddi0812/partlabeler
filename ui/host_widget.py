"""Notebook host for the annotator (Jupyter, JupyterLab, VS Code, Colab).

    from ui.host_widget import Annotator
    Annotator("data/projects/my_run")        # last expression in a cell displays it
"""
from pathlib import Path

import anywidget

from engine.api import Session
from engine.project import Project


class Annotator(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "canvas.js"

    def __init__(self, project, **kwargs):
        super().__init__(**kwargs)
        self.session = Session(project if isinstance(project, Project) else Project(project), self.send)
        self.on_msg(lambda _w, msg, _b: self.session.handle(msg))
