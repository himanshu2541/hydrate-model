"""
canvas.py
---------
Wraps a matplotlib Figure inside a ttk Frame with a slim custom toolbar.
Provides `update_plot()` to redraw with new data.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import theme as th


class _SlimToolbar(NavigationToolbar2Tk):
    """Navigation toolbar with only the most useful tools and dark styling."""
    toolitems = (
        ("Home",    "Reset original view",  "home",    "home"),
        ("Back",    "Previous view",         "back",    "back"),
        ("Forward", "Next view",             "forward", "forward"),
        (None, None, None, None),
        ("Pan",     "Pan axes",              "move",    "pan"),
        ("Zoom",    "Zoom to rectangle",     "zoom_to_rect", "zoom"),
        (None, None, None, None),
        ("Save",    "Save figure",           "filesave","save_figure"),
    )

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent, pack_toolbar=False)
        self.config(background=th.MANTLE)
        for child in self.winfo_children():
            try:
                child.config(background=th.MANTLE,
                             foreground=th.TEXT,
                             relief="flat",
                             activebackground=th.SURFACE1)
            except Exception:
                pass


class PlotCanvas(ttk.Frame):
    """
    A ttk.Frame that contains:
      ┌──────────────────────────────────┐
      │  [slim toolbar]                  │
      ├──────────────────────────────────┤
      │  matplotlib Figure (fills rest)  │
      └──────────────────────────────────┘
    """

    def __init__(self, parent, figsize=(12, 8), dpi=96, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style="TFrame")

        self.fig = Figure(figsize=figsize, dpi=dpi,
                          facecolor=th.BASE, tight_layout=True)

        self._canvas = FigureCanvasTkAgg(self.fig, master=self)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.configure(bg=th.BASE, highlightthickness=0)

        self._toolbar = _SlimToolbar(self._canvas, self)

        # Pack: toolbar at top (thin), canvas fills the rest
        self._toolbar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 0))
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, pady=0)
        self._canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def draw(self):
        """Flush all pending drawing commands."""
        self._canvas.draw_idle()

    def get_figure(self) -> Figure:
        return self.fig

    def save_png(self, filepath: str, dpi: int = 150):
        self.fig.savefig(filepath, dpi=dpi, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())
        print(f"[PlotCanvas] Saved → {filepath}")