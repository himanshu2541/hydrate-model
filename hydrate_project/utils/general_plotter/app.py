"""
app.py
------
Main application window.  Owns the layout:

  ┌──────────────┬──────────────────────────────────────────┐
  │              │  toolbar                                 │
  │   Sidebar    ├──────────────────────────────────────────┤
  │  (controls)  │                                          │
  │              │         PlotCanvas (matplotlib)          │
  │              │                                          │
  └──────────────┴──────────────────────────────────────────┘

Keyboard shortcut: Enter / Return → update plot.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional
import datetime

from hydrate_project.utils.general_plotter import theme as th
from hydrate_project.utils.general_plotter.canvas import PlotCanvas
from hydrate_project.utils.general_plotter.sidebar import Sidebar
from hydrate_project.utils.general_plotter.core import build_grid


class PlotBuilderApp(tk.Tk):
    """Top-level window for the General Plotter."""

    def __init__(
        self,
        results_dict: dict,
        experimental_data: Optional[dict] = None,
        title: str = "Hydrate — General Plot Builder",
    ):
        super().__init__()
        self._results = results_dict
        self._exp_data = experimental_data

        # ── window setup ───────────────────────────────────────────────────
        self.title(title)
        self.geometry("1400x820")
        self.minsize(900, 600)
        th.apply(self)
        self._set_icon()

        # ── layout ─────────────────────────────────────────────────────────
        self._build_layout()

        # ── status bar ─────────────────────────────────────────────────────
        self._build_statusbar()

        # ── keybindings ────────────────────────────────────────────────────
        self.bind("<Return>", lambda _e: self._on_update())
        self.bind("<Control-s>", lambda _e: self._on_save())

        # ── initial draw ───────────────────────────────────────────────────
        self.after(100, self._on_update)

    # ── layout builders ────────────────────────────────────────────────────

    def _build_layout(self):
        main = ttk.Frame(self, style="TFrame")
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Sidebar
        self._sidebar = Sidebar(
            main,
            results_dict=self._results,
            on_update=self._on_update,
            on_save=self._on_save,
        )
        self._sidebar.grid(row=0, column=0, sticky="nsew")

        # Vertical divider
        ttk.Separator(main, orient="vertical").grid(
            row=0, column=1, sticky="ns", padx=0
        )

        # Canvas
        self._canvas = PlotCanvas(main, figsize=(13, 8), dpi=96)
        self._canvas.grid(row=0, column=2, sticky="nsew", padx=0)
        main.columnconfigure(2, weight=1)

    def _build_statusbar(self):
        bar = ttk.Frame(self, style="TFrame", padding=(8, 3))
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, side=tk.BOTTOM)
        self._status_var = tk.StringVar(
            value="Ready  •  Press Enter or click Update Plot"
        )
        ttk.Label(bar, textvariable=self._status_var, style="Muted.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(bar, text="Press  Enter  to update", style="Muted.TLabel").pack(
            side=tk.RIGHT
        )

    def _set_icon(self):
        """Try to set a minimal window icon."""
        try:
            icon = tk.PhotoImage(width=16, height=16)
            icon.put(th.BLUE, to=(0, 0, 15, 15))
            self.iconphoto(True, icon)
        except Exception:
            pass

    # ── callbacks ──────────────────────────────────────────────────────────

    def _on_update(self, *_):
        cfg = self._sidebar.get_config()
        self._status_var.set("Rendering…")
        self.update_idletasks()
        try:
            build_grid(
                fig=self._canvas.get_figure(),
                results_dict=self._results,
                row_vars=cfg.row_vars,
                col_vars=cfg.col_vars,
                x_var=cfg.x_var,
                comparison=cfg.comparison,
                exp_data=self._exp_data,
                exp_overlay=cfg.exp_overlay,
                dark=cfg.dark_mode,
            )
            self._canvas.draw()
            n_cells = len(cfg.row_vars) * len(cfg.col_vars)
            self._status_var.set(
                f"✓  {n_cells} panel{'s' if n_cells != 1 else ''} drawn  •  "
                f"{len(cfg.row_vars)} row(s) × {len(cfg.col_vars)} col(s)  •  "
                f"x={cfg.x_var}  •  mode={cfg.comparison}"
            )
        except Exception as exc:
            self._status_var.set(f"⚠  Error: {exc}")
            messagebox.showerror("Plot error", str(exc))

    def _on_save(self, *_):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"hydrate_plot_{ts}.png"
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
            initialfile=default_name,
            title="Save plot as…",
        )
        if path:
            try:
                ext = path.rsplit(".", 1)[-1].lower()
                self._canvas.get_figure().savefig(
                    path,
                    dpi=150,
                    bbox_inches="tight",
                    facecolor=self._canvas.get_figure().get_facecolor(),
                    format=ext,
                )
                self._status_var.set(f"✓  Saved → {path}")
            except Exception as exc:
                messagebox.showerror("Save error", str(exc))
