"""
app.py
------
Plot-builder window.  Two flavours:

  PlotBuilderApp    – standalone root Tk window  (direct launch)
  PlotBuilderWindow – Toplevel child window       (inside LauncherApp)

Both share identical behaviour through _PlotBuilderMixin.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional
import datetime

from hydrate_project.ui.general_plotter import theme as th
from hydrate_project.ui.general_plotter.canvas import PlotCanvas
from hydrate_project.ui.general_plotter.sidebar import Sidebar
from hydrate_project.ui.general_plotter.core import build_grid


class _PlotBuilderMixin:
    """All UI and logic shared between the Tk and Toplevel variants."""

    def _init_content(
        self,
        results_dict,
        experimental_data,
        title,
        series_label="EOS models",
        default_row_vars=None,
        default_col_vars=None,
    ):
        self._results = results_dict
        self._exp_data = experimental_data
        self._series_label = series_label
        self._default_row_vars = default_row_vars
        self._default_col_vars = default_col_vars
        self.title(title)
        self.geometry("1400x820")
        self.minsize(900, 600)
        th.apply(self)
        self._set_icon()
        self._build_layout()
        self._build_statusbar()
        self.bind("<Return>", lambda _e: self._on_update())
        self.bind("<Control-s>", lambda _e: self._on_save())
        self.after(100, self._on_update)

    def _build_layout(self):
        main = ttk.Frame(self, style="TFrame")
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        self._sidebar = Sidebar(
            main,
            results_dict=self._results,
            on_update=self._on_update,
            on_save=self._on_save,
            series_label=self._series_label,
            default_row_vars=self._default_row_vars,
            default_col_vars=self._default_col_vars,
        )
        self._sidebar.grid(row=0, column=0, sticky="nsew")
        ttk.Separator(main, orient="vertical").grid(row=0, column=1, sticky="ns")
        self._canvas = PlotCanvas(main, figsize=(13, 8), dpi=96)
        self._canvas.grid(row=0, column=2, sticky="nsew")
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
        try:
            icon = tk.PhotoImage(width=16, height=16)
            icon.put(th.BLUE, to=(0, 0, 15, 15))
            self.iconphoto(True, icon)
        except Exception:
            pass

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
            n = len(cfg.row_vars) * len(cfg.col_vars)
            self._status_var.set(
                f"✓  {n} panel{'s' if n != 1 else ''} drawn  •  "
                f"{len(cfg.row_vars)} row(s) × {len(cfg.col_vars)} col(s)  •  "
                f"x={cfg.x_var}  •  mode={cfg.comparison}"
            )
        except Exception as exc:
            self._status_var.set(f"⚠  Error: {exc}")
            messagebox.showerror("Plot error", str(exc))

    def _on_save(self, *_):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
            initialfile=f"hydrate_plot_{ts}.png",
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


class PlotBuilderApp(_PlotBuilderMixin, tk.Tk):
    """Standalone root-window — use when launching directly."""

    def __init__(
        self,
        results_dict,
        experimental_data=None,
        title="Hydrate — General Plot Builder",
        series_label="EOS models",
        default_row_vars=None,
        default_col_vars=None,
    ):
        tk.Tk.__init__(self)
        self._init_content(
            results_dict,
            experimental_data,
            title,
            series_label,
            default_row_vars,
            default_col_vars,
        )


class PlotBuilderWindow(_PlotBuilderMixin, tk.Toplevel):
    """Toplevel child window — use when embedded inside LauncherApp."""

    def __init__(
        self,
        master,
        results_dict,
        experimental_data=None,
        title="Hydrate — General Plot Builder",
        series_label="EOS models",
        default_row_vars=None,
        default_col_vars=None,
    ):
        tk.Toplevel.__init__(self, master)
        self._init_content(
            results_dict,
            experimental_data,
            title,
            series_label,
            default_row_vars,
            default_col_vars,
        )
