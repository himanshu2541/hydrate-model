"""
temperature_panel.py
---------------------
Temperature scan range panel: T_min / T_max / T_step with a live point-count
preview.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from hydrate_project.services import validators as _validators


class TemperaturePanel(ttk.LabelFrame):
    """Temperature Scan Range section."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            text="  Temperature Scan Range  ",
            style="TLabelframe",
            padding=(10, 8),
            **kwargs,
        )
        T_row = ttk.Frame(self, style="TFrame")
        T_row.pack(fill=tk.X)

        self._vars: dict[str, tk.StringVar] = {}
        for label, key, default in [
            ("From (K)", "min", "273.15"),
            ("To (K)", "max", "283.15"),
            ("Step (K)", "step", "0.5"),
        ]:
            f = ttk.Frame(T_row, style="TFrame")
            f.pack(side=tk.LEFT, padx=(0, 16))
            ttk.Label(f, text=label, style="Muted.TLabel").pack(anchor=tk.W)
            var = tk.StringVar(value=default)
            self._vars[key] = var
            ttk.Entry(f, textvariable=var, width=10, font=("Segoe UI", 10)).pack()
            var.trace_add("write", lambda *_: self._update_preview())

        self._preview = ttk.Label(self, text="", style="Muted.TLabel")
        self._preview.pack(anchor=tk.W, pady=(4, 0))
        self._update_preview()

    # ── public API ─────────────────────────────────────────────────────────

    def get_T_range(self):
        """Parse and validate T_min/T_max/T_step -> np.ndarray of temperatures."""
        try:
            tmin = float(self._vars["min"].get())
            tmax = float(self._vars["max"].get())
            tstep = float(self._vars["step"].get())
        except ValueError:
            raise ValueError("Temperature inputs must be numeric.")
        return _validators.build_T_range(tmin, tmax, tstep)

    # ── internal ───────────────────────────────────────────────────────────

    def _update_preview(self):
        try:
            tmin = float(self._vars["min"].get())
            tmax = float(self._vars["max"].get())
            tstep = float(self._vars["step"].get())
            n = max(0, int(round((tmax - tmin) / tstep)) + 1)
            self._preview.configure(text=f"→ {n} temperature points")
        except Exception:
            self._preview.configure(text="")
