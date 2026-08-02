"""
liquid_panel.py
----------------
Liquid-phase composition panel: H2O baseline + optional promoter (e.g. DIOX)
mole fraction.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from hydrate_project.services import validators as _validators


class LiquidPanel(ttk.LabelFrame):
    """Liquid Phase Composition section: H2O + optional promoter."""

    def __init__(self, parent, promoters: dict[str, tuple], **kwargs):
        super().__init__(
            parent,
            text="  Liquid Phase Composition  ",
            style="TLabelframe",
            padding=(10, 8),
            **kwargs,
        )
        self._promoters = promoters
        self._promoter_var = tk.StringVar(value="none")
        self._promoter_frac = tk.StringVar(value="0.0556")

        disp_row = ttk.Frame(self, style="TFrame")
        disp_row.pack(fill=tk.X)
        self._liq_display = ttk.Label(
            disp_row,
            text="H₂O: 1.0000",
            style="Section.TLabel",
            font=("Segoe UI", 10, "bold"),
        )
        self._liq_display.pack(side=tk.LEFT)

        prom_toggle = ttk.Frame(self, style="TFrame")
        prom_toggle.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(prom_toggle, text="Promoter:", style="Muted.TLabel", width=10).pack(
            side=tk.LEFT
        )

        prom_opts = ["None"] + [f"{v[0]}  ({k})" for k, v in promoters.items()]
        self._prom_display_map = {"None": "none"}
        for k, v in promoters.items():
            self._prom_display_map[f"{v[0]}  ({k})"] = k

        self._prom_cb_var = tk.StringVar(value="None")
        prom_cb = ttk.Combobox(
            prom_toggle,
            textvariable=self._prom_cb_var,
            values=prom_opts,
            state="readonly",
            width=24,
            font=("Segoe UI", 9),
        )
        prom_cb.pack(side=tk.LEFT, padx=(0, 12))
        prom_cb.bind("<<ComboboxSelected>>", self._on_promoter_change)

        self._prom_frac_frame = ttk.Frame(self, style="TFrame")
        self._prom_frac_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(
            self._prom_frac_frame, text="Mol fraction:", style="Muted.TLabel"
        ).pack(side=tk.LEFT)
        ttk.Entry(
            self._prom_frac_frame,
            textvariable=self._promoter_frac,
            width=10,
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT, padx=(6, 10))
        self._prom_hint = ttk.Label(
            self._prom_frac_frame, text="", style="Muted.TLabel"
        )
        self._prom_hint.pack(side=tk.LEFT)
        self._prom_frac_frame.pack_forget()  # hidden until promoter selected
        self._promoter_frac.trace_add("write", lambda *_: self._refresh_display())

    # ── public API ─────────────────────────────────────────────────────────

    def get_composition(self) -> dict[str, float]:
        key = self._promoter_var.get()
        if key == "none":
            return {"H2O": 1.0}
        xp = _validators.parse_promoter_fraction(self._promoter_frac.get())
        return {"H2O": round(1.0 - xp, 8), key: round(xp, 8)}

    # ── internal ───────────────────────────────────────────────────────────

    def _on_promoter_change(self, *_):
        display_val = self._prom_cb_var.get()
        key = self._prom_display_map.get(display_val, "none")
        self._promoter_var.set(key)

        if key == "none":
            self._prom_frac_frame.pack_forget()
        else:
            info = self._promoters[key]
            self._promoter_frac.set(str(round(info[1], 4)))
            self._prom_hint.configure(text=info[2])
            self._prom_frac_frame.pack(fill=tk.X, pady=(6, 0))

        self._refresh_display()

    def _refresh_display(self, *_):
        key = self._promoter_var.get()
        if key == "none":
            self._liq_display.configure(text="H₂O: 1.0000")
            return
        try:
            xp = float(self._promoter_frac.get())
            xp = max(0.0, min(xp, 0.9999))
        except ValueError:
            xp = 0.0
        xw = round(1.0 - xp, 6)
        name = self._promoters[key][0]
        self._liq_display.configure(text=f"H₂O: {xw:.4f}    {name}: {xp:.4f}")
