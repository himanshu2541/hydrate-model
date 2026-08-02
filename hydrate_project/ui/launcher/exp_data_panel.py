"""
exp_data_panel.py
-------------------
Experimental data panel: none / literature preset / manual entry, used for
AAD comparison against computed results.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, scrolledtext

from hydrate_project.ui.general_plotter import theme as th
from hydrate_project.services.literature_data import PRESET_DATA
from hydrate_project.services import validators as _validators


class ExpDataPanel(ttk.LabelFrame):
    """Experimental Data section: none / literature preset / manual entry."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            text="  Experimental Data  (optional — for AAD)  ",
            style="TLabelframe",
            padding=(10, 8),
            **kwargs,
        )
        self._mode = tk.StringVar(value="none")
        self._preset_var = tk.StringVar(value=next(iter(PRESET_DATA)))

        mode_row = ttk.Frame(self, style="TFrame")
        mode_row.pack(fill=tk.X)
        for val, lbl in [
            ("none", "None"),
            ("preset", "From literature"),
            ("custom", "Enter manually"),
        ]:
            ttk.Radiobutton(
                mode_row,
                text=lbl,
                value=val,
                variable=self._mode,
                style="TRadiobutton",
                command=self._on_mode_change,
            ).pack(side=tk.LEFT, padx=(0, 16))

        self._preset_frame = ttk.Frame(self, style="TFrame")
        self._preset_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(self._preset_frame, text="Dataset:", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Combobox(
            self._preset_frame,
            textvariable=self._preset_var,
            values=list(PRESET_DATA.keys()),
            state="readonly",
            font=("Segoe UI", 9),
            width=62,
        ).pack(side=tk.LEFT)
        self._preset_frame.pack_forget()

        self._custom_frame = ttk.Frame(self, style="TFrame")
        self._custom_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(
            self._custom_frame,
            text="One  T(K), P(MPa)  pair per line (comma or space separated):",
            style="Muted.TLabel",
        ).pack(anchor=tk.W)
        self._custom_text = scrolledtext.ScrolledText(
            self._custom_frame,
            height=6,
            width=40,
            font=("Consolas", 9),
            bg=th.SURFACE0,
            fg=th.TEXT,
            insertbackground=th.TEXT,
            relief="flat",
            highlightthickness=1,
            highlightcolor=th.BLUE,
            highlightbackground=th.SURFACE1,
        )
        self._custom_text.pack(fill=tk.X)
        self._custom_text.insert("end", "# T(K)   P(MPa)\n273.9   5.56\n275.7   6.90\n")
        self._custom_frame.pack_forget()

    # ── public API ─────────────────────────────────────────────────────────

    def get_exp_data(self) -> dict | None:
        mode = self._mode.get()
        if mode == "preset":
            preset = PRESET_DATA[self._preset_var.get()]
            return {"T (K)": preset["T (K)"], "P_eq (MPa)": preset["P_eq (MPa)"]}
        if mode == "custom":
            text = self._custom_text.get("1.0", tk.END)
            return _validators.parse_custom_exp_data(text)
        return None

    # ── internal ───────────────────────────────────────────────────────────

    def _on_mode_change(self):
        mode = self._mode.get()
        self._preset_frame.pack_forget()
        self._custom_frame.pack_forget()
        if mode == "preset":
            self._preset_frame.pack(fill=tk.X, pady=(6, 0))
        elif mode == "custom":
            self._custom_frame.pack(fill=tk.X, pady=(6, 0))
