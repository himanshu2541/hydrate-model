"""
eos_panel.py
-------------
Equation-of-state model selection panel: checkboxes for PR / SRK / PT.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

EOS_NAMES = ["Peng-Robinson", "Soave-Redlich-Kwong", "Patel-Teja"]


class EosPanel(ttk.LabelFrame):
    """Equation of State Models section."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            text="  Equation of State Models  ",
            style="TLabelframe",
            padding=(10, 8),
            **kwargs,
        )
        row = ttk.Frame(self, style="TFrame")
        row.pack(fill=tk.X)
        self._vars: dict[str, tk.BooleanVar] = {}
        for name in EOS_NAMES:
            v = tk.BooleanVar(value=True)
            self._vars[name] = v
            ttk.Checkbutton(row, text=name, variable=v, style="TCheckbutton").pack(
                side=tk.LEFT, padx=(0, 16)
            )

    def get_selected(self) -> list[str]:
        selected = [n for n, v in self._vars.items() if v.get()]
        if not selected:
            raise ValueError("Select at least one EOS model.")
        return selected
