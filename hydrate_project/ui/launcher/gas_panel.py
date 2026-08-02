"""
gas_panel.py
------------
Gas-phase composition panel: a dynamic list of (gas, mole fraction) rows
with a live sum validator.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from hydrate_project.ui.general_plotter import theme as th
from hydrate_project.services import validators as _validators


class GasPanel(ttk.LabelFrame):
    """Gas Phase Composition section: dynamic rows + sum validator."""

    def __init__(self, parent, available_gases: list[str], **kwargs):
        super().__init__(
            parent,
            text="  Gas Phase Composition  ",
            style="TLabelframe",
            padding=(10, 8),
            **kwargs,
        )
        self._available_gases = available_gases
        self._rows: list[dict] = []

        self._rows_frame = ttk.Frame(self, style="TFrame")
        self._rows_frame.pack(fill=tk.X)

        add_row = ttk.Frame(self, style="TFrame")
        add_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            add_row, text="＋  Add gas", style="TButton", command=self._add_row_dialog
        ).pack(side=tk.LEFT)
        self._sum_label = ttk.Label(add_row, text="Sum: —", style="Muted.TLabel")
        self._sum_label.pack(side=tk.RIGHT, padx=4)

    # ── public API ─────────────────────────────────────────────────────────

    def add_row(self, gas: str = "CO2", frac: float = 0.5):
        row = ttk.Frame(self._rows_frame, style="TFrame")
        row.pack(fill=tk.X, pady=2)

        gas_var = tk.StringVar(value=gas)
        frac_var = tk.StringVar(value=str(round(frac, 4)))

        ttk.Label(row, text="Gas:", style="Muted.TLabel", width=4).pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=gas_var,
            values=self._available_gases,
            state="readonly",
            width=10,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(row, text="Mole fraction:", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=frac_var, width=10, font=("Segoe UI", 10)).pack(
            side=tk.LEFT, padx=(4, 12)
        )

        entry = {"gas_var": gas_var, "frac_var": frac_var, "frame": row}
        ttk.Button(
            row,
            text="✕",
            style="TButton",
            width=3,
            command=lambda e=entry: self._delete_row(e),
        ).pack(side=tk.LEFT)

        self._rows.append(entry)
        frac_var.trace_add("write", lambda *_: self._update_sum())
        gas_var.trace_add("write", lambda *_: self._update_sum())
        self._update_sum()

    def get_composition(self) -> dict[str, float]:
        """Parse rows into a gas_comp dict, validated to sum to 1.0."""
        gas_comp: dict[str, float] = {}
        for r in self._rows:
            g = r["gas_var"].get()
            try:
                f = float(r["frac_var"].get())
            except ValueError:
                raise ValueError(f"Invalid mole fraction for {g}.")
            gas_comp[g] = f
        _validators.validate_gas_composition(gas_comp)
        return gas_comp

    # ── internal ───────────────────────────────────────────────────────────

    def _add_row_dialog(self):
        used = {r["gas_var"].get() for r in self._rows}
        avail = [g for g in self._available_gases if g not in used]
        if not avail:
            messagebox.showinfo(
                "No more gases", "All available gases are already added."
            )
            return
        self.add_row(avail[0], 0.0)

    def _delete_row(self, entry: dict):
        if len(self._rows) <= 1:
            messagebox.showwarning(
                "Cannot delete", "At least one gas component is required."
            )
            return
        entry["frame"].destroy()
        self._rows.remove(entry)
        self._update_sum()

    def _update_sum(self):
        total = 0.0
        for r in self._rows:
            try:
                total += float(r["frac_var"].get())
            except ValueError:
                pass
        ok = abs(total - 1.0) < 1e-6
        self._sum_label.configure(
            text=f"Sum: {total:.4f}  {'✓' if ok else '⚠ must equal 1.000'}",
            foreground=th.GREEN if ok else th.RED,
        )
