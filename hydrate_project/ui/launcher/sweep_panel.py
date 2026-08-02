"""
sweep_panel.py
----------------
Composition Comparison panel: runs the solver across several DIOX mol%
loadings (fixed gas ratio) or several gas ratios (fixed DIOX%), then opens
the Plot Builder pre-configured to show separation-factor columns so the
different compositions can be compared on one chart.
"""

from __future__ import annotations

import threading
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import Callable

from hydrate_project.ui.general_plotter import theme as th
from hydrate_project.services import validators as _validators
from hydrate_project.services.sweep_runner import run_diox_sweep, run_gas_ratio_sweep


def _default_sf_columns(gas_names: list[str]) -> list[str]:
    gas_names = list(gas_names)
    cols = [f"Ideal_SF_{g}" for g in gas_names]
    if len(gas_names) >= 2:
        cols.insert(0, f"SF_{gas_names[0]}_{gas_names[1]}")
    return cols or ["P_eq (MPa)"]


class SweepPanel(ttk.LabelFrame):
    """Composition Comparison section: DIOX% sweep or gas-ratio sweep."""

    def __init__(
        self,
        parent,
        promoters: dict[str, tuple],
        get_gas_comp: Callable[[], dict],
        get_liq_comp: Callable[[], dict],
        get_T_range: Callable,
        get_eos_names: Callable[[], list[str]],
        on_results: Callable,
        **kwargs,
    ):
        super().__init__(
            parent,
            text="  Composition Comparison  (separation factor)  ",
            style="TLabelframe",
            padding=(10, 8),
            **kwargs,
        )
        self._promoters = promoters
        self._get_gas_comp = get_gas_comp
        self._get_liq_comp = get_liq_comp
        self._get_T_range = get_T_range
        self._get_eos_names = get_eos_names
        self._on_results = on_results

        self._axis = tk.StringVar(value="diox")

        mode_row = ttk.Frame(self, style="TFrame")
        mode_row.pack(fill=tk.X)
        ttk.Radiobutton(
            mode_row,
            text="Vary DIOX %",
            value="diox",
            variable=self._axis,
            style="TRadiobutton",
            command=self._on_axis_change,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Radiobutton(
            mode_row,
            text="Vary gas ratio",
            value="gas_ratio",
            variable=self._axis,
            style="TRadiobutton",
            command=self._on_axis_change,
        ).pack(side=tk.LEFT)

        # ── DIOX% sweep inputs ───────────────────────────────────────────
        self._diox_frame = ttk.Frame(self, style="TFrame")
        self._diox_frame.pack(fill=tk.X, pady=(8, 0))

        prom_row = ttk.Frame(self._diox_frame, style="TFrame")
        prom_row.pack(fill=tk.X)
        ttk.Label(prom_row, text="Promoter:", style="Muted.TLabel", width=10).pack(
            side=tk.LEFT
        )
        prom_keys = list(promoters.keys())
        self._prom_var = tk.StringVar(value=prom_keys[0] if prom_keys else "")
        ttk.Combobox(
            prom_row,
            textvariable=self._prom_var,
            values=prom_keys,
            state="readonly",
            width=12,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)

        diox_row = ttk.Frame(self._diox_frame, style="TFrame")
        diox_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(diox_row, text="Mol % values:", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self._diox_values_var = tk.StringVar(value="2, 5.56, 8, 10")
        ttk.Entry(
            diox_row, textvariable=self._diox_values_var, font=("Segoe UI", 10)
        ).pack(fill=tk.X, pady=(2, 0))

        # ── Gas-ratio sweep inputs ───────────────────────────────────────
        self._ratio_frame = ttk.Frame(self, style="TFrame")
        ttk.Label(
            self._ratio_frame,
            text="One gas composition per line, e.g.  CO2=0.4, H2=0.6",
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(6, 0))
        self._ratio_text = scrolledtext.ScrolledText(
            self._ratio_frame,
            height=4,
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
        self._ratio_text.pack(fill=tk.X)
        self._ratio_text.insert(
            "end", "CO2=0.392, H2=0.608\nCO2=0.579, H2=0.421\n"
        )
        self._ratio_frame.pack_forget()

        # ── EOS choice + run ────────────────────────────────────────────
        run_row = ttk.Frame(self, style="TFrame")
        run_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(run_row, text="EOS:", style="Muted.TLabel").pack(side=tk.LEFT)
        self._eos_var = tk.StringVar(value="")
        self._eos_cb = ttk.Combobox(
            run_row,
            textvariable=self._eos_var,
            values=[],
            state="readonly",
            width=20,
            font=("Segoe UI", 9),
        )
        self._eos_cb.pack(side=tk.LEFT, padx=(6, 12))

        self._run_btn = ttk.Button(
            run_row,
            text="▶  Run Comparison",
            style="TButton",
            command=self._on_run,
        )
        self._run_btn.pack(side=tk.LEFT)

        self._status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._status_var, style="Muted.TLabel").pack(
            anchor=tk.W, pady=(6, 0)
        )

    # ── internal ───────────────────────────────────────────────────────────

    def _on_axis_change(self):
        if self._axis.get() == "diox":
            self._ratio_frame.pack_forget()
            self._diox_frame.pack(fill=tk.X, pady=(8, 0))
        else:
            self._diox_frame.pack_forget()
            self._ratio_frame.pack(fill=tk.X, pady=(8, 0))

    def _refresh_eos_choices(self):
        try:
            eos_names = self._get_eos_names()
        except Exception:
            eos_names = []
        self._eos_cb.configure(values=eos_names)
        if eos_names and self._eos_var.get() not in eos_names:
            self._eos_var.set(eos_names[0])

    def _on_run(self):
        self._refresh_eos_choices()
        eos_name = self._eos_var.get()
        if not eos_name:
            messagebox.showerror("Input error", "Select an EOS model.")
            return

        try:
            T_range = self._get_T_range()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return

        axis = self._axis.get()
        if axis == "diox":
            try:
                gas_comp = self._get_gas_comp()
                fractions = _validators.parse_diox_fraction_list(
                    self._diox_values_var.get()
                )
            except ValueError as exc:
                messagebox.showerror("Input error", str(exc))
                return
            promoter_key = self._prom_var.get()
            series_label = "compositions"
            default_cols = _default_sf_columns(list(gas_comp.keys()))

            def _sweep():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return run_diox_sweep(
                        gas_comp, promoter_key, fractions, T_range, eos_name, self._status_cb
                    )

        else:
            try:
                gas_comp_list = _validators.parse_gas_ratio_lines(
                    self._ratio_text.get("1.0", tk.END)
                )
            except ValueError as exc:
                messagebox.showerror("Input error", str(exc))
                return
            try:
                liq_comp = self._get_liq_comp()
            except ValueError as exc:
                messagebox.showerror("Input error", str(exc))
                return
            series_label = "compositions"
            default_cols = _default_sf_columns(list(gas_comp_list[0].keys()))

            def _sweep():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return run_gas_ratio_sweep(
                        gas_comp_list, liq_comp, T_range, eos_name, self._status_cb
                    )

        self._run_btn.configure(state=tk.DISABLED)
        self._status_var.set("Running comparison sweep…")

        def _worker():
            results, err = _sweep()
            self.after(0, lambda: self._on_done(results, err, series_label, default_cols))

        threading.Thread(target=_worker, daemon=True).start()

    def _status_cb(self, msg: str):
        self.after(0, lambda: self._status_var.set(msg))

    def _on_done(self, results, error, series_label, default_cols):
        self._run_btn.configure(state=tk.NORMAL)
        if error is not None:
            self._status_var.set(f"⚠  Error: {error}")
            messagebox.showerror("Sweep error", str(error))
            return
        if not results:
            self._status_var.set("⚠  No results.")
            return
        self._status_var.set(f"✓  {len(results)} composition(s) computed.")
        self._on_results(results, series_label, default_cols)
