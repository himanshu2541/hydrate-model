"""
app.py
------
Pre-computation configuration GUI for the Hydrate Equilibrium Model.

This is the slim "shell" that composes the individual panel widgets
(gas/liquid composition, temperature range, EOS selection, cache info,
experimental data) and wires the Run button to the backend solver in
hydrate_project.services. All science/orchestration logic lives in
services/; this module only owns Tkinter widgets and window flow.
"""

from __future__ import annotations

import threading
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from hydrate_project.ui.general_plotter import theme as th
from hydrate_project.ui.general_plotter.app import PlotBuilderWindow
from hydrate_project.ui.properties_window import PropertiesWindow

from hydrate_project.services.cache import get_cache
from hydrate_project.services.solver_runner import run_model as _run_model

from .gas_panel import GasPanel
from .liquid_panel import LiquidPanel
from .temperature_panel import TemperaturePanel
from .eos_panel import EosPanel
from .cache_panel import CachePanel
from .exp_data_panel import ExpDataPanel
from .sweep_panel import SweepPanel


# ── Available components (sourced from the database at import time) ───────────


def _load_component_lists():
    """Pull gas and promoter lists from Database so there's one source of truth."""
    try:
        from hydrate_project.core.database import Database

        db = Database()
        gases = list(db.GAS_DB.keys())
        promoters = {}
        for k, v in db.PROMOTER_DB.items():
            promoters[k] = (
                v.get("display_name", k),
                v.get("stoichiometric_x", 0.05),
                f"Stoichiometric: {v.get('stoichiometric_x', 0.05)*100:.2f} mol%",
            )
        return gases, promoters
    except Exception:
        # Fallback in case DB can't be imported yet
        return (
            ["CO2", "H2"],
            {"DIOX": ("1,4-Dioxane", 0.0556, "Stoichiometric: 5.56 mol%")},
        )


AVAILABLE_GASES, PROMOTERS = _load_component_lists()
GAS_LABELS = {k: k for k in AVAILABLE_GASES}  # plain fallback; override if needed
GAS_LABELS.update({"CO2": "CO₂", "H2": "H₂"})


# ── Launcher GUI ──────────────────────────────────────────────────────────────


class LauncherApp(tk.Tk):
    """Main entry-point window for model configuration and execution."""

    def __init__(self):
        super().__init__()
        self.title("Hydrate Equilibrium Model")
        self.geometry("880x900")
        self.resizable(True, True)
        th.apply(self)

        self._results: Optional[dict] = None
        self._exp_data: Optional[dict] = None

        self._build_ui()

        # Default gas rows
        self._gas_panel.add_row("CO2", 0.40)
        self._gas_panel.add_row("H2", 0.60)

    # ── UI skeleton ───────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = ttk.Frame(self, style="TFrame", padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        scroll_canvas = tk.Canvas(outer, bg=th.BASE, highlightthickness=0)
        vsb = th.styled_scrollbar(
            outer, orient=tk.VERTICAL, command=scroll_canvas.yview
        )
        scroll_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = ttk.Frame(scroll_canvas, style="TFrame", padding=(0, 4))
        win_id = scroll_canvas.create_window((0, 0), window=self._inner, anchor="nw")

        self._inner.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")),
        )
        scroll_canvas.bind(
            "<Configure>", lambda e: scroll_canvas.itemconfig(win_id, width=e.width)
        )
        scroll_canvas.bind_all(
            "<MouseWheel>",
            lambda e: scroll_canvas.yview_scroll(int(-e.delta / 120), "units"),
        )

        P = {"pady": (0, 14), "fill": tk.X}

        # Header
        hdr = ttk.Frame(self._inner, style="TFrame")
        hdr.pack(**P)
        ttk.Label(hdr, text="🔬  Hydrate Equilibrium Model", style="Title.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(
            hdr,
            text="Configure inputs, then click  ▶ Run Calculation",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Separator(self._inner, orient="horizontal").pack(fill=tk.X, pady=(0, 10))

        self._gas_panel = GasPanel(self._inner, AVAILABLE_GASES)
        self._gas_panel.pack(**P)

        self._liquid_panel = LiquidPanel(self._inner, PROMOTERS)
        self._liquid_panel.pack(**P)

        self._temperature_panel = TemperaturePanel(self._inner)
        self._temperature_panel.pack(**P)

        self._eos_panel = EosPanel(self._inner)
        self._eos_panel.pack(**P)

        self._cache_panel = CachePanel(self._inner)
        self._cache_panel.pack(**P)

        self._exp_data_panel = ExpDataPanel(self._inner)
        self._exp_data_panel.pack(**P)

        self._sweep_panel = SweepPanel(
            self._inner,
            PROMOTERS,
            get_gas_comp=lambda: self._gas_panel.get_composition(),
            get_liq_comp=lambda: self._liquid_panel.get_composition(),
            get_T_range=lambda: self._temperature_panel.get_T_range(),
            get_eos_names=lambda: self._eos_panel.get_selected(),
            on_results=self._open_sweep_plot_builder,
        )
        self._sweep_panel.pack(**P)

        # ── Run button + status ───────────────────────────────────────────────
        ttk.Separator(self._inner, orient="horizontal").pack(fill=tk.X, pady=(0, 10))
        self._run_btn = ttk.Button(
            self._inner,
            text="▶   Run Calculation",
            style="Primary.TButton",
            command=self._on_run,
        )
        self._run_btn.pack(fill=tk.X, pady=(0, 6))

        # Secondary action row
        btn_row = ttk.Frame(self._inner, style="TFrame")
        btn_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            btn_row,
            text="📊  View Properties",
            style="TButton",
            command=self._open_properties_window,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(
            btn_row,
            text="📈  Open Plot Builder",
            style="TButton",
            command=self._open_plot_builder,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._progress = ttk.Progressbar(self._inner, mode="indeterminate")
        self._progress.pack(fill=tk.X, pady=(0, 6))
        self._progress.pack_forget()

        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            self._inner,
            textvariable=self._status_var,
            style="Muted.TLabel",
            wraplength=820,
        ).pack(anchor=tk.W)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_inputs(self):
        gas_comp = self._gas_panel.get_composition()
        liq_comp = self._liquid_panel.get_composition()
        T_range = self._temperature_panel.get_T_range()
        eos_names = self._eos_panel.get_selected()
        exp_data = self._exp_data_panel.get_exp_data()
        return gas_comp, liq_comp, T_range, eos_names, exp_data

    # ── Run ───────────────────────────────────────────────────────────────────

    def _on_run(self):
        try:
            gas_comp, liq_comp, T_range, eos_names, exp_data = self._validate_inputs()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return

        self._exp_data = exp_data

        # ── Cache check ──────────────────────────────────────────────────────
        cache = get_cache()
        cached = cache.get(gas_comp, liq_comp, T_range, eos_names)
        if cached is not None:
            self._results = cached
            n_total = sum(len(df) for df in cached.values())
            self._status_var.set(
                f"⚡  Cache HIT — {len(cached)} model(s), "
                f"{n_total} rows loaded instantly.  Opening Plot Builder…"
            )
            self._cache_panel.refresh()
            self._open_plot_builder()
            return

        # ── Cache miss → compute ─────────────────────────────────────────────
        self._run_btn.configure(state=tk.DISABLED)
        self._progress.pack(fill=tk.X, pady=(0, 6))
        self._progress.start(12)
        self._status_var.set("Cache miss — starting calculation…")

        def _status_cb(msg: str):
            self.after(0, lambda m=msg: self._status_var.set(m))

        # Capture for thread closure
        _gas, _liq, _T, _eos = gas_comp, liq_comp, T_range, eos_names

        def _worker():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results, err = _run_model(_gas, _liq, _T, _eos, _status_cb)
            if err is None and results:
                try:
                    cache.put(_gas, _liq, _T, _eos, results)
                except Exception as ce:
                    _status_cb(f"[cache write failed: {ce}]")
            self.after(0, lambda: self._on_done(results, err))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, results: dict, error: Optional[Exception]):
        self._progress.stop()
        self._progress.pack_forget()
        self._run_btn.configure(state=tk.NORMAL)
        self._cache_panel.refresh()

        if error:
            self._status_var.set(f"⚠  Error: {error}")
            messagebox.showerror("Computation error", str(error))
            return
        if not results:
            self._status_var.set("⚠  No results — check inputs.")
            return

        self._results = results
        n_total = sum(len(df) for df in results.values())
        self._status_var.set(
            f"✓  Computed & cached — {len(results)} model(s), "
            f"{n_total} total rows.  Opening Plot Builder…"
        )
        self._open_plot_builder()

    def _open_plot_builder(self):
        if not self._results:
            return
        PlotBuilderWindow(
            master=self,
            results_dict=self._results,
            experimental_data=self._exp_data,
            title="Hydrate — General Plot Builder",
        ).focus_set()

    def _open_sweep_plot_builder(self, results_dict, series_label, default_col_vars):
        PlotBuilderWindow(
            master=self,
            results_dict=results_dict,
            experimental_data=None,
            title="Hydrate — Composition Comparison",
            series_label=series_label,
            default_row_vars=["P_eq (MPa)"],
            default_col_vars=default_col_vars,
        ).focus_set()

    def _open_properties_window(self):
        if not self._results:
            messagebox.showinfo("No results", "Run a calculation first.")
            return
        PropertiesWindow(
            master=self,
            results_dict=self._results,
            experimental_data=self._exp_data,
        ).focus_set()
