"""
launcher_app.py
---------------
Pre-computation configuration GUI for the Hydrate Equilibrium Model.

Features
--------
  • Gas phase composition  (dynamic rows, live sum validator)
  • Liquid phase composition  (H₂O baseline + optional promoter)
  • Temperature scan range   (T_min / T_max / T_step with point count preview)
  • EOS model selection      (PR, SRK, PT)
  • SQLite FIFO result cache (automatic hit/miss with clear-cache button)
  • Experimental data        (none / literature presets / manual entry)
  • Progress bar + background worker thread
  • Opens Plot Builder on completion
"""

from __future__ import annotations

import threading
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import Optional
import numpy as np
import pandas as pd

from hydrate_project.utils.general_plotter import theme as th
from hydrate_project.utils.cache import get_cache
from hydrate_project.utils.general_plotter.app import PlotBuilderWindow
from hydrate_project.utils.properties_window import PropertiesWindow


# ── Literature presets ────────────────────────────────────────────────────────

PRESET_DATA: dict[str, dict] = {
    "CO₂/H₂ (39.2/60.8 mol%) — Kumar et al. 2006  [sI, bulk water]": {
        "T (K)": [273.9, 274.8, 275.7, 276.5, 277.3, 278.0, 278.4],
        "P_eq (MPa)": [5.56, 6.15, 6.90, 7.75, 8.80, 9.95, 10.74],
    },
    "CO₂/H₂ (57.9/42.1 mol%) — Kumar et al. 2006  [sI, bulk water]": {
        "T (K)": [274.6, 275.5, 276.5, 277.5, 278.5, 279.5, 280.5, 281.4],
        "P_eq (MPa)": [2.77, 3.20, 3.80, 4.55, 5.45, 6.55, 7.90, 8.21],
    },
    "CO₂/H₂ (39.9/60.1 mol%) — Belandria et al. 2011  [sI, bulk water]": {
        "T (K)": [273.6, 274.5, 275.4, 276.4, 277.3, 278.3, 279.3, 280.3, 281.2],
        "P_eq (MPa)": [1.88, 2.20, 2.60, 3.10, 3.70, 4.50, 5.50, 6.70, 8.57],
    },
    "CO₂ (Pure) — Sloan & Koh 2008  [sI, bulk water]": {
        "T (K)": [
            273.15,
            274.15,
            275.15,
            276.15,
            277.15,
            278.15,
            279.15,
            280.15,
            281.15,
            282.15,
            283.15,
        ],
        "P_eq (MPa)": [1.25, 1.4, 1.5, 1.75, 2.0, 2.25, 2.6, 3.0, 3.4, 3.9, 4.5],
    },
}

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


# ── Solver worker ─────────────────────────────────────────────────────────────


def _run_model(
    gas_comp: dict,
    liq_comp: dict,
    T_range: np.ndarray,
    eos_names: list[str],
    status_cb: callable,
) -> tuple[dict, Optional[Exception]]:
    """Run solver for every selected EOS.  Returns (results_dict, error)."""
    try:
        from hydrate_project.core.database import Database
        # from hydrate_project.thermo_model.john_holder import JohnHolderModel
        # from hydrate_project.thermo_model.klauda_sandler import KlaudaSandlerModel
        from hydrate_project.thermo_model.klauda_sandler_empirical import KlaudaSandlerEmpiricalModel
        from hydrate_project.eos_model.pr_eos import PREOS
        from hydrate_project.eos_model.srk_eos import SRKEOS
        from hydrate_project.eos_model.pt_eos import PTEOS
        from hydrate_project.solvers.equilibrium import EquilibriumSolver

        EOS_MAP = {
            "Peng-Robinson": PREOS,
            "Soave-Redlich-Kwong": SRKEOS,
            "Patel-Teja": PTEOS,
        }

        db = Database()
        # hydrate_core = KlaudaSandlerModel(database=db)
        hydrate_core = KlaudaSandlerEmpiricalModel(database=db)
        # hydrate_core = JohnHolderModel(database=db)
        results: dict[str, pd.DataFrame] = {}

        for eos_name in eos_names:
            cls = EOS_MAP[eos_name]
            eos_inst = cls(composition=gas_comp, database=db)
            solver = EquilibriumSolver(
                liq_phase_composition=liq_comp,
                database=db,
                hydrate_model=hydrate_core,
                eos_model=eos_inst,
            )
            status_cb(f"[{eos_name}]  scanning {len(T_range)} temperature points…")
            df = solver.find_optimum_structure(
                T_range=T_range, P_initial_guess=2.5e6, solver_method="bisect"
            )
            results[eos_name] = df
            status_cb(f"[{eos_name}]  ✓ done")

        return results, None

    except Exception as exc:
        return {}, exc


# ── Launcher GUI ──────────────────────────────────────────────────────────────


class LauncherApp(tk.Tk):
    """Main entry-point window for model configuration and execution."""

    def __init__(self):
        super().__init__()
        self.title("Hydrate Equilibrium Model")
        self.geometry("880x900")
        self.resizable(True, True)
        th.apply(self)

        self._gas_rows: list[dict] = []
        self._exp_mode = tk.StringVar(value="none")
        self._preset_var = tk.StringVar(value=next(iter(PRESET_DATA)))
        self._results: Optional[dict] = None
        self._exp_data: Optional[dict] = None

        # Promoter state
        self._promoter_var = tk.StringVar(value="none")  # "none" or key in PROMOTERS
        self._promoter_frac = tk.StringVar(value="0.0556")

        self._build_ui()

        # Default gas rows
        self._add_gas_row("CO2", 0.40)
        self._add_gas_row("H2", 0.60)
        self._refresh_liquid_display()

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

        # ── Section: Gas phase ────────────────────────────────────────────────
        gas_sec = self._section("Gas Phase Composition")
        gas_sec.pack(**P)
        self._gas_rows_frame = ttk.Frame(gas_sec, style="TFrame")
        self._gas_rows_frame.pack(fill=tk.X)
        add_row = ttk.Frame(gas_sec, style="TFrame")
        add_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            add_row,
            text="＋  Add gas",
            style="TButton",
            command=self._add_gas_row_dialog,
        ).pack(side=tk.LEFT)
        self._sum_label = ttk.Label(add_row, text="Sum: —", style="Muted.TLabel")
        self._sum_label.pack(side=tk.RIGHT, padx=4)

        # ── Section: Liquid phase / Promoter ─────────────────────────────────
        liq_sec = self._section("Liquid Phase Composition")
        liq_sec.pack(**P)

        # Static display row
        disp_row = ttk.Frame(liq_sec, style="TFrame")
        disp_row.pack(fill=tk.X)
        self._liq_display = ttk.Label(
            disp_row,
            text="H₂O: 1.0000",
            style="Section.TLabel",
            font=("Segoe UI", 10, "bold"),
        )
        self._liq_display.pack(side=tk.LEFT)

        # Promoter toggle row
        prom_toggle = ttk.Frame(liq_sec, style="TFrame")
        prom_toggle.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(prom_toggle, text="Promoter:", style="Muted.TLabel", width=10).pack(
            side=tk.LEFT
        )

        prom_opts = ["None"] + [f"{v[0]}  ({k})" for k, v in PROMOTERS.items()]
        self._prom_display_map = {"None": "none"}
        for k, v in PROMOTERS.items():
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

        # Concentration entry (shown only when promoter != None)
        self._prom_frac_frame = ttk.Frame(liq_sec, style="TFrame")
        self._prom_frac_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(
            self._prom_frac_frame, text="Mol fraction:", style="Muted.TLabel"
        ).pack(side=tk.LEFT)
        prom_ent = ttk.Entry(
            self._prom_frac_frame,
            textvariable=self._promoter_frac,
            width=10,
            font=("Segoe UI", 10),
        )
        prom_ent.pack(side=tk.LEFT, padx=(6, 10))
        self._prom_hint = ttk.Label(
            self._prom_frac_frame, text="", style="Muted.TLabel"
        )
        self._prom_hint.pack(side=tk.LEFT)
        self._prom_frac_frame.pack_forget()  # hidden until promoter selected
        self._promoter_frac.trace_add(
            "write", lambda *_: self._refresh_liquid_display()
        )

        # ── Section: Temperature ──────────────────────────────────────────────
        T_sec = self._section("Temperature Scan Range")
        T_sec.pack(**P)
        T_row = ttk.Frame(T_sec, style="TFrame")
        T_row.pack(fill=tk.X)
        for label, attr, default in [
            ("From (K)", "_T_min", "273.15"),
            ("To (K)", "_T_max", "283.15"),
            ("Step (K)", "_T_step", "0.5"),
        ]:
            f = ttk.Frame(T_row, style="TFrame")
            f.pack(side=tk.LEFT, padx=(0, 16))
            ttk.Label(f, text=label, style="Muted.TLabel").pack(anchor=tk.W)
            var = tk.StringVar(value=default)
            setattr(self, attr + "_var", var)
            ttk.Entry(f, textvariable=var, width=10, font=("Segoe UI", 10)).pack()
            var.trace_add("write", lambda *_: self._update_t_preview())
        self._t_preview = ttk.Label(T_sec, text="", style="Muted.TLabel")
        self._t_preview.pack(anchor=tk.W, pady=(4, 0))
        self._update_t_preview()

        # ── Section: EOS ──────────────────────────────────────────────────────
        eos_sec = self._section("Equation of State Models")
        eos_sec.pack(**P)
        eos_row = ttk.Frame(eos_sec, style="TFrame")
        eos_row.pack(fill=tk.X)
        self._eos_vars: dict[str, tk.BooleanVar] = {}
        for name in ["Peng-Robinson", "Soave-Redlich-Kwong", "Patel-Teja"]:
            v = tk.BooleanVar(value=True)
            self._eos_vars[name] = v
            ttk.Checkbutton(eos_row, text=name, variable=v, style="TCheckbutton").pack(
                side=tk.LEFT, padx=(0, 16)
            )

        # ── Section: Cache info ───────────────────────────────────────────────
        cache_sec = self._section("Result Cache  (SQLite · FIFO eviction)")
        cache_sec.pack(**P)
        cache_top = ttk.Frame(cache_sec, style="TFrame")
        cache_top.pack(fill=tk.X)
        self._cache_label = ttk.Label(
            cache_top, text="Checking cache…", style="Muted.TLabel"
        )
        self._cache_label.pack(side=tk.LEFT)
        ttk.Button(
            cache_top,
            text="🗑  Clear cache",
            style="TButton",
            command=self._on_clear_cache,
        ).pack(side=tk.RIGHT)
        self._update_cache_label()

        # ── Section: Experimental data ────────────────────────────────────────
        exp_sec = self._section("Experimental Data  (optional — for AAD)")
        exp_sec.pack(**P)
        mode_row = ttk.Frame(exp_sec, style="TFrame")
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
                variable=self._exp_mode,
                style="TRadiobutton",
                command=self._on_exp_mode_change,
            ).pack(side=tk.LEFT, padx=(0, 16))

        self._preset_frame = ttk.Frame(exp_sec, style="TFrame")
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

        self._custom_frame = ttk.Frame(exp_sec, style="TFrame")
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

    # ── Helper ────────────────────────────────────────────────────────────────

    def _section(self, title: str) -> ttk.LabelFrame:
        return ttk.LabelFrame(
            self._inner, text=f"  {title}  ", style="TLabelframe", padding=(10, 8)
        )

    # ── Gas composition ───────────────────────────────────────────────────────

    def _add_gas_row(self, gas: str = "CO2", frac: float = 0.5):
        row = ttk.Frame(self._gas_rows_frame, style="TFrame")
        row.pack(fill=tk.X, pady=2)

        gas_var = tk.StringVar(value=gas)
        frac_var = tk.StringVar(value=str(round(frac, 4)))

        ttk.Label(row, text="Gas:", style="Muted.TLabel", width=4).pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=gas_var,
            values=AVAILABLE_GASES,
            state="readonly",
            width=10,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(row, text="Mole fraction:", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=frac_var, width=10, font=("Segoe UI", 10)).pack(
            side=tk.LEFT, padx=(4, 12)
        )

        entry = {"gas_var": gas_var, "frac_var": frac_var, "frame": row}
        del_btn = ttk.Button(
            row,
            text="✕",
            style="TButton",
            width=3,
            command=lambda e=entry: self._delete_gas_row(e),
        )
        del_btn.pack(side=tk.LEFT)

        self._gas_rows.append(entry)
        frac_var.trace_add("write", lambda *_: self._update_sum())
        gas_var.trace_add("write", lambda *_: self._update_sum())
        self._update_sum()

    def _add_gas_row_dialog(self):
        used = {r["gas_var"].get() for r in self._gas_rows}
        avail = [g for g in AVAILABLE_GASES if g not in used]
        if not avail:
            messagebox.showinfo(
                "No more gases", "All available gases are already added."
            )
            return
        self._add_gas_row(avail[0], 0.0)

    def _delete_gas_row(self, entry: dict):
        if len(self._gas_rows) <= 1:
            messagebox.showwarning(
                "Cannot delete", "At least one gas component is required."
            )
            return
        entry["frame"].destroy()
        self._gas_rows.remove(entry)
        self._update_sum()

    def _update_sum(self):
        total = 0.0
        for r in self._gas_rows:
            try:
                total += float(r["frac_var"].get())
            except ValueError:
                pass
        ok = abs(total - 1.0) < 1e-6
        self._sum_label.configure(
            text=f"Sum: {total:.4f}  {'✓' if ok else '⚠ must equal 1.000'}",
            foreground=th.GREEN if ok else th.RED,
        )

    # ── Promoter / liquid phase ───────────────────────────────────────────────

    def _on_promoter_change(self, *_):
        display_val = self._prom_cb_var.get()
        key = self._prom_display_map.get(display_val, "none")
        self._promoter_var.set(key)

        if key == "none":
            self._prom_frac_frame.pack_forget()
        else:
            info = PROMOTERS[key]
            self._promoter_frac.set(str(round(info[1], 4)))
            self._prom_hint.configure(text=info[2])
            self._prom_frac_frame.pack(fill=tk.X, pady=(6, 0))

        self._refresh_liquid_display()

    def _refresh_liquid_display(self, *_):
        """Update the H₂O / promoter fraction display label."""
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
        name = PROMOTERS[key][0]
        self._liq_display.configure(text=f"H₂O: {xw:.4f}    {name}: {xp:.4f}")

    # ── Temperature ───────────────────────────────────────────────────────────

    def _update_t_preview(self):
        try:
            tmin = float(self._T_min_var.get())
            tmax = float(self._T_max_var.get())
            tstep = float(self._T_step_var.get())
            n = max(0, int(round((tmax - tmin) / tstep)) + 1)
            self._t_preview.configure(text=f"→ {n} temperature points")
        except Exception:
            self._t_preview.configure(text="")

    # ── Cache label ───────────────────────────────────────────────────────────

    def _update_cache_label(self):
        try:
            info = get_cache().info()
            self._cache_label.configure(
                text=f"{info.total_entries} / {info.max_entries} entries  •  "
                f"session: {info.hits_session} hits, "
                f"{info.misses_session} misses  •  "
                f"DB: {info.db_path}"
            )
        except Exception as exc:
            self._cache_label.configure(text=f"Cache unavailable: {exc}")

    def _on_clear_cache(self):
        if messagebox.askyesno(
            "Clear cache", "Delete all cached results?\nThis cannot be undone."
        ):
            get_cache().clear()
            self._update_cache_label()
            self._status_var.set("Cache cleared.")

    # ── Experimental data ─────────────────────────────────────────────────────

    def _on_exp_mode_change(self):
        mode = self._exp_mode.get()
        self._preset_frame.pack_forget()
        self._custom_frame.pack_forget()
        if mode == "preset":
            self._preset_frame.pack(fill=tk.X, pady=(6, 0))
        elif mode == "custom":
            self._custom_frame.pack(fill=tk.X, pady=(6, 0))

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_inputs(self):
        # --- Gas composition ---
        gas_comp: dict[str, float] = {}
        total = 0.0
        for r in self._gas_rows:
            g = r["gas_var"].get()
            try:
                f = float(r["frac_var"].get())
            except ValueError:
                raise ValueError(f"Invalid mole fraction for {g}.")
            if f < 0:
                raise ValueError(f"Mole fraction for {g} cannot be negative.")
            gas_comp[g] = f
            total += f
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Gas mole fractions must sum to 1.0  (current = {total:.4f})."
            )

        # --- Liquid composition (H₂O + optional promoter) ---
        key = self._promoter_var.get()
        if key == "none":
            liq_comp = {"H2O": 1.0}
        else:
            try:
                xp = float(self._promoter_frac.get())
            except ValueError:
                raise ValueError("Promoter mole fraction must be a number.")
            if not (0.0 < xp < 1.0):
                raise ValueError("Promoter mole fraction must be between 0 and 1.")
            liq_comp = {"H2O": round(1.0 - xp, 8), key: round(xp, 8)}

        # --- Temperature range ---
        try:
            tmin = float(self._T_min_var.get())
            tmax = float(self._T_max_var.get())
            tstep = float(self._T_step_var.get())
        except ValueError:
            raise ValueError("Temperature inputs must be numeric.")
        if tmin >= tmax:
            raise ValueError("T_min must be less than T_max.")
        if tstep <= 0:
            raise ValueError("T_step must be positive.")
        T_range = np.arange(tmin, tmax + tstep / 2, tstep)

        # --- EOS ---
        eos_names = [n for n, v in self._eos_vars.items() if v.get()]
        if not eos_names:
            raise ValueError("Select at least one EOS model.")

        # --- Experimental data ---
        exp_data = None
        mode = self._exp_mode.get()
        if mode == "preset":
            exp_data = dict(PRESET_DATA[self._preset_var.get()])
        elif mode == "custom":
            lines = self._custom_text.get("1.0", tk.END).strip().splitlines()
            Ts, Ps = [], []
            for ln in lines:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.replace(",", " ").split()
                if len(parts) < 2:
                    raise ValueError(f"Cannot parse: '{ln}'  (expected T P).")
                Ts.append(float(parts[0]))
                Ps.append(float(parts[1]))
            if Ts:
                exp_data = {"T (K)": Ts, "P_eq (MPa)": Ps}

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
            self._update_cache_label()
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
        self._update_cache_label()

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
        # Offer both windows without blocking
        PlotBuilderWindow(
            master=self,
            results_dict=self._results,
            experimental_data=self._exp_data,
            title="Hydrate — General Plot Builder",
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
