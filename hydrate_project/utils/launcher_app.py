"""
launcher_app.py
---------------
Pre-computation configuration GUI for the Hydrate Equilibrium Model.

Allows the user to:
  • Set gas phase composition  (mole fractions sum to 1)
  • Choose temperature scan range  (T_min, T_max, T_step)
  • Select which EOS models to run
  • Optionally supply experimental data (presets or manual entry)
  • Run the solver with a live progress bar
  • Launch the General Plot Builder when done
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
from hydrate_project.utils.general_plotter.app import PlotBuilderWindow


# ── Preset experimental data from literature ─────────────────────────────────
# Data extracted from figures / tables in the uploaded papers.
# These are approximate values; users can always enter custom data.

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
}

# Available gases (must match keys in core/database.py GUEST_DB)
AVAILABLE_GASES = ["CO2", "H2", "DIOX"]
GAS_LABELS = {"CO2": "CO₂", "H2": "H₂", "DIOX": "1,4-Dioxane"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_model(
    gas_comp: dict,
    liq_comp: dict,
    T_range: np.ndarray,
    eos_names: list[str],
    status_cb: callable,
) -> tuple[dict, Optional[Exception]]:
    """Run the solver for every selected EOS.  Returns (results_dict, error)."""
    try:
        from hydrate_project.core.database import Database
        from hydrate_project.thermo_model.john_holder import JohnHolderModel
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
        hydrate_core = JohnHolderModel(database=db)
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


# ── Launcher application ──────────────────────────────────────────────────────


class LauncherApp(tk.Tk):
    """Main entry-point window for model configuration and execution."""

    def __init__(self):
        super().__init__()
        self.title("Hydrate Equilibrium Model")
        self.geometry("860x780")
        self.resizable(True, True)
        th.apply(self)

        self._gas_rows: list[dict] = []  # list of {gas_var, frac_var, frame}
        self._exp_mode = tk.StringVar(value="none")
        self._preset_var = tk.StringVar(value=next(iter(PRESET_DATA)))
        self._results: Optional[dict] = None
        self._exp_data: Optional[dict] = None

        self._build_ui()

        # Seed two default gas rows
        self._add_gas_row("CO2", 0.40)
        self._add_gas_row("H2", 0.60)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = ttk.Frame(self, style="TFrame", padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # Scrollable canvas so the window adapts to small screens
        canvas = tk.Canvas(outer, bg=th.BASE, highlightthickness=0)
        vsb = th.styled_scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = ttk.Frame(canvas, style="TFrame", padding=(0, 4))
        inner_id = canvas.create_window((0, 0), window=self._inner, anchor="nw")

        def _resize(e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _canvas_resize(e):
            canvas.itemconfig(inner_id, width=e.width)

        self._inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", _canvas_resize)
        canvas.bind_all(
            "<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units")
        )

        P = {"pady": (0, 14), "fill": tk.X}

        # ── Header ────────────────────────────────────────────────────────────
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

        # ── Gas composition ───────────────────────────────────────────────────
        self._gas_frame_outer = self._section("Gas Phase Composition")
        self._gas_frame_outer.pack(**P)
        self._gas_rows_frame = ttk.Frame(self._gas_frame_outer, style="TFrame")
        self._gas_rows_frame.pack(fill=tk.X)
        add_btn_row = ttk.Frame(self._gas_frame_outer, style="TFrame")
        add_btn_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            add_btn_row,
            text="＋  Add gas",
            style="TButton",
            command=self._add_gas_row_dialog,
        ).pack(side=tk.LEFT)
        self._sum_label = ttk.Label(add_btn_row, text="Sum: —", style="Muted.TLabel")
        self._sum_label.pack(side=tk.RIGHT, padx=4)

        # ── Temperature range ─────────────────────────────────────────────────
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

        # ── EOS selection ─────────────────────────────────────────────────────
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

        # ── Experimental data ─────────────────────────────────────────────────
        exp_sec = self._section("Experimental Data  (optional — for AAD calculation)")
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
        preset_cb = ttk.Combobox(
            self._preset_frame,
            textvariable=self._preset_var,
            values=list(PRESET_DATA.keys()),
            state="readonly",
            font=("Segoe UI", 9),
            width=62,
        )
        preset_cb.pack(side=tk.LEFT)
        self._preset_frame.pack_forget()  # hidden initially

        self._custom_frame = ttk.Frame(exp_sec, style="TFrame")
        self._custom_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(
            self._custom_frame,
            text="Enter one T(K), P(MPa) pair per line  " "(comma or space separated):",
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
        self._custom_frame.pack_forget()  # hidden initially

        # ── Actions ───────────────────────────────────────────────────────────
        act_sec = ttk.Frame(self._inner, style="TFrame")
        act_sec.pack(fill=tk.X, pady=(4, 0))
        ttk.Separator(self._inner, orient="horizontal").pack(fill=tk.X, pady=(0, 10))

        self._run_btn = ttk.Button(
            act_sec,
            text="▶   Run Calculation",
            style="Primary.TButton",
            command=self._on_run,
        )
        self._run_btn.pack(fill=tk.X, pady=(0, 8))

        self._progress = ttk.Progressbar(act_sec, mode="indeterminate", length=300)
        self._progress.pack(fill=tk.X, pady=(0, 6))
        self._progress.pack_forget()

        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            act_sec, textvariable=self._status_var, style="Muted.TLabel", wraplength=780
        ).pack(anchor=tk.W)

    # ── Gas composition rows ──────────────────────────────────────────────────

    def _section(self, title: str) -> ttk.LabelFrame:
        return ttk.LabelFrame(
            self._inner,
            text=f"  {title}  ",
            style="TLabelframe",
            padding=(10, 8),
        )

    def _add_gas_row(self, gas: str = "CO2", frac: float = 0.5):
        row = ttk.Frame(self._gas_rows_frame, style="TFrame")
        row.pack(fill=tk.X, pady=2)

        gas_var = tk.StringVar(value=gas)
        frac_var = tk.StringVar(value=str(round(frac, 4)))

        ttk.Label(row, text="Gas:", style="Muted.TLabel", width=4).pack(side=tk.LEFT)
        cb = ttk.Combobox(
            row,
            textvariable=gas_var,
            values=AVAILABLE_GASES,
            state="readonly",
            width=10,
            font=("Segoe UI", 9),
        )
        cb.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row, text="Mole fraction:", style="Muted.TLabel").pack(side=tk.LEFT)
        ent = ttk.Entry(row, textvariable=frac_var, width=10, font=("Segoe UI", 10))
        ent.pack(side=tk.LEFT, padx=(4, 12))

        del_btn = ttk.Button(
            row,
            text="✕",
            style="TButton",
            width=3,
            command=lambda r=row, d=None: self._delete_gas_row(r),
        )
        del_btn.pack(side=tk.LEFT)

        entry = {"gas_var": gas_var, "frac_var": frac_var, "frame": row}
        # store reference in del_btn closure
        del_btn.configure(command=lambda e=entry: self._delete_gas_row(e))

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

    # ── Temperature preview ───────────────────────────────────────────────────

    def _update_t_preview(self):
        try:
            tmin = float(self._T_min_var.get())
            tmax = float(self._T_max_var.get())
            tstep = float(self._T_step_var.get())
            n = max(0, int(round((tmax - tmin) / tstep)) + 1)
            self._t_preview.configure(text=f"→ {n} temperature points")
        except Exception:
            self._t_preview.configure(text="")

    # ── Experimental data mode ────────────────────────────────────────────────

    def _on_exp_mode_change(self):
        mode = self._exp_mode.get()
        self._preset_frame.pack_forget()
        self._custom_frame.pack_forget()
        if mode == "preset":
            self._preset_frame.pack(fill=tk.X, pady=(6, 0))
        elif mode == "custom":
            self._custom_frame.pack(fill=tk.X, pady=(6, 0))

    # ── Input validation ──────────────────────────────────────────────────────

    def _validate_inputs(self):
        # Gas composition
        gas_comp = {}
        total = 0.0
        for r in self._gas_rows:
            g = r["gas_var"].get()
            try:
                f = float(r["frac_var"].get())
            except ValueError:
                raise ValueError(f"Invalid mole fraction for gas {g}.")
            if f < 0:
                raise ValueError(f"Mole fraction for {g} cannot be negative.")
            gas_comp[g] = f
            total += f
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Mole fractions must sum to 1.0 (current sum = {total:.4f})."
            )

        # Temperature range
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

        # EOS selection
        eos_names = [name for name, v in self._eos_vars.items() if v.get()]
        if not eos_names:
            raise ValueError("Select at least one EOS model.")

        # Experimental data
        exp_data = None
        mode = self._exp_mode.get()
        if mode == "preset":
            key = self._preset_var.get()
            exp_data = dict(PRESET_DATA[key])
        elif mode == "custom":
            lines = self._custom_text.get("1.0", tk.END).strip().splitlines()
            Ts, Ps = [], []
            for ln in lines:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.replace(",", " ").split()
                if len(parts) < 2:
                    raise ValueError(
                        f"Cannot parse line: '{ln}'  " f"(expected T P, got '{ln}')."
                    )
                Ts.append(float(parts[0]))
                Ps.append(float(parts[1]))
            if Ts:
                exp_data = {"T (K)": Ts, "P_eq (MPa)": Ps}

        return gas_comp, {"H2O": 1.0}, T_range, eos_names, exp_data

    # ── Run ───────────────────────────────────────────────────────────────────

    def _on_run(self):
        try:
            gas_comp, liq_comp, T_range, eos_names, exp_data = self._validate_inputs()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return

        self._exp_data = exp_data
        self._run_btn.configure(state=tk.DISABLED)
        self._progress.pack(fill=tk.X, pady=(0, 6))
        self._progress.start(12)
        self._status_var.set("Starting calculation…")

        def _status_cb(msg: str):
            self.after(0, lambda m=msg: self._status_var.set(m))

        def _worker():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results, err = _run_model(
                    gas_comp, liq_comp, T_range, eos_names, _status_cb
                )
            self.after(0, lambda: self._on_done(results, err))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, results: dict, error: Optional[Exception]):
        self._progress.stop()
        self._progress.pack_forget()
        self._run_btn.configure(state=tk.NORMAL)

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
            f"✓  Completed  {len(results)} model(s), "
            f"{n_total} total data rows.  "
            "Opening Plot Builder…"
        )
        self._open_plot_builder()

    # ── Open plot builder ─────────────────────────────────────────────────────

    def _open_plot_builder(self):
        if not self._results:
            return
        win = PlotBuilderWindow(
            master=self,
            results_dict=self._results,
            experimental_data=self._exp_data,
            title="Hydrate — General Plot Builder",
        )
        win.focus_set()
