"""
properties_window.py
--------------------
Thermodynamic Properties Viewer + Data Extractor.

Opens as a Toplevel child of LauncherApp and shows every calculated
quantity in a scrollable table.  A separate Export tab lets the user
filter by T range, choose columns, and export to CSV or clipboard.

Layout
------
  [EOS tabs at top]
  ┌──────────────────────────────────────────────────────────────────┐
  │  Summary cards (P range, T range, converged %, avg Z)            │
  ├──────────────────────────────────────────────────────────────────┤
  │  Treeview table (all properties, scrollable)                     │
  ├──────────────────────────────────────────────────────────────────┤
  │  [Filter T from ... to ...]  [Export CSV]  [Copy Clipboard]      │
  └──────────────────────────────────────────────────────────────────┘

  A second "Data Extractor" tab lets the user select columns and rows,
  preview, and export.
"""

from __future__ import annotations

import csv
import io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import numpy as np
import pandas as pd

from hydrate_project.ui.general_plotter import theme as th
from hydrate_project.services.derived import add_dissociation_thermo


# ── Column display metadata ───────────────────────────────────────────────────

# (df_column, display_header, width_px, format_string)
_COL_META: list[tuple[str, str, int, str]] = [
    ("T (K)", "T (K)", 70, "{:.2f}"),
    ("P_eq (MPa)", "P_eq (MPa)", 90, "{:.4f}"),
    ("Optimum_Structure", "Structure", 72, "{}"),
    ("Z", "Z", 62, "{:.4f}"),
    ("a_w", "aᵥ", 68, "{:.5f}"),
    ("gamma_w", "γ_w", 68, "{:.5f}"),
    ("Delta_Mu_w", "Δμ_w (J/mol)", 100, "{:.2f}"),
    ("Delta_Mu_H", "Δμ_H (J/mol)", 100, "{:.2f}"),
    # per-gas columns resolved dynamically
]

_FLOAT_FMT = "{:.5g}"

_SUMMARY_COLS = [
    ("P_eq (MPa)", "P_eq"),
    ("Z", "Z"),
    ("a_w", "aᵥ"),
]


def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if isinstance(val, float):
        return _FLOAT_FMT.format(val)
    return str(val)


class PropertiesWindow(tk.Toplevel):
    """
    Thermodynamic properties viewer + CSV/clipboard data extractor.

    Parameters
    ----------
    master          : parent tkinter widget
    results_dict    : {eos_name: pd.DataFrame}  — from the solver
    experimental_data : {"T (K)": [...], "P_eq (MPa)": [...]} | None
    """

    def __init__(
        self,
        master: tk.Misc,
        results_dict: dict[str, pd.DataFrame],
        experimental_data: Optional[dict] = None,
        title: str = "Thermodynamic Properties",
    ):
        super().__init__(master)
        self._results = results_dict
        self._exp = experimental_data
        self._eos_names = list(results_dict.keys())

        # Prepare enriched DataFrames (with computed dH_diss/dG_diss/dS_diss)
        self._rich: dict[str, pd.DataFrame] = {
            name: add_dissociation_thermo(df) for name, df in results_dict.items()
        }

        self.title(title)
        self.geometry("1200x780")
        self.minsize(900, 600)
        th.apply(self)
        self._build_ui()
        self.after(80, self._populate_table)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── header ──────────────────────────────────────────────────────────
        hdr = ttk.Frame(self, style="TFrame", padding=(14, 10, 14, 6))
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="📊  Thermodynamic Properties", style="Title.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(
            hdr,
            text="Tabular view of all computed quantities per temperature step",
            style="Subtitle.TLabel",
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X)

        # ── main notebook ────────────────────────────────────────────────────
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Tab 1 — Table viewer
        self._tab_table = ttk.Frame(self._nb, style="TFrame")
        self._nb.add(self._tab_table, text="  Properties Table  ")

        # Tab 2 — Data extractor
        self._tab_extract = ttk.Frame(self._nb, style="TFrame")
        self._nb.add(self._tab_extract, text="  Data Extractor  ")

        self._build_table_tab()
        self._build_extractor_tab()

    # ── Table tab ─────────────────────────────────────────────────────────────

    def _build_table_tab(self):
        tab = self._tab_table

        # EOS selector row
        top = ttk.Frame(tab, style="TFrame", padding=(6, 6))
        top.pack(fill=tk.X)
        ttk.Label(top, text="EOS model:", style="Muted.TLabel").pack(side=tk.LEFT)
        self._eos_var = tk.StringVar(value=self._eos_names[0])
        ttk.Combobox(
            top,
            textvariable=self._eos_var,
            values=self._eos_names,
            state="readonly",
            width=26,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(6, 20))
        self._eos_var.trace_add("write", lambda *_: self._populate_table())

        # T filter
        ttk.Label(top, text="Filter T:", style="Muted.TLabel").pack(side=tk.LEFT)
        self._Tmin_v = tk.StringVar(value="")
        self._Tmax_v = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self._Tmin_v, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text="—", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self._Tmax_v, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text="K", style="Muted.TLabel").pack(side=tk.LEFT, padx=(2, 12))
        ttk.Button(
            top, text="Apply", style="TButton", command=self._populate_table
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            top, text="Clear filter", style="TButton", command=self._clear_filter
        ).pack(side=tk.LEFT, padx=2)

        # Summary cards
        self._summary_frame = ttk.Frame(tab, style="TFrame", padding=(6, 0))
        self._summary_frame.pack(fill=tk.X)

        # Treeview
        tree_frame = ttk.Frame(tab, style="TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 2))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(tree_frame, show="headings", selectmode="extended")
        vsb = th.styled_scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self._tree.yview
        )
        hsb = th.styled_scrollbar(
            tree_frame, orient=tk.HORIZONTAL, command=self._tree.xview
        )
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Row striping
        self._tree.tag_configure("odd", background=th.SURFACE0)
        self._tree.tag_configure("even", background=th.BASE)
        self._tree.tag_configure("nan", foreground=th.OVERLAY)

        # Bottom bar
        bot = ttk.Frame(tab, style="TFrame", padding=(6, 4))
        bot.pack(fill=tk.X)
        self._row_label = ttk.Label(bot, text="", style="Muted.TLabel")
        self._row_label.pack(side=tk.LEFT)
        ttk.Button(
            bot,
            text="💾  Export CSV",
            style="Primary.TButton",
            command=self._export_csv_table,
        ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(
            bot, text="📋  Copy to clipboard", style="TButton", command=self._copy_table
        ).pack(side=tk.RIGHT, padx=4)

    # ── Extractor tab ─────────────────────────────────────────────────────────

    def _build_extractor_tab(self):
        tab = self._tab_extract
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1)

        # Instructions
        ttk.Label(
            tab,
            text="Select EOS model, choose columns, set optional T filter, then export.",
            style="Subtitle.TLabel",
            padding=(10, 8),
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Separator(tab, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="ew"
        )

        # Left: EOS + T range
        left = ttk.Frame(tab, style="TFrame", padding=(10, 8))
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 4))

        ttk.Label(left, text="EOS model", style="Section.TLabel").pack(
            anchor=tk.W, pady=(0, 4)
        )
        self._ext_eos_var = tk.StringVar(value=self._eos_names[0])
        ttk.Combobox(
            left,
            textvariable=self._ext_eos_var,
            values=self._eos_names,
            state="readonly",
            width=22,
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(left, text="T range (K)", style="Section.TLabel").pack(
            anchor=tk.W, pady=(0, 4)
        )
        tf = ttk.Frame(left, style="TFrame")
        tf.pack(anchor=tk.W, pady=(0, 12))
        ttk.Label(tf, text="From", style="Muted.TLabel").grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Label(tf, text="To", style="Muted.TLabel").grid(
            row=1, column=0, sticky=tk.W
        )
        self._ext_Tmin = tk.StringVar(value="")
        self._ext_Tmax = tk.StringVar(value="")
        ttk.Entry(tf, textvariable=self._ext_Tmin, width=10).grid(
            row=0, column=1, padx=6, pady=2
        )
        ttk.Entry(tf, textvariable=self._ext_Tmax, width=10).grid(
            row=1, column=1, padx=6, pady=2
        )

        ttk.Button(
            left, text="👁  Preview", style="TButton", command=self._ext_preview
        ).pack(fill=tk.X, pady=(4, 2))
        ttk.Button(
            left,
            text="💾  Export CSV",
            style="Primary.TButton",
            command=self._ext_export,
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            left, text="📋  Copy clipboard", style="TButton", command=self._ext_copy
        ).pack(fill=tk.X, pady=2)

        # Middle: column selector
        mid = ttk.LabelFrame(
            tab, text="  Columns to include  ", style="TLabelframe", padding=(8, 6)
        )
        mid.grid(row=2, column=1, sticky="nsew", padx=4)
        mid.rowconfigure(1, weight=1)

        ttk.Label(mid, text="Ctrl+click to select multiple", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        self._ext_col_lb = th.styled_listbox(mid, height=20, selectmode=tk.MULTIPLE)
        ext_sb = th.styled_scrollbar(
            mid, orient=tk.VERTICAL, command=self._ext_col_lb.yview
        )
        self._ext_col_lb.configure(yscrollcommand=ext_sb.set)
        self._ext_col_lb.grid(row=1, column=0, sticky="nsew")
        ext_sb.grid(row=1, column=1, sticky="ns")
        mid.columnconfigure(0, weight=1)

        sel_row = ttk.Frame(mid, style="TFrame")
        sel_row.grid(row=2, column=0, columnspan=2, pady=(4, 0), sticky="ew")
        ttk.Button(
            sel_row,
            text="Select all",
            style="TButton",
            command=lambda: self._ext_col_lb.selection_set(0, tk.END),
        ).pack(side=tk.LEFT)
        ttk.Button(
            sel_row,
            text="Clear all",
            style="TButton",
            command=lambda: self._ext_col_lb.selection_clear(0, tk.END),
        ).pack(side=tk.LEFT, padx=4)

        # Right: preview text
        right = ttk.LabelFrame(
            tab, text="  Preview (first 10 rows)  ", style="TLabelframe", padding=(8, 6)
        )
        right.grid(row=2, column=2, sticky="nsew", padx=(4, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        tab.columnconfigure(2, weight=2)

        self._ext_preview_text = tk.Text(
            right,
            wrap="none",
            font=("Consolas", 8),
            bg=th.SURFACE0,
            fg=th.TEXT,
            relief="flat",
            state=tk.DISABLED,
        )
        t_sb_v = th.styled_scrollbar(
            right, orient=tk.VERTICAL, command=self._ext_preview_text.yview
        )
        t_sb_h = th.styled_scrollbar(
            right, orient=tk.HORIZONTAL, command=self._ext_preview_text.xview
        )
        self._ext_preview_text.configure(
            yscrollcommand=t_sb_v.set, xscrollcommand=t_sb_h.set
        )
        self._ext_preview_text.grid(row=0, column=0, sticky="nsew")
        t_sb_v.grid(row=0, column=1, sticky="ns")
        t_sb_h.grid(row=1, column=0, sticky="ew")

        self._nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

    # ── Data logic ────────────────────────────────────────────────────────────

    def _get_filtered_df(
        self, eos_var: tk.StringVar, tmin_var: tk.StringVar, tmax_var: tk.StringVar
    ) -> pd.DataFrame:
        name = eos_var.get()
        df = self._rich.get(name, pd.DataFrame()).copy()
        try:
            tmin = float(tmin_var.get())
            df = df[df["T (K)"] >= tmin]
        except (ValueError, KeyError):
            pass
        try:
            tmax = float(tmax_var.get())
            df = df[df["T (K)"] <= tmax]
        except (ValueError, KeyError):
            pass
        return df

    def _get_columns(self, df: pd.DataFrame) -> list[str]:
        priority = [
            "T (K)",
            "P_eq (MPa)",
            "Optimum_Structure",
            "Z",
            "dG_diss (kJ/mol)",
            "dS_diss (kJ/mol.K)",
            "dH_diss (kJ/mol)",
            "a_w",
            "gamma_w",
            "Delta_Mu_w",
            "Delta_Mu_H",
        ]
        rest = [c for c in df.columns if c not in priority]
        return [c for c in priority if c in df.columns] + rest

    def _populate_table(self, *_):
        df = self._get_filtered_df(self._eos_var, self._Tmin_v, self._Tmax_v)
        cols = self._get_columns(df)

        # Build Treeview columns
        self._tree["columns"] = cols
        for col in cols:
            width = 90
            if col == "T (K)":
                width = 70
            elif col in ("P_eq (MPa)", "Z"):
                width = 82
            elif col == "Optimum_Structure":
                width = 72
            elif len(col) > 18:
                width = 115
            anchor = tk.CENTER if col not in ("Optimum_Structure",) else tk.W
            self._tree.heading(col, text=col, anchor=anchor)
            self._tree.column(
                col, width=width, anchor=anchor, minwidth=55, stretch=False
            )

        # Clear existing rows
        self._tree.delete(*self._tree.get_children())

        # Insert rows
        n_ok = 0
        for idx, (_, row) in enumerate(df.iterrows()):
            tag = "odd" if idx % 2 else "even"
            vals = []
            has_nan = False
            for col in cols:
                v = row.get(col, float("nan"))
                if isinstance(v, float) and np.isnan(v):
                    has_nan = True
                vals.append(_fmt(v))
            if has_nan:
                tag = "nan"
            else:
                n_ok += 1
            self._tree.insert("", tk.END, values=vals, tags=(tag,))

        self._row_label.configure(
            text=f"{len(df)} rows  •  {n_ok} converged  •  {len(df)-n_ok} failed"
        )
        self._update_summary(df)

    def _update_summary(self, df: pd.DataFrame):
        for w in self._summary_frame.winfo_children():
            w.destroy()

        def _card(label: str, val: str, color: str = th.TEXT):
            f = ttk.Frame(self._summary_frame, style="TFrame", padding=(10, 4))
            f.pack(side=tk.LEFT, padx=(0, 10))
            ttk.Label(f, text=label, style="Muted.TLabel").pack(anchor=tk.W)
            ttk.Label(
                f,
                text=val,
                foreground=color,
                font=("Segoe UI", 10, "bold"),
                background=th.BASE,
            ).pack(anchor=tk.W)

        if df.empty:
            _card("No data", "—")
            return

        T = df["T (K)"].dropna()
        P = df["P_eq (MPa)"].dropna()
        Z = df["Z"].dropna() if "Z" in df.columns else pd.Series(dtype=float)
        n = len(df)
        n_ok = df["P_eq (MPa)"].notna().sum()

        _card(
            "T range",
            f"{T.min():.2f} – {T.max():.2f} K" if len(T) > 1 else f"{T.iloc[0]:.2f} K",
        )
        if len(P):
            _card("P range", f"{P.min():.3f} – {P.max():.3f} MPa", th.BLUE)
        if len(Z):
            _card("Z̄ (avg)", f"{Z.mean():.4f}", th.GREEN if Z.mean() < 1 else th.TEXT)
        _card("Converged", f"{n_ok}/{n}", th.GREEN if n_ok == n else th.YELLOW)
        if "dH_diss (kJ/mol)" in df.columns:
            dH = df["dH_diss (kJ/mol)"].dropna()
            if len(dH):
                _card("ΔH_diss", f"{dH.median():.1f} kJ/mol", th.MAUVE)
        if "dS_diss (kJ/mol.K)" in df.columns:
            dS = df["dS_diss (kJ/mol.K)"].dropna()
            if len(dS):
                _card("ΔS_diss", f"{dS.median():.3f} kJ/mol.K", th.MAUVE)
                
        if "dG_diss (kJ/mol)" in df.columns:
            dG = df["dG_diss (kJ/mol)"].dropna()
            if len(dG):
                _card("ΔG_diss", f"{dG.median():.1f} kJ/mol", th.MAUVE)

    def _clear_filter(self):
        self._Tmin_v.set("")
        self._Tmax_v.set("")
        self._populate_table()

    def _on_tab_change(self, *_):
        if self._nb.index(self._nb.select()) == 1:
            self._populate_extractor_cols()

    def _populate_extractor_cols(self):
        name = self._ext_eos_var.get()
        df = self._rich.get(name, pd.DataFrame())
        cols = self._get_columns(df)
        self._ext_col_lb.delete(0, tk.END)
        for c in cols:
            self._ext_col_lb.insert(tk.END, c)
        self._ext_col_lb.selection_set(0, tk.END)  # select all by default

    def _get_ext_selection(self) -> tuple[pd.DataFrame, list[str]]:
        df = self._get_filtered_df(self._ext_eos_var, self._ext_Tmin, self._ext_Tmax)
        idxs = self._ext_col_lb.curselection()
        if not idxs:
            cols = self._get_columns(df)
        else:
            cols = [self._ext_col_lb.get(i) for i in idxs]
        available = [c for c in cols if c in df.columns]
        return df[available], available

    def _df_to_csv_str(self, df: pd.DataFrame) -> str:
        buf = io.StringIO()
        df.to_csv(buf, index=False, float_format="%.6g")
        return buf.getvalue()

    def _df_to_table_str(self, df: pd.DataFrame) -> str:
        return df.to_string(index=False, float_format=lambda x: f"{x:.5g}")

    # ── Table tab actions ─────────────────────────────────────────────────────

    def _export_csv_table(self):
        df = self._get_filtered_df(self._eos_var, self._Tmin_v, self._Tmax_v)
        self._save_csv(df)

    def _copy_table(self):
        df = self._get_filtered_df(self._eos_var, self._Tmin_v, self._Tmax_v)
        self.clipboard_clear()
        self.clipboard_append(self._df_to_csv_str(df))
        self._row_label.configure(text="✓  Copied to clipboard")

    # ── Extractor tab actions ─────────────────────────────────────────────────

    def _ext_preview(self):
        df, _ = self._get_ext_selection()
        preview = df.head(10)
        txt = self._df_to_table_str(preview)
        self._ext_preview_text.configure(state=tk.NORMAL)
        self._ext_preview_text.delete("1.0", tk.END)
        self._ext_preview_text.insert(tk.END, txt)
        self._ext_preview_text.configure(state=tk.DISABLED)

    def _ext_export(self):
        df, _ = self._get_ext_selection()
        self._save_csv(df)

    def _ext_copy(self):
        df, _ = self._get_ext_selection()
        self.clipboard_clear()
        self.clipboard_append(self._df_to_csv_str(df))
        messagebox.showinfo("Copied", f"Copied {len(df)} rows to clipboard.")

    def _save_csv(self, df: pd.DataFrame):
        if df.empty:
            messagebox.showwarning("No data", "Nothing to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
            initialfile="hydrate_properties.csv",
            title="Save properties as CSV…",
        )
        if path:
            try:
                df.to_csv(path, index=False, float_format="%.6g")
                self._row_label.configure(text=f"✓  Saved {len(df)} rows → {path}")
            except Exception as exc:
                messagebox.showerror("Save error", str(exc))
