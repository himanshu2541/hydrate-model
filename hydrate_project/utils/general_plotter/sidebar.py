"""
sidebar.py
----------
The left control panel.  All user selections are exposed via:
    sidebar.get_config() -> PlotConfig

Uses proper scrollable Listboxes, Combobox, Radiobuttons, Checkbuttons.
No matplotlib widgets — pure tkinter/ttk.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass, field
from typing import Optional, Callable

from hydrate_project.utils.general_plotter import theme as th
from hydrate_project.utils.general_plotter.core import get_numeric_columns


# ── data class for selections ──────────────────────────────────────────────

@dataclass
class PlotConfig:
    x_var:       str
    row_vars:    list[str]
    col_vars:    list[str]
    comparison:  str          # "eos" | "single"
    exp_overlay: bool
    dark_mode:   bool


# ── reusable widgets ───────────────────────────────────────────────────────

class _Section(ttk.LabelFrame):
    """A styled LabelFrame used as a section card in the sidebar."""
    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, text=f"  {title}  ",
                         style="TLabelframe", padding=(8, 6), **kwargs)


class _MultiListbox(ttk.Frame):
    """
    A scrollable Listbox (multi-select) with a search bar.
    Exposes `.get_selected() -> list[str]` and `.set_selected(items)`.
    """
    def __init__(self, parent, options: list[str],
                 height: int = 7, initial: list[str] | None = None, **kwargs):
        super().__init__(parent, style="TFrame", **kwargs)
        self._options = options

        # Search bar
        search_frame = ttk.Frame(self, style="TFrame")
        search_frame.pack(fill=tk.X, pady=(0, 4))
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", self._on_search)
        search_entry = tk.Entry(
            search_frame,
            textvariable=self._search_var,
            bg=th.SURFACE0, fg=th.SUBTEXT,
            insertbackground=th.TEXT,
            relief="flat",
            highlightthickness=1,
            highlightcolor=th.BLUE,
            highlightbackground=th.SURFACE1,
            font=("Segoe UI", 8),
        )
        search_entry.pack(fill=tk.X)
        # Placeholder effect
        search_entry.insert(0, "Filter…")
        search_entry.bind("<FocusIn>",  lambda e: self._clear_placeholder(search_entry))
        search_entry.bind("<FocusOut>", lambda e: self._set_placeholder(search_entry))

        # Listbox + scrollbar
        list_frame = ttk.Frame(self, style="TFrame")
        list_frame.pack(fill=tk.BOTH, expand=True)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self._lb = th.styled_listbox(list_frame, height=height,
                                     selectmode=tk.MULTIPLE)
        sb = th.styled_scrollbar(list_frame, orient=tk.VERTICAL,
                                 command=self._lb.yview)
        self._lb.config(yscrollcommand=sb.set)
        self._lb.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self._populate(options, initial or [])

    def _clear_placeholder(self, entry):
        if entry.get() == "Filter…":
            entry.delete(0, tk.END)
            entry.config(fg=th.TEXT)

    def _set_placeholder(self, entry):
        if entry.get() == "":
            entry.insert(0, "Filter…")
            entry.config(fg=th.SUBTEXT)

    def _on_search(self, *_):
        query = self._search_var.get().lower()
        if query == "filter…":
            query = ""
        current = self.get_selected()
        self._populate(
            [o for o in self._options if query in o.lower()],
            current,
        )

    def _populate(self, opts: list[str], selected: list[str]):
        self._lb.delete(0, tk.END)
        for opt in opts:
            self._lb.insert(tk.END, opt)
        # Restore selection
        for i, item in enumerate(opts):
            if item in selected:
                self._lb.selection_set(i)

    def get_selected(self) -> list[str]:
        indices = self._lb.curselection()
        return [self._lb.get(i) for i in indices]

    def set_selected(self, items: list[str]):
        self._lb.selection_clear(0, tk.END)
        for i in range(self._lb.size()):
            if self._lb.get(i) in items:
                self._lb.selection_set(i)


class _RadioGroup(ttk.Frame):
    def __init__(self, parent, options: list[tuple[str, str]],
                 default: str = "", **kwargs):
        super().__init__(parent, style="TFrame", **kwargs)
        self._var = tk.StringVar(value=default)
        for value, label in options:
            rb = ttk.Radiobutton(
                self, text=label, value=value,
                variable=self._var, style="TRadiobutton",
            )
            rb.pack(anchor=tk.W, pady=2)

    def get(self) -> str:
        return self._var.get()

    def set(self, value: str):
        self._var.set(value)


# ── main sidebar ───────────────────────────────────────────────────────────

class Sidebar(ttk.Frame):
    """
    Full control panel.  Fixed width, scrollable content.

    Layout (top → bottom):
        ┌─────────────────┐
        │  ⚙ Plot Builder │  ← header
        ├─────────────────┤
        │  X Axis         │  ← combobox
        ├─────────────────┤
        │  Row Variables  │  ← multi-listbox
        ├─────────────────┤
        │  Col Variables  │  ← multi-listbox
        ├─────────────────┤
        │  Comparison     │  ← radio
        ├─────────────────┤
        │  Options        │  ← checkbuttons
        ├─────────────────┤
        │  [▶ Update]     │  ← primary button
        │  [💾 Save PNG]  │
        └─────────────────┘
    """

    WIDTH = 260

    def __init__(self, parent,
                 results_dict: dict,
                 on_update: Callable,
                 on_save: Callable,
                 **kwargs):
        super().__init__(parent, style="Sidebar.TFrame",
                         width=self.WIDTH, **kwargs)
        self.pack_propagate(False)

        self._on_update = on_update
        self._on_save   = on_save
        self._columns   = get_numeric_columns(results_dict)
        self._eos_names = list(results_dict.keys())

        self._build_header()
        self._build_body()

    # ── header ─────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = ttk.Frame(self, style="Sidebar.TFrame", padding=(12, 12, 12, 8))
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="⚙  Plot Builder", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(hdr, text="Configure and preview your chart grid",
                  style="Subtitle.TLabel").pack(anchor=tk.W, pady=(2, 0))
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X)

    # ── scrollable body ────────────────────────────────────────────────────
    def _build_body(self):
        # Outer container + canvas for scrolling
        outer = ttk.Frame(self, style="Sidebar.TFrame")
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer, bg=th.MANTLE,
                           highlightthickness=0, bd=0)
        vsb = th.styled_scrollbar(outer, orient=tk.VERTICAL,
                                  command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas, style="Sidebar.TFrame", padding=(10, 6))
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_frame_config(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_config(e):
            canvas.itemconfig(inner_id, width=e.width)

        inner.bind("<Configure>", _on_frame_config)
        canvas.bind("<Configure>", _on_canvas_config)

        # Mousewheel scrolling
        def _on_scroll(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_scroll)

        self._populate_sections(inner)

    # ── sections ───────────────────────────────────────────────────────────
    def _populate_sections(self, parent: ttk.Frame):
        PAD = {"pady": (0, 10), "fill": tk.X}

        # ── X Axis ─────────────────────────────────────────────────────────
        sec_x = _Section(parent, "X Axis")
        sec_x.pack(**PAD)
        x_opts = [c for c in self._columns if c in ("T (K)", "P_eq (MPa)")] \
                 + [c for c in self._columns if c not in ("T (K)", "P_eq (MPa)")]
        self._x_var = tk.StringVar(value=x_opts[0] if x_opts else "")
        x_cb = ttk.Combobox(sec_x, textvariable=self._x_var,
                            values=x_opts, state="readonly",
                            style="TCombobox", font=("Segoe UI", 9))
        x_cb.pack(fill=tk.X)

        # ── Row Variables ──────────────────────────────────────────────────
        sec_row = _Section(parent, "Row Variables  (Y-left)")
        sec_row.pack(**PAD)
        ttk.Label(sec_row,
                  text="Ctrl+click to select multiple",
                  style="Muted.TLabel").pack(anchor=tk.W, pady=(0, 4))
        self._row_lb = _MultiListbox(sec_row, self._columns, height=7,
                                     initial=["P_eq (MPa)"])
        self._row_lb.pack(fill=tk.BOTH, expand=True)

        # ── Col Variables ──────────────────────────────────────────────────
        sec_col = _Section(parent, "Col Variables  (Y-right / twin)")
        sec_col.pack(**PAD)
        ttk.Label(sec_col,
                  text="Overlaid on twin right axis",
                  style="Muted.TLabel").pack(anchor=tk.W, pady=(0, 4))
        self._col_lb = _MultiListbox(sec_col, self._columns, height=7,
                                     initial=["Theta_Small_CO2"])
        self._col_lb.pack(fill=tk.BOTH, expand=True)

        # ── Comparison Mode ────────────────────────────────────────────────
        sec_cmp = _Section(parent, "Comparison Mode")
        sec_cmp.pack(**PAD)
        self._comp_radio = _RadioGroup(
            sec_cmp,
            options=[
                ("eos",    f"Overlay all  ({len(self._eos_names)} EOS models)"),
                ("single", f"Single  ({self._eos_names[0]})"),
            ],
            default="eos",
        )
        self._comp_radio.pack(fill=tk.X)

        # ── Options ────────────────────────────────────────────────────────
        sec_opt = _Section(parent, "Options")
        sec_opt.pack(**PAD)
        self._exp_var  = tk.BooleanVar(value=True)
        self._dark_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sec_opt, text="Show experimental data",
                        variable=self._exp_var,
                        style="TCheckbutton").pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(sec_opt, text="Dark plot background",
                        variable=self._dark_var,
                        style="TCheckbutton").pack(anchor=tk.W, pady=2)

        # ── Action buttons ─────────────────────────────────────────────────
        btn_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        btn_frame.pack(fill=tk.X, pady=(4, 8))

        ttk.Button(btn_frame, text="▶   Update Plot",
                   style="Primary.TButton",
                   command=self._on_update).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btn_frame, text="💾   Save PNG",
                   style="TButton",
                   command=self._on_save).pack(fill=tk.X)

    # ── public API ─────────────────────────────────────────────────────────
    def get_config(self) -> PlotConfig:
        rows = self._row_lb.get_selected() or ["P_eq (MPa)"]
        cols = self._col_lb.get_selected() or ["T (K)"]
        return PlotConfig(
            x_var       = self._x_var.get() or "T (K)",
            row_vars    = rows,
            col_vars    = cols,
            comparison  = self._comp_radio.get(),
            exp_overlay = self._exp_var.get(),
            dark_mode   = self._dark_var.get(),
        )