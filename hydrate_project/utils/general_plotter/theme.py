"""
theme.py
--------
Applies a cohesive dark (Catppuccin Mocha-inspired) ttk.Style to a Tk root.
Call `apply(root)` before building any widgets.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk

# ── palette ────────────────────────────────────────────────────────────────
BASE = "#1e1e2e"  # window background
MANTLE = "#181825"  # darker surface
CRUST = "#11111b"  # deepest surface
SURFACE0 = "#313244"  # card / panel bg
SURFACE1 = "#45475a"  # border / divider
SURFACE2 = "#585b70"  # muted elements

TEXT = "#cdd6f4"  # primary text
SUBTEXT = "#a6adc8"  # secondary text
OVERLAY = "#6c7086"  # placeholder / disabled

BLUE = "#89b4fa"
GREEN = "#a6e3a1"
MAUVE = "#cba6f7"
PEACH = "#fab387"
RED = "#f38ba8"
YELLOW = "#f9e2af"

BTN_BG = SURFACE0
BTN_HOVER = SURFACE1
BTN_ACTIVE = SURFACE2
BTN_PRIMARY = BLUE
ACCENT = BLUE


def apply(root: tk.Tk | tk.Toplevel):
    """Configure the ttk.Style for the entire application."""
    style = ttk.Style(root)

    # Use clam as the base (works on all platforms)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # ── General ──────────────────────────────────────────────────────────
    root.configure(bg=BASE)
    style.configure(
        ".",
        background=BASE,
        foreground=TEXT,
        fieldbackground=MANTLE,
        troughcolor=MANTLE,
        bordercolor=SURFACE1,
        darkcolor=CRUST,
        lightcolor=SURFACE0,
        selectbackground=BLUE,
        selectforeground=BASE,
        insertcolor=TEXT,
        font=("Segoe UI", 9),
    )

    # ── Frame ─────────────────────────────────────────────────────────────
    style.configure("TFrame", background=BASE)
    style.configure("Card.TFrame", background=SURFACE0, relief="flat", borderwidth=1)
    style.configure("Sidebar.TFrame", background=MANTLE)

    # ── LabelFrame ────────────────────────────────────────────────────────
    style.configure(
        "TLabelframe",
        background=SURFACE0,
        bordercolor=SURFACE1,
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label",
        background=SURFACE0,
        foreground=BLUE,
        font=("Segoe UI", 9, "bold"),
    )

    # ── Label ─────────────────────────────────────────────────────────────
    style.configure("TLabel", background=BASE, foreground=TEXT)
    style.configure("Sidebar.TLabel", background=MANTLE, foreground=TEXT)
    style.configure(
        "Section.TLabel",
        background=MANTLE,
        foreground=BLUE,
        font=("Segoe UI", 8, "bold"),
    )
    style.configure(
        "Muted.TLabel", background=MANTLE, foreground=OVERLAY, font=("Segoe UI", 8)
    )
    style.configure(
        "Title.TLabel",
        background=MANTLE,
        foreground=TEXT,
        font=("Segoe UI", 11, "bold"),
    )
    style.configure(
        "Subtitle.TLabel", background=MANTLE, foreground=SUBTEXT, font=("Segoe UI", 8)
    )

    # ── Button ────────────────────────────────────────────────────────────
    style.configure(
        "TButton",
        background=SURFACE0,
        foreground=TEXT,
        bordercolor=SURFACE1,
        focuscolor="none",
        relief="flat",
        padding=(10, 6),
        font=("Segoe UI", 9),
    )
    style.map(
        "TButton",
        background=[("active", BTN_HOVER), ("pressed", BTN_ACTIVE)],
        foreground=[("active", TEXT)],
        bordercolor=[("active", SURFACE2)],
    )

    style.configure(
        "Primary.TButton",
        background=BLUE,
        foreground=BASE,
        font=("Segoe UI", 9, "bold"),
        relief="flat",
        padding=(10, 7),
    )
    style.map(
        "Primary.TButton",
        background=[("active", MAUVE), ("pressed", "#7287fd")],
        foreground=[("active", BASE)],
    )

    style.configure(
        "Danger.TButton",
        background=RED,
        foreground=BASE,
        font=("Segoe UI", 9),
        relief="flat",
        padding=(10, 6),
    )
    style.map("Danger.TButton", background=[("active", "#eba0ac")])

    style.configure(
        "Success.TButton",
        background=GREEN,
        foreground=BASE,
        font=("Segoe UI", 9),
        relief="flat",
        padding=(10, 6),
    )
    style.map("Success.TButton", background=[("active", "#94e2b8")])

    # ── Checkbutton ───────────────────────────────────────────────────────
    style.configure(
        "TCheckbutton",
        background=MANTLE,
        foreground=TEXT,
        focuscolor="none",
        indicatorcolor=SURFACE0,
        indicatorrelief="flat",
    )
    style.map(
        "TCheckbutton",
        indicatorcolor=[("selected", BLUE), ("active", SURFACE1)],
        foreground=[("active", TEXT)],
    )

    # ── Radiobutton ───────────────────────────────────────────────────────
    style.configure(
        "TRadiobutton",
        background=MANTLE,
        foreground=TEXT,
        focuscolor="none",
        indicatorcolor=SURFACE0,
    )
    style.map(
        "TRadiobutton",
        indicatorcolor=[("selected", BLUE)],
        foreground=[("active", TEXT)],
    )

    # ── Combobox ──────────────────────────────────────────────────────────
    style.configure(
        "TCombobox",
        fieldbackground=SURFACE0,
        background=SURFACE0,
        foreground=TEXT,
        arrowcolor=TEXT,
        bordercolor=SURFACE1,
        selectbackground=BLUE,
        selectforeground=BASE,
        padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", SURFACE0)],
        foreground=[("readonly", TEXT)],
        selectbackground=[("readonly", BLUE)],
    )

    # ── Scrollbar ─────────────────────────────────────────────────────────
    style.configure(
        "TScrollbar",
        background=SURFACE0,
        troughcolor=MANTLE,
        arrowcolor=OVERLAY,
        bordercolor=MANTLE,
        relief="flat",
        arrowsize=12,
    )
    style.map("TScrollbar", background=[("active", SURFACE1), ("pressed", SURFACE2)])

    # ── Separator ─────────────────────────────────────────────────────────
    style.configure("TSeparator", background=SURFACE1)

    # ── Notebook ──────────────────────────────────────────────────────────
    style.configure("TNotebook", background=BASE, bordercolor=SURFACE1)
    style.configure(
        "TNotebook.Tab",
        background=SURFACE0,
        foreground=SUBTEXT,
        padding=(12, 5),
        bordercolor=SURFACE1,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", BASE), ("active", SURFACE1)],
        foreground=[("selected", TEXT)],
    )

    return style


# ── Custom Listbox wrapper (no ttk Listbox) ───────────────────────────────


def styled_listbox(parent, **kwargs) -> tk.Listbox:
    """Return a dark-themed tk.Listbox with consistent colours."""
    defaults = dict(
        bg=SURFACE0,
        fg=TEXT,
        selectbackground=BLUE,
        selectforeground=BASE,
        activestyle="none",
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightcolor=SURFACE1,
        highlightbackground=SURFACE1,
        font=("Segoe UI", 9),
        exportselection=False,
    )
    defaults.update(kwargs)
    return tk.Listbox(parent, **defaults) # type: ignore


def styled_scrollbar(parent, **kwargs) -> ttk.Scrollbar:
    return ttk.Scrollbar(parent, style="TScrollbar", **kwargs)
