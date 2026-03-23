"""
theme.py
--------
Applies a clean light ttk.Style to a Tk root.
Call `apply(root)` before building any widgets.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk

# ── palette ────────────────────────────────────────────────────────────────
BASE = "#ffffff"  # window background
MANTLE = "#f4f4f5"  # sidebar / panel background
CRUST = "#e4e4e7"  # deepest surface
SURFACE0 = "#fafafa"  # card / input background
SURFACE1 = "#d4d4d8"  # border / divider
SURFACE2 = "#a1a1aa"  # muted border / active

TEXT = "#18181b"  # primary text
SUBTEXT = "#52525b"  # secondary text
OVERLAY = "#a1a1aa"  # placeholder / disabled

BLUE = "#2563eb"  # primary accent
GREEN = "#16a34a"  # success
MAUVE = "#7c3aed"  # purple
PEACH = "#ea580c"  # orange
RED = "#dc2626"  # danger
YELLOW = "#ca8a04"  # warning

BTN_BG = SURFACE0
BTN_HOVER = CRUST
BTN_ACTIVE = SURFACE1
BTN_PRIMARY = BLUE
ACCENT = BLUE


def apply(root: tk.Tk | tk.Toplevel):
    """Configure the ttk.Style for the entire application."""
    style = ttk.Style(root)

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
        fieldbackground=SURFACE0,
        troughcolor=CRUST,
        bordercolor=SURFACE1,
        darkcolor=SURFACE1,
        lightcolor=BASE,
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
        background=[("active", MAUVE), ("pressed", "#1d4ed8")],
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
    style.map("Danger.TButton", background=[("active", "#ef4444")])

    style.configure(
        "Success.TButton",
        background=GREEN,
        foreground=BASE,
        font=("Segoe UI", 9),
        relief="flat",
        padding=(10, 6),
    )
    style.map("Success.TButton", background=[("active", "#15803d")])

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
        arrowcolor=SUBTEXT,
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
        background=CRUST,
        troughcolor=MANTLE,
        arrowcolor=SUBTEXT,
        bordercolor=SURFACE1,
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
        background=MANTLE,
        foreground=SUBTEXT,
        padding=(12, 5),
        bordercolor=SURFACE1,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", BASE), ("active", CRUST)],
        foreground=[("selected", TEXT)],
    )

    return style


# ── Custom Listbox wrapper (no ttk Listbox) ───────────────────────────────


def styled_listbox(parent, **kwargs) -> tk.Listbox:
    """Return a light-themed tk.Listbox with consistent colours."""
    defaults = dict(
        bg=SURFACE0,
        fg=TEXT,
        selectbackground=BLUE,
        selectforeground=BASE,
        activestyle="none",
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightcolor=BLUE,
        highlightbackground=SURFACE1,
        font=("Segoe UI", 9),
        exportselection=False,
    )
    defaults.update(kwargs)
    return tk.Listbox(parent, **defaults) # type: ignore


def styled_scrollbar(parent, **kwargs) -> ttk.Scrollbar:
    return ttk.Scrollbar(parent, style="TScrollbar", **kwargs)
