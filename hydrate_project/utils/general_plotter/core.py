"""
core.py
-------
Pure matplotlib drawing logic.  No tkinter, no GUI code here.
All functions accept a matplotlib Figure/Axes and data; they return nothing.

Grid logic:
  - Row vars  → each selected variable gets its own ROW of subplots
  - Col vars  → each selected variable gets its own COLUMN of subplots
  - Cell (r,c) shows exactly ONE variable: col_vars[c]
    (col selection drives what is plotted; row selection creates repeated
     rows showing the same column variables, useful when comparing groups)

  In the common case where row_vars == col_vars (same list), cell (r,c)
  shows col_vars[c] — a standard multi-panel layout.

  No twin / secondary axis is used anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

# ── Style constants ────────────────────────────────────────────────────────
BG_PLOT = "#ffffff"
BG_AXES = "#fafafa"
FG_TEXT = "#18181b"
GRID_CLR = "#e4e4e7"
SPINE_CLR = "#d4d4d8"

EOS_PALETTE = [
    "#2563eb",  # blue
    "#16a34a",  # green
    "#dc2626",  # red
    "#ea580c",  # orange
    "#7c3aed",  # purple
    "#0891b2",  # cyan
]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LSTYLES = ["-", "--", "-.", ":", "-", "--"]

EXP_COLOR = "#ca8a04"  # amber for experimental scatter


# ── helpers ────────────────────────────────────────────────────────────────


def _safe_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col in df.columns and df[col].notna().any():
        return df[col]
    return None


def get_numeric_columns(results_dict: dict[str, pd.DataFrame]) -> list[str]:
    """Union of numeric columns across all DataFrames, T first then P."""
    cols: set[str] = set()
    for df in results_dict.values():
        cols.update(df.select_dtypes(include=np.number).columns)
    priority = ["T (K)", "P_eq (MPa)"]
    rest = sorted(cols - set(priority))
    return [c for c in priority if c in cols] + rest


# ── axes styling ───────────────────────────────────────────────────────────


def style_axes(ax: plt.Axes, dark: bool = False):
    if dark:
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4", labelsize=8)
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        ax.grid(True, color="#313244", linewidth=0.6, linestyle=":")
    else:
        ax.set_facecolor(BG_AXES)
        ax.tick_params(colors=FG_TEXT, labelsize=8)
        ax.xaxis.label.set_color(FG_TEXT)
        ax.yaxis.label.set_color(FG_TEXT)
        ax.title.set_color(FG_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=0.6, linestyle=":")


def style_legend(ax: plt.Axes, dark: bool = False):
    leg = ax.get_legend()
    if leg is None:
        return
    frame_color = "#313244" if dark else "#ffffff"
    border_color = "#45475a" if dark else SPINE_CLR
    text_color = "#cdd6f4" if dark else FG_TEXT
    leg.get_frame().set_facecolor(frame_color)
    leg.get_frame().set_edgecolor(border_color)
    for txt in leg.get_texts():
        txt.set_color(text_color)
        txt.set_fontsize(7)


# ── single-cell drawing ────────────────────────────────────────────────────


def draw_cell(
    ax: plt.Axes,
    results_dict: dict[str, pd.DataFrame],
    x_var: str,
    y_var: str,
    models: list[str],
    exp_data: Optional[dict],
    exp_overlay: bool,
    dark: bool = False,
):
    """
    Draw one grid cell on a single Y axis — no twin/secondary axis.

    Parameters
    ----------
    y_var   : the single Y variable to plot in this cell.
    models  : list of model keys from results_dict to overlay.
    """
    handles, labels = [], []
    plotted = False

    for m_idx, model in enumerate(models):
        df = results_dict[model]
        if _safe_series(df, x_var) is None or _safe_series(df, y_var) is None:
            continue
        valid = df[[x_var, y_var]].dropna()
        if valid.empty:
            continue

        c = EOS_PALETTE[m_idx % len(EOS_PALETTE)]
        mk = MARKERS[m_idx % len(MARKERS)]
        ls = LSTYLES[0]

        (lh,) = ax.plot(
            valid[x_var],
            valid[y_var],
            color=c,
            marker=mk,
            linestyle=ls,
            linewidth=1.8,
            markersize=4,
            label=model if len(models) > 1 else y_var,
        )
        handles.append(lh)
        labels.append(lh.get_label())
        plotted = True

    # Experimental overlay (only when the Y variable is equilibrium pressure)
    if exp_overlay and exp_data and y_var == "P_eq (MPa)":
        sc = ax.scatter(
            exp_data["T (K)"],
            exp_data["P_eq (MPa)"],
            color=EXP_COLOR,
            marker="x",
            s=55,
            zorder=7,
            linewidths=1.8,
            label="Experimental",
        )
        handles.append(sc)
        labels.append("Experimental")

    if not plotted:
        no_data_color = "#585b70" if dark else SPINE_CLR
        ax.text(
            0.5,
            0.5,
            "No data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=no_data_color,
            fontsize=9,
            style="italic",
        )

    if handles:
        ax.legend(
            handles, labels, fontsize=6.5, loc="best", framealpha=0.9, handlelength=1.4
        )
        style_legend(ax, dark)

    style_axes(ax, dark)


# ── full grid drawing ──────────────────────────────────────────────────────


def build_grid(
    fig: plt.Figure,
    results_dict: dict[str, pd.DataFrame],
    row_vars: list[str],
    col_vars: list[str],
    x_var: str,
    comparison: str,
    exp_data: Optional[dict],
    exp_overlay: bool,
    dark: bool = False,
) -> list[plt.Axes]:
    """
    Clear *fig* and draw a (len(row_vars) × len(col_vars)) grid.

    cell(r, c) independently shows col_vars[c] for row r — one Y axis each.
    row_vars[r] is shown as the Y variable for that row when col_vars is
    the same list; when they differ, each row just repeats the column layout.

    In practice:
      - To get a 2×2 grid of 4 different variables, select 2 row vars and
        2 col vars (same 2 vars in both lists).
      - To get a 2×1 grid, select 2 row vars and 1 col var.
    """
    fig.clear()
    bg = "#1e1e2e" if dark else BG_PLOT
    fig.patch.set_facecolor(bg)

    nrows = max(len(row_vars), 1)
    ncols = max(len(col_vars), 1)
    models = (
        list(results_dict.keys())
        if comparison == "eos"
        else [list(results_dict.keys())[0]]
    )

    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.4)
    axes_out: list[plt.Axes] = []

    for r in range(nrows):
        for c, y_var in enumerate(col_vars):
            ax = fig.add_subplot(gs[r, c])
            draw_cell(
                ax=ax,
                results_dict=results_dict,
                x_var=x_var,
                y_var=y_var,
                models=models,
                exp_data=exp_data,
                exp_overlay=exp_overlay,
                dark=dark,
            )
            ax.set_title(y_var, fontsize=8, pad=5, color="#cdd6f4" if dark else FG_TEXT)
            if r == nrows - 1:
                ax.set_xlabel(x_var, fontsize=8)
            ax.set_ylabel(y_var, fontsize=8)
            axes_out.append(ax)

    return axes_out
