"""
core.py
-------
Pure matplotlib drawing logic.  No tkinter, no GUI code here.
All functions accept a matplotlib Figure/Axes and data; they return nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

# ── Style constants ────────────────────────────────────────────────────────
BG_PLOT   = "#1e1e2e"
BG_AXES   = "#181825"
FG_TEXT   = "#cdd6f4"
GRID_CLR  = "#313244"
SPINE_CLR = "#45475a"

EOS_PALETTE = [
    "#89b4fa",  # blue
    "#a6e3a1",  # green
    "#f38ba8",  # red
    "#fab387",  # peach
    "#cba6f7",  # mauve
    "#89dceb",  # sky
]
MARKERS  = ["o", "s", "^", "D", "v", "P"]
LSTYLES  = ["-", "--", "-.", ":", "-", "--"]

EXP_COLOR  = "#f9e2af"   # yellow for experimental scatter
TWIN_COLOR = "#f38ba8"   # right-axis colour


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

def style_axes(ax: plt.Axes, dark: bool = True):
    if not dark:
        return
    ax.set_facecolor(BG_AXES)
    ax.tick_params(colors=FG_TEXT, labelsize=8)
    ax.xaxis.label.set_color(FG_TEXT)
    ax.yaxis.label.set_color(FG_TEXT)
    ax.title.set_color(FG_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.6, linestyle=":")


def style_legend(ax: plt.Axes):
    leg = ax.get_legend()
    if leg is None:
        return
    leg.get_frame().set_facecolor("#313244")
    leg.get_frame().set_edgecolor(SPINE_CLR)
    for txt in leg.get_texts():
        txt.set_color(FG_TEXT)
        txt.set_fontsize(7)


# ── single-cell drawing ────────────────────────────────────────────────────

def draw_cell(
    ax: plt.Axes,
    results_dict: dict[str, pd.DataFrame],
    x_var: str,
    y_primary: str,
    y_secondary: Optional[str],
    models: list[str],
    exp_data: Optional[dict],
    exp_overlay: bool,
    dark: bool = True,
):
    """Draw one grid cell.  y_secondary is plotted on a twin right axis if set."""
    twin_ax: Optional[plt.Axes] = None
    handles, labels = [], []
    plotted = False

    # ── primary y ─────────────────────────────────────────────────────────
    for m_idx, model in enumerate(models):
        df = results_dict[model]
        xs = _safe_series(df, x_var)
        ys = _safe_series(df, y_primary)
        if xs is None or ys is None:
            continue
        valid = df[[x_var, y_primary]].dropna()
        if valid.empty:
            continue
        c  = EOS_PALETTE[m_idx % len(EOS_PALETTE)]
        mk = MARKERS[m_idx % len(MARKERS)]
        lh, = ax.plot(
            valid[x_var], valid[y_primary],
            color=c, marker=mk, linestyle=LSTYLES[0],
            linewidth=1.8, markersize=4,
            label=model if len(models) > 1 else y_primary,
        )
        handles.append(lh)
        labels.append(model if len(models) > 1 else y_primary)
        plotted = True

    # ── secondary y (twin axis) ────────────────────────────────────────────
    if y_secondary and y_secondary != y_primary:
        twin_ax = ax.twinx()
        if dark:
            twin_ax.set_facecolor(BG_AXES)
            twin_ax.tick_params(colors=TWIN_COLOR, labelsize=7)
            twin_ax.yaxis.label.set_color(TWIN_COLOR)
            for spine in twin_ax.spines.values():
                spine.set_edgecolor(SPINE_CLR)

        for m_idx, model in enumerate(models):
            df = results_dict[model]
            xs = _safe_series(df, x_var)
            ys = _safe_series(df, y_secondary)
            if xs is None or ys is None:
                continue
            valid = df[[x_var, y_secondary]].dropna()
            if valid.empty:
                continue
            c  = TWIN_COLOR
            mk = MARKERS[(m_idx + 3) % len(MARKERS)]
            lh, = twin_ax.plot(
                valid[x_var], valid[y_secondary],
                color=c, marker=mk, linestyle=LSTYLES[1],
                linewidth=1.4, markersize=3, alpha=0.85,
                label=f"{model} [{y_secondary}]" if len(models) > 1 else y_secondary,
            )
            handles.append(lh)
            labels.append(lh.get_label())
            plotted = True

        twin_ax.set_ylabel(y_secondary, fontsize=8, color=TWIN_COLOR)

    # ── experimental overlay ───────────────────────────────────────────────
    if exp_overlay and exp_data and y_primary == "P_eq (MPa)":
        sc = ax.scatter(
            exp_data["T (K)"], exp_data["P_eq (MPa)"],
            color=EXP_COLOR, marker="x", s=55, zorder=7, linewidths=1.8,
            label="Experimental",
        )
        handles.append(sc)
        labels.append("Experimental")

    # ── empty state ────────────────────────────────────────────────────────
    if not plotted:
        ax.text(0.5, 0.5, "No data available",
                transform=ax.transAxes, ha="center", va="center",
                color=SPINE_CLR, fontsize=9, style="italic")

    # ── legend ─────────────────────────────────────────────────────────────
    if handles:
        ax.legend(handles, labels, fontsize=6.5, loc="best",
                  framealpha=0.85, handlelength=1.4)
        style_legend(ax)

    style_axes(ax, dark)


# ── full grid drawing ──────────────────────────────────────────────────────

def build_grid(
    fig: plt.Figure,
    results_dict: dict[str, pd.DataFrame],
    row_vars: list[str],
    col_vars: list[str],
    x_var: str,
    comparison: str,        # "eos" | "single"
    exp_data: Optional[dict],
    exp_overlay: bool,
    dark: bool = True,
) -> list[plt.Axes]:
    """
    Clear *fig* and draw a (len(row_vars) × len(col_vars)) grid.

    Cell (r, c):
      - Primary Y  = row_vars[r]
      - Secondary Y (twin right axis) = col_vars[c]  if ≠ primary

    Returns the list of created axes.
    """
    fig.clear()
    if dark:
        fig.patch.set_facecolor(BG_PLOT)

    nrows = max(len(row_vars), 1)
    ncols = max(len(col_vars), 1)
    models = list(results_dict.keys()) if comparison == "eos" else [list(results_dict.keys())[0]]

    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.45)
    axes_out: list[plt.Axes] = []

    for r, y_row in enumerate(row_vars):
        for c, y_col in enumerate(col_vars):
            ax = fig.add_subplot(gs[r, c])
            y_sec = y_col if y_col != y_row else None
            draw_cell(
                ax=ax,
                results_dict=results_dict,
                x_var=x_var,
                y_primary=y_row,
                y_secondary=y_sec,
                models=models,
                exp_data=exp_data,
                exp_overlay=exp_overlay,
                dark=dark,
            )
            # Axis labels
            cell_title = f"{y_row}" if y_row == y_col else f"{y_row}  /  {y_col}"
            ax.set_title(cell_title, fontsize=8, pad=4,
                         color=FG_TEXT if dark else "black")
            if r == nrows - 1:
                ax.set_xlabel(x_var, fontsize=8)
            if c == 0:
                ax.set_ylabel(y_row, fontsize=8)
            axes_out.append(ax)

    return axes_out