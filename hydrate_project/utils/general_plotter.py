"""
GeneralPlotter
==============
An interactive matplotlib-based grid plotter for hydrate project results.

Usage
-----
    plotter = GeneralPlotter(all_results, experimental_data)
    plotter.show()

    # Or build a static grid programmatically:
    plotter.plot_grid(
        row_vars=["P_eq (MPa)", "ΔH_diss (kJ/mol)"],
        col_vars=["Theta_Small_CO2", "Theta_Large_CO2"],
        comparison="eos",          # "eos" | "structure" | "single"
        x_axis="T (K)",
        exp_overlay=True,
    )
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import CheckButtons, RadioButtons, Button, TextBox
import numpy as np
import pandas as pd
from typing import Optional


# ── colour / marker cycles ──────────────────────────────────────────────────
_COLORS   = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
_MARKERS  = ["o", "s", "^", "D", "v", "P"]
_LSTYLES  = ["-", "--", "-.", ":", "-", "--"]


# ── helpers ─────────────────────────────────────────────────────────────────

def _safe_col(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Return a column if it exists and has at least one non-NaN value, else None."""
    if col in df.columns and df[col].notna().any():
        return df[col]
    return None


def _get_numeric_cols(results_dict: dict) -> list[str]:
    """Union of all numeric columns across every DataFrame."""
    cols: set[str] = set()
    for df in results_dict.values():
        cols.update(c for c in df.select_dtypes(include=np.number).columns)
    # Sort sensibly: T first, then P, then the rest alphabetically
    priority = ["T (K)", "P_eq (MPa)"]
    rest = sorted(cols - set(priority))
    return [c for c in priority if c in cols] + rest


# ── main class ───────────────────────────────────────────────────────────────

class GeneralPlotter:
    """
    Parameters
    ----------
    results_dict : dict[str, pd.DataFrame]
        Keys are comparison labels (e.g. EOS names), values are result DataFrames.
    experimental_data : dict | None
        {"T (K)": [...], "P_eq (MPa)": [...]} or None.
    """

    def __init__(self, results_dict: dict, experimental_data: dict | None = None):
        self.results = results_dict
        self.exp     = experimental_data
        self.labels  = list(results_dict.keys())
        self.all_cols = _get_numeric_cols(results_dict)

        # Default selections for the interactive GUI
        self._x_var    = "T (K)"
        self._row_vars = ["P_eq (MPa)", "ΔH_diss (kJ/mol)"]
        self._col_vars = ["Theta_Small_CO2", "Theta_Large_CO2"]
        self._comp_mode = "eos"   # "eos" | "single"
        self._exp_overlay = True
        self._grid_rows = 2
        self._grid_cols = 2

    # ── public API ────────────────────────────────────────────────────────────

    def plot_grid(
        self,
        row_vars: list[str],
        col_vars: list[str],
        x_axis: str = "T (K)",
        comparison: str = "eos",
        exp_overlay: bool = True,
        title: str = "",
        figsize: tuple = None,
    ) -> plt.Figure:
        """
        Create a (len(row_vars) × len(col_vars)) subplot grid.

        Parameters
        ----------
        row_vars   : Y-axis variables, one per row.
        col_vars   : Y-axis variables, one per column.
                     Each cell (r, c) plots row_vars[r] AND col_vars[c] together
                     on the same axes (left/right y-axis if units differ).
                     If you only want a 1-D sweep pass a single-element list for
                     the dimension you want fixed.
        x_axis     : Column to use as the shared X axis.
        comparison : "eos"    → overlay all EOS models on every cell
                     "single" → use only the first model
        exp_overlay: Overlay experimental data when the Y variable is "P_eq (MPa)".
        title      : Optional suptitle.
        figsize    : Override figure size.
        """
        nrows = len(row_vars)
        ncols = len(col_vars)

        auto_w = max(4.5 * ncols, 8)
        auto_h = max(3.5 * nrows, 5)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize or (auto_w, auto_h),
            squeeze=False,
        )

        for r, yvar_row in enumerate(row_vars):
            for c, yvar_col in enumerate(col_vars):
                ax = axes[r][c]
                yvar = yvar_row if yvar_row == yvar_col else yvar_row  # primary
                # If row and col variables are different, plot both
                yvars_to_plot = list(dict.fromkeys([yvar_row, yvar_col]))

                self._draw_cell(
                    ax=ax,
                    x_var=x_axis,
                    y_vars=yvars_to_plot,
                    comparison=comparison,
                    exp_overlay=exp_overlay,
                )

                if r == nrows - 1:
                    ax.set_xlabel(x_axis, fontsize=9)
                if c == 0:
                    ax.set_ylabel(yvar_row, fontsize=9)
                ax.set_title(yvar_col if yvar_row != yvar_col else yvar_row, fontsize=9)

        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

        fig.tight_layout()
        return fig

    def show(self):
        """Launch the interactive GUI for selecting what to plot."""
        self._launch_gui()

    # ── drawing helper ────────────────────────────────────────────────────────

    def _draw_cell(
        self,
        ax: plt.Axes,
        x_var: str,
        y_vars: list[str],
        comparison: str,
        exp_overlay: bool,
    ):
        """Draw one subplot cell."""
        models = self.labels if comparison == "eos" else [self.labels[0]]

        plotted_any = False
        twin_ax = None
        legend_handles = []

        for y_idx, yvar in enumerate(y_vars):
            current_ax = ax if y_idx == 0 else (twin_ax := ax.twinx())

            for m_idx, model in enumerate(models):
                df  = self.results[model]
                x_s = _safe_col(df, x_var)
                y_s = _safe_col(df, yvar)

                if x_s is None or y_s is None:
                    continue

                valid = df[[x_var, yvar]].dropna()
                if valid.empty:
                    continue

                color  = _COLORS[(m_idx + y_idx * len(models)) % len(_COLORS)]
                marker = _MARKERS[m_idx % len(_MARKERS)]
                ls     = _LSTYLES[y_idx % len(_LSTYLES)]

                label = f"{model}" if len(y_vars) == 1 else f"{model} [{yvar}]"
                line, = current_ax.plot(
                    valid[x_var], valid[yvar],
                    color=color, marker=marker, linestyle=ls,
                    linewidth=1.6, markersize=4, label=label,
                )
                legend_handles.append(line)
                plotted_any = True

            if y_idx > 0 and twin_ax is not None:
                twin_ax.set_ylabel(yvar, fontsize=8, color=_COLORS[y_idx % len(_COLORS)])
                twin_ax.tick_params(axis="y", labelcolor=_COLORS[y_idx % len(_COLORS)])

        # Experimental overlay (only when plotting pressure)
        if exp_overlay and self.exp and "P_eq (MPa)" in y_vars:
            scatter = ax.scatter(
                self.exp["T (K)"], self.exp["P_eq (MPa)"],
                color="black", marker="x", s=50, zorder=6, label="Experimental",
            )
            legend_handles.append(scatter)

        if not plotted_any:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)

        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=7, loc="best",
                      framealpha=0.7, handlelength=1.5)

        ax.grid(True, linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=8)

    # ── interactive GUI ───────────────────────────────────────────────────────

    def _launch_gui(self):
        """
        Build a control panel + live preview window.
        Layout:
          Left column  → controls (check-boxes, radio-buttons, text inputs)
          Right area   → live matplotlib preview
        """
        matplotlib.use("TkAgg") if matplotlib.get_backend() == "" else None

        self._gui_fig = plt.figure(figsize=(18, 10))
        self._gui_fig.patch.set_facecolor("#1e1e2e")

        # Split figure: controls on left (20%), preview on right (80%)
        gs_outer = gridspec.GridSpec(
            1, 2, figure=self._gui_fig,
            width_ratios=[1, 4], wspace=0.05,
        )

        # ── Control panel ──────────────────────────────────────────────────
        ctrl_ax = self._gui_fig.add_subplot(gs_outer[0])
        ctrl_ax.set_facecolor("#2a2a3e")
        ctrl_ax.set_xticks([])
        ctrl_ax.set_yticks([])
        for spine in ctrl_ax.spines.values():
            spine.set_edgecolor("#555")

        # Preview area (will be replaced on each update)
        self._preview_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_outer[1], hspace=0.45, wspace=0.4,
        )
        self._preview_axes: list[plt.Axes] = []

        # ── Widget positions (axes units of ctrl_ax: 0→1) ─────────────────
        panel_pos = ctrl_ax.get_position()
        fig_w, fig_h = self._gui_fig.get_size_inches()

        def make_widget_ax(y_frac, height_frac=0.04, x_frac=0.01, w_frac=0.18):
            """Create an axes in figure coordinates within the control panel."""
            return self._gui_fig.add_axes([x_frac, y_frac, w_frac, height_frac])

        # Title label
        ctrl_ax.text(
            0.5, 0.97, "⚙  Plot Builder",
            transform=ctrl_ax.transAxes,
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="#cdd6f4",
        )

        # ── X-axis selector (radio) ────────────────────────────────────────
        ctrl_ax.text(0.05, 0.91, "X axis", transform=ctrl_ax.transAxes,
                     fontsize=9, color="#a6e3a1", fontweight="bold")
        x_opts = [c for c in ["T (K)", "P_eq (MPa)"] if c in self.all_cols][:4]
        ax_x_radio = make_widget_ax(0.78, height_frac=0.025 * len(x_opts), w_frac=0.17)
        self._x_radio = RadioButtons(ax_x_radio, x_opts, active=0,
                                     activecolor="#89b4fa")
        ax_x_radio.set_facecolor("#2a2a3e")
        self._x_radio.on_clicked(self._on_x_change)

        # ── Y-variable checklist (rows) ────────────────────────────────────
        ctrl_ax.text(0.05, 0.75, "Row variables (Y)", transform=ctrl_ax.transAxes,
                     fontsize=9, color="#a6e3a1", fontweight="bold")
        y_opts_row = self.all_cols[:12]  # Show up to 12 options
        n_row = len(y_opts_row)
        actives_row = [c in self._row_vars for c in y_opts_row]
        ax_row_check = make_widget_ax(0.58, height_frac=0.018 * n_row, w_frac=0.17)
        self._row_check = CheckButtons(ax_row_check, y_opts_row, actives_row)
        ax_row_check.set_facecolor("#2a2a3e")
        self._row_check.on_clicked(self._on_row_change)

        # ── Y-variable checklist (cols) ────────────────────────────────────
        ctrl_ax.text(0.05, 0.54, "Col variables (Y)", transform=ctrl_ax.transAxes,
                     fontsize=9, color="#a6e3a1", fontweight="bold")
        y_opts_col = self.all_cols[:12]
        n_col = len(y_opts_col)
        actives_col = [c in self._col_vars for c in y_opts_col]
        ax_col_check = make_widget_ax(0.37, height_frac=0.018 * n_col, w_frac=0.17)
        self._col_check = CheckButtons(ax_col_check, y_opts_col, actives_col)
        ax_col_check.set_facecolor("#2a2a3e")
        self._col_check.on_clicked(self._on_col_change)

        # ── Comparison mode (radio) ────────────────────────────────────────
        ctrl_ax.text(0.05, 0.33, "Comparison mode", transform=ctrl_ax.transAxes,
                     fontsize=9, color="#a6e3a1", fontweight="bold")
        ax_comp = make_widget_ax(0.27, height_frac=0.055, w_frac=0.17)
        self._comp_radio = RadioButtons(ax_comp, ["eos", "single"], active=0,
                                        activecolor="#89b4fa")
        ax_comp.set_facecolor("#2a2a3e")
        self._comp_radio.on_clicked(self._on_comp_change)

        # ── Experimental overlay toggle ────────────────────────────────────
        ctrl_ax.text(0.05, 0.25, "Experimental overlay", transform=ctrl_ax.transAxes,
                     fontsize=9, color="#a6e3a1", fontweight="bold")
        ax_exp = make_widget_ax(0.21, height_frac=0.03, w_frac=0.17)
        self._exp_check = CheckButtons(ax_exp, ["Show experimental"], [self._exp_overlay])
        ax_exp.set_facecolor("#2a2a3e")
        self._exp_check.on_clicked(self._on_exp_change)

        # ── Plot button ────────────────────────────────────────────────────
        ax_btn = make_widget_ax(0.14, height_frac=0.04, w_frac=0.17)
        self._plot_btn = Button(ax_btn, "▶  Update Plot",
                                color="#313244", hovercolor="#45475a")
        self._plot_btn.label.set_color("#cdd6f4")
        self._plot_btn.on_clicked(self._on_plot_click)

        # ── Export button ──────────────────────────────────────────────────
        ax_export = make_widget_ax(0.08, height_frac=0.04, w_frac=0.17)
        self._export_btn = Button(ax_export, "💾  Save PNG",
                                  color="#313244", hovercolor="#45475a")
        self._export_btn.label.set_color("#cdd6f4")
        self._export_btn.on_clicked(self._on_export_click)

        # ── Initial preview ────────────────────────────────────────────────
        self._rebuild_preview()
        plt.show()

    # ── widget callbacks ──────────────────────────────────────────────────────

    def _on_x_change(self, label):
        self._x_var = label

    def _on_row_change(self, label):
        if label in self._row_vars:
            self._row_vars.remove(label)
        else:
            self._row_vars.append(label)

    def _on_col_change(self, label):
        if label in self._col_vars:
            self._col_vars.remove(label)
        else:
            self._col_vars.append(label)

    def _on_comp_change(self, label):
        self._comp_mode = label

    def _on_exp_change(self, label):
        self._exp_overlay = not self._exp_overlay

    def _on_plot_click(self, event):
        self._rebuild_preview()
        self._gui_fig.canvas.draw_idle()

    def _on_export_click(self, event):
        fname = "hydrate_plot_export.png"
        self._gui_fig.savefig(fname, dpi=150, bbox_inches="tight",
                              facecolor=self._gui_fig.get_facecolor())
        print(f"[GeneralPlotter] Saved → {fname}")

    # ── preview rebuild ───────────────────────────────────────────────────────

    def _rebuild_preview(self):
        """Clear and redraw all preview subplots."""
        for ax in self._preview_axes:
            ax.remove()
        self._preview_axes.clear()

        row_vars = self._row_vars or ["P_eq (MPa)"]
        col_vars = self._col_vars or ["T (K)"]
        nrows, ncols = len(row_vars), len(col_vars)

        gs = gridspec.GridSpecFromSubplotSpec(
            nrows, ncols,
            subplot_spec=self._preview_gs[:, :],
            hspace=0.55, wspace=0.45,
        )

        for r, yvar_row in enumerate(row_vars):
            for c, yvar_col in enumerate(col_vars):
                ax = self._gui_fig.add_subplot(gs[r, c])
                ax.set_facecolor("#181825")
                ax.tick_params(colors="#cdd6f4", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#555")

                yvars = list(dict.fromkeys([yvar_row, yvar_col]))
                self._draw_cell(
                    ax=ax,
                    x_var=self._x_var,
                    y_vars=yvars,
                    comparison=self._comp_mode,
                    exp_overlay=self._exp_overlay,
                )

                cell_title = yvar_col if yvar_row != yvar_col else yvar_row
                ax.set_title(cell_title, fontsize=7, color="#cdd6f4", pad=3)
                if r == nrows - 1:
                    ax.set_xlabel(self._x_var, fontsize=7, color="#a6adc8")
                if c == 0:
                    ax.set_ylabel(yvar_row, fontsize=7, color="#a6adc8")

                # Style legend
                leg = ax.get_legend()
                if leg:
                    leg.get_frame().set_facecolor("#313244")
                    for text in leg.get_texts():
                        text.set_color("#cdd6f4")
                        text.set_fontsize(6)

                self._preview_axes.append(ax)


# ── standalone demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate synthetic data for a quick demo
    T = np.linspace(273, 283, 20)
    rng = np.random.default_rng(42)

    def fake_df(offset=0):
        return pd.DataFrame({
            "T (K)":             T,
            "P_eq (MPa)":        2.0 * np.exp(0.05 * (T - 273)) + offset + rng.normal(0, 0.1, len(T)),
            "ΔH_diss (kJ/mol)":  -58.0 + offset * 0.5 + rng.normal(0, 0.5, len(T)),
            "ΔS_diss (J/mol.K)": -210.0 + offset + rng.normal(0, 1, len(T)),
            "Theta_Small_CO2":   0.85 - offset * 0.02 + rng.normal(0, 0.01, len(T)),
            "Theta_Large_CO2":   0.92 - offset * 0.02 + rng.normal(0, 0.01, len(T)),
            "Z_gas":             0.85 + offset * 0.01 + rng.normal(0, 0.005, len(T)),
        })

    demo_results = {
        "Peng-Robinson":       fake_df(0),
        "Soave-Redlich-Kwong": fake_df(0.3),
        "Patel-Teja":          fake_df(-0.2),
    }

    demo_exp = {
        "T (K)":      [273.15, 275.15, 277.15, 279.15, 281.15],
        "P_eq (MPa)": [2.1,    2.8,    3.8,    5.1,    6.9],
    }

    plotter = GeneralPlotter(demo_results, demo_exp)
    plotter.show()