"""
general_plotter
===============
Interactive grid-plot builder for hydrate project results.

Quick start
-----------
    from hydrate_project.utils.general_plotter import GeneralPlotter

    plotter = GeneralPlotter(all_results, experimental_data)
    plotter.show()              # launch GUI

    # Or build a static figure programmatically (no GUI):
    fig = plotter.plot_grid(
        row_vars   = ["P_eq (MPa)", "ΔH_diss (kJ/mol)"],
        col_vars   = ["Theta_Small_CO2", "Theta_Large_CO2"],
        x_axis     = "T (K)",
        comparison = "eos",
        exp_overlay= True,
    )
    fig.savefig("output.png", dpi=150, bbox_inches="tight")

Module layout
-------------
    core.py    — pure matplotlib drawing logic (no tkinter)
    theme.py   — dark ttk.Style + colour constants
    canvas.py  — FigureCanvasTkAgg wrapper
    sidebar.py — control panel widgets
    app.py     — main Tk window
"""
from __future__ import annotations

from typing import Optional
import matplotlib.pyplot as plt

from .core import build_grid, get_numeric_columns


class GeneralPlotter:
    """
    Public façade.  Wraps the GUI app and the headless plot builder.

    Parameters
    ----------
    results_dict : dict[str, pd.DataFrame]
        Keys are model names; values are result DataFrames from the solver.
    experimental_data : dict | None
        {"T (K)": [...], "P_eq (MPa)": [...]}  or None.
    """

    def __init__(self, results_dict: dict, experimental_data: Optional[dict] = None):
        self._results = results_dict
        self._exp     = experimental_data

    # ── GUI ────────────────────────────────────────────────────────────────

    def show(self, title: str = "Hydrate — General Plot Builder"):
        """Launch the interactive tkinter GUI (blocks until window closes)."""
        from .app import PlotBuilderApp
        app = PlotBuilderApp(self._results, self._exp, title=title)
        app.mainloop()

    # ── headless / scripting ──────────────────────────────────────────────

    def plot_grid(
        self,
        row_vars: list[str],
        col_vars: list[str],
        x_axis: str = "T (K)",
        comparison: str = "eos",
        exp_overlay: bool = True,
        dark: bool = True,
        figsize: tuple[float, float] | None = None,
        title: str = "",
    ) -> plt.Figure:
        """
        Build and return a matplotlib Figure without opening a GUI window.

        Parameters
        ----------
        row_vars   : Y variables for each row of the grid.
        col_vars   : Y variables for each column (twin-axis overlay per cell).
        x_axis     : Column used as the shared X axis.
        comparison : "eos" → overlay all models; "single" → first model only.
        exp_overlay: Overlay experimental scatter when y == "P_eq (MPa)".
        dark       : Use the dark Catppuccin theme.
        figsize    : Override (width, height) in inches.
        title      : Optional figure suptitle.
        """
        nrows = max(len(row_vars), 1)
        ncols = max(len(col_vars), 1)
        auto_w = max(4.5 * ncols, 8)
        auto_h = max(3.5 * nrows, 5)
        fig = plt.figure(figsize=figsize or (auto_w, auto_h))

        build_grid(
            fig          = fig,
            results_dict = self._results,
            row_vars     = row_vars,
            col_vars     = col_vars,
            x_var        = x_axis,
            comparison   = comparison,
            exp_data     = self._exp,
            exp_overlay  = exp_overlay,
            dark         = dark,
        )
        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01,
                         color="white" if dark else "black")
        fig.tight_layout()
        return fig

    # ── convenience ───────────────────────────────────────────────────────

    @property
    def available_columns(self) -> list[str]:
        """All numeric columns available across all result DataFrames."""
        return get_numeric_columns(self._results)