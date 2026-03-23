"""
main.py
-------
Entry point for the Hydrate Equilibrium Thermodynamic Model.

Launches the LauncherApp GUI where the user can:
  - Set gas composition and temperature scan range
  - Choose EOS models (PR, SRK, PT)
  - Optionally enter experimental data for AAD calculation
  - Run the solver and inspect results in the Plot Builder

Usage:
    uv run python -m hydrate_project.main
    # or
    uv run model
"""

import warnings
warnings.filterwarnings("ignore")

from hydrate_project.utils.launcher_app import LauncherApp


def main():
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()