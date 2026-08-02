"""
main.py
-------
Entry point for the Hydrate Equilibrium Thermodynamic Model.

Default UI is now the browser app (hydrate_project/web/): same gas/liquid/
temperature/EOS/experimental-data/cache/sweep controls and Plot Builder as
before, served over HTTP so it's testable with Playwright. The Tkinter
launcher (hydrate_project/ui/) is kept as a fallback -- `uv run model-tk`.

Usage:
    uv run model            # web UI at http://127.0.0.1:8765
    uv run model-tk          # legacy Tkinter UI
"""

import warnings
warnings.filterwarnings("ignore")


def main():
    """Launch the web UI (default)."""
    import uvicorn

    print("Hydrate Equilibrium Model -- open http://127.0.0.1:8765 in your browser")
    uvicorn.run("hydrate_project.web.api:app", host="127.0.0.1", port=8765)


def main_tk():
    """Launch the legacy Tkinter UI."""
    from hydrate_project.ui.launcher.app import LauncherApp

    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()