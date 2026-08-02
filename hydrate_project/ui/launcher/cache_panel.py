"""
cache_panel.py
---------------
Result cache info panel: shows SQLite FIFO cache stats with a clear button.
"""

from __future__ import annotations

from tkinter import ttk, messagebox

from hydrate_project.services.cache import get_cache


class CachePanel(ttk.LabelFrame):
    """Result Cache section."""

    def __init__(self, parent, on_cleared=None, **kwargs):
        super().__init__(
            parent,
            text="  Result Cache  (SQLite · FIFO eviction)  ",
            style="TLabelframe",
            padding=(10, 8),
            **kwargs,
        )
        self._on_cleared = on_cleared

        top = ttk.Frame(self, style="TFrame")
        top.pack(fill="x")
        self._label = ttk.Label(top, text="Checking cache…", style="Muted.TLabel")
        self._label.pack(side="left")
        ttk.Button(
            top, text="🗑  Clear cache", style="TButton", command=self._on_clear_cache
        ).pack(side="right")
        self.refresh()

    def refresh(self):
        try:
            info = get_cache().info()
            self._label.configure(
                text=f"{info.total_entries} / {info.max_entries} entries  •  "
                f"session: {info.hits_session} hits, "
                f"{info.misses_session} misses  •  "
                f"DB: {info.db_path}"
            )
        except Exception as exc:
            self._label.configure(text=f"Cache unavailable: {exc}")

    def _on_clear_cache(self):
        if messagebox.askyesno(
            "Clear cache", "Delete all cached results?\nThis cannot be undone."
        ):
            get_cache().clear()
            self.refresh()
            if self._on_cleared:
                self._on_cleared()
