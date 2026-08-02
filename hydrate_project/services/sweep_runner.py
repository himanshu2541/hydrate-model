"""
sweep_runner.py
-----------------
Orchestrates a series of single-EOS solver runs across a swept parameter —
either DIOX (promoter) mol% or gas composition ratio — so the results can be
overlaid on one chart (e.g. separation factor vs T for several DIOX loadings).

Each sweep point goes through the same services.solver_runner.run_model /
services.cache path as a normal single run, so repeated sweeps over
previously-seen configurations are served from cache.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from hydrate_project.services.cache import get_cache
from hydrate_project.services.solver_runner import run_model


def _run_one(gas_comp, liq_comp, T_range, eos_name, status_cb):
    cache = get_cache()
    cached = cache.get(gas_comp, liq_comp, T_range, [eos_name])
    if cached is not None:
        return cached[eos_name], None
    results, err = run_model(gas_comp, liq_comp, T_range, [eos_name], status_cb)
    if err is not None:
        return None, err
    cache.put(gas_comp, liq_comp, T_range, [eos_name], results)
    return results[eos_name], None


def run_diox_sweep(
    gas_comp: dict,
    promoter_key: str,
    diox_fractions: list[float],
    T_range: np.ndarray,
    eos_name: str,
    status_cb: callable,
) -> tuple[dict[str, pd.DataFrame], Optional[Exception]]:
    """Sweep DIOX (or other promoter) mol% at a fixed gas composition.

    Returns ({"DIOX 5.56%": df, ...}, error).
    """
    results: dict[str, pd.DataFrame] = {}
    for frac in diox_fractions:
        liq_comp = {"H2O": round(1.0 - frac, 8), promoter_key: round(frac, 8)}
        label = f"{promoter_key} {frac * 100:.2f}%"
        status_cb(f"[{label}]  running…")
        df, err = _run_one(gas_comp, liq_comp, T_range, eos_name, status_cb)
        if err is not None:
            return {}, err
        results[label] = df
        status_cb(f"[{label}]  ✓ done")
    return results, None


def run_gas_ratio_sweep(
    gas_comp_list: list[dict],
    liq_comp: dict,
    T_range: np.ndarray,
    eos_name: str,
    status_cb: callable,
) -> tuple[dict[str, pd.DataFrame], Optional[Exception]]:
    """Sweep gas composition ratio at a fixed liquid-phase composition.

    Returns ({"CO2:H2 = 40:60", df, ...}, error).
    """
    results: dict[str, pd.DataFrame] = {}
    for gas_comp in gas_comp_list:
        label = " : ".join(f"{g} {f * 100:.1f}%" for g, f in gas_comp.items())
        status_cb(f"[{label}]  running…")
        df, err = _run_one(gas_comp, liq_comp, T_range, eos_name, status_cb)
        if err is not None:
            return {}, err
        results[label] = df
        status_cb(f"[{label}]  ✓ done")
    return results, None
