"""
derived.py
-----------
Display-only derived quantities layered on top of solver output -- not
physics-core, so this is not core/. Ported from the Tkinter
PropertiesWindow so the web Properties table gets the same columns without
re-deriving the formula in JS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

R = 8.314462618  # J/(mol*K)


def _compute_dH_diss(df: pd.DataFrame) -> pd.Series:
    """Clausius-Clapeyron estimate of dissociation enthalpy (kJ/mol):
    dH_diss = -R * d(ln P) / d(1/T), central difference with one-sided edges.
    """
    T = df["T (K)"].values.astype(float)
    P = df["P_eq (MPa)"].values.astype(float)

    inv_T = np.where(T > 0, 1.0 / T, np.nan)
    ln_P = np.where(P > 0, np.log(P), np.nan)

    dH = np.full(len(T), np.nan)
    for i in range(1, len(T) - 1):
        denom = inv_T[i + 1] - inv_T[i - 1]
        if denom != 0 and not np.isnan(ln_P[i - 1]) and not np.isnan(ln_P[i + 1]):
            dH[i] = -R * (ln_P[i + 1] - ln_P[i - 1]) / denom / 1000.0

    if len(T) >= 2:
        d0 = inv_T[1] - inv_T[0]
        if d0 != 0 and not np.isnan(ln_P[0]) and not np.isnan(ln_P[1]):
            dH[0] = -R * (ln_P[1] - ln_P[0]) / d0 / 1000.0
        dn = inv_T[-1] - inv_T[-2]
        if dn != 0 and not np.isnan(ln_P[-2]) and not np.isnan(ln_P[-1]):
            dH[-1] = -R * (ln_P[-1] - ln_P[-2]) / dn / 1000.0

    return pd.Series(dH, index=df.index, name="dH_diss (kJ/mol)")


def add_dissociation_thermo(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with dH_diss/dG_diss/dS_diss columns appended."""
    rdf = df.copy()
    try:
        dH_series = _compute_dH_diss(rdf)
        rdf["dH_diss (kJ/mol)"] = dH_series
        rdf["dG_diss (kJ/mol)"] = 0.0  # dG = 0 at equilibrium by definition
        rdf["dS_diss (kJ/mol.K)"] = dH_series / rdf["T (K)"]
    except Exception:
        pass
    return rdf
