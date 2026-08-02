"""
literature_data.py
-------------------
Single source of truth for literature equilibrium datasets used both by the
UI (experimental-data presets) and by the accuracy validation harness
(tests/test_accuracy_regression.py), which needs the exact gas/liquid
composition each dataset was measured at to be able to re-run the solver
headlessly and compare against it.
"""

from __future__ import annotations

PRESET_DATA: dict[str, dict] = {
    "CO₂/H₂ (39.2/60.8 mol%) — Kumar et al. 2006  [sI, bulk water]": {
        "gas_comp": {"CO2": 0.392, "H2": 0.608},
        "liq_comp": {"H2O": 1.0},
        "T (K)": [273.9, 274.6, 275.1, 275.6, 276.0, 276.4, 276.7, 277.5, 277.7, 278.4],
        "P_eq (MPa)": [5.56, 6.04, 6.41, 6.84, 7.16, 7.56, 7.95, 9.15, 9.42, 10.74],
    },
    "CO₂/H₂ (57.9/42.1 mol%) — Kumar et al. 2006  [sI, bulk water]": {
        "gas_comp": {"CO2": 0.579, "H2": 0.421},
        "liq_comp": {"H2O": 1.0},
        "T (K)": [274.6, 277.8, 279.4, 280.7, 281.4],
        "P_eq (MPa)": [2.77, 4.61, 5.99, 7.41, 8.31],
    },
    "CO₂ (Pure) — Sloan & Koh 2008  [sI, bulk water]": {
        "gas_comp": {"CO2": 1.0},
        "liq_comp": {"H2O": 1.0},
        "T (K)": [
            273.15,
            274.15,
            275.15,
            276.15,
            277.15,
            278.15,
            279.15,
            280.15,
            281.15,
            282.15,
            283.15,
        ],
        "P_eq (MPa)": [1.25, 1.4, 1.5, 1.75, 2.0, 2.25, 2.6, 3.0, 3.4, 3.9, 4.5],
    },
}
