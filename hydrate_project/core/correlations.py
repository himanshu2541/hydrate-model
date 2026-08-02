"""
Klauda & Sandler (2000/2003) pure correlation functions.

Single source of truth for the water/ice/empty-hydrate vapor-pressure and
molar-volume forms in KS_MODEL_SPEC.md §3. Every hydrate model
(thermo_model/klauda_sandler*.py) must import these instead of re-deriving
them, so the two model variants cannot silently drift apart.

Two transcription traps fixed here per spec §3.2/§3.3 — get them wrong and
every downstream number is off by orders of magnitude:
  - eqs. 7c/7d (P_sat_liquid_water, P_sat_ice): the OCR of the source PDF
    flips the sign of the constant and 1/T terms.
  - eqs. 8a/8b (V_empty_hydrate): the lattice-parameter bracket is cubed;
    the exponent is lost in the PDF text layer.

No class, no state — every function takes plain numeric/string arguments.
"""

from __future__ import annotations

import numpy as np

NA = 6.02214076e23  # 1/mol, CODATA
R = 8.314462618  # J/(mol*K), CODATA


def P_sat_liquid_water(T: float) -> float:
    """Vapor pressure of liquid water (Pa). K&S eq. 7c."""
    ln_p = 4.1539 * np.log(T) - 5500.9332 / T + 7.6537 - 16.1277e-3 * T
    return np.exp(ln_p)


def P_sat_ice(T: float) -> float:
    """Vapor pressure of ice (Pa). K&S eq. 7d."""
    ln_p = 4.6056 * np.log(T) - 5501.1243 / T + 2.9446 - 8.1431e-3 * T
    return np.exp(ln_p)


def V_empty_hydrate(T: float, P_MPa: float, structure: str) -> float:
    """Molar volume of the empty hydrate lattice (m^3/mol). K&S eqs. 8a/8b.

    P_MPa in MPa. The pressure-dependent quadratic coefficient
    (5.448e-12) is shared between sI and sII per BK2011 Table 1 --
    # VERIFY: BK2011 Table 1 before treating the sI value as independently
    # confirmed (see KS_MODEL_SPEC.md §3.3); the difference from the other
    # candidate coefficient (5.448e-13) is ~5e-10 m^3/mol at 10 MPa, i.e.
    # negligible for this system, but it has not been re-derived here.
    """
    if structure == "sI":
        Nw = 46.0
        a = 11.835 + 2.217e-5 * T + 2.242e-6 * T**2
    elif structure == "sII":
        Nw = 136.0
        a = 17.13 + 2.249e-4 * T + 2.013e-6 * T**2 - 1.009e-9 * T**3
    else:
        raise ValueError(f"Unknown structure: {structure!r}")

    V_lattice = (a**3) * 1e-30 * NA / Nw
    V_pressure = -8.006e-9 * P_MPa + 5.448e-12 * P_MPa**2
    return V_lattice + V_pressure


def V_liquid_water(T: float, P_MPa: float) -> float:
    """Molar volume of liquid water (m^3/mol). K&S eq. 8c. P_MPa in MPa."""
    ln_V = (
        -10.9241
        + 2.5e-4 * (T - 273.15)
        - 3.532e-4 * (P_MPa - 0.101325)
        + 1.559e-7 * (P_MPa - 0.101325) ** 2
    )
    return np.exp(ln_V)


def V_ice(T: float) -> float:
    """Molar volume of ice (m^3/mol). K&S eq. 8d."""
    return 1.912e-5 + 8.387e-10 * T + 4.016e-12 * T**2


def henry_constant(T: float, params: dict, P0: float = 101325.0) -> float:
    """Henry's-law constant (Pa) for a gas dissolved in water. K&S eq. 10.

    -ln(H/P0) = H1 + H2/T + H3*ln(T) + H4*T

    `params` is a {"H1", "H2", "H3", "H4"} dict; callers look it up from
    whichever table it lives in (database.py for literature [V] guests,
    core/fitted_params.py for guests K&S never covered) -- this function
    only implements the equation.
    """
    rhs = params["H1"] + params["H2"] / T + params["H3"] * np.log(T) + params["H4"] * T
    return P0 * np.exp(-rhs)
