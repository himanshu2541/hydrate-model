"""
validators.py
--------------
Pure validation/parsing functions for launcher inputs. These take already-read
primitives (not tkinter widgets) and raise ValueError with a user-facing
message on bad input. Kept tkinter-free so they're independently testable and
reusable from the sweep feature.
"""

from __future__ import annotations

import numpy as np


def validate_gas_composition(gas_comp: dict[str, float]) -> None:
    total = sum(gas_comp.values())
    for g, f in gas_comp.items():
        if f < 0:
            raise ValueError(f"Mole fraction for {g} cannot be negative.")
    if abs(total - 1.0) > 1e-4:
        raise ValueError(
            f"Gas mole fractions must sum to 1.0  (current = {total:.4f})."
        )


def parse_promoter_fraction(raw: str) -> float:
    try:
        xp = float(raw)
    except ValueError:
        raise ValueError("Promoter mole fraction must be a number.")
    if not (0.0 < xp < 1.0):
        raise ValueError("Promoter mole fraction must be between 0 and 1.")
    return xp


def build_T_range(tmin: float, tmax: float, tstep: float) -> np.ndarray:
    if tmin >= tmax:
        raise ValueError("T_min must be less than T_max.")
    if tstep <= 0:
        raise ValueError("T_step must be positive.")
    return np.arange(tmin, tmax + tstep / 2, tstep)


def parse_custom_exp_data(text: str) -> dict | None:
    """Parse 'T(K), P(MPa)' lines (comma or space separated; '#' comments)."""
    lines = text.strip().splitlines()
    Ts, Ps = [], []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.replace(",", " ").split()
        if len(parts) < 2:
            raise ValueError(f"Cannot parse: '{ln}'  (expected T P).")
        Ts.append(float(parts[0]))
        Ps.append(float(parts[1]))
    if Ts:
        return {"T (K)": Ts, "P_eq (MPa)": Ps}
    return None


def parse_gas_ratio_lines(text: str) -> list[dict[str, float]]:
    """Parse one gas composition per line, e.g. 'CO2=0.4, H2=0.6' (# comments allowed)."""
    compositions = []
    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        comp: dict[str, float] = {}
        for pair in ln.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Cannot parse '{pair}'  (expected GAS=fraction).")
            gas, frac_str = pair.split("=", 1)
            try:
                comp[gas.strip()] = float(frac_str.strip())
            except ValueError:
                raise ValueError(f"Invalid mole fraction in '{pair}'.")
        if comp:
            validate_gas_composition(comp)
            compositions.append(comp)
    if not compositions:
        raise ValueError("Enter at least one gas composition line (e.g. 'CO2=0.4, H2=0.6').")
    return compositions


def parse_diox_fraction_list(raw: str) -> list[float]:
    """Parse a comma-separated list of DIOX mol% values (e.g. '2, 5.56, 8') -> fractions."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Enter at least one DIOX % value (comma-separated).")
    fractions = []
    for p in parts:
        try:
            pct = float(p)
        except ValueError:
            raise ValueError(f"Cannot parse DIOX % value: '{p}'.")
        if not (0.0 < pct < 100.0):
            raise ValueError(f"DIOX % value out of range (0-100): '{p}'.")
        fractions.append(pct / 100.0)
    return fractions
