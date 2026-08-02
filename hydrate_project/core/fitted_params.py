"""
The fenced parameter layer (KS_MODEL_SPEC.md §6).

Klauda & Sandler never parameterised H2 or 1,4-dioxane, so a handful of
numbers in this model are not literature [V]/[P] constants -- they are
either genuinely regressed here, a one-off literature citation for a guest
K&S doesn't cover, or (in one case) an un-cross-validated placeholder. All
of them live in exactly this module, each with a mandatory provenance
record, so a reader can tell at a glance which numbers came from K&S and
which did not.

Rules (enforced by tests/test_integrity.py):
  1. core/database.py, core/correlations.py, thermo_model/ contain only
     [V]/[P]/[S] literature constants -- no FittedParam import there.
     Consumers (solvers/equilibrium.py, eos_model/*.py,
     water_activity_model/mod_unifac.py, thermo_model/klauda_sandler.py)
     import FITTED_PARAMS directly instead.
  2. Every entry declares fitted_to, n_train, n_holdout, bounds, date.
  3. aard_train/aard_test are Optional: several of these numbers were never
     actually regressed against a held-out set (some are literature
     citations, one is an ad hoc scalar tune, one is an undocumented
     placeholder). Where that is true, aard_test is None and fitted_to
     says so explicitly -- inventing a held-out score would be worse than
     admitting there isn't one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class FittedParam:
    value: Any
    symbol: str
    units: str
    fitted_to: str
    n_train: int
    n_holdout: int
    aard_train: Optional[float]
    aard_test: Optional[float]
    bounds: Optional[tuple]
    date: str

    def __post_init__(self):
        if not self.fitted_to.strip():
            raise ValueError(f"{self.symbol}: fitted_to must not be empty")
        if self.bounds is not None and isinstance(self.value, (int, float)):
            lo, hi = self.bounds
            if not (lo < self.value < hi):
                raise ValueError(
                    f"{self.symbol}={self.value} is outside its declared "
                    f"bounds {self.bounds} -- that is a bug report, not a result."
                )


FITTED_PARAMS: dict[str, FittedParam] = {
    "v_inf_CO2": FittedParam(
        value=32.0e-6,
        symbol="v_inf_CO2",
        units="m^3/mol",
        fitted_to=(
            "Klauda & Sandler 2000, infinite-dilution partial molar volume "
            "of CO2 in water used in their Poynting correction (eq. 19). "
            "Literature citation, not independently re-regressed or "
            "cross-validated against a held-out set in this codebase."
        ),
        n_train=0,
        n_holdout=0,
        aard_train=None,
        aard_test=None,
        bounds=(10e-6, 60e-6),
        date="2026-08-02",
    ),
    "v_inf_H2": FittedParam(
        value=26.1e-6,
        symbol="v_inf_H2",
        units="m^3/mol",
        fitted_to=(
            "Brelvi & O'Connell (AIChE J, 1972) scaled-particle-theory "
            "estimate for H2 at infinite dilution in water, 273-300 K. "
            "K&S 2000 does not cover H2. Literature citation, not "
            "independently re-regressed or cross-validated against a "
            "held-out set in this codebase. (An earlier value of 15 mL/mol "
            "in this codebase understated the Poynting correction by ~40%; "
            "this is the corrected literature value, not a curve-fit.)"
        ),
        n_train=0,
        n_holdout=0,
        aard_train=None,
        aard_test=None,
        bounds=(10e-6, 60e-6),
        date="2026-08-02",
    ),
    "k_ij_CO2_H2": FittedParam(
        value=0.162,
        symbol="k_ij_CO2_H2",
        units="dimensionless",
        fitted_to=(
            "Single scalar hand-tuned (not a least-squares fit against a "
            "held-out split) to bring CO2/H2 mixture hydrate pressure "
            "predictions closer to Kumar et al. 2006 data. Whole-pipeline "
            "AAD with this value, from validation_report.md: 39.2/60.8 "
            "mol% CO2/H2 mix -- 6.53%/1.67%/1.88% (PR/SRK/PT); 57.9/42.1 "
            "mol% mix -- 90.95%/59.29%/69.83% (PR/SRK/PT). The second "
            "dataset's large AAD shows this single scalar does not "
            "generalise across composition -- treat it as a rough "
            "correction, not a validated binary interaction parameter."
        ),
        n_train=0,
        n_holdout=0,
        aard_train=None,
        aard_test=None,
        bounds=(-0.5, 0.5),
        date="2026-08-02",
    ),
    "henry_H2": FittedParam(
        value={"H1": -86.8550, "H2": 4178.717, "H3": 10.4935, "H4": 0.00632},
        symbol="henry_H2",
        units="H1: dimensionless, H2: K, H3: dimensionless, H4: 1/K (K&S eq. 10 form)",
        fitted_to=(
            "4-parameter least-squares fit to Battino (IUPAC Solubility "
            "Data Series, Vol. 5/6, 1981/1984) H2-in-water mole-fraction "
            "solubility data: T=273.15/278.15/283.15/298.15 K -> "
            "kH=5.706e9/6.224e9/6.706e9/7.237e9 Pa. All 4 points were used "
            "to fit; none were held out, so aard_test is None rather than "
            "a fabricated number -- do not read this as cross-validated."
        ),
        n_train=4,
        n_holdout=0,
        aard_train=2.02,  # recomputed directly from the 4 points above
        aard_test=None,
        bounds=None,
        date="2026-08-02",
    ),
    "henry_DIOX": FittedParam(
        value={"H1": -200.0, "H2": 10000.0, "H3": 0.0, "H4": 0.0},
        symbol="henry_DIOX",
        units="H1: dimensionless, H2: K, H3: dimensionless, H4: 1/K (K&S eq. 10 form)",
        fitted_to=(
            "UNVALIDATED PLACEHOLDER. Not fitted to any literature "
            "solubility data for 1,4-dioxane in water -- do not trust "
            "quantitatively. Kept only so the solver has a finite value to "
            "iterate against; ModifiedUnifac.calc_henry_constant warns "
            "whenever this entry is used."
        ),
        n_train=0,
        n_holdout=0,
        aard_train=None,
        aard_test=None,
        bounds=None,
        date="2026-08-02",
    ),
    "dioxane_kihara": FittedParam(
        value={"sigma": 3.38, "eps_k": 840.7, "a": 0.85},
        symbol="dioxane_kihara",
        units="sigma: Angstrom, eps_k: K, a: Angstrom",
        fitted_to=(
            "Provenance not documented anywhere in the codebase history "
            "prior to this fencing pass -- these values predate this "
            "session and their source (regression vs. literature table) "
            "could not be reconstructed. Flagged here rather than silently "
            "carried forward as if verified; needs sourcing from a Kihara "
            "parameter table for 1,4-dioxane or an honest re-fit with a "
            "held-out set before being trusted quantitatively."
        ),
        n_train=0,
        n_holdout=0,
        aard_train=None,
        aard_test=None,
        bounds=None,
        date="2026-08-02",
    ),
}


def get(symbol: str) -> FittedParam:
    return FITTED_PARAMS[symbol]
