"""
solver_runner.py
-----------------
Backend orchestration: wires together the database, hydrate model, EOS model
and equilibrium solver for a given input configuration.  No UI imports here —
this module must remain importable and runnable headlessly (used by the UI
worker thread and by the accuracy validation harness).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from hydrate_project.core.database import Database
from hydrate_project.thermo_model.klauda_sandler import KlaudaSandlerModel
from hydrate_project.thermo_model.klauda_sandler_empirical import (
    KlaudaSandlerEmpiricalModel,
)
from hydrate_project.eos_model.pr_eos import PREOS
from hydrate_project.eos_model.srk_eos import SRKEOS
from hydrate_project.eos_model.pt_eos import PTEOS
from hydrate_project.solvers.equilibrium import EquilibriumSolver

log = logging.getLogger(__name__)

EOS_MAP = {
    "Peng-Robinson": PREOS,
    "Soave-Redlich-Kwong": SRKEOS,
    "Patel-Teja": PTEOS,
}


def run_model(
    gas_comp: dict,
    liq_comp: dict,
    T_range: np.ndarray,
    eos_names: list[str],
    status_cb: callable,
) -> tuple[dict, Optional[Exception]]:
    """Run the equilibrium solver for every selected EOS.

    Returns (results_dict, error). results_dict maps eos_name -> DataFrame.
    On failure, results_dict is {} and error holds the raised exception.
    """
    try:
        db = Database()

        active_gases = {g: f for g, f in gas_comp.items() if f > 1e-6}
        has_promoter = any(k != "H2O" for k, v in liq_comp.items() if v > 1e-6)

        if has_promoter:
            # Promoters require Kihara integration to calculate cavity occupancy
            hydrate_core = KlaudaSandlerModel(database=db)
            log.info("Detected promoter -> using standard KlaudaSandlerModel")
        elif len(active_gases) == 1 and "CO2" in active_gases:
            # Pure CO2 uses the standard Kihara model
            hydrate_core = KlaudaSandlerModel(database=db)
            log.info("Detected pure CO2 -> using standard KlaudaSandlerModel")
        else:
            # Gas mixtures (without promoters) use the empirical correlations
            hydrate_core = KlaudaSandlerEmpiricalModel(database=db)
            log.info("Detected gas mixture -> using KlaudaSandlerEmpiricalModel")

        results: dict[str, pd.DataFrame] = {}

        for eos_name in eos_names:
            cls = EOS_MAP[eos_name]
            eos_inst = cls(composition=gas_comp, database=db)
            solver = EquilibriumSolver(
                liq_phase_composition=liq_comp,
                database=db,
                hydrate_model=hydrate_core,
                eos_model=eos_inst,
            )
            status_cb(f"[{eos_name}]  scanning {len(T_range)} temperature points…")
            df = solver.find_optimum_structure(
                T_range=T_range, P_initial_guess=2.5e6, solver_method="brentq"
            )
            results[eos_name] = df
            status_cb(f"[{eos_name}]  ✓ done")

        return results, None

    except Exception as exc:
        return {}, exc
