"""
Test headless p_eq evaluation and parameter check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hydrate_project.core.database import Database
from hydrate_project.thermo_model.klauda_sandler import KlaudaSandlerModel
from hydrate_project.eos_model.pr_eos import PREOS
from hydrate_project.eos_model.srk_eos import SRKEOS
from hydrate_project.eos_model.pt_eos import PTEOS
from hydrate_project.solvers.equilibrium import EquilibriumSolver

EOS_MAP = {
    "Peng-Robinson": PREOS,
    "Soave-Redlich-Kwong": SRKEOS,
    "Patel-Teja": PTEOS,
}

def p_eq(
    T: float | list[float] | np.ndarray,
    y_CO2: float,
    x_diox: float = 0.0556,
    eos_name: str = "Peng-Robinson",
    P_guess_MPa: float = 3.0,
) -> dict | pd.DataFrame:
    """Calculate equilibrium hydrate pressure for CO2/H2 gas + H2O/1,4-dioxane liquid.
    
    Parameters
    ----------
    T : float, list of floats, or ndarray
        Temperature in Kelvin.
    y_CO2 : float
        Mole fraction of CO2 in feed gas (balance is H2: 1 - y_CO2).
    x_diox : float, default 0.0556
        Mole fraction of 1,4-dioxane in liquid phase (balance is H2O: 1 - x_diox).
    eos_name : str, default 'Peng-Robinson'
        EOS to use: 'Peng-Robinson', 'Soave-Redlich-Kwong', or 'Patel-Teja'.
    P_guess_MPa : float, default 3.0
        Initial pressure guess in MPa.

    Returns
    -------
    dict (if scalar T) or pd.DataFrame (if array-like T) with equilibrium properties.
    """
    is_scalar = isinstance(T, (int, float))
    T_arr = np.array([T]) if is_scalar else np.array(T)

    gas_comp = {"CO2": float(y_CO2), "H2": float(1.0 - y_CO2)}
    liq_comp = {"H2O": float(1.0 - x_diox)}
    if x_diox > 1e-6:
        liq_comp["DIOX"] = float(x_diox)

    db = Database()
    km = KlaudaSandlerModel(database=db)
    eos_cls = EOS_MAP[eos_name]
    eos_inst = eos_cls(composition=gas_comp, database=db)
    solver = EquilibriumSolver(
        liq_phase_composition=liq_comp,
        database=db,
        hydrate_model=km,
        eos_model=eos_inst,
    )

    df = solver.find_optimum_structure(
        T_range=T_arr,
        P_initial_guess=P_guess_MPa * 1e6,
        solver_method="brentq",
    )

    if is_scalar:
        row = df.iloc[0].to_dict()
        return row
    return df

if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    res = p_eq(276.15, y_CO2=0.4, x_diox=0.0556)
    dt = time.perf_counter() - t0
    print(f"Time: {dt:.3f} s")
    print("Result for T=276.15 K, y_CO2=0.4, x_diox=0.0556:")
    for k in ["T (K)", "Optimum_Structure", "P_eq (MPa)", "z_Hyd_CO2", "z_Hyd_H2", "z_Hyd_DIOX"]:
        print(f"  {k}: {res.get(k)}")
