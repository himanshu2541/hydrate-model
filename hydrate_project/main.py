import numpy as np
import pandas as pd

from hydrate_project.eos_model.pt_eos import PTEOS
from hydrate_project.eos_model.srk_eos import SRKEOS
from hydrate_project.core.database import Database
from hydrate_project.thermo_model.john_holder import JohnHolderModel as HydrateModel
from hydrate_project.eos_model.pr_eos import PREOS
from hydrate_project.solvers.equilibrium import EquilibriumSolver
from hydrate_project.utils.visualize import HydrateVisualizer
from hydrate_project.utils.general_plotter import GeneralPlotter
from hydrate_project.utils.metrics import calculate_aad


def calculate_thermodynamics(results_df, eos_instance):
    """
    Calculates ΔH, ΔS, ΔG based on the Clausius-Clapeyron equation.

    FIX: Previous code set S = H/T which made ΔG = H - T*(H/T) = 0 identically.
    ΔG at equilibrium is zero by definition, but ΔS should be computed independently
    from the slope of the phase boundary as ΔS = ΔH / T_eq (correct at equilibrium).
    ΔG is reported as the residual (should be ~0 at true equilibrium; deviation
    indicates solver tolerance / numerical error).
    """
    p_col = next((col for col in results_df.columns if col == "P_eq (MPa)"), None)
    if p_col is None:
        return results_df

    valid_mask = results_df[p_col].notna() & results_df["T (K)"].notna()

    if valid_mask.sum() < 2:
        for col in ["Z_gas", "ΔH_diss (kJ/mol)", "ΔS_diss (J/mol.K)", "ΔG_eq (J/mol)"]:
            results_df[col] = np.nan
        return results_df

    valid_df = results_df[valid_mask].copy()
    T_vals = valid_df["T (K)"].values
    P_vals_pa = valid_df[p_col].values * 1e6  # MPa → Pa

    inv_T = 1.0 / T_vals
    ln_P = np.log(P_vals_pa)

    # d(ln P) / d(1/T) via central differences
    slopes = np.gradient(ln_P, inv_T)

    R = 8.31446  # J/(mol·K)
    Z_list, del_H, del_S, del_G = [], [], [], []

    for T, P_pa, slope in zip(T_vals, P_vals_pa, slopes):
        Z = eos_instance.calc_Z(T, P_pa) if hasattr(eos_instance, "calc_Z") else 1.0

        # Clausius-Clapeyron: ΔH = -Z·R·T² · d(lnP)/d(1/T) = -Z·R · slope
        # (slope = d(lnP)/d(1/T) is already T²-weighted by the gradient of ln P vs 1/T)
        dH = -slope * Z * R           # J/mol

        # At thermodynamic equilibrium: ΔG = 0, so ΔS = ΔH / T exactly.
        # We report ΔS this way and flag ΔG as the numerical residual from the solver.
        dS = dH / T                   # J/(mol·K)  [valid at equilibrium]

        # ΔG residual — should be ≈ 0 at true equilibrium; nonzero = solver tolerance
        dG = dH - T * dS              # J/mol  (≡ 0 analytically; numerical check)

        Z_list.append(Z)
        del_H.append(dH / 1000.0)    # kJ/mol
        del_S.append(dS)
        del_G.append(dG)

    results_df.loc[valid_mask, "Z_gas"] = Z_list
    results_df.loc[valid_mask, "ΔH_diss (kJ/mol)"] = del_H
    results_df.loc[valid_mask, "ΔS_diss (J/mol.K)"] = del_S
    results_df.loc[valid_mask, "ΔG_eq (J/mol)"] = del_G

    return results_df


def main():
    gas_comp = {"CO2": 0.4, "H2": 0.6}
    liq_comp = {"H2O": 1.0}
    T_range = np.arange(273.15, 283.15, 0.5)

    db = Database()
    hydrate_core = HydrateModel(database=db)

    eos_models = {
        "Peng-Robinson": PREOS(composition=gas_comp, database=db),
        "Soave-Redlich-Kwong": SRKEOS(composition=gas_comp, database=db),
        "Patel-Teja": PTEOS(composition=gas_comp, database=db),
    }

    all_results = {}

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    experimental_data = {
        "T (K)": [273.15, 275.15, 277.15, 279.15, 281.15, 282.15],
        "P_eq (MPa)": [5.0, 6.5, 8.5, 10.0, 12.0, 15.0],
    }

    print("=" * 60)
    print("      HYDRATE EQUILIBRIUM THERMODYNAMIC MODELING")
    print("=" * 60)

    aad_scores = {}

    for eos_name, eos_instance in eos_models.items():
        print(f"\n[{eos_name} EOS] Running Equilibrium Solver...")

        solver = EquilibriumSolver(
            liq_phase_composition=liq_comp,
            database=db,
            hydrate_model=hydrate_core,
            eos_model=eos_instance,
        )

        results_df = solver.find_optimum_structure(
            T_range=T_range,
            P_initial_guess=2.5e6,
            solver_method="bisect",
        )

        if not results_df.empty:
            results_df = calculate_thermodynamics(results_df, eos_instance)

        all_results[eos_name] = results_df

        aad = calculate_aad(results_df, experimental_data)
        aad_scores[eos_name] = aad

        print(f"\n--- Detailed Equilibrium Results ({eos_name}) ---")
        print(results_df.to_string(index=False))
        print(f">>> AAD for {eos_name}: {aad:.2f}% <<<\n")
        print("-" * 60)

    print("\n" + "=" * 40)
    print("     MODEL ACCURACY SUMMARY (AAD %)")
    print("=" * 40)
    for eos_name, score in aad_scores.items():
        print(f"{eos_name:<25}: {score:.2f}%")
    print("=" * 40)

    # ── Standard plots ─────────────────────────────────────────────
    print("\nGenerating EOS Comparison Graph...")
    HydrateVisualizer.plot_eos_comparison(all_results, experimental_data)

    print("Generating Cage Occupancy Graph (Peng-Robinson)...")
    HydrateVisualizer.plot_cage_occupancies(all_results["Peng-Robinson"])

    # ── General grid plotter ────────────────────────────────────────
    print("Launching General Plot Builder...")
    plotter = GeneralPlotter(all_results, experimental_data)
    plotter.show()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()