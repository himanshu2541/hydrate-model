"""
Generate P_eq grid across (T, y_CO2, x_diox) and fit analytical surrogate for Aspen Plus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scripts.test_headless import p_eq

def run_grid():
    T_vals = np.linspace(274.15, 282.15, 9)     # 1 C to 9 C
    y_vals = np.array([0.20, 0.30, 0.392, 0.50, 0.60]) # CO2 mole fraction
    x_vals = np.array([0.03, 0.045, 0.0556])    # Dioxane mole fraction

    records = []
    print(f"Generating grid: {len(T_vals)} x {len(y_vals)} x {len(x_vals)} = {len(T_vals)*len(y_vals)*len(x_vals)} points...")

    for T in T_vals:
        for y in y_vals:
            for x in x_vals:
                res = p_eq(T, y_CO2=y, x_diox=x, eos_name="Peng-Robinson")
                p_val = res.get("P_eq (MPa)")
                struct = res.get("Optimum_Structure")
                z_co2 = res.get("z_Hyd_CO2", np.nan)
                z_h2 = res.get("z_Hyd_H2", np.nan)
                records.append({
                    "T_K": T,
                    "y_CO2": y,
                    "x_diox": x,
                    "Structure": struct,
                    "P_eq_MPa": p_val,
                    "z_Hyd_CO2": z_co2,
                    "z_Hyd_H2": z_h2,
                })

    df = pd.DataFrame(records)
    df.to_csv("scripts/surrogate_grid_data.csv", index=False)
    print(f"Saved {len(df)} points to scripts/surrogate_grid_data.csv")

    # Filter valid sII points
    valid = df[df["P_eq_MPa"].notna() & (df["P_eq_MPa"] > 0)].copy()
    valid["ln_P"] = np.log(valid["P_eq_MPa"])
    valid["inv_T"] = 1.0 / valid["T_K"]
    valid["y_over_T"] = valid["y_CO2"] / valid["T_K"]
    valid["x_over_T"] = valid["x_diox"] / valid["T_K"]

    # Design matrix X: [1, 1/T, y, x, y/T, x/T]
    X = np.column_stack([
        np.ones(len(valid)),
        valid["inv_T"],
        valid["y_CO2"],
        valid["x_diox"],
        valid["y_over_T"],
        valid["x_over_T"],
    ])
    y_vec = valid["ln_P"].values

    # Least squares fit
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y_vec, rcond=None)
    
    valid["ln_P_pred"] = X @ coeffs
    valid["P_pred_MPa"] = np.exp(valid["ln_P_pred"])
    
    # Metrics
    aard = np.mean(np.abs((valid["P_pred_MPa"] - valid["P_eq_MPa"]) / valid["P_eq_MPa"])) * 100
    rmse = np.sqrt(np.mean((valid["P_pred_MPa"] - valid["P_eq_MPa"])**2))
    r2 = 1.0 - np.sum((valid["P_eq_MPa"] - valid["P_pred_MPa"])**2) / np.sum((valid["P_eq_MPa"] - np.mean(valid["P_eq_MPa"]))**2)

    print("\n" + "="*50)
    print("SURROGATE FIT RESULTS (for Aspen Plus Calculator)")
    print("="*50)
    print(f"ln(P_eq [MPa]) = a0 + a1*(1/T) + a2*y + a3*x + a4*(y/T) + a5*(x/T)")
    labels = ["a0 (const)", "a1 (1/T)", "a2 (y_CO2)", "a3 (x_diox)", "a4 (y/T)", "a5 (x/T)"]
    for lbl, c in zip(labels, coeffs):
        print(f"  {lbl:15s} = {c:16.8e}")

    print("\nModel Quality Metrics on Thermodynamic Grid:")
    print(f"  R^2        = {r2:.6f}")
    print(f"  RMSE (MPa) = {rmse:.4f}")
    print(f"  AARD (%)   = {aard:.2f}%")
    print("="*50)

    # Also compute average CO2 selectivity for the Sep block
    # Gas basis selectivity: z_CO2 / (z_CO2 + z_H2)
    valid["CO2_selectivity"] = valid["z_Hyd_CO2"] / (valid["z_Hyd_CO2"] + valid["z_Hyd_H2"])
    print(f"\nAverage Hydrate CO2 purity (gas basis): {valid['CO2_selectivity'].mean()*100:.1f}%")
    print(f"Min CO2 purity: {valid['CO2_selectivity'].min()*100:.1f}%, Max: {valid['CO2_selectivity'].max()*100:.1f}%")

if __name__ == "__main__":
    run_grid()
