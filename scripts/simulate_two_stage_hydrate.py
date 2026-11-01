"""
simulate_two_stage_hydrate.py
-----------------------------
Rigorous Aspen-equivalent Process Simulator for Two-Stage Hydrate CO2 Capture
with Complete Promoter/Water Solution Recycling.

Compares:
  - Case A: Unpromoted (sI), P1 = 70 bar, T1 = 1.0 C
  - Case B: 1,4-Dioxane Promoted (sII), P1 = 25 bar, T1 = 1.0 C
"""

import numpy as np
import pandas as pd

R_GAS = 8.31446  # kJ/(kmol K)

def calc_compressor_power(F_kmol_h, T_in_K, P_in_bar, P_out_bar, gamma, eta_is=0.75, max_rp=4.0):
    total_rp = P_out_bar / P_in_bar
    if total_rp <= 1.0:
        return 0.0, 1, total_rp, T_in_K
    
    # Calculate number of stages
    N_stages = int(np.ceil(np.log(total_rp) / np.log(max_rp)))
    N_stages = max(1, N_stages)
    rp_stage = total_rp ** (1.0 / N_stages)
    
    # Molar flow in kmol/s
    n_dot = F_kmol_h / 3600.0
    
    # Isentropic power per stage
    exponent = (gamma - 1.0) / gamma
    T_out_is = T_in_K * (rp_stage ** exponent)
    T_out_actual = T_in_K + (T_out_is - T_in_K) / eta_is
    
    W_stage_is = n_dot * (gamma / (gamma - 1.0)) * R_GAS * T_in_K * (rp_stage ** exponent - 1.0)
    W_stage_actual = W_stage_is / eta_is  # kW
    
    W_total_kW = N_stages * W_stage_actual
    return W_total_kW, N_stages, rp_stage, T_out_actual

def calc_pump_power(F_kmol_h, MW_solvent, rho_kg_m3, P_in_bar, P_out_bar, eta_pump=0.70):
    m_dot_kg_s = (F_kmol_h * MW_solvent) / 3600.0
    V_dot_m3_s = m_dot_kg_s / rho_kg_m3
    delta_P_Pa = (P_out_bar - P_in_bar) * 1.0e5
    if delta_P_Pa <= 0:
        return 0.0
    W_pump_kW = (V_dot_m3_s * delta_P_Pa) / (eta_pump * 1000.0)
    return W_pump_kW

def simulate_flowsheet(
    case_name="Case B (Promoted)",
    P_feed_bar=25.0,
    T_feed_C=30.0,
    P1_bar=25.0,
    P2_bar=25.0,
    T1_C=1.0,
    T2_C=1.0,
    x_diox=0.0556,
    L_G_ratio=12.0,
    F_feed_kmol_h=100.0,
    y_CO2_feed=0.40,
    P_diss_bar=4.0,
    T_diss_C=12.0,
    alpha1=0.72,
    alpha2=0.88,
    cop_carnot_fraction=0.30,
):
    T_feed_K = T_feed_C + 273.15
    P_feed_bar = float(P_feed_bar)
    T1_K = T1_C + 273.15
    T2_K = T2_C + 273.15
    T_diss_K = T_diss_C + 273.15
    
    # Molar weights
    MW_CO2 = 44.01
    MW_H2 = 2.016
    MW_H2O = 18.015
    MW_DIOX = 88.11
    
    MW_gas_feed = y_CO2_feed * MW_CO2 + (1.0 - y_CO2_feed) * MW_H2
    gamma_syngas = 1.35
    gamma_CO2_rich = 1.29
    
    MW_solvent = (1.0 - x_diox) * MW_H2O + x_diox * MW_DIOX
    rho_solvent = 1000.0 if x_diox == 0 else 1025.0
    Cp_solvent = (1.0 - x_diox) * 75.3 + x_diox * 152.0  # kJ/(kmol K)
    
    # 1. Feed Compression (COMP-1)
    W_comp1, N_stages1, rp1, T_dis1 = calc_compressor_power(
        F_feed_kmol_h, T_feed_K, P_feed_bar, P1_bar, gamma_syngas, eta_is=0.75, max_rp=5.0
    )
    
    # 2. Feed Chilling (CHILL-1)
    Cp_gas = y_CO2_feed * 37.1 + (1.0 - y_CO2_feed) * 28.8
    Q_chill1 = (F_feed_kmol_h / 3600.0) * Cp_gas * (T_feed_K - T1_K)
    
    # 3. Stage 1 Hydrate Reaction (FORM-1)
    n_CO2_feed = F_feed_kmol_h * y_CO2_feed
    n_H2_feed = F_feed_kmol_h * (1.0 - y_CO2_feed)
    
    n_CO2_hyd1 = n_CO2_feed * alpha1
    z_CO2_hyd1 = 0.985 if x_diox > 0 else 0.983
    n_H2_hyd1 = n_CO2_hyd1 * (1.0 - z_CO2_hyd1) / z_CO2_hyd1
    
    n_CO2_cg1 = n_CO2_feed - n_CO2_hyd1
    n_H2_cg1 = n_H2_feed - n_H2_hyd1
    F_cg1 = n_CO2_cg1 + n_H2_cg1
    y_H2_cg1 = n_H2_cg1 / F_cg1
    
    # Liquid solvent in Stage 1
    F_solv1 = F_feed_kmol_h * L_G_ratio
    
    # Hydrate crystallization heat
    dH_cryst = 58.0 if x_diox > 0 else 62.5  # kJ/mol CO2
    Q_cryst1 = (n_CO2_hyd1 / 3600.0) * dH_cryst * 1000.0  # kW
    
    # 4. Stage 1 Dissociation (DISS-1)
    n_CO2_int = n_CO2_hyd1
    n_H2_int = n_H2_hyd1
    F_int = n_CO2_int + n_H2_int
    y_CO2_int = n_CO2_int / F_int
    
    # 5. Stage 1 Liquid Recycle (PUMP-1 & CHILL-2)
    W_pump1 = calc_pump_power(F_solv1, MW_solvent, rho_solvent, P_diss_bar, P1_bar, eta_pump=0.70)
    Q_chill2 = (F_solv1 / 3600.0) * Cp_solvent * (T_diss_K - T1_K)
    
    # 6. Stage 2 Intermediate Gas Compression (COMP-2)
    W_comp2, N_stages2, rp2, T_dis2 = calc_compressor_power(
        F_int, T_diss_K, P_diss_bar, P2_bar, gamma_CO2_rich, eta_is=0.75, max_rp=6.5
    )
    
    # 7. Stage 2 Gas Chilling (CHILL-3)
    Cp_int = y_CO2_int * 37.1 + (1.0 - y_CO2_int) * 28.8
    Q_chill3 = (F_int / 3600.0) * Cp_int * (T_feed_K - T2_K)
    
    # 8. Stage 2 Hydrate Reaction (FORM-2)
    F_solv2 = F_solv1 * 0.35  # Polishing stage needs smaller liquid inventory
    n_CO2_target = n_CO2_int * alpha2
    z_CO2_hyd2 = 0.992 if x_diox > 0 else 0.989
    n_H2_target = n_CO2_target * (1.0 - z_CO2_hyd2) / z_CO2_hyd2
    
    n_CO2_cg2 = n_CO2_int - n_CO2_target
    n_H2_cg2 = n_H2_int - n_H2_target
    F_cg2 = n_CO2_cg2 + n_H2_cg2
    
    Q_cryst2 = (n_CO2_target / 3600.0) * dH_cryst * 1000.0  # kW
    
    # 9. Stage 2 Dissociation (DISS-2)
    F_target = n_CO2_target + n_H2_target
    purity_CO2 = (n_CO2_target / F_target) * 100.0
    
    # 10. Stage 2 Liquid Recycle (PUMP-2 & CHILL-4)
    W_pump2 = calc_pump_power(F_solv2, MW_solvent, rho_solvent, P_diss_bar, P2_bar, eta_pump=0.70)
    Q_chill4 = (F_solv2 / 3600.0) * Cp_solvent * (T_diss_K - T2_K)
    
    # Overall Performance Metrics
    rec_CO2 = (n_CO2_target / n_CO2_feed) * 100.0
    rec_H2 = ((n_H2_cg1 + n_H2_cg2) / n_H2_feed) * 100.0
    m_dot_CO2_captured_kg_h = n_CO2_target * MW_CO2
    
    # Refrigeration COP
    COP_carnot = T1_K / (T_feed_K - T1_K)
    COP_actual = cop_carnot_fraction * COP_carnot
    
    # Total refrigeration electrical power
    Q_total_chill = Q_chill1 + Q_chill2 + Q_chill3 + Q_chill4 + Q_cryst1 + Q_cryst2
    W_refrig_electric = Q_total_chill / COP_actual
    
    # Total Power
    W_compressors = W_comp1 + W_comp2
    W_pumps = W_pump1 + W_pump2
    W_total = W_compressors + W_pumps + W_refrig_electric
    
    # Specific Energy Consumption (SEC)
    SEC_kWh_per_kg_CO2 = W_total / m_dot_CO2_captured_kg_h
    SEC_MJ_per_kg_CO2 = SEC_kWh_per_kg_CO2 * 3.6
    
    results = {
        "Case": case_name,
        "P1 (bar)": P1_bar,
        "P2 (bar)": P2_bar,
        "Promoter mol%": x_diox * 100.0,
        "CO2 Recovery (%)": rec_CO2,
        "H2 Recovery (%)": rec_H2,
        "CO2 Purity (mol%)": purity_CO2,
        "H2 Clean Gas 1 (mol%)": y_H2_cg1 * 100.0,
        "CO2 Captured (kg/h)": m_dot_CO2_captured_kg_h,
        "W_COMP-1 (kW)": W_comp1,
        "W_COMP-2 (kW)": W_comp2,
        "W_Compressors Total (kW)": W_compressors,
        "W_Pumps Total (kW)": W_pumps,
        "Q_Chilling Thermal (kW)": Q_total_chill,
        "COP Refrig": COP_actual,
        "W_Refrig Electric (kW)": W_refrig_electric,
        "W_Total Electric (kW)": W_total,
        "SEC (kWh/kg CO2)": SEC_kWh_per_kg_CO2,
        "SEC (MJ/kg CO2)": SEC_MJ_per_kg_CO2,
    }
    return results

if __name__ == "__main__":
    res_a = simulate_flowsheet(
        case_name="Case A (Unpromoted, 70 bar)",
        P1_bar=70.0,
        P2_bar=70.0,
        x_diox=0.0,
        alpha1=0.72,
        alpha2=0.88,
    )
    res_b = simulate_flowsheet(
        case_name="Case B (1,4-Dioxane Promoted, 25 bar)",
        P1_bar=25.0,
        P2_bar=25.0,
        x_diox=0.0556,
        alpha1=0.72,
        alpha2=0.88,
    )
    
    df_compare = pd.DataFrame([res_a, res_b])
    print("=" * 80)
    print("MTP TWO-STAGE HYDRATE CAPTURE: CASE A vs. CASE B SUMMARY")
    print("=" * 80)
    for col in df_compare.columns:
        val_a = df_compare.loc[0, col]
        val_b = df_compare.loc[1, col]
        if isinstance(val_a, float):
            print(f"  {col:32s}: Case A = {val_a:10.2f} | Case B = {val_b:10.2f}")
        else:
            print(f"  {col:32s}: Case A = {val_a} | Case B = {val_b}")
    print("=" * 80)
