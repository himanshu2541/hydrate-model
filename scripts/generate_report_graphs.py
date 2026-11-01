"""
generate_report_graphs.py
--------------------------
Generates publication-quality figures for reports and thesis:
  1. Phase Equilibrium (P-T) Boundary: Pure CO2 vs CO2+H2 vs CO2+H2+DIOX (Promoter)
  2. Separation Factor & H2 Enrichment vs Temperature
  3. Fractional Cage Occupancies (Small & Large Cages) vs Temperature
  4. Hydrate Selectivity & Equilibrium Pressure vs Feed CO2 Composition
  5. 4-Panel Comprehensive Summary Dashboard

Features:
  - Legends placed neatly at the bottom outside the plotting area for maximum clarity
  - Automatic export of all plotted data to CSV files in `reports/data/`
  - High-DPI outputs copied to reports and artifact directory
"""

import os
import shutil
import tempfile

# Ensure Matplotlib uses a writable config dir in restricted environments
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydrate_project.services.solver_runner import run_model
from hydrate_project.services.literature_data import PRESET_DATA

# Output directories
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
DATA_DIR = os.path.join(BASE_DIR, "reports", "data")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

ARTIFACT_DIR = r"C:\Users\hskus\.gemini\antigravity\brain\ef9c5d82-4990-4c9a-a408-e99eaa6ede22"

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.5,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "figure.autolayout": False,  # Managed manually to prevent clipping with external legends
})


def export_and_log_csv(df: pd.DataFrame, filename: str, title: str):
    """Helper to save CSV to reports/data and artifact directory, and log summary."""
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    
    # Copy to artifact dir
    if os.path.isdir(ARTIFACT_DIR):
        shutil.copy2(filepath, os.path.join(ARTIFACT_DIR, filename))
        
    print(f"\n[DATA LOG] {title}")
    print(f"  -> File: {filepath} ({len(df)} rows)")
    print(df.head(4).to_string(index=False))
    if len(df) > 4:
        print(f"  ... ({len(df) - 4} more rows)")
    return filepath


def generate_figure1_phase_equilibrium():
    print("\n" + "=" * 60)
    print("Generating Figure 1: Phase Equilibrium P-T Diagram...")
    print("=" * 60)
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)

    # 1. Experimental Data: Kumar et al. (2006) for 39.2/60.8 CO2/H2
    preset_mix = PRESET_DATA["CO₂/H₂ (39.2/60.8 mol%) — Kumar et al. 2006  [sI, bulk water]"]
    T_exp_mix = np.array(preset_mix["T (K)"])
    P_exp_mix = np.array(preset_mix["P_eq (MPa)"])

    # 2. Experimental Data: Sloan & Koh (2008) for Pure CO2
    preset_pure = PRESET_DATA["CO₂ (Pure) — Sloan & Koh 2008  [sI, bulk water]"]
    T_exp_pure = np.array(preset_pure["T (K)"])
    P_exp_pure = np.array(preset_pure["P_eq (MPa)"])

    # 3. Model Curves: CO2/H2 (40/60)
    T_model_mix = np.linspace(273.5, 278.6, 25)
    res_srk, _ = run_model({"CO2": 0.40, "H2": 0.60}, {"H2O": 1.0}, T_model_mix, ["Soave-Redlich-Kwong"], lambda m: None)
    res_pr, _ = run_model({"CO2": 0.40, "H2": 0.60}, {"H2O": 1.0}, T_model_mix, ["Peng-Robinson"], lambda m: None)
    df_srk = res_srk["Soave-Redlich-Kwong"]
    df_pr = res_pr["Peng-Robinson"]

    # 4. Model Curve: Pure CO2
    T_model_pure = np.linspace(273.15, 281.0, 20)
    res_pure, _ = run_model({"CO2": 1.0}, {"H2O": 1.0}, T_model_pure, ["Soave-Redlich-Kwong"], lambda m: None)
    df_pure = res_pure["Soave-Redlich-Kwong"]

    # 5. Model Curve: CO2/H2 (40/60) with 1,4-Dioxane Promoter (5.56 mol% in water)
    T_model_prom = np.linspace(273.5, 281.0, 16)
    res_prom, _ = run_model({"CO2": 0.40, "H2": 0.60}, {"H2O": 0.9444, "DIOX": 0.0556}, T_model_prom, ["Soave-Redlich-Kwong"], lambda m: None)
    df_prom = res_prom["Soave-Redlich-Kwong"]

    # --- Plot experimental points ---
    ax.plot(T_exp_mix, P_exp_mix, "rs", label="CO₂/H₂ (39.2/60.8) Exp. [Kumar et al. 2006]", zorder=5)
    ax.plot(T_exp_pure, P_exp_pure, "ko", markerfacecolor="none", label="Pure CO₂ Exp. [Sloan & Koh 2008]", zorder=4)

    # --- Plot model lines ---
    ax.plot(df_srk["T (K)"], df_srk["P_eq (MPa)"], "r-", label="CO₂/H₂ (40/60) - SRK Model (sI)")
    ax.plot(df_pr["T (K)"], df_pr["P_eq (MPa)"], "r--", label="CO₂/H₂ (40/60) - PR Model (sI)")
    ax.plot(df_pure["T (K)"], df_pure["P_eq (MPa)"], "k-", label="Pure CO₂ - SRK Model (sI)")
    ax.plot(df_prom["T (K)"], df_prom["P_eq (MPa)"], "b-^", label="CO₂/H₂ (40/60) + 5.56% 1,4-Dioxane (sII)")

    ax.set_xlabel("Temperature $T$ (K)")
    ax.set_ylabel("Equilibrium Pressure $P_{eq}$ (MPa)")
    ax.set_title("Hydrate Phase Equilibrium Boundaries: Effect of $H_2$ Dilution & Promoter")
    ax.grid(True)
    ax.set_xlim(272.5, 283.8)
    ax.set_ylim(0.5, 12.8)

    # Shaded regions / notes positioned cleanly without obstructing data
    ax.annotate("Hydrate Stable Region\n(High P / Low T)", xy=(273.5, 11.2), color="navy", fontweight="bold", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="#eef4ff", ec="#8bb2e8", lw=0.8, alpha=0.9))
    ax.annotate("Fluid Stable Region\n(Low P / High T)", xy=(279.5, 1.2), color="darkred", fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff2f2", ec="#e8a8a8", lw=0.8, alpha=0.9))
    ax.annotate("~70% Pressure Reduction\nwith 1,4-Dioxane Promoter", 
                xy=(276.5, 2.5), xytext=(278.5, 5.8), ha="center",
                arrowprops=dict(facecolor="blue", arrowstyle="->", lw=1.5),
                fontsize=9.5, color="blue", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f0f7ff", ec="#a0c5f0", lw=0.8, alpha=0.9))

    # Legend at the bottom OUTSIDE the plotting area
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, framealpha=0.95, edgecolor="gray")

    filepath = os.path.join(FIGURES_DIR, "figure1_phase_equilibrium.png")
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1: {filepath}")

    # --- Data Logging and CSV Export ---
    # 1. Experimental Data CSV
    df_exp1 = pd.DataFrame({
        "System": "CO2/H2 (39.2/60.8 mol%)",
        "Reference": "Kumar et al. (2006)",
        "Structure": "sI",
        "T_K": np.round(T_exp_mix, 2),
        "T_C": np.round(T_exp_mix - 273.15, 2),
        "P_eq_MPa": np.round(P_exp_mix, 3)
    })
    df_exp2 = pd.DataFrame({
        "System": "Pure CO2",
        "Reference": "Sloan & Koh (2008)",
        "Structure": "sI",
        "T_K": np.round(T_exp_pure, 2),
        "T_C": np.round(T_exp_pure - 273.15, 2),
        "P_eq_MPa": np.round(P_exp_pure, 3)
    })
    df_exp_all = pd.concat([df_exp1, df_exp2], ignore_index=True)
    export_and_log_csv(df_exp_all, "figure1_experimental_data.csv", "Figure 1 Experimental Benchmark Points")

    # 2. Model Phase Equilibrium Curves CSV (Tidy Format)
    records = []
    for _, row in df_srk.iterrows():
        records.append({"System": "CO2/H2 (40/60 mol%)", "Model": "SRK", "Structure": "sI", 
                        "T_K": round(row["T (K)"], 2), "T_C": round(row["T (K)"] - 273.15, 2), "P_eq_MPa": round(row["P_eq (MPa)"], 3)})
    for _, row in df_pr.iterrows():
        records.append({"System": "CO2/H2 (40/60 mol%)", "Model": "PR", "Structure": "sI", 
                        "T_K": round(row["T (K)"], 2), "T_C": round(row["T (K)"] - 273.15, 2), "P_eq_MPa": round(row["P_eq (MPa)"], 3)})
    for _, row in df_pure.iterrows():
        records.append({"System": "Pure CO2", "Model": "SRK", "Structure": "sI", 
                        "T_K": round(row["T (K)"], 2), "T_C": round(row["T (K)"] - 273.15, 2), "P_eq_MPa": round(row["P_eq (MPa)"], 3)})
    for _, row in df_prom.iterrows():
        records.append({"System": "CO2/H2 (40/60) + 5.56% 1,4-Dioxane", "Model": "SRK", "Structure": "sII", 
                        "T_K": round(row["T (K)"], 2), "T_C": round(row["T (K)"] - 273.15, 2), "P_eq_MPa": round(row["P_eq (MPa)"], 3)})
    df_model_all = pd.DataFrame(records)
    export_and_log_csv(df_model_all, "figure1_model_phase_boundaries.csv", "Figure 1 Thermodynamic Model Curves")

    return filepath


def generate_figure2_separation_factor():
    print("\n" + "=" * 60)
    print("Generating Figure 2: Separation Factor and Hydrate Purity...")
    print("=" * 60)
    fig, ax1 = plt.subplots(figsize=(8.5, 5.8), dpi=300)

    preset = PRESET_DATA["CO₂/H₂ (39.2/60.8 mol%) — Kumar et al. 2006  [sI, bulk water]"]
    T_range = np.array(sorted(set(preset["T (K)"])))
    res, _ = run_model({"CO2": 0.40, "H2": 0.60}, {"H2O": 1.0}, T_range, ["Soave-Redlich-Kwong"], lambda m: None)
    df = res["Soave-Redlich-Kwong"]

    T_C = df["T (K)"] - 273.15
    sf = df["SF_CO2_H2"]
    co2_hyd = df["z_Hyd_CO2"] * 100.0
    h2_hyd = df["z_Hyd_H2"] * 100.0

    color1 = "crimson"
    ax1.set_xlabel("Temperature $T$ (°C)")
    ax1.set_ylabel("Separation Factor ($SF_{\\mathrm{CO_2/H_2}}$)", color=color1)
    line1 = ax1.plot(T_C, sf, color=color1, marker="o", lw=2.2, label="Separation Factor ($SF_{\\mathrm{CO_2/H_2}}$)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True)
    ax1.set_ylim(2.40, 2.52)

    ax2 = ax1.twinx()
    color2 = "teal"
    ax2.set_ylabel("Hydrate Guest Purity (mol%)", color=color2)
    line2 = ax2.plot(T_C, co2_hyd, color=color2, marker="s", lw=2.2, label="$\\mathrm{CO_2}$ in Hydrate (mol%)")
    line3 = ax2.plot(T_C, h2_hyd, color="darkorange", marker="^", lw=2.0, linestyle="--", label="$\\mathrm{H_2}$ in Hydrate (mol%)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 105)

    # Combine legends into one clean box at the bottom OUTSIDE the plotting area
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True, framealpha=0.95, edgecolor="gray")

    ax1.set_title("Carbon Capture Selectivity: Separation Factor & Hydrate Purity vs. Temperature")
    filepath = os.path.join(FIGURES_DIR, "figure2_separation_factor.png")
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 2: {filepath}")

    # --- Data Logging and CSV Export ---
    df_sf_export = pd.DataFrame({
        "Temperature_K": np.round(df["T (K)"], 2),
        "Temperature_C": np.round(T_C, 2),
        "P_eq_MPa": np.round(df["P_eq (MPa)"], 3),
        "Separation_Factor_CO2_H2": np.round(sf, 4),
        "Hydrate_CO2_mol_pct": np.round(co2_hyd, 2),
        "Hydrate_H2_mol_pct": np.round(h2_hyd, 2),
        "Gas_Feed_CO2_mol_pct": 40.0,
        "Gas_Feed_H2_mol_pct": 60.0,
    })
    export_and_log_csv(df_sf_export, "figure2_separation_factor_data.csv", "Figure 2 Separation Factor & Hydrate Purity Data")

    return filepath


def generate_figure3_cage_occupancy():
    print("\n" + "=" * 60)
    print("Generating Figure 3: Fractional Cage Occupancy...")
    print("=" * 60)
    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=300)

    preset = PRESET_DATA["CO₂/H₂ (39.2/60.8 mol%) — Kumar et al. 2006  [sI, bulk water]"]
    T_range = np.array(sorted(set(preset["T (K)"])))
    res, _ = run_model({"CO2": 0.40, "H2": 0.60}, {"H2O": 1.0}, T_range, ["Soave-Redlich-Kwong"], lambda m: None)
    df = res["Soave-Redlich-Kwong"]

    T_K = df["T (K)"]

    ax.plot(T_K, df["Theta_Large_CO2"], "b-s", label="$\\theta_{\\mathrm{large}}^{\\mathrm{CO_2}}$ ($5^{12}6^2$ Large Cage)", lw=2.2)
    ax.plot(T_K, df["Theta_Small_CO2"], "g-o", label="$\\theta_{\\mathrm{small}}^{\\mathrm{CO_2}}$ ($5^{12}$ Small Cage)", lw=2.2)
    ax.plot(T_K, df["Theta_Small_H2"], "m--^", label="$\\theta_{\\mathrm{small}}^{\\mathrm{H_2}}$ ($5^{12}$ Small Cage)", lw=2.0)
    ax.plot(T_K, df["Theta_Large_H2"], "r:d", label="$\\theta_{\\mathrm{large}}^{\\mathrm{H_2}}$ ($5^{12}6^2$ Large Cage)", lw=2.0)

    ax.set_xlabel("Temperature $T$ (K)")
    ax.set_ylabel("Fractional Cage Occupancy ($\\theta_i$)")
    ax.set_title("Structure I Cage Occupancies: Selective Encapsulation of $\\mathrm{CO_2}$ over $\\mathrm{H_2}$")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True)

    # Legend at the bottom OUTSIDE the plotting area
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, framealpha=0.95, edgecolor="gray")

    # Callout text boxes positioned neatly
    ax.annotate("Large cages saturated by $\\mathrm{CO_2}$ (~98%)\ndriving thermodynamic stability",
                xy=(275.5, 0.978), xytext=(274.0, 0.80),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
                fontsize=9.2, bbox=dict(boxstyle="round,pad=0.3", fc="#fffae6", ec="#d4b106", lw=0.8, alpha=0.9))

    ax.annotate("$\\mathrm{H_2}$ occupancy < 5% due to\nweak dispersion energy ($\\epsilon/k = 31.7$ K)",
                xy=(276.5, 0.04), xytext=(274.2, 0.22),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
                fontsize=9.2, bbox=dict(boxstyle="round,pad=0.3", fc="#ffeef0", ec="#d98282", lw=0.8, alpha=0.9))

    filepath = os.path.join(FIGURES_DIR, "figure3_cage_occupancy.png")
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 3: {filepath}")

    # --- Data Logging and CSV Export ---
    df_cage_export = pd.DataFrame({
        "Temperature_K": np.round(T_K, 2),
        "Temperature_C": np.round(T_K - 273.15, 2),
        "P_eq_MPa": np.round(df["P_eq (MPa)"], 3),
        "Theta_Large_CO2_5_12_6_2": np.round(df["Theta_Large_CO2"], 4),
        "Theta_Small_CO2_5_12": np.round(df["Theta_Small_CO2"], 4),
        "Theta_Small_H2_5_12": np.round(df["Theta_Small_H2"], 4),
        "Theta_Large_H2_5_12_6_2": np.round(df["Theta_Large_H2"], 4),
        "Hydrate_CO2_mol_pct": np.round(df["z_Hyd_CO2"] * 100.0, 2),
        "Hydrate_H2_mol_pct": np.round(df["z_Hyd_H2"] * 100.0, 2),
    })
    export_and_log_csv(df_cage_export, "figure3_cage_occupancy_data.csv", "Figure 3 Fractional Cage Occupancy Data")

    return filepath


def generate_figure4_feed_composition_effect():
    print("\n" + "=" * 60)
    print("Generating Figure 4: Feed Composition Effect...")
    print("=" * 60)
    fig, ax1 = plt.subplots(figsize=(8.5, 5.8), dpi=300)

    # Scan feed CO2 fraction from 10% to 90% at T = 274.0 K
    feed_co2_fractions = np.linspace(0.10, 0.90, 17)
    P_eq_list = []
    co2_hyd_purity = []
    h2_hyd_purity = []
    sf_list = []

    for y_co2 in feed_co2_fractions:
        y_h2 = 1.0 - y_co2
        res, _ = run_model({"CO2": y_co2, "H2": y_h2}, {"H2O": 1.0}, np.array([274.0]), ["Soave-Redlich-Kwong"], lambda m: None)
        df = res["Soave-Redlich-Kwong"]
        P_eq_list.append(df["P_eq (MPa)"].values[0])
        c_hyd = df["z_Hyd_CO2"].values[0] * 100.0
        h_hyd = df["z_Hyd_H2"].values[0] * 100.0
        co2_hyd_purity.append(c_hyd)
        h2_hyd_purity.append(h_hyd)
        sf_list.append(df["SF_CO2_H2"].values[0])

    color1 = "navy"
    ax1.set_xlabel("Feed Gas $\\mathrm{CO_2}$ Mole Percentage (mol%)")
    ax1.set_ylabel("Equilibrium Pressure $P_{eq}$ (MPa)", color=color1)
    line1 = ax1.plot(feed_co2_fractions * 100.0, P_eq_list, color=color1, marker="o", lw=2.2, label="$P_{eq}$ at 274.0 K (MPa)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = "darkgreen"
    ax2.set_ylabel("$\\mathrm{CO_2}$ Purity in Captured Hydrate (mol%)", color=color2)
    line2 = ax2.plot(feed_co2_fractions * 100.0, co2_hyd_purity, color=color2, marker="s", lw=2.2, linestyle="--", label="Hydrate $\\mathrm{CO_2}$ Purity (mol%)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(95.0, 100.2)

    # Combine legends into one clean box at the bottom OUTSIDE the plotting area
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, framealpha=0.95, edgecolor="gray")

    ax1.set_title("Effect of Feed Gas Composition on Equilibrium Pressure & Hydrate Purity ($T = 274.0$ K)")
    filepath = os.path.join(FIGURES_DIR, "figure4_feed_composition_effect.png")
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 4: {filepath}")

    # --- Data Logging and CSV Export ---
    df_feed_export = pd.DataFrame({
        "Feed_CO2_mol_pct": np.round(feed_co2_fractions * 100.0, 1),
        "Feed_H2_mol_pct": np.round((1.0 - feed_co2_fractions) * 100.0, 1),
        "Temperature_K": 274.0,
        "Temperature_C": 0.85,
        "P_eq_MPa": np.round(P_eq_list, 3),
        "Hydrate_CO2_Purity_mol_pct": np.round(co2_hyd_purity, 2),
        "Hydrate_H2_mol_pct": np.round(h2_hyd_purity, 2),
        "Separation_Factor": np.round(sf_list, 4)
    })
    export_and_log_csv(df_feed_export, "figure4_feed_composition_effect_data.csv", "Figure 4 Feed Composition Sensitivity Data")

    return filepath


def generate_figure5_comprehensive_dashboard():
    print("\n" + "=" * 60)
    print("Generating Figure 5: Comprehensive 4-Panel Dashboard...")
    print("=" * 60)
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # Panel (a): P-T Curves
    preset_mix = PRESET_DATA["CO₂/H₂ (39.2/60.8 mol%) — Kumar et al. 2006  [sI, bulk water]"]
    T_exp_mix = np.array(preset_mix["T (K)"])
    P_exp_mix = np.array(preset_mix["P_eq (MPa)"])
    preset_pure = PRESET_DATA["CO₂ (Pure) — Sloan & Koh 2008  [sI, bulk water]"]

    T_model_mix = np.linspace(273.5, 278.6, 20)
    res_srk, _ = run_model({"CO2": 0.40, "H2": 0.60}, {"H2O": 1.0}, T_model_mix, ["Soave-Redlich-Kwong"], lambda m: None)
    df_srk = res_srk["Soave-Redlich-Kwong"]

    T_model_prom = np.linspace(273.5, 281.0, 16)
    res_prom, _ = run_model({"CO2": 0.40, "H2": 0.60}, {"H2O": 0.9444, "DIOX": 0.0556}, T_model_prom, ["Soave-Redlich-Kwong"], lambda m: None)
    df_prom = res_prom["Soave-Redlich-Kwong"]

    axs[0, 0].plot(T_exp_mix, P_exp_mix, "rs", label="CO₂/H₂ Exp. [Kumar]")
    axs[0, 0].plot(preset_pure["T (K)"], preset_pure["P_eq (MPa)"], "ko", mfc="none", label="Pure CO₂ Exp. [Sloan]")
    axs[0, 0].plot(df_srk["T (K)"], df_srk["P_eq (MPa)"], "r-", label="CO₂/H₂ Model (sI)")
    axs[0, 0].plot(df_prom["T (K)"], df_prom["P_eq (MPa)"], "b-^", label="CO₂/H₂ + DIOX (sII)")
    axs[0, 0].set_title("(a) Phase Equilibrium Boundaries")
    axs[0, 0].set_xlabel("Temperature $T$ (K)")
    axs[0, 0].set_ylabel("Equilibrium Pressure $P_{eq}$ (MPa)")
    axs[0, 0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True, edgecolor="gray")
    axs[0, 0].grid(True)

    # Panel (b): Separation Factor & Gas Enrichment
    T_range = np.array(sorted(set(preset_mix["T (K)"])))
    res_b, _ = run_model({"CO2": 0.40, "H2": 0.60}, {"H2O": 1.0}, T_range, ["Soave-Redlich-Kwong"], lambda m: None)
    df_b = res_b["Soave-Redlich-Kwong"]
    axs[0, 1].plot(df_b["T (K)"], df_b["SF_CO2_H2"], "purple", marker="o", lw=2.2, label="Separation Factor $SF_{\\mathrm{CO_2/H_2}}$")
    axs[0, 1].set_title("(b) Carbon Capture Separation Factor")
    axs[0, 1].set_xlabel("Temperature $T$ (K)")
    axs[0, 1].set_ylabel("Separation Factor ($SF$)")
    axs[0, 1].set_ylim(2.40, 2.52)
    axs[0, 1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=True, edgecolor="gray")
    axs[0, 1].grid(True)

    # Panel (c): Cage Occupancy
    axs[1, 0].plot(df_b["T (K)"], df_b["Theta_Large_CO2"], "b-s", label="$\\theta_{\\mathrm{large}}^{\\mathrm{CO_2}}$ ($5^{12}6^2$)", lw=2.0)
    axs[1, 0].plot(df_b["T (K)"], df_b["Theta_Small_CO2"], "g-o", label="$\\theta_{\\mathrm{small}}^{\\mathrm{CO_2}}$ ($5^{12}$)", lw=2.0)
    axs[1, 0].plot(df_b["T (K)"], df_b["Theta_Small_H2"], "m--^", label="$\\theta_{\\mathrm{small}}^{\\mathrm{H_2}}$ ($5^{12}$)", lw=1.8)
    axs[1, 0].plot(df_b["T (K)"], df_b["Theta_Large_H2"], "r:d", label="$\\theta_{\\mathrm{large}}^{\\mathrm{H_2}}$ ($5^{12}6^2$)", lw=1.8)
    axs[1, 0].set_title("(c) Structure I Cage Occupancies ($\\theta$)")
    axs[1, 0].set_xlabel("Temperature $T$ (K)")
    axs[1, 0].set_ylabel("Occupancy Fraction ($\\theta$)")
    axs[1, 0].set_ylim(-0.02, 1.05)
    axs[1, 0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True, edgecolor="gray")
    axs[1, 0].grid(True)

    # Panel (d): Feed Sensitivity
    feed_co2_fractions = np.linspace(0.15, 0.85, 12)
    P_eq_sens = []
    for y_co2 in feed_co2_fractions:
        res_d, _ = run_model({"CO2": y_co2, "H2": 1.0 - y_co2}, {"H2O": 1.0}, np.array([274.0]), ["Soave-Redlich-Kwong"], lambda m: None)
        df_d = res_d["Soave-Redlich-Kwong"]
        P_eq_sens.append(df_d["P_eq (MPa)"].values[0])

    axs[1, 1].plot(feed_co2_fractions * 100, P_eq_sens, "navy", marker="s", lw=2.2, label="$P_{eq}$ at 274.0 K (MPa)")
    axs[1, 1].set_title("(d) Sensitivity to Feed $\\mathrm{CO_2}$ Fraction")
    axs[1, 1].set_xlabel("Feed $\\mathrm{CO_2}$ Percentage (mol%)")
    axs[1, 1].set_ylabel("Equilibrium Pressure $P_{eq}$ (MPa)")
    axs[1, 1].grid(True)
    axs[1, 1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=True, edgecolor="gray")

    fig.subplots_adjust(hspace=0.45, wspace=0.25, bottom=0.10)
    filepath = os.path.join(FIGURES_DIR, "figure5_comprehensive_dashboard.png")
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 5: {filepath}")
    return filepath


def main():
    print("=" * 70)
    print("Running Full Figure Suite & Data Logger for Hydrate Equilibrium Study")
    print("=" * 70)

    f1 = generate_figure1_phase_equilibrium()
    f2 = generate_figure2_separation_factor()
    f3 = generate_figure3_cage_occupancy()
    f4 = generate_figure4_feed_composition_effect()
    f5 = generate_figure5_comprehensive_dashboard()

    # Copy figure files to artifact directory for presentation
    for f in [f1, f2, f3, f4, f5]:
        dest = os.path.join(ARTIFACT_DIR, os.path.basename(f))
        shutil.copy2(f, dest)
        print(f"Updated artifact image: {dest}")

    print("\n" + "=" * 70)
    print("All figures and CSV datasets generated and saved successfully!")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Data CSVs saved in: {DATA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
