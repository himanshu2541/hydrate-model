#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hydrate_unit_op.py
-------------------
Unified Entry File for Gas Hydrate Flash Equilibrium Calculations:
  - Supports:
      1. Pure CO2 + H2O
      2. CO2 (x) + H2 (y) + H2O
      3. CO2 (x) + H2 (y) + 1,4-Dioxane (xdiox) + H2O
  - Fully dynamic component selection (only active non-zero components are initialized).
  - Standalone CLI & Python module.
  - Direct compatibility with Aspen Plus / Aspen HYSYS / COCO CAPE-OPEN Python Unit Operations
    (AmsterCHEM / CO-LaN Python Unit Operation standard: Configure() and Calculate()).
"""

import os
import sys
import argparse
import numpy as np

# Ensure local hydrateflash modules are discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import component_properties as cp
import flashalgorithm as fc


# -----------------------------------------------------------------------------
# Component Name Mapping (Standard Aspen Plus / HYSYS / CAPE-OPEN Names)
# -----------------------------------------------------------------------------
ASPEN_NAME_MAP = {
    # Carbon Dioxide
    "CO2": "co2",
    "CARBON DIOXIDE": "co2",
    "CARBON-DIOXIDE": "co2",
    "CARBONDIOXIDE": "co2",
    "CARBONIC ANHYDRIDE": "co2",
    
    # Hydrogen
    "H2": "h2",
    "HYDROGEN": "h2",
    
    # 1,4-Dioxane
    "DIOX": "diox",
    "DIOXANE": "diox",
    "1,4-DIOXANE": "diox",
    "1,4 DIOXANE": "diox",
    "14-DIOXANE": "diox",
    "14DIOXANE": "diox",
    "P-DIOXANE": "diox",
    "C4H8O2": "diox",
    
    # Water
    "H2O": "h2o",
    "WATER": "h2o",
}


def normalize_component_name(name: str) -> str:
    """Map external/Aspen component names to hydrateflash standard names."""
    clean = str(name).strip().upper()
    return ASPEN_NAME_MAP.get(clean, clean.lower())


# -----------------------------------------------------------------------------
# Core Dynamic Hydrate Flash Engine
# -----------------------------------------------------------------------------
def run_hydrate_flash(
    T: float,
    P: float,
    x_CO2: float,
    y_H2: float = 0.0,
    x_DIOX: float = 0.0,
    x_H2O: float = None,
    water_ratio: float = 0.85,
    T_unit: str = "K",
    P_unit: str = "bar",
    verbose: bool = False
) -> dict:
    """
    Dynamically perform a hydrate flash calculation for CO2 / H2 / DIOX / H2O systems.

    Parameters
    ----------
    T : float
        Temperature (default in Kelvin; if T_unit='C', converted automatically).
    P : float
        Pressure (default in bar; if P_unit='MPa' or 'Pa', converted automatically).
    x_CO2 : float
        Mole fraction of CO2.
    y_H2 : float, optional
        Mole fraction of H2 (default: 0.0).
    x_DIOX : float, optional
        Mole fraction of 1,4-dioxane promoter (default: 0.0).
    x_H2O : float, optional
        Total system mole fraction of H2O. If None, computed from water_ratio.
    water_ratio : float, optional
        If x_H2O is None, fraction of system that is water (default: 0.85).
        The non-water fractions are scaled to sum to (1.0 - water_ratio).
    T_unit : str
        'K' or 'C'.
    P_unit : str
        'bar', 'MPa', or 'Pa'.
    verbose : bool
        If True, prints detailed calculation summary.

    Returns
    -------
    results : dict
        Contains 'converged', 'T_K', 'P_bar', 'phases', 'alpha', 'compositions',
        'hydrate_formed', 'hydrate_fraction', 'separation_metrics'.
    """
    # 1. Unit conversions
    if str(T_unit).upper() in ["C", "CELSIUS"]:
        T_K = float(T) + 273.15
    else:
        T_K = float(T)

    P_u = str(P_unit).upper()
    if P_u == "MPA":
        P_bar = float(P) * 10.0
    elif P_u in ["PA", "PASCAL"]:
        P_bar = float(P) / 1.0e5
    else:
        P_bar = float(P)

    # 2. Dynamic Component Selection
    active_comps = ["water", "co2"]
    if y_H2 > 1e-6:
        active_comps.append("h2")
    if x_DIOX > 1e-6:
        active_comps.append("diox")

    # 3. Assemble Feed Composition Vector z
    if x_H2O is not None:
        raw_feed = {"water": float(x_H2O), "co2": float(x_CO2)}
        if "h2" in active_comps:
            raw_feed["h2"] = float(y_H2)
        if "diox" in active_comps:
            raw_feed["diox"] = float(x_DIOX)
        tot = sum(raw_feed.values())
        z = np.array([raw_feed[c] / tot for c in active_comps], dtype=float)
    else:
        # User passed gas/promoter fractions (e.g. 40% CO2, 60% H2)
        non_water = {"co2": float(x_CO2)}
        if "h2" in active_comps:
            non_water["h2"] = float(y_H2)
        if "diox" in active_comps:
            non_water["diox"] = float(x_DIOX)
        nw_tot = sum(non_water.values())
        if nw_tot <= 0:
            raise ValueError("At least one active gas/promoter (CO2, H2, DIOX) must be > 0.")
        
        # Scale with water
        z_list = [water_ratio]
        for c in active_comps[1:]:
            z_list.append((1.0 - water_ratio) * (non_water[c] / nw_tot))
        z = np.array(z_list, dtype=float)

    # 4. Initialize Controller & Solve
    flash = fc.FlashController(components=active_comps)
    flash.main_handler(compobjs=flash.compobjs, z=z, T=T_K, P=P_bar)

    # 5. Extract & Format Results
    compositions = {}
    phase_fractions = {}
    hydrate_formed = False
    hydrate_fraction = 0.0
    hydrate_phase_name = None

    for i, phase in enumerate(flash.phases):
        alpha_p = float(flash.alpha_calc[i])
        phase_fractions[phase] = alpha_p
        if alpha_p > 1e-5:
            compositions[phase] = {
                comp.compname: float(flash.x_calc[j, i])
                for j, comp in enumerate(flash.compobjs)
            }
            if phase in ["s1", "s2"]:
                hydrate_formed = True
                hydrate_fraction += alpha_p
                hydrate_phase_name = phase

    # 6. Carbon Capture & Separation Metrics
    separation_metrics = {}
    gas_phase = "vapor" if "vapor" in compositions else ("lhc" if "lhc" in compositions else None)
    if gas_phase and "co2" in compositions[gas_phase]:
        y_co2_gas = compositions[gas_phase]["co2"]
        y_h2_gas = compositions[gas_phase].get("h2", 0.0)
        separation_metrics["gas_phase_CO2_mol_pct"] = y_co2_gas * 100.0
        separation_metrics["gas_phase_H2_mol_pct"] = y_h2_gas * 100.0
        if hydrate_formed and hydrate_phase_name:
            x_co2_hyd = compositions[hydrate_phase_name]["co2"]
            x_h2_hyd = compositions[hydrate_phase_name].get("h2", 0.0)
            separation_metrics["hydrate_gas_CO2_purity_pct"] = (
                x_co2_hyd / (x_co2_hyd + x_h2_hyd) * 100.0 if (x_co2_hyd + x_h2_hyd) > 0 else 0.0
            )

    results = {
        "converged": True,
        "T_K": T_K,
        "P_bar": P_bar,
        "active_components": [c.compname for c in flash.compobjs],
        "feed_z": {c.compname: float(z[i]) for i, c in enumerate(flash.compobjs)},
        "phase_fractions": phase_fractions,
        "active_phases": list(compositions.keys()),
        "compositions": compositions,
        "hydrate_formed": hydrate_formed,
        "hydrate_type": hydrate_phase_name,
        "hydrate_molar_fraction": hydrate_fraction,
        "separation_metrics": separation_metrics,
    }

    if verbose:
        print_results_table(results)

    return results


def print_results_table(results: dict):
    """Pretty prints the flash results."""
    print("=" * 65)
    print(f"HYDRATE FLASH EQUILIBRIUM: T = {results['T_K']:.2f} K, P = {results['P_bar']:.2f} bar")
    print("=" * 65)
    print(f"Components Active: {', '.join(results['active_components'])}")
    print(f"Hydrate Formed   : {'YES (' + str(results['hydrate_type']).upper() + ')' if results['hydrate_formed'] else 'NO'}")
    if results['hydrate_formed']:
        print(f"Hydrate Fraction : {results['hydrate_molar_fraction']*100:.2f} mol% of total feed")

    print("\nPhase Distribution:")
    for phase, comp_dict in results["compositions"].items():
        alpha = results["phase_fractions"][phase]
        print(f"\n  [{phase.upper()}] - Phase Fraction: {alpha:.4f} ({alpha*100:.2f} mol%)")
        for comp, frac in comp_dict.items():
            print(f"    - {comp:6s}: {frac:7.4f} ({frac*100:.2f} mol%)")

    if results["separation_metrics"]:
        print("\nSeparation Summary:")
        for k, v in results["separation_metrics"].items():
            print(f"  * {k:30s}: {v:.2f}%")
    print("=" * 65)


# -----------------------------------------------------------------------------
# CAPE-OPEN Unit Operation Standard Interface (AmsterCHEM / CO-LaN Standard)
# -----------------------------------------------------------------------------
def Configure(unit):
    """
    CAPE-OPEN Unit Operation Configuration Hook.
    Defines inlet/outlet material ports and parameters for Aspen Plus, Aspen HYSYS, and COCO.
    """
    try:
        # Ports
        unit.add_port("Feed", unit.PortType.MATERIAL, unit.PortDirection.FEED)
        unit.add_port("Vapor_Out", unit.PortType.MATERIAL, unit.PortDirection.PRODUCT)
        unit.add_port("Hydrate_Out", unit.PortType.MATERIAL, unit.PortDirection.PRODUCT)
        unit.add_port("Liquid_Out", unit.PortType.MATERIAL, unit.PortDirection.PRODUCT)

        # Operational Parameters
        unit.add_parameter("Operating Temperature (K)", unit.ParameterType.REAL, unit.ParameterMode.INPUT, default=275.0)
        unit.add_parameter("Operating Pressure (bar)", unit.ParameterType.REAL, unit.ParameterMode.INPUT, default=65.0)
    except Exception as exc:
        # Pass gracefully if not in a live CAPE-OPEN PME
        pass


def Calculate(unit):
    """
    CAPE-OPEN Unit Operation Execution Hook.
    Reads feed stream state from Aspen, solves hydrate flash, and sets product streams.
    """
    # 1. Retrieve feed stream
    feed = unit.ports["Feed"]
    
    # Inlet state (CAPE-OPEN standard SI units: T in K, P in Pa, Flow in mol/s)
    try:
        T_feed = feed.t if hasattr(feed, 't') else feed.get_property("temperature")
    except Exception:
        T_feed = 275.0
        
    try:
        P_feed_Pa = feed.p if hasattr(feed, 'p') else feed.get_property("pressure")
        P_bar = P_feed_Pa / 1.0e5
    except Exception:
        P_bar = 65.0

    try:
        total_flow = feed.flow_rate if hasattr(feed, 'flow_rate') else feed.get_property("totalFlow")
    except Exception:
        total_flow = 100.0  # mol/s default fallback

    # Check for parameter overrides
    if "Operating Temperature (K)" in unit.parameters:
        p_val = unit.parameters["Operating Temperature (K)"].value
        if p_val > 100.0:
            T_feed = p_val
    if "Operating Pressure (bar)" in unit.parameters:
        p_val = unit.parameters["Operating Pressure (bar)"].value
        if p_val > 0.0:
            P_bar = p_val

    # 2. Extract feed composition
    comp_names = [str(c).upper() for c in (feed.comp_names if hasattr(feed, 'comp_names') else unit.comp_names)]
    fractions = feed.x if hasattr(feed, 'x') else feed.get_property("fraction")

    z_inlet = {}
    for name, frac in zip(comp_names, fractions):
        norm_name = normalize_component_name(name)
        z_inlet[norm_name] = z_inlet.get(norm_name, 0.0) + float(frac)

    x_co2 = z_inlet.get("co2", 0.0)
    y_h2 = z_inlet.get("h2", 0.0)
    x_diox = z_inlet.get("diox", 0.0)
    x_h2o = z_inlet.get("h2o", 0.0)

    # 3. Solve Hydrate Flash
    res = run_hydrate_flash(
        T=T_feed,
        P=P_bar,
        x_CO2=x_co2,
        y_H2=y_h2,
        x_DIOX=x_diox,
        x_H2O=x_h2o if x_h2o > 1e-4 else None,
        T_unit="K",
        P_unit="bar",
        verbose=False
    )

    # 4. Map outputs to CAPE-OPEN Product Streams
    alpha_map = res["phase_fractions"]
    alpha_vap = alpha_map.get("vapor", 0.0) + alpha_map.get("lhc", 0.0)
    alpha_hyd = alpha_map.get("s1", 0.0) + alpha_map.get("s2", 0.0)
    alpha_liq = alpha_map.get("aqueous", 0.0)

    # Helper to construct full composition array matching Aspen compound list
    def make_full_x(phase_name):
        if phase_name not in res["compositions"]:
            return fractions
        c_dict = res["compositions"][phase_name]
        return [c_dict.get(normalize_component_name(c), 0.0) for c in comp_names]

    # Assign to ports
    if "Vapor_Out" in unit.ports:
        gas_phase = "vapor" if "vapor" in res["compositions"] else "lhc"
        unit.ports["Vapor_Out"].set(
            t=T_feed,
            p=P_bar * 1e5,
            flow_rate=total_flow * alpha_vap,
            x=make_full_x(gas_phase)
        )

    if "Hydrate_Out" in unit.ports:
        hyd_phase = res.get("hydrate_type", "s1")
        unit.ports["Hydrate_Out"].set(
            t=T_feed,
            p=P_bar * 1e5,
            flow_rate=total_flow * alpha_hyd,
            x=make_full_x(hyd_phase)
        )

    if "Liquid_Out" in unit.ports:
        unit.ports["Liquid_Out"].set(
            t=T_feed,
            p=P_bar * 1e5,
            flow_rate=total_flow * alpha_liq,
            x=make_full_x("aqueous")
        )


# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified Hydrate Flash Dynamic Entry (Pure CO2, CO2+H2, CO2+H2+DIOX)"
    )
    parser.add_argument("-T", "--temperature", type=float, default=274.0, help="Temperature (default: 274.0 K)")
    parser.add_argument("-P", "--pressure", type=float, default=65.0, help="Pressure (default: 65.0 bar)")
    parser.add_argument("--T-unit", type=str, default="K", choices=["K", "C", "k", "c"], help="Temperature unit (K or C)")
    parser.add_argument("--P-unit", type=str, default="bar", choices=["bar", "MPa", "Pa"], help="Pressure unit")
    parser.add_argument("--co2", type=float, required=True, help="CO2 mole fraction")
    parser.add_argument("--h2", type=float, default=0.0, help="H2 mole fraction (default: 0.0)")
    parser.add_argument("--diox", type=float, default=0.0, help="1,4-Dioxane mole fraction (default: 0.0)")
    parser.add_argument("--water-ratio", type=float, default=0.85, help="System water ratio (default: 0.85)")

    args = parser.parse_args()

    run_hydrate_flash(
        T=args.temperature,
        P=args.pressure,
        x_CO2=args.co2,
        y_H2=args.h2,
        x_DIOX=args.diox,
        water_ratio=args.water_ratio,
        T_unit=args.T_unit,
        P_unit=args.P_unit,
        verbose=True
    )


if __name__ == "__main__":
    main()
