"""
example_h2_and_diox.py
-----------------------
Demonstrates the extended capabilities of hydrateflash with:
1. H2 (hydrogen) gas component (parameterized from literature & hydrate_project)
2. 1,4-Dioxane (liquid thermodynamic promoter forming Structure II hydrate)
3. Multi-phase flash calculation for CO2 + H2 + H2O (syngas / pre-combustion capture)
4. Multi-phase flash calculation for CH4 + 1,4-Dioxane + H2O (Structure II promoter)
"""

import numpy as np
import component_properties as cp
import flashalgorithm as fc


def test_component_properties():
    print("=" * 60)
    print("1. Component Properties Verification")
    print("=" * 60)
    comps = ["h2o", "co2", "h2", "diox"]
    for name in comps:
        c = cp.Component(name)
        print(f"[{c.compname:5s}] Tc = {c.Tc:6.2f} K | Pc = {c.Pc:6.2f} bar | MW = {c.MW:6.2f} g/mol | diam = {c.diam:.2f} Å")
        print(f"        Kihara: a = {c.HvdWPM['kih']['a']}, sigma = {c.HvdWPM['kih']['sig']} Å, eps/k = {c.HvdWPM['kih']['epsk']} K")
        print(f"        SRK omega = {c.SRK['omega']}, kij(co2) = {c.SRK['kij'].get('co2', 0.0)}")


def test_co2_h2_flash():
    print("\n" + "=" * 60)
    print("2. Flash Calculation: CO2 + H2 (40:60) + Water")
    print("=" * 60)
    flash = fc.FlashController(components=["water", "co2", "h2"])
    print(f"Phases considered: {flash.phases}")
    
    # 80 mol% water, 8 mol% CO2, 12 mol% H2 (CO2:H2 = 40:60 gas ratio)
    T = 275.0  # K (~1.85 °C)
    P = 65.0   # bar (6.5 MPa)
    z = np.array([0.80, 0.08, 0.12])
    
    flash.main_handler(flash.compobjs, z=z, T=T, P=P)
    print(f"\nConditions: T = {T} K, P = {P} bar")
    print("Stable phases and equilibrium compositions:")
    for i, phase in enumerate(flash.phases):
        if flash.alpha_calc[i] > 1e-5:
            print(f"\n  Phase: {phase.upper()} (Fraction = {flash.alpha_calc[i]:.4f})")
            for j, comp in enumerate(flash.compobjs):
                print(f"    - {comp.compname:5s}: {flash.x_calc[j, i]:.4f} ({flash.x_calc[j, i]*100:.2f} mol%)")


def test_dioxane_promoter_flash():
    print("\n" + "=" * 60)
    print("3. Flash Calculation: CH4 + 1,4-Dioxane (sII Promoter) + Water")
    print("=" * 60)
    flash = fc.FlashController(components=["water", "methane", "diox"])
    print(f"Phases considered: {flash.phases}")
    
    # Water + Methane + 1,4-Dioxane promoter
    T = 278.0  # K (~4.85 °C)
    P = 20.0   # bar (2.0 MPa)
    z = np.array([0.90, 0.05, 0.05])
    
    flash.main_handler(flash.compobjs, z=z, T=T, P=P)
    print(f"\nConditions: T = {T} K, P = {P} bar")
    print("Stable phases and equilibrium compositions:")
    for i, phase in enumerate(flash.phases):
        if flash.alpha_calc[i] > 1e-5:
            print(f"\n  Phase: {phase.upper()} (Fraction = {flash.alpha_calc[i]:.4f})")
            for j, comp in enumerate(flash.compobjs):
                print(f"    - {comp.compname:5s}: {flash.x_calc[j, i]:.4f} ({flash.x_calc[j, i]*100:.2f} mol%)")


if __name__ == "__main__":
    test_component_properties()
    test_co2_h2_flash()
    test_dioxane_promoter_flash()
