"""
Klauda & Sandler (2000) fugacity-based hydrate equilibrium model.

Key differences from vdWP/John-Holder:
  1. Fugacity equality  f_w^H = f_w^π  (no reference chemical potential)
  2. Guest-specific vapor pressure of empty hydrate lattice (Table 6 / QL1 form)
  3. Lorentz-Berthelot combining rules for Kihara params (Eq. 13)
  4. All three shells contribute to cell potential W(r) (Eq. 12);
     shells 2 & 3 evaluated at cavity centre (flat approximation)
  5. Pressure-dependent hydrate molar volume (Eqs. 27-28)

Equilibrium condition (Appendix):
  f_w^H(T,P)  = f_w^β(T,P) · exp(−Δμ_w^H / RT)  ... hydrate side
  f_w^π(T,P)  = a_w · P_w^sat,π · exp(V_w^π(P − P_sat) / RT)  ... water side

Solver objective (compatible with existing EquilibriumSolver interface):
  chemical_potential_difference_water(T, P, a_w, struct)  → RT·ln(f_w^π)
  chemical_potential_difference_hydrate(T, P, f_dict, struct) → RT·ln(f_w^H)
  objective = mu_w − mu_H = RT·(ln f_w^π − ln f_w^H) = 0  ✓
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


class KlaudaSandlerEmpiricalModel:

    def __init__(self, database):
        self.database = database
        self.R = database.R

    # ── Langmuir constant ─────────────────────────────────────────────────

    def calc_langmuir_constant(self, T, gas, cavity_type, structure):
        """
        Returns C in m³/J.
        """
        db = self.database

        # formula Cml(T) = exp(Aml + Bml/T + Dml/T²)
        params = db.KS_EMP_LANGMUIR_PARAMS[structure][cavity_type].get(gas)
        if params is None:
            raise NotImplementedError(
                f"KlaudaSandlerEmpiricalModel has no empirical Langmuir parameters "
                f"for '{gas}' in {structure} {cavity_type} (only fitted for CO2/H2 "
                f"gas mixtures). Liquid promoters like DIOX require the Kihara-"
                f"integration KlaudaSandlerModel instead."
            )

        A, B, D = params["A"], params["B"], params["D"]
        C = np.exp(A + B / T + D / (T ** 2))
        log.debug(
            "Calculated Langmuir constant for %s in %s %s at T=%s K: %s",
            gas, structure, cavity_type, T, C,
        )
        return C

    def calc_cage_occupancy(self, T, fugacities, structure, cavity_type):
        """Standard Langmuir occupancy (vdWP Eq. 3)."""
        C = {
            g: self.calc_langmuir_constant(T, g, cavity_type, structure)
            for g in fugacities
        }
        denom = 1.0 + sum(C[g] * f for g, f in fugacities.items())
        return {g: C[g] * f / denom for g, f in fugacities.items()}

    # ── Δμ_w^H (hydrate occupancy term) ──────────────────────────────────

    def _delta_mu_hydrate_over_RT(self, T, fugacities, structure):
        """
        Δμ_w^H / RT  = −sum_m ν_m · ln(1 − sum_j θ_mj)
        Mathematically exact form: sum_m ν_m · ln(1 + sum_j C_mj f_j)
        """
        sp = self.database.STRUCTURE_DB[structure]

        # Small cavity
        C_s = {
            g: self.calc_langmuir_constant(T, g, "small", structure) for g in fugacities
        }
        sum_Cf_s = sum(C_s[g] * f for g, f in fugacities.items())

        # Large cavity
        C_l = {
            g: self.calc_langmuir_constant(T, g, "large", structure) for g in fugacities
        }
        sum_Cf_l = sum(C_l[g] * f for g, f in fugacities.items())

        # Exact logarithmic calculation to prevent float64 precision loss
        # np.log1p(x) calculates ln(1 + x) exactly, even for massive numbers
        ln_unocc_s = -np.log1p(sum_Cf_s)
        ln_unocc_l = -np.log1p(sum_Cf_l)

        return -(sp["small"]["nu"] * ln_unocc_s + sp["large"]["nu"] * ln_unocc_l)

    # ── Vapor pressure helpers (QL1 form, K&S Eq. 23) ─────────────────────

    def _ln_psat_water(self, T):
        """ln(P_sat [Pa]) for ice (T<T0) or liquid water. K&S Table 5."""
        phase = "ice" if T < self.database.T0 else "liquid"
        p = self.database.WATER_VP_PARAMS[phase]
        ln_psat = p["A"] * np.log(T) + p["B"] / T + p["C"] + p["D"] * T

        return ln_psat

    def _mixture_vp_params(self, T, fugacities, structure):
        """
        Hydrate-occupancy weighted average of K&S Table 6 parameters.
        (K&S 2003 mixing rule for mixtures).
        """
        vp_db = self.database.KS_EMP_VAPOR_PRESSURE_PARAMS[structure]

        # Optimization: if pure gas, skip occupancy math
        if len(fugacities) == 1:
            gas = next(iter(fugacities))
            return vp_db.get(gas, next(iter(vp_db.values())))

        # Calculate actual hydrate occupancies for mixing weights
        occ_s = self.calc_cage_occupancy(T, fugacities, structure, "small")
        occ_l = self.calc_cage_occupancy(T, fugacities, structure, "large")

        sp = self.database.STRUCTURE_DB[structure]
        nu_s = sp["small"]["nu"]
        nu_l = sp["large"]["nu"]

        hydrate_moles = {}
        total_moles = 0.0
        for gas in fugacities.keys():
            moles = nu_s * occ_s.get(gas, 0.0) + nu_l * occ_l.get(gas, 0.0)
            hydrate_moles[gas] = moles
            total_moles += moles

        if total_moles <= 0.0:
            gas = next(iter(fugacities))
            return vp_db.get(gas, next(iter(vp_db.values())))

        A = B = C = D = 0.0
        for gas in fugacities.keys():
            w = hydrate_moles[gas] / total_moles  # Hydrate-free basis fraction
            p = vp_db.get(gas, next(iter(vp_db.values())))
            A += w * p["A"]
            B += w * p["B"]
            C += w * p["C"]
            D += w * p["D"]

        return {"A": A, "B": B, "C": C, "D": D}

    def _ln_psat_empty_hydrate(self, T, fugacities, structure):
        """ln(P_sat^β [Pa]) using mixture-averaged QL1 parameters."""
        p = self._mixture_vp_params(T, fugacities, structure)
        ln_psat = p["A"] * np.log(T) + p["B"] / T + p["C"] + p["D"] * T
        # print(f"Calculated empty hydrate vapor pressure at T={T} K for {structure} with fugacities {fugacities}: ln(P_sat^β) = {ln_psat}")
        return ln_psat

    # ── Molar volumes ──────────────────────────────────────────────────────

    def _V_hydrate(self, T, P, structure):
        """
        Pressure-dependent molar volume of empty hydrate (m³/mol).
        K&S Eqs. 27-28.  P in Pa.
        """
        NA = self.database.NA
        P_MPa = P / 1e6
        if structure == "sI":
            Nw = 46.0
            a_sI = 11.835 + 2.217e-5 * T + 2.242e-6 * T**2
            Vt = (a_sI**3) * 1e-30 * NA / Nw
        else:  # sII
            Nw = 136.0
            a_sII = 17.13 + 2.249e-4 * T + 2.013e-6 * T**2 + 1.009e-9 * T**3
            Vt = (a_sII**3) * 1e-30 * NA / Nw

        Vc = -8.006e-9 * P_MPa + 5.448e-12 * P_MPa**2
        return Vt + Vc

    def _V_ice(self, T):
        """Ice molar volume (m³/mol).  K&S Eq. 20."""
        return 1.912e-5 + 8.387e-10 * T + 4.016e-12 * T**2

    def _V_liquid(self, T, P):
        """Liquid water molar volume (m³/mol).  K&S Eq. 21.  P in Pa."""
        P_MPa = P / 1e6
        ln_V = (
            -10.9241
            + 2.5e-4 * (T - 273.15)
            - 3.532e-4 * (P_MPa - 0.101325)
            + 1.559e-7 * (P_MPa - 0.101325) ** 2
        )
        return np.exp(ln_V)

    # ── Main fugacity calculations ─────────────────────────────────────────

    def _ln_fugacity_hydrate_water(self, T, P, fugacities, structure):
        """
        ln f_w^H(T,P)  =  ln f_w^β  −  Δμ_w^H / RT
        K&S Eq. A2.
        """
        ln_Psat_b = self._ln_psat_empty_hydrate(T, fugacities, structure)
        Psat_b = np.exp(ln_Psat_b)
        # print(f"Empty hydrate vapor pressure at T={T} K: P_sat = {Psat_b} Pa")
        V_b = self._V_hydrate(T, P, structure)
        # Fugacity of empty lattice (Poynting)
        ln_f_beta = ln_Psat_b + V_b * (P - Psat_b) / (self.R * T)
        # Subtract Δμ_w^H / RT  (positive value → reduces fugacity)
        dmu = self._delta_mu_hydrate_over_RT(T, fugacities, structure)
        ln_f_w_H = ln_f_beta - dmu
        # print(
        #     f"Hydrate-side water fugacity at T={T} K, P={P} Pa: f_w^H = {np.exp(ln_f_w_H)} Pa"
        # )
        return ln_f_w_H

    def _ln_fugacity_water_phase(self, T, P, a_w):
        """
        ln f_w^π(T,P)  for ice or liquid water.
        K&S Eqs. A3-A4.
        """
        ln_Psat_w = self._ln_psat_water(T)
        Psat_w = np.exp(ln_Psat_w)
        # print(f"Water vapor pressure at T={T} K: P_sat = {Psat_w} Pa")
        if T < self.database.T0:
            V_w = self._V_ice(T)
            ln_fw = ln_Psat_w + V_w * (P - Psat_w) / (self.R * T)
        else:
            V_w = self._V_liquid(T, P)
            ln_fw = (
                np.log(max(a_w, 1e-15)) + ln_Psat_w + V_w * (P - Psat_w) / (self.R * T)
            )
        # print(
        #     f"Water-side water fugacity at T={T} K, P={P} Pa, a_w={a_w}: f_w^π = {np.exp(ln_fw)} Pa"
        # )
        return ln_fw

    # ── Solver-compatible interface ────────────────────────────────────────
    # The existing EquilibriumSolver evaluates:
    #   objective = mu_w − mu_H = 0
    # We return RT·ln(f) for each side so the difference = RT·ln(f_w^π/f_w^H).

    def chemical_potential_difference_water(self, T, P, a_w, structure):
        """Returns RT · ln(f_w^π) — used as 'mu_w' in solver objective."""
        mu_W = self.R * T * self._ln_fugacity_water_phase(T, P, a_w)
        # print(
        #     f"Calculated water chemical potential difference (RT·ln(f_w^π)) at T={T} K, P={P} Pa, a_w={a_w}: {mu_W} J/mol"
        # )
        return mu_W

    def chemical_potential_difference_hydrate(self, T, fugacities, structure, P=None):
        """
        Returns RT · ln(f_w^H) — used as 'mu_H' in solver objective.
        P must be supplied for the Poynting correction.
        Falls back to zero Poynting if P is None (use only for diagnostics).
        """
        if P is None:
            # Rough estimate: ignore Poynting (diagnostic only)
            P = 1e6

        mu_H = self.R * T * self._ln_fugacity_hydrate_water(T, P, fugacities, structure)
        # print(
        #     f"Calculated hydrate chemical potential difference (RT·ln(f_w^H)) at T={T} K, P={P} Pa: {mu_H} J/mol"
        # )
        return mu_H
