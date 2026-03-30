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

import numpy as np
from scipy.integrate import quad


class KlaudaSandlerModel:

    def __init__(self, database):
        self.database = database
        self.R = database.R
        self.kihara_params = {}
    # ── Kihara cell potential ──────────────────────────────────────────────

    def _kihara_shell_potential(self, r, sigma, eps, a, R_shell, z):
        """
        Spherically-averaged Kihara cell potential for one shell.
        K&S Eqs. 6-7.  Returns potential in Joules.
        """
        if r >= (R_shell - a):
            return 1e50
        if r < 1e-12:
            r = 1e-12

        def delta(N):
            x = r / R_shell
            y = a / R_shell
            return (1.0 / N) * ((1 - x - y) ** (-N) - (1 + x - y) ** (-N))

        s12 = sigma**12
        s6 = sigma**6
        term_rep = (s12 / (R_shell**11 * r)) * (delta(10) + (a / R_shell) * delta(11))
        term_att = (s6 / (R_shell**5 * r)) * (delta(4) + (a / R_shell) * delta(5))
        return 2 * z * eps * (term_rep - term_att)

    def _get_combined_kihara(self, gas, structure):
        """
        Lorentz-Berthelot combining rules (K&S Eq. 13):
          σ = (σ_g + σ_w) / 2
          ε = sqrt(ε_g · ε_w)
          a = (a_g + a_w) / 2
        Uses KS_KIHARA_PARAMS (Tee et al. 1966) — NOT the fitted GAS_DB values.
        """
        ANGSTROM = 1e-10
        db = self.database

        if gas+structure not in self.kihara_params:
          # Guest: from the K&S-specific Tee et al. table
          ks = db.KS_KIHARA_PARAMS
          gp = ks.get(gas, db.GAS_DB[gas])  # fall back to GAS_DB if missing
          wp = ks["H2O"]

          a_g = gp["a"] * ANGSTROM
          s_g = gp["sigma"] * ANGSTROM
          e_g = gp["eps_k"] * db.KB

          a_w = wp["a"] * ANGSTROM
          s_w = wp["sigma"] * ANGSTROM
          e_w = wp["eps_k"] * db.KB

          a = (a_g + a_w) / 2.0
          sigma = (s_g + s_w) / 2.0
          eps = np.sqrt(e_g * e_w)
          print(f"Combined Kihara params for {gas} in {structure}: a={a} m, σ={sigma} m, ε={eps} J")
          self.kihara_params[gas+structure] = (a, sigma, eps)
          return a, sigma, eps
        else:
          return self.kihara_params[gas+structure]

    # ── Langmuir constant ─────────────────────────────────────────────────

    def calc_langmuir_constant(self, T, gas, cavity_type, structure):
        """
        K&S Eq. 4 with multi-shell W(r) from Eq. 12.
        Shells 2 & 3: evaluated at r = 0 (flat over integration range).
        Returns C in m³/J.
        """
        db = self.database
        ANGSTROM = 1e-10
        a, sigma, eps = self._get_combined_kihara(gas, structure)

        shells = db.STRUCTURE_DB[structure][cavity_type]["shells"]
        R1 = shells["1"]["R"] * ANGSTROM
        limit = R1 - a - 1e-12

        # Pre-compute outer shell contributions at r→0 (centre)
        W_outer = 0.0
        for key in ["2", "3"]:
            if key in shells:
                R_sh = shells[key]["R"] * ANGSTROM
                z_sh = shells[key]["z"]
                W_outer += self._kihara_shell_potential(
                    1e-12, sigma, eps, a, R_sh, z_sh
                )

        def integrand(r):
            R1_sh = shells["1"]["R"] * ANGSTROM
            z1 = shells["1"]["z"]
            W1 = self._kihara_shell_potential(r, sigma, eps, a, R1_sh, z1)
            W = W1 + W_outer
            if W > 100 * db.KB * T:
                return 0.0
            return np.exp(-W / (db.KB * T)) * r * r

        try:
            val, _ = quad(integrand, 0.0, limit, limit=200)
            C = (4.0 * np.pi / (db.KB * T)) * val
        except Exception:
            C = 0.0
        return max(C, 0.0)

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
        K&S Eq. 2.  Returns the *positive* value.
        """
        sp = self.database.STRUCTURE_DB[structure]
        occ_s = self.calc_cage_occupancy(T, fugacities, structure, "small")
        occ_l = self.calc_cage_occupancy(T, fugacities, structure, "large")
        ts = max(1.0 - sum(occ_s.values()), 1e-15)
        tl = max(1.0 - sum(occ_l.values()), 1e-15)
        return -(sp["small"]["nu"] * np.log(ts) + sp["large"]["nu"] * np.log(tl))

    # ── Vapor pressure helpers (QL1 form, K&S Eq. 23) ─────────────────────

    def _ln_psat_water(self, T):
        """ln(P_sat [Pa]) for ice (T<T0) or liquid water. K&S Table 5."""
        phase = "ice" if T < self.database.T0 else "liquid"
        p = self.database.WATER_VP_PARAMS[phase]
        return p["A"] * np.log(T) + p["B"] / T + p["C"] + p["D"] * T

    def _mixture_vp_params(self, fugacities, structure):
        """
        Fugacity-weighted average of K&S Table 6 parameters.
        For pure systems this is exact; for mixtures it is the recommended
        first-order mixing rule (K&S 2003 approach).
        """
        vp_db = self.database.KS_VAPOR_PRESSURE_PARAMS[structure]
        total = sum(fugacities.values())
        if total <= 0.0:
            gas = next(iter(fugacities))
            return vp_db.get(gas, next(iter(vp_db.values())))

        A = B = C = D = 0.0
        for gas, f in fugacities.items():
            w = f / total
            p = vp_db.get(gas, next(iter(vp_db.values())))
            A += w * p["A"]
            B += w * p["B"]
            C += w * p["C"]
            D += w * p["D"]
        return {"A": A, "B": B, "C": C, "D": D}

    def _ln_psat_empty_hydrate(self, T, fugacities, structure):
        """ln(P_sat^β [Pa]) using mixture-averaged QL1 parameters."""
        p = self._mixture_vp_params(fugacities, structure)
        return p["A"] * np.log(T) + p["B"] / T + p["C"] + p["D"] * T

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
            Vt = (11.835 + 2.217e-5 * T + 2.242e-6 * T**2) * 1e-30 * NA / Nw
        else:  # sII
            Nw = 136.0
            Vt = (
                (17.13 + 2.249e-4 * T + 2.013e-6 * T**2 + 1.009e-9 * T**3)
                * 1e-30
                * NA
                / Nw
            )
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
        V_b = self._V_hydrate(T, P, structure)
        # Fugacity of empty lattice (Poynting)
        ln_f_beta = ln_Psat_b + V_b * (P - Psat_b) / (self.R * T)
        # Subtract Δμ_w^H / RT  (positive value → reduces fugacity)
        dmu = self._delta_mu_hydrate_over_RT(T, fugacities, structure)
        return ln_f_beta - dmu

    def _ln_fugacity_water_phase(self, T, P, a_w):
        """
        ln f_w^π(T,P)  for ice or liquid water.
        K&S Eqs. A3-A4.
        """
        ln_Psat_w = self._ln_psat_water(T)
        Psat_w = np.exp(ln_Psat_w)
        if T < self.database.T0:
            V_w = self._V_ice(T)
            ln_fw = ln_Psat_w + V_w * (P - Psat_w) / (self.R * T)
        else:
            V_w = self._V_liquid(T, P)
            ln_fw = (
                np.log(max(a_w, 1e-15)) + ln_Psat_w + V_w * (P - Psat_w) / (self.R * T)
            )
        return ln_fw

    # ── Solver-compatible interface ────────────────────────────────────────
    # The existing EquilibriumSolver evaluates:
    #   objective = mu_w − mu_H = 0
    # We return RT·ln(f) for each side so the difference = RT·ln(f_w^π/f_w^H).

    def chemical_potential_difference_water(self, T, P, a_w, structure):
        """Returns RT · ln(f_w^π) — used as 'mu_w' in solver objective."""
        return self.R * T * self._ln_fugacity_water_phase(T, P, a_w)

    def chemical_potential_difference_hydrate(self, T, fugacities, structure, P=None):
        """
        Returns RT · ln(f_w^H) — used as 'mu_H' in solver objective.
        P must be supplied for the Poynting correction.
        Falls back to zero Poynting if P is None (use only for diagnostics).
        """
        if P is None:
            # Rough estimate: ignore Poynting (diagnostic only)
            P = 1e6
        return self.R * T * self._ln_fugacity_hydrate_water(T, P, fugacities, structure)

    # ── Compatibility shims (called by _calculate_state) ──────────────────

    def calc_cage_occupancy(self, T, fugacities, structure, cavity_type):
        """Already defined above — exposed for EquilibriumSolver._calculate_state."""
        C = {
            g: self.calc_langmuir_constant(T, g, cavity_type, structure)
            for g in fugacities
        }
        denom = 1.0 + sum(C[g] * f for g, f in fugacities.items())
        return {g: C[g] * f / denom for g, f in fugacities.items()}
