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
from scipy.integrate import quad

from ..core import correlations as corr
from ..core.exceptions import LangmuirIntegrationError
from ..core.fitted_params import FITTED_PARAMS

log = logging.getLogger(__name__)

# Kihara params for guests missing from KS_KIHARA_PARAMS/GAS_DB entirely.
# See core/fitted_params.py "dioxane_kihara" for provenance (undocumented).
_FENCED_KIHARA = {"DIOX": FITTED_PARAMS["dioxane_kihara"].value}


class KlaudaSandlerModel:

    def __init__(self, database):
        self.database = database
        self.R = database.R
        self.kihara_params = {}
        self._warned_proxies: set[str] = set()

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

        if str(gas + structure) not in self.kihara_params:
            # Guest: from the K&S-specific Tee et al. table, then the fenced
            # layer for guests K&S never parameterised (e.g. DIOX), then
            # GAS_DB as a last resort.
            ks = db.KS_KIHARA_PARAMS
            gp = ks.get(gas) or _FENCED_KIHARA.get(gas) or db.GUEST_DB[gas]
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
            log.debug(
                "Combined Kihara params for %s in %s: a=%s m, sigma=%s m, eps=%s J",
                gas, structure, a, sigma, eps,
            )
            self.kihara_params[str(gas + structure)] = (a, sigma, eps)
            return a, sigma, eps
        else:
            return self.kihara_params[str(gas + structure)]

    # ── Kihara cell potential ──────────────────────────────────────────────

    def _kihara_shell_potential(self, r, sigma, eps, a, R_shell, z):
        """
        Spherically-averaged Kihara cell potential for one shell.
        K&S Eqs. 6-7.  Returns potential in Joules.
        """
        if r >= (R_shell - a):
            log.debug(
                "r=%s m exceeds shell radius minus core (R_shell - a = %s m); "
                "returning large potential.",
                r, R_shell - a,
            )
            return 1e50

        def delta(N):
            x = r / R_shell
            y = a / R_shell
            return (1.0 / N) * ((1 - x - y) ** (-N) - (1 + x - y) ** (-N))

        s12 = sigma**12
        s6 = sigma**6
        term_rep = (s12 / (R_shell**11 * r)) * (
            delta(10) + (a / R_shell) * delta(11)
        )  # unitless
        term_att = (s6 / (R_shell**5 * r)) * (
            delta(4) + (a / R_shell) * delta(5)
        )  # unitless

        w_r = 2 * z * eps * (term_rep - term_att)  # in Joules
        return w_r

    # ── Langmuir constant ─────────────────────────────────────────────────

    def calc_langmuir_constant(self, T, gas, cavity_type, structure):
        """
        K&S Eq. 4 with multi-shell W(r) from Eq. 12.
        Shells 2 & 3: evaluated at r = 0 (flat over integration range).
        Returns C in m³/J.
        """
        # Large liquid promoters (DIOX, THF, CP) physically cannot fit inside the 
        # squashed sI large cavity. We must bypass the Kihara spherical illusion.
        sII_only_promoters = ["DIOX", "THF", "CP", "C5H10"] 
        if structure == "sI" and gas in sII_only_promoters:
            return 0.0

        db = self.database
        ANGSTROM = 1e-10
        struct_props = db.STRUCTURE_DB[structure][cavity_type]
        a, sigma, eps = self._get_combined_kihara(gas, structure)  # returns in SI units

        R1 = struct_props["shells"]["1"]["R"] * ANGSTROM  # convert to meters
        limit = R1 - a  # in meters

        def integrand(r):
            w_total = 0.0
            for shell in struct_props["shells"].values():
                R_sh = shell["R"] * ANGSTROM  # convert to meters
                z_sh = shell["z"]
                w_total += self._kihara_shell_potential(
                    r, sigma, eps, a, R_sh, z_sh
                )  # in Joules

            return np.exp(-w_total / (db.KB * T)) * r * r

        try:
            integral, _ = quad(integrand, 1e-12, limit)
        except Exception as exc:
            raise LangmuirIntegrationError(
                f"Kihara quadrature failed for {gas} in {structure} {cavity_type} "
                f"at T={T} K: {exc}"
            ) from exc
        C = (4 * np.pi / (db.KB * T)) * integral  # in m³/J

        log.debug(
            "Calculated Langmuir constant for %s in %s with %s: C = %s",
            gas, structure, cavity_type, C,
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
        """ln(P_sat [Pa]) for ice (T<T0) or liquid water. K&S eqs. 7c/7d."""
        p_sat = corr.P_sat_ice(T) if T < self.database.T0 else corr.P_sat_liquid_water(T)
        return np.log(p_sat)

    def _mixture_vp_params(self, T, fugacities, structure):
        """
        Hydrate-occupancy weighted average of K&S Table 6 parameters.
        (K&S 2003 mixing rule for mixtures).
        """
        vp_db = self.database.KS_VAPOR_PRESSURE_PARAMS[structure]

        sII_promoters = ["DIOX", "THF", "CP", "C5H10"]
        heavy_fallback = "C3H8" if "C3H8" in vp_db else next(iter(vp_db.keys()))

        for gas in fugacities:
            if gas in sII_promoters and structure == "sII":
                # Bypass the mixing rule completely. The lattice belongs to the promoter.
                if gas not in self._warned_proxies:
                    self._warned_proxies.add(gas)
                    log.warning(
                        "No K&S empty-hydrate vapor-pressure parameters for %s in "
                        "%s; substituting %s's parameters as a proxy. This is not "
                        "physically validated for %s and results may be inaccurate.",
                        gas, structure, heavy_fallback, gas,
                    )
                return vp_db[heavy_fallback]

        # Optimization: if pure gas, skip occupancy math
        if len(fugacities) == 1:
            gas = next(iter(fugacities))
            return vp_db.get(gas, next(iter(vp_db.values())))

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
        
        # Determine a stable fallback for unparameterized heavy promoters
        # Propane (C3H8) is the standard reference sII former in K&S tables.
        heavy_fallback = "C3H8" if "C3H8" in vp_db else next(iter(vp_db.keys()))

        for gas in fugacities.keys():
            w = hydrate_moles[gas] / total_moles  # Hydrate-free basis fraction
            
            if gas not in vp_db and gas not in self.database.GUEST_DB:
                # If it's a liquid promoter not in the K&S gas table, use heavy fallback
                p = vp_db[heavy_fallback]
            else:
                p = vp_db.get(gas, vp_db[heavy_fallback])
                
            A += w * p["A"]
            B += w * p["B"]
            C += w * p["C"]
            D += w * p["D"]

        return {"A": A, "B": B, "C": C, "D": D}

    def _ln_psat_empty_hydrate(self, T, fugacities, structure):
        """ln(P_sat^β [Pa]) using mixture-averaged QL1 parameters."""
        p = self._mixture_vp_params(T, fugacities, structure)
        ln_psat = p["A"] * np.log(T) + p["B"] / T + p["C"] + p["D"] * T
        log.debug(
            "Calculated empty hydrate vapor pressure at T=%s K for %s with "
            "fugacities %s: ln(P_sat^beta) = %s",
            T, structure, fugacities, ln_psat,
        )
        return ln_psat

    # ── Molar volumes ──────────────────────────────────────────────────────

    def _V_hydrate(self, T, P, structure):
        """Pressure-dependent molar volume of empty hydrate (m^3/mol). P in Pa."""
        return corr.V_empty_hydrate(T, P / 1e6, structure)

    def _V_ice(self, T):
        """Ice molar volume (m^3/mol)."""
        return corr.V_ice(T)

    def _V_liquid(self, T, P):
        """Liquid water molar volume (m^3/mol). P in Pa."""
        return corr.V_liquid_water(T, P / 1e6)

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
