import logging

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from ..water_activity_model.mod_unifac import ModifiedUnifac
from ..core.exceptions import HydrateModelError, WaterActivityError, ConvergenceError
from ..core.fitted_params import FITTED_PARAMS

log = logging.getLogger(__name__)

# Infinite-dilution partial molar volumes for the Poynting correction
# (K&S 2000 eq. 19). Neither is an in-house regression -- see
# core/fitted_params.py "v_inf_CO2"/"v_inf_H2" for provenance.
_V_INF = {
    "CO2": FITTED_PARAMS["v_inf_CO2"].value,
    "H2": FITTED_PARAMS["v_inf_H2"].value,
}


class EquilibriumSolver:
    def __init__(self, liq_phase_composition, database, hydrate_model, eos_model):
        self.database = database
        self.hydrate_model = hydrate_model
        self.eos = eos_model
        self.liq_phase_composition = liq_phase_composition

        # Auto-detect promoter from liquid composition
        self.promoter_name = None
        self.promoter_frac = 0.0
        if self.liq_phase_composition:
            for comp, frac in self.liq_phase_composition.items():
                if comp != "H2O" and comp not in self.eos.gases:
                    self.promoter_name = comp
                    self.promoter_frac = frac
                    break

    def _get_liquid_and_fugacities(self, T, P):
        """Centralized method to calculate fugacities and liquid water activity.

        Partial molar volumes at infinite dilution (v_inf, see _V_INF above)
        are used in the Poynting correction for gas solubility (K&S 2000,
        eq. 19): x_gas = f_gas / (H(T) * exp(v_inf * P / RT)).
        """
        f_dict, phi_val = self.eos.calc_fugacities(T, P)

        try:
            unifac_pure = ModifiedUnifac({"H2O": 1.0}, self.database)
            x_gas_total = 0.0

            for gas in list(f_dict.keys()):
                H_val_base = unifac_pure.calc_henry_constant(gas, T)
                poynting_factor = np.exp(
                    (_V_INF.get(gas, 32e-6) * P) / (self.database.R * T)
                )
                x_gas = f_dict[gas] / (H_val_base * poynting_factor)
                x_gas_total += x_gas

            # Calculate true mole fraction of water in the total liquid
            x_w = max(1.0 - x_gas_total - self.promoter_frac, 0.0)

            # Isolate the solvent (Water + Promoter) for UNIFAC
            solvent_total = x_w + self.promoter_frac
            unifac_comps = {}
            if solvent_total > 0:
                unifac_comps["H2O"] = x_w / solvent_total
                if self.promoter_frac > 0 and self.promoter_name:
                    unifac_comps[self.promoter_name] = (
                        self.promoter_frac / solvent_total
                    )
            else:
                unifac_comps["H2O"] = 1.0

            # Calculate activity coefficient using ONLY the normalized solvent matrix
            unifac_mix = ModifiedUnifac(unifac_comps, self.database)
            gamma_dict = unifac_mix.calc_gamma(T)

            # The activity of water is the true mole fraction * activity coefficient
            aw_val = x_w * gamma_dict.get("H2O", 1.0)

            promoter_data = self.database.GUEST_DB.get(self.promoter_name, {})
            if self.promoter_frac > 0 and self.promoter_name:
                delta_H_vap = promoter_data.get("delta_H_vap", 34700.0)
                P_sat = promoter_data.get("P_sat_ref", 9300.0) * np.exp(
                    (delta_H_vap / self.database.R) * (1 / promoter_data.get("T_sat_ref", 293.15) - 1 / T)
                )
                f_dict[self.promoter_name] = (
                    self.promoter_frac * gamma_dict.get(self.promoter_name, 1.0) * P_sat
                )

            return f_dict, phi_val, aw_val, gamma_dict.get("H2O", 1.0)

        except Exception as exc:
            raise WaterActivityError(
                f"UNIFAC water-activity calculation failed at T={T} K, P={P} Pa: {exc}"
            ) from exc

    def _calculate_state(self, T, P, structure):
        """Calculate all thermodynamic properties at a given T, P."""
        if np.isnan(P) or P <= 0:
            return None

        f_dict, phi_val, aw_val, gamma_val = self._get_liquid_and_fugacities(T, P)

        mu_w = self.hydrate_model.chemical_potential_difference_water(
            T, P, aw_val, structure
        )
        mu_h = self.hydrate_model.chemical_potential_difference_hydrate(
            T, f_dict, structure, P=P
        )

        occ_small = self.hydrate_model.calc_cage_occupancy(
            T, f_dict, structure, "small"
        )
        occ_large = self.hydrate_model.calc_cage_occupancy(
            T, f_dict, structure, "large"
        )

        nu_small = self.database.STRUCTURE_DB[structure]["small"]["nu"]
        nu_large = self.database.STRUCTURE_DB[structure]["large"]["nu"]

        all_guests = list(self.eos.gases)
        if self.promoter_name and self.promoter_name not in all_guests:
            all_guests.append(self.promoter_name)

        hydrate_moles = {}
        total_hydrate_moles = 0.0
        
        # Iterate over all guests (including DIOX) to find the correct hydrate fractions
        for guest in all_guests:
            moles = nu_small * occ_small.get(guest, 0) + nu_large * occ_large.get(guest, 0)
            hydrate_moles[guest] = moles
            total_hydrate_moles += moles

        z_hydrate = {
            guest: (
                hydrate_moles[guest] / total_hydrate_moles
                if total_hydrate_moles > 0
                else 0.0
            )
            for guest in all_guests
        }

        phi_by_gas = {gas: phi_val[i] for i, gas in enumerate(self.eos.gases)}

        try:
            Z_val = self.eos.calc_Z(T, P)
        except Exception:
            Z_val = float("nan")

        state = {
            "P_eq (MPa)": P / 1e6,
            "Z": Z_val,
            "a_w": aw_val,
            "gamma_w": gamma_val,
            "Delta_Mu_w": mu_w,
            "Delta_Mu_H": mu_h,
        }

        for guest in all_guests:
            state[f"f_{guest} (MPa)"] = f_dict.get(guest, 0) / 1e6
            
            # Fugacity coefficients (Phi) are only tracked for EOS gases. 
            # We set NaN for liquid promoters like DIOX.
            state[f"Phi_{guest}"] = phi_by_gas.get(guest, "nan")
            
            state[f"Theta_Small_{guest}"] = occ_small.get(guest, 0)
            state[f"Theta_Large_{guest}"] = occ_large.get(guest, 0)
            state[f"z_Hyd_{guest}"] = z_hydrate.get(guest, 0)

        # Calculate Ideal Separation Factor (Enrichment Ratio)
        if len(self.eos.gases) >= 2:
            gas1, gas2 = self.eos.gases[0], self.eos.gases[1]
            y1 = self.eos.y[
                0
            ]  # Mole fraction of primary gas (e.g., CO2) in the vapor phase

            if y1 > 0:
                # Defined as: y_gas_hydrate / y_gas_vapor
                state[f"SF_{gas1}_{gas2}"] = z_hydrate[gas1] / y1
            else:
                state[f"SF_{gas1}_{gas2}"] = "nan"

        # Optional: Track the ideal separation factor for EVERY gas individually
        for i, gas in enumerate(self.eos.gases):
            y_gas = self.eos.y[i]
            if y_gas > 0:
                state[f"Ideal_SF_{gas}"] = z_hydrate[gas] / y_gas
            else:
                state[f"Ideal_SF_{gas}"] = "nan"

        return state

    def _find_bracket(self, objective, P_initial_guess, P_max=100e6):
        """Expand outward from P_initial_guess until objective changes sign.

        root_scalar's derivative-free methods (newton-as-secant, in
        particular) are being asked to bracket a piecewise-smooth objective
        (it contains `quad` calls with a hard wall at R - a); brentq needs an
        explicit sign-changing bracket instead. Start at
        [0.5*P_guess, 2*P_guess] and double outward until P_max.
        """
        lo, hi = 0.5 * P_initial_guess, 2.0 * P_initial_guess
        f_lo, f_hi = objective(lo), objective(hi)
        while f_lo * f_hi > 0 and hi < P_max:
            lo /= 2.0
            hi *= 2.0
            f_lo, f_hi = objective(lo), objective(hi)
        if f_lo * f_hi > 0:
            return None
        return lo, hi

    def evaluate_structure(self, T, P_initial_guess, structure, method="brentq"):
        """
        Runs the pressure iteration loop and returns the full thermodynamic state.
        FIX: objective now calls _get_liquid_and_fugacities to stay consistent
        (includes Poynting correction and temperature-dependent P_sat for promoter).
        """

        def objective(P):
            if P <= 0:
                return 1e6 - P

            f_dict, _, aw_val, _ = self._get_liquid_and_fugacities(T, P)

            mu_w = self.hydrate_model.chemical_potential_difference_water(
                T, P, aw_val, structure
            )
            mu_h = self.hydrate_model.chemical_potential_difference_hydrate(
                T, f_dict, structure, P=P
            )
            return mu_w - mu_h

        try:
            if method == "brentq":
                bracket = self._find_bracket(objective, P_initial_guess)
                if bracket is None:
                    raise ConvergenceError(
                        f"No sign change found for {structure} at T={T} K within "
                        f"P in [0, 100 MPa] starting from guess {P_initial_guess} Pa."
                    )
                sol = root_scalar(objective, bracket=list(bracket), method="brentq")
            elif method == "newton":
                sol = root_scalar(
                    objective, x0=P_initial_guess, method="newton", maxiter=50
                )
            elif method == "secant":
                sol = root_scalar(
                    objective,
                    x0=P_initial_guess,
                    x1=P_initial_guess * 1.1,
                    method="secant",
                    maxiter=50,
                )
            elif method == "bisect":
                sol = root_scalar(
                    objective, bracket=[1, 100e6], method="bisect", xtol=1.0
                )
            else:
                raise ValueError(f"Unknown solver method: {method}")

            if sol.converged:
                return self._calculate_state(T, sol.root, structure)
            log.debug(
                "Root-find did not converge for %s at T=%s K (method=%s).",
                structure, T, method,
            )
            return None
        except HydrateModelError as e:
            log.debug(
                "%s at T=%s K, structure=%s: %s", type(e).__name__, T, structure, e
            )
            return None
        except (ValueError, RuntimeError) as e:
            log.debug(
                "Root-find failed for %s at T=%s K (method=%s): %s",
                structure, T, method, e,
            )
            return None

    def find_optimum_structure(
        self, T_range, P_initial_guess=2.5e6, solver_method="brentq"
    ):
        """Compares sI and sII and returns a DataFrame of results."""
        all_results = []

        for T in T_range:
            state_sI = self.evaluate_structure(
                T, P_initial_guess, "sI", method=solver_method
            )
            state_sII = self.evaluate_structure(
                T, P_initial_guess, "sII", method=solver_method
            )

            P_sI = state_sI["P_eq (MPa)"] if state_sI else np.nan
            P_sII = state_sII["P_eq (MPa)"] if state_sII else np.nan

            log.info("T=%.2f K: P_sI=%.3f MPa, P_sII=%.3f MPa", T, P_sI, P_sII)

            opt_struct = None
            opt_state = None

            if not np.isnan(P_sI) and not np.isnan(P_sII):
                opt_struct = "sI" if P_sI < P_sII else "sII"
                opt_state = state_sI if P_sI < P_sII else state_sII
            elif not np.isnan(P_sI):
                opt_struct, opt_state = "sI", state_sI
            elif not np.isnan(P_sII):
                opt_struct, opt_state = "sII", state_sII

            row = {
                "T (K)": T,
                "Optimum_Structure": opt_struct if opt_struct else "None",
            }

            if opt_state:
                row.update(opt_state)
            else:
                # Build NaN placeholders from first available state structure
                template_state = self._calculate_state(T, P_initial_guess, "sI")
                if template_state:
                    row.update({k: np.nan for k in template_state})

            all_results.append(row)

        return pd.DataFrame(all_results)
