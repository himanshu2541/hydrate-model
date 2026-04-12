import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from ..water_activity_model.mod_unifac import ModifiedUnifac


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

        Partial molar volumes at infinite dilution (v_inf) are used in the
        Poynting correction for gas solubility (K&S 2000, Eq. 19):

            x_gas = f_gas / (H(T) * exp(v_inf * P / RT))

        Literature sources for v_inf:
          CO2 : 32 mL/mol  — Klauda & Sandler 2000, consistent with their eq.
          H2  : 26.1 mL/mol — Brelvi & O'Connell (AIChE J, 1972); also
                consistent with scaled-particle theory for H2 at infinite
                dilution in water near 273–300 K.

        Note: the original code used v_inf(H2) = 15 mL/mol which is too low
        (it underestimates the Poynting correction for H2 by ~40%).  The
        correct value is ~26 mL/mol.  Together with the H2 Henry's-law fix in
        database.py this produces the largest improvement to mixture results.
        """
        f_dict, phi_val = self.eos.calc_fugacities(T, P)

        try:
            unifac_pure = ModifiedUnifac({"H2O": 1.0}, self.database)
            x_gas_total = 0.0

            v_inf = {"CO2": 32.0e-6, "H2": 26.1e-6}

            for gas in list(f_dict.keys()):
                H_val_base = unifac_pure.calc_henry_constant(gas, T)
                poynting_factor = np.exp(
                    (v_inf.get(gas, 32e-6) * P) / (self.database.R * T)
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

        except Exception:
            aw_val = max(
                1.0
                - sum(f_dict.get(g, 0) / 7.35e7 for g in f_dict.keys())
                - self.promoter_frac,
                0.0,
            )
            return f_dict, phi_val, aw_val, 1.0

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

        hydrate_moles = {}
        total_hydrate_moles = 0.0
        for gas in self.eos.gases:
            moles = nu_small * occ_small.get(gas, 0) + nu_large * occ_large.get(gas, 0)
            hydrate_moles[gas] = moles
            total_hydrate_moles += moles

        z_hydrate = {
            gas: (
                hydrate_moles[gas] / total_hydrate_moles
                if total_hydrate_moles > 0
                else 0.0
            )
            for gas in self.eos.gases
        }

        # FIX: look up phi by gas name instead of hardcoding index 0 for CO2
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

        # Dynamically store fugacity and phi for each gas
        for gas in self.eos.gases:
            state[f"f_{gas} (MPa)"] = f_dict.get(gas, 0) / 1e6
            state[f"Phi_{gas}"] = phi_by_gas.get(gas, 1.0)
            state[f"Theta_Small_{gas}"] = occ_small.get(gas, 0)
            state[f"Theta_Large_{gas}"] = occ_large.get(gas, 0)
            state[f"z_Hyd_{gas}"] = z_hydrate[gas]

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
                state[f"SF_{gas1}_{gas2}"] = np.nan

        # Optional: Track the ideal separation factor for EVERY gas individually
        for i, gas in enumerate(self.eos.gases):
            y_gas = self.eos.y[i]
            if y_gas > 0:
                state[f"Ideal_SF_{gas}"] = z_hydrate[gas] / y_gas
            else:
                state[f"Ideal_SF_{gas}"] = np.nan

        return state

    def evaluate_structure(self, T, P_initial_guess, structure, method="newton"):
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
            if method == "newton":
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
            return None
        except Exception:
            return None

    def find_optimum_structure(
        self, T_range, P_initial_guess=2.5e6, solver_method="newton"
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

            print(f"T={T:.2f} K: P_sI={P_sI:.3f} MPa, P_sII={P_sII:.3f} MPa")

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
