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

        # Auto-detect promoter from liquid composition (e.g., {"H2O": 0.9444, "DIOX": 0.0556})
        self.promoter_name = None
        self.promoter_frac = 0.0
        if self.liq_phase_composition:
            for comp, frac in self.liq_phase_composition.items():
                if comp != "H2O" and comp not in self.eos.gases:
                    self.promoter_name = comp
                    self.promoter_frac = frac
                    break

    def _calculate_state(self, T, P, structure):
        """Helper function to calculate and return all thermodynamic properties at a given T, P."""
        if np.isnan(P) or P <= 0:
            return None

        # 1. Vapor Phase
        f_dict, phi_val = self.eos.calc_fugacities(T, P)

        # 2. Liquid Phase
        try:
            unifac = ModifiedUnifac(self.liq_phase_composition, self.database)
            H_val = unifac.calc_henry_constant("CO2", T)
            x_gas = f_dict["CO2"] / H_val
            x_w = 1.0 - x_gas

            unifac_mix = ModifiedUnifac({"H2O": x_w, "CO2": x_gas}, self.database)
            gamma_val = unifac_mix.calc_gamma(T)["H2O"]
            aw_val = x_w * gamma_val
        except Exception:
            gamma_val = 1.0
            aw_val = 1.0 - (f_dict.get("CO2", 0) / 7.35e7)

        # 3. Hydrate Phase & Occupancies
        mu_w = self.hydrate_model.chemical_potential_difference_water(
            T, P, aw_val, structure
        )
        mu_h = self.hydrate_model.chemical_potential_difference_hydrate(
            T, f_dict, structure
        )

        occ_small = self.hydrate_model.calc_cage_occupancy(
            T, f_dict, structure, "small"
        )
        occ_large = self.hydrate_model.calc_cage_occupancy(
            T, f_dict, structure, "large"
        )

        # ---------------------------------------------------------
        # 4. HYDRATE PHASE COMPOSITION & SEPARATION FACTOR
        # ---------------------------------------------------------
        nu_small = self.database.STRUCTURE_DB[structure]["small"]["nu"]
        nu_large = self.database.STRUCTURE_DB[structure]["large"]["nu"]

        hydrate_moles = {}
        total_hydrate_moles = 0.0

        for gas in self.eos.gases:
            # Moles of gas per mole of water in the hydrate lattice
            moles = nu_small * occ_small.get(gas, 0) + nu_large * occ_large.get(gas, 0)
            hydrate_moles[gas] = moles
            total_hydrate_moles += moles

        z_hydrate = {}
        for gas in self.eos.gases:
            z_hydrate[gas] = (
                hydrate_moles[gas] / total_hydrate_moles
                if total_hydrate_moles > 0
                else 0.0
            )

        # Build the final state dictionary
        state = {
            "P_eq (MPa)": P / 1e6,
            "f_CO2 (MPa)": f_dict.get("CO2", 0) / 1e6,
            "Phi_CO2": phi_val[0] if len(phi_val) > 0 else 1.0,
            "a_w": aw_val,
            "gamma_w": gamma_val,
            "Delta_Mu_w": mu_w,
            "Delta_Mu_H": mu_h,
            "Theta_Small_CO2": occ_small.get("CO2", 0),
            "Theta_Large_CO2": occ_large.get("CO2", 0),
        }

        # Dynamically append Hydrate Mole Fractions for all gases
        for gas in self.eos.gases:
            state[f"z_Hyd_{gas}"] = z_hydrate[gas]

        # Calculate Separation Factor if it's a binary gas mixture (e.g. CO2/H2)
        if len(self.eos.gases) >= 2:
            gas1, gas2 = self.eos.gases[0], self.eos.gases[1]
            y1, y2 = self.eos.y[0], self.eos.y[1]

            # Avoid division by zero
            if y1 > 0 and y2 > 0 and z_hydrate[gas2] > 0:
                sf = (z_hydrate[gas1] / y1) / (z_hydrate[gas2] / y2)
                state[f"SF_{gas1}_{gas2}"] = sf
            else:
                state[f"SF_{gas1}_{gas2}"] = np.nan

        return state

    def evaluate_structure(self, T, P_initial_guess, structure, method="newton"):
        """Runs the pressure iteration loop and returns the full thermodynamic state."""

        def objective(P):
            if P <= 0:
                return 1e6 - P  # Steer away from non-physical pressures

            f_dict, _ = self.eos.calc_fugacities(T, P)

            try:
                unifac_pure = ModifiedUnifac({"H2O": 1.0}, self.database)
                x_gas_total = sum(
                    f_dict[g] / unifac_pure.calc_henry_constant(g, T)
                    for g in f_dict.keys()
                )
                x_w = max(1.0 - x_gas_total - self.promoter_frac, 0.0)

                mix_comps = {
                    g: f_dict[g] / unifac_pure.calc_henry_constant(g, T)
                    for g in f_dict.keys()
                }
                mix_comps["H2O"] = x_w
                if self.promoter_frac > 0 and self.promoter_name:
                    mix_comps[self.promoter_name] = self.promoter_frac

                unifac_mix = ModifiedUnifac(mix_comps, self.database)
                gamma_dict = unifac_mix.calc_gamma(T)
                aw_val = x_w * gamma_dict.get("H2O", 1.0)

                if self.promoter_frac > 0 and self.promoter_name:
                    P_sat = 10000.0
                    f_dict[self.promoter_name] = (
                        self.promoter_frac
                        * gamma_dict.get(self.promoter_name, 1.0)
                        * P_sat
                    )

            except Exception:
                aw_val = max(
                    1.0
                    - sum(f_dict.get(g, 0) / 7.35e7 for g in f_dict.keys())
                    - self.promoter_frac,
                    0.0,
                )

            mu_w = self.hydrate_model.chemical_potential_difference_water(
                T, P, aw_val, structure
            )
            mu_h = self.hydrate_model.chemical_potential_difference_hydrate(
                T, f_dict, structure
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
                    objective, bracket=[1e5, 50e6], method="bisect", xtol=1.0
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
        """Compares sI and sII and returns a comprehensive list of dictionaries."""
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
            opt_struct = "None"
            opt_state = None

            if not np.isnan(P_sI) and not np.isnan(P_sII):
                opt_struct = "sI" if P_sI < P_sII else "sII"
                opt_state = state_sI if P_sI < P_sII else state_sII
            elif not np.isnan(P_sI):
                opt_struct, opt_state = "sI", state_sI
            elif not np.isnan(P_sII):
                opt_struct, opt_state = "sII", state_sII

            row = {"T (K)": T, "Optimum_Structure": opt_struct}

            if opt_state:
                row.update(opt_state)
            else:
                # Need to fill all possible columns with NaNs if the calculation fails
                keys_to_nan = [
                    "P_eq (MPa)",
                    "f_CO2 (MPa)",
                    "Phi_CO2",
                    "a_w",
                    "gamma_w",
                    "Delta_Mu_w",
                    "Delta_Mu_H",
                    "Theta_Small_CO2",
                    "Theta_Large_CO2",
                ]
                for gas in self.eos.gases:
                    keys_to_nan.append(f"z_Hyd_{gas}")
                if len(self.eos.gases) >= 2:
                    keys_to_nan.append(f"SF_{self.eos.gases[0]}_{self.eos.gases[1]}")

                row.update({k: np.nan for k in keys_to_nan})

            all_results.append(row)

        return pd.DataFrame(all_results)
