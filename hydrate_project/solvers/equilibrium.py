import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from ..water_activity_model.mod_unifac import ModifiedUnifac

class EquilibriumSolver:
    def __init__(self, database, hydrate_model, eos_model):
        self.database = database
        self.hydrate_model = hydrate_model
        self.eos = eos_model

    def _calculate_state(self, T, P, structure):
        """Helper function to calculate and return all thermodynamic properties at a given T, P."""
        if np.isnan(P):
            return None

        # 1. Vapor Phase
        f_dict, phi_val = self.eos.calc_fugacities(T, P)

        # 2. Liquid Phase
        try:
            unifac = ModifiedUnifac({"H2O": 1.0, "CO2": 0.0}, self.database)
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
        mu_w = self.hydrate_model.chemical_potential_difference_water(T, P, aw_val, structure)
        mu_h = self.hydrate_model.chemical_potential_difference_hydrate(T, f_dict, structure)
        
        occ_small = self.hydrate_model.calc_cage_occupancy(T, f_dict, structure, "small")
        occ_large = self.hydrate_model.calc_cage_occupancy(T, f_dict, structure, "large")

        return {
            "P_eq (MPa)": P / 1e6,
            "f_CO2 (MPa)": f_dict.get("CO2", 0) / 1e6,
            "Phi_CO2": phi_val[0] if len(phi_val) > 0 else 1.0,
            "a_w": aw_val,
            "gamma_w": gamma_val,
            "Delta_Mu_w": mu_w,
            "Delta_Mu_H": mu_h,
            "Theta_Small_CO2": occ_small.get("CO2", 0),
            "Theta_Large_CO2": occ_large.get("CO2", 0)
        }

    def evaluate_structure(self, T, P_initial_guess, structure, method="newton"):
        """Runs the pressure iteration loop and returns the full thermodynamic state."""
        
        def objective(P):
            f_dict, _ = self.eos.calc_fugacities(T, P)
            try:
                unifac = ModifiedUnifac({"H2O": 1.0, "CO2": 0.0}, self.database)
                x_w = 1.0 - (f_dict["CO2"] / unifac.calc_henry_constant("CO2", T))
                aw_val = x_w * ModifiedUnifac({"H2O": x_w, "CO2": 1.0 - x_w}, self.database).calc_gamma(T)["H2O"]
            except Exception:
                aw_val = 1.0 - (f_dict.get("CO2", 0) / 7.35e7)

            mu_w = self.hydrate_model.chemical_potential_difference_water(T, P, aw_val, structure)
            mu_h = self.hydrate_model.chemical_potential_difference_hydrate(T, f_dict, structure)
            return mu_w - mu_h

        try:
            if method == "newton":
                sol = root_scalar(objective, x0=P_initial_guess, method='newton', maxiter=50)
            elif method == "secant":
                sol = root_scalar(objective, x0=P_initial_guess, x1=P_initial_guess*1.1, method='secant', maxiter=50)
            elif method == "bisect":
                sol = root_scalar(objective, bracket=[1e5, 50e6], method='bisect', xtol=1.0)
            else:
                raise ValueError(f"Unknown solver method: {method}")

            if sol.converged:
                # Recalculate and store all properties at the found equilibrium pressure
                return self._calculate_state(T, sol.root, structure)
            return None
        except Exception:
            return None

    def find_optimum_structure(self, T_range, P_initial_guess=2.5e6, solver_method="newton"):
        """Compares sI and sII and returns a comprehensive list of dictionaries."""
        
        all_results = []

        for T in T_range:
            state_sI = self.evaluate_structure(T, P_initial_guess, "sI", method=solver_method)
            state_sII = self.evaluate_structure(T, P_initial_guess, "sII", method=solver_method)

            P_sI = state_sI["P_eq (MPa)"] if state_sI else np.nan
            P_sII = state_sII["P_eq (MPa)"] if state_sII else np.nan

            opt_struct = "None"
            opt_state = None

            if not np.isnan(P_sI) and not np.isnan(P_sII):
                opt_struct = "sI" if P_sI < P_sII else "sII"
                opt_state = state_sI if P_sI < P_sII else state_sII
            elif not np.isnan(P_sI):
                opt_struct, opt_state = "sI", state_sI
            elif not np.isnan(P_sII):
                opt_struct, opt_state = "sII", state_sII

            # Compile row data
            row = {"T (K)": T, "Optimum_Structure": opt_struct}
            
            # Add optimum state data to the row
            if opt_state:
                row.update(opt_state)
            else:
                # Fill with NaNs if no equilibrium found
                row.update({k: np.nan for k in ["P_eq (MPa)", "f_CO2 (MPa)", "Phi_CO2", "a_w", "gamma_w", "Delta_Mu_w", "Delta_Mu_H", "Theta_Small_CO2", "Theta_Large_CO2"]})

            all_results.append(row)

        # Convert to Pandas DataFrame for beautiful formatting
        df_results = pd.DataFrame(all_results)
        return df_results