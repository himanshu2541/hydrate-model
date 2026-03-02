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

        # Vapor Phase
        f_dict, phi_val = self.eos.calc_fugacities(T, P)

        # Liquid Phase
        try:
            unifac_pure = ModifiedUnifac({"H2O": 1.0}, self.database)
            x_gas_total = 0.0
            mix_comps = {}
            
            # Dynamically calculate dissolved gas fractions
            for gas in f_dict.keys():
                H_val = unifac_pure.calc_henry_constant(gas, T)
                x_gas = f_dict[gas] / H_val
                x_gas_total += x_gas
                mix_comps[gas] = x_gas

            # Apply penalty to water fraction from the liquid promoter
            x_w = max(1.0 - x_gas_total - self.promoter_frac, 0.0)
            mix_comps["H2O"] = x_w
            
            if self.promoter_frac > 0 and self.promoter_name:
                mix_comps[self.promoter_name] = self.promoter_frac

            # Calculate Activity Coefficient with UNIFAC
            unifac_mix = ModifiedUnifac(mix_comps, self.database)
            gamma_dict = unifac_mix.calc_gamma(T)
            gamma_val = gamma_dict.get("H2O", 1.0)
            aw_val = x_w * gamma_val
            
            # Promoter Fugacity (Allows the promoter to enter the hydrate cages!)
            if self.promoter_frac > 0 and self.promoter_name:
                # Approximate P_sat for promoter (In highly advanced models, use Antoine Eq here)
                P_sat = 10000.0 
                f_dict[self.promoter_name] = self.promoter_frac * gamma_dict.get(self.promoter_name, 1.0) * P_sat

        except Exception:
            gamma_val = 1.0
            aw_val = max(1.0 - sum(f_dict.get(g, 0) / 7.35e7 for g in f_dict.keys()) - self.promoter_frac, 0.0)

        # 3. Hydrate Phase & Occupancies
        mu_w = self.hydrate_model.chemical_potential_difference_water(T, P, aw_val, structure)
        mu_h = self.hydrate_model.chemical_potential_difference_hydrate(T, f_dict, structure)
        
        occ_small = self.hydrate_model.calc_cage_occupancy(T, f_dict, structure, "small")
        occ_large = self.hydrate_model.calc_cage_occupancy(T, f_dict, structure, "large")

        state = {
            "P_eq (MPa)": P / 1e6,
            "a_w": aw_val,
            "gamma_w": gamma_val,
            "Delta_Mu_w": mu_w,
            "Delta_Mu_H": mu_h,
        }

        # Dynamically append features for all gases (Maintains existing CO2 output perfectly)
        for i, gas in enumerate(self.eos.gases):
            state[f"f_{gas} (MPa)"] = f_dict.get(gas, 0) / 1e6
            state[f"Phi_{gas}"] = phi_val[i] if i < len(phi_val) else 1.0
            state[f"Theta_Small_{gas}"] = occ_small.get(gas, 0)
            state[f"Theta_Large_{gas}"] = occ_large.get(gas, 0)
            
        # Append extra features for the Promoter if one is present
        if self.promoter_frac > 0 and self.promoter_name:
            state[f"f_{self.promoter_name} (MPa)"] = f_dict.get(self.promoter_name, 0) / 1e6
            state[f"Theta_Small_{self.promoter_name}"] = occ_small.get(self.promoter_name, 0)
            state[f"Theta_Large_{self.promoter_name}"] = occ_large.get(self.promoter_name, 0)

        return state

    def evaluate_structure(self, T, P_initial_guess, structure, method="newton"):
        """Runs the pressure iteration loop and returns the full thermodynamic state."""
        
        def objective(P):
            if P <= 0:
                return 1e6 - P # Steer away from non-physical pressures
            
            f_dict, _ = self.eos.calc_fugacities(T, P)
            
            try:
                unifac_pure = ModifiedUnifac({"H2O": 1.0}, self.database)
                x_gas_total = sum(f_dict[g] / unifac_pure.calc_henry_constant(g, T) for g in f_dict.keys())
                x_w = max(1.0 - x_gas_total - self.promoter_frac, 0.0)
                
                mix_comps = {g: f_dict[g] / unifac_pure.calc_henry_constant(g, T) for g in f_dict.keys()}
                mix_comps["H2O"] = x_w
                if self.promoter_frac > 0 and self.promoter_name:
                    mix_comps[self.promoter_name] = self.promoter_frac
                    
                unifac_mix = ModifiedUnifac(mix_comps, self.database)
                gamma_dict = unifac_mix.calc_gamma(T)
                aw_val = x_w * gamma_dict.get("H2O", 1.0)
                
                if self.promoter_frac > 0 and self.promoter_name:
                    P_sat = 10000.0 
                    f_dict[self.promoter_name] = self.promoter_frac * gamma_dict.get(self.promoter_name, 1.0) * P_sat
                    
            except Exception:
                aw_val = max(1.0 - sum(f_dict.get(g, 0) / 7.35e7 for g in f_dict.keys()) - self.promoter_frac, 0.0)

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

            row = {"T (K)": T, "Optimum_Structure": opt_struct}
            
            if opt_state:
                row.update(opt_state)
            else:
                # Ensure missing columns are filled with NaNs if calculation fails
                keys_to_nan = ["P_eq (MPa)", "a_w", "gamma_w", "Delta_Mu_w", "Delta_Mu_H"]
                for gas in self.eos.gases:
                    keys_to_nan.extend([f"f_{gas} (MPa)", f"Phi_{gas}", f"Theta_Small_{gas}", f"Theta_Large_{gas}"])
                if self.promoter_frac > 0 and self.promoter_name:
                    keys_to_nan.extend([f"f_{self.promoter_name} (MPa)", f"Theta_Small_{self.promoter_name}", f"Theta_Large_{self.promoter_name}"])
                    
                row.update({k: np.nan for k in keys_to_nan})

            all_results.append(row)

        return pd.DataFrame(all_results)