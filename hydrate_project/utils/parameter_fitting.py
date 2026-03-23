import numpy as np
from scipy.optimize import minimize


class KiharaOptimizer:
    def __init__(self, solver, experimental_data, promoter_name="DIOX", promoter_frac=0.0556):
        self.solver = solver
        self.exp_data = experimental_data
        self.promoter = promoter_name
        self.promoter_frac = promoter_frac

    def objective_function(self, params):
        """
        Minimize AAD between experimental pressure and model pressure.
        params = [sigma, eps_k, a]
        FIX: evaluate_structure does not accept promoter_frac/promoter_name kwargs;
        those are set on the solver's liquid composition at construction time.
        The Kihara params are updated directly in the database.
        """
        sigma, eps_k, a = params

        self.solver.database.GUEST_DB[self.promoter]["sigma"] = sigma
        self.solver.database.GUEST_DB[self.promoter]["eps_k"] = eps_k
        self.solver.database.GUEST_DB[self.promoter]["a"] = a

        error = 0.0
        valid_points = 0

        for T_exp, P_exp in zip(self.exp_data["T (K)"], self.exp_data["P_eq (MPa)"]):
            # FIX: evaluate_structure signature is (T, P_initial_guess, structure, method)
            state = self.solver.evaluate_structure(
                T=T_exp,
                P_initial_guess=P_exp * 1e6,
                structure="sII",
                method="bisect",
            )

            if state is None or np.isnan(state.get("P_eq (MPa)", np.nan)):
                error += 1000.0
            else:
                P_calc = state["P_eq (MPa)"]
                error += abs(P_calc - P_exp) / P_exp
                valid_points += 1

        if valid_points == 0:
            return 1e6

        aad_percentage = (error / valid_points) * 100
        print(f"Trying: sigma={sigma:.4f}, eps/k={eps_k:.2f}, a={a:.4f} --> Error: {aad_percentage:.2f}%")
        return aad_percentage

    def run_optimization(self, initial_guess=None):
        """Runs Nelder-Mead simplex to find best Kihara parameters."""
        if initial_guess is None:
            initial_guess = [3.48, 380.0, 0.85]

        print(f"--- Starting Kihara Parameter Regression for {self.promoter} ---")

        bounds = [(3.0, 4.0), (300.0, 550.0), (0.5, 1.0)]

        result = minimize(
            self.objective_function,
            initial_guess,
            method="Nelder-Mead",
            bounds=bounds,
            options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 200},
        )

        if result.success:
            print("\n✅ OPTIMIZATION SUCCESSFUL!")
            print(f"Optimized Parameters for {self.promoter}:")
            print(f"  Sigma (σ)      : {result.x[0]:.4f} Å")
            print(f"  Epsilon/k (ε/k): {result.x[1]:.2f} K")
            print(f"  Core radius (a): {result.x[2]:.4f} Å")
            print(f"  Final AAD      : {result.fun:.2f}%")
        else:
            print("\n❌ OPTIMIZATION FAILED.")
            print(result.message)

        return result.x