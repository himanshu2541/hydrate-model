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
        The objective is to minimize the Average Absolute Deviation (AAD)
        between the experimental pressure and the predicted pressure.
        params = [sigma, eps_k, a]
        """
        sigma, eps_k, a = params

        # 1. Update the database with the current "trial" parameters
        # (Assuming you successfully renamed GAS_DB to GUEST_DB)
        self.solver.database.GUEST_DB[self.promoter]["sigma"] = sigma
        self.solver.database.GUEST_DB[self.promoter]["eps_k"] = eps_k
        self.solver.database.GUEST_DB[self.promoter]["a"] = a

        error = 0.0
        valid_points = 0

        # 2. Loop through the experimental data
        for T_exp, P_exp in zip(self.exp_data["T (K)"], self.exp_data["P_eq (MPa)"]):
            
            # Predict the pressure using the trial parameters (always sII for DiOX)
            state = self.solver.evaluate_structure(
                T=T_exp, 
                P_initial_guess=P_exp * 1e6, # Use exp pressure as a great initial guess 
                structure="sII", 
                method="bisect", # Bisect is safest during aggressive optimization
                promoter_frac=self.promoter_frac,
                promoter_name=self.promoter
            )

            if state is None or np.isnan(state["P_eq (MPa)"]):
                # Massive penalty if the model fails to converge with these parameters
                error += 1000.0 
            else:
                P_calc = state["P_eq (MPa)"]
                # Calculate Absolute Relative Error for this point
                error += abs(P_calc - P_exp) / P_exp
                valid_points += 1

        # Return Average Error (or a huge number if everything failed)
        if valid_points == 0:
            return 1e6
            
        aad_percentage = (error / valid_points) * 100
        
        # Print progress to the terminal so you can watch it learn
        print(f"Trying: sigma={sigma:.4f}, eps/k={eps_k:.2f}, a={a:.4f} --> Error: {aad_percentage:.2f}%")
        
        return aad_percentage

    def run_optimization(self, initial_guess=[3.48, 380.0, 0.85]):
        """Runs the Nelder-Mead simplex algorithm to find the best parameters."""
        print(f"--- Starting Kihara Parameter Regression for {self.promoter} ---")
        
        # Set bounds to keep physics realistic
        # sigma: 3.0 to 4.0 Angstroms
        # eps_k: 300 to 550 K
        # a: 0.5 to 1.0 Angstroms
        bounds = [(3.0, 4.0), (300.0, 550.0), (0.5, 1.0)]

        # Nelder-Mead is robust for thermodynamic models where derivatives are jumpy
        result = minimize(
            self.objective_function, 
            initial_guess, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 200}
        )

        if result.success:
            print("\n✅ OPTIMIZATION SUCCESSFUL!")
            print(f"Optimized Parameters for {self.promoter}:")
            print(f"Sigma (σ)   : {result.x[0]:.4f} Å")
            print(f"Epsilon/k (ε/k): {result.x[1]:.2f} K")
            print(f"Core radius (a): {result.x[2]:.4f} Å")
            print(f"Final AAD Error: {result.fun:.2f}%")
        else:
            print("\n❌ OPTIMIZATION FAILED.")
            print(result.message)
            
        return result.x