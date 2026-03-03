import numpy as np
import pandas as pd

from hydrate_project.eos_model.pt_eos import PTEOS
from hydrate_project.eos_model.srk_eos import SRKEOS
from hydrate_project.core.database import Database
from hydrate_project.thermo_model.john_holder import JohnHolderModel as HydrateModel
from hydrate_project.eos_model.pr_eos import PREOS
from hydrate_project.solvers.equilibrium import EquilibriumSolver
from hydrate_project.utils.visualize import HydrateVisualizer
from hydrate_project.utils.metrics import calculate_aad

def main():
    # gas_comp = {"CO2": 0.4, "H2": 0.6}
    gas_comp = {"CO2": 1}
    
    # liq_comp = {"H2O": 1-0.0556, "DIOX": 0.0556}  # Example liquid phase composition with a small amount of dioxane as an inhibitor
    liq_comp = {"H2O": 1}
    T_range = np.arange(273.15, 283.15, 0.5) 
    
    db = Database()
    hydrate_core = HydrateModel(database=db)

    # Initialize all equation of state models
    eos_models = {
        "Peng-Robinson": PREOS(composition=gas_comp, database=db),
        "Soave-Redlich-Kwong": SRKEOS(composition=gas_comp, database=db),
        "Patel-Teja": PTEOS(composition=gas_comp, database=db)
    }
    
    all_results = {}
    
    # Set pandas display options for neat terminal logging
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', 1000)

    # Define standard Experimental Data for CO2 Hydrate (approx 273K - 282K)
    experimental_data = {
        "T (K)": [273.15, 275.15, 277.15, 279.15, 281.15, 282.15],
        "P_eq (MPa)": [1.26, 1.67, 2.26, 3.06, 4.18, 4.49] # for pure CO2 hydrate
        # "P_eq (MPa)": [5.0, 6.5, 8.5, 10.0, 12.0, 15.0]  
    }

    # experimental_data = {
    #     "T (K)": [],
    #     "P_eq (MPa)": []
    # }

    print("="*60)
    print("      HYDRATE EQUILIBRIUM THERMODYNAMIC MODELING")
    print("="*60)

    # 1. Run Solvers, Log Values, and Calculate Errors
    aad_scores = {}

    for eos_name, eos_instance in eos_models.items():
        print(f"\n[{eos_name} EOS] Running Equilibrium Solver...")
        
        # Inject the liq_phase_composition here!
        solver = EquilibriumSolver(
            liq_phase_composition=liq_comp, 
            database=db, 
            hydrate_model=hydrate_core, 
            eos_model=eos_instance
        )
        
        results_df = solver.find_optimum_structure(
            T_range=T_range, 
            P_initial_guess=2.5e6, 
            solver_method="bisect"  # Changed to bisect to avoid crashing on phase boundaries
        )
        
        all_results[eos_name] = results_df
        
        # Calculate AAD for this specific model
        aad = calculate_aad(results_df, experimental_data)
        aad_scores[eos_name] = aad

        # Log detailed values to terminal
        print(f"\n--- Detailed Equilibrium Results ({eos_name}) ---")
        print(results_df.to_string(index=False))
        print(f">>> Average Absolute Deviation (AAD) for {eos_name}: {aad:.2f}% <<<\n")
        print("-" * 60)

    # 2. Print Error Summary
    print("\n" + "="*40)
    print("     MODEL ACCURACY SUMMARY (AAD %)")
    print("="*40)
    for eos_name, score in aad_scores.items():
        print(f"{eos_name:<25}: {score:.2f}%")
    print("="*40)

    # 3. Generate Graphs
    print("\nGenerating EOS Comparison Graph with Experimental Data...")
    HydrateVisualizer.plot_eos_comparison(all_results, experimental_data)

    # print("Generating Phase Boundary Graph (Peng-Robinson as reference)...")
    # HydrateVisualizer.plot_phase_boundary(all_results["Peng-Robinson"])

    print("Generating Cage Occupancy Graph (Peng-Robinson as reference)...")
    HydrateVisualizer.plot_cage_occupancies(all_results["Peng-Robinson"])

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore') 
    main()