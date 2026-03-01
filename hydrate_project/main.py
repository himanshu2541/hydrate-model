import numpy as np
import pandas as pd
from .core.database import Database
from .thermo_model.john_holder import JohnHolderModel as HydrateModel
from .eos_model.pr_eos import PREOS
from .solvers.equilibrium import EquilibriumSolver
from .utils.visualize import HydrateVisualizer

def main():
    gas_comp = {"CO2": 1.0}
    T_range = np.arange(273.15, 283.15, 1.0)
    
    db = Database()
    hydrate_core = HydrateModel(database=db)
    pr_eos = PREOS(composition=gas_comp, database=db)

    solver = EquilibriumSolver(
        database=db, 
        hydrate_model=hydrate_core, 
        eos_model=pr_eos
    )

    print("Running solver... This may take a moment depending on the method.")
    # The solver now returns a Pandas DataFrame
    results_df = solver.find_optimum_structure(
        T_range=T_range, 
        P_initial_guess=2.5e6, 
        solver_method="newton" 
    )

    # 1. Display nicely formatted results in the console
    pd.set_option('display.max_columns', None) # Ensures all columns print
    pd.set_option('display.width', 1000)
    print("\n--- Detailed Equilibrium Results ---")
    print(results_df.to_string(index=False))

    # 2. Generate Graphs
    print("\nGenerating Phase Boundary Graph...")
    HydrateVisualizer.plot_phase_boundary(results_df)

    print("Generating Cage Occupancy Graph...")
    HydrateVisualizer.plot_cage_occupancies(results_df)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore') 
    main()