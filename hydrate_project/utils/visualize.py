# utils/visualize.py
import matplotlib.pyplot as plt

class HydrateVisualizer:
    @staticmethod
    def plot_phase_boundary(df):
        """Plots the Equilibrium Pressure vs Temperature."""
        # Drop rows where equilibrium wasn't found
        valid_df = df.dropna(subset=['P_eq (MPa)'])

        plt.figure(figsize=(8, 6))
        
        # Plot the data
        plt.plot(valid_df["T (K)"], valid_df["P_eq (MPa)"], marker='o', linestyle='-', color='b', label='Hydrate Phase Boundary')
        
        plt.title('Phase Equilibrium Boundary', fontsize=14)
        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel('Equilibrium Pressure (MPa)', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.yscale('log') # Pressure is often better viewed on a log scale for hydrates
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cage_occupancies(df):
        """Plots Small and Large Cage Occupancies vs Temperature."""
        valid_df = df.dropna(subset=['Theta_Small_CO2', 'Theta_Large_CO2'])

        plt.figure(figsize=(8, 6))
        
        plt.plot(valid_df["T (K)"], valid_df["Theta_Small_CO2"], marker='s', linestyle='-', color='g', label='Small Cage (CO2)')
        plt.plot(valid_df["T (K)"], valid_df["Theta_Large_CO2"], marker='^', linestyle='-', color='r', label='Large Cage (CO2)')
        
        plt.title('Cage Occupancies vs Temperature', fontsize=14)
        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel('Fractional Occupancy ($\theta$)', fontsize=12)
        plt.ylim(0.5, 1.05)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()