import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class HydrateVisualizer:
    
    @staticmethod
    def _break_line_at_jumps(df, threshold=1.5):
        """
        Detects sudden jumps in pressure (phase transitions) and inserts
        a NaN row to prevent matplotlib from connecting the disjointed lines.
        """
        valid_df = df.dropna(subset=['P_eq (MPa)']).sort_values(by="T (K)").reset_index(drop=True)
        jumps = valid_df.index[valid_df["P_eq (MPa)"].diff().abs() > threshold].tolist()
        
        if not jumps:
            return valid_df
            
        nan_rows = pd.DataFrame(np.nan, index=jumps, columns=valid_df.columns)
        nan_rows.index = nan_rows.index - 0.5 
        
        df_broken = pd.concat([valid_df, nan_rows]).sort_index().reset_index(drop=True)
        return df_broken

    @staticmethod
    def plot_phase_boundary(df):
        """Plots the Equilibrium Pressure vs Temperature."""
        plot_df = HydrateVisualizer._break_line_at_jumps(df)

        plt.figure(figsize=(8, 6))
        
        plt.plot(plot_df["T (K)"], plot_df["P_eq (MPa)"], marker='o', linestyle='-', color='b', label='Hydrate Phase Boundary')
        
        plt.title('Phase Equilibrium Boundary', fontsize=14)
        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel('Equilibrium Pressure (MPa)', fontsize=12)
        plt.grid(False)

        
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
        plt.ylabel('Fractional Occupancy (θ)', fontsize=12)
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_eos_comparison(results_dict, exp_data=None):
        """Plots multiple EOS predictions against experimental data."""
        plt.figure(figsize=(10, 7))
        
        colors = ['b', 'g', 'r', 'c', 'm']
        styles = ['-', '--', '-.', ':']

        # Plot each EOS model
        for i, (eos_name, df) in enumerate(results_dict.items()):
            plot_df = HydrateVisualizer._break_line_at_jumps(df)
            
            plt.plot(
                plot_df["T (K)"], 
                plot_df["P_eq (MPa)"], 
                marker = ['o', 's', '^', 'D', 'v'][i % 5],
                color=colors[i % len(colors)], 
                linestyle=styles[i % len(styles)], 
                linewidth=2,
                label=f'{eos_name} Model'
            )

        # Plot Experimental Data if provided
        if exp_data:
            plt.scatter(
                exp_data["T (K)"], 
                exp_data["P_eq (MPa)"], 
                color='black', 
                marker='x', 
                s=60, 
                zorder=5, 
                label='Experimental Data'
            )
            
            # Automatically zoom the Y-axis to fit just above the highest experimental point
            max_exp_p = max(exp_data["P_eq (MPa)"])
            plt.ylim(0, max_exp_p * 1.2)

        plt.title('Hydrate Phase Boundary: EOS Comparison', fontsize=16)
        plt.xlabel('Temperature (K)', fontsize=14)
        plt.ylabel('Equilibrium Pressure (MPa)', fontsize=14)
        plt.grid(False)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()