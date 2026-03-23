import numpy as np

def calculate_thermodynamics(results_df, eos_instance):
    """
    Calculates Del_H, Del_S, Del_G based on the Clausius-Clapeyron Equation.
    Safely filters out non-converged solver points (NaNs).
    """
    p_col = [col for col in results_df.columns if 'P' in col][0]
    valid_mask = results_df[p_col].notna() & results_df['T (K)'].notna()
    
    # We need at least 2 valid points to calculate a derivative slope
    if valid_mask.sum() < 2:
        for col in ['Z_gas', 'ΔH_diss (kJ/mol)', 'ΔS_diss (J/mol.K)', 'ΔG_eq (J/mol)']:
            results_df[col] = np.nan
        return results_df
        
    valid_df = results_df[valid_mask]
    T_vals = valid_df['T (K)'].values
    P_vals_pa = valid_df[p_col].values * 1e6  # Convert MPa to Pascals
    
    inv_T = 1.0 / T_vals
    ln_P = np.log(P_vals_pa)
    slopes = np.gradient(ln_P, inv_T) 
    
    R = 8.31446  
    Z_list, del_H, del_S, del_G = [], [], [], []
    
    for T, P_pa, slope in zip(T_vals, P_vals_pa, slopes):
        Z = eos_instance.calc_Z(T, P_pa) if hasattr(eos_instance, 'calc_Z') else 1.0 
        H = -slope * Z * R  
        S = H / T  
        G = H - (T * S) 
        
        Z_list.append(Z)
        del_H.append(H / 1000.0)
        del_S.append(S)
        del_G.append(G)
        
    results_df.loc[valid_mask, 'Z_gas'] = Z_list
    results_df.loc[valid_mask, 'ΔH_diss (kJ/mol)'] = del_H
    results_df.loc[valid_mask, 'ΔS_diss (J/mol.K)'] = del_S
    results_df.loc[valid_mask, 'ΔG_eq (J/mol)'] = del_G
    
    return results_df