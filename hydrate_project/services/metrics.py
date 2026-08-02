import numpy as np

def calculate_aad(computed_df, exp_data):
    """
    Calculates the Average Absolute Deviation (AAD %) between computed
    pressures and experimental pressures using linear interpolation.
    """
    valid_df = computed_df.dropna(subset=["P_eq (MPa)"])
    if valid_df.empty:
        return np.nan

    T_exp = np.array(exp_data["T (K)"])
    P_exp = np.array(exp_data["P_eq (MPa)"])

    # Ensure calculation arrays are sorted by Temperature for interpolation
    T_calc = valid_df["T (K)"].values
    P_calc = valid_df["P_eq (MPa)"].values

    sort_idx = np.argsort(T_calc)
    T_calc = T_calc[sort_idx]
    P_calc = P_calc[sort_idx]

    # Interpolate calculated pressures at the exact experimental temperatures
    P_calc_interp = np.interp(T_exp, T_calc, P_calc)

    # AAD Formula: (100 / N) * SUM(|P_exp - P_calc| / P_exp)
    aad = np.mean(np.abs((P_exp - P_calc_interp) / P_exp)) * 100
    return aad
