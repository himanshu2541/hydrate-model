import numpy as np
from ..core.database import Database

class ModifiedUnifac:
    def __init__(self, x_composition, database: Database):
        self.comps = x_composition
        self.molecules = list(x_composition.keys())  # e.g., ['H2O', 'CO2']
        self.x = np.array([x_composition.get(mol, 0.0) for mol in self.molecules])
        self.database = database

    def _get_interaction_param(self, m, n, T):
        """Calculates a_mn(T) = a_mn,1 + a_mn,2 * (T - 298.15)"""

        if (m, n) in self.database.MOD_UNIFAC_INTERACTIONS:
            p = self.database.MOD_UNIFAC_INTERACTIONS[(m, n)]
            return p[0] + p[1] * (T - 298.15)
        return 0.0

    def calc_gamma(self, T):
        """
        Calculates Activity Coefficients (gamma) for all components in the mixture
        using the Modified UNIFAC model (Larsen et al., 1987 framework).
        Returns: dict {'H2O': gamma_w, 'CO2': gamma_co2}
        """
        # Component Properties (r_i, q_i)
        # r_i = sum(nu_k * R_k), q_i = sum(nu_k * Q_k)
        r = []
        q = []

        # Molecular breakdown into groups
        # Groups present in the mixture
        groups_in_mix = set()

        # nu[i][k] = number of groups k in molecule i
        mol_group_counts = []

        for mol in self.molecules:
            # Get UNIFAC group mapping for molecule
            if mol in self.database.UNIFAC_MAPPING:
                mapping = self.database.UNIFAC_MAPPING[mol]["unifac_groups"]
            elif mol == "H2O":
                mapping = {6: 1}  # Explicitly add water if missing
            else:
                mapping = {}

            mol_group_counts.append(mapping)

            r_i = 0.0
            q_i = 0.0

            for group_id, count in mapping.items():
                groups_in_mix.add(group_id)
                props = self.database.MOD_UNIFAC_GROUPS[group_id]
                r_i += count * props["R"]
                q_i += count * props["Q"]
            r.append(r_i)
            q.append(q_i)

        r = np.array(r)
        q = np.array(q)

        # 2. Combinatorial Part (Modified Flory-Huggins)
        # Larsen et al. 1987 uses p = 2/3
        p = 2.0 / 3.0
        r_pow = np.power(r, p)

        # J_i = r_i^(2/3) / sum(x_j * r_j^(2/3))
        # L_i = q_i / sum(x_j * q_j) (Standard Flory-Huggins uses volume fractions, Modified uses surface/volume mix)
        # Note: Larsen formula for Combinatorial:
        # ln gamma_i_comb = ln(omega_i/x_i) + 1 - omega_i/x_i
        # where omega_i = x_i * r_i^(2/3) / sum(x_j * r_j^(2/3))
        denom_comb = np.sum(self.x * r_pow)
        if denom_comb == 0: return {m: 1.0 for m in self.molecules} # Avoid division by zero
        
        omega = (self.x * r_pow) / denom_comb

        ln_gamma_comb = np.zeros_like(self.x)
        mask = self.x > 1e-12
        ln_gamma_comb[mask] = np.log(omega[mask] / self.x[mask]) + 1 - (omega[mask] / self.x[mask])


        # 3. Residual Part
        # ln gamma_i_res = sum_k nu_ki * (ln Gamma_k - ln Gamma_k_i)

        # Need group fractions in mixture (X_m)
        # X_m = sum_j (x_j * nu_jm) / sum_j sum_n (x_j * nu_jn)
        all_groups = list(groups_in_mix)
        group_map = {grp: i for i, grp in enumerate(all_groups)}
        num_groups = len(all_groups)
        
        X_m = np.zeros(num_groups)
        total_groups = 0.0
        for i, mol in enumerate(self.molecules):
            for grp, count in mol_group_counts[i].items():
                idx = group_map[grp]
                X_m[idx] += self.x[i] * count
                total_groups += self.x[i] * count
        
        if total_groups > 0: X_m /= total_groups

        # Calculate Group Area Fractions (Theta m)
        # Theta_m = (Q_m * X_m) / sum_n (Q_n * X_n)
        Q_k = np.array([self.database.MOD_UNIFAC_GROUPS[g]["Q"] for g in all_groups])
        Theta_m = (Q_k * X_m) / np.sum(Q_k * X_m) if np.sum(Q_k * X_m) > 0 else np.zeros(num_groups)

        # Temperature Dependent Parameters Psi_nm
        # Psi_nm = exp(-a_nm / T)
        Psi = np.zeros((num_groups, num_groups))
        for i in range(num_groups):
            for j in range(num_groups):
                Psi[i, j] = np.exp(-self._get_interaction_param(all_groups[i], all_groups[j], T) / T)

        # Calculate ln Gamma_k (Group activity coefficient in mixture)
        # ln Gamma_k = Q_k * [1 - ln(sum_m Theta_m * Psi_mk) - sum_m (Theta_m * Psi_km / sum_n Theta_n * Psi_nm)]

        ln_Gamma_mix = np.zeros(num_groups)
        for k in range(num_groups):
            sum1 = np.sum(Theta_m * Psi[:, k])
            sum2 = 0.0
            for m in range(num_groups):
                denom = np.sum(Theta_m * Psi[:, m])
                if denom > 0: sum2 += (Theta_m[m] * Psi[k, m]) / denom
            if sum1 > 0: ln_Gamma_mix[k] = Q_k[k] * (1.0 - np.log(sum1) - sum2)

        # Calculate ln Gamma_k_i (Group activity coefficient in pure component i)
        
        # Pure component reference
        ln_gamma_res = np.zeros(len(self.molecules))
        for i, mol in enumerate(self.molecules):
            total_pure = 0.0
            X_pure = np.zeros(num_groups)
            mapping = mol_group_counts[i]
            for grp, count in mapping.items():
                X_pure[group_map[grp]] = count
                total_pure += count
            X_pure /= total_pure
            
            Theta_pure = (Q_k * X_pure) / np.sum(Q_k * X_pure)
            ln_G_pure = np.zeros(num_groups)
            for k in range(num_groups):
                sum1 = np.sum(Theta_pure * Psi[:, k])
                sum2 = 0.0
                for m in range(num_groups):
                    denom = np.sum(Theta_pure * Psi[:, m])
                    if denom > 0: sum2 += (Theta_pure[m] * Psi[k, m]) / denom
                if sum1 > 0: ln_G_pure[k] = Q_k[k] * (1.0 - np.log(sum1) - sum2)
            
            sum_res = 0.0
            for grp, count in mapping.items():
                idx = group_map[grp]
                sum_res += count * (ln_Gamma_mix[idx] - ln_G_pure[idx])
            ln_gamma_res[i] = sum_res

        return {mol: np.exp(ln_gamma_comb[i] + ln_gamma_res[i]) for i, mol in enumerate(self.molecules)}

    def calc_henry_constant(self, gas, T):
        """
        Calculates Henry's Law Constant H (in Pa) using Eq 18 from Klauda & Sandler (2000).
        -ln(H / 101325) = H1 + H2/T + H3*ln(T) + H4*T
        """
        if gas not in self.database.HENRY_PARAMS:
            return 1e9  # return high value (low solubility) if gas not in DB

        p = self.database.HENRY_PARAMS[gas]

        rhs = p["H1"] + p["H2"] / T + p["H3"] * np.log(T) + p["H4"] * T
        # -ln(H/P0) = rhs  => ln(H/P0) = -rhs
        # H = P0 * exp(-rhs)
        H_pa = self.database.P0 * np.exp(-rhs)  # in Pa

        # print(f"Henry's Law Constant for {gas} at T={T:.2f}K: H = {H_pa:.2e} Pa")
        return H_pa

    def calc_activity_coefficients(self, T, P):
        """
        Calculates water activity a_w = x_w * gamma_w.
        1. x_w is calculated using Henry's Law from Klauda & Sandler (Table 4).
        2. gamma_w (Modified UNIFAC) is approx 1.0 here (requires external database).
        """
        # Calculate x_CO2 using Henry's Law: x = P_partial / H
        # neglecting poynting correction

        x_gas_total = 0.0
        for gas, y_frac in self.comps.items():
            H_pa = self.calc_henry_constant(gas, T)
            P_partial = y_frac * P
            x_gas = P_partial / H_pa
            x_gas_total += x_gas

        x_w = 1.0 - x_gas_total
        if x_w < 0:
            x_w = 0.0

        # Activity Coefficient (gamma_w)
        # The paper uses Modified UNIFAC here.
        gamma_w = self.calc_gamma(T).get("H2O", 1.0)

        a_w = x_w * gamma_w

        # print(f"x_w: {x_w:.4e}, gamma_w: {gamma_w:.4f}, a_w: {a_w:.4f}")

        return a_w