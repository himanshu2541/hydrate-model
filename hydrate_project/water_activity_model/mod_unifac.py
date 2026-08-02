import logging

import numpy as np
from ..core.database import Database

log = logging.getLogger(__name__)
_warned_henry_placeholders: set[str] = set()


class ModifiedUnifac:
    def __init__(self, x_composition, database: Database):
        self.comps = x_composition
        self.molecules = list(x_composition.keys())
        self.x = np.array([x_composition.get(mol, 0.0) for mol in self.molecules])
        self.database = database

    def _get_interaction_param(self, m, n, T):
        """Calculates a_nm + b_nm*T + c_nm*T^2 for Dortmund Mod. UNIFAC"""
        if (m, n) in self.database.MOD_UNIFAC_INTERACTIONS:
            p = self.database.MOD_UNIFAC_INTERACTIONS[(m, n)]
            # Uses the 3 parameters from the database [a, b, c]
            return p[0] + p[1] * T + p[2] * (T**2)
        return 0.0

    def calc_gamma(self, T):
        r = []
        q = []
        groups_in_mix = set()
        mol_group_counts = []

        for mol in self.molecules:
            if mol in self.database.UNIFAC_MAPPING:
                mapping = self.database.UNIFAC_MAPPING[mol]["unifac_groups"]
            elif mol == "H2O":
                mapping = {7: 1}  # Dortmund uses Group 7 for Water
            else:
                mapping = {}

            mol_group_counts.append(mapping)
            r_i, q_i = 0.0, 0.0

            for group_id, count in mapping.items():
                groups_in_mix.add(group_id)
                props = self.database.MOD_UNIFAC_GROUPS[group_id]
                r_i += count * props["R"]
                q_i += count * props["Q"]
            r.append(r_i)
            q.append(q_i)

        r = np.array(r)
        q = np.array(q)

        # 2. Combinatorial Part (Modified UNIFAC Dortmund)
        p_exponent = 0.75  # Dortmund exponent
        r_pow = np.power(r, p_exponent)

        denom_phi_prime = np.sum(self.x * r_pow)
        denom_phi = np.sum(self.x * r)
        denom_theta = np.sum(self.x * q)

        if denom_phi_prime == 0 or denom_phi == 0 or denom_theta == 0:
            return {m: 1.0 for m in self.molecules}

        phi_prime = (self.x * r_pow) / denom_phi_prime
        phi = (self.x * r) / denom_phi
        theta = (self.x * q) / denom_theta

        ln_gamma_comb = np.zeros_like(self.x)
        mask = self.x > 1e-12

        # Dortmund Combinatorial Equation
        ln_gamma_comb[mask] = (
            np.log(phi_prime[mask] / self.x[mask])
            + 1.0
            - (phi_prime[mask] / self.x[mask])
            - 5.0
            * q[mask]
            * (np.log(phi[mask] / theta[mask]) + 1.0 - (phi[mask] / theta[mask]))
        )

        # 3. Residual Part
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

        if total_groups > 0:
            X_m /= total_groups

        Q_k = np.array([self.database.MOD_UNIFAC_GROUPS[g]["Q"] for g in all_groups])
        Theta_m = (
            (Q_k * X_m) / np.sum(Q_k * X_m)
            if np.sum(Q_k * X_m) > 0
            else np.zeros(num_groups)
        )

        # Dortmund Psi calculation
        Psi = np.zeros((num_groups, num_groups))
        for i in range(num_groups):
            for j in range(num_groups):
                Psi[i, j] = np.exp(
                    -self._get_interaction_param(all_groups[i], all_groups[j], T) / T
                )

        ln_Gamma_mix = np.zeros(num_groups)
        for k in range(num_groups):
            sum1 = np.sum(Theta_m * Psi[:, k])
            sum2 = 0.0
            for m in range(num_groups):
                denom = np.sum(Theta_m * Psi[:, m])
                if denom > 0:
                    sum2 += (Theta_m[m] * Psi[k, m]) / denom
            if sum1 > 0:
                ln_Gamma_mix[k] = Q_k[k] * (1.0 - np.log(sum1) - sum2)

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
                    if denom > 0:
                        sum2 += (Theta_pure[m] * Psi[k, m]) / denom
                if sum1 > 0:
                    ln_G_pure[k] = Q_k[k] * (1.0 - np.log(sum1) - sum2)

            sum_res = 0.0
            for grp, count in mapping.items():
                idx = group_map[grp]
                sum_res += count * (ln_Gamma_mix[idx] - ln_G_pure[idx])
            ln_gamma_res[i] = sum_res

        return {
            mol: np.exp(ln_gamma_comb[i] + ln_gamma_res[i])
            for i, mol in enumerate(self.molecules)
        }

    def calc_henry_constant(self, gas, T):
        if gas not in self.database.HENRY_PARAMS:
            return 1e9

        if gas in getattr(self.database, "UNVALIDATED_HENRY_PARAMS", set()):
            if gas not in _warned_henry_placeholders:
                _warned_henry_placeholders.add(gas)
                log.warning(
                    "Henry's-law constant for '%s' is an unvalidated placeholder "
                    "(not fitted to literature solubility data); results "
                    "involving it should be treated as low-confidence.",
                    gas,
                )

        p = self.database.HENRY_PARAMS[gas]
        rhs = p["H1"] + p["H2"] / T + p["H3"] * np.log(T) + p["H4"] * T
        H_pa = self.database.P0 * np.exp(-rhs)
        return H_pa
