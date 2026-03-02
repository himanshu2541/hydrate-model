import numpy as np
from scipy.integrate import quad

class JohnHolderModel:
    def __init__(self, database):
        self.database = database
        self.R = self.database.R

    def _kihara_potential(self, r, sigma, eps, a, R, z):
        """Calculates W(r) in Joules."""
        # Singularity check: if r approaches the wall (R-a), potential explodes
        if r >= (R - a):
            return 1e50  # effectively infinity

        # At r=0, the 1/r term and the delta expansion resolve to a constant.
        # However, for numerical simplicity, we can just use a tiny epsilon
        if r < 1e-12:
            r = 1e-12

        # Pre-calculate powers
        s12 = sigma**12
        s6 = sigma**6
        R11 = R**11
        R5 = R**5

        def delta(N):
            x = r / R
            y = a / R

            term1 = (1 - x - y) ** (-N)
            term2 = (1 + x - y) ** (-N)

            return (1.0 / N) * (term1 - term2)

        term_rep = (s12 / (R11 * r)) * (delta(10) + (a / R) * delta(11))
        term_att = (s6 / (R5 * r)) * (delta(4) + (a / R) * delta(5))

        return 2 * z * eps * (term_rep - term_att)

    def _q_star_calculation(self, gas_props, struct_props, ref_props, Rc):
        a0 = struct_props.get("a_0", 0)
        n0 = struct_props.get("n_0", 0)

        a_g = gas_props["a"] * 1e-10
        sigma_g = gas_props["sigma"] * 1e-10
        eps_k_g = gas_props["eps_k"]

        # a_g = 0.677e-10
        # eps_k_g = 506.25
        # sigma_g = 3.407e-10

        a_w = ref_props["a_w"] * 1e-10
        sigma_w = ref_props["sigma_w"] * 1e-10
        eps_k_w = ref_props["eps_k_w"]

        # mixing rules
        a = (a_g + a_w) / 2
        sigma = (sigma_g + sigma_w) / 2
        eps_k = np.sqrt(eps_k_g * eps_k_w)

        T0 = self.database.T0

        if a0 > 0:
            omega = gas_props["omega"]
            term = (sigma / (Rc - a)) * (eps_k / T0) * abs(omega)
            Q_star = np.exp(-a0 * (term**n0))
            return max(0.5, min(Q_star * 40, 1.0))  # Q* (0.5 to 1.0)
        return 1.0

    def calc_langmuir_constant(self, T, gas, cavity_type, structure):
        db = self.database
        gas_props = db.GAS_DB[gas]
        struct_props = db.STRUCTURE_DB[structure][cavity_type]

        ANGSTROM = 1e-10

        sigma_g = gas_props["sigma"] * ANGSTROM
        a_g = gas_props["a"] * ANGSTROM
        eps_k_g = gas_props["eps_k"]

        sigma_w = db.REFERENCE_PROPS[structure]["sigma_w"] * ANGSTROM
        a_w = db.REFERENCE_PROPS[structure]["a_w"] * ANGSTROM
        eps_w_k = db.REFERENCE_PROPS[structure]["eps_k_w"]

        # Mixed parameters
        sigma = 0.5 * (sigma_g + sigma_w)
        a = 0.5 * (a_g + a_w)
        eps_k = np.sqrt(eps_k_g * eps_w_k)

        eps = eps_k * db.KB

        # Integration limit
        Rc = struct_props["shells"]["1"]["R"] * ANGSTROM

        # Safety limit to avoid singularity
        limit = Rc - a - 1e-12

        def integrand(r):
            w_total = 0.0
            # Sum potential from all shells
            for shell in struct_props["shells"].values():
                R_sh = shell["R"] * ANGSTROM
                z_sh = shell["z"]
                w_total += self._kihara_potential(r, sigma, eps, a, R_sh, z_sh)

            # Cap high energy to avoid overflow
            if w_total > 100 * db.KB * T:
                return 0.0
            return np.exp(-w_total / (db.KB * T)) * (r**2)

        try:
            integral, _ = quad(integrand, 0, limit)
            C_star = (4 * np.pi / (db.KB * T)) * integral
        except:
            C_star = 0.0

        # John-Holder Q* Correction
        Q_star = self._q_star_calculation(gas_props, struct_props, db.REFERENCE_PROPS[structure], Rc)

        # print(f"[DEBUG] Langmuir Constant C for {gas} in {cavity_type} cage at T={T:.2f}K: C*={C_star:.4e}, Q*={Q_star:.4e}, C={C_star * Q_star:.4e}")

        # print(
        #     f"[DEBUG] Langmuir Constant C for {gas} in {cavity_type} cage at T={T:.2f}K: C*={C_star:.4e} 1/Pa"
        # )
        return C_star * Q_star

    def calc_cage_occupancy(self, T, fugacities, structure, cavity_type):
        C_vals = {}
        for gas in fugacities.keys():
            C_vals[gas] = self.calc_langmuir_constant(T, gas, cavity_type, structure)

        denominator = 1.0
        for gas, f in fugacities.items():
            denominator += C_vals[gas] * f

        occupancies = {}
        for gas, f in fugacities.items():
            occupancies[gas] = (C_vals[gas] * f) / denominator

        # print(f"Cage Occupancies ({cavity_type}) at T={T:.2f}K: {occupancies}")
        return occupancies

    def chemical_potential_difference_hydrate(self, T, fugacities, structure):
        struct_props = self.database.STRUCTURE_DB[structure]
        occupancy_small = self.calc_cage_occupancy(T, fugacities, structure, "small")
        occupancy_large = self.calc_cage_occupancy(T, fugacities, structure, "large")

        summation_occupancy_small = sum(occupancy_small.values())
        summation_occupancy_large = sum(occupancy_large.values())

        # Protect against log(0)
        val_s = 1.0 - summation_occupancy_small
        val_l = 1.0 - summation_occupancy_large

        # print(f"1 - theta_small: {val_s:.4e}, 1 - theta_large: {val_l:.4e}")
        
        if val_s <= 1e-15:
            val_s = 1e-15
        if val_l <= 1e-15:
            val_l = 1e-15

        del_mu_H = (
            -self.database.R
            * T
            * (
                struct_props["small"]["nu"] * np.log(val_s)
                + struct_props["large"]["nu"] * np.log(val_l)
            )
        )
        # print(f"Del Mu_H at T={T:.2f}K: {del_mu_H:.2f} J/mol")
        return del_mu_H

    def chemical_potential_difference_water(self, T, P, a_w, structure):
        ref_props = self.database.REFERENCE_PROPS[structure]
        T0 = self.database.T0

        if T < T0:
            dMu0 = ref_props["dMu0"]
            dH0 = ref_props["dH0_ice"]
            dV = ref_props["dV_ice"]
        else:
            dMu0 = ref_props["dMu0"]
            dH0 = ref_props["dH0_liq"]
            dV = ref_props["dV_liq"]

        def heat_integrand(T_in):
            dCp0 = ref_props["del_CP0_ice"] if T_in < T0 else ref_props["del_CP0_liq"]
            dCp0_b = (
                ref_props["del_CP0_ice_b_factor"]
                if T_in < T0
                else ref_props["del_CP0_liq_b_factor"]
            )
            return (dH0 + dCp0 * (T_in - T0) + 0.5 * dCp0_b * (T_in - T0) ** 2) / (
                self.database.R * T_in**2
            )

        heat_integral, _ = quad(heat_integrand, T0, T)

        # Simplified volume integral
        vol_integral = (dV / (self.database.R * T)) * (P - self.database.P0)
        
        rhs = (
            dMu0 / (self.database.R * T0)
            - heat_integral
            + vol_integral
            - np.log(a_w + 1e-20)
        )

        del_mu_W = self.database.R * T * rhs
        return del_mu_W


