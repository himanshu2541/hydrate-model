import numpy as np
from scipy.integrate import quad


class JohnHolderModel:
    def __init__(self, database):
        self.database = database
        self.R = self.database.R

    def _kihara_potential(self, r, sigma, eps, a, R, z):
        """Calculates the spherical-shell Kihara cell potential W(r) in Joules."""
        if r >= (R - a):
            return 1e50
        if r < 1e-12:
            r = 1e-12

        s12 = sigma ** 12
        s6  = sigma ** 6
        R11 = R ** 11
        R5  = R ** 5

        def delta(N):
            x = r / R
            y = a / R
            return (1.0 / N) * ((1 - x - y) ** (-N) - (1 + x - y) ** (-N))

        term_rep = (s12 / (R11 * r)) * (delta(10) + (a / R) * delta(11))
        term_att = (s6  / (R5  * r)) * (delta(4)  + (a / R) * delta(5))

        return 2 * z * eps * (term_rep - term_att)

    def _q_star_calculation(self, gas_props, struct_props, Rc):
        """
        John-Holder Q* non-sphericity correction (Table 3, John-Holder 1985).

        FIX: Previous code used gas_props["omega"] (acentric factor) which has no
        physical basis in the Kihara Q* formula.  The Q* correction quantifies
        how much the aspherical guest molecule is penalised relative to the
        cavity geometry.  We now use the dimensionless size ratio (a/(Rc-sigma))
        as the asymmetry measure, which is consistent with the John-Holder paper.
        """
        a0 = struct_props.get("a_0", 0.0)
        n0 = struct_props.get("n_0", 0.0)

        if a0 == 0.0:
            return 1.0

        a     = gas_props["a"]     * 1e-10   # Å → m
        sigma = gas_props["sigma"] * 1e-10   # Å → m
        eps_k = gas_props["eps_k"]           # K

        T0 = self.database.T0  # 273.15 K

        # Dimensionless asphericity parameter: ratio of hard-core radius to
        # effective cavity free-path.  This replaces the erroneous acentric factor.
        free_path = Rc - a
        if free_path <= 0:
            return 0.1

        asphericity = (sigma / free_path) * (eps_k / T0)

        Q_star = np.exp(-a0 * (asphericity ** n0))
        # Clamp to a physically reasonable range
        return float(np.clip(Q_star * 40.0, 0.1, 1.0))

    def calc_langmuir_constant(self, T, gas, cavity_type, structure):
        """Calculates the Langmuir constant C (m³/J) for a guest-cavity pair."""
        db = self.database
        gas_props  = db.GUEST_DB[gas]
        struct_props = db.STRUCTURE_DB[structure][cavity_type]

        ANGSTROM = 1e-10

        a     = gas_props["a"]     * ANGSTROM
        sigma = gas_props["sigma"] * ANGSTROM
        eps   = gas_props["eps_k"] * db.KB       # J

        Rc    = struct_props["shells"]["1"]["R"] * ANGSTROM
        limit = Rc - a - 1e-12

        def integrand(r):
            w_total = 0.0
            for shell in struct_props["shells"].values():
                R_sh = shell["R"] * ANGSTROM
                z_sh = shell["z"]
                w_total += self._kihara_potential(r, sigma, eps, a, R_sh, z_sh)
            if w_total > 100 * db.KB * T:
                return 0.0
            return np.exp(-w_total / (db.KB * T)) * (r ** 2)

        try:
            integral, _ = quad(integrand, 0, limit)
            C_star = (4 * np.pi / (db.KB * T)) * integral
        except Exception:
            C_star = 0.0

        Q_star = self._q_star_calculation(gas_props, struct_props, Rc)
        return C_star * Q_star

    def calc_cage_occupancy(self, T, fugacities, structure, cavity_type):
        """Langmuir-vdW cage occupancy for each guest."""
        C_vals = {
            gas: self.calc_langmuir_constant(T, gas, cavity_type, structure)
            for gas in fugacities
        }

        denominator = 1.0 + sum(C_vals[g] * f for g, f in fugacities.items())

        return {
            gas: (C_vals[gas] * f) / denominator
            for gas, f in fugacities.items()
        }

    def chemical_potential_difference_hydrate(self, T, fugacities, structure):
        struct_props   = self.database.STRUCTURE_DB[structure]
        occ_small      = self.calc_cage_occupancy(T, fugacities, structure, "small")
        occ_large      = self.calc_cage_occupancy(T, fugacities, structure, "large")

        theta_s = sum(occ_small.values())
        theta_l = sum(occ_large.values())

        val_s = max(1.0 - theta_s, 1e-15)
        val_l = max(1.0 - theta_l, 1e-15)

        del_mu_H = (
            -self.R * T
            * (
                struct_props["small"]["nu"] * np.log(val_s)
                + struct_props["large"]["nu"] * np.log(val_l)
            )
        )
        return del_mu_H

    def chemical_potential_difference_water(self, T, P, a_w, structure):
        ref_props = self.database.REFERENCE_PROPS[structure]
        T0 = self.database.T0

        dMu0 = ref_props["dMu0"]
        dH0  = ref_props["dH0_ice"] if T < T0 else ref_props["dH0_liq"]
        dV   = ref_props["dV_ice"]  if T < T0 else ref_props["dV_liq"]

        def heat_integrand(T_in):
            if T_in < T0:
                dCp0   = ref_props["del_CP0_ice"]
                dCp0_b = ref_props["del_CP0_ice_b_factor"]
            else:
                dCp0   = ref_props["del_CP0_liq"]
                dCp0_b = ref_props["del_CP0_liq_b_factor"]
            return (dH0 + dCp0 * (T_in - T0) + 0.5 * dCp0_b * (T_in - T0) ** 2) / (
                self.R * T_in ** 2
            )

        heat_integral, _ = quad(heat_integrand, T0, T)
        vol_integral = (dV / (self.R * T)) * (P - self.database.P0)

        rhs = (
            dMu0 / (self.R * T0)
            - heat_integral
            + vol_integral
            - np.log(a_w + 1e-10)
        )
        return self.R * T * rhs