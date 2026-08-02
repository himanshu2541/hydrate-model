import numpy as np
from .base import EquationOfState


class PTEOS(EquationOfState):
    def __init__(self, composition, database):
        super().__init__(composition, database)
        self.y = np.array([composition.get(gas, 0.0) for gas in self.gases])
        self.R = self.database.R
        self.kij = getattr(self.database, "KIJ_DB", {})
        self._cache = {}

    def _binary_interaction_parameter(self, gas1, gas2):
        return self.kij.get((gas1, gas2), self.kij.get((gas2, gas1), 0.0))

    def _get_eos_params_and_Z(self, T, P):
        cache_key = (T, P)
        if cache_key in self._cache:
            return self._cache[cache_key]

        n = len(self.gases)
        ai, bi, ci = np.zeros(n), np.zeros(n), np.zeros(n)

        for i, gas in enumerate(self.gases):
            props = self.database.GUEST_DB[gas]
            Tc, Pc, omega = props["Tc"], props["Pc"], props["omega"]
            Tr = T / Tc

            zeta_c = 0.329032 - 0.076799 * omega + 0.0211947 * omega**2
            Omega_c = 1.0 - 3.0 * zeta_c

            ob_coeffs = [1.0, (2.0 - 3.0 * zeta_c), 3.0 * (zeta_c**2), -(zeta_c**3)]
            ob_roots = np.roots(ob_coeffs)
            ob_real = np.real(ob_roots[np.isreal(ob_roots) & (np.real(ob_roots) > 0)])
            Omega_b = np.min(ob_real) if len(ob_real) > 0 else 0.0
            Omega_a = (
                3.0 * (zeta_c**2)
                + 3.0 * (1.0 - 2.0 * zeta_c) * Omega_b
                + (Omega_b**2)
                + 1.0
                - 3.0 * zeta_c
            )

            F = 0.452413 + 1.30982 * omega - 0.295937 * omega**2

            # THERMODYNAMIC FIX: Boston-Mathias Extrapolation
            if Tr > 1.0:
                d = 1.0 + F / 2.0
                c = 1.0 - 1.0 / d
                alpha = np.exp(2.0 * c * (1.0 - Tr**d))
            else:
                alpha = (1.0 + F * (1.0 - np.sqrt(Tr))) ** 2

            ai[i] = Omega_a * ((self.R * Tc) ** 2 / Pc) * alpha
            bi[i] = Omega_b * (self.R * Tc) / Pc
            ci[i] = Omega_c * (self.R * Tc) / Pc

        am, bm, cm = 0.0, 0.0, 0.0
        a_mix_matrix = np.zeros((n, n))

        for i in range(n):
            bm += self.y[i] * bi[i]
            cm += self.y[i] * ci[i]
            for j in range(n):
                kij = self._binary_interaction_parameter(self.gases[i], self.gases[j])
                a_ij = np.sqrt(ai[i] * ai[j]) * (1.0 - kij)
                a_mix_matrix[i, j] = a_ij
                am += self.y[i] * self.y[j] * a_ij

        A = am * P / (self.R**2 * T**2)
        B = bm * P / (self.R * T)
        C = cm * P / (self.R * T)

        coeffs = [
            1.0,
            (C - 1.0),
            (A - 2.0 * B * C - B**2 - B - C),
            (C * B**2 + B * C - A * B),
        ]
        roots = np.roots(coeffs)
        Z_roots = np.real(roots[np.isreal(roots) & (roots > 0)])
        Z = np.max(Z_roots) if len(Z_roots) > 0 else 1.0

        result = (Z, A, B, C, am, bm, cm, ai, bi, ci, a_mix_matrix)
        self._cache[cache_key] = result
        return result

    def calc_Z(self, T, P) -> float:
        if P < 1.0:
            return 1.0
        Z, *_ = self._get_eos_params_and_Z(T, P)
        return float(Z)

    def calc_fugacities(self, T, P) -> tuple[dict, np.ndarray]:
        if P < 1.0:
            return {gas: P * self.y[i] for i, gas in enumerate(self.gases)}, np.ones(
                len(self.gases)
            )
        Z, A, B, C, am, bm, cm, ai, bi, ci, a_mix_matrix = self._get_eos_params_and_Z(
            T, P
        )

        phi = np.zeros(len(self.gases))
        f_dict = {}
        Q = np.sqrt(B**2 + C**2 + 6.0 * B * C)

        for i, gas in enumerate(self.gases):
            sum_yaj = np.sum(self.y * a_mix_matrix[i, :])
            term1 = (bi[i] / bm) * (Z - 1.0)
            term2 = np.log(Z - B)
            log_term = np.log((Z + 0.5 * (B + C + Q)) / (Z + 0.5 * (B + C - Q)))
            term3 = (A / Q) * ((2.0 * sum_yaj / am) - (bi[i] / bm)) * log_term

            phi[i] = np.exp(term1 - term2 - term3)
            f_dict[gas] = self.y[i] * phi[i] * P

        return f_dict, phi
