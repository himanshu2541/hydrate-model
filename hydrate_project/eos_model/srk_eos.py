import numpy as np
from .base import EquationOfState


class SRKEOS(EquationOfState):
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
        ai, bi = np.zeros(n), np.zeros(n)

        for i, gas in enumerate(self.gases):
            props = self.database.GUEST_DB[gas]
            Tc, Pc, omega = props["Tc"], props["Pc"], props["omega"]
            Tr = T / Tc
            m = 0.480 + 1.574 * omega - 0.176 * omega**2

            # THERMODYNAMIC FIX: Boston-Mathias Extrapolation
            if Tr > 1.0:
                d = 1.0 + m / 2.0
                c = 1.0 - 1.0 / d
                alpha = np.exp(2.0 * c * (1.0 - Tr**d))
            else:
                alpha = (1.0 + m * (1.0 - np.sqrt(Tr))) ** 2

            ai[i] = 0.42748 * ((self.R * Tc) ** 2 / Pc) * alpha
            bi[i] = 0.08664 * (self.R * Tc) / Pc

        am, bm = 0.0, 0.0
        a_mix_matrix = np.zeros((n, n))

        for i in range(n):
            bm += self.y[i] * bi[i]
            for j in range(n):
                kij = self._binary_interaction_parameter(self.gases[i], self.gases[j])
                a_ij = np.sqrt(ai[i] * ai[j]) * (1.0 - kij)
                a_mix_matrix[i, j] = a_ij
                am += self.y[i] * self.y[j] * a_ij

        A = am * P / (self.R**2 * T**2)
        B = bm * P / (self.R * T)

        coeffs = [1.0, -1.0, (A - B - B**2), -(A * B)]
        roots = np.roots(coeffs)
        Z_roots = np.real(roots[np.isreal(roots) & (roots > 0)])
        Z = np.max(Z_roots) if len(Z_roots) > 0 else 1.0

        result = (Z, A, B, am, bm, ai, bi, a_mix_matrix)
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
        Z, A, B, am, bm, ai, bi, a_mix_matrix = self._get_eos_params_and_Z(T, P)

        phi = np.zeros(len(self.gases))
        f_dict = {}

        for i, gas in enumerate(self.gases):
            sum_yaj = np.sum(self.y * a_mix_matrix[i, :])
            term1 = (bi[i] / bm) * (Z - 1.0)
            term2 = np.log(Z - B)
            term3 = (
                (A / B) * ((2.0 * sum_yaj / am) - (bi[i] / bm)) * np.log(1.0 + B / Z)
            )

            phi[i] = np.exp(term1 - term2 - term3)
            f_dict[gas] = self.y[i] * phi[i] * P

        return f_dict, phi
