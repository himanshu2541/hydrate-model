import numpy as np
from .base import EquationOfState

class PTEOS(EquationOfState):
    def __init__(self, composition, database):
        super().__init__(composition, database)
        self.y = np.array([composition[gas] for gas in self.gases])
        self.R = self.database.R

    def _binary_interaction_parameter(self, gas1, gas2):
        return 0.0
    
    def _calc_fugacity_coefficients(self, T, P):
        if P < 1.0: return np.ones(len(self.gases))

        n = len(self.gases)
        ai, bi, ci = np.zeros(n), np.zeros(n), np.zeros(n)

        for i, gas in enumerate(self.gases):
            props = self.database.GAS_DB[gas]
            Tc, Pc, omega = props["Tc"], props["Pc"], props["omega"]
            Tr = T / Tc

            # PT EOS Constants based on critical compressibility correlation
            zeta_c = 0.329032 - 0.076799 * omega + 0.0211947 * omega**2
            
            # Omega calculation (simplified analytical estimation for PT)
            Omega_c = 1 - 3 * zeta_c
            Omega_a = 3 * zeta_c**2 + 3 * (1 - 2 * zeta_c) * Omega_c + Omega_c**2 + 1 - 3 * zeta_c
            Omega_b = zeta_c # Simplified approximation for b

            # PT Alpha function
            F = 0.452413 + 1.30982 * omega - 0.295937 * omega**2
            alpha = (1 + F * (1 - np.sqrt(Tr)))**2
            
            ai[i] = Omega_a * ((self.R * Tc)**2 / Pc) * alpha
            bi[i] = Omega_b * (self.R * Tc) / Pc
            ci[i] = Omega_c * (self.R * Tc) / Pc
        
        am, bm, cm = 0.0, 0.0, 0.0

        for i in range(n):
            bm += self.y[i] * bi[i]
            cm += self.y[i] * ci[i]
            for j in range(n):
                kij = self._binary_interaction_parameter(self.gases[i], self.gases[j])
                a_ij = np.sqrt(ai[i] * ai[j]) * (1 - kij)
                am += self.y[i] * self.y[j] * a_ij
        
        A = am * P / (self.R**2 * T**2)
        B = bm * P / (self.R * T)
        C = cm * P / (self.R * T)

        # PT Cubic Equation: Z^3 + (C - 1)Z^2 + (A - 2BC - B^2 - B - C)Z + (BC^2 + B^2C + B^2 + BC - AC) = 0
        coeffs = [1, (C - 1), (A - 2*B*C - B**2 - B - C), (B*C**2 + B**2*C + B**2 + B*C - A*C)]
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)].real
        if len(real_roots) == 0:
            return np.ones(n)
        valid_roots = [z for z in real_roots if z > max(B, C)]
        
        if not valid_roots:
            return np.ones(n)
            
        best_Z = valid_roots[0]
        min_G_res = float('inf')
        
        for Z in valid_roots:
            G_res = Z - 1 - np.log(Z - B) - (A / (B + C)) * np.log((Z + B) / (Z - C))
            
            if G_res < min_G_res:
                min_G_res = G_res
                best_Z = Z
                
        Z = best_Z

        ln_phi = np.zeros(n)
        for i in range(n):
            term1 = np.log(Z - B) if (Z - B) > 0 else -999.0
            
            # Fugacity derivation for PT is complex; using standard cubic approximation here
            # For exact analytical derivation, a dedicated mixing rule matrix is usually applied.
            # Below is a streamlined approximation for the residual.
            sum_ai_aj = sum(self.y[j] * np.sqrt(ai[i] * ai[j]) for j in range(n))
            
            term2 = (bi[i] / bm) * (Z - 1)
            # Placeholder for the exact PT integral term to prevent math domain errors on complex roots
            term3 = (A / (B + C)) * (2 * sum_ai_aj / am - bi[i] / bm) * np.log((Z + B) / (Z - C)) if (Z-C) > 0 else 0
            
            ln_phi[i] = term2 - term1 - term3
        
        return np.exp(ln_phi)

    def calc_fugacities(self, T, P):
        phi = self._calc_fugacity_coefficients(T, P)
        fugacities = {gas: phi[i] * self.y[i] * P for i, gas in enumerate(self.gases)}
        return fugacities, phi