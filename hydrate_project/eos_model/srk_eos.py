import numpy as np
from .base import EquationOfState

class SRKEOS(EquationOfState):
    def __init__(self, composition, database):
        super().__init__(composition, database)
        self.y = np.array([composition[gas] for gas in self.gases])
        self.R = self.database.R

    def _binary_interaction_parameter(self, gas1, gas2):
        return 0.0
    
    def _calc_fugacity_coefficients(self, T, P):
        if P < 1.0: return np.ones(len(self.gases))

        n = len(self.gases)
        ai = np.zeros(n)
        bi = np.zeros(n)

        for i, gas in enumerate(self.gases):
            props = self.database.GUEST_DB[gas]
            Tc, Pc, omega = props["Tc"], props["Pc"], props["omega"]
            Tr = T / Tc

            # SRK specific alpha formulation
            m = 0.480 + 1.574 * omega - 0.176 * omega**2
            alpha = (1 + m * (1 - np.sqrt(Tr)))**2
            
            ai[i] = 0.42748 * ((self.R * Tc)**2 / Pc) * alpha
            bi[i] = 0.08664 * (self.R * Tc) / Pc
        
        am = 0.0
        bm = 0.0

        for i in range(n):
            bm += self.y[i] * bi[i]
            for j in range(n):
                kij = self._binary_interaction_parameter(self.gases[i], self.gases[j])
                a_ij = np.sqrt(ai[i] * ai[j]) * (1 - kij)
                am += self.y[i] * self.y[j] * a_ij
        
        A = am * P / (self.R**2 * T**2)
        B = bm * P / (self.R * T)

        # SRK Cubic Equation: Z^3 - Z^2 + (A - B - B^2)Z - AB = 0
        coeffs = [1, -1, (A - B - B**2), -A * B]
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)].real
        valid_roots = [z for z in real_roots if z > B]
        
        if not valid_roots:
            return np.ones(n)
            
        best_Z = valid_roots[0]
        min_G_res = float('inf')
        
        for Z in valid_roots:
            G_res = Z - 1 - np.log(Z - B) - (A / B) * np.log(1 + B / Z)
            
            if G_res < min_G_res:
                min_G_res = G_res
                best_Z = Z
                
        Z = best_Z

        ln_phi = np.zeros(n)
        for i in range(n):
            term1 = (bi[i] / bm) * (Z - 1)
            term2 = np.log(Z - B) if (Z - B) > 0 else -999.0 
            
            sum_ai_aj = 0.0
            for j in range(n):
                kij = self._binary_interaction_parameter(self.gases[i], self.gases[j])
                sum_ai_aj += self.y[j] * np.sqrt(ai[i] * ai[j]) * (1 - kij)
                
            if am > 0 and B > 0:
                # SRK specific fugacity coefficient term
                term3 = (A / B) * (2 * sum_ai_aj / am - bi[i] / bm) * np.log(1 + B / Z)
            else:
                term3 = 0.0
            ln_phi[i] = term1 - term2 - term3
        
        return np.exp(ln_phi)
    
    def calc_fugacities(self, T, P):
        phi = self._calc_fugacity_coefficients(T, P)
        fugacities = {gas: phi[i] * self.y[i] * P for i, gas in enumerate(self.gases)}
        return fugacities, phi