import numpy as np
from .base import EquationOfState

class PREOS(EquationOfState):
    def __init__(self, composition, database):
        super().__init__(composition, database)
        self.y = np.array([composition[gas] for gas in self.gases])
        self.R = self.database.R

    def _binary_interaction_parameter(self, gas1, gas2):
        # For simplicity, we assume kij = 0 for all pairs
        return 0.0
    
    def _calc_fugacity_coefficients(self, T, P):
        # Handle low pressure explicitly
        if P < 1.0: return np.ones(len(self.gases))

        n = len(self.gases)
        ai = np.zeros(n)
        bi = np.zeros(n)

        for i, gas in enumerate(self.gases):
            props = self.database.GUEST_DB[gas]
            Tc, Pc, omega = props["Tc"], props["Pc"], props["omega"]
            Tr, Pr = T / Tc, P / Pc

            # Calculate 'a' and 'b' parameters for Peng-Robinson EOS
            kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
            alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
            ai[i] = 0.45724 * ((self.R * Tc)**2 / Pc) * alpha
            bi[i] = 0.07780 * (self.R * Tc )/ Pc
        
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

        # Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0
        coeffs = [1, -(1 - B), (A - 3*B**2 - 2*B), -(A*B - B**2 - B**3)]
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)].real
        valid_roots = [z for z in real_roots if z > B]
        
        if not valid_roots:
            return np.ones(n)
            
        # Select the root that minimizes Residual Gibbs Free Energy (Stable Phase)
        best_Z = valid_roots[0]
        min_G_res = float('inf')
        
        for Z in valid_roots:
            term3 = (A / (2 * np.sqrt(2) * B)) * np.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))
            G_res = Z - 1 - np.log(Z - B) - term3
            
            if G_res < min_G_res:
                min_G_res = G_res
                best_Z = Z
                
        Z = best_Z

        ln_phi = np.zeros(n)
        for i in range(n):
            term1 = (bi[i] / bm) * (Z - 1)
            term2 = np.log(Z - B) if (Z - B) > 0 else -999.0 # Avoid log of non-positive
            sum_ai_aj = 0.0
            for j in range(n):
                kij = self._binary_interaction_parameter(self.gases[i], self.gases[j])
                sum_ai_aj += self.y[j] * np.sqrt(ai[i] * ai[j]) * (1 - kij)
            if am > 0 and B > 0:
                term3 = (A / (2 * np.sqrt(2) * B)) * (2 * sum_ai_aj / am - bi[i] / bm) * np.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))
            else:
                term3 = 0.0
            ln_phi[i] = term1 - term2 - term3
        
        
        return np.exp(ln_phi)
    
    def calc_fugacities(self, T, P):
        phi = self._calc_fugacity_coefficients(T, P)
        fugacities = {}
        for i, gas in enumerate(self.gases):
            fugacities[gas] = phi[i] * self.y[i] * P

        # print(f"At T={T:.2f}K and P={P/1e6:.4f}MPa: Fugacities: {fugacities}, Fugacity Coefficients: {phi}")
        return fugacities, phi

    def calc_Z(self, T, P):
        """Calculates and returns the compressibility factor (Z) for the mixture."""

        if np.isnan(P) or np.isnan(T) or np.isinf(P) or np.isinf(T):
            return 1.0
        
        if P < 1.0: return 1.0

        n = len(self.gases)
        ai, bi = np.zeros(n), np.zeros(n)

        for i, gas in enumerate(self.gases):
            props = self.database.GUEST_DB[gas]
            Tc, Pc, omega = props["Tc"], props["Pc"], props["omega"]
            Tr, Pr = T / Tc, P / Pc

            kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
            alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
            ai[i] = 0.45724 * ((self.R * Tc)**2 / Pc) * alpha
            bi[i] = 0.07780 * (self.R * Tc) / Pc
        
        am, bm = 0.0, 0.0
        for i in range(n):
            bm += self.y[i] * bi[i]
            for j in range(n):
                kij = self._binary_interaction_parameter(self.gases[i], self.gases[j])
                a_ij = np.sqrt(ai[i] * ai[j]) * (1 - kij)
                am += self.y[i] * self.y[j] * a_ij
        
        A = am * P / (self.R**2 * T**2)
        B = bm * P / (self.R * T)

        # Solve Z cubic equation
        coeffs = [1, -(1 - B), (A - 3*B**2 - 2*B), -(A*B - B**2 - B**3)]
        roots = np.roots(coeffs)
        valid_roots = [z for z in roots[np.isreal(roots)].real if z > B]
        
        if not valid_roots:
            return 1.0
            
        best_Z = valid_roots[0]
        min_G_res = float('inf')
        
        # Find the most stable root
        for Z in valid_roots:
            term3 = (A / (2 * np.sqrt(2) * B)) * np.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))
            G_res = Z - 1 - np.log(Z - B) - term3
            if G_res < min_G_res:
                min_G_res = G_res
                best_Z = Z
                
        return best_Z