from abc import ABC, abstractmethod
import numpy as np

from ..core.fitted_params import FITTED_PARAMS

# k_ij(CO2, H2) is a hand-tuned scalar with no held-out validation (see
# core/fitted_params.py "k_ij_CO2_H2") -- it does not live in
# database.KIJ_DB with the sourced/standard pairs.
_FENCED_KIJ = {frozenset({"CO2", "H2"}): "k_ij_CO2_H2"}


class EquationOfState(ABC):
    def __init__(self, composition, database):
        self.composition = composition
        self.database = database
        self.gases = list(composition.keys())
        self.kij = getattr(self.database, "KIJ_DB", {})

    def _binary_interaction_parameter(self, gas1, gas2):
        fenced = _FENCED_KIJ.get(frozenset({gas1, gas2}))
        if fenced is not None:
            return FITTED_PARAMS[fenced].value
        return self.kij.get((gas1, gas2), self.kij.get((gas2, gas1), 0.0))

    @abstractmethod
    def calc_fugacities(self, T, P) -> tuple[dict, np.ndarray]:
        """Calculate fugacities for each component.
        
        Returns:
            tuple: (fugacities dict, fugacity coefficients array)
        """
        pass

    @abstractmethod
    def calc_Z(self, T, P) -> float:
        """Calculate the compressibility factor Z for the gas mixture."""
        pass