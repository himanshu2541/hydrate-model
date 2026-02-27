from abc import ABC, abstractmethod
import numpy as np

class EquationOfState(ABC):
    def __init__(self, composition, database):
        self.composition = composition
        self.database = database
        self.gases = list(composition.keys())

    @abstractmethod
    def calc_fugacities(self, T, P) -> tuple[dict, np.ndarray]:
        """Calculate fugacities for each component.
        
        Returns:
            tuple: (fugacities dict, fugacity coefficients array)
        """
        pass