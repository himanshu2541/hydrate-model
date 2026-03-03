class Database:
    def __init__(self):
        self.R = 8.314  # J/(mol.K) - Gas constant
        self.KB = 1.38064852e-23  # J/K - Boltzmann constant
        self.NA = 6.022e23  # Avogadro Number
        self.T0 = 273.15  # K - Reference Temperature
        self.P0 = 1.01325e5  # Pa - Reference Pressure

        # Kihara parameters (sigma, epsilon/k, a) for John-Holder model
        self.GUEST_DB = {
            "CO2": {
                "Tc": 304.12,
                "Pc": 73.74e5,
                "omega": 0.225,
                "sigma": 2.98,
                "eps_k": 170.1,
                "a": 0.677,
                "is_linear": True,
            },
            "H2": {
                "Tc": 33.19,
                "Pc": 13.13e5,
                "omega": -0.216,
                "sigma": 3.11,
                "eps_k": 27.2,
                "a": 0.34,
                "is_linear": False,
            },
            "DIOX": {
                "Tc": 544.0,  # K
                "Pc": 51.0e5,  # Pa
                "omega": 0.23,
                "sigma": 3.48,
                "eps_k": 380.0,
                "a": 0.85,
                "is_linear": False,
            },
        }

        # -----------------------------------------------------------------------------
        # HYDRATE STRUCTURE PROPERTIES (From John-Holder Paper Table 3)
        # -----------------------------------------------------------------------------
        # nu: cavities per water molecule
        # R_c: cavity radius (Angstroms)
        # z: coordination number
        # a_0, n_0: John-Holder correction parameters (Table 3, John-Holder 1985)
        self.STRUCTURE_DB = {
            "sI": {
                "small": {
                    "type": "5^12",
                    "nu": 2 / 46,
                    "shells": {
                        "1": {"R": 3.906, "z": 20},
                        "2": {"R": 6.593, "z": 20},
                        "3": {"R": 8.086, "z": 80},
                    },
                    "a_0": 35.3446,
                    "n_0": 0.973,
                },
                "large": {
                    "type": "5^12 6^2",
                    "nu": 6 / 46,
                    "shells": {
                        "1": {"R": 4.326, "z": 24},
                        "2": {"R": 7.078, "z": 24},
                        "3": {"R": 8.285, "z": 50},
                    },
                    "a_0": 14.1161,
                    "n_0": 0.826,
                },
                "lattice_type": "sI",
            },
            "sII": {
                "small": {
                    "type": "5^12",
                    "nu": 16 / 136,
                    "shells": {
                        "1": {"R": 3.902, "z": 20},
                        "2": {"R": 6.667, "z": 20},
                        "3": {"R": 8.079, "z": 50},
                    },
                    "a_0": 35.3446,
                    "n_0": 0.973,
                },
                "large": {
                    "type": "5^12 6^4",
                    "nu": 8 / 136,
                    "shells": {
                        "1": {"R": 4.682, "z": 28},
                        "2": {"R": 7.464, "z": 28},
                        "3": {"R": 8.782, "z": 50},
                    },
                    "a_0": 782.8469,
                    "n_0": 2.3129,
                },
                "lattice_type": "sII",
            },
        }

        self.REFERENCE_PROPS = {
            "sI": {
                "dMu0": 1108.0,  # J/mol
                "dH0_ice": 1714.0,  # J/mol
                "dH0_liq": -4297,  # J/mol
                "dV_ice": 3.0e-6,  # m3/mol (approx: ~3 cc/mol)
                "dV_liq": 4.6e-6,  # m3/mol (approx: ~4.6 cc/mol)
                "del_CP0_ice": 3.315,  # J/(mol.K) for T < T0
                "del_CP0_liq": -34.583,  # J/(mol.K) for T > T0
                "del_CP0_ice_b_factor": 0.012,
                "del_CP0_liq_b_factor": 0.189,
                "a_w": 0,
                "sigma_w": 3.56438,  # Angstroms
                "eps_k_w": 102.134,  # K
            },
            "sII": {
                "dMu0": 931.0,  # J/mol
                "dH0_ice": 1400.0,  # J/mol
                "dH0_liq": -4611.0,  # J/mol
                "dV_ice": 3.4e-6,  # m3/mol
                "dV_liq": 5.0e-6,  # m3/mol
                "del_CP0_ice": 1.029,  # J/(mol.K) for T < T0
                "del_CP0_liq": -36.8607,
                "del_CP0_ice_b_factor": 0.00377,
                "del_CP0_liq_b_factor": 0.181,
                "a_w": 0,
                "sigma_w": 3.56438,  # Angstroms
                "eps_k_w": 102.134,  # K
            },
        }

        # -----------------------------------------------------------------------------
        # HENRY'S LAW CONSTANT PARAMETERS
        # -----------------------------------------------------------------------------
        # Klauda and Sandler 2000, Table 4
        self.HENRY_PARAMS = {
            "CO2": {"H1": -159.8680, "H2": 8742.426, "H3": 21.6712, "H4": -0.00110}
        }

        self.MOD_UNIFAC_GROUPS = {
            6: {
                "name": "H2O",
                "R": 0.9200,
                "Q": 1.4000,
            },  # Standard Larsen values for Water
            22: {"name": "H2", "R": 0.8320, "Q": 1.1410},  # Dahl Table II
            26: {"name": "CO2", "R": 2.5920, "Q": 2.5220},  # Dahl Table II
            1: {"name": "CH2", "R": 0.6744, "Q": 0.5400},
            13: {"name": "CH2O", "R": 0.9183, "Q": 0.7800},
        }

        self.UNIFAC_MAPPING = {
            "CO2": {"unifac_groups": {26: 1}},
            "H2": {"unifac_groups": {22: 1}},
            "H2O": {"unifac_groups": {6: 1}},
            "DIOX": {"unifac_groups": {1: 4, 13: 2}},  # Dioxane has 4 CH2 and 2 CH2O groups
        }

        # Interaction Parameters a_mn,1 and a_mn,2
        # a_mn(T) = a_mn,1 + a_mn,2 * (T - 298.15)
        # Values extracted from Dahl Table III(a) and III(b)
        # Format: (m, n): [a1, a2] -> Interaction of group m with group n
        self.MOD_UNIFAC_INTERACTIONS = {
            # H2O (6) - CO2 (26) interactions
            (6, 26): [226.6, -0.2410],  # From Table III(a) Col 26, Row 6
            (26, 6): [1067.0, -0.4180],  # From Table III(b) Row 26, Col 6
            # H2O (6) - H2 (22) interactions
            (6, 22): [949.9, -0.3100],  # From Table III(a) Col 22, Row 6
            (22, 6): [1586.0, 3.924],  # From Table III(b) Row 22, Col 6

            # Assuming zero interactions for groups not listed in Dahl's tables, which is common practice when data is unavailable
            (6, 1): [0.0, 0.0], (1, 6): [0.0, 0.0],
            (6, 13): [0.0, 0.0], (13, 6): [0.0, 0.0],

            # Gas-Gas interactions (assumed zero as per Dahl paper text)
            (22, 26): [0.0, 0.0],
            (26, 22): [0.0, 0.0],
            (6, 6): [0.0, 0.0],
            (22, 22): [0.0, 0.0],
            (26, 26): [0.0, 0.0],
        }
