class Database:
    def __init__(self):
        self.R = 8.314
        self.KB = 1.38064852e-23
        self.NA = 6.022e23
        self.T0 = 273.15
        self.P0 = 1.01325e5

        # ── K&S 2000 Table 3 corrected Kihara params (Tee et al. 1966) ────────────
        self.KS_KIHARA_PARAMS: dict = {
            "CH4": {"sigma": 3.505, "eps_k": 232.2, "a": 0.28},
            "C2H6": {"sigma": 4.022, "eps_k": 404.3, "a": 0.574},
            "C3H8": {"sigma": 4.519, "eps_k": 493.71, "a": 0.6502},
            "i-C4H10": {"sigma": 4.746, "eps_k": 628.6, "a": 0.859},
            "H2S": {"sigma": 3.607, "eps_k": 459.6, "a": 0.3508},
            "N2": {"sigma": 3.469, "eps_k": 142.1, "a": 0.341},
            "CO2": {"sigma": 3.335, "eps_k": 513.85, "a": 0.677},
            # H2: not in K&S 2000; Tee et al. 1966 spherical-core values
            "H2": {"sigma": 2.9608, "eps_k": 31.7, "a": 0.000},
            # Water (K&S Table 3)
            "H2O": {"sigma": 3.564, "eps_k": 102.134, "a": 0.000},
        }

        # ── K&S 2000 Table 6: empty hydrate lattice vapor pressure (QL1 form) ─────
        # ln(P_sat^β [Pa]) = A*ln(T) + B/T + C + D*T
        # C = 2.7789 for all (fitted to CH4 sI I-H-V only, then held fixed)
        self.KS_VAPOR_PRESSURE_PARAMS: dict = {
            "sI": {
                "H2": {"A": 4.69453, "B": -5345.39, "C": 2.7789, "D": -0.008424},
                "CO2": {"A": 4.6188, "B": -5020.8289, "C": 2.7789, "D": -8.3355e-3},
            },
            "sII": {
                "H2": {"A": 4.69736, "B": -5458.15, "C": 2.7789, "D": -0.009235},
                "CO2": {"A": 4.84222, "B": -5020.8289, "C": 2.7789, "D": -8.3455e-3},
            },
        }

        self.KS_EMP_VAPOR_PRESSURE_PARAMS: dict = {
            "sI": {
                "H2": {"A": 4.6977, "B": -5822.39, "C":2.7897, "D": -0.008024},
                "CO2": {"A": 4.59071, "B": -5345.28, "C": 2.7897, "D": -0.007515},
            },
            "sII": {
                "H2": {"A": 4.99736, "B": -5458.15, "C":2.7897, "D": -0.009235},
                "CO2": {"A": 4.84222, "B": -5621.08, "C": 2.7897, "D": -0.009199},
            },
        }

        self.KS_EMP_LANGMUIR_PARAMS: dict = {
            "sI": {
                "small": {
                    "H2": {"A": -21.6228, "B": 1020.2356, "D": -6733.3429},
                    "CO2": {"A": -24.9824, "B": 2743.7375, "D": 31948.6496},
                },
                "large": {
                    "H2": {"A": -20.2942, "B": 966.9431, "D": -11765.0392},
                    "CO2": {"A": -22.4037, "B": 3171.7604, "D": 26783.0000},
                },
            },
            "sII": {
                "small": {
                    "H2": {"A": -21.6122, "B": 1018.4156, "D": -7082.4990},
                    "CO2": {"A": -25.1752, "B": 3089.4741, "D": 48259.6778},
                },
                "large": {
                    "H2": {"A": -19.6865, "B": 870.0524, "D": -14208.0553},
                    "CO2": {"A": -21.0917, "B": 2405.3662, "D": 28783.0000},
                },
            },
        }

        # ── K&S 2000 Table 5: QL1 vapor pressure of pure water phases ─────────────
        # ln(P_sat [Pa]) = A*ln(T) + B/T + C + D*T
        self.WATER_VP_PARAMS: dict = {
            "ice": {"A": 4.6056, "B": -5501.1243, "C": 2.9446, "D": -8.1431e-3},
            "liquid": {"A": 4.1539, "B": -5500.9332, "C": 7.6537, "D": -16.1277e-3},
        }

        # ── Gas-phase formers ─────────────────────────────────────────────────
        self.GAS_DB: dict = {
            "CO2": {
                "Tc": 304.12,
                "Pc": 73.74e5,
                "omega": 0.225,
                "sigma": self.KS_KIHARA_PARAMS["CO2"]["sigma"],
                "eps_k": self.KS_KIHARA_PARAMS["CO2"]["eps_k"],  # K
                "a": self.KS_KIHARA_PARAMS["CO2"]["a"],  # Å
                "is_linear": True,
            },
            "H2": {
                "Tc": 43.6,
                "Pc": 20.47e5,
                "omega": -0.216,
                "sigma": self.KS_KIHARA_PARAMS["H2"]["sigma"],
                "eps_k": self.KS_KIHARA_PARAMS["H2"]["eps_k"],  # K
                "a": self.KS_KIHARA_PARAMS["H2"]["a"],  # Å
                "is_linear": False,
            },
        }

        # ── Liquid-phase thermodynamic promoters ──────────────────────────────
        self.PROMOTER_DB: dict = {
            "DIOX": {
                "display_name": "1,4-Dioxane",
                "Tc": 587.0,
                "Pc": 51.4e5,
                "omega": 0.281,
                "sigma": 3.48,
                "eps_k": 583.0,
                "a": 0.85,
                "is_linear": False,
                "stoichiometric_x": 0.0556,
                "delta_H_vap": 34700.0,
                "P_sat_ref": 9300.0,
                "T_sat_ref": 293.15,
                "unifac_groups": {1: 4, 13: 2},
            },
        }

        self._guest_db_cache: dict | None = None

        # ── Hydrate structure parameters ──────────────────────────────────────
        # SINGLE-SHELL per cavity type — consistent with K&S 2000 Kihara params.
        # Cavity radii (Rc) and coordination numbers (z):
        #   K&S 2000, Table 1 / van Stackelberg & Müller 1954
        # nu  — cavities per water molecule
        # a_0, n_0 — JH1985 Q* correlation (only used if USE_Q_STAR=True)
        self.STRUCTURE_DB: dict = {
            "sI": {
                "small": {
                    "type": "5^12",
                    "nu": 2 / 46,
                    "shells": {
                        "1": {"R": 3.906, "z": 20},  # K&S 2000 Table 1
                        "2": {"R": 6.593, "z": 20},
                        "3": {"R": 8.086, "z": 50},
                    },
                    "a_0": 35.3446,
                    "n_0": 0.973,
                },
                "large": {
                    "type": "5^12 6^2",
                    "nu": 6 / 46,
                    "shells": {
                        "1": {"R": 4.326, "z": 24},  # K&S 2000 Table 1
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
                        "1": {"R": 3.902, "z": 20},  # K&S 2000 Table 1
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
                        "1": {"R": 4.682, "z": 28},  # K&S 2000 Table 1
                        "2": {"R": 7.464, "z": 28},
                        "3": {"R": 8.782, "z": 50},
                    },
                    "a_0": 782.8469,
                    "n_0": 2.3129,
                },
                "lattice_type": "sII",
            },
        }

        # ── Reference chemical potential parameters ────────────────────────────
        # Source: K&S 2000, Table 2 (self-consistent with their Kihara params).
        #
        # Sign: Δ = (empty hydrate lattice) − (pure liquid water reference)
        # dMu0    : Δμ°_w at T0, P0  [J/mol]
        # dH0_liq : Δh°_w (liquid water basis)  [J/mol]
        # dH0_ice : Δh°_w (ice basis) = dH0_liq + ΔH_fusion_ice ≈ dH0_liq + 6008
        # dV      : ΔV_w  [m³/mol]
        # del_CP0_liq / _b_factor : ΔCp  [J/(mol·K)] and linear coeff [J/(mol·K²)]
        self.REFERENCE_PROPS: dict = {
            "sI": {
                "dMu0": 1120.0,  # J/mol
                "dH0_ice": 1714.0,  # J/mol
                "dH0_liq": -4297.0,  # J/mol
                "dV_ice": 3.0e-6,
                "dV_liq": 4.6e-6,
                "del_CP0_ice": 3.315,
                "del_CP0_liq": -34.582,  # J/(mol·K)
                "del_CP0_ice_b_factor": 0.012,
                "del_CP0_liq_b_factor": 0.189,  # J/(mol·K²)
                "a_w": 0,
                "sigma_w": 3.56438,
                "eps_k_w": 102.134,
            },
            "sII": {
                "dMu0": 931.0,  # J/mol
                "dH0_ice": 1400.0,  # J/mol
                "dH0_liq": -4611.0,  # J/mol
                "dV_ice": 3.4e-6,
                "dV_liq": 5.0e-6,
                "del_CP0_ice": 1.029,
                "del_CP0_liq": -36.861,  # J/(mol·K)
                "del_CP0_ice_b_factor": 0.004,
                "del_CP0_liq_b_factor": 0.181,  # J/(mol·K²)
                "a_w": 0,
                "sigma_w": 3.56438,
                "eps_k_w": 102.134,
            },
        }

        self.KIJ_DB: dict = {
            ("CO2", "CO2"): 0.0,
            ("H2", "H2"): 0.0,
            ("CO2", "H2"): 0.162,  # Crucial for accurate CO2-H2 mixture fugacity
            ("H2", "CO2"): 0.162,  # Keep it symmetric
            # Ensure your other interactions (like water) are defined
            ("CO2", "H2O"): 0.1896,  # Standard PR value for CO2-H2O
            ("H2", "H2O"): 0.0,  # H2/H2O interaction is negligible
        }

        # ── Henry's law: K&S 2000, Table 4 ───────────────────────────────────
        # −ln(H/P0) = H1 + H2/T + H3·ln(T) + H4·T
        #
        # CO2 : Klauda & Sandler 2000, Table 4.
        #
        # H2  : K&S 2000 do not include H2.  Parameters fitted here to
        #        Battino (IUPAC Solubility Data Series, Vol. 5/6, 1981/1984)
        #        experimental mole-fraction solubility data in the hydrate
        #        temperature window (273–300 K):
        #
        #          T = 273.15 K → x_H2 = 1.776×10⁻⁵ → kH = 5.706×10⁹ Pa
        #          T = 278.15 K → x_H2 = 1.628×10⁻⁵ → kH = 6.224×10⁹ Pa
        #          T = 283.15 K → x_H2 = 1.511×10⁻⁵ → kH = 6.706×10⁹ Pa
        #          T = 298.15 K → x_H2 = 1.400×10⁻⁵ → kH = 7.237×10⁹ Pa
        #
        #        Two-parameter least-squares (H3 = H4 = 0) over 273–298 K:
        #          H1 = -13.767,  H2 = 772.7
        #
        #        This gives kH ≈ 5 880 MPa at 276 K, vs. the erroneous
        #        fallback of 1 000 MPa that caused ~1.7 MPa underprediction
        #        of equilibrium pressure in CO2/H2 mixture hydrates.
        #
        #        Root cause of the bug: without this entry, calc_henry_constant
        #        returned 1e9 Pa (1 000 MPa), overestimating x_H2 by ~6×,
        #        which lowered a_w and mu_w, so the bisection solver found
        #        equilibrium at too-low pressure.
        # K&S 2003 Table 1: Henry constants
        self.HENRY_PARAMS: dict = {
            "CO2": {"H1": -159.868, "H2": 8742.426, "H3": 21.6712, "H4": -0.00110},
            "H2": {"H1": -86.8550, "H2": 4178.717, "H3": 10.4935, "H4": 0.00632},
        }

        # ── Modified UNIFAC (Dortmund) Groups ──────────────
        self.MOD_UNIFAC_GROUPS: dict = {
            1: {"name": "CH2", "R": 0.6325, "Q": 0.7081},
            7: {"name": "H2O", "R": 1.7334, "Q": 2.4561},
            13: {"name": "CH2O", "R": 1.1434, "Q": 1.2495},
        }

        self.UNIFAC_MAPPING: dict = {
            "H2O": {"unifac_groups": {7: 1}},
            "DIOX": {"unifac_groups": {1: 2, 13: 2}},
        }

        # [a, b, c] parameters for Dortmund temperature dependence
        self.MOD_UNIFAC_INTERACTIONS: dict = {
            # Alkane (1) & Water (7)
            (7, 1): [-17.253, 0.8389, 0.9021e-3],
            (1, 7): [1391.3, -3.6156, 0.1144e-2],
            # Alkane (1) & Ether (13)
            (1, 13): [233.10, -0.3155, 0.0],
            (13, 1): [-9.6540, -0.3242e-1, 0.0],
            # Ether (13) & Water (7)
            (13, 7): [140.70, 0.5679e-1, 0.0],
            (7, 13): [-197.50, 0.1766, 0.0],
        }

    @property
    def GUEST_DB(self) -> dict:
        if self._guest_db_cache is None:
            self._guest_db_cache = {**self.GAS_DB, **self.PROMOTER_DB}
        return self._guest_db_cache

    @GUEST_DB.setter
    def GUEST_DB(self, value):
        self._guest_db_cache = value

    def is_gas(self, key: str) -> bool:
        return key in self.GAS_DB

    def is_promoter(self, key: str) -> bool:
        return key in self.PROMOTER_DB
