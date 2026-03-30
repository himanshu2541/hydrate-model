class Database:
    def __init__(self):
        self.R = 8.314
        self.KB = 1.38064852e-23
        self.NA = 6.022e23
        self.T0 = 273.15
        self.P0 = 1.01325e5

        # # Keep False — incompatible with K&S 2000 Kihara params
        # self.USE_Q_STAR: bool = False

        # ── Gas-phase formers ─────────────────────────────────────────────────
        self.GAS_DB: dict = {
            "CO2": {
                "Tc": 304.12,
                "Pc": 73.74e5,
                "omega": 0.225,
                "sigma": 2.9608, 
                "eps_k":  188.97,  # K  
                "a":  0.677,
                "is_linear": True,
            },
            "H2": {
                "Tc": 33.19,
                "Pc": 13.13e5,
                "omega": -0.216,
                "sigma": 2.916,  
                "eps_k": 31.31,  # K
                "a": 0.00,  # Å 
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
            ('CO2', 'CO2'): 0.0,
            ('H2', 'H2'): 0.0,
            ('CO2', 'H2'): -0.017,  # Crucial for accurate CO2-H2 mixture fugacity
            ('H2', 'CO2'): -0.017,  # Keep it symmetric
            # Ensure your other interactions (like water) are defined
            ('CO2', 'H2O'): 0.1896, # Standard PR value for CO2-H2O
            ('H2', 'H2O'): 0.0      # H2/H2O interaction is negligible
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
        self.HENRY_PARAMS: dict = {
            "CO2": {"H1": -159.868, "H2": 8742.426, "H3": 21.6712, "H4": -0.00110},
            "H2":  {"H1":  -13.767, "H2":    772.7,  "H3":    0.0,  "H4":   0.0},
        }

        # ── Modified UNIFAC (Dahl, Fredenslund & Rasmussen 1991) ──────────────
        self.MOD_UNIFAC_GROUPS: dict = {
            1: {"name": "CH2", "R": 0.6744, "Q": 0.5400},
            6: {"name": "H2O", "R": 0.9200, "Q": 1.4000},
            13: {"name": "CH2O", "R": 0.9183, "Q": 0.7800},
            22: {"name": "H2", "R": 0.8320, "Q": 1.1410},
            26: {"name": "CO2", "R": 2.5920, "Q": 2.5220},
        }

        self.UNIFAC_MAPPING: dict = {
            "CO2": {"unifac_groups": {26: 1}},
            "H2": {"unifac_groups": {22: 1}},
            "H2O": {"unifac_groups": {6: 1}},
            "DIOX": {"unifac_groups": {1: 4, 13: 2}},
        }

        self.MOD_UNIFAC_INTERACTIONS: dict = {
            (6, 26): [226.6, -0.2410],
            (26, 6): [1067.0, -0.4180],
            (6, 22): [949.9, -0.3100],
            (22, 6): [1586.0, 3.9240],
            (6, 1): [0.0, 0.0],
            (1, 6): [0.0, 0.0],
            (6, 13): [0.0, 0.0],
            (13, 6): [0.0, 0.0],
            (22, 26): [0.0, 0.0],
            (26, 22): [0.0, 0.0],
            (6, 6): [0.0, 0.0],
            (22, 22): [0.0, 0.0],
            (26, 26): [0.0, 0.0],
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