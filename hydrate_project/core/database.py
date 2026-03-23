"""
-----------
Central parameter store for the hydrate equilibrium model.

Component taxonomy
------------------
GAS_DB      — molecules that appear in the **gas phase** and occupy hydrate
              cages as the primary formers (CO₂, H₂).
              These are passed to the EOS for fugacity calculation and to
              the John-Holder model for Langmuir constant integration.

PROMOTER_DB — molecules added to the **liquid phase** that shift the
              hydrate equilibrium to milder conditions.  They occupy large
              cages and are handled by the liquid-activity / promoter
              pathway in the equilibrium solver (e.g. 1,4-Dioxane).
              Promoters still need Kihara parameters (cavity potential)
              and UNIFAC groups (activity coefficient), but they are never
              used as EOS components or as gas-phase fugacity sources.

GUEST_DB    — read-only property that returns the *union* of GAS_DB and
              PROMOTER_DB.  Kept for backward-compatibility with any solver
              or thermo-model code that iterates over all cage occupants.

All other dictionaries (STRUCTURE_DB, REFERENCE_PROPS, etc.) are unchanged.
"""


class Database:
    def __init__(self):
        self.R = 8.314
        self.KB = 1.38064852e-23
        self.NA = 6.022e23
        self.T0 = 273.15
        self.P0 = 1.01325e5

        # Keep False — see module docstring
        self.USE_Q_STAR: bool = False

        # ── Gas-phase formers ─────────────────────────────────────────────────
        # CO2 Kihara: Klauda & Sandler 2000, Table 3
        # H2  Kihara: Munck 1988 / Kumar 2006 / Belandria 2011
        # Previous bad H2 values: σ=3.11, ε/k=27.2, a=0.34
        self.GAS_DB: dict = {
            "CO2": {
                "Tc": 304.12,
                "Pc": 73.74e5,
                "omega": 0.225,
                "sigma": 2.9605,
                "eps_k": 169.09,
                "a": 0.677,  
                "is_linear": True,
            },
            "H2": {
                "Tc": 33.19,
                "Pc": 13.13e5,
                "omega": -0.216,
                "sigma": 2.641,  
                "eps_k": 30.15,
                "a": 0.000, 
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
        # Radii and z-values: Klauda & Sandler 2000, Table 1
        # a_0, n_0: JH1985 Q* correlation (used only when USE_Q_STAR=True)
        self.STRUCTURE_DB: dict = {
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

        # ── Reference chemical potential parameters ────────────────────────────
        # Sign: Δ = (empty hydrate lattice) − (liquid water or ice)
        # sI  — Holder et al. 1988 / Parrish & Prausnitz 1972
        # sII — Sloan & Koh 2008, Table 4-1
        # dH0_ice ≈ dH0_liq + 6008 (ice-fusion enthalpy at T0)
        self.REFERENCE_PROPS: dict = {
            "sI": {
                "dMu0": 1264.0,  # J/mol  (was 1120)
                "dH0_ice": 1300.0,  # J/mol  (≈ -4858 + 6008 = 1150; rounding)
                "dH0_liq": -4858.0,  # J/mol  (was -4297)
                "dV_ice": 3.0e-6,
                "dV_liq": 4.6e-6,
                "del_CP0_ice": 3.315,
                "del_CP0_liq": -37.32,  # J/(mol·K)  (was -34.583)
                "del_CP0_ice_b_factor": 0.012,
                "del_CP0_liq_b_factor": 0.179,  # J/(mol·K²)  (was 0.189)
                "a_w": 0,
                "sigma_w": 3.56438,
                "eps_k_w": 102.134,
            },
            "sII": {
                "dMu0": 883.0,  # J/mol  (was 931)
                "dH0_ice": 200.0,  # J/mol
                "dH0_liq": -5931.0,  # J/mol  (was -4611)
                "dV_ice": 3.4e-6,
                "dV_liq": 5.0e-6,
                "del_CP0_ice": 1.029,
                "del_CP0_liq": -41.07,  # J/(mol·K)  (was -36.86)
                "del_CP0_ice_b_factor": 0.00377,
                "del_CP0_liq_b_factor": 0.155,  # J/(mol·K²)  (was 0.181)
                "a_w": 0,
                "sigma_w": 3.56438,
                "eps_k_w": 102.134,
            },
        }

        # ── Henry's law: Klauda & Sandler 2000, Table 4 ───────────────────────
        # −ln(H/P0) = H1 + H2/T + H3·ln(T) + H4·T
        self.HENRY_PARAMS: dict = {
            "CO2": {"H1": -159.868, "H2": 8742.426, "H3": 21.6712, "H4": -0.00110},
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
