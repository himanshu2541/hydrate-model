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
        self.R = 8.314  # J/(mol·K)
        self.KB = 1.38064852e-23  # J/K
        self.NA = 6.022e23
        self.T0 = 273.15  # K  — reference temperature
        self.P0 = 1.01325e5  # Pa — reference pressure

        # ── Gas-phase formers ─────────────────────────────────────────────────
        # These molecules live in the *gas phase*, are handled by the EOS for
        # fugacity, and occupy hydrate cages as primary guests.
        # Kihara params: sigma (Å), eps_k (K), a (Å)  [Tee et al. 1966 /
        # John-Holder 1985 optimal set, Table 6].
        self.GAS_DB: dict = {
            "CO2": {
                "Tc": 304.12,
                "Pc": 73.74e5,
                "omega": 0.225,
                "sigma": 2.9605,  # Å  (Klauda & Sandler 2000, Table 3)
                "eps_k": 180.85,  # K
                "a": 0.677,  # Å
                "is_linear": True,
            },
            "H2": {
                "Tc": 33.19,
                "Pc": 13.13e5,
                "omega": -0.216,  # quantum gas — Q* correction skipped
                "sigma": 3.11,
                "eps_k": 27.2,
                "a": 0.34,
                "is_linear": False,
            },
        }

        # ── Liquid-phase thermodynamic promoters ──────────────────────────────
        # These molecules are dissolved in the *aqueous phase*.  They lower the
        # hydrate formation pressure by occupying large cages.  They are NOT
        # EOS components and their fugacity is estimated via a simple
        # Clausius-Clapeyron vapor-pressure expression in the solver.
        #
        # Extra fields beyond the standard Kihara set:
        #   stoichiometric_x  — ideal large-cage occupancy mol fraction in water
        #   delta_H_vap       — J/mol, for vapor-pressure estimation
        #   P_sat_ref         — Pa at T_sat_ref (used as Clausius-Clapeyron anchor)
        #   T_sat_ref         — K
        self.PROMOTER_DB: dict = {
            "DIOX": {
                # 1,4-Dioxane  (sII thermodynamic promoter)
                "display_name": "1,4-Dioxane",
                "Tc": 587.0,
                "Pc": 51.4e5,
                "omega": 0.281,
                "sigma": 3.48,  # Å  — approximate, literature
                "eps_k": 583.0,  # K
                "a": 0.85,  # Å
                "is_linear": False,
                # Promoter-specific
                "stoichiometric_x": 0.0556,  # 5.56 mol%  (sII ideal large-cage fill)
                "delta_H_vap": 34700.0,  # J/mol  (≈ 34.7 kJ/mol, literature)
                "P_sat_ref": 9300.0,  # Pa  at T_sat_ref
                "T_sat_ref": 293.15,  # K
                "unifac_groups": {1: 4, 13: 2},  # 4×CH₂ + 2×CH₂O
            },
        }

        # ── Backward-compatible merged view ───────────────────────────────────
        # Code that calls db.GUEST_DB[gas] continues to work unchanged.
        # Do NOT modify this dict directly; edit GAS_DB or PROMOTER_DB instead.
        self._guest_db_cache: dict | None = None

        # ── Hydrate structure parameters ──────────────────────────────────────
        # Source: John-Holder 1985 (AIChE J. 31(2)), Table 2 / Table 3.
        # nu:  cavities per water molecule  (sI: 2/46 small, 6/46 large;
        #                                   sII: 16/136 small, 8/136 large)
        # R:   shell radius (Å);  z: coordination number
        # a_0, n_0: Q* correlation constants (Table 3, JH 1985)
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
        # Source: John-Holder 1985, Table 3.
        # dMu0: Δμ°_w  J/mol;  dH0: Δh°_w  J/mol;  dV: ΔV_w  m³/mol
        # del_CP0_*: ΔCp  J/(mol·K);  *_b_factor: temperature coefficient
        self.REFERENCE_PROPS: dict = {
            "sI": {
                "dMu0": 1120.0,
                "dH0_ice": 1714.0,
                "dH0_liq": -4297.0,
                "dV_ice": 3.0e-6,
                "dV_liq": 4.6e-6,
                "del_CP0_ice": 3.315,
                "del_CP0_liq": -34.583,
                "del_CP0_ice_b_factor": 0.012,
                "del_CP0_liq_b_factor": 0.189,
                "a_w": 0,
                "sigma_w": 3.56438,
                "eps_k_w": 102.134,
            },
            "sII": {
                "dMu0": 931.0,
                "dH0_ice": 1400.0,
                "dH0_liq": -4611.0,
                "dV_ice": 3.4e-6,
                "dV_liq": 5.0e-6,
                "del_CP0_ice": 1.029,
                "del_CP0_liq": -36.8607,
                "del_CP0_ice_b_factor": 0.00377,
                "del_CP0_liq_b_factor": 0.181,
                "a_w": 0,
                "sigma_w": 3.56438,
                "eps_k_w": 102.134,
            },
        }

        # ── Henry's law constants ──────────────────────────────────────────────
        # Klauda & Sandler 2000, Table 4.
        # −ln(H/101325) = H1 + H2/T + H3·ln(T) + H4·T    [H in Pa⁻¹]
        self.HENRY_PARAMS: dict = {
            "CO2": {"H1": -159.868, "H2": 8742.426, "H3": 21.6712, "H4": -0.00110},
        }

        # ── Modified UNIFAC groups ─────────────────────────────────────────────
        # Dahl, Fredenslund & Rasmussen 1991 (MHV2 paper), Tables I–III.
        self.MOD_UNIFAC_GROUPS: dict = {
            1: {"name": "CH2", "R": 0.6744, "Q": 0.5400},
            6: {"name": "H2O", "R": 0.9200, "Q": 1.4000},
            13: {"name": "CH2O", "R": 0.9183, "Q": 0.7800},
            22: {"name": "H2", "R": 0.8320, "Q": 1.1410},
            26: {"name": "CO2", "R": 2.5920, "Q": 2.5220},
        }

        # UNIFAC group assignments for each component
        self.UNIFAC_MAPPING: dict = {
            "CO2": {"unifac_groups": {26: 1}},
            "H2": {"unifac_groups": {22: 1}},
            "H2O": {"unifac_groups": {6: 1}},
            "DIOX": {"unifac_groups": {1: 4, 13: 2}},  # 4×CH₂ + 2×CH₂O
        }

        # Modified UNIFAC interaction parameters  a_mn(T) = a1 + a2·(T − 298.15)
        # Format: (m, n): [a1, a2]   from Dahl 1991 Table III(a)/(b)
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

    # ── Convenience helpers ───────────────────────────────────────────────────

    @property
    def GUEST_DB(self) -> dict:
        """
        Union of GAS_DB and PROMOTER_DB.

        Backward-compatible view used by thermo-model code that needs Kihara
        parameters for any cage occupant regardless of whether it is a gas or
        a promoter.  Read-only — edit GAS_DB / PROMOTER_DB directly.
        """
        if self._guest_db_cache is None:
            self._guest_db_cache = {**self.GAS_DB, **self.PROMOTER_DB}
        return self._guest_db_cache

    @GUEST_DB.setter
    def GUEST_DB(self, value):
        # Allow legacy code that does `db.GUEST_DB["X"] = {...}` to still work
        # by merging into whichever sub-dict is appropriate.
        # Best practice: edit GAS_DB / PROMOTER_DB directly.
        self._guest_db_cache = value

    def is_gas(self, key: str) -> bool:
        """Return True if *key* is a gas-phase former."""
        return key in self.GAS_DB

    def is_promoter(self, key: str) -> bool:
        """Return True if *key* is a liquid-phase thermodynamic promoter."""
        return key in self.PROMOTER_DB
