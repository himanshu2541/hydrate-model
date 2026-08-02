"""
tests/test_correlations.py
---------------------------
Regression targets from KS_MODEL_SPEC.md §7. These catch the whole class of
transcription errors (dropped cube, flipped sign, Pa/MPa mix-up) in
milliseconds -- if one of these fails, do not "fix" correlations.py to match;
re-check against the spec first.
"""

from __future__ import annotations

import pytest

from hydrate_project.core import correlations as corr
from hydrate_project.core.database import Database


@pytest.mark.parametrize(
    "T,expected_pa",
    [
        (263.15, 285.5),
        (273.15, 609.7),
        (283.15, 1227.0),
        (298.15, 3172.4),
    ],
)
def test_P_sat_liquid_water(T, expected_pa):
    got = corr.P_sat_liquid_water(T)
    assert got == pytest.approx(expected_pa, rel=0.01)


@pytest.mark.parametrize(
    "T,expected_pa",
    [
        (263.15, 260.5),
        (273.15, 612.9),
    ],
)
def test_P_sat_ice(T, expected_pa):
    got = corr.P_sat_ice(T)
    assert got == pytest.approx(expected_pa, rel=0.01)


def test_V_empty_hydrate_sI():
    got = corr.V_empty_hydrate(273.15, 0.1, "sI")
    assert got == pytest.approx(2.2669e-5, rel=0.001)


def test_V_empty_hydrate_sII():
    got = corr.V_empty_hydrate(273.15, 0.1, "sII")
    assert got == pytest.approx(2.3010e-5, rel=0.001)


def test_V_liquid_water():
    got = corr.V_liquid_water(273.15, 0.1)
    assert got == pytest.approx(1.8019e-5, rel=0.001)


def test_V_ice():
    got = corr.V_ice(273.15)
    assert got == pytest.approx(1.9649e-5, rel=0.001)


@pytest.mark.parametrize(
    "T,expected_mpa",
    [
        (273.15, 73.5),
        (298.15, 165.8),
    ],
)
def test_henry_constant_CO2(T, expected_mpa):
    db = Database()
    got_mpa = corr.henry_constant(T, db.HENRY_PARAMS["CO2"], P0=db.P0) / 1e6
    assert got_mpa == pytest.approx(expected_mpa, rel=0.02)


def test_R_equals_kB_times_NA():
    db = Database()
    assert db.R == pytest.approx(db.KB * db.NA, rel=1e-9)
