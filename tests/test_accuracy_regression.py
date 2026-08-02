"""
test_accuracy_regression.py
------------------------------
Runs the solver against every literature dataset in services/literature_data.py
and reports the AAD% (average absolute deviation in equilibrium pressure)
against each EOS model.

This is a *diagnostic* harness, not a correctness proof: the sanity ceiling
below is deliberately generous (it exists to catch outright breakage —
NaN/inf results, exceptions, or wildly diverging predictions — not to claim
the model is "accurate"). Run with `pytest -s` to see the real AAD% numbers;
`python scripts/run_validation_report.py` writes them to validation_report.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrate_project.services.literature_data import PRESET_DATA
from hydrate_project.services.solver_runner import run_model
from hydrate_project.services.metrics import calculate_aad

EOS_NAMES = ["Peng-Robinson", "Soave-Redlich-Kwong", "Patel-Teja"]

# Generous sanity ceiling: catches crashes / NaN / wildly diverging results,
# not "this is accurate". Some literature datasets (notably the 57.9/42.1
# CO2/H2 mixture) are known to deviate far more than this from the model —
# see validation_report.md for the real per-dataset numbers.
SANITY_CEILING_PCT = 200.0


def _cases():
    for name, preset in PRESET_DATA.items():
        for eos in EOS_NAMES:
            yield name, eos


@pytest.mark.parametrize("dataset_name,eos_name", list(_cases()))
def test_solver_vs_literature(dataset_name, eos_name):
    preset = PRESET_DATA[dataset_name]
    T_range = np.array(sorted(set(preset["T (K)"])))

    results, err = run_model(
        preset["gas_comp"], preset["liq_comp"], T_range, [eos_name], lambda m: None
    )
    assert err is None, f"Solver raised an exception: {err!r}"
    assert eos_name in results and not results[eos_name].empty

    df = results[eos_name]
    aad = calculate_aad(df, preset)
    print(f"\n[AAD] {dataset_name} | {eos_name}: {aad:.2f}%")

    assert not np.isnan(aad), "AAD is NaN — solver produced no usable P_eq values."
    assert aad < SANITY_CEILING_PCT, (
        f"AAD {aad:.1f}% exceeds the sanity ceiling of {SANITY_CEILING_PCT}% — "
        f"this looks like outright breakage, not just inaccuracy."
    )
