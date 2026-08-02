"""
tests/test_integrity.py
-------------------------
Guards the fenced parameter layer (KS_MODEL_SPEC.md §6): every regressed /
un-sourced value lives in core/fitted_params.py with a mandatory provenance
record, and does not leak into the sourced-constants modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hydrate_project.core.fitted_params import FITTED_PARAMS

SOURCED_MODULES = [
    Path("hydrate_project/core/database.py"),
    Path("hydrate_project/core/correlations.py"),
]


@pytest.mark.parametrize("path", SOURCED_MODULES, ids=lambda p: p.name)
def test_no_fitted_params_import_in_sourced_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "fitted_params" in node.module:
            pytest.fail(f"{path} imports fitted_params -- fitted values must not leak into sourced tables")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "fitted_params" in alias.name:
                    pytest.fail(f"{path} imports fitted_params -- fitted values must not leak into sourced tables")


@pytest.mark.parametrize("symbol,param", list(FITTED_PARAMS.items()))
def test_fitted_param_has_provenance(symbol, param):
    assert param.fitted_to.strip(), f"{symbol}: fitted_to must not be empty"
    assert param.date, f"{symbol}: date must be set"


@pytest.mark.parametrize("symbol,param", list(FITTED_PARAMS.items()))
def test_fitted_param_value_within_bounds(symbol, param):
    if param.bounds is not None and isinstance(param.value, (int, float)):
        lo, hi = param.bounds
        assert lo < param.value < hi, f"{symbol}={param.value} outside bounds {param.bounds}"


@pytest.mark.parametrize("symbol,param", list(FITTED_PARAMS.items()))
def test_no_fabricated_test_aard(symbol, param):
    """If a held-out AARD is reported, a train AARD must exist too (and
    vice versa is fine -- several entries here are literature citations or
    ad hoc tunes with neither, which is honest, not a violation)."""
    if param.aard_test is not None:
        assert param.aard_train is not None, (
            f"{symbol}: has aard_test but no aard_train -- that's backwards"
        )
