"""
web/api.py
-----------
FastAPI backend for the browser UI. This is a thin wrapper: all physics and
orchestration logic stays in services/ (already tkinter-free and reused
as-is from the Tk launcher) -- this module only translates HTTP <-> those
functions and serves the static frontend.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from hydrate_project.core.database import Database
from hydrate_project.services import validators as _validators
from hydrate_project.services.cache import get_cache
from hydrate_project.services.derived import add_dissociation_thermo
from hydrate_project.services.literature_data import PRESET_DATA
from hydrate_project.services.projects import get_project_store
from hydrate_project.services.solver_runner import EOS_MAP, run_model
from hydrate_project.services.sweep_runner import run_diox_sweep, run_gas_ratio_sweep
from hydrate_project.services.metrics import calculate_aad
from .schemas import (
    CacheEntry,
    CacheInfoResponse,
    ConfigResponse,
    CreateProjectRequest,
    CustomExpDataRequest,
    CustomExpDataResponse,
    LiteratureDataset,
    ProjectDetail,
    ProjectSummary,
    PromoterInfo,
    RenameProjectRequest,
    RunRequest,
    RunResponse,
    SaveProjectConfigRequest,
    SweepDioxRequest,
    SweepRatioRequest,
    SweepResponse,
)

log = logging.getLogger(__name__)

app = FastAPI(title="Hydrate Equilibrium Model")

EOS_NAMES = list(EOS_MAP.keys())


# ── helpers ────────────────────────────────────────────────────────────────


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of records (NaN -> null, not the bare
    float NaN that json.dumps would emit as invalid-JSON `NaN` literal)."""
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _results_to_json(results: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    return {
        name: _df_to_records(add_dissociation_thermo(df))
        for name, df in results.items()
    }


def _default_sf_columns(gas_names: list[str]) -> list[str]:
    gas_names = list(gas_names)
    cols = [f"Ideal_SF_{g}" for g in gas_names]
    if len(gas_names) >= 2:
        cols.insert(0, f"SF_{gas_names[0]}_{gas_names[1]}")
    return cols or ["P_eq (MPa)"]


def _cache_entry(row: dict) -> CacheEntry:
    return CacheEntry(
        id=row["id"],
        gas=row["gas"],
        liq=row["liq"],
        T_min=row["T_min"],
        T_max=row["T_max"],
        T_step=row["T_step"],
        eos=row["eos"],
        inserted_at=row["inserted_at"],
        accessed_at=row["accessed_at"],
    )


# ── config / reference data ─────────────────────────────────────────────────


@app.get("/api/config", response_model=ConfigResponse)
def get_config():
    db = Database()
    promoters = {
        k: PromoterInfo(
            display_name=v.get("display_name", k),
            stoichiometric_x=v.get("stoichiometric_x", 0.05),
            hint=f"Stoichiometric: {v.get('stoichiometric_x', 0.05) * 100:.2f} mol%",
        )
        for k, v in db.PROMOTER_DB.items()
    }
    literature = {
        name: LiteratureDataset(T=preset["T (K)"], P=preset["P_eq (MPa)"])
        for name, preset in PRESET_DATA.items()
    }
    return ConfigResponse(
        gases=list(db.GAS_DB.keys()),
        promoters=promoters,
        eos_names=EOS_NAMES,
        literature=literature,
    )


@app.post("/api/exp-data/custom", response_model=CustomExpDataResponse)
def parse_custom_exp_data(req: CustomExpDataRequest):
    try:
        parsed = _validators.parse_custom_exp_data(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if parsed is None:
        return CustomExpDataResponse()
    return CustomExpDataResponse(T=parsed["T (K)"], P=parsed["P_eq (MPa)"])


# ── run ──────────────────────────────────────────────────────────────────────


@app.post("/api/run", response_model=RunResponse)
def run(req: RunRequest):
    try:
        _validators.validate_gas_composition(req.gas_comp)
        T_range = _validators.build_T_range(req.T_min, req.T_max, req.T_step)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not req.eos_names:
        raise HTTPException(status_code=400, detail="Select at least one EOS model.")
    unknown = [n for n in req.eos_names if n not in EOS_MAP]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown EOS model(s): {unknown}")

    cache = get_cache()
    cached = cache.get(req.gas_comp, req.liq_comp, T_range, req.eos_names)
    if cached is not None:
        return RunResponse(results=_results_to_json(cached), cached=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, err = run_model(
            req.gas_comp, req.liq_comp, T_range, req.eos_names, lambda _m: None
        )

    if err is not None:
        return RunResponse(results={}, cached=False, error=str(err))

    try:
        cache.put(req.gas_comp, req.liq_comp, T_range, req.eos_names, results)
    except Exception as exc:
        log.warning("Cache write failed: %s", exc)

    return RunResponse(results=_results_to_json(results), cached=False)


# ── sweeps ───────────────────────────────────────────────────────────────────


@app.post("/api/sweep/diox", response_model=SweepResponse)
def sweep_diox(req: SweepDioxRequest):
    try:
        T_range = _validators.build_T_range(req.T_min, req.T_max, req.T_step)
        fractions = _validators.parse_diox_fraction_list(req.diox_values)
        _validators.validate_gas_composition(req.gas_comp)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, err = run_diox_sweep(
            req.gas_comp, req.promoter_key, fractions, T_range, req.eos_name, lambda _m: None
        )

    default_cols = _default_sf_columns(list(req.gas_comp.keys()))
    if err is not None:
        return SweepResponse(results={}, series_label="compositions", default_cols=default_cols, error=str(err))
    return SweepResponse(
        results=_results_to_json(results), series_label="compositions", default_cols=default_cols
    )


@app.post("/api/sweep/ratio", response_model=SweepResponse)
def sweep_ratio(req: SweepRatioRequest):
    try:
        T_range = _validators.build_T_range(req.T_min, req.T_max, req.T_step)
        gas_comp_list = _validators.parse_gas_ratio_lines(req.gas_ratio_lines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, err = run_gas_ratio_sweep(
            gas_comp_list, req.liq_comp, T_range, req.eos_name, lambda _m: None
        )

    default_cols = _default_sf_columns(list(gas_comp_list[0].keys()))
    if err is not None:
        return SweepResponse(results={}, series_label="compositions", default_cols=default_cols, error=str(err))
    return SweepResponse(
        results=_results_to_json(results), series_label="compositions", default_cols=default_cols
    )


# ── cache ────────────────────────────────────────────────────────────────────


@app.get("/api/cache", response_model=CacheInfoResponse)
def cache_info():
    cache = get_cache()
    info = cache.info()
    entries = [_cache_entry(row) for row in cache.list_entries()]
    return CacheInfoResponse(
        total_entries=info.total_entries,
        max_entries=info.max_entries,
        db_path=info.db_path,
        last_access=info.last_access,
        hits_session=info.hits_session,
        misses_session=info.misses_session,
        entries=entries,
    )


@app.post("/api/cache/clear")
def cache_clear():
    get_cache().clear()
    return {"ok": True}


# ── projects ─────────────────────────────────────────────────────────────────


def _project_summary(row: dict) -> ProjectSummary:
    return ProjectSummary(
        id=row["id"], name=row["name"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _project_detail(row: dict) -> ProjectDetail:
    return ProjectDetail(**_project_summary(row).model_dump(), config=row["config"])


@app.get("/api/projects", response_model=list[ProjectSummary])
def list_projects():
    return [_project_summary(r) for r in get_project_store().list()]


@app.post("/api/projects", response_model=ProjectDetail)
def create_project(req: CreateProjectRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name cannot be empty.")
    return _project_detail(get_project_store().create(name))


@app.get("/api/projects/{project_id}", response_model=ProjectDetail)
def get_project(project_id: int):
    row = get_project_store().get(project_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return _project_detail(row)


@app.put("/api/projects/{project_id}", response_model=ProjectSummary)
def rename_project(project_id: int, req: RenameProjectRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name cannot be empty.")
    if get_project_store().get(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    get_project_store().rename(project_id, name)
    return _project_summary(get_project_store().get(project_id))


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: int):
    if get_project_store().get(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    get_project_store().delete(project_id)
    return {"ok": True}


@app.put("/api/projects/{project_id}/config", response_model=ProjectDetail)
def save_project_config(project_id: int, req: SaveProjectConfigRequest):
    if get_project_store().get(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    get_project_store().update_config(project_id, req.config.model_dump())
    return _project_detail(get_project_store().get(project_id))


# ── static frontend ──────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"


def _mount_vendor_plotly():
    try:
        import plotly

        vendor_dir = Path(plotly.__file__).parent / "package_data"
        if (vendor_dir / "plotly.min.js").exists():
            app.mount("/vendor", StaticFiles(directory=vendor_dir), name="vendor")
            return
    except Exception as exc:
        log.warning("Could not mount vendored plotly.min.js: %s", exc)
    log.warning(
        "plotly.min.js not found; the Plot Builder chart will not render. "
        "Run `uv sync` (plotly is a declared dependency, used only for its "
        "bundled JS asset)."
    )


_mount_vendor_plotly()
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
