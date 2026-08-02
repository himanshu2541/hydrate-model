"""Pydantic request/response models for the web API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class RunRequest(BaseModel):
    gas_comp: dict[str, float]
    liq_comp: dict[str, float]
    T_min: float
    T_max: float
    T_step: float
    eos_names: list[str]


class RunResponse(BaseModel):
    results: dict[str, list[dict]]
    cached: bool
    error: Optional[str] = None


class SweepDioxRequest(BaseModel):
    gas_comp: dict[str, float]
    promoter_key: str
    diox_values: str
    T_min: float
    T_max: float
    T_step: float
    eos_name: str


class SweepRatioRequest(BaseModel):
    gas_ratio_lines: str
    liq_comp: dict[str, float]
    T_min: float
    T_max: float
    T_step: float
    eos_name: str


class SweepResponse(BaseModel):
    results: dict[str, list[dict]]
    series_label: str
    default_cols: list[str]
    error: Optional[str] = None


class CustomExpDataRequest(BaseModel):
    text: str


class CustomExpDataResponse(BaseModel):
    T: Optional[list[float]] = None
    P: Optional[list[float]] = None


class PromoterInfo(BaseModel):
    display_name: str
    stoichiometric_x: float
    hint: str


class LiteratureDataset(BaseModel):
    T: list[float]
    P: list[float]


class ConfigResponse(BaseModel):
    gases: list[str]
    promoters: dict[str, PromoterInfo]
    eos_names: list[str]
    literature: dict[str, LiteratureDataset]


class CacheEntry(BaseModel):
    id: int
    gas: dict[str, float]
    liq: dict[str, float]
    T_min: float
    T_max: float
    T_step: float
    eos: list[str]
    inserted_at: str
    accessed_at: Optional[str] = None


class CacheInfoResponse(BaseModel):
    total_entries: int
    max_entries: int
    db_path: str
    last_access: Optional[str] = None
    hits_session: int
    misses_session: int
    entries: list[CacheEntry]


class ProjectConfig(BaseModel):
    gas_comp: dict[str, float] = {}
    liq_comp: dict[str, float] = {}
    T_min: Optional[float] = None
    T_max: Optional[float] = None
    T_step: Optional[float] = None
    eos_names: list[str] = []
    expdata_mode: str = "none"
    expdata_preset: Optional[str] = None
    expdata_custom_text: Optional[str] = None


class ProjectSummary(BaseModel):
    id: int
    name: str
    created_at: str
    updated_at: str


class ProjectDetail(ProjectSummary):
    config: ProjectConfig


class CreateProjectRequest(BaseModel):
    name: str


class RenameProjectRequest(BaseModel):
    name: str


class SaveProjectConfigRequest(BaseModel):
    config: ProjectConfig
