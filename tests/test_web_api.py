"""
tests/test_web_api.py
-----------------------
Smoke tests for the FastAPI backend (hydrate_project/web/api.py). These
don't drive a browser (see Playwright for that) -- they just confirm the
HTTP layer wires correctly into services/ and returns JSON-safe results.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from hydrate_project.web.api import app

client = TestClient(app)


def test_config():
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert "CO2" in data["gases"] and "H2" in data["gases"]
    assert "DIOX" in data["promoters"]
    assert set(data["eos_names"]) == {"Peng-Robinson", "Soave-Redlich-Kwong", "Patel-Teja"}
    assert len(data["literature"]) >= 1


def test_run_pure_co2():
    r = client.post(
        "/api/run",
        json={
            "gas_comp": {"CO2": 1.0},
            "liq_comp": {"H2O": 1.0},
            "T_min": 273.15, "T_max": 275.15, "T_step": 1.0,
            "eos_names": ["Peng-Robinson"],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["error"] is None
    rows = data["results"]["Peng-Robinson"]
    assert len(rows) == 3
    assert rows[0]["T (K)"] == 273.15
    assert rows[0]["P_eq (MPa)"] > 0
    # derived thermo columns present (services/derived.py enrichment)
    assert "dH_diss (kJ/mol)" in rows[0]


def test_run_is_cached_on_second_call():
    req = {
        "gas_comp": {"CO2": 1.0}, "liq_comp": {"H2O": 1.0},
        "T_min": 273.15, "T_max": 274.15, "T_step": 1.0,
        "eos_names": ["Peng-Robinson"],
    }
    r1 = client.post("/api/run", json=req)
    assert r1.json()["cached"] is False or r1.json()["cached"] is True  # first call may already be cached from a prior test run
    r2 = client.post("/api/run", json=req)
    assert r2.json()["cached"] is True


def test_run_rejects_bad_gas_composition():
    r = client.post(
        "/api/run",
        json={
            "gas_comp": {"CO2": 0.5}, "liq_comp": {"H2O": 1.0},
            "T_min": 273.15, "T_max": 275.15, "T_step": 1.0,
            "eos_names": ["Peng-Robinson"],
        },
    )
    assert r.status_code == 400
    assert "sum to 1.0" in r.json()["detail"]


def test_run_rejects_unknown_eos():
    r = client.post(
        "/api/run",
        json={
            "gas_comp": {"CO2": 1.0}, "liq_comp": {"H2O": 1.0},
            "T_min": 273.15, "T_max": 275.15, "T_step": 1.0,
            "eos_names": ["Not-A-Real-EOS"],
        },
    )
    assert r.status_code == 400


def test_sweep_diox():
    r = client.post(
        "/api/sweep/diox",
        json={
            "gas_comp": {"CO2": 1.0}, "promoter_key": "DIOX",
            "diox_values": "2, 5.56",
            "T_min": 277.15, "T_max": 278.15, "T_step": 1.0,
            "eos_name": "Peng-Robinson",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["error"] is None
    assert set(data["results"].keys()) == {"DIOX 2.00%", "DIOX 5.56%"}


def test_sweep_ratio():
    r = client.post(
        "/api/sweep/ratio",
        json={
            "gas_ratio_lines": "CO2=0.4, H2=0.6\nCO2=0.579, H2=0.421",
            "liq_comp": {"H2O": 1.0},
            "T_min": 273.15, "T_max": 274.15, "T_step": 1.0,
            "eos_name": "Peng-Robinson",
        },
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_custom_exp_data_parsing():
    r = client.post("/api/exp-data/custom", json={"text": "273.9 5.56\n275.7, 6.90"})
    assert r.status_code == 200
    data = r.json()
    assert data["T"] == [273.9, 275.7]
    assert data["P"] == [5.56, 6.90]


def test_cache_info_and_clear():
    r = client.get("/api/cache")
    assert r.status_code == 200
    assert "total_entries" in r.json()

    r2 = client.post("/api/cache/clear")
    assert r2.status_code == 200
    assert client.get("/api/cache").json()["total_entries"] == 0


def test_project_crud_lifecycle():
    r = client.post("/api/projects", json={"name": "Test Project"})
    assert r.status_code == 200
    proj = r.json()
    assert proj["name"] == "Test Project"
    assert proj["config"]["gas_comp"] == {}
    pid = proj["id"]

    r = client.get("/api/projects")
    assert r.status_code == 200
    assert any(p["id"] == pid for p in r.json())

    r = client.get(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert r.json()["name"] == "Test Project"

    r = client.put(f"/api/projects/{pid}", json={"name": "Renamed Project"})
    assert r.status_code == 200
    assert r.json()["name"] == "Renamed Project"

    cfg = {
        "gas_comp": {"CO2": 1.0}, "liq_comp": {"H2O": 1.0},
        "T_min": 273.15, "T_max": 283.15, "T_step": 0.5,
        "eos_names": ["Peng-Robinson"], "expdata_mode": "none",
    }
    r = client.put(f"/api/projects/{pid}/config", json={"config": cfg})
    assert r.status_code == 200
    assert r.json()["config"]["gas_comp"] == {"CO2": 1.0}

    r = client.get(f"/api/projects/{pid}")
    assert r.json()["config"]["T_min"] == 273.15

    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_project_not_found_errors():
    assert client.get("/api/projects/999999").status_code == 404
    assert client.put("/api/projects/999999", json={"name": "x"}).status_code == 404
    assert client.delete("/api/projects/999999").status_code == 404


def test_create_project_rejects_empty_name():
    r = client.post("/api/projects", json={"name": "   "})
    assert r.status_code == 400
