import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.run import HisimConfig
from app.services import hisim_runner, store, thermal_runner


@pytest.fixture(autouse=True)
def real_engine_config_is_never_touched():
    """No test in this module may rewrite HISIM-SystolicArray/config.py (a leaked write once broke real runs)."""
    from app.services.paths import HISIM_DIR

    path = HISIM_DIR / "config.py"
    before = path.read_bytes()
    yield
    assert path.read_bytes() == before, "a test rewrote the real engine config.py"


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path)
    return tmp_path


def _make(status="completed"):
    record = hisim_runner._new_record(HisimConfig(), None)
    record.status = status
    store.save_record(record)
    return record


def test_load_thermal_maps(runs_dir):
    record = _make()
    with pytest.raises(FileNotFoundError):
        thermal_runner.load_thermal_maps(record.id)
    thermal = runs_dir / record.id / "thermal"
    thermal.mkdir()
    (thermal / "maps.json").write_text(json.dumps({"dies": {"0": {"grid": [[300.0]]}}}))
    assert thermal_runner.load_thermal_maps(record.id)["dies"]["0"]["grid"] == [[300.0]]


def test_api_serves_maps(runs_dir):
    client = TestClient(app)
    ok = _make()
    assert client.get(f"/api/runs/{ok.id}/thermal/maps").status_code == 404
    thermal = runs_dir / ok.id / "thermal"
    thermal.mkdir()
    (thermal / "maps.json").write_text(json.dumps({"dies": {}}))
    assert client.get(f"/api/runs/{ok.id}/thermal/maps").json() == {"dies": {}}


# ---- thermal pass for one already-completed run (dashboard "Run thermal" button) ----
@pytest.fixture()
def no_worker(monkeypatch):
    started = []
    monkeypatch.setattr(thermal_runner, "_start_worker", lambda ids: started.append(ids))
    return started


def test_queue_thermal_queues_a_completed_run(runs_dir, no_worker):
    ok = _make()
    result = thermal_runner.queue_thermal([ok.id])
    assert result["queued"] == [ok.id] and no_worker == [[ok.id]]
    assert store.load_record(ok.id).thermal_status == "queued"


def test_queue_thermal_skips_missing_unfinished_and_busy_runs(runs_dir, no_worker):
    failed = _make("failed")
    busy = _make()
    busy.thermal_status = "running"
    store.save_record(busy)
    result = thermal_runner.queue_thermal([failed.id, busy.id, "nope"])
    assert result["queued"] == [] and no_worker == []
    assert {s["id"]: s["reason"] for s in result["skipped"]} == {
        failed.id: "run is not completed", busy.id: "thermal already running", "nope": "run not found"}


def test_merge_pass_result_updates_record_and_result_file(runs_dir):
    record = _make()
    result_file = runs_dir / record.id / "gui_result.json"
    result_file.write_text(json.dumps({"metrics": {"total_latency_s": 1.0}}))
    thermal_runner.merge_pass_result(record, {"status": "completed", "thermal": {"peak_k": 350.0},
                                              "metrics": {"peak_temperature_k": 350.0}})
    merged = json.loads(result_file.read_text())
    assert merged["metrics"] == {"total_latency_s": 1.0, "peak_temperature_k": 350.0}
    assert merged["thermal"]["peak_k"] == 350.0
    loaded = store.load_record(record.id)
    assert loaded.metrics["peak_temperature_k"] == 350.0 and loaded.thermal_status == "done"


def test_merge_pass_result_records_failure(runs_dir):
    record = _make()
    thermal_runner.merge_pass_result(record, {"thermal": None, "thermal_error": "boom"})
    loaded = store.load_record(record.id)
    assert loaded.thermal_status == "failed" and loaded.thermal_error == "boom"


def test_api_thermal_button_endpoint(runs_dir, no_worker):
    client = TestClient(app)
    ok = _make()
    assert client.post("/api/runs/missing/thermal").status_code == 400
    resp = client.post(f"/api/runs/{ok.id}/thermal")
    assert resp.status_code == 200 and resp.json()["queued"] == [ok.id]
    assert client.post("/api/thermal/batch", json={"run_ids": [ok.id]}).status_code in (404, 405)  # one run only


def test_failed_pass_reports_the_engines_own_message_not_a_traceback(runs_dir, monkeypatch, tmp_path):
    from types import SimpleNamespace

    record = _make()
    engine = tmp_path / "engine"
    engine.mkdir()
    monkeypatch.setattr(hisim_runner, "_validate_mapping_files", lambda config: None)
    monkeypatch.setattr(hisim_runner, "_prepare_run_workspace", lambda target, config: (engine, tmp_path / "inputs"))

    def fake_run(cmd, cwd, env, **kw):
        assert cwd == engine and env["HISIM_GUI_THERMAL_ONLY"] == "1"   # runs in the private engine copy
        (runs_dir / record.id / "thermal_result.json").write_text(
            json.dumps({"status": "failed", "errors": ["Error: ThermalTooLarge: Thermal grid too large (14,276,064 cells)."]}))
        return SimpleNamespace(returncode=1, stdout="", stderr="Traceback (most recent call last):\n ... long ...")

    monkeypatch.setattr(thermal_runner.subprocess, "run", fake_run)
    thermal_runner._run_pass(record.id)
    loaded = store.load_record(record.id)
    assert loaded.thermal_status == "failed"
    assert loaded.thermal_error == "ThermalTooLarge: Thermal grid too large (14,276,064 cells)."
