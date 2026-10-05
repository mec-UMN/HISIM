import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.run import HisimConfig
from app.services import demo, hisim_runner, store


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    bundle = tmp_path / "demo"
    monkeypatch.setattr(store, "RUNS_DIR", runs)
    monkeypatch.setattr(demo, "RUNS_DIR", runs)
    monkeypatch.setattr(demo, "DEMO_DIR", bundle)
    return runs, bundle


def _bundle(bundle):
    record = hisim_runner._new_record(HisimConfig(), "Demo · showcase")
    record.status = "completed"
    record.stdout_path = "/somewhere/else/runs/x/stdout.log"
    run_dir = bundle / "runs" / record.id
    run_dir.mkdir(parents=True)
    (run_dir / "record.json").write_text(record.model_dump_json())
    (run_dir / "gui_result.json").write_text(json.dumps({"metrics": {"total_latency_s": 1.0}}))
    manifest = {"showcase": record.id, "thermal": record.id, "sweep_groups": [], "runs": [record.id]}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return record.id


def test_load_demo_installs_the_bundle_and_is_idempotent(dirs):
    runs, bundle = dirs
    run_id = _bundle(bundle)
    manifest = demo.load_demo()
    assert manifest["showcase"] == run_id
    assert (runs / run_id / "gui_result.json").exists()
    loaded = store.load_record(run_id)
    assert loaded.stdout_path == str(runs / run_id / "stdout.log")      # paths point at the installed copy
    demo.load_demo()                                                    # second load is harmless
    assert store.load_record(run_id).name == "Demo · showcase"


def test_missing_bundle_is_a_clear_404(dirs):
    client = TestClient(app)
    response = client.post("/api/demo/load")
    assert response.status_code == 404
    assert "build_demo.py" in response.json()["detail"]


def test_api_load_demo(dirs):
    _, bundle = dirs
    run_id = _bundle(bundle)
    response = TestClient(app).post("/api/demo/load")
    assert response.status_code == 200 and response.json()["showcase"] == run_id
