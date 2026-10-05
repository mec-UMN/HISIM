from app.schemas.run import HisimConfig
from app.services import hisim_runner, store


def test_startup_fails_runs_that_a_dead_server_left_queued_or_running(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path)
    states = {}
    for status in ("queued", "running", "completed"):
        record = hisim_runner._new_record(HisimConfig(), status)
        record.status = status
        store.save_record(record)
        states[status] = record.id
    busy_thermal = hisim_runner._new_record(HisimConfig(), "thermal")
    busy_thermal.status = "completed"
    busy_thermal.thermal_status = "running"
    store.save_record(busy_thermal)

    assert hisim_runner.recover_interrupted_runs() == 3
    for status in ("queued", "running"):
        loaded = store.load_record(states[status])
        assert loaded.status == "failed" and "restarted" in loaded.error
    assert store.load_record(states["completed"]).status == "completed"
    thermal = store.load_record(busy_thermal.id)
    assert thermal.status == "completed" and thermal.thermal_status == "failed" and "restarted" in thermal.thermal_error


def test_runs_owned_by_another_live_server_are_left_alone(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(store, "RUNS_DIR", tmp_path)
    record = hisim_runner._new_record(HisimConfig(), "other server")
    record.status = "running"
    record.owner_pid = os.getppid()          # a live process that is not this one
    store.save_record(record)
    assert hisim_runner.recover_interrupted_runs() == 0
    assert store.load_record(record.id).status == "running"
