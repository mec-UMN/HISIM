from app.schemas.run import HisimConfig, SweepCreate
from app.services import hisim_runner


def test_thermal_sim_defaults_off_and_is_written_to_config():
    assert HisimConfig().thermal_sim is False
    assert HisimConfig().to_config_values()["THERMAL_SIM"] is False
    assert HisimConfig(thermal_sim=True).to_config_values()["THERMAL_SIM"] is True


def test_record_defaults_have_no_thermal():
    record = hisim_runner._new_record(HisimConfig(), None)
    assert record.thermal_status == "none" and record.thermal_error is None


def test_sweep_children_never_run_thermal(monkeypatch):
    monkeypatch.setattr(hisim_runner, "_validate_mapping_files", lambda c: None)
    monkeypatch.setattr(hisim_runner, "available_models", lambda: [])
    monkeypatch.setattr(hisim_runner, "save_record", lambda r: None)
    monkeypatch.setattr(hisim_runner.threading, "Thread", lambda **kw: type("T", (), {"start": lambda s: None})())
    payload = SweepCreate(config=HisimConfig(thermal_sim=True), knob="stack_count", values=[1, 2])
    records = hisim_runner.create_sweep(payload)
    assert [r.config.thermal_sim for r in records] == [False, False]


def test_apply_thermal_payload_sets_metrics_and_status():
    record = hisim_runner._new_record(HisimConfig(thermal_sim=True), None)
    hisim_runner.apply_thermal_payload(record, {"thermal": {"peak_k": 350.0}, "metrics": {
        "peak_temperature_k": 350.0, "avg_temperature_k": 330.0, "min_temperature_k": 320.0}})
    assert record.thermal_status == "done"
    assert record.metrics["peak_temperature_k"] == 350.0


def test_apply_thermal_payload_records_failure():
    record = hisim_runner._new_record(HisimConfig(thermal_sim=True), None)
    hisim_runner.apply_thermal_payload(record, {"thermal": None, "thermal_error": "boom"})
    assert record.thermal_status == "failed" and record.thermal_error == "boom"


def test_apply_thermal_payload_noop_when_thermal_not_requested():
    record = hisim_runner._new_record(HisimConfig(), None)
    hisim_runner.apply_thermal_payload(record, {"metrics": {}})
    assert record.thermal_status == "none"


def test_thermal_is_one_knob_next_to_the_auto_mapping_toggles():
    from app.services.knobs import knob_schema

    fields = [f["id"] for g in knob_schema()["groups"] for f in g["fields"]]
    assert fields.count("thermal_sim") == 1
    assert fields.index("thermal_sim") == fields.index("create_default_files") + 1
