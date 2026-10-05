import pytest

from app.schemas.run import HisimConfig, RunCreate, SweepCreate
from app.services import hisim_runner
from app.services.knobs import knob_schema


def test_sa_dimensions_and_array_count_are_power_of_two_knobs():
    fields = {f["id"]: f for g in knob_schema()["groups"] for f in g["fields"]}
    for knob in ("sa_size_x", "sa_size_y", "n_sa"):
        assert fields[knob]["scale"] == "pow2"


def test_a_new_run_with_a_non_power_of_two_size_is_rejected(monkeypatch):
    monkeypatch.setattr(hisim_runner, "_validate_mapping_files", lambda c: None)
    with pytest.raises(ValueError, match="power of two"):
        hisim_runner.create_run(RunCreate(config=HisimConfig(sa_size_x=470)))


def test_a_sweep_over_non_power_of_two_values_is_rejected(monkeypatch):
    monkeypatch.setattr(hisim_runner, "_validate_mapping_files", lambda c: None)
    monkeypatch.setattr(hisim_runner, "available_models", lambda: [])
    with pytest.raises(ValueError, match="power of two"):
        hisim_runner.create_sweep(SweepCreate(config=HisimConfig(), knob="n_sa", values=[1, 3]))


def test_old_records_with_other_sizes_still_load():
    assert HisimConfig(sa_size_x=470).sa_size_x == 470
