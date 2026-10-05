import pandas as pd
import pytest
from pydantic import ValidationError

from pathlib import Path
import sys


ENGINE_DIR = Path(__file__).parents[2] / "HISIM-SystolicArray"
BACKEND_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ENGINE_DIR))

from Module_1_Compute.HISIM_2_0_Files.Mem import determine_buffer_energy
from app.schemas.run import HisimConfig
from app.schemas.run import SweepCreate
from app.services.hisim_runner import create_sweep
from app.services import hisim_runner


def test_tbc_lut_values_are_calibrated_without_mutating_string_series():
    read = pd.Series(["TBC"], dtype="string")
    write = pd.Series(["TBC"], dtype="string")

    energy = determine_buffer_energy(
        pd.Series([1.0]),
        pd.Series([1.0]),
        1_000_000_000,
        read,
        write,
        1.0,
        1.0,
        1.0,
        1e-9,
        1e-9,
    )

    assert energy > 0
    assert read.iloc[0] == "TBC"
    assert write.iloc[0] == "TBC"


def test_config_rejects_invalid_values_when_a_sweep_assigns_them():
    config = HisimConfig()

    with pytest.raises(ValidationError):
        config.n_bank = "not-a-number"

    with pytest.raises(ValidationError):
        config.n_2d_links_per_tile = 0
    with pytest.raises(ValidationError):
        HisimConfig(type_default_files="not-a-topology")
    with pytest.raises(ValidationError):
        HisimConfig(col_mux=1)


def test_sweep_rejects_an_uncalibrated_memory_setting_before_launch():
    payload = SweepCreate(config=HisimConfig(), knob="col_mux", values=[1])
    with pytest.raises(ValueError, match="Invalid value"):
        create_sweep(payload)


def test_sweep_uses_the_design_model_even_if_a_client_sends_other_models(monkeypatch):
    saved = []

    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(hisim_runner, "save_record", saved.append)
    monkeypatch.setattr(hisim_runner.threading, "Thread", NoopThread)

    payload = SweepCreate(
        config=HisimConfig(aimodel="gpt2"),
        knob="n_bank",
        values=[1, 2],
        models=["llama", "vitbase"],
    )
    created = create_sweep(payload)

    assert len(created) == 2
    assert {record.config.aimodel for record in created} == {"gpt2"}
    assert [record.sweep_value for record in created] == ["1", "2"]
    assert len(saved) == 2


def test_explicit_multi_model_sweep_runs_each_selected_model_for_each_value(monkeypatch):
    """Catches accidentally ignoring an explicitly requested model comparison."""
    saved = []

    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(hisim_runner, "save_record", saved.append)
    monkeypatch.setattr(hisim_runner.threading, "Thread", NoopThread)

    created = create_sweep(
        SweepCreate(
            config=HisimConfig(aimodel="gpt2"),
            knob="n_bank",
            values=[1, 2],
            compare_models=True,
            models=["llama", "vitbase"],
        )
    )

    assert len(created) == 6
    assert [record.config.aimodel for record in created] == [
        "gpt2", "gpt2", "llama", "llama", "vitbase", "vitbase",
    ]
    assert [record.sweep_value for record in created] == ["1", "2", "1", "2", "1", "2"]
    assert len(saved) == 6


def test_sweep_names_use_x10_not_e_notation():
    assert hisim_runner._display_sweep_value(1e-7) == "1.00 × 10^-7"
    assert hisim_runner._display_sweep_value(1e8) == "1.00 × 10^8"


def test_failed_sweep_child_is_terminal_and_does_not_block_the_next(monkeypatch):
    saved = []
    records = [
        hisim_runner._new_record(HisimConfig(), "first"),
        hisim_runner._new_record(HisimConfig(), "second"),
    ]
    calls = []

    def fake_execute(record):
        calls.append(record.name)
        if record.name == "first":
            raise RuntimeError("simulator failed")
        record.status = "completed"

    monkeypatch.setattr(hisim_runner, "_execute_run", fake_execute)
    monkeypatch.setattr(hisim_runner, "save_record", saved.append)

    hisim_runner._execute_sweep(records)

    assert calls == ["first", "second"]
    assert records[0].status == "failed"
    assert records[0].error == "simulator failed"
    assert records[1].status == "completed"


def test_friendly_error_prefers_validation_failure_over_stderr_noise():
    message = hisim_runner._friendly_error(
        "Matplotlib is building the font cache; this may take a moment.\n",
        "VALIDATION FAILED -- HISIM will not run.\n[K003] Clock frequency\n",
    )

    assert "VALIDATION FAILED" in message
    assert "Clock frequency" in message
