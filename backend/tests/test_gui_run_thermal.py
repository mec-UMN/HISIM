import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).parents[2] / "HISIM-SystolicArray"
sys.path.insert(0, str(ENGINE_DIR))

import gui_run


def test_thermal_metrics_maps_summary_to_metric_keys():
    summary = {"peak_k": 350.0, "min_k": 320.0, "avg_k": 330.0, "dies": []}
    assert gui_run.thermal_metrics(summary) == {
        "peak_temperature_k": 350.0, "min_temperature_k": 320.0, "avg_temperature_k": 330.0}


def test_run_thermal_step_calls_solver_and_exporter(monkeypatch, tmp_path):
    calls = {}

    def fake_analysis(G_chip, G_stack, G_sys=None, mesh_size=None, preset=None, rule=None, out_dir=None,
                      mesh_size_resolution=None):
        calls.update(G_sys=G_sys, mesh_size=mesh_size, rule=rule, out_dir=out_dir, res=mesh_size_resolution)
        return 300.0, 310.0, 320.0

    monkeypatch.setattr(gui_run, "_thermal_analysis", lambda: fake_analysis)
    monkeypatch.setattr(gui_run, "_export_maps",
                        lambda d, a, b, c: {"peak_k": c, "min_k": a, "avg_k": b, "dies": []})
    monkeypatch.setattr(gui_run.config, "THERMAL_FLOORPLAN", "rowshelf", raising=False)
    monkeypatch.setattr(gui_run, "_plan_resolution", lambda *a, **k: {
        "unknowns": 10, "coarsen_factor": 1.0, "resolution_mm": None, "final_unknowns": 10})
    out = gui_run.run_thermal_step("gc", "gs", "gsys", 4, str(tmp_path))
    assert calls == {"G_sys": "gsys", "mesh_size": 4, "rule": "rowshelf", "out_dir": str(tmp_path), "res": None}
    assert out["peak_k"] == 320.0 and out["solve_time_s"] >= 0


def _stubs(monkeypatch, plan):
    calls = {}

    def fake_analysis(*a, **k):
        calls["res"] = k.get("mesh_size_resolution")
        return 300.0, 310.0, 320.0

    monkeypatch.setattr(gui_run, "_thermal_analysis", lambda: fake_analysis)
    monkeypatch.setattr(gui_run, "_export_maps", lambda d, a, b, c: {"peak_k": c, "min_k": a, "avg_k": b, "dies": []})
    monkeypatch.setattr(gui_run, "_plan_resolution", lambda *a, **k: plan)
    return calls


def test_big_design_is_solved_on_a_coarser_grid_and_says_so(monkeypatch, tmp_path):
    calls = _stubs(monkeypatch, {"unknowns": 14_000_000, "coarsen_factor": 5.9, "resolution_mm": 3.7,
                                 "final_unknowns": 390_000})
    out = gui_run.run_thermal_step("gc", "gs", "gsys", 4, str(tmp_path))
    assert calls["res"] == 3.7
    assert out["coarsen_factor"] == 5.9 and out["unknowns"] == 390_000


def test_design_that_cannot_be_made_small_enough_is_skipped_with_a_clear_message(monkeypatch, tmp_path):
    calls = _stubs(monkeypatch, {"unknowns": 9e9, "coarsen_factor": 50.0, "resolution_mm": 30.0,
                                 "final_unknowns": 5_000_000})
    try:
        gui_run.run_thermal_step("gc", "gs", "gsys", 4, str(tmp_path))
    except gui_run.ThermalTooLarge as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("expected ThermalTooLarge")
    assert "res" not in calls


def test_known_thermal_model_limits_are_explained_in_plain_language():
    msg = gui_run.friendly_thermal_error(ValueError("C1_x3_y1: 3D link power on a boundary NoP site"))
    assert "3.5D" in msg and "not supported" in msg
    msg = gui_run.friendly_thermal_error(NotImplementedError("stacks with different tier counts ({'S1': 1, 'S2': 2}) are not supported"))
    assert "same number of tiers" in msg
    assert gui_run.friendly_thermal_error(RuntimeError("boom")) == "RuntimeError: boom"
    too_big = gui_run.ThermalTooLarge("Thermal grid too large (5 cells).")
    assert gui_run.friendly_thermal_error(too_big) == "Thermal grid too large (5 cells)."
