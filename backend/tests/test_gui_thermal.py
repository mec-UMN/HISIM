import json
import sys
from pathlib import Path

import numpy as np

ENGINE_DIR = Path(__file__).parents[2] / "HISIM-SystolicArray"
sys.path.insert(0, str(ENGINE_DIR))

from gui_thermal import export_maps


def _fake_result(tmp_path):
    T = np.zeros((4, 2, 3))
    T[0] = 300.0
    T[1] = 310.0
    T[2] = 320.0
    T[3] = 400.0  # non-active silicon of die 1 must be ignored
    np.save(tmp_path / "T_full.npy", T)
    np.savez(tmp_path / "grid.npz", x_edges=np.array([0, 1e-3, 2e-3, 3e-3]), y_edges=np.array([0, 1e-3, 2e-3]),
             z_edges=np.arange(5) * 1e-6)
    subs = [
        {"k": 0, "role": "active", "die": 0},
        {"k": 1, "role": "active", "die": 0},
        {"k": 2, "role": "active", "die": 1},
        {"k": 3, "role": "silicon", "die": 1},
    ]
    (tmp_path / "layers.json").write_text(json.dumps({"sublayers": subs}))
    return tmp_path


def test_export_maps_averages_active_sublayers_per_die(tmp_path):
    summary = export_maps(str(_fake_result(tmp_path)))
    assert [d["die"] for d in summary["dies"]] == [0, 1]
    assert summary["dies"][0]["peak_k"] == 305.0
    assert summary["dies"][1]["peak_k"] == 320.0
    assert summary["peak_k"] == 320.0
    assert summary["min_k"] == 305.0
    maps = json.loads((tmp_path / "maps.json").read_text())
    assert maps["x_edges_mm"] == [0.0, 1.0, 2.0, 3.0]
    assert maps["dies"]["0"]["grid"] == [[305.0] * 3] * 2


def test_export_maps_prefers_solver_totals(tmp_path):
    summary = export_maps(str(_fake_result(tmp_path)), t_min=1.0, t_avg=2.0, t_max=3.0)
    assert (summary["min_k"], summary["avg_k"], summary["peak_k"]) == (1.0, 2.0, 3.0)


def test_export_maps_single_die(tmp_path):
    _fake_result(tmp_path)
    (tmp_path / "layers.json").write_text(json.dumps({"sublayers": [{"k": 0, "role": "active", "die": 0}]}))
    assert len(export_maps(str(tmp_path))["dies"]) == 1


def test_export_maps_without_active_layers_raises(tmp_path):
    _fake_result(tmp_path)
    (tmp_path / "layers.json").write_text(json.dumps({"sublayers": [{"k": 0, "role": "silicon", "die": 0}]}))
    try:
        export_maps(str(tmp_path))
    except ValueError as exc:
        assert "active" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_coarsen_factor_is_one_below_the_limit_and_grows_above_it():
    from gui_thermal import coarsen_factor

    assert coarsen_factor(100_000, limit=400_000) == 1.0
    assert coarsen_factor(400_000, limit=400_000) == 1.0
    assert abs(coarsen_factor(1_600_000, limit=400_000) - 2.0) < 1e-9


def test_fit_resolution_coarsens_until_it_fits():
    from gui_thermal import fit_resolution

    count = lambda edge: int(10_000_000 / (edge / 0.5) ** 2)   # unknowns fall with the square of the edge
    plan = fit_resolution(count, base_edge=0.5, limit=400_000)
    assert plan["unknowns"] == 10_000_000
    assert plan["final_unknowns"] <= 400_000
    assert 4.0 < plan["coarsen_factor"] < 6.0
    assert plan["resolution_mm"] == 0.5 * plan["coarsen_factor"]


def test_fit_resolution_gives_up_when_coarsening_does_not_help():
    from gui_thermal import fit_resolution

    count = lambda edge: 13_000_000 - int(min(edge, 10) * 1000)   # grid is pinned by tile boundaries
    plan = fit_resolution(count, base_edge=0.5, limit=400_000)
    assert plan["coarsen_factor"] == 1.0 and plan["resolution_mm"] is None
    assert plan["final_unknowns"] == plan["unknowns"] == count(0.5)


def test_export_maps_reports_the_ambient_temperature_when_known(tmp_path):
    _fake_result(tmp_path)
    assert export_maps(str(tmp_path))["ambient_k"] is None
    (tmp_path / "meta.json").write_text(json.dumps({"config": {"bc": {"T_amb": 298.0}}}))
    assert export_maps(str(tmp_path))["ambient_k"] == 298.0
