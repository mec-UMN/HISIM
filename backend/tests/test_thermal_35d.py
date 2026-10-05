"""3.5D packages in the thermal model (hisim_thermal v1.1).

The v1.1 core rejects 3.5D on purpose ("M3_5D ... is not implemented / validated in v1.1", neutral_input.py), and the
SA adapter has no placement for the 3D-link power that 3.5D puts on the bottom die's 2.5D interface site, nor for the
upper tiers' boundary NoP routers. The fixture is the real 3.5D demo design (resnet50, 4 stacks x 2 tiers), so this
test turns green on its own once the thermal model supports 3.5D; strict xfail then forces updating it.
"""
import pickle

import pytest
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).parents[2] / "HISIM-SystolicArray"
sys.path.insert(0, str(ENGINE_DIR))

from hisim_thermal import get_config
from hisim_thermal.adapters.sa import from_sa

FIXTURE = Path(__file__).parent / "fixtures" / "sa_35d_resnet50_4x2.pkl"


def _graphs():
    with open(FIXTURE, "rb") as handle:
        return pickle.load(handle)


def _p3(g_chip, inside):
    total = 0.0
    for node, attrs in g_chip.nodes(data=True):
        x, y = (int(float(v)) for v in str(attrs.get("NoC Position", "0,0")).split(","))
        if (0 <= x < 3 and 0 <= y < 3) == inside:
            total += float(attrs.get("3D link Power", 0) or 0)
    return total


@pytest.mark.xfail(strict=True, reason="hisim_thermal v1.1 does not support 3.5D (M3_5D) packages")
def test_35d_package_converts_and_keeps_all_3d_link_power():
    g = _graphs()
    ti = from_sa(g["G_chip"], g["G_stack"], g["G_sys"], get_config("v1.1", {}), mesh_size=g["mesh"], rule="rowshelf")
    boundary = _p3(g["G_chip"], inside=False)
    assert boundary > 0, "fixture should contain boundary 3D-link power"
    tsv_links = [link for link in ti.links if link.kind == "tsv"]
    assert abs(sum(link.p_src_W for link in tsv_links) - (_p3(g["G_chip"], True) + boundary)) < 1e-12
    boundary_links = [link for link in tsv_links if "boundary" in link.id]
    assert len(boundary_links) == 4                                   # one interface site per stack
    assert all(link.region["src_tiles"][0].endswith(tuple(f".phy{e}" for e in "NSEW")) for link in boundary_links)
