from pathlib import Path
import sys

import networkx as nx


ENGINE_DIR = Path(__file__).parents[2] / "HISIM-SystolicArray"
sys.path.insert(0, str(ENGINE_DIR))

from Module_0_AI_Map.util_chip.HISIM_2_0_Files import HW_Map
from gui_run import build_topology


def test_generic_3d_template_creates_one_stack_with_multiple_tiers(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node("In")
    for index in range(1, 4):
        graph.add_node(f"L{index}", Type="Relu", prec=8)
    graph.add_edges_from([("In", "L1"), ("L1", "L2"), ("L2", "L3")])

    monkeypatch.setattr(HW_Map, "chip_count", 2)
    monkeypatch.setattr(HW_Map, "tile_count_dict", {"SA": 1, "CPU": 1, "Mem_I": 1, "Mem_W": 1, "Mem_O": 1})
    monkeypatch.setattr(HW_Map, "SET_SUFF_BANKS", False)

    chip_map, mem_spec, sa_spec, layer_spec, sys_map, net_spec = HW_Map.create_default_files(graph, "3D_Mesh")

    assert chip_map
    assert mem_spec
    assert set(entry[1] for entry in sys_map.values()) == {"S1"}
    assert {entry[2] for entry in sys_map.values()} == {"TR0", "TR1"}


def test_gui_topology_omits_empty_padding_tiles_and_their_links():
    chip = nx.Graph()
    chip.add_node(
        "C1_T1",
        **{
            "HW Type": "SA",
            "Chiplet ID": "C1",
            "Tile ID": "T1",
            "Stack ID": "S1",
            "NoC Position": "0,0",
            "NoP Position": "0,0,0",
        },
    )
    chip.add_node(
        "1_x-1_y-1",
        **{
            "HW Type": "Empty",
            "Chiplet ID": "C1_x-1_y-1",
            "Tile ID": "1_x-1_y-1",
            "Stack ID": "S1",
            "NoC Position": "-1,-1",
            "NoP Position": "0,0,0",
        },
    )
    chip.add_edge("C1_T1", "1_x-1_y-1")

    system = nx.Graph()
    system.add_node("C1", **{"Stack ID": "S1", "Tier ID": "TR0", "NoP Position": "0,0,0"})
    system.add_node("C1_x-1_y-1", **{"Stack ID": "S1", "Tier ID": "TR0", "NoP Position": "0,0,0"})
    system.add_edge("C1", "C1_x-1_y-1")
    stack = nx.Graph()
    stack.add_node("S1_TR0", **{"NoP Position": "0,0,0"})

    topology = build_topology(chip, system, stack, {}, {})

    assert [tile["id"] for tile in topology["tiles"]] == ["C1_T1"]
    assert [chiplet["id"] for chiplet in topology["chiplets"]] == ["C1"]
    assert topology["noc_edges"] == []
    assert topology["nop_edges"] == []
