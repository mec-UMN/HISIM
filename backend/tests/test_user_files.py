import sys
from pathlib import Path


ENGINE_DIR = Path(__file__).parents[2] / "HISIM-SystolicArray"
sys.path.insert(0, str(ENGINE_DIR))

from validate_maps import import_user_files  # noqa: E402


def test_import_user_files_stages_all_six_files_for_hisim(tmp_path):
    root = tmp_path / "HISIM-SystolicArray"
    user_base = tmp_path / "uploaded_files"
    user_dir = user_base / "gpt2"
    names = [
        "Chip_Map", "Sys_Map", "Layer_Mapping", "SA_Spec", "Mem_Spec", "Network_Spec"
    ]
    for name in names:
        (user_dir / f"{name}_gpt2.csv").parent.mkdir(parents=True, exist_ok=True)
        (user_dir / f"{name}_gpt2.csv").write_text(f"{name}\n", encoding="utf-8")

    import_user_files("gpt2", str(root), str(user_base), quiet=True)

    destinations = [
        root / "Chip_Map_gpt2.csv",
        root / "Sys_Map_gpt2.csv",
        root / "Layer_Mapping_gpt2.csv",
        root / "Module_1_Compute" / "HISIM_2_0_Files" / "HW_configs" / "SA_Spec_gpt2.csv",
        root / "Module_1_Compute" / "HISIM_2_0_Files" / "HW_configs" / "Mem_Spec_gpt2.csv",
        root / "Module_2_Network" / "HISIM_2_0_Files" / "Network_configs" / "Network_Spec_gpt2.csv",
    ]
    assert [path.read_text(encoding="utf-8") for path in destinations] == [f"{name}\n" for name in names]


def test_gui_driver_stages_and_validates_user_files_before_loading_maps():
    source = (ENGINE_DIR / "gui_run.py").read_text(encoding="utf-8")
    runtime_source = source[source.index("def main") :]
    assert "from validate_maps import export_user_files, import_user_files" in source
    assert runtime_source.index("load_ai_network") < runtime_source.index("import_user_files") < runtime_source.index("load_ai_chip")
