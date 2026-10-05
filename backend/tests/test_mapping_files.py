from pathlib import Path
import sys

import pytest

BACKEND_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services import mapping_files
from app.services import hisim_runner
from app.schemas.run import HisimConfig


def test_all_six_input_kinds_use_model_scoped_uploaded_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(mapping_files, "HISIM_DIR", tmp_path / "HISIM-SystolicArray")
    monkeypatch.setattr(mapping_files, "UPLOADED_FILES_DIR", tmp_path / "uploaded_files")

    kinds = {"chip_map", "sys_map", "layer_map", "sa_spec", "mem_spec", "network_spec"}
    for kind in kinds:
        mapping_files.write_mapping_file("gpt2", kind, b"header\nvalue\n")

    status = mapping_files.mapping_files_status("gpt2")
    assert set(status) == kinds
    assert all(item["exists"] and item["source"] == "uploaded" for item in status.values())
    assert all(
        mapping_files.uploaded_file_path("gpt2", kind).parent == tmp_path / "uploaded_files" / "gpt2"
        for kind in kinds
    )


def test_download_prefers_uploaded_file_but_falls_back_to_generated_file(tmp_path, monkeypatch):
    monkeypatch.setattr(mapping_files, "HISIM_DIR", tmp_path / "HISIM-SystolicArray")
    monkeypatch.setattr(mapping_files, "UPLOADED_FILES_DIR", tmp_path / "uploaded_files")
    monkeypatch.setattr(mapping_files, "GENERATED_FILES_DIR", tmp_path / "generated_files")

    generated = mapping_files.mapping_file_path("gpt2", "sa_spec")
    generated.parent.mkdir(parents=True)
    generated.write_text("generated\n", encoding="utf-8")
    assert mapping_files.preferred_mapping_file_path("gpt2", "sa_spec") == generated

    mapping_files.write_mapping_file("gpt2", "sa_spec", b"uploaded\n")
    preferred = mapping_files.preferred_mapping_file_path("gpt2", "sa_spec")
    assert preferred == mapping_files.uploaded_file_path("gpt2", "sa_spec")
    assert preferred.read_text(encoding="utf-8") == "uploaded\n"


def test_generated_files_are_kept_separate_from_user_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(mapping_files, "HISIM_DIR", tmp_path / "HISIM-SystolicArray")
    monkeypatch.setattr(mapping_files, "UPLOADED_FILES_DIR", tmp_path / "uploaded_files")
    monkeypatch.setattr(mapping_files, "GENERATED_FILES_DIR", tmp_path / "generated_files")

    generated = mapping_files.generated_file_path("gpt2", "sa_spec")
    generated.parent.mkdir(parents=True)
    generated.write_text("generated\n", encoding="utf-8")

    assert mapping_files.preferred_mapping_file_path("gpt2", "sa_spec") == generated
    assert mapping_files.mapping_files_status("gpt2")["sa_spec"]["source"] == "generated"

    mapping_files.write_mapping_file("gpt2", "sa_spec", b"uploaded\n")
    assert mapping_files.preferred_mapping_file_path("gpt2", "sa_spec") == mapping_files.uploaded_file_path("gpt2", "sa_spec")


def test_missing_input_file_has_a_valid_csv_template():
    filename, content = mapping_files.input_file_template("gpt2", "sa_spec")

    assert filename == "SA_Spec_gpt2_template.csv"
    assert content.splitlines()[0] == (
        "Chiplet ID,Tile ID,HW Type,SA_size_x,SA_size_y,n_SA,prec,Clock Frequency (Hz)"
    )


def test_each_input_kind_has_a_template_with_the_expected_header():
    expected = {
        "chip_map": "Chiplet ID,Tile ID,HW Type,NoC Position,AI Layer,NodeName",
        "sys_map": "Chiplet ID,Stack ID,Tier ID,NoP Position",
        "layer_map": "Layer ID,Type,Function,in1_dim1,in1_dim2,in1_dim3,in1_dim4,in2_dim1,in2_dim2,in2_dim3,in2_dim4,Parallel,A,B,C",
        "sa_spec": "Chiplet ID,Tile ID,HW Type,SA_size_x,SA_size_y,n_SA,prec,Clock Frequency (Hz)",
        "mem_spec": "Chiplet ID,Tile ID,HW Type,Nbank,NW,NB,CM,Clock Frequency (Hz)",
        "network_spec": "Stack ID,N_2D_Links_per_tile,N_3D_Links_per_tile,N_2.5D_channels_per_chiplet_edge",
    }

    for kind, header in expected.items():
        _, content = mapping_files.input_file_template("gpt2", kind)
        assert content.splitlines()[0] == header


def test_mapping_overview_summarizes_uploaded_csv_contents(tmp_path, monkeypatch):
    monkeypatch.setattr(mapping_files, "HISIM_DIR", tmp_path / "HISIM-SystolicArray")
    monkeypatch.setattr(mapping_files, "UPLOADED_FILES_DIR", tmp_path / "uploaded_files")
    files = {
        "chip_map": "Chiplet ID,Tile ID,HW Type,NoC Position,AI Layer,NodeName\nC1,T1,SA,0,0,L0\nC1,T2,Mem Tile,1,0,L0\n",
        "sys_map": "Chiplet ID,Stack ID,Tier ID,NoP Position\nC1,S1,1,0,0,0\n",
        "layer_map": "Layer ID,Type,Function,in1_dim1,in1_dim2,in1_dim3,in1_dim4,in2_dim1,in2_dim2,in2_dim3,in2_dim4,Parallel,A,B,C\nL0,MatMul,AxB mul BxC,1,1,1,1,1,1,1,1,Auto,A,B,C,\n",
        "sa_spec": "Chiplet ID,Tile ID,HW Type,SA_size_x,SA_size_y,n_SA,prec,Clock Frequency (Hz)\nC1,T1,SA,32,16,2,8,1000000000\n",
        "mem_spec": "Chiplet ID,Tile ID,HW Type,Nbank,NW,NB,CM,Clock Frequency (Hz)\nC1,T2,Mem Tile,4,1024,320,4,1000000000\n",
        "network_spec": "Stack ID,N_2D_Links_per_tile,N_3D_Links_per_tile,N_2.5D_channels_per_chiplet_edge\nS1,80,40,2\n",
    }
    for kind, content in files.items():
        mapping_files.write_mapping_file("gpt2", kind, content.encode())

    overview = mapping_files.mapping_files_overview("gpt2")

    assert overview["model"] == "gpt2"
    assert overview["source"] == "uploaded"
    assert overview["tiles"] == 2
    assert overview["chiplets"] == 1
    assert overview["hardware_types"] == {"SA": 1, "Mem Tile": 1}
    assert overview["sa"]["sizes"] == ["32×16"]
    assert overview["sa"]["precisions"] == [8]
    assert overview["memory"]["total_banks"] == 4
    assert overview["network"]["noc_2d_links"] == [80]


def test_unknown_kind_and_unsafe_model_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(mapping_files, "HISIM_DIR", tmp_path / "HISIM-SystolicArray")
    monkeypatch.setattr(mapping_files, "UPLOADED_FILES_DIR", tmp_path / "uploaded_files")

    with pytest.raises(ValueError, match="Unknown input file kind"):
        mapping_files.uploaded_file_path("gpt2", "not_a_file")
    with pytest.raises(ValueError, match="Invalid model name"):
        mapping_files.uploaded_file_path("../other", "sa_spec")


def test_runner_accepts_a_complete_uploaded_set_when_regeneration_is_disabled(tmp_path, monkeypatch):
    engine = tmp_path / "HISIM-SystolicArray"
    monkeypatch.setattr(mapping_files, "HISIM_DIR", engine)
    monkeypatch.setattr(mapping_files, "UPLOADED_FILES_DIR", tmp_path / "uploaded_files")
    monkeypatch.setattr(mapping_files, "GENERATED_FILES_DIR", tmp_path / "generated_files")
    monkeypatch.setattr(hisim_runner, "HISIM_DIR", engine)

    config = HisimConfig(aimodel="gpt2", create_default_files=False)
    with pytest.raises(ValueError, match="Chip_Map"):
        hisim_runner._validate_mapping_files(config)

    for kind in mapping_files.KIND_PREFIXES:
        mapping_files.write_mapping_file("gpt2", kind, b"header\nvalue\n")

    hisim_runner._validate_mapping_files(config)


def test_result_metadata_records_requested_and_verified_input_source(monkeypatch):
    summary = {"model": "gpt2", "source": "uploaded", "files_available": 6}
    monkeypatch.setattr(hisim_runner, "mapping_files_overview", lambda model: summary)

    payload = hisim_runner._annotate_result_inputs(
        {"status": "completed"}, HisimConfig(aimodel="gpt2", create_default_files=False)
    )

    assert payload["input_mode_requested"] == "uploaded"
    assert payload["input_mode_used"] == "uploaded"
    assert payload["input_file_overview"] == summary


def test_run_workspace_isolated_from_the_engine_and_stages_six_inputs(tmp_path, monkeypatch):
    engine = tmp_path / "HISIM-SystolicArray"
    engine.mkdir()
    source_config = "import os\nmain_dir = os.path.dirname(__file__)\naimodel = 'gpt2'\n"
    (engine / "config.py").write_text(source_config, encoding="utf-8")
    (engine / "gui_run.py").write_text("print('driver')\n", encoding="utf-8")
    source_inputs = tmp_path / "source-inputs"
    source_inputs.mkdir()
    for kind in mapping_files.KIND_PREFIXES:
        path = source_inputs / f"{kind}.csv"
        path.write_text(f"{kind}\n", encoding="utf-8")

    monkeypatch.setattr(hisim_runner, "HISIM_DIR", engine)
    monkeypatch.setattr(
        hisim_runner,
        "preferred_mapping_file_path",
        lambda _model, kind: source_inputs / f"{kind}.csv",
    )

    workspace, input_base = hisim_runner._prepare_run_workspace(
        tmp_path / "run", HisimConfig(aimodel="gpt2", create_default_files=False)
    )

    assert (workspace / "config.py").read_text(encoding="utf-8") != source_config
    assert (engine / "config.py").read_text(encoding="utf-8") == source_config
    assert f"USER_FILES_DIR={str(input_base)!r}" in (workspace / "config.py").read_text(encoding="utf-8")
    assert {path.name for path in (input_base / "gpt2").iterdir()} == {
        f"{prefix}_gpt2.csv" for prefix in mapping_files.KIND_PREFIXES.values()
    }
