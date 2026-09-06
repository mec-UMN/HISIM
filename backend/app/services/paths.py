from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
HISIM_DIR = ROOT_DIR / "HISIM-SystolicArray"
RUNS_DIR = ROOT_DIR / "runs"
POSTPROCESSING_DIR = HISIM_DIR / "Postprocessing_Files"
WEBUI_DIR = ROOT_DIR / "webui"
UPLOADED_FILES_DIR = ROOT_DIR / "uploaded_files"
GENERATED_FILES_DIR = ROOT_DIR / "generated_files"
