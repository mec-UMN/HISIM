# HISIM Workbench

HISIM Workbench is a browser-based interface for **HISIM 2.0**, an analytical simulator for power, performance, area, cost, and interconnect behavior in 2D, 2.5D, 3D, and 3.5D AI accelerator packages.

The application serves the GUI and API from one local process. It supports single simulations, design-space sweeps, run comparison, and validated custom mapping/specification CSVs for heterogeneous chiplet designs.

## Repository layout

```text
HISIM_GUI_Official/
├── HISIM-SystolicArray/  # HISIM analytical engine and reference model inputs
├── backend/              # FastAPI API and simulation orchestration
├── webui/                # HTML, CSS, and JavaScript interface
├── uploaded_files/       # Runtime location for user-provided CSVs (initially empty)
├── generated_files/      # Runtime location for generated CSVs (initially empty)
├── runs/                 # Runtime run history, logs, and artifacts (initially empty)
├── requirements.txt      # Complete runtime dependency set
└── LICENSE
```

## Quick start

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.9 or later is required. `pygraphviz` is optional and is needed only when HISIM debug figure generation is enabled.

### 3. Start the workbench

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/ in a browser.

The simulator writes runtime data into `runs/`, `uploaded_files/`, and `generated_files/`. These directories are intentionally empty in a fresh clone and are excluded from Git.

## Using the GUI

1. Select an AI model and design knobs.
2. Choose **Run simulation** for one configuration, or enable **Design-space sweep** and provide values for one knob.
3. Enable **Compare multiple models** only when a sweep should run each selected model at every value.
4. Use **Runs & Compare** to inspect completed sweep runs, sweep response, trade-offs, and scorecards.

For custom heterogeneous chiplet designs, open **HISIM input files** in the Design Knobs panel. Download the current CSVs or templates, upload replacements, then select **Run uploaded files**. The input validator checks all six files before invoking HISIM:

- Chip Map
- System Map
- Layer Map
- SA Spec
- Memory Spec
- Network Spec

## Command-line HISIM

The analytical engine can also run without the GUI:

```bash
cd HISIM-SystolicArray
python HISIM.py
```

Set `CREATE_DEFAULT_FILES = True` in `HISIM-SystolicArray/config.py` to generate default inputs. Set it to `False` when supplying a complete validated custom file set.

## Notes

- The web interface loads Plotly, Chart.js, D3, and Three.js from public CDNs. Internet access is required the first time a browser loads the application unless these assets are vendored locally.
- Simulation runs are isolated from one another, so one failed sweep point is recorded as failed and does not block the remaining points.
- Debug plots require Graphviz plus `pygraphviz`; this is not required for normal GUI simulations.

## License

This distribution is released under the MIT License. See [LICENSE](LICENSE). Please retain the included HISIM attribution and citation information in [HISIM-SystolicArray/README.md](HISIM-SystolicArray/README.md) when using the analytical engine in research.
