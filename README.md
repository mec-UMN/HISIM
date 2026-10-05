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


## Thermal analysis

Turn on **Thermal analysis** in the Integration panel (next to the mapping-file switches) to also solve the
steady 3D temperature field of the package with `hisim_thermal` v1.1 (`HISIM-SystolicArray/hisim_thermal`). It is
off by default and does not change any other result.

- **Dashboard → Thermal:** peak, average and minimum temperature, the rise above ambient per die, and a
  **Run thermal** button that adds thermal results to an already finished run using its saved configuration.
- **Architecture 3D → Thermal:** the temperature map of every die on one colour scale, with PNG and CSV
  downloads covering all dies.
- Large designs are sized before solving: the mesh is coarsened when that helps (noted in the UI), and designs
  that are still too large are skipped with an explanation; the regular results are kept.
- 2D, 3D and 2.5D packages are supported. **3.5D is not supported by the v1.1 thermal model yet**
  (`backend/tests/test_thermal_35d.py` documents the gap and starts passing once the model supports it).
- Sweeps never run thermal analysis. Raw thermal output of a run is written to `runs/<id>/thermal/`.
- The delivered thermal package, its changelog and checksums are kept in `docs/thermal-v1.1/`. Parameter
  values are still being calibrated upstream; the GUI integration does not hard-code them.

## Demo and run history

- **Demo** (top bar) loads a pre-computed bundle from `demo/` instantly, without running the simulator: a 3.5D
  `resnet50` package (4 stacks × 2 tiers), the same workload on 2.5D with thermal maps, and three sweeps
  (systolic arrays per tile, stack count, 2.5D channels per chiplet edge). Rebuild it with
  `python backend/scripts/build_demo.py` while the server is running.
- `runs/` ships with previous simulation results, so the dashboard, Architecture 3D and Runs & Compare are
  populated on first start. Only what the GUI reads is kept per run (record, results, logs, the config used and
  thermal maps); per-run engine copies and scratch files are ignored by git.

## Notes

- The web interface loads Plotly, Chart.js, D3, and Three.js from public CDNs. Internet access is required the first time a browser loads the application unless these assets are vendored locally.
- Simulation runs are isolated from one another, so one failed sweep point is recorded as failed and does not block the remaining points.
- Debug plots require Graphviz plus `pygraphviz`; this is not required for normal GUI simulations.

## License
