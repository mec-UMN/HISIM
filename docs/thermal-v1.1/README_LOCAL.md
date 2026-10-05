# HISIM (IMC + SA) with thermal module v1.1 — local-run package

This package holds the two models of the original HISIM-main (HEAD bbdb07a):
- `HISIM-IMC/` (HISIM 1.0);
- `HISIM-SystolicArray/` (HISIM 2.0).

Each folder has its own embedded copy of `hisim_thermal/`, the v1.1 thermal module. The two copies are identical.

After you unzip the package and install the dependencies, you can run either model from its folder. You don't need any environment variables or any other folders.

- Version and per-file md5: `BUILD_INFO.txt` and `MD5SUMS`.
- Changes from the original: `CHANGELOG.md`.

## 1. Install (once)

The package was tested with Python 3.12 and the versions pinned in `requirements.txt`.

conda:
```
conda create -n hisim_v11 python=3.12 -y
conda activate hisim_v11
pip install -r requirements.txt
```

venv:
```
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`pyamg` is required. Without it, any grid above 60000 unknowns raises an error; the solver does not fall back to another method. The IMC default mesh (8×8 cells per tile) has about 160 thousand unknowns on the README case.

## 2. Run HISIM-IMC

Run from inside `HISIM-IMC/`, because the upstream Debug / Results paths are relative to the current directory:
```
cd HISIM-IMC
python analy_model.py --chip_architect M3D --placement_method 5 --N_stack 1 --N_tier 3 --N_tile 64 --N_pe 36 --xbar_size 1024 --ai_model vit --thermal
```

- `--thermal` turns on the thermal simulation. All other arguments are the same as in the original HISIM.
- Results go to `HISIM-IMC/Results/result_thermal/hisim_thermal/`.
- For 2.5D with 3 or 4 chiplets, two extra side layouts are also written, `layout_L/` and `layout_ring/`. They do not enter the PPA.
- The PPA temperature column is `peak_temperature (K)`, in kelvin.
- **Self-check:** the command above should print `peaks domain 315.112 K ... tile 315.112 K, router 313.799 K, tsv 314.855 K`. It took about 20–40 s on the test server.

## 3. Run HISIM-SystolicArray

Run from inside `HISIM-SystolicArray/` for the same reason:
```
cd HISIM-SystolicArray
python HISIM.py
```

- The thermal switches are in `config.py`:
  - `THERMAL_SIM = True` runs the thermal module after the network model;
  - `THERMAL_PRESET = 'v1.1'`;
  - `THERMAL_FLOORPLAN = 'rowshelf'` (or `'grid'`).
- Everything else (model, architecture, tile counts, ...) is set in `config.py`, as in the original HISIM 2.0.
- Results go to `HISIM-SystolicArray/Results/thermal/`.
- **Self-check:** the default `config.py` should print `peaks: domain 298.321 K ... router 298.229 K, aib 298.309 K`. It took about 2.5–3 min on the test server. Most of that time is the upstream compute part.

## 4. Presets and parameters

**Preset.** The first of these that is set wins:
1. the function argument `preset=`;
2. the environment variable `HISIM_THERMAL_PRESET`;
3. `THERMAL_PRESET` in the file.

| Preset | Meaning | Applies to |
|---|---|---|
| `v1.1` | Default; the new physical model | IMC, SA |
| `v1compat` | Numerically equivalent to the v1 reference implementation; M2D / M3D only | IMC |
| `legacy` | Original thermal code plus patches L1–L7 | IMC |
| `legacy_exact` | Original thermal code; only torch removed and output directories created | IMC |

SA's original thermal module was an empty stub, so there is no old version to fall back to. SA is validated with `v1.1` only.

**Overrides.** Pass JSON in the environment variable `HISIM_THERMAL_OVERRIDES`.

Linux / macOS:
```
HISIM_THERMAL_OVERRIDES='{"bc": {"h_top": 2e4}, "mesh": {"cells_per_tile": 4}}' python analy_model.py ... --thermal
```

Windows PowerShell:
```
$env:HISIM_THERMAL_OVERRIDES='{"bc": {"h_top": 2e4}}'; python analy_model.py ... --thermal
```

**Default mesh.**
- IMC with v1.1 uses 8×8 cells per tile (`cells_per_tile = 8`). This level is certified by GCI.
- SA defaults to r = 1.

## 5. Output files (one directory per run)

| File | Content |
|---|---|
| `peaks.json` | Peaks by category: domain / device / active / imc / tile / router / tsv / aib / emib |
| `tiles.json` | Mean temperature, max temperature and power of each tile |
| `T_tile.npy`, `tile_labels.npz` | IMC N×N tile label maps (`T_mean`, `T_max`, `P_W`, `mapped`, ...) |
| `T_full.npy` | Full temperature field [k, iy, ix] (K) |
| `P_full.npy`, `K_full.npy`, `N_full.npy`, `regions.json` | Per-cell power, conductivity, region codes |
| `grid.npz`, `layers.json` | Grid coordinates (m), stack |
| `meta.json` | Configuration and provenance, power ledger, energy balance, solver info |
| `input.json` | Frozen neutral input; can be re-solved on its own |
| `figures/`, `viewer.html` | Figures, and an interactive viewer that opens in a browser |

**Use the tile mean temperature for labels** (`T_mean` in `tiles.json` or `tile_labels.npz`). The tile max temperature does not converge under GCI and is not certified.

## 6. Notes and known limitations

- **Do not run two jobs in the same folder at once.** `hisim_model.py` clears `Results/result_thermal/1stacks` when it is imported. To run in parallel, make several copies of the folder.
- **Upstream issues are not corrected** (decision D15); the thermal module takes the upstream values as given:
  - NoC (router / TSV) power is about 103× too small;
  - tile area is overestimated by 1.6–132×.
- **The SA floorplan is an assumption** (rowshelf). The die area comes out 1.1–2.9× the upstream area.
- **SA's default mesh is not certified.** At r = 1 the GCI error is about 9.8%. For more accuracy, use `{"mesh": {"lat_level": 2}}`, which is slower.
- M3_5D is not supported and raises NotImplementedError.
- **Validated only on Linux x86_64.** The run path calls no shell commands and uses no multiprocessing, and pyamg has pip wheels for Windows and macOS. Neither platform has been tested.
- The upstream sweep scripts (`run.py`, `run_HISIM_*dse.py`) call the main program through `os.system("python ...")`. For them, `python` on your PATH must point to the environment with the dependencies installed.

## 7. Validation of this package (2026-10-02)

Setup: a fresh Python 3.12 venv with only the packages in `requirements.txt`, and no environment variables.

| Check | Result |
|---|---|
| IMC README case (the command in section 2) | Finished in 37 s. Peak 315.112 K, the same as the official workspace result (results/final, e2e m3d_readme_r8). Energy balance relative error 4.9e-14 |
| SA with the default `config.py` | Finished in 2 min 49 s. Peak 298.321 K. Energy balance relative error 2.0e-11 |
| Unit tests (the workspace tests/, run against this package's code) | 84 passed; 2 slow tests skipped by design |

These checks were run before the English translation of the documentation. The only code change in the translation is a comment in `HISIM-SystolicArray/config.py`; no code logic changed.
