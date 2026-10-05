# CHANGELOG — hisim_thermal v1.1

This is an English translation of the development workspace's CHANGELOG as of 2026-10-02 16:26. Some references point to the development workspace and are not included in this package:
- "plan §x" means the v1.1 implementation plan;
- "README §x" means the workspace README;
- `results/`, `diff/`, `tests/` and `tools/` are workspace folders.

## Changes from the original HISIM (HEAD bbdb07a)

### HISIM-IMC
- `Module_Thermal/thermal_model.py` is now a thin wrapper. The signature and return value are unchanged. It dispatches by preset:
  - `v1.1` / `v1compat` → the new package;
  - `legacy` / `legacy_exact` → the frozen original code.
- `Module_Thermal/legacy/`: the three original thermal files plus patches L1–L7 (diff in the workspace's `diff/legacy_vs_head.diff`):
  - L1: remove torch;
  - L2: create the output directory before saving figures;
  - L3: use slicing for placement 3/4;
  - L4: group `power_tile_reorg` by column 15 (M17);
  - L5: with 4 chiplets, set the AIB/EMIB counts to 8/4;
  - L6: raise ValueError for 5 or more chiplets;
  - L7: assert the AIB row count.
- `Module_Thermal/legacy_exact/`: only L1 and L2.
- `hisim_model.py:170, :469`: the column name `peak_temperature (C)` becomes `peak_temperature (K)` (D8). The values are unchanged.

### HISIM-SystolicArray
- `Module_4_Thermal/HISIM_2_0_Files/thermal.py`: the original file held only interface comments; importing it also read a pkl and dropped into pdb. `thermal_analysis()` is now implemented with the original interface.
- `HISIM.py`: calls the thermal module after the network model (`config.THERMAL_SIM`).
- `config.py`: new settings `THERMAL_SIM`, `THERMAL_PRESET`, `THERMAL_FLOORPLAN`.

### Physics and numerics (v1.1 preset, with the issue IDs of the original code)

| ID | Original | v1.1 |
|---|---|---|
| M05 | Router / TSV power written into every cell | Each tier's total is spread over its cells; conservation is checked |
| — | Ghost nodes pinned at 298 K | Robin boundaries on top and bottom; no ghost nodes |
| — | Cu k = 15 / 20, α = 3.5 | Cu 400, α = 1 |
| — | Package layers fixed at 5 × 0.5 mm sublayers; dict_z had no effect | Stack per plan §4.1 / §4.2; graded sublayers; thickness assertions |
| — | Only the top die has BEOL; air layers | Every die is [BEOL, active, Si] with a bond layer between dies; no air layers |
| B3 | H2_5D imc_size = (√(A+1)−1) | √A |
| A2 / A3 | Crash with 4 chiplets; UnboundLocalError with ≥5 | Chain layout, any N |
| M17 | Power shifted forward with ≥3 chiplets | Grouped by column 15 (stack) |
| A4 | Placement 3/4 dropped the last tile when the slot count is odd | Slicing |
| — | 2.5D strip width rounded to a multiple of b (area inflated) | Width set from each row's physical area |
| — | 2.5D gap columns filled with air | Underfill in the C4 layer, mold compound above |
| — | 2.5D tile power split over the imc cell and 2 tsv cells | Injected into the imc region only (intentional change) |
| — | Depends on torch | Depends only on numpy / scipy / matplotlib; pyamg optional, but grids above 6e4 unknowns need it |

### Added
- Neutral input (`hisim_thermal.ThermalInput`), with one adapter each for IMC and SA.
- Outputs: peak categories, tile labels, a trace table, and full metadata.
- Validation tools: v1 alignment, same-grid comparison with 3D-ICE, GCI certification, comparison with MFIT.

## Changes from the v1 reference implementation

| Item | v1 | v1.1 |
|---|---|---|
| Stack | Heat sink 5 mm, TIM2 50 µm, spreader 2 mm, TIM1 30 µm; h_top on the heat-sink top; only the top die has BEOL; an 11 µm bond band under every tier, including the bottom one | Cu lid 2 mm, TIM1 20 µm; h_top on the lid top; every die has BEOL; bond layers only between dies |
| Bottom | h_bottom = 0 (adiabatic) | 10 |
| Leakage | On by default, selected by P > 0 | Off by default; when on, uses the real "mapped" flag |
| Lateral mesh | 2×2 per tile | Choosable r (GCI certification sets the production level) |
| 2.5D, SA | Not covered | Both covered |
| v1compat preset | — | Numerically equivalent to v1 (I2, 13 cases). Leakage uses `leakage_slots = p_positive`, because v1's "mapped" in fact means P > 0. v1's router_fix=False switch is not kept; use legacy to reproduce the original behavior |

## Additions after the final review (v1.0)
- GCI:
  - the observed order is capped at 2 from above, and values below 0.5 are used as they are (both more conservative);
  - each level gets a bound on the solve error (G⁻¹|r|, residual in extended precision);
  - a strict verdict per the original wording of plan D11, with the relaxed verdicts still reported;
  - certificates record each level's grid ID, the scope of validity, and the family role (joint = acceptance family, lateral / vertical = diagnostic families).
- 3D-ICE comparison: with h_bottom = 0, an h → 0 sequence is used (1e-2, 1e-4, 1e-6 W/m²K).
- AIB source ledger is split by row into Tx / Rx / P_wire and checked row by row; the bridge metadata records the strip width and where it came from.
- The total stack thickness is checked independently against the formulas of plan §4.1 / §4.2.
- SA: the Empty outer ring is recognised correctly when mesh_size is not passed.
- v1 alignment tool: also runs inside a snapshot; new options --h_bottom / --include_leakage / --leakage_slots, plus placement 3 / 4 cases (including 9×9 with an odd slot count).
- The metadata records the upstream tsvPitch (recorded only, not used).
- Invalid input now always raises instead of being accepted silently (final audit):
  - overrides that change the preset;
  - duplicate entries in tsv_via_path;
  - the same tile appearing twice in a link region;
  - duplicate injection regions.
- Figures and viewer (final audit):
  - the sensitivity tornado chart draws low and high values side by side, so they no longer hide each other;
  - in the viewer's log-scale power map, zero-power cells are grey, and the colour bar notes "= 0".
- New tests: U18 (via segment shares), P1 (trace and P_full agree both ways), U6 per-layer router totals, SA mesh inference, GCI details, invalid inputs; 81 tests in total at the time of this entry.

## Default mesh
- With v1.1, the HISIM-IMC wrapper defaults to `cells_per_tile = 8`, as certified by GCI (README §4). The core configuration default is still 1. With r = 1 the grid matches the original 2×2 grid cell for cell.

## Review rounds
- **v0.1**: 5 reviewer agents, 35 findings, all addressed (workspace `results/reviews/v0.1/` and the work log).
- **v0.2**: 4 reviewer agents, 31 findings, all addressed. numerics_gci failed partway; v1.0's sa_numerics covered it.
- **v1.0 (final review)**: one agent per hisim/*.md design document, checking it item by item, plus sa_numerics (10 findings, addressed). Results are in the workspace's `results/reviews/v1.0/`.
