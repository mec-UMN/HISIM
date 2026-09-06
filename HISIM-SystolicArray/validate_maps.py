#!/usr/bin/env python3
"""
validate_maps.py -- pre-flight validator for HISIM's input CSV files.

WHAT THIS IS FOR
----------------
HISIM has no self-consistency checking. It trusts the CSVs completely and
computes whatever they describe. Some inconsistencies raise a clean exception;
others SILENTLY produce plausible-looking wrong numbers -- for example:

  * two tiles on one NoC position     -> one tile drops out of the network model
  * a layer on two compute tiles      -> tile_map is last-write-wins, one is lost
  * an SA-eligible layer with no
    weight memory tile                -> weight traffic silently vanishes
  * an SA tile with no layers, or a
    stale SA_Spec row                 -> num_adders_tile stays NaN -> NaN TOTAL area
  * a per-tile clock != f_noc         -> compute is retimed, the network is not

This script validates all six input files as ONE self-consistent description of
a machine plus a mapping, so those problems surface before the run instead of
inside the results.

    Chip_Map_<model>.csv       tile -> chiplet, NoC position, HW type, layers
    Sys_Map_<model>.csv        chiplet -> stack, tier, NoP position
    Layer_Mapping_<model>.csv  layer -> intra-tile dataflow
    SA_Spec_<model>.csv        per-tile systolic array geometry
    Mem_Spec_<model>.csv       per-tile memory geometry
    Network_Spec_<model>.csv   per-stack link widths

WHAT YOU ARE FREE TO CHANGE
---------------------------
Everything the constraints allow. This validator does not impose a policy of
its own -- it only enforces what HISIM's own code requires. In particular you
may freely build:

  * personalised mappings   -- any layer on any compatible tile
  * heterogeneous chiplets  -- different tile COUNTS per chiplet
  * heterogeneous tiles     -- different SA geometry / memory size per tile
  * any placement           -- any NoC / NoP coordinates

subject to the constraints listed under "CONSTRAINTS" below, all of which come
from HISIM's consumer code rather than from this script.

CONSTRAINTS (each enforced by a check below)
--------------------------------------------
  1. Chiplets sharing a Stack ID must have the SAME mesh size. Mesh size is
     inferred as max(x, y) + 1, so a sparser chiplet must place a tile at the
     far edge to pad its bounding box. Tile COUNTS may still differ.
  2. Chiplet IDs must be contiguous C1..CN (mesh_increment looks up C<n-1>).
  3. Each stack needs exactly one TR0 chiplet; its members share one (x,y) NoP
     position and occupy distinct tiers; Tier ID must equal the NoP z.
  4. NoC positions unique within a chiplet; NoP positions unique across the
     system.
  5. One SA_Spec row per SA tile and one Mem_Spec row per memory tile, in BOTH
     directions. One Network_Spec row per Stack ID. CPU tiles take no spec row.
  6. An SA tile may not be idle -- an SA tile with no layers, or an SA_Spec row
     with no tile, yields NaN area.
  7. Every layer needs exactly one compute tile and exactly one output memory
     tile; exactly one DDR tile exists.
  8. Layer_Mapping is SYMBOLIC. Only {n_SA, SA_size_x, SA_size_y, Auto,
     Temporal, NA} are legal; literal sizes such as 16 are not supported. To
     vary the array size, edit SA_Spec per tile.
  9. The whole system runs at ONE clock. Network.py costs all traffic at the
     global f_noc / f_nop in Network.json and never reads the per-tile clock,
     and CPU tiles are hardcoded to 1e9, so a differing per-tile clock is
     silently mis-modelled.
 10. All geometry values must be positive.

USAGE
-----
    python validate_maps.py                 # uses config.aimodel
    python validate_maps.py --model llama
    python validate_maps.py -v              # also list what passed
    python validate_maps.py --strict        # treat warnings as failures

Exit status: 0 clean, 1 findings, 2 usage/IO problem.

HISIM.py calls validate_or_exit() automatically before the model runs.

SEVERITY
--------
    ERROR  HISIM will crash, or will silently produce wrong numbers.
    WARN   Legal and runnable, but worth knowing -- usually a cost side effect.

Every finding cites the consumer code that imposes the rule.
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

# --------------------------------------------------------------------------
# Constants mirrored from HISIM's consumer code
# --------------------------------------------------------------------------

# Compute.py seeds dict_wrap with exactly these keys; an unknown token in
# Parallel/A/B/C raises on dict_wrap["wrap_"+loc].
WRAP_TOKENS = {"Auto", "n_SA", "SA_size_y", "SA_size_x", "Temporal"}

# Compute.py does sa_df.loc[row_sa, loc] for every NON-LAST token of a
# '*'-chain, so a non-last token must also be a column of SA_Spec.
SA_SPEC_COLUMN_TOKENS = {"n_SA", "SA_size_y", "SA_size_x"}

HW_TYPES = {"SA", "CPU", "Mem Tile"}

# Roles come from literal_eval(NodeName)[0][0] -- the FIRST CHARACTER of the
# FIRST element (Compute.mem_requirement).
MEM_ROLES = {"I": "Input", "W": "Weight", "O": "Output", "D": "DDR"}

CLOCK = "Clock Frequency (Hz)"

CHIP_MAP_COLS = ["Chiplet ID", "Tile ID", "HW Type", "NoC Position",
                 "AI Layer", "NodeName"]
SYS_MAP_COLS = ["Chiplet ID", "Stack ID", "Tier ID", "NoP Position"]
LAYER_MAP_COLS = (["Layer ID", "Type", "Function"]
                  + [f"in1_dim{i}" for i in range(1, 5)]
                  + [f"in2_dim{i}" for i in range(1, 5)]
                  + ["Parallel", "A", "B", "C"])

LOOP_COLS = ["Parallel", "A", "B", "C"]
IN1_COLS = [f"in1_dim{i}" for i in range(1, 5)]
IN2_COLS = [f"in2_dim{i}" for i in range(1, 5)]


def is_sa_eligible(layer_type):
    """Layer types HISIM models on a systolic array.

    Mirrors the predicate repeated in HW_Map.create_default_files and
    Compute.mem_requirement.
    """
    t = str(layer_type)
    return (t.startswith("MatMul") or t == "Gemm" or t == "Conv"
            or t.endswith("Attention"))


def _layer_sort_key(layer):
    m = re.fullmatch(r"L(\d+)", str(layer))
    return (0, int(m.group(1))) if m else (1, str(layer))


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


class Report:
    def __init__(self):
        self.findings = []          # (severity, code, where, message, hint)
        self.passed = []

    def error(self, code, where, message, hint=""):
        self.findings.append(("ERROR", code, where, message, hint))

    def warn(self, code, where, message, hint=""):
        self.findings.append(("WARN", code, where, message, hint))

    def ok(self, label):
        self.passed.append(label)

    @property
    def n_errors(self):
        return sum(1 for f in self.findings if f[0] == "ERROR")

    @property
    def n_warnings(self):
        return sum(1 for f in self.findings if f[0] == "WARN")

    def summarise(self, severity):
        """'C013 x1, M001 x232' -- one compact line per severity."""
        by_code = defaultdict(int)
        for sev, code, _, _, _ in self.findings:
            if sev == severity:
                by_code[code] += 1
        return ", ".join(f"{c} x{n}" for c, n in sorted(by_code.items()))

    def render(self, verbose=False, max_per_code=8, severities=("ERROR", "WARN")):
        out = []
        if verbose and self.passed:
            out.append("Checks passed:")
            out += [f"  ok    {p}" for p in self.passed]
            out.append("")

        if not self.findings:
            out.append("No problems found.")
            return "\n".join(out)

        for want in severities:
            group = [f for f in self.findings if f[0] == want]
            if not group:
                continue
            out.append(f"{want}S ({len(group)}):")
            shown = defaultdict(int)
            hidden = defaultdict(int)
            for sev, code, where, message, hint in group:
                shown[code] += 1
                if not verbose and shown[code] > max_per_code:
                    hidden[code] += 1
                    continue
                out.append(f"  [{code}] {where}")
                out.append(f"        {message}")
                if hint:
                    out.append(f"        -> {hint}")
            for code, n in sorted(hidden.items()):
                out.append(f"  [{code}] ... and {n} more of the same "
                           f"(use -v to list all)")
            out.append("")
        return "\n".join(out)


# --------------------------------------------------------------------------
# Cell parsing helpers
# --------------------------------------------------------------------------


def parse_list_cell(value):
    """literal_eval a cell the way Compute.py does -> (list, error_str)."""
    if pd.isna(value):
        return None, "cell is empty"
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError) as exc:
        return None, f"is not a Python literal ({exc})"
    if not isinstance(parsed, list):
        return None, f"is a {type(parsed).__name__}, expected a list"
    return parsed, None


def parse_coord(value, n_expected):
    """Parse an 'x,y' / 'x,y,z' cell -> (tuple, error_str)."""
    if pd.isna(value):
        return None, "cell is empty"
    parts = str(value).split(",")
    if len(parts) != n_expected:
        return None, f"has {len(parts)} component(s), expected {n_expected}"
    try:
        return tuple(int(p) for p in parts), None
    except ValueError:
        return None, f"'{value}' is not all integers"


# --------------------------------------------------------------------------
# AI model graph
#
# Read Network.csv / Edge.csv directly. Importing HW_Map would REGENERATE the
# very files we are validating whenever config.CREATE_DEFAULT_FILES is True.
# --------------------------------------------------------------------------


def load_ai_model(ai_dir, rep):
    net_csv = os.path.join(ai_dir, "Network.csv")
    edge_csv = os.path.join(ai_dir, "Edge.csv")
    for path in (net_csv, edge_csv):
        if not os.path.isfile(path):
            rep.error("X001", os.path.basename(path), f"missing: {path}",
                      "layer coverage cannot be checked without the AI model")
            return None, None
    try:
        net = pd.read_csv(net_csv)
        edges_df = pd.read_csv(edge_csv)
    except Exception as exc:
        rep.error("X001", "AI model", f"could not read: {exc}")
        return None, None

    layers = {}
    for _, row in net.iterrows():
        attrs = row.to_dict()
        t = attrs["Type"]
        # HW_Map.load_ai_network relabels a Mul with no in2_dim2 as ScalarMul.
        d2 = attrs.get("in2_dim2")
        if t == "Mul" and (d2 is None or pd.isna(d2)):
            t = "ScalarMul"
        attrs["Type"] = t
        layers[str(attrs["Layer_ID"])] = attrs

    edges = [(str(r.iloc[0]), str(r.iloc[1])) for _, r in edges_df.iterrows()]
    return layers, edges


# --------------------------------------------------------------------------
# Sys_Map
# --------------------------------------------------------------------------


def check_sys_map(sys_df, net_spec_df, rep):
    """-> {chiplet_id: {stack, tier, tier_z, nop, line}}"""
    where = "Sys_Map"
    missing = [c for c in SYS_MAP_COLS if c not in sys_df.columns]
    if missing:
        rep.error("S001", where,
                  f"missing required column(s): {', '.join(missing)}",
                  f"expected header: {','.join(SYS_MAP_COLS)}")
        return {}

    chiplets, seen_nop, stacks = {}, {}, defaultdict(list)

    for idx, row in sys_df.iterrows():
        line = idx + 2
        cid = str(row["Chiplet ID"]).strip()
        loc = f"{where} line {line} ({cid})"

        if cid in chiplets:
            rep.error("S002", loc, f"duplicate Chiplet ID '{cid}'",
                      "load_ai_chip builds G_sys keyed by Chiplet ID; the later "
                      "row silently wins")
            continue
        if not re.fullmatch(r"C\d+", cid):
            rep.error("S003", loc, f"Chiplet ID '{cid}' is not of the form C<int>",
                      "HW_Map.natural_key assumes one letter then an integer")
            continue

        sid = str(row["Stack ID"]).strip()
        if not re.fullmatch(r"S\d+", sid):
            rep.error("S004", loc, f"Stack ID '{sid}' is not of the form S<int>")

        tier = str(row["Tier ID"]).strip()
        tier_z = None
        if not re.fullmatch(r"TR\d+", tier):
            rep.error("S005", loc, f"Tier ID '{tier}' is not of the form TR<int>",
                      "load_ai_chip does int(row[2][2:]) on this cell")
        else:
            tier_z = int(tier[2:])

        nop, err = parse_coord(row["NoP Position"], 3)
        if err:
            rep.error("S006", loc, f"NoP Position {err}", 'expected "x,y,z"')
            continue
        if any(v < 0 for v in nop):
            rep.error("S006", loc, f"NoP Position {nop} has a negative component")

        if tier_z is not None and tier_z != nop[2]:
            rep.error("S007", loc,
                      f"Tier ID '{tier}' disagrees with the z of NoP Position {nop}",
                      "Network.calc_edge_charc routes on the z of NoP Position "
                      "while other code keys off Tier ID; they must agree")

        if nop in seen_nop:
            rep.error("S010", loc,
                      f"NoP Position {nop} already used by {seen_nop[nop]}",
                      "load_ai_chip builds nop_chip_dict keyed by this tuple; "
                      "duplicates silently overwrite")
        else:
            seen_nop[nop] = cid

        chiplets[cid] = {"stack": sid, "tier": tier, "tier_z": tier_z,
                         "nop": nop, "line": line}
        stacks[sid].append(cid)

    if not chiplets:
        rep.error("S001", where, "no valid chiplet rows")
        return {}

    nums = sorted(int(c[1:]) for c in chiplets)
    if nums != list(range(1, len(nums) + 1)):
        rep.error("S008", where,
                  f"Chiplet IDs are not contiguous C1..C{len(nums)}: found "
                  + ", ".join("C" + str(n) for n in nums),
                  "load_ai_chip computes mesh_increment for C<n> from C<n-1> and "
                  "then indexes C<len(tile_ids)>; a gap raises KeyError")

    for sid, members in sorted(stacks.items()):
        tr0 = [c for c in members if chiplets[c]["tier_z"] == 0]
        if not tr0:
            rep.error("S009", f"{where} stack {sid}",
                      f"stack {sid} has no TR0 chiplet",
                      "load_ai_chip does [c for c in chip_ids if "
                      "Tier ID=='TR0'][0] -> IndexError")
        elif len(tr0) > 1:
            rep.error("S009", f"{where} stack {sid}",
                      f"stack {sid} has {len(tr0)} TR0 chiplets "
                      f"({', '.join(tr0)})",
                      "load_ai_chip takes [0] arbitrarily, so the stack's "
                      "geometry depends on row order")

        xy = {chiplets[c]["nop"][:2] for c in members}
        if len(xy) > 1:
            rep.error("S011", f"{where} stack {sid}",
                      f"chiplets in stack {sid} disagree on (x,y) NoP position: "
                      f"{sorted(xy)}",
                      "a stack is ONE physical location -- its chiplets differ "
                      "only in z, so a stack moves as a unit")

        zs = [chiplets[c]["tier_z"] for c in members]
        dup = sorted({z for z in zs if z is not None and zs.count(z) > 1})
        if dup:
            rep.error("S012", f"{where} stack {sid}",
                      f"stack {sid} has multiple chiplets on tier(s) {dup}",
                      "two chiplets cannot occupy the same tier of one stack")

    if net_spec_df is not None and "Stack ID" in net_spec_df.columns:
        spec_stacks = {str(s).strip() for s in net_spec_df["Stack ID"]}
        for sid in sorted(stacks):
            if sid not in spec_stacks:
                rep.error("X002", "Network_Spec",
                          f"no row for Stack ID '{sid}' used in Sys_Map",
                          "Network.calc_edge_charc does nw_df.loc[...].iloc[0] "
                          "-> IndexError")
        for sid in sorted(spec_stacks - set(stacks)):
            rep.warn("X003", "Network_Spec",
                     f"row for Stack ID '{sid}' that no chiplet belongs to",
                     "unused; usually a leftover from an earlier configuration")

    rep.ok(f"Sys_Map: {len(chiplets)} chiplet(s) in {len(stacks)} stack(s)")
    return chiplets


# --------------------------------------------------------------------------
# Chip_Map
# --------------------------------------------------------------------------


def check_chip_map(chip_df, chiplets, ai_layers, ai_edges,
                   sa_spec_df, mem_spec_df, rep):
    """-> tiles dict {key: {chiplet, tile, hw, noc, layers, nodes, line}}"""
    where = "Chip_Map"
    missing = [c for c in CHIP_MAP_COLS if c not in chip_df.columns]
    if missing:
        rep.error("C001", where,
                  f"missing required column(s): {', '.join(missing)}",
                  f"expected header: {','.join(CHIP_MAP_COLS)}")
        return {}

    tiles = {}
    noc_seen = defaultdict(dict)          # chiplet -> {(x,y): tile_key}
    compute_of = defaultdict(list)        # layer -> [tile_key]      ("_C")
    mem_of = defaultdict(list)            # (layer, role) -> [tile_key]

    for idx, row in chip_df.iterrows():
        line = idx + 2
        cid = str(row["Chiplet ID"]).strip()
        tid = str(row["Tile ID"]).strip()
        key = f"{cid}_{tid}"
        loc = f"{where} line {line} ({key})"

        if key in tiles:
            rep.error("C002", loc, f"duplicate tile '{key}'",
                      "load_ai_chip builds G_chip keyed by "
                      "'<Chiplet ID>_<Tile ID>'; the later row silently wins")
            continue
        if not re.fullmatch(r"T\d+", tid):
            rep.error("C003", loc, f"Tile ID '{tid}' is not of the form T<int>",
                      "load_ai_chip does int(row[1][1:]) on this cell")
            continue
        if cid not in chiplets:
            rep.error("C004", loc, f"Chiplet ID '{cid}' has no row in Sys_Map",
                      "load_ai_chip does G_sys.nodes[chip_id]['Stack ID'] "
                      "-> KeyError")
            continue

        hw = str(row["HW Type"]).strip()
        if hw not in HW_TYPES:
            rep.error("C007", loc,
                      f"HW Type '{hw}' is not one of {sorted(HW_TYPES)}",
                      "anything else is skipped by Compute.load_compute_tiles "
                      "and its layers never enter tile_map")
            continue

        noc, err = parse_coord(row["NoC Position"], 2)
        if err:
            rep.error("C005", loc, f"NoC Position {err}", 'expected "x,y"')
            continue
        if any(v < 0 for v in noc):
            rep.error("C005", loc,
                      f"NoC Position {noc} has a negative component",
                      "mesh size is inferred as max(x,y)+1; negatives corrupt it")

        if noc in noc_seen[cid]:
            rep.error("C006", loc,
                      f"NoC Position {noc} already used by {noc_seen[cid][noc]} "
                      f"on the same chiplet",
                      "SILENT: load_ai_chip builds noc_tile_dict keyed by "
                      "(x,y,z); duplicates overwrite and one tile drops out of "
                      "the network model entirely")
        else:
            noc_seen[cid][noc] = key

        ai_list, err = parse_list_cell(row["AI Layer"])
        if err:
            rep.error("C008", loc, f"AI Layer {err}",
                      "Compute.load_compute_tiles does ast.literal_eval on this "
                      "cell; it must look like \"['L3', 'L4']\"")
            continue
        node_list, err = parse_list_cell(row["NodeName"])
        if err:
            rep.error("C008", loc, f"NodeName {err}",
                      "Compute.mem_requirement does ast.literal_eval on this cell")
            continue
        ai_list = [str(x) for x in ai_list]

        if not ai_list:
            if hw == "SA":
                rep.error("C009", loc, "SA tile has an empty AI Layer list",
                          "SILENT: num_adders_tile is assigned only inside "
                          "Compute.py's per-layer loop, so an idle SA tile keeps "
                          "it NaN -> NaN tile area -> NaN TOTAL area. An SA tile "
                          "cannot be left idle; remove it and its SA_Spec row")
            else:
                rep.warn("C009", loc,
                         "AI Layer list is empty; this tile does no work",
                         "it still costs area and can inflate the mesh")

        if hw in ("SA", "CPU"):
            if len(node_list) != len(ai_list):
                rep.error("C010", loc,
                          f"AI Layer has {len(ai_list)} entries but NodeName has "
                          f"{len(node_list)}",
                          "for a compute tile NodeName is the per-layer op type "
                          "and must be positionally parallel to AI Layer")
            for layer in ai_list:
                compute_of[layer].append(key)
        else:                                   # Mem Tile
            if not node_list:
                rep.error("C010", loc, "NodeName list is empty",
                          "Compute.mem_requirement reads "
                          "literal_eval(NodeName)[0] -> IndexError")
                continue
            if len(node_list) > 1:
                rep.warn("C011", loc,
                         f"NodeName has {len(node_list)} entries ({node_list}); "
                         f"only the first is used",
                         "a memory tile has ONE role for ALL its layers -- "
                         "Compute.mem_requirement reads NodeName[0][0]. Split "
                         "into separate tiles for mixed roles")
            role = str(node_list[0])[0]
            if role not in MEM_ROLES:
                rep.error("C012", loc,
                          f"memory role '{node_list[0]}' resolves to '{role}', "
                          f"not one of {sorted(MEM_ROLES)}",
                          "only the FIRST CHARACTER of NodeName[0] is read: "
                          "I=Input, W=Weight, O=Output, D=DDR")
            else:
                for layer in ai_list:
                    if layer == "DDR":
                        mem_of[("DDR", "Mem")].append(key)
                    else:
                        mem_of[(layer, role)].append(key)

        tiles[key] = {"chiplet": cid, "tile": tid, "hw": hw, "noc": noc,
                      "layers": ai_list, "nodes": node_list, "line": line}

    if not tiles:
        rep.error("C001", where, "no valid tile rows")
        return {}

    # ---- mesh geometry (constraint 1) ----
    mesh_size = {}
    for cid, coords in noc_seen.items():
        pts = list(coords)
        mesh_size[cid] = max(max(x for x, _ in pts), max(y for _, y in pts)) + 1
        holes = mesh_size[cid] ** 2 - len(pts)
        if holes:
            rep.warn("C013", f"{where} chiplet {cid}",
                     f"{len(pts)} tile(s) occupy a {mesh_size[cid]}x"
                     f"{mesh_size[cid]} mesh ({holes} empty position(s))",
                     "mesh size is inferred as max(x,y)+1, so empty positions "
                     "still cost routers and area")

    by_stack = defaultdict(set)
    for cid in mesh_size:
        by_stack[chiplets[cid]["stack"]].add(cid)
    for sid, members in sorted(by_stack.items()):
        if len({mesh_size[c] for c in members}) > 1:
            rep.error("C014", f"{where} stack {sid}",
                      f"chiplets in stack {sid} have different mesh sizes: "
                      + ", ".join(f"{c}={mesh_size[c]}"
                                  for c in sorted(members)),
                      "load_ai_chip raises ValueError('Inconsistent mesh sizes "
                      "in stack.'). Tile COUNTS may differ, but the bounding box "
                      "must match -- pad the sparser chiplet by placing a tile "
                      "at the far edge")

    # ---- spec-file correspondence (constraint 5) ----
    def spec_keys(df, name):
        if df is None:
            return None
        if "Chiplet ID" not in df.columns or "Tile ID" not in df.columns:
            rep.error("X004", name, "missing 'Chiplet ID' / 'Tile ID' columns")
            return None
        return {f"{str(r['Chiplet ID']).strip()}_{str(r['Tile ID']).strip()}"
                for _, r in df.iterrows()}

    sa_keys = spec_keys(sa_spec_df, "SA_Spec")
    mem_keys = spec_keys(mem_spec_df, "Mem_Spec")

    if sa_keys is not None:
        nan_hint = ("SILENT: area_sa_tile iterates SA_Spec, but num_adders_tile "
                    "is assigned only inside Compute.py's per-layer loop. A spec "
                    "row with no live tile keeps it NaN, so tile area is NaN and "
                    "the TOTAL area/cost silently becomes NaN")
        for key, t in sorted(tiles.items()):
            if t["hw"] == "SA" and key not in sa_keys:
                rep.error("X005", f"{where} line {t['line']} ({key})",
                          f"SA tile '{key}' has no row in SA_Spec",
                          "Compute.load_compute_tiles does sa_df[...].index[0] "
                          "-> IndexError. Add a row: "
                          f"{t['chiplet']},{t['tile']},SA,<x>,<y>,<n_SA>,<prec>,"
                          "<clock>")
        for key in sorted(sa_keys):
            if key not in tiles:
                rep.error("X006", "SA_Spec",
                          f"row for '{key}' which is not in Chip_Map", nan_hint)
            elif tiles[key]["hw"] != "SA":
                rep.error("X006", "SA_Spec",
                          f"row for '{key}', but Chip_Map calls it "
                          f"'{tiles[key]['hw']}'", nan_hint)

    if mem_keys is not None:
        for key, t in sorted(tiles.items()):
            if t["hw"] == "Mem Tile" and key not in mem_keys:
                rep.error("X007", f"{where} line {t['line']} ({key})",
                          f"memory tile '{key}' has no row in Mem_Spec",
                          "SILENT: Compute.mem_requirement iterates Mem_Spec, so "
                          "this tile is skipped entirely and its layers never "
                          "enter tile_map -- surfacing later as a KeyError in "
                          "Network.network_map")
        for key in sorted(mem_keys):
            if key not in tiles:
                rep.error("X008", "Mem_Spec",
                          f"row for '{key}' which is not in Chip_Map",
                          "Compute.mem_requirement does G_chip.nodes[key] "
                          "-> KeyError")
            elif tiles[key]["hw"] != "Mem Tile":
                rep.error("X008", "Mem_Spec",
                          f"row for '{key}', but Chip_Map calls it "
                          f"'{tiles[key]['hw']}'",
                          "it will be read as if it were a memory tile")

    # ---- layer coverage (constraint 7) ----
    if ai_layers is not None:
        known = set(ai_layers) | {"DDR"}
        for key, t in sorted(tiles.items()):
            for layer in t["layers"]:
                if layer not in known:
                    rep.error("C016", f"{where} line {t['line']} ({key})",
                              f"references layer '{layer}', which is not in the "
                              f"AI model's Network.csv",
                              "G_ai_model.nodes[layer] -> KeyError")

        for layer in sorted(ai_layers, key=_layer_sort_key):
            claims = compute_of.get(layer, [])
            if not claims:
                rep.error("C017", f"{where} layer {layer}",
                          f"layer {layer} is not assigned to any SA or CPU tile",
                          "Network.network_map does tile_map[layer+'_C'] "
                          "-> KeyError")
            elif len(claims) > 1:
                rep.error("C018", f"{where} layer {layer}",
                          f"layer {layer} is assigned to {len(claims)} compute "
                          f"tiles ({', '.join(claims)})",
                          "SILENT: tile_map[layer+'_C'] is assigned with no "
                          "duplicate check, so the last row wins and the other "
                          "assignment is discarded")

            outs = mem_of.get((layer, "O"), [])
            if not outs:
                rep.error("C019", f"{where} layer {layer}",
                          f"layer {layer} has no Output ('O') memory tile",
                          "Network.network_map does tile_map[layer+'_O'] "
                          "-> KeyError")
            elif len(outs) > 1:
                rep.error("C020", f"{where} layer {layer}",
                          f"layer {layer}'s output is claimed by {len(outs)} "
                          f"memory tiles ({', '.join(outs)})",
                          "SILENT: tile_map[layer+'_O'] is last-write-wins")

            for role in ("I", "W"):
                dup = mem_of.get((layer, role), [])
                if len(dup) > 1:
                    rep.error("C020", f"{where} layer {layer}",
                              f"layer {layer}'s {MEM_ROLES[role]} memory is "
                              f"claimed by {len(dup)} tiles "
                              f"({', '.join(dup)})",
                              "SILENT: tile_map is last-write-wins")

            eligible = is_sa_eligible(ai_layers[layer]["Type"])
            has_w = bool(mem_of.get((layer, "W")))
            n_in = sum(1 for s, d in (ai_edges or []) if d == layer)
            if eligible and not has_w and n_in <= 1:
                rep.warn("C028", f"{where} layer {layer}",
                         f"layer {layer} ({ai_layers[layer]['Type']}) has no "
                         f"Weight memory tile",
                         "SILENT: Network.network_map guards the weight lookup "
                         "with `if layer+'_W' in tile_map`, so this layer's "
                         "weight traffic is simply never generated. Legitimate "
                         "only for layers with dynamic weights (QKV-style, >1 "
                         "in-edge)")
            if has_w and not eligible:
                rep.warn("C021", f"{where} layer {layer}",
                         f"layer {layer} is of type "
                         f"'{ai_layers[layer]['Type']}' but has a Weight tile",
                         "only MatMul*/Gemm/Conv/*Attention layers consume "
                         "weights; this tile's traffic is never generated")

        for key, t in sorted(tiles.items()):
            if t["hw"] not in ("SA", "CPU"):
                continue
            for layer in t["layers"]:
                if layer not in ai_layers:
                    continue
                lt = ai_layers[layer]["Type"]
                if t["hw"] == "SA" and not is_sa_eligible(lt):
                    rep.warn("C022", f"{where} line {t['line']} ({key})",
                             f"layer {layer} of type '{lt}' is on an SA tile",
                             "it needs a Layer_Mapping row and is costed as a "
                             "systolic-array op; CPU is the intended tile type")
                if t["hw"] == "CPU" and is_sa_eligible(lt):
                    rep.warn("C023", f"{where} line {t['line']} ({key})",
                             f"layer {layer} of type '{lt}' is on a CPU tile",
                             "its MACs are costed as a CPU op, not a systolic "
                             "array -- usually a large mis-estimate")
            if len(t["nodes"]) == len(t["layers"]):
                for layer, node in zip(t["layers"], t["nodes"]):
                    if layer in ai_layers and \
                            str(node) != str(ai_layers[layer]["Type"]):
                        rep.warn("C024", f"{where} line {t['line']} ({key})",
                                 f"NodeName '{node}' for layer {layer} disagrees "
                                 f"with its Network.csv type "
                                 f"'{ai_layers[layer]['Type']}'",
                                 "Network.network_map builds HW Type strings "
                                 "from these; keep them in sync")

        if "L1" not in ai_layers:
            rep.error("C027", where, "the AI model has no layer 'L1'",
                      "Compute.load_compute_tiles does "
                      "G_ai_model.nodes['L1'].keys() to build its column list")

    ddr = mem_of.get(("DDR", "Mem"), [])
    if not ddr:
        rep.error("C025", where,
                  "no DDR memory tile (a 'Mem Tile' row with AI Layer ['DDR'] "
                  "and NodeName ['DDR'])",
                  "Network.network_map does tile_map['DDR_Mem'] -> KeyError")
    elif len(ddr) > 1:
        rep.error("C025", where,
                  f"{len(ddr)} DDR memory tiles ({', '.join(ddr)}); exactly one "
                  f"is supported",
                  "SILENT: tile_map['DDR_Mem'] is last-write-wins")

    if ai_edges and any(s == "In" for s, _ in ai_edges) \
            and not mem_of.get(("L1", "I")):
        rep.error("C026", where,
                  "the AI graph has an edge from 'In' but no Input ('I') memory "
                  "tile is assigned to L1",
                  "Network.network_map does tile_map['L1_I'] -> KeyError")

    rep.ok(f"Chip_Map: {len(tiles)} tile(s); mesh "
           + ", ".join(f"{c}={s}x{s}" for c, s in sorted(mesh_size.items())))
    return tiles


# --------------------------------------------------------------------------
# Layer_Mapping (constraint 8)
# --------------------------------------------------------------------------


def check_layer_mapping(map_df, tiles, ai_layers, rep):
    where = "Layer_Mapping"
    missing = [c for c in LAYER_MAP_COLS if c not in map_df.columns]
    if missing:
        rep.error("L001", where,
                  f"missing required column(s): {', '.join(missing)}",
                  f"expected header: {','.join(LAYER_MAP_COLS)}")
        return

    rows = {}
    for idx, row in map_df.iterrows():
        layer = str(row["Layer ID"]).strip()
        if layer in rows:
            rep.error("L002", f"{where} line {idx + 2} ({layer})",
                      f"duplicate Layer ID '{layer}'",
                      "Compute.py takes mapping_row[dim].values[0], the FIRST "
                      "match; the second row is silently ignored")
            continue
        rows[layer] = (idx + 2, row)

    sa_layers = set()
    for t in tiles.values():
        if t["hw"] == "SA":
            sa_layers.update(t["layers"])
    sa_layers.discard("DDR")

    for layer in sorted(sa_layers, key=_layer_sort_key):
        if layer not in rows:
            rep.error("L003", f"{where} layer {layer}",
                      f"layer {layer} is on an SA tile but has no Layer_Mapping "
                      f"row",
                      "Compute.py does mapping_row[dim].values[0] -> IndexError "
                      "on an empty selection")

    for layer in sorted(set(rows) - sa_layers, key=_layer_sort_key):
        rep.warn("L004", f"{where} line {rows[layer][0]} ({layer})",
                 f"layer {layer} has a mapping row but is on no SA tile",
                 "the row is never read; usually stale")

    for layer in sorted(rows, key=_layer_sort_key):
        line, row = rows[layer]
        loc = f"{where} line {line} ({layer})"

        if ai_layers and layer in ai_layers:
            declared, actual = str(row["Type"]).strip(), \
                str(ai_layers[layer]["Type"])
            if declared != actual:
                rep.warn("L005", loc,
                         f"Type '{declared}' disagrees with the AI model's "
                         f"'{actual}'",
                         "documentation only, but a mismatch usually means the "
                         "row is stale")

        # Compute.py slices these cells as strings, so a non-string (NaN, which
        # is what pandas makes of "NA", or a bare number) is fatal on the
        # prefix-matching path.
        labels, nonstr = {}, []
        for col in IN1_COLS + IN2_COLS:
            v = row[col]
            if isinstance(v, str):
                labels[col] = v.strip()
            else:
                labels[col] = None
                nonstr.append(col)

        wrap_target = {}
        for dim in LOOP_COLS:
            cell = row[dim]
            if pd.isna(cell):
                continue                      # 'NA' -> NaN: loop unused
            tokens = str(cell).strip().split("*")

            bad = False
            for pos, tok in enumerate(tokens):
                if tok not in WRAP_TOKENS:
                    rep.error("L006", loc,
                              f"column '{dim}' contains token '{tok}', which is "
                              f"not one of {sorted(WRAP_TOKENS)}",
                              "these columns are SYMBOLIC -- a literal size such "
                              "as '16' is NOT supported. A bare number is read by "
                              "pandas as an int and dies on .split(); a number in "
                              "a '*' chain reaches dict_wrap['wrap_16'] -> "
                              "KeyError. To vary the array size edit SA_Spec per "
                              "tile. Use 'NA' (not an empty cell) for an unused "
                              "loop")
                    bad = True
                    continue
                if pos != len(tokens) - 1 and tok not in SA_SPEC_COLUMN_TOKENS:
                    rep.error("L007", loc,
                              f"column '{dim}': '{tok}' is not the last element "
                              f"of '{cell}'",
                              f"for every non-last token Compute.py does "
                              f"sa_df.loc[row, '{tok}'], and SA_Spec has no such "
                              f"column -> KeyError. Only "
                              f"{sorted(SA_SPEC_COLUMN_TOKENS)} may precede a '*'")
                    bad = True
                if tok in wrap_target and tok != "Temporal":
                    rep.error("L008", loc,
                              f"'{tok}' is claimed by both loop "
                              f"'{wrap_target[tok]}' and loop '{dim}'",
                              f"SILENT: dict_wrap is ONE dict per LAYER, so the "
                              f"second assignment to wrap_{tok} overwrites the "
                              f"first and one loop's unrolling factor is lost")
                wrap_target[tok] = dim
            if bad:
                continue

            if "Temporal" in tokens:
                rep.warn("L009", loc, f"column '{dim}' maps to 'Temporal'",
                         "Compute.py recomputes wrap_Temporal from the in1_dim* "
                         "cells afterwards, so this has no effect")

            # Resolve the loop name to AI-model dimensions exactly as Compute.py
            # does: exact label match, else a prefix fallback (how Conv rows bind
            # B -> B1,B2,B3 and C -> C1-Unroll-out1_dim3).
            exact = [c for c in IN1_COLS if labels[c] == dim] or \
                    [c for c in IN2_COLS if labels[c] == dim]
            if exact:
                targets = exact
            else:
                if nonstr:
                    rep.error("L010", loc,
                              f"column '{dim}' is set to '{cell}', but no dim "
                              f"cell is labelled '{dim}' and "
                              f"{', '.join(nonstr)} is/are empty or 'NA'",
                              "with no exact match Compute.py falls into its "
                              "prefix-matching branch, which slices EVERY "
                              "in1_dim*/in2_dim* cell as a string -> TypeError on "
                              f"the NA cells. Label a dim cell exactly '{dim}' "
                              f"(or '{dim}1','{dim}2',... with all eight cells "
                              "filled, as the Conv rows do)")
                    continue
                p1 = [c for c in IN1_COLS if labels[c].startswith(dim)]
                p2 = [c for c in IN2_COLS if labels[c].startswith(dim)]
                targets = p1 if len(p1) >= len(p2) else p2
                if not targets:
                    reads_op_split = len(tokens) > 1 and \
                        any(t != "n_SA" for t in tokens[:-1])
                    if reads_op_split:
                        rep.error("L013", loc,
                                  f"column '{dim}' is set to '{cell}' but no dim "
                                  f"cell is labelled '{dim}' or starts with it",
                                  f"map_dim['{dim}'] falls back to 1, so "
                                  f"op_split_dim is never computed; Compute.py "
                                  f"then reads it for the non-last token of "
                                  f"'{cell}' and either raises NameError or "
                                  f"silently reuses the previous loop's value")
                    else:
                        rep.warn("L013", loc,
                                 f"column '{dim}' is set to '{cell}' but no dim "
                                 f"cell is labelled '{dim}' or starts with it",
                                 f"this loop contributes nothing (map_dim=1, "
                                 f"wrap_{tokens[-1]}=1). HISIM's own "
                                 f"depthwise-Conv rows look like this, so it may "
                                 f"be intentional")
                    continue
                vals = [labels[c] for c in targets]
                if any("Unroll" in seg for v in vals for seg in v.split("-")):
                    targets = [v.split("-")[-1] for v in vals]

            if ai_layers and layer in ai_layers:
                for col in targets:
                    if col not in ai_layers[layer]:
                        rep.error("L011", loc,
                                  f"loop '{dim}' resolves to '{col}', which is "
                                  f"not a column of the AI model's Network.csv",
                                  f"Compute.py does "
                                  f"G_ai_model.nodes[layer]['{col}'] -> KeyError")
                    elif pd.isna(ai_layers[layer][col]):
                        rep.warn("L012", loc,
                                 f"loop '{dim}' resolves to '{col}', which is NA "
                                 f"for layer {layer} in Network.csv",
                                 "map_dim becomes NaN and this layer's MAC count "
                                 "and utilisation propagate NaN into the results")

        if not any(pd.notna(row[d]) for d in LOOP_COLS):
            rep.warn("L014", loc,
                     "all of Parallel/A/B/C are NA; this layer maps to nothing",
                     "its MAC count collapses to 1 and the tile looks free")

    rep.ok(f"Layer_Mapping: {len(rows)} row(s) covering "
           f"{len(sa_layers & set(rows))}/{len(sa_layers)} SA-mapped layer(s)")


# --------------------------------------------------------------------------
# Clocks (constraint 9)
# --------------------------------------------------------------------------


def check_clocks(sa_spec_df, mem_spec_df, tiles, network_json, rep):
    """Every component must run at ONE frequency.

    Network.py costs all traffic at the global f_noc / f_nop in Network.json,
    while CPU tiles use config.def_clk_hz. A per-tile clock that disagrees
    changes compute latency but not link latency.
    """
    seen = defaultdict(list)

    for name, df in (("SA_Spec", sa_spec_df), ("Mem_Spec", mem_spec_df)):
        if df is None or CLOCK not in df.columns:
            continue
        for idx, row in df.iterrows():
            who = (f"{name} line {idx + 2} "
                   f"({str(row.get('Chiplet ID', '?')).strip()}_"
                   f"{str(row.get('Tile ID', '?')).strip()})")
            try:
                f = float(row[CLOCK])
            except (TypeError, ValueError):
                rep.error("K001", who,
                          f"clock frequency '{row[CLOCK]}' is not a number")
                continue
            if f <= 0:
                rep.error("K002", who, f"clock frequency is {f:g} Hz",
                          "latency is divided by this -- zero raises "
                          "ZeroDivisionError, negative silently inverts results")
                continue
            seen[f].append(who)

    if not seen:
        return

    if any(t["hw"] == "CPU" for t in tiles.values()):
        try:
            import config

            cpu_clock = float(config.def_clk_hz)
        except (ImportError, AttributeError, TypeError, ValueError):
            cpu_clock = 1e9
        seen[cpu_clock].append("CPU tiles (config.def_clk_hz)")

    for key in ("f_noc", "f_nop"):
        if network_json and key in network_json:
            try:
                seen[float(network_json[key])].append(
                    f"Network.json '{key}' (the NoC/NoP clock)")
            except (TypeError, ValueError):
                pass

    if len(seen) == 1:
        f = next(iter(seen))
        n = sum(len(v) for v in seen.values())
        rep.ok(f"Clocks: uniform at {f:g} Hz across {n} component(s)")
        return

    ranked = sorted(seen.items(), key=lambda kv: -len(kv[1]))
    majority = ranked[0][0]
    rep.error("K003", "Clock frequency",
              "the system is NOT running at a single clock: "
              + ", ".join(f"{f:g} Hz x{len(v)}" for f, v in ranked),
              f"all components must share one frequency; the majority is "
              f"{majority:g} Hz. Update config.def_clk_hz and Network.json "
              f"together, or regenerate the files through the GUI.")
    for f, wheres in ranked[1:]:
        for w in wheres:
            rep.error("K003", w, f"runs at {f:g} Hz, not {majority:g} Hz")


# --------------------------------------------------------------------------
# Geometry sanity (constraint 10)
# --------------------------------------------------------------------------


def check_positive_specs(sa_spec_df, mem_spec_df, net_spec_df, rep):
    groups = [
        ("SA_Spec", sa_spec_df, ["SA_size_x", "SA_size_y", "n_SA", "prec"],
         ["Chiplet ID", "Tile ID"]),
        ("Mem_Spec", mem_spec_df, ["Nbank", "NW", "NB", "CM"],
         ["Chiplet ID", "Tile ID"]),
        ("Network_Spec", net_spec_df,
         ["N_2D_Links_per_tile", "N_3D_Links_per_tile",
          "N_2.5D_channels_per_chiplet_edge"], ["Stack ID"]),
    ]
    for name, df, cols, ident in groups:
        if df is None:
            continue
        for col in cols:
            if col not in df.columns:
                rep.error("V001", name, f"missing required column '{col}'")
                continue
            for idx, row in df.iterrows():
                who = "_".join(str(row[c]).strip() for c in ident
                               if c in df.columns)
                try:
                    v = float(row[col])
                except (TypeError, ValueError):
                    rep.error("V002", f"{name} line {idx + 2} ({who})",
                              f"'{col}' = '{row[col]}' is not a number")
                    continue
                if v <= 0:
                    rep.error("V003", f"{name} line {idx + 2} ({who})",
                              f"'{col}' = {v:g}; must be > 0",
                              "zero or negative geometry produces NaN/inf area, "
                              "divide-by-zero, or inverted latency")


# --------------------------------------------------------------------------
# Memory capacity (advisory)
# --------------------------------------------------------------------------


def check_memory_capacity(tiles, mem_spec_df, ai_layers, rep):
    """Report which layers will spill to DDR before it shows up in the results.

    Mirrors Compute.mem_requirement: each layer is checked INDIVIDUALLY against
    the whole tile's capacity (layers time-share it), and a shortfall is not an
    error -- it silently becomes off-chip DDR traffic.
    """
    if mem_spec_df is None or ai_layers is None:
        return
    need = {"I": "in1_dim", "W": "in2_dim", "O": "out1_dim"}

    cap = {}
    for _, row in mem_spec_df.iterrows():
        try:
            key = (f"{str(row['Chiplet ID']).strip()}_"
                   f"{str(row['Tile ID']).strip()}")
            cap[key] = float(row["NB"]) * float(row["NW"]) * float(row["Nbank"])
        except (TypeError, ValueError, KeyError):
            continue

    spills = []
    for key, t in sorted(tiles.items()):
        if t["hw"] != "Mem Tile" or key not in cap or not t["nodes"]:
            continue
        role = str(t["nodes"][0])[0]
        if role not in need:
            continue
        for layer in t["layers"]:
            if layer == "DDR" or layer not in ai_layers:
                continue
            attrs = ai_layers[layer]
            bits = 1
            for i in range(1, 5):
                v = attrs.get(f"{need[role]}{i}")
                if v is not None and not pd.isna(v):
                    bits *= int(float(v))
            try:
                bits *= int(float(attrs.get("prec", 1) or 1))
            except (TypeError, ValueError):
                pass
            if bits > cap[key]:
                spills.append((key, layer, MEM_ROLES[role], bits, cap[key]))

    for key, layer, role, bits, c in spills:
        rep.warn("M001", f"Chip_Map ({key})",
                 f"layer {layer}'s {role} needs {bits / 8e6:.3f} MB but the tile "
                 f"holds {c / 8e6:.3f} MB",
                 "SILENT: Compute.mem_requirement turns the shortfall into "
                 "off-chip DDR traffic rather than an error. Raise Nbank/NW/NB "
                 "in Mem_Spec, or set config.SET_SUFF_BANKS = True")
    if not spills:
        rep.ok("Memory: every layer fits in its assigned tile (no DDR spill)")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def resolve_paths(model, root):
    return {
        "Chip_Map": os.path.join(root, f"Chip_Map_{model}.csv"),
        "Sys_Map": os.path.join(root, f"Sys_Map_{model}.csv"),
        "Layer_Mapping": os.path.join(root, f"Layer_Mapping_{model}.csv"),
        "SA_Spec": os.path.join(root, "Module_1_Compute", "HISIM_2_0_Files",
                                "HW_configs", f"SA_Spec_{model}.csv"),
        "Mem_Spec": os.path.join(root, "Module_1_Compute", "HISIM_2_0_Files",
                                 "HW_configs", f"Mem_Spec_{model}.csv"),
        "Network_Spec": os.path.join(root, "Module_2_Network", "HISIM_2_0_Files",
                                     "Network_configs",
                                     f"Network_Spec_{model}.csv"),
        "Network_json": os.path.join(root, "Module_2_Network", "HISIM_2_0_Files",
                                     "Network.json"),
        "AI_dir": os.path.join(root, "Module_0_AI_Map",
                               "HISIM_2_0_AI_layer_information", model),
    }


# The six files a run consumes, in the order they are reported.
INPUT_FILES = ["Chip_Map", "Sys_Map", "Layer_Mapping",
               "SA_Spec", "Mem_Spec", "Network_Spec"]


def default_user_files_base(root):
    """<repo>/uploaded_files -- a sibling of the HISIM-SystolicArray directory.

    Derived from `root` rather than hardcoded, so a fresh clone works anywhere.
    """
    return os.path.join(os.path.dirname(os.path.abspath(root).rstrip(os.sep)),
                        "uploaded_files")


def user_files_dir(model, root, base=None):
    """Where this model's editable copies live: <base>/<model>/."""
    if base is None:
        base = default_user_files_base(root)
    return os.path.join(base, model)


def _user_file_path(name, model, udir, base):
    """Prefer <base>/<model>/X_<model>.csv, fall back to a flat <base>/X_<model>.csv."""
    fname = f"{name}_{model}.csv"
    per_model = os.path.join(udir, fname)
    if os.path.isfile(per_model):
        return per_model
    flat = os.path.join(base, fname)
    if os.path.isfile(flat):
        return flat
    return per_model                      # report the preferred location


def export_user_files(model, root, base=None, quiet=False):
    """Publish the freshly generated CSVs into the user's editable folder.

    Called when config.CREATE_DEFAULT_FILES is True, so that a user always has a
    valid, complete starting point to edit. Any previous set is rolled into
    <model>/previous/ rather than being destroyed.
    """
    import shutil
    if base is None:
        base = default_user_files_base(root)
    udir = user_files_dir(model, root, base)
    paths = resolve_paths(model, root)

    missing = [n for n in INPUT_FILES if not os.path.isfile(paths[n])]
    if missing:
        print(f"  note: not exporting -- generated files missing: "
              f"{', '.join(missing)}")
        return None

    existing = [n for n in INPUT_FILES
                if os.path.isfile(os.path.join(udir, f"{n}_{model}.csv"))]
    if existing:
        prev = os.path.join(udir, "previous")
        os.makedirs(prev, exist_ok=True)
        for n in existing:
            shutil.copy2(os.path.join(udir, f"{n}_{model}.csv"),
                         os.path.join(prev, f"{n}_{model}.csv"))

    os.makedirs(udir, exist_ok=True)
    for n in INPUT_FILES:
        shutil.copy2(paths[n], os.path.join(udir, f"{n}_{model}.csv"))

    if not quiet:
        print(f"Exported the generated input files for '{model}' to:")
        print(f"  {udir}")
        if existing:
            print(f"  (the previous set was kept in {os.path.join(udir, 'previous')})")
        print("  Edit them there, then set config.CREATE_DEFAULT_FILES = False "
              "to run on your own files.")
    return udir


def import_user_files(model, root, base=None, quiet=False):
    """Stage the user's CSVs into the locations HISIM reads.

    Called when config.CREATE_DEFAULT_FILES is False. Exits with an explanatory
    message if the user's folder is missing or incomplete, rather than letting
    HISIM silently run on whatever happens to be left in the tree.
    """
    import shutil
    if base is None:
        base = default_user_files_base(root)
    udir = user_files_dir(model, root, base)
    paths = resolve_paths(model, root)

    # A GUI user may replace only one or two files (for example SA_Spec and
    # Mem_Spec) while keeping the generated maps. Prefer an uploaded file for
    # each kind, then fall back to the legacy in-tree file for that same model.
    # This keeps partial edits useful while still requiring every consumer
    # input to exist before validation.
    uploaded = {n: _user_file_path(n, model, udir, base) for n in INPUT_FILES}
    src = {
        n: uploaded[n] if os.path.isfile(uploaded[n]) else paths[n]
        for n in INPUT_FILES
    }
    missing = [n for n in INPUT_FILES if not os.path.isfile(src[n])]
    if missing:
        print("=" * 74)
        print(f"No usable input files for '{model}' in:")
        print(f"  {udir} (with fallback to the simulator's generated files)")
        print(f"Missing: {', '.join(f'{n}_{model}.csv' for n in missing)}")
        print()
        print("Generate a starting set first:")
        print("  1. in config.py set CREATE_DEFAULT_FILES = True")
        print("  2. python HISIM.py            (writes the six files to the "
              "folder above)")
        print("  3. edit them, set CREATE_DEFAULT_FILES = False, run again")
        print("=" * 74)
        sys.exit(1)

    for n in INPUT_FILES:
        dst = paths[n]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(src[n]) != os.path.abspath(dst):
            shutil.copy2(src[n], dst)

    if not quiet:
        print(f"Loaded your input files for '{model}' from:")
        print(f"  {os.path.dirname(src['Chip_Map'])}")
    return udir


def _read_csv(path, rep, code, label, required=True):
    if not os.path.isfile(path):
        if required:
            rep.error(code, label, f"missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        rep.error(code, label, f"could not read {path}: {exc}")
        return None


def validate(model, root, user_base=None):
    """Validate every input CSV for `model`. Returns a Report.

    If `user_base` is given, the six input files are read from the user's folder
    (<user_base>/<model>/) instead of the in-tree locations, so an uploaded set
    can be checked without staging it first. The AI model and Network.json are
    always read from the tree.
    """
    rep = Report()
    paths = resolve_paths(model, root)
    if user_base is not None:
        udir = user_files_dir(model, root, user_base)
        for n in INPUT_FILES:
            paths[n] = _user_file_path(n, model, udir, user_base)

    chip_df = _read_csv(paths["Chip_Map"], rep, "C001", "Chip_Map")
    sys_df = _read_csv(paths["Sys_Map"], rep, "S001", "Sys_Map")
    map_df = _read_csv(paths["Layer_Mapping"], rep, "L001", "Layer_Mapping")
    sa_spec_df = _read_csv(paths["SA_Spec"], rep, "X004", "SA_Spec", False)
    mem_spec_df = _read_csv(paths["Mem_Spec"], rep, "X004", "Mem_Spec", False)
    net_spec_df = _read_csv(paths["Network_Spec"], rep, "X004", "Network_Spec",
                            False)

    if chip_df is None or sys_df is None or map_df is None:
        return rep

    network_json = None
    if os.path.isfile(paths["Network_json"]):
        try:
            with open(paths["Network_json"]) as f:
                network_json = json.load(f)
        except Exception as exc:
            rep.warn("K004", "Network.json", f"could not read: {exc}",
                     "tile clocks cannot be cross-checked against f_noc / f_nop")

    ai_layers, ai_edges = load_ai_model(paths["AI_dir"], rep)

    chiplets = check_sys_map(sys_df, net_spec_df, rep)
    tiles = check_chip_map(chip_df, chiplets, ai_layers, ai_edges,
                           sa_spec_df, mem_spec_df, rep)
    check_layer_mapping(map_df, tiles, ai_layers, rep)
    check_clocks(sa_spec_df, mem_spec_df, tiles, network_json, rep)
    check_positive_specs(sa_spec_df, mem_spec_df, net_spec_df, rep)
    check_memory_capacity(tiles, mem_spec_df, ai_layers, rep)
    return rep


def validate_or_exit(model, root, strict=False, verbose=False, quiet=False):
    """Validate and abort the process if anything is wrong.

    Called by HISIM.py before the model runs. Raises SystemExit(1) on failure so
    the run stops before producing numbers from an invalid configuration.
    """
    banner = "=" * 74
    if not quiet:
        print(banner)
        print(f"Validating input CSV files for '{model}'")
        print(banner)

    rep = validate(model, root)
    failed = rep.n_errors > 0 or (strict and rep.n_warnings > 0)
    detail = "  (run `python validate_maps.py -v` or set " \
             "config.VALIDATE_VERBOSE = True for details)"

    if verbose:
        print(rep.render(True))
    elif failed:
        # Show the errors in full; a large model can legitimately produce
        # hundreds of DDR-spill warnings and they would bury the actual fault.
        print(rep.render(False, severities=("ERROR",)))
        if rep.n_warnings:
            print(f"Warnings: {rep.summarise('WARN')}")
            print(detail)
    elif rep.n_warnings:
        print(f"Warnings: {rep.summarise('WARN')}")
        print(detail)
    print(f"{rep.n_errors} error(s), {rep.n_warnings} warning(s).")

    if failed:
        print(banner)
        print("VALIDATION FAILED -- HISIM will not run.")
        print("Fix the errors above, or set config.VALIDATE_MAPS = False to "
              "skip this check.")
        print(banner)
        sys.exit(1)

    if not quiet:
        print("Validation passed. Running HISIM.")
        print(banner)
    return rep


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Validate HISIM's input CSV files before a run.")
    ap.add_argument("--model", default=None,
                    help="AI model name (default: config.aimodel)")
    ap.add_argument("--root", default=here,
                    help="HISIM-SystolicArray directory (default: this script's "
                         "directory)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list every finding and the checks that passed")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on warnings as well as errors")
    ap.add_argument("--user-files", nargs="?", const="", default=None,
                    metavar="DIR",
                    help="validate the six CSVs in the user folder "
                         "(<DIR>/<model>/) instead of the in-tree copies. With no "
                         "DIR, uses config.USER_FILES_DIR, else <repo>/uploaded_files")
    args = ap.parse_args(argv)

    model = args.model
    if model is None:
        sys.path.insert(0, args.root)
        try:
            import config
            model = config.aimodel
        except Exception as exc:
            print(f"error: could not import config from {args.root} ({exc}); "
                  f"pass --model", file=sys.stderr)
            return 2

    user_base = None
    if args.user_files is not None:
        user_base = args.user_files or None
        if user_base is None:
            sys.path.insert(0, args.root)
            try:
                import config
                user_base = getattr(config, "USER_FILES_DIR", None)
            except Exception:
                user_base = None
        if user_base is None:
            user_base = default_user_files_base(args.root)

    print(f"Validating model '{model}'")
    print(f"  root: {args.root}")
    if user_base:
        print(f"  input files: {user_files_dir(model, args.root, user_base)}")
    print()

    rep = validate(model, args.root, user_base)
    print(rep.render(args.verbose))
    print(f"{rep.n_errors} error(s), {rep.n_warnings} warning(s).")

    if rep.n_errors:
        print("FAIL -- HISIM would crash or produce wrong numbers with these "
              "files.")
        return 1
    if args.strict and rep.n_warnings:
        print("FAIL (--strict) -- warnings present.")
        return 1
    print("PASS -- safe to run HISIM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
