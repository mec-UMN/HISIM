import time
import os
import pandas as pd
import sys
import math
import json
import networkx as nx
from matplotlib.patches import Rectangle
from prettytable import PrettyTable
#sys.path.append(os.path.abspath("../.."))
from Module_1_Compute.HISIM_2_0_Files.Mem import *
from Module_1_Compute.HISIM_2_0_Files.SA import *
from Module_1_Compute.HISIM_2_0_Files.CPU import *
import matplotlib.pyplot as plt
import numpy as np
import itertools
from functools import reduce
import operator
import config

aimodel=config.aimodel
main_dir = config.main_dir
DEBUG=config.DEBUG

current_dir = os.path.dirname(__file__)
target_file_path = os.path.join(current_dir, 'HW_configs')

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def load_hw_specs(mem_config, SA_config, mapping_config,lut_config):
    mem_df = pd.read_csv(mem_config)
    sa_df = pd.read_csv(SA_config) 
    mapping_df = pd.read_csv(mapping_config)
    lut_df = pd.read_csv(lut_config)
    #print(mem_df) 
    #print(sa_df)
    return mem_df, sa_df, mapping_df,lut_df

def determine_reuse_mul(G_ai_model, layer_idx, mem_place):
    node=G_ai_model.nodes[layer_idx]
    reuse_mul = 1
    if mem_place == "W":
        for i in range(4, 0, -1):
            dim = node.get(f"wdim{i}", 1)
            #print("checking:", dim, "at i=", i)

            # Skip if it's NA or NaN
            if dim == "NA" or pd.isna(dim):
                continue

            reuse_mul *= int(float(dim))  # convert safely
            break
    elif mem_place == "I":
        prev_layer_idx= [list(G_ai_model.in_edges(layer_idx))[i][0] for i in range(len(list(G_ai_model.in_edges(layer_idx))))]
        next_layer_idx_list = [list(G_ai_model.out_edges(i)) for i in prev_layer_idx]
        reuse_mul*=len(list(itertools.chain.from_iterable(next_layer_idx_list)))
        #import pdb; pdb.set_trace()
    return reuse_mul

def mem_requirement(mem_df, G_ai_model, G_chip,tile_map, ip_list):
    mem_req={}
    for layer_idx in G_ai_model.nodes():
        if layer_idx !="In":
            values = [G_ai_model.nodes[layer_idx].get(f"indim{i}", 1) for i in range(1, 5)]
            mem_req[layer_idx + "_I"] = math.prod([int(v) if not math.isnan(v) else 1 for v in values])*G_ai_model.nodes[layer_idx].get("prec", 1)
            if G_ai_model.nodes[layer_idx]["Type"]== "Matmul":
                values = [G_ai_model.nodes[layer_idx].get(f"wdim{i}", 1) for i in range(1, 5)]
                mem_req[layer_idx + "_W"] = math.prod([int(v) if not math.isnan(v) else 1 for v in values])*G_ai_model.nodes[layer_idx].get("prec", 1)
            values = [G_ai_model.nodes[layer_idx].get(f"outdim{i}", 1) for i in range(1, 5)]
            mem_req[layer_idx + "_O"] = math.prod([int(v) if not math.isnan(v) else 1 for v in values])*G_ai_model.nodes[layer_idx].get("prec", 1)
        
    mem_size={}
    for idx, row in mem_df.iterrows():
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        layer_idx=G_chip.nodes[chip_tile_idx]["AI Layer"]
        #print(layer_idx, chip_tile_idx)
        layer_type=G_ai_model.nodes[layer_idx]["Type"] if layer_idx!="DDR" else "DDR"
        mem_place=G_chip.nodes[chip_tile_idx]["Comment"]
        mem_df.loc[idx, "Next Layer Type"] = layer_type
        mem_df.loc[idx, "mem_place"] = mem_place[0] if layer_idx!="DDR" else "Mem"
        for i in range(1, 5):
            if mem_place[0] == "I":
                mem_df.loc[idx, f"dim{i}"] = G_ai_model.nodes[layer_idx].get(f"indim{i}", 1) 
            elif mem_place[0] == "W":
                mem_df.loc[idx, f"dim{i}"] = G_ai_model.nodes[layer_idx].get(f"wdim{i}", 1) 
            elif mem_place[0] == "O":
                mem_df.loc[idx, f"dim{i}"] = G_ai_model.nodes[layer_idx].get(f"outdim{i}", 1)
        #print(row)
        mem_name = layer_idx+"_"+mem_place[0] if layer_idx!="DDR" else "DDR_Mem"
        mem_size[mem_name]=row["NB"]*row["NW"]*row["Nbank"]*row["prec"]
        mem_df.loc[idx, "Accesses_r"]=math.ceil(mem_req.get(mem_name,0)/row["NB"]) if layer_idx!="DDR" else 0
        mem_df.loc[idx, "Accesses_w"]=math.ceil(mem_req.get(mem_name,0)/row["NB"]) if layer_idx!="DDR" else 0
        mem_df.loc[idx, "DDR_access"]=mem_size[mem_name]<mem_req.get(mem_name,0) if layer_idx!="DDR" else True
        #DDR always needs to access/store the input and output layers of the algortihm and also weights of the algortihm
        if len(list(G_ai_model.in_edges(layer_idx))) == 0 or layer_idx+"_"+mem_place[0] == "L1_I":
            mem_df.loc[idx, "DDR_access"]=True
        #if mem_place[0] == "W":
            #mem_df.loc[idx, "DDR_access"]=True
        if len(list(G_ai_model.out_edges(layer_idx)))== 0 and mem_place[0] == "O":
            mem_df.loc[idx, "DDR_access"]=True
                

        if mem_df.loc[idx, "DDR_access"] and layer_idx!="DDR":
            mem_df.loc[idx, f"ddr_reuse_mul_r"] = 1 if mem_place[0] == "O" else determine_reuse_mul(G_ai_model, layer_idx, mem_place[0])
        else:
            mem_df.loc[idx, f"ddr_reuse_mul_r"] = 1
        mem_df.loc[idx, f"ddr_reuse_mul_w"] = 1
        mem_df.loc[idx,"F_non_zeros_r"]=0.5
        mem_df.loc[idx,"F_non_zeros_w"]=0.5
        mem_df.loc[idx, "mem_req"]=mem_req.get(mem_name,0)
        tile_map[mem_name]=chip_tile_idx
        if "Mem_"+str(row["NB"])+"bits x"+str(row["NW"])+"words x"+str(row["Nbank"])+"banks" not in ip_list:
            ip_list["Mem_"+str(row["NB"])+"bits x"+str(row["NW"])+"words x"+str(row["Nbank"])+"banks"] = [chip_tile_idx]
        else:
            ip_list["Mem_"+str(row["NB"])+"bits x"+str(row["NW"])+"words x"+str(row["Nbank"])+"banks"].append(chip_tile_idx)
    
    #bool_mem_flags = {k: mem_size[k] > mem_req.get(k,0) for k in mem_size}
    #bool_ddr_access_flags = {k: mem_size[k] < mem_req.get(k,0) for k in mem_size}
    #DDR always needs to access/store the input and output layers of the algortihm
    #bool_ddr_access_flags["L1_I"]=True
    #bool_ddr_access_flags[list(tile_map.keys())[-1]]=True
    #import pdb; pdb.set_trace()

    DDR_access_table = PrettyTable(["AI Layer", "Mem Stores I/O/W", "Chiplet ID", "Tile ID", "Memory Size (MB)", "Memory required by AI Layer (MB)", "DDR access required"])
    [DDR_access_table.add_row([k.split('_')[0], k.split('_')[1], tile_map.get(k,0).split('_')[0], tile_map.get(k,0).split('_')[1], v*1E-6/8, mem_req.get(k,0)*1E-6/8, mem_df[(mem_df["Chiplet ID"] == tile_map.get(k,0).split('_')[0]) & (mem_df["Tile ID"] == tile_map.get(k,0).split('_')[1])]["DDR_access"].values[0]]) 
    for k,v in mem_size.items() if k!="DDR_Mem"]
    #print(DDR_access_table)
    #import pdb; pdb.set_trace()
    return DDR_access_table, tile_map, mem_req, ip_list, mem_df, mem_size

def load_compute_tiles(G_ai_model, G_chip):
    mem_df, sa_df, mapping_df, lut_df = load_hw_specs(f"{target_file_path}/Mem_Spec_{aimodel}.csv", f"{target_file_path}/SA_Spec_{aimodel}.csv", f"{main_dir}/Layer_Mapping_{aimodel}.csv", f"{target_file_path}/Mem_LUT.csv")

    tile_map = {}
    rows = []
    ip_list={}

    # Precompute the base columns
    base_cols = ["Chiplet ID", "Tile ID", "HW Type", "Clock Frequency (Hz)"]
    layer_cols = list(G_ai_model.nodes["L1"].keys()) 
    all_cols = base_cols + layer_cols

    for chip_tile_idx in G_chip.nodes():
        layer_idx = G_chip.nodes[chip_tile_idx].get("AI Layer", 1)
        hw_type = G_chip.nodes[chip_tile_idx].get("HW Type", 1)
        chip_id=G_chip.nodes[chip_tile_idx].get("Chiplet ID", 1)
        tile_id=G_chip.nodes[chip_tile_idx].get("Tile ID", 1)

        if hw_type == "CPU":
            row_dict = {
                "Chiplet ID": chip_id,
                "Tile ID": tile_id,
                "HW Type": hw_type,
                "Clock Frequency (Hz)": 1e9,
                **dict(G_ai_model.nodes[layer_idx]),            
            }
            rows.append(row_dict)
            tile_map[f"{layer_idx}_C"] = chip_tile_idx
            if "CPU" not in ip_list:
                ip_list["CPU"] =  [chip_tile_idx]
            else:
                ip_list["CPU"].append(chip_tile_idx)

        elif hw_type == "SA":
            row_sa = sa_df[(sa_df["Chiplet ID"] == chip_id) & (sa_df["Tile ID"] == tile_id)].index[0]
            mapping_row = mapping_df[mapping_df["Layer ID"] == layer_idx]
            n_SA=sa_df.loc[row_sa, "n_SA"]
            SA_size_y=sa_df.loc[row_sa, "SA_size_y"]
            SA_size_x=sa_df.loc[row_sa, "SA_size_x"]
           
            #Obtain mapping with respect to matrix dimensions
            map_dim = {}
            map_sa_loc ={}
            # Columns to check
            indim_cols = [f"indim{i}" for i in range(1,5)]
            wdim_cols  = [f"wdim{i}"  for i in range(1,5)]
            #import pdb; pdb.set_trace()

            layer_macs = 1

            for dim in ["Parallel","A","B","C"]:
                map_sa_loc[dim]=mapping_row[dim].values[0] 
                map_sa_loc_list = map_sa_loc[dim].split('*') if pd.notna(map_sa_loc[dim]) else []

                # check indim first
                val_list = [col for col in indim_cols if mapping_row[col].values[0]==dim]
                # if nothing in indim, check wdim
                if not val_list:
                    val_list = [col for col in wdim_cols if mapping_row[col].values[0]==dim]

                if len(val_list)>1:
                    print("Error: Multiple mapping found for dimension {dim}")
                if val_list:  # store the mapped dimension name/value
                    map_dim[dim] = G_ai_model.nodes[layer_idx][val_list[0]]
                    layer_macs *= map_dim[dim]
                    op_split_dim=math.ceil((map_dim[dim])**(1/len(map_sa_loc_list)))
                #print(map_sa_loc)
                count=0
                
                for idx, loc in enumerate(map_sa_loc_list):
                    if idx == len(map_sa_loc_list)-1:
                        # assign the remaining dimension to the last location
                        sa_df.loc[row_sa, "wrap_"+loc]=map_dim[dim]//count if count>0 else map_dim[dim]
                    else:
                        if loc =="n_SA":
                            sa_df.loc[row_sa, "wrap_"+loc]=n_SA 
                            count+=sa_df.loc[row_sa, "wrap_"+loc]
                        else:
                            sa_df.loc[row_sa, "wrap_"+loc]=op_split_dim if op_split_dim>sa_df.loc[row_sa, loc] else sa_df.loc[row_sa, loc]
                            count+=sa_df.loc[row_sa, "wrap_"+loc]    
                        #import pdb; pdb.set_trace()

            sa_df.loc[row_sa, "layer_macs"] = layer_macs
            if n_SA == 1:
                num_adders_tile = SA_size_y*SA_size_x
                num_adder_stages=1
            else:
                num_adders_tile = (n_SA-1)*SA_size_y*SA_size_x  # Total number of adders within one tile
                num_adder_stages=math.ceil(math.log2(n_SA)) if map_dim.get("Parallel", 1)!="n_SA" else 1 
            sa_df.loc[row_sa, "num_adders_tile"] = num_adders_tile
            sa_df.loc[row_sa, "num_adder_stages"] = num_adder_stages

            # Temporal: multiply all temporal and parallelism entries
            temporal_vals = [col for col in indim_cols if mapping_row[col].values[0]=="Temporal"]
            map_dim["Temporal"] = prod([G_ai_model.nodes[layer_idx][val] for val in temporal_vals])
            sa_df.loc[row_sa, "wrap_Temporal"] = map_dim["Temporal"]
            tile_map[f"{layer_idx}_C"] = chip_tile_idx
            if f"SA_{n_SA}_{SA_size_y}x{SA_size_x}" not in ip_list:
                ip_list[f"SA_{n_SA}_{SA_size_y}x{SA_size_x}"] = [chip_tile_idx]
            else:
                ip_list[f"SA_{n_SA}_{SA_size_y}x{SA_size_x}"].append(chip_tile_idx)

    cpu_df = pd.DataFrame(rows, columns=all_cols)

    return cpu_df, tile_map, mem_df, sa_df, mapping_df, lut_df, ip_list


def plot_component(areas, latencies, energies, noc_positions, filename, title, scale_factor, labels=None, colors=None, spacing=0.5, annotate=True):
    if colors is None:
        colors ={k: "skyblue" for k in areas.keys()}
    if labels is None:
        labels ={k: "" for k in areas.keys()}
    #print(areas, noc_positions)
    coords = {k: v for k, v in noc_positions.items()}
    #import pdb; pdb.set_trace()

    # Compute side lengths proportional to sqrt(area)
    sides = {k: math.sqrt(a) for k, a in areas.items()}

    scale_factor=math.ceil(max(sides.values()))
    # Scale coordinates based on side length (optional spacing)
    scaled_coords = {k: (coords[k][0]*scale_factor, coords[k][1]*scale_factor) for k in coords}
    #import pdb; pdb.set_trace()
    fig, ax = plt.subplots()

    # Track extents for axis scaling
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for comp in areas:
        #print(scaled_coords, coords, noc_positions, sides, comp)
        x, y = scaled_coords[comp]
        s = sides[comp]
        # Draw square centered at (x, y)
        ax.add_patch(Rectangle((x, y), s, s, fill=True, color=colors[comp]))
        if annotate:
            if "router" not in comp and "3dlink" not in comp and len(comp.split('_')) <= 2:
                ax.text(x+s/2, y, comp+"\n"+labels[comp], ha='center', va='bottom', fontsize=6)
            else:
                if "router" in comp and latencies!={}:
                    ax.text(math.floor(x)+0.5,math.ceil(y), f"r:\n{latencies[comp]:.2e}s\n{energies[comp]:.2e}J", ha='left', va='top', fontsize=4)
                elif "3dlink" in comp and latencies!={}:
                    ax.text(math.floor(x),math.ceil(y), f"3d:\n{latencies[comp]:.2e}s\n{energies[comp]:.2e}J", ha='left', va='top', fontsize=4)
        # Update bounds
        min_x = min(min_x, x )
        max_x = max(max_x, x + s)
        min_y = min(min_y, y )
        max_y = max(max_y, y + s)
    #place note: not to scale on the plot
    ax.text(min_x, min_y-2*spacing, "Note: Not to scale", ha='left', va='top', fontsize=8, color="red")
    # Set axis limits based on content
    ax.set_xlim(min_x - spacing, max_x + spacing)
    ax.set_ylim(min_y - spacing, max_y + spacing)
    ax.set_xticks(range(math.ceil(min_y - spacing),  math.ceil(max_x + spacing), scale_factor))
    ax.set_yticks(range(math.ceil(min_y - spacing),  math.ceil(max_y + spacing), scale_factor))
    ax.grid(True, linestyle="--", alpha=0.5)

    # Auto figure size proportional to layout
    width = (max_x - min_x) + 2 * spacing
    height = (max_y - min_y) + 2 * spacing
    fig.set_size_inches(width, height)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{title} Map")
    
    
    fig.savefig(filename, dpi=300, bbox_inches="tight")

def compute_main_fn(G_ai_model, G_chip, tile_ids):
    file_path = current_dir+'/Compute.json'
    with open(file_path, 'r') as f:
        compute_json_data  = json.load(f)
    cpu_df, tile_map, mem_df, sa_df, mapping_df, lut_df, ip_list = load_compute_tiles(G_ai_model, G_chip)
    DDR_access_table, tile_map, mem_req, ip_list, mem_df, mem_size = mem_requirement(mem_df, G_ai_model, G_chip, tile_map, ip_list)
    #import pdb; pdb.set_trace()
    tile_area={}
    tile_area, tile_buf_df=area_sa_tile(sa_df,compute_json_data,tile_area, lut_df)
    tile_area=area_cpu_tile(cpu_df,compute_json_data, tile_area)
    tile_area=area_mem_tile(mem_df, compute_json_data, tile_area, lut_df)
    tile_latency = {}
    tile_latency_breakdown = {}
    tile_latency, tile_latency_breakdown, compute_latency = latency_sa_tile(sa_df, compute_json_data, tile_latency, tile_latency_breakdown)
    tile_latency, tile_latency_breakdown, compute_latency = latency_cpu_tile(cpu_df, compute_json_data, tile_latency, tile_latency_breakdown, compute_latency)
    tile_latency, tile_latency_breakdown, mem_latency, ddr_latency = latency_mem_tile(mem_df, compute_json_data, tile_latency, tile_latency_breakdown)
    tile_latency, tile_latency_breakdown, mem_latency, DDR_access_table, mem_req, mem_df=update_ddr_mem_latency(mem_df, compute_json_data, tile_latency,tile_latency_breakdown, mem_latency, tile_map, DDR_access_table, mem_req, mem_size)
    #import pdb; pdb.set_trace()
    tile_energy  = {}
    tile_energy, tile_energy_breakdown, compute_energy = energy_sa_tile(sa_df, compute_json_data, tile_energy, tile_latency_breakdown, tile_latency,tile_buf_df, lut_df)
    tile_energy, compute_energy = energy_cpu_tile(cpu_df, compute_json_data, tile_energy, tile_latency,compute_energy) 
    tile_energy, mem_energy, ddr_energy = energy_mem_tile(mem_df, compute_json_data, tile_energy, tile_latency_breakdown, lut_df)
    #import pdb; pdb.set_trace()

    tile_power = {k: tile_energy[k]/tile_latency[k] if tile_latency[k]>0 else 0 for k in tile_energy}
    ddr_energy_list={k:ddr_energy.get(k, 0) for k in tile_energy}
    ddr_latency_list={k:ddr_latency.get(k, 0) for k in tile_latency}
    for name, default in [("Tile_Area", tile_area),
                        ("Tile_Power", tile_power),
                        ("Tile_Energy", tile_energy),
                        ("DDR_Energy", ddr_energy_list),
                        ("Tile_Latency", tile_latency),
                        ("DDR_Latency", ddr_latency_list),]:
            nx.set_node_attributes(G_chip,
            default if isinstance(default, dict) else {k: default for k in tile_area},
            name)
    compute_area={k: v for k, v in tile_area.items() if G_chip.nodes[k]["HW Type"].endswith("SA") or G_chip.nodes[k]["HW Type"].endswith("CPU")}
    mem_area={k: v for k, v in tile_area.items() if G_chip.nodes[k]["HW Type"].startswith("Mem")}
    #import pdb; pdb.set_trace()
    print("----------Compute and Memory Summary---------------")
    print("Compute Area of the Chip is:", sum(compute_area.values()), "mm2")
    print("Memory Area of the Chip is:", sum(mem_area.values()), "mm2")
    print("Compute Latency of the Chip is:", sum(compute_latency.values()), "s")
    print("Memory Latency of the Chip is:", sum(mem_latency.values()), "s")
    print("DDR Latency of the Chip is:",  sum(ddr_latency.values()), "s")
    print("Compute Energy of the Chip is:", sum(compute_energy.values()), "J")
    print("Memory Energy of the Chip is:", sum(mem_energy.values()), "J")
    print("DDR Energy of the Chip is:",  sum(ddr_energy.values()), "J")
    print("----------------------------------------")
    if DEBUG:
        with open(f"{target_file_path}/ddr_table.txt", "w") as f:
            f.write(str(DDR_access_table))

        df = pd.DataFrame(DDR_access_table._rows, columns=DDR_access_table.field_names)
        df = df[df["AI Layer"] != "DDR"]
        df["Chiplet_Tile"] = df["Chiplet ID"] + "_" + df["Tile ID"].astype(str)
        ax = df.plot(
            x="Chiplet_Tile", 
            y=["Memory Size (MB)", "Memory required by AI Layer (MB)"], 
            kind="bar", 
            figsize=(8, 6)
        )
        for i, val in enumerate(df["DDR access required"]):
            ax.text(i, max(df["Memory Size (MB)"].iloc[i], df["Memory required by AI Layer (MB)"].iloc[i]) * 1.02,
                    f"DDR\n={str(val)[:4]}", ha="center", fontsize=9, color="red")

        plt.ylabel("Data Volume in MB")
        plt.title("Memory Usage per Chiplet/Tile")
        plt.xticks(rotation=45, ha="right")
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.5)
        os.makedirs(f"{target_file_path}", exist_ok=True)
        plt.savefig(f"{target_file_path}/ddr_bar_graph.png", dpi=300, bbox_inches="tight")

        # Extract keys and values
        labels = list(tile_area.keys())
        values = list(tile_area.values())

        plt.figure(figsize=(12, 6))
        plt.bar(labels, values)

        plt.xticks(rotation=90)  # Rotate labels for readability
        plt.ylabel("Area in mm2")
        plt.title("Area of each tile")

        plt.tight_layout()
        os.makedirs(f"{target_file_path}", exist_ok=True)
        plt.savefig(f"{target_file_path}/area_bar_graph.png", dpi=300, bbox_inches="tight")

        # Extract keys and values
        labels = list(tile_latency.keys())
        values = [tile_latency[k] + ddr_latency.get(k, 0) for k in tile_latency]
        #import pdb; pdb.set_trace()
        plt.figure(figsize=(12, 6))
        plt.bar(labels, values)

        plt.xticks(rotation=90)  # Rotate labels for readability
        plt.ylabel("Latency in seconds")
        plt.title("Latency of each tile")
        plt.yscale("log")
        plt.text(0, max(values)*1.1, "Note: Y-axis is in log scale", fontsize=10, color="red")
        plt.tight_layout()
        os.makedirs(f"{target_file_path}", exist_ok=True)
        plt.savefig(f"{target_file_path}/latency_bar_graph.png", dpi=300, bbox_inches="tight")

        # Extract keys and values
        labels = list(tile_energy.keys())
        values = [tile_energy[k] + ddr_energy.get(k, 0) for k in tile_energy]

        plt.figure(figsize=(12, 6))
        plt.bar(labels, values)

        plt.xticks(rotation=90)  # Rotate labels for readability
        plt.ylabel("Energy in Joules")
        plt.title("Energy Consumption of each tile")
        plt.yscale("log")
        plt.text(0, max(values)*1.1, "Note: Y-axis is in log scale", fontsize=10, color="red")
        plt.tight_layout()
        os.makedirs(f"{target_file_path}", exist_ok=True)
        plt.savefig(f"{target_file_path}/energy_bar_graph.png", dpi=300, bbox_inches="tight")

        for chip_idx, tile_idx_dict in tile_ids.items():
            #print(chip_idx, tile_idx_dict)
            chip_tile_idx_list=[chip_idx+"_T"+str(tile_idx) for tile_idx in tile_idx_dict]
            filename=f"{main_dir}/Results/Chip_Map/Chip_map_{chip_idx}.png"
            chip_tile_area= {k: v for k, v in tile_area.items() if k in chip_tile_idx_list}
            chip_tile_latency= {k: v for k, v in tile_latency.items() if k in chip_tile_idx_list}
            chip_tile_energy= {k: v for k, v in tile_energy.items() if k in chip_tile_idx_list}
            #nx.set_node_attributes(G_chip, tile_area, "Area")
            chip_noc_positions= {k: tuple(map(int, v.split(','))) for k, v in nx.get_node_attributes(G_chip, "NoC Position").items() if k in chip_tile_idx_list}   
            # Compute side lengths proportional to sqrt(area)
            sides = {k: math.sqrt(a) for k, a in chip_tile_area.items()}
            scale_factor=math.ceil(max(sides.values()))
            plot_component(chip_tile_area, chip_tile_latency, chip_tile_energy, chip_noc_positions , filename, title=f"Chip {chip_idx}", scale_factor=scale_factor, labels= {k : G_chip.nodes[k].get("HW Type", "") for k in chip_tile_area})
        #plt.show()
    #import pdb; pdb.set_trace()
    #print(tile_area)
    return G_chip, tile_map, mem_req, ip_list

#import pdb; pdb.set_trace()
