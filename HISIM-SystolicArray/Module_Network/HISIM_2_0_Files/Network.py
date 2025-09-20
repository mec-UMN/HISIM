import math
import numpy as np
import sys
import os
from tabulate import tabulate
sys.path.append(os.path.abspath("../.."))
from Module_Network.orion_power_area import power_router_single, area_router_single
from Module_Network.aib_2_5d import area_aib, aib
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../.."))
import pickle
import config
import json
import pandas as pd
import networkx as nx

aimodel=config.aimodel
main_dir = config.main_dir
DEBUG=config.DEBUG
current_dir = os.path.dirname(__file__)
target_file_path = os.path.join(current_dir, 'Network_configs')
parent_dir = os.path.dirname(current_dir)

def calc_edge_charc(source_tile_idx, dest_tile_idx,G_sys, G_chip, nw_df, mesh_size,nop_json_data):
    source_chip_idx=source_tile_idx.split("_")[0]
    dest_chip_idx=dest_tile_idx.split("_")[0]
    hops2d,hops3d=0, 0
    mesh_size_s, mesh_size_d=mesh_size[source_chip_idx],mesh_size[dest_chip_idx]
    nop_router_locs_s={"North": (mesh_size_s//2-1, mesh_size_s,0), "South": (mesh_size_s//2-1, -1,0), "East": (mesh_size_s, mesh_size_s//2-1,0), "West": (-1, mesh_size_s//2-1,0)}
    nop_router_locs_d={"North": (mesh_size_d//2-1, mesh_size_d,0), "South": (mesh_size_d//2-1, -1,0), "East": (mesh_size_d, mesh_size_d//2-1,0), "West": (-1, mesh_size_d//2-1,0)}
    x_noc_s, y_noc_s = map(int, G_chip.nodes[source_tile_idx]["NoC Position"].split(','))
    x_noc_d, y_noc_d= map(int, G_chip.nodes[dest_tile_idx]["NoC Position"].split(','))
    x_nop_s, y_nop_s,z_nop_s = map(int, G_sys.nodes[source_chip_idx]["NoP Position"].split(','))
    x_nop_d, y_nop_d,z_nop_d= map(int, G_sys.nodes[dest_chip_idx]["NoP Position"].split(','))
    nop_location_s, nop_location_d= "NA", "NA"
    if (x_nop_s, y_nop_s) == (x_nop_d, y_nop_d):
        connection_type="2d" if z_nop_d==z_nop_s else "3d"
        hops2d= abs(y_noc_d-y_noc_s)+abs(x_noc_d-x_noc_s)
        hops3d= abs(z_nop_d-z_nop_s)
        hops2_5d=0
        bus_width=nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[source_chip_idx]["Stack ID"], "N_2D_Links_per_tile"].iloc[0] if hops2d!=0 else nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[source_chip_idx]["Stack ID"], "N_3D_Links_per_tile"].iloc[0]
    else:
        connection_type="2_5d" if z_nop_d==z_nop_s==0 else "3_5d"
        hops2_5d=abs(y_nop_d-y_nop_s)+abs(x_nop_d-x_nop_s)
        if y_nop_d>y_nop_s:
            nop_location_s="North"
            nop_location_d="South"
        elif y_nop_d<y_nop_s:
            nop_location_s="South"
            nop_location_d="North"
        elif x_nop_d>x_nop_s:
            nop_location_s="East"
            nop_location_d="West"
        else:
            nop_location_s="West"
            nop_location_d="East"
        x_nop_r_s, y_nop_r_s, _ = nop_router_locs_s[nop_location_s]
        x_nop_r_d, y_nop_r_d, _ = nop_router_locs_d[nop_location_d]
        hops2d = abs(y_nop_r_s-y_noc_s)+abs(x_nop_r_s-x_noc_s)
        hops2d+= abs(y_nop_r_d-y_noc_d)+abs(x_nop_r_d-x_noc_d)
        hops2d+=hops2_5d-1                  # To account for the hops required to communicate between two NoP routers within the intermediate chiplet when travelling from chiplet a to chiplet b
        #import pdb; pdb.set_trace()
        hops3d=abs(z_nop_d-0)+abs(z_nop_s-0) 
        bus_width=min(nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[source_chip_idx]["Stack ID"], "N_2D_Links_per_tile"].iloc[0], nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[dest_chip_idx]["Stack ID"], "N_2D_Links_per_tile"].iloc[0])
        no_2_5d_channels=min(nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[source_chip_idx]["Stack ID"], "N_2.5D_channels_per_chiplet"].iloc[0], nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[dest_chip_idx]["Stack ID"], "N_2.5D_channels_per_chiplet"].iloc[0])
        w_2_5d=no_2_5d_channels*(nop_json_data["n_Tx_config"]+nop_json_data["n_Rx_config"])
        bus_width=min(bus_width, w_2_5d)
    return hops2d,hops2_5d, hops3d, connection_type, bus_width, nop_location_s, nop_location_d


def network_map(G_chip, G_sys, G_ai_model, G_stack, nw_df, tile_map, mem_req, mesh_size, nop_json_data):
    #import pdb; pdb.set_trace()
    #G_network = nx.DiGraph()  
    for edge in G_ai_model.edges():
        source_layer_idx=edge[0]
        dest_layer_idx=edge[1]
        dest_tile_idx=tile_map[dest_layer_idx+"_C"]
        if source_layer_idx!="In":
            source_tile_idx, Q=tile_map[source_layer_idx+"_O"], mem_req[source_layer_idx+"_O"]
            #nodes=[(source_tile_idx,{"HW Type": source_layer_idx+" Out Mem"}), (dest_tile_idx,{"HW Type":dest_layer_idx+" "+G_chip.nodes[dest_tile_idx]["HW Type"]})]
            values = [G_ai_model.nodes[source_layer_idx].get(f"outdim{i}", 1) for i in range(1, 5)]
            G_chip.nodes[source_tile_idx]["HW Type"]=source_layer_idx+" Out Mem"
            if dest_layer_idx not in  G_chip.nodes[dest_tile_idx]["HW Type"]:
                G_chip.nodes[dest_tile_idx]["HW Type"]=dest_layer_idx+" "+G_chip.nodes[dest_tile_idx]["HW Type"]
        else:
            source_tile_idx, Q=tile_map["L1_I"], mem_req["L1_I"]
            #nodes=[(source_tile_idx,{"HW Type": "Input Mem"})]
            G_chip.nodes[source_tile_idx]["HW Type"]="Input Mem"
            #G_network.add_nodes_from(nodes)
        hops2d,hops2_5d, hops3d, connection_type, bus_width, nop_location_s, nop_location_d=calc_edge_charc(source_tile_idx, dest_tile_idx,G_sys, G_chip, nw_df, mesh_size,nop_json_data)
        G_chip.add_edge(source_tile_idx, dest_tile_idx, connection_type=connection_type, bus_width=bus_width, Q=Q, hops2d=hops2d,hops2_5d=hops2_5d, hops3d=hops3d, nop_location_s=nop_location_s, nop_location_d=nop_location_d)
        if G_chip.nodes[dest_tile_idx].get(f"Comment", 1)=="Matmul" and dest_layer_idx+"_W" in tile_map:
            source_tile_idx, Q=tile_map[dest_layer_idx+"_W"], mem_req[dest_layer_idx+"_W"]
            hops2d, hops2_5d, hops3d, connection_type, bus_width, nop_location_s, nop_location_d=calc_edge_charc(source_tile_idx, dest_tile_idx,G_sys, G_chip, nw_df, mesh_size,nop_json_data)
            #nodes=[(source_tile_idx,{"HW Type": dest_layer_idx+" Wg Mem"})]
            #G_network.add_nodes_from(nodes)
            G_chip.nodes[source_tile_idx]["HW Type"]=dest_layer_idx+" Wg Mem"
            G_chip.add_edge(source_tile_idx, dest_tile_idx, connection_type=connection_type, bus_width=bus_width, Q=Q, hops2d=hops2d, hops2_5d=hops2_5d, hops3d=hops3d, nop_location_s=nop_location_s, nop_location_d=nop_location_d)

    
    for layer_idx in G_ai_model.nodes():
        if layer_idx!="In":
            source_tile_idx=tile_map[layer_idx+"_C"]
            dest_tile_idx, Q=tile_map[layer_idx+"_O"],mem_req[layer_idx+"_O"]
            #nodes=[(source_tile_idx,{"HW Type": layer_idx+" "+G_chip.nodes[source_tile_idx]["HW Type"]}), (dest_tile_idx,{"HW Type":layer_idx+" Out Mem"})]
            #G_network.add_nodes_from(nodes)
            hops2d,hops2_5d, hops3d, connection_type, bus_width, nop_location_s, nop_location_d=calc_edge_charc(source_tile_idx, dest_tile_idx,G_sys, G_chip, nw_df, mesh_size, nop_json_data)
            if layer_idx not in  G_chip.nodes[source_tile_idx]["HW Type"]:
                G_chip.nodes[source_tile_idx]["HW Type"]= layer_idx+" "+G_chip.nodes[source_tile_idx]["HW Type"]
            G_chip.nodes[dest_tile_idx]["HW Type"]= layer_idx+" Out Mem"
            G_chip.add_edge(source_tile_idx, dest_tile_idx, connection_type=connection_type, bus_width=bus_width, Q=Q, hops2d=hops2d, hops2_5d=hops2_5d, hops3d=hops3d, nop_location_s=nop_location_s, nop_location_d=nop_location_d)
    #import pdb; pdb.set_trace()
    attrs = {
            "router Area": 0,
            "router Energy": 0,
            "router Latency": 0,
            "router Power": 0,
            "3D link Area": 0,
            "3D link Latency": 0,
            "3D link Energy": 0,
            "3D link Power": 0,
        }

    nx.set_node_attributes(
        G_chip,
        {node: attrs for node in G_chip.nodes()}
    )
    attrs = {
            '2.5d link Area': 0,
            '2.5d link Latency': 0,
            '2.5d link Energy': 0,
            '2.5d link Power': 0,
        }

    nx.set_node_attributes(
        G_stack,
        {node: attrs for node in G_stack.nodes()}
    )


    if DEBUG:
        pos = nx.nx_agraph.graphviz_layout(G_chip, prog="dot")  

        # Node labels
        labels = {node: node+'\n'+G_chip.nodes[node]["HW Type"] for node in G_chip.nodes()} 

        # Plot
        plt.figure(figsize=(0.5*len(labels), 0.5*len(labels)))
        nx.draw(G_chip, pos, labels=labels,
                node_size=1500, node_color="lightblue",
                font_size=10, font_weight="bold",
                edge_color="gray", arrows=True)

        edge_labels = nx.get_edge_attributes(G_chip, "connection_type")  # pick which attr to show
        nx.draw_networkx_edge_labels(G_chip, pos, edge_labels=edge_labels, font_size=8)
        plt.title(f"{aimodel} Device Connectivity Graph", fontsize=14)
        plt.savefig(f"{main_dir}/Results/network_graph.png", dpi=300, bbox_inches="tight")

        edge_data = []
        for u, v, attrs in G_chip.edges(data=True):
            row = {"Source": u, "Target": v, **attrs}
            edge_data.append(row)

        # Convert to DataFrame
        edge_df = pd.DataFrame(edge_data)

        # Pretty print
        table_str = tabulate(edge_df, headers="keys", tablefmt="pretty", showindex=False)
        with open(f"{target_file_path}/network_table.txt", "w") as f:
            f.write(table_str)
    #import pdb; pdb.set_trace()
    return G_chip

def determine_network_area(network_json_data, G_chip, G_sys, G_stack, nw_df, stack_ids, nop_json_data):
    #area
    beachhead={} #stored in mm
    link_feasibility={}
    router_area={}
    metal_layer_pitch_nm= network_json_data["noc_metal_layer_pitch_nm"]
    link_3d_Pitch_um=network_json_data["link_3d_Pitch"] 
    area={chiplet_idx:0 for chiplet_idx in G_sys.nodes()}
    for chip_tile_idx, node in G_chip.nodes(data=True):
        #print(tile)
        chiplet_idx=chip_tile_idx.split("_")[0]
        beachhead[chip_tile_idx]={}
        edge_tile=math.sqrt(node.get("Tile_Area", 0)) #in mm
        beachhead[chip_tile_idx]["2D"]=edge_tile # in mm
        beachhead[chip_tile_idx]["3D"]=node.get("Tile_Area", 0) #in mm^2
        #import pdb; pdb.set_trace()
        no_tiers=len(stack_ids[node.get("Stack ID", 1)])
        no_2d_links = nw_df[nw_df["Stack ID"] == node.get("Stack ID", 1)]["N_2D_Links_per_tile"].values[0]
        if no_tiers==1:
            link_feasibility[chip_tile_idx] = beachhead[chip_tile_idx]["2D"]>no_2d_links*metal_layer_pitch_nm*1e-6
            if not link_feasibility[chip_tile_idx]:
                print(f"Error: Tile {chip_tile_idx} does not have sufficient area for the required number of 2D links. Please increase the tile area or reduce the number of 2D links. Number of maximum 2D links with current tile area is {beachhead[chip_tile_idx]['2D']//(metal_layer_pitch_nm*1e-6)}")
                sys.exit(1)
            router_area[chip_tile_idx]=area_router_single(no_2d_links, "2D_NoC") 
            link_3d_area=0
        else:
            no_3d_links = nw_df[nw_df["Stack ID"] == node.get("Stack ID", 1)]["N_3D_Links_per_tile"].values[0]
            link_3d_area=no_3d_links*link_3d_Pitch_um*link_3d_Pitch_um/0.7*1e-6
            link_feasibility[chip_tile_idx]= beachhead[chip_tile_idx]["3D"]>link_3d_area
            if not link_feasibility[chip_tile_idx]:
                print(f"Error: Tile {chip_tile_idx} does not have sufficient area for the required number of 3D links. Please increase the tile area or reduce the number of 3D links. Number of maximum 3D links with current tile area is {beachhead[chip_tile_idx]['3D']//(link_3d_Pitch_um*link_3d_Pitch_um/0.7*1e-6)}")
                sys.exit(1)
            router_area[chip_tile_idx]=area_router_single(no_3d_links+no_2d_links, "TSV")
        if router_area[chip_tile_idx]>0.5*node.get("Tile_Area", 0):
            print(f"Error: Tile {chip_tile_idx} has router area {router_area[chip_tile_idx]:.4f} mm^2 greater than half of the tile area {node.get('Tile_Area', 0):.4f} mm^2. Please increase the tile area or reduce the number of links.")
            sys.exit(1)
        edge_router=math.sqrt(router_area[chip_tile_idx])
        link_3d_area_available= 2*edge_router*edge_tile #in mm^2
        if link_3d_area_available<link_3d_area:
            print(f"Error: Tile {chip_tile_idx} does not have sufficient area for the required number of 3D links due to router size. Please increase the tile area or reduce the number of 3D links. Number of maximum 3D links with current tile area and router size is {link_3d_area_available//(link_3d_Pitch_um*link_3d_Pitch_um/0.7*1e-6)}")
            sys.exit(1)
        node["3D link Area"]= link_3d_area
        node["router Area"]=  router_area[chip_tile_idx]
        area[chiplet_idx]+=(edge_tile+edge_router)**2
    #import pdb; pdb.set_trace()
    beachhead_chiplets={}
    nop_interface="aib" #or "direct_signaling"
    for chiplet_idx in G_sys.nodes():
        if  G_sys.nodes[chiplet_idx].get("Tier ID", 1)== "TR0":
            beachhead_chiplets[chiplet_idx]= math.sqrt(area[chiplet_idx])
            no_channels = nw_df[nw_df["Stack ID"] == G_sys.nodes[chiplet_idx]["Stack ID"]]["N_2.5D_channels_per_chiplet"].values[0]
            if nop_interface == "aib":
                n_Tx=nop_json_data["n_Tx_config"]
                n_Rx=nop_json_data["n_Rx_config"]
                area_nop, BW_nop=area_aib(None, 0, n_Tx, n_Rx, no_channels)
                beachhead_required=BW_nop/nop_json_data["aib_BW_Gbps_mm"]
                #import pdb; pdb.set_trace() 
                if beachhead_chiplets[chiplet_idx]<beachhead_required:
                    print(f"Error: Chiplet {chiplet_idx} does not have sufficient area for the required number of 2.5D links. Please increase the chiplet area or reduce the number of 2.5D channels. Number of maximum 2.5D channels with current chiplet area is  {(beachhead_chiplets[chiplet_idx]*nop_json_data['aib_BW_Gbps_mm'])//((n_Tx+n_Rx)*nop_json_data['aib_ns_fwd_clk_GHz'])}")
                    sys.exit(1)
                #print(BW_nop)
                G_sys.nodes[chiplet_idx]["2.5D link Area"]= area_nop*4
                area[chiplet_idx]+=area_nop*4
        G_sys.nodes[chiplet_idx]["Chiplet Utilized Area"]= area[chiplet_idx]
        
    for stack_id in G_stack.nodes():
        G_stack.nodes[stack_id]["2.5d link Area"]= area_nop
        #import pdb; pdb.set_trace()
        area_stack= max([G_sys.nodes[chiplet_idx]["Chiplet Utilized Area"] for chiplet_idx in stack_ids[stack_id.split("_")[0]]])
        for chiplet_idx in stack_ids[stack_id.split("_")[0]]:
            G_sys.nodes[chiplet_idx]["Chiplet Actual Area"] = area_stack
        
    #import pdb; pdb.set_trace()
    return G_chip,G_sys,G_stack, beachhead, link_feasibility, router_area, beachhead_chiplets

#print keys and values of G_chip, G_sys, G_stack in a pretty table and save in a file
def print_summary(G_chip, G_sys, G_stack):
    chiplet_data = []
    for chiplet_idx, attrs in G_chip.nodes(data=True):
        row = {"Chiplet": chiplet_idx, **attrs}
        chiplet_data.append(row)

    # Convert to DataFrame
    chiplet_df = pd.DataFrame(chiplet_data)

    # Pretty print
    table_str = tabulate(chiplet_df, headers="keys", tablefmt="pretty", showindex=False)
    with open(f"{target_file_path}/chiplet_table.txt", "w") as f:
        f.write(table_str)
    
    sys_data = []
    for sys_idx, attrs in G_sys.nodes(data=True):
        row = {"Chiplet": sys_idx, **attrs}
        sys_data.append(row)

    # Convert to DataFrame
    sys_df = pd.DataFrame(sys_data)

    # Pretty print
    table_str = tabulate(sys_df, headers="keys", tablefmt="pretty", showindex=False)
    with open(f"{target_file_path}/sys_table.txt", "w") as f:
        f.write(table_str)

    stack_data = []
    for stack_id, attrs in G_stack.nodes(data=True):
        row = {"Stack": stack_id, **attrs}
        stack_data.append(row)

    # Convert to DataFrame
    stack_df = pd.DataFrame(stack_data)

    # Pretty print
    table_str = tabulate(stack_df, headers="keys", tablefmt="pretty", showindex=False)
    with open(f"{target_file_path}/stack_table.txt", "w") as f:
        f.write(table_str)

def update_dict(nop_latency, key, value):
    if key in nop_latency:
        nop_latency[key] += value
    else:
        nop_latency[key] = value
    return nop_latency

def determine_network_latency(G_chip, G_stack, network_json_data, nw_df):
    noc_latency_2d={}
    noc_latency_3d={}
    nop_latency={}
    noc_latency_breakdown_2d={}
    noc_latency_breakdown_3d={}
    #(hop*(trc+tva+tsa+tst+tl)+tenq*Q/BW)/f_noc
    for key, value in network_json_data.items():
        globals()[key] = value
    #import pdb; pdb.set_trace()

    for edge in G_chip.edges():
        source_chip_tile_idx, dest_chip_tile_idx=edge
        edge_attrs=G_chip.edges[edge]
        source_chip_node=G_chip.nodes[source_chip_tile_idx]
        connection_type=edge_attrs.get("connection_type", 0)
        hops2d=edge_attrs.get("hops2d", 0)
        Q_2d=edge_attrs.get("Q", 0)
        bus_width_2d=edge_attrs.get("bus_width", 1)
        latency_2d=hops2d*(trc_cycles+tva_cycles+tsa_cycles+tst_cycles+tl_cycles)+tenq_cycles*Q_2d/bus_width_2d
        noc_latency_breakdown_2d[(source_chip_tile_idx, dest_chip_tile_idx)]={"2d latency": latency_2d/f_noc, 
                                "act_fac_channel": (hops2d*(tl_cycles)+Q_2d/bus_width_2d)/ latency_2d if latency_2d!=0 else 0,
                                "act_fac_inputbuffer": tenq_cycles*Q_2d/bus_width_2d/ latency_2d if latency_2d!=0 else 0,
                                "act_fac_switch": hops2d*(trc_cycles+tva_cycles+tsa_cycles)/ latency_2d if latency_2d!=0 else 0,
                                "act_fac_switchctrl": (hops2d*(trc_cycles+tva_cycles+tsa_cycles)+Q_2d/bus_width_2d)/ latency_2d if latency_2d!=0 else 0,
                                "act_fac_output": (hops2d*(tst_cycles)+Q_2d/bus_width_2d)/ latency_2d if latency_2d!=0 else 0,
                                "act_fac_outputclk": (hops2d*(tst_cycles))/ latency_2d if latency_2d!=0 else 0
        }
        latency_2d/=f_noc


        latency_2_5d=0
        latency_3d=0
        if connection_type in ["2_5d", "3_5d"]:
            #import pdb; pdb.set_trace()
            aib_out=aib(edge_attrs.get("Q", 0)*1e-6/8, None, 0, 0.8, n_ch=nw_df[nw_df["Stack ID"] == G_chip.nodes[source_chip_tile_idx]["Stack ID"]]["N_2.5D_channels_per_chiplet"].values[0])
            latency_2_5d=aib_out[2]*1e-9
            #import pdb; pdb.set_trace()
            nop_latency=update_dict(nop_latency, (source_chip_tile_idx.split("_")[0], dest_chip_tile_idx.split("_")[0], connection_type), latency_2_5d)
            nop_position_id_s =source_chip_node["Stack ID"]+"_"+edge_attrs.get("nop_location_s", None)[0]
            #print(G_stack.nodes[nop_position_id_s]["2.5d link Latency"])
            G_stack.nodes[nop_position_id_s]["2.5d link Latency"]+=latency_2_5d
        if connection_type in ["3d", "3_5d"]:
            hops3d=edge_attrs.get("hops3d", 0)
            Q_3d=edge_attrs.get("Q", 0)
            bus_width_3d=edge_attrs.get("bus_width", 1)
            latency_3d=hops3d*(trc_cycles+tva_cycles+tsa_cycles+tst_cycles+tl_cycles)+tenq_cycles*Q_3d/bus_width_3d
            noc_latency_breakdown_3d[(source_chip_tile_idx, dest_chip_tile_idx)]={"3d latency": latency_3d/f_noc,
                                    "act_fac_channel": (hops3d*(tl_cycles)+Q_3d/bus_width_3d)/ latency_3d if latency_3d!=0 else 0,
                                    "act_fac_inputbuffer": tenq_cycles*Q_3d/bus_width_3d/ latency_3d if latency_3d!=0 else 0,
                                    "act_fac_switch": hops3d*(trc_cycles+tva_cycles+tsa_cycles)/ latency_3d if latency_3d!=0 else 0,
                                    "act_fac_switchctrl": (hops3d*(trc_cycles+tva_cycles+tsa_cycles)+Q_3d/bus_width_3d)/ latency_3d if latency_3d!=0 else 0,
                                    "act_fac_output": (hops3d*(tst_cycles)+Q_3d/bus_width_3d)/ latency_3d if latency_3d!=0 else 0,
                                    "act_fac_outputclk": (hops3d*(tst_cycles))/ latency_3d if latency_3d!=0 else 0
            }
            latency_3d/=f_noc

        #import pdb; pdb.set_trace()    
        noc_latency_2d[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)] = latency_2d
        noc_latency_3d[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)] = latency_3d
        source_chip_node["router Latency"]+= latency_2d
        source_chip_node["3D link Latency"]+= latency_3d
       

    return noc_latency_2d, noc_latency_3d, nop_latency, noc_latency_breakdown_2d, noc_latency_breakdown_3d
#print(tile_map)

def determine_network_energy(G_chip, G_stack, network_json_data, nw_df, noc_latency_breakdown_2d, noc_latency_breakdown_3d):
    noc_energy_2d={}
    noc_energy_3d={}
    nop_energy={}
    for edge in G_chip.edges():
        source_chip_tile_idx, dest_chip_tile_idx=edge
        edge_attrs=G_chip.edges[edge]
        source_chip_node=G_chip.nodes[source_chip_tile_idx]
        connection_type=edge_attrs.get("connection_type", 0)
        bus_width_2d=edge_attrs.get("bus_width", 1)
        _, channel_power_2d,router_power_2d =power_router_single(bus_width_2d, "2D_NoC", noc_latency_breakdown_2d[(source_chip_tile_idx, dest_chip_tile_idx)])
        channel_power_2d*=1e-3*f_noc/1e9 # converting to W and scaling to respective frequency
        router_power_2d*=1e-3*f_noc/1e9
        #import pdb; pdb.set_trace()
        energy_2d=(channel_power_2d+router_power_2d)*noc_latency_breakdown_2d[(source_chip_tile_idx, dest_chip_tile_idx)]["2d latency"]

        energy_2_5d=0
        energy_3d=0
        if connection_type in ["2_5d", "3_5d"]:
            aib_out=aib(edge_attrs.get("Q", 0)*1e-6/8, None, 0, 0.8, n_ch=nw_df[nw_df["Stack ID"] == G_chip.nodes[source_chip_tile_idx]["Stack ID"]]["N_2.5D_channels_per_chiplet"].values[0])
            energy_2_5d=aib_out[1]*1e-12 #converting to J
            nop_energy=update_dict(nop_energy, (source_chip_tile_idx.split("_")[0], dest_chip_tile_idx.split("_")[0], connection_type), energy_2_5d)
            nop_position_id_s = source_chip_node["Stack ID"]+"_"+edge_attrs.get("nop_location_s", None)[0]
            G_stack.nodes[nop_position_id_s]["2.5d link Energy"]+=energy_2_5d
        if connection_type in ["3d", "3_5d"]:
            bus_width_3d=edge_attrs.get("bus_width", 1)
            _, channel_power_3d,router_power_3d=power_router_single(bus_width_3d, "TSV", noc_latency_breakdown_3d[(source_chip_tile_idx, dest_chip_tile_idx)])
            channel_power_3d*=1e-3*f_noc/1e9 # converting to W and scaling to respective frequency
            router_power_3d*=1e-3*f_noc/1e9
            energy_3d=(channel_power_3d+router_power_3d)*noc_latency_breakdown_3d[(source_chip_tile_idx, dest_chip_tile_idx)]["3d latency"]
        noc_energy_2d[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)] = energy_2d
        noc_energy_3d[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)] = energy_3d
        source_chip_node["router Energy"]+= energy_2d
        source_chip_node["3D link Energy"]+= energy_3d

    return noc_energy_2d, noc_energy_3d, nop_energy

def network_main_fn(G_ai_model, G_chip, G_sys, G_stack, tile_map, mem_req, mesh_size, stack_ids):
    file_path = current_dir+'/Network.json'
    with open(file_path, 'r') as f:
        network_json_data  = json.load(f)
    nw_df= pd.read_csv(f"{target_file_path}/Network_Spec_{aimodel}.csv")
     
    nop_json_path = os.path.join(parent_dir, 'aib_params.json')
    with open(nop_json_path, 'r') as f:
        nop_json_data  = json.load(f)
    #import pdb; pdb.set_trace()
    G_chip=network_map(G_chip, G_sys, G_ai_model, G_stack, nw_df, tile_map, mem_req, mesh_size, nop_json_data)

    G_chip,G_sys,G_stack, beachhead, link_feasibility, router_area, beachhead_chiplets=determine_network_area(network_json_data, G_chip, G_sys, G_stack, nw_df, stack_ids, nop_json_data)
    
    #latency
    noc_latency_2d, noc_latency_3d,nop_latency, noc_latency_breakdown_2d, noc_latency_breakdown_3d=determine_network_latency(G_chip, G_stack, network_json_data, nw_df)
    
    #import pdb; pdb.set_trace()

    #power function update
    #calculate power
    noc_energy_2d, noc_energy_3d,nop_energy=determine_network_energy(G_chip, G_stack,network_json_data, nw_df, noc_latency_breakdown_2d, noc_latency_breakdown_3d)

    for node in G_chip.nodes():
        G_chip.nodes[node]["router Power"]= G_chip.nodes[node]["router Energy"]/G_chip.nodes[node]["router Latency"] if G_chip.nodes[node]["router Latency"]>0 else 0
        G_chip.nodes[node]["3D link Power"]= G_chip.nodes[node]["3D link Energy"]/G_chip.nodes[node]["3D link Latency"] if G_chip.nodes[node]["3D link Latency"]>0 else 0
    
    for node in G_stack.nodes():
        G_stack.nodes[node]["2.5d link Power"]= G_stack.nodes[node]["2.5d link Energy"]/G_stack.nodes[node]["2.5d link Latency"] if G_stack.nodes[node]["2.5d link Latency"]>0 else 0
    
    print("----------Network Summary---------------")
    print("Total NoC Area (in mm^2)", sum(router_area.values()))
    print("Total NoP Area (in mm^2)", sum([G_stack.nodes[stack_id]["2.5d link Area"] for stack_id in G_stack.nodes()]))
    print("Total 2D latency (in s)", sum(noc_latency_2d.values()))
    print("Total 2D energy (in J)", sum(noc_energy_2d.values()))
    print("Total 2.5D latency (in s)", sum(nop_latency.values()))
    print("Total 2.5D energy (in J)", sum(nop_energy.values()))
    print("Total 3D latency (in s)", sum(noc_latency_3d.values()))
    print("Total 3D energy (in J)", sum(noc_energy_3d.values()))
    print("----------------------------------------")

    print("----------Combined Summary (Compute+Network+Memory)---------------")
    stack_area={G_sys.nodes[chip_idx]["Stack ID"] : G_sys.nodes[chip_idx]["Chiplet Actual Area"] for chip_idx in G_sys.nodes()}
    #import pdb; pdb.set_trace()
    print("Total Chip Area (in mm^2)", sum(stack_area.values()))
    #import pdb; pdb.set_trace()
    print("Total Chip Latency (in s)", sum([G_chip.nodes[node]["Tile_Latency"] for node in G_chip.nodes()])+sum(noc_latency_2d.values())+sum(nop_latency.values())+ sum(noc_latency_3d.values()))
    print("Total Chip Energy (in J)", sum([G_chip.nodes[node]["Tile_Energy"] for node in G_chip.nodes()])+sum(noc_energy_2d.values())+sum(nop_energy.values())+ sum(noc_energy_3d.values()))
    print("----------------------------------------")
    if DEBUG:
        print_summary(G_chip, G_sys, G_stack)

        #import pdb; pdb.set_trace()

        totalhops2d=sum([G_chip.edges[edge]["hops2d"] for edge in G_chip.edges])
        totalhops2_5d=sum([G_chip.edges[edge]["hops2_5d"] for edge in G_chip.edges])
        totalhops3d=sum([G_chip.edges[edge]["hops3d"] for edge in G_chip.edges])
        #BW_3d_list=[G_chip.edges[edge]["BW"] for edge in G_chip.edges if G_chip.edges[edge]["hops3d"]>0]
        #BW_2d_list=[G_chip.edges[edge]["BW"] for edge in G_chip.edges]

        #print("Total 2D hops: ", totalhops2d, " Total 2.5D hops: ", totalhops2_5d, " Total 3D hops: ", totalhops3d)
        #print("Effective BW of communicating tiles - IntraTier: ", BW_2d_list)
        #print("Effective BW of communicating tiles - InterTier: ", BW_3d_list)
             
    outs=(G_chip, G_sys)
    with open('G_chip_G_stack'+'.pkl', 'wb') as file:
        pickle.dump(outs, file)   
    #import pdb; pdb.set_trace()
