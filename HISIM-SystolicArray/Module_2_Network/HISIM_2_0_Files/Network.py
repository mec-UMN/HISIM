import math
import numpy as np
import sys
import os
from tabulate import tabulate
from Module_2_Network.orion_power_area import power_router_single, area_router_single
from Module_2_Network.aib_2_5d import area_aib, aib
import matplotlib.pyplot as plt
import pickle
import config
import json
import pandas as pd
import networkx as nx
from Module_1_Compute.HISIM_2_0_Files.Compute import plot_component

aimodel=config.aimodel
main_dir = config.main_dir
DEBUG=config.DEBUG
current_dir = os.path.dirname(__file__)
target_file_path = os.path.join(current_dir, 'Network_configs')
parent_dir = os.path.dirname(current_dir)

def calc_edge_charc(source_tile_idx, dest_tile_idx,G_sys, G_chip, G_stack, nw_df,nop_json_data):
    source_chip_idx=source_tile_idx.split('_')[0]
    dest_chip_idx=dest_tile_idx.split('_')[0]
    hops2d,hops3d=0, 0
    x_noc_s, y_noc_s = map(int, G_chip.nodes[source_tile_idx]["NoC Position"].split(','))
    x_noc_d, y_noc_d= map(int, G_chip.nodes[dest_tile_idx]["NoC Position"].split(','))
    x_nop_s, y_nop_s,z_nop_s = map(int, G_sys.nodes[source_chip_idx]["NoP Position"].split(','))
    x_nop_d, y_nop_d,z_nop_d= map(int, G_sys.nodes[dest_chip_idx]["NoP Position"].split(','))
    nop_location_s, nop_location_d= "NA", "NA"
    source_stack_id=G_sys.nodes[source_chip_idx]["Stack ID"]
    dest_stack_id=G_sys.nodes[dest_chip_idx]["Stack ID"]
    if (x_nop_s, y_nop_s) == (x_nop_d, y_nop_d):
        if z_nop_d==z_nop_s:
            connection_type="2d"
            router_list_2d={source_stack_id: [(x,y_noc_s, z_nop_s) for x in range(min(x_noc_s,x_noc_d), max(x_noc_s,x_noc_d)+1)]+ [(x_noc_d,y,z_nop_s) for y in range(min(y_noc_s,y_noc_d), max(y_noc_s,y_noc_d)+1) if y!=y_noc_s]} #if condition to avoid duplication
            router_list_3d={}
        else:
            connection_type="3d"
            if (x_noc_d, y_noc_d) == (x_noc_s, y_noc_s):
                router_list_2d={}
            else:
                router_list_2d={source_stack_id: [(x,y_noc_s, z_nop_s) for x in range(min(x_noc_s,x_noc_d), max(x_noc_s,x_noc_d)+1)]+ [(x_noc_d,y,z_nop_s) for y in range(min(y_noc_s,y_noc_d), max(y_noc_s,y_noc_d)+1) if y!=y_noc_s and y!=y_noc_d]}
            router_list_3d={source_stack_id: [(x_noc_d,y_noc_d, z) for z in range(min(z_nop_s,z_nop_d), max(z_nop_s,z_nop_d)+1)]}
        hops2d= abs(y_noc_d-y_noc_s)+abs(x_noc_d-x_noc_s)
        hops3d= abs(z_nop_d-z_nop_s)
        bus_width=nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[source_chip_idx]["Stack ID"], "N_2D_Links_per_tile"].iloc[0] if hops2d!=0 else nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[source_chip_idx]["Stack ID"], "N_3D_Links_per_tile"].iloc[0]
        hops2_5d=0
        no_2_5d_channels_active=0
        router_list_2_5d={}
    else:
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
        x_noc_nop_r_s, y_noc_nop_r_s, _ = G_stack.nodes[source_stack_id+"_"+nop_location_s[0]]["nop_noc_router_location"]
        x_noc_nop_r_d, y_noc_nop_r_d, _ = G_stack.nodes[dest_stack_id+"_"+nop_location_d[0]]["nop_noc_router_location"]
        #import pdb; pdb.set_trace()
        router_list_2d={source_stack_id: [(x,y_noc_s, z_nop_s) for x in range(min(x_noc_s,x_noc_nop_r_s), max(x_noc_s,x_noc_nop_r_s)+1)]+[(x_noc_nop_r_s,y,z_nop_s) for y in range(min(y_noc_s,y_noc_nop_r_s), max(y_noc_s,y_noc_nop_r_s)+1) if y!=y_noc_s]}
        router_list_2d.update({dest_stack_id: [(x,y_noc_nop_r_d, z_nop_d) for x in range(min(x_noc_nop_r_d,x_noc_d), max(x_noc_nop_r_d,x_noc_d)+1)]+[(x_noc_d,y,z_nop_d) for y in range(min(y_noc_nop_r_d,y_noc_d), max(y_noc_nop_r_d,y_noc_d)+1) if y!=y_noc_nop_r_d]})
        router_list_2_5d=[(x,y_nop_s,0) for x in range(min(x_nop_s, x_nop_d), max(x_nop_s, x_nop_d)+1)]+[(x_nop_d,y,0) for y in range(min(y_nop_s, y_nop_d), max(y_nop_s, y_nop_d)+1) if y!=y_nop_s]
        if z_nop_d==z_nop_s==0:
            connection_type="2_5d"
            router_list_3d={}
        else:
            connection_type="3_5d"
            router_list_3d={source_stack_id: [(x_noc_nop_r_s,y_noc_nop_r_s, z) for z in range(0, z_nop_s+1) if z!=z_nop_s]}
            router_list_3d.update({dest_stack_id: [(x_noc_nop_r_d,y_noc_nop_r_d, z) for z in range(0, z_nop_d+1) if z!=z_nop_d]})
        hops2d = abs(y_noc_nop_r_s-y_noc_s)+abs(x_noc_nop_r_s-x_noc_s)
        hops2d+= abs(y_noc_nop_r_d-y_noc_d)+abs(x_noc_nop_r_d-x_noc_d)
        hops2_5d=abs(y_nop_d-y_nop_s)+abs(x_nop_d-x_nop_s)
        hops2d+=hops2_5d-1                  # To account for the hops required to communicate between two NoP routers within the intermediate chiplet when travelling from chiplet a to chiplet b
        #import pdb; pdb.set_trace()
        hops3d=abs(z_nop_d-0)+abs(z_nop_s-0) 
        bus_width=min(nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[source_chip_idx]["Stack ID"], "N_2D_Links_per_tile"].iloc[0], nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[dest_chip_idx]["Stack ID"], "N_2D_Links_per_tile"].iloc[0])
        no_2_5d_channels=min(nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[source_chip_idx]["Stack ID"], "N_2.5D_channels_per_chiplet_edge"].iloc[0], nw_df.loc[nw_df["Stack ID"] == G_sys.nodes[dest_chip_idx]["Stack ID"], "N_2.5D_channels_per_chiplet_edge"].iloc[0])
        w_2_5d=no_2_5d_channels*(nop_json_data["n_Tx_config"]+nop_json_data["n_Rx_config"])
        bus_width=min(bus_width, w_2_5d)
        no_2_5d_channels_active=math.ceil(bus_width/(nop_json_data["n_Tx_config"]+nop_json_data["n_Rx_config"]))

    return hops2d,hops2_5d, hops3d, connection_type, bus_width, no_2_5d_channels_active, nop_location_s, nop_location_d, router_list_2d, router_list_3d, router_list_2_5d


def network_map(G_chip, G_sys, G_ai_model, G_stack, noc_tile_dict, nop_chip_dict, nw_df, tile_map, mem_req, mesh_size, mesh_size_nop,nop_json_data):
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
        hops2d,hops2_5d, hops3d, connection_type, bus_width, no_2_5d_channels_active, nop_location_s, nop_location_d, router_list_2d, router_list_3d, router_list_2_5d=calc_edge_charc(source_tile_idx, dest_tile_idx,G_sys, G_chip, G_stack, nw_df,nop_json_data)
        G_chip.add_edge(source_tile_idx, dest_tile_idx, connection_type=connection_type, bus_width=bus_width, Q=Q, hops2d=hops2d,hops2_5d=hops2_5d, hops3d=hops3d, nop_location_s=nop_location_s, nop_location_d=nop_location_d, router_list_2d=router_list_2d, router_list_3d=router_list_3d, nop_router_list=router_list_2_5d)
        if G_chip.nodes[dest_tile_idx].get(f"Comment", 1)=="Matmul" and dest_layer_idx+"_W" in tile_map:
            source_tile_idx, Q=tile_map[dest_layer_idx+"_W"], mem_req[dest_layer_idx+"_W"]
            hops2d, hops2_5d, hops3d, connection_type, bus_width, no_2_5d_channels_active, nop_location_s, nop_location_d, router_list_2d, router_list_3d, router_list_2_5d=calc_edge_charc(source_tile_idx, dest_tile_idx,G_sys, G_chip, G_stack, nw_df,nop_json_data)
            #nodes=[(source_tile_idx,{"HW Type": dest_layer_idx+" Wg Mem"})]
            #G_network.add_nodes_from(nodes)
            G_chip.nodes[source_tile_idx]["HW Type"]=dest_layer_idx+" Wg Mem"
            G_chip.add_edge(source_tile_idx, dest_tile_idx, connection_type=connection_type, bus_width=bus_width, Q=Q, hops2d=hops2d, hops2_5d=hops2_5d, hops3d=hops3d, nop_location_s=nop_location_s, nop_location_d=nop_location_d, router_list_2d=router_list_2d, router_list_3d=router_list_3d, nop_router_list=router_list_2_5d)

    
    for layer_idx in G_ai_model.nodes():
        if layer_idx!="In":
            source_tile_idx=tile_map[layer_idx+"_C"]
            dest_tile_idx, Q=tile_map[layer_idx+"_O"],mem_req[layer_idx+"_O"]
            #nodes=[(source_tile_idx,{"HW Type": layer_idx+" "+G_chip.nodes[source_tile_idx]["HW Type"]}), (dest_tile_idx,{"HW Type":layer_idx+" Out Mem"})]
            #G_network.add_nodes_from(nodes)
            hops2d,hops2_5d, hops3d, connection_type, bus_width, no_2_5d_channels_active, nop_location_s, nop_location_d, router_list_2d, router_list_3d, router_list_2_5d=calc_edge_charc(source_tile_idx, dest_tile_idx,G_sys, G_chip, G_stack, nw_df, nop_json_data)
            if layer_idx not in  G_chip.nodes[source_tile_idx]["HW Type"]:
                G_chip.nodes[source_tile_idx]["HW Type"]= layer_idx+" "+G_chip.nodes[source_tile_idx]["HW Type"]
            G_chip.nodes[dest_tile_idx]["HW Type"]= layer_idx+" Out Mem"
            G_chip.add_edge(source_tile_idx, dest_tile_idx, connection_type=connection_type, bus_width=bus_width, Q=Q, hops2d=hops2d, hops2_5d=hops2_5d, hops3d=hops3d, nop_location_s=nop_location_s, nop_location_d=nop_location_d, router_list_2d=router_list_2d, router_list_3d=router_list_3d, nop_router_list=router_list_2_5d)
    

    #add nodes for DDR mem tile to mem tile connections
    ddr_chip_tile_idx=tile_map["DDR_Mem"]
    for chip_tile_idx in G_chip.nodes():
        node_data=G_chip.nodes[chip_tile_idx]
        if "DDR_Latency" in node_data:
            if node_data["DDR_Latency"]!=0:
                source_tile_idx=ddr_chip_tile_idx
                dest_tile_idx=chip_tile_idx
                Q=mem_req[ddr_chip_tile_idx][chip_tile_idx]
                #nodes=[(source_tile_idx,{"HW Type": "DDR Mem"}), (dest_tile_idx,{"HW Type":G_chip.nodes[dest_tile_idx]["HW Type"]})]
                #G_network.add_nodes_from(nodes)
                hops2d,hops2_5d, hops3d, connection_type, bus_width, no_2_5d_channels_active, nop_location_s, nop_location_d, router_list_2d, router_list_3d, router_list_2_5d=calc_edge_charc(source_tile_idx, dest_tile_idx,G_sys, G_chip, G_stack, nw_df,nop_json_data)
                #add bidirectional edge
                G_chip.add_edge(source_tile_idx, dest_tile_idx, connection_type=connection_type, bus_width=bus_width, Q=Q, hops2d=hops2d,hops2_5d=hops2_5d, hops3d=hops3d, nop_location_s=nop_location_s, nop_location_d=nop_location_d, router_list_2d=router_list_2d, router_list_3d=router_list_3d, nop_router_list=router_list_2_5d)
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
        os.makedirs(f"{main_dir}/Results", exist_ok=True)
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

    #add empty tiles to G_chip
    for chiplet_idx in G_sys.nodes():
        stack_id=G_sys.nodes[chiplet_idx]["Stack ID"]
        _, _, z_nop = map(int, G_sys.nodes[chiplet_idx]["NoP Position"].split(','))
        #Assign empty tiles to unassigned NoC positions in the mesh. these positions also include -1 and mesh_size+1 positions to account for placement of NoP interface
        for i in range(-1, mesh_size[chiplet_idx]+1):
            for j in range(-1, mesh_size[chiplet_idx]+1):
                noc_position= (i,j,z_nop)
                if noc_position not in noc_tile_dict[stack_id].keys():
                    #import pdb; pdb.set_trace()
                    tile_id= f"{chiplet_idx}_x{i}_y{j}"
                    noc_pos_str = f"{i},{j}"
                    G_chip.add_node(
                                    tile_id,
                                    **{
                                        "Chiplet ID": tile_id,
                                        "Tile ID": tile_id,
                                        "HW Type": "Empty",
                                        "NoC Position": noc_pos_str,
                                        "Stack ID": stack_id,
                                        "NoP Position": G_sys.nodes[chiplet_idx]["NoP Position"],
                                        "Comment": "Empty",
                                        "AI Layer": "Empty",
                                        "Tile_Area": 0,
                                        "Tile_Power": 0,
                                        "Tile_Energy": 0,
                                        "DDR_Energy": 0,
                                        "Tile_Latency": 0,
                                        "DDR_Latency": 0,
                                    }
                                )

                    noc_tile_dict[stack_id][noc_position]=tile_id
                    #import pdb; pdb.set_trace()
    for i in range(0, mesh_size_nop):
        for j in range(0, mesh_size_nop):
            nop_position= (i,j,0)
            if nop_position not in nop_chip_dict.keys():
                stack_id=f"S_{i}_{j}"
                G_sys.add_node(stack_id, **{"Chiplet ID": "Empty","Stack ID": stack_id, "NoP Position": f"{i},{j},0", "Tier ID": "TR0", "2.5D link Area": 0, "Chiplet Utilized Area":0, "Chiplet Actual Area":0, "NoP Router Area":0, "NoP Router Latency":0, "NoP Router Energy":0})
                nop_chip_dict[nop_position]=stack_id

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

    #import pdb; pdb.set_trace()
    return G_chip, noc_tile_dict, G_stack, nop_chip_dict, G_sys

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
        chiplet_idx=chip_tile_idx.split('_')[0]
        beachhead[chip_tile_idx]={}
        edge_tile=math.sqrt(node.get("Tile_Area", 0)) #in mm
        beachhead[chip_tile_idx]["2D"]=edge_tile # in mm
        beachhead[chip_tile_idx]["3D"]=node.get("Tile_Area", 0) #in mm^2
        #import pdb; pdb.set_trace()
        no_tiers=len(stack_ids[node.get("Stack ID", 1)])
        no_2d_links = nw_df[nw_df["Stack ID"] == node.get("Stack ID", 1)]["N_2D_Links_per_tile"].values[0]
        if no_tiers==1:
            link_feasibility[chip_tile_idx] = beachhead[chip_tile_idx]["2D"]>no_2d_links*metal_layer_pitch_nm*1e-6
            if not link_feasibility[chip_tile_idx] and node["HW Type"]!="Empty":
                print(f"Error: Tile {chip_tile_idx} does not have sufficient area for the required number of 2D links. Please increase the tile area or reduce the number of 2D links. Number of maximum 2D links with current tile area is {beachhead[chip_tile_idx]['2D']//(metal_layer_pitch_nm*1e-6)}")
                sys.exit(1)
            router_area[chip_tile_idx]=area_router_single(no_2d_links, "2D_NoC") 
            link_3d_area=0
        else:
            no_3d_links = nw_df[nw_df["Stack ID"] == node.get("Stack ID", 1)]["N_3D_Links_per_tile"].values[0]
            link_3d_area=no_3d_links*link_3d_Pitch_um*link_3d_Pitch_um/0.7*1e-6
            link_feasibility[chip_tile_idx]= beachhead[chip_tile_idx]["3D"]>link_3d_area
            if not link_feasibility[chip_tile_idx] and node["HW Type"]!="Empty":
                print(f"Error: Tile {chip_tile_idx} does not have sufficient area for the required number of 3D links. Please increase the tile area or reduce the number of 3D links. Number of maximum 3D links with current tile area is {beachhead[chip_tile_idx]['3D']//(link_3d_Pitch_um*link_3d_Pitch_um/0.7*1e-6)}")
                sys.exit(1)
            router_area[chip_tile_idx]=area_router_single(min(no_3d_links,no_2d_links), "TSV")
        #import pdb; pdb.set_trace()
        if router_area[chip_tile_idx]>0.5*node.get("Tile_Area", 0) and node["HW Type"]!="Empty":
            print(f"Error: Tile {chip_tile_idx} has router area {router_area[chip_tile_idx]:.4f} mm^2 greater than half of the tile area {node.get('Tile_Area', 0):.4f} mm^2. Please increase the tile area or reduce the number of links.")
            sys.exit(1)
        edge_router=math.sqrt(router_area[chip_tile_idx])
        link_3d_area_available= 2*edge_router*edge_tile #in mm^2
        if link_3d_area_available<link_3d_area:
            if node["HW Type"]!="Empty":
                print(f"Error: Required number of 3D links from tile {chip_tile_idx} does not have sufficient area due to router size. Please increase the tile area or reduce the number of 3D links. Number of maximum 3D links with current tile area and router size is {link_3d_area_available//(link_3d_Pitch_um*link_3d_Pitch_um/0.7*1e-6)}")
                sys.exit(1)
            else:
                if link_3d_area>2*router_area[chip_tile_idx]:   
                    print(f"Error: Area of 3D links from empty tile {chip_tile_idx} greater than twice of the router area. Please reduce the number of 3D links.")
                    sys.exit(1)
        node["3D link Area"]= link_3d_area
        node["router Area"]=  router_area[chip_tile_idx]
        #area[chiplet_idx]+=(edge_tile+edge_router)**2
        area[chiplet_idx]+=(edge_tile**2)+router_area[chip_tile_idx]+link_3d_area
    #import pdb; pdb.set_trace()
    n_signal_IO={}
    beachhead_chiplets={}
    nop_channels=24
    nop_interface="aib" #or "direct_signaling"
    for chiplet_idx in G_sys.nodes():
        if  G_sys.nodes[chiplet_idx].get("Tier ID", 1)== "TR0" and G_sys.nodes[chiplet_idx].get("Chiplet ID", 1)!="Empty":
            beachhead_chiplets[chiplet_idx]= math.sqrt(area[chiplet_idx])
            no_channels = nw_df[nw_df["Stack ID"] == G_sys.nodes[chiplet_idx]["Stack ID"]]["N_2.5D_channels_per_chiplet_edge"].values[0]
            nop_channels = min(no_channels, nop_channels)
            if nop_interface == "aib":
                n_Tx=nop_json_data["n_Tx_config"]
                n_Rx=nop_json_data["n_Rx_config"]
                n_IO=nop_json_data["n_IO"]*(no_channels)
                area_nop, BW_nop=area_aib(None, 0, n_Tx, n_Rx, no_channels)
                beachhead_required=BW_nop/nop_json_data["aib_BW_Gbps_mm"]
                #import pdb; pdb.set_trace() 
                if beachhead_chiplets[chiplet_idx]<beachhead_required:
                    print(f"Error: Chiplet {chiplet_idx} does not have sufficient area for the required number of 2.5D links. Please increase the chiplet area or reduce the number of 2.5D channels. Number of maximum 2.5D channels with current chiplet area is  {(beachhead_chiplets[chiplet_idx]*nop_json_data['aib_BW_Gbps_mm'])//((n_Tx+n_Rx)*nop_json_data['aib_ns_fwd_clk_GHz'])}")
                    sys.exit(1)
                #print(BW_nop)
                G_sys.nodes[chiplet_idx]["2.5D link Area"]= area_nop*4
                area[chiplet_idx]+=area_nop*4
                n_signal_IO[G_sys.nodes[chiplet_idx]["Stack ID"]]=n_IO*4
                nop_width=no_channels*(nop_json_data["n_IO"])
        else:
            G_sys.nodes[chiplet_idx]["2.5D link Area"]= 0
            G_sys.nodes[chiplet_idx]["NoP Router Area"] = 0
        G_sys.nodes[chiplet_idx]["Chiplet Utilized Area"]= area[chiplet_idx]
    
    
    for stack_id in G_stack.nodes():
        G_stack.nodes[stack_id]["2.5d link Area"]= area_nop
        #import pdb; pdb.set_trace()
        area_stack= max([G_sys.nodes[chiplet_idx]["Chiplet Utilized Area"] for chiplet_idx in stack_ids[stack_id.split('_')[0]]])
        for chiplet_idx in stack_ids[stack_id.split('_')[0]]:
            G_sys.nodes[chiplet_idx]["Chiplet Actual Area"] = area_stack

    nop_width=nop_channels*(nop_json_data["n_IO"])
    nop_router_area= area_router_single(nop_width, "2D_NoC")
    for chiplet_idx in G_sys.nodes():
        G_sys.nodes[chiplet_idx]["NoP Router Area"] = nop_router_area if G_sys.nodes[chiplet_idx].get("Tier ID", 1)== "TR0" else 0
                    
    #import pdb; pdb.set_trace()
    return G_chip,G_sys,G_stack, beachhead, link_feasibility, router_area, beachhead_chiplets, n_signal_IO

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
        row = {**attrs}
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

def determine_network_latency(G_chip, G_stack, G_sys, noc_tile_dict, nop_chip_dict, network_json_data, nw_df):
    noc_latency_2d, noc_latency_3d, nop_latency = {}, {}, {}
    noc_latency_breakdown_2d, noc_latency_breakdown_3d, nop_latency_breakdown = {}, {}, {}
    #(hop*(trc+tva+tsa+tst+tl)+tenq*Q/BW)/f_noc
    for key, value in network_json_data.items():
        globals()[key] = value
    #import pdb; pdb.set_trace()

    for edge in G_chip.edges():
        source_chip_tile_idx, dest_chip_tile_idx=edge
        edge_attrs=G_chip.edges[edge]
        source_chip_node=G_chip.nodes[source_chip_tile_idx]
        dest_chip_node=G_chip.nodes[dest_chip_tile_idx]
        connection_type=edge_attrs.get("connection_type", 0)
        hops2d=edge_attrs.get("hops2d", 0)
        Q_2d=edge_attrs.get("Q", 0)
        bus_width_2d=edge_attrs.get("bus_width", 1)
        latency_2d=hops2d*(trc_cycles_noc+tva_cycles_noc+tsa_cycles_noc+tst_cycles_noc+tl_cycles_noc)+tenq_cycles_noc*Q_2d/bus_width_2d
        noc_latency_breakdown_2d[(source_chip_tile_idx, dest_chip_tile_idx)]={"2d latency": latency_2d/f_noc, 
                                "act_fac_channel": (hops2d*(tl_cycles_noc)+Q_2d/bus_width_2d)/ latency_2d if latency_2d!=0 else 0,
                                "act_fac_inputbuffer": tenq_cycles_noc*Q_2d/bus_width_2d/ latency_2d if latency_2d!=0 else 0,
                                "act_fac_switch": hops2d*(trc_cycles_noc+tva_cycles_noc+tsa_cycles_noc)/ latency_2d if latency_2d!=0 else 0,
                                "act_fac_switchctrl": (hops2d*(trc_cycles_noc+tva_cycles_noc+tsa_cycles_noc)+Q_2d/bus_width_2d)/ latency_2d if latency_2d!=0 else 0,
                                "act_fac_output": (hops2d*(tst_cycles_noc)+Q_2d/bus_width_2d)/ latency_2d if latency_2d!=0 else 0,
                                "act_fac_outputclk": (hops2d*(tst_cycles_noc))/ latency_2d if latency_2d!=0 else 0
        }
        latency_2d/=f_noc
        #import pdb; pdb.set_trace()
        noc_latency_2d[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)] = latency_2d
        #print(edge_attrs.get("router_list_2d", []).items())
        for router_node_stack_idx, noc_positions in edge_attrs.get("router_list_2d", []).items():
            #import pdb; pdb.set_trace()
            for noc_position in noc_positions:
                chip_tile_idx=noc_tile_dict[router_node_stack_idx][noc_position]
                router_node = G_chip.nodes[chip_tile_idx]
                router_node["router Latency"] += latency_2d
                #import pdb; pdb.set_trace()

        latency_2_5d=0
        latency_3d=0
        if connection_type in ["2_5d", "3_5d"]:
            hops2_5d=edge_attrs.get("hops2_5d", 0)
            Q_2_5d=edge_attrs.get("Q", 0)
            bus_width_2_5d=edge_attrs.get("bus_width", 1)
            #import pdb; pdb.set_trace()
            #source NoP interface
            no_2_5d_channels_active=edge_attrs.get("no_2_5d_channels_active", 1)
            aib_out=aib(Q_2_5d*1e-6/8, None, 0, 0.8, n_ch=no_2_5d_channels_active)
            nop_interface_latency_2_5d=aib_out[2]*1e-9
            nop_position_id_s = source_chip_node["Stack ID"]+"_"+edge_attrs.get("nop_location_s", None)[0]
            G_stack.nodes[nop_position_id_s]["2.5d link Latency"]+=nop_interface_latency_2_5d/2
            nop_position_id_d = dest_chip_node["Stack ID"]+"_"+edge_attrs.get("nop_location_d", None)[0]
            G_stack.nodes[nop_position_id_d]["2.5d link Latency"]+=nop_interface_latency_2_5d/2
            #destination NoP interface

            router_latency_2_5d=hops2_5d*(trc_cycles_nop+tva_cycles_nop+tsa_cycles_nop+tst_cycles_nop+tl_cycles_nop)+tenq_cycles_nop*Q_2_5d/bus_width_2_5d
            router_latency_2_5d/=f_nop
            #import pdb; pdb.set_trace()
            nop_latency=update_dict(nop_latency, (source_chip_tile_idx.split('_')[0], dest_chip_tile_idx.split('_')[0], connection_type), nop_interface_latency_2_5d+router_latency_2_5d)
            nop_latency_breakdown[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)] = {"nop interface latency": nop_interface_latency_2_5d,
                                    "nop router latency": router_latency_2_5d,
                                    "act_fac_channel": (hops2_5d*(tl_cycles_nop)+Q_2_5d/bus_width_2_5d)/ (router_latency_2_5d*f_nop) if router_latency_2_5d!=0 else 0,
                                    "act_fac_inputbuffer": tenq_cycles_nop*Q_2_5d/bus_width_2_5d/ (router_latency_2_5d*f_nop) if router_latency_2_5d!=0 else 0,
                                    "act_fac_switch": hops2_5d*(trc_cycles_nop+tva_cycles_nop+tsa_cycles_nop)/ (router_latency_2_5d*f_nop) if router_latency_2_5d!=0 else 0,
                                    "act_fac_switchctrl": (hops2_5d*(trc_cycles_nop+tva_cycles_nop+tsa_cycles_nop)+Q_2_5d/bus_width_2_5d)/ (router_latency_2_5d*f_nop) if router_latency_2_5d!=0 else 0,
                                    "act_fac_output": (hops2_5d*(tst_cycles_nop)+Q_2_5d/bus_width_2_5d)/ (router_latency_2_5d*f_nop) if router_latency_2_5d!=0 else 0,
                                    "act_fac_outputclk": (hops2_5d*(tst_cycles_nop))/ (router_latency_2_5d*f_nop) if router_latency_2_5d!=0 else 0}
            #import pdb; pdb.set_trace()
            for nop_position in edge_attrs.get("nop_router_list", []):
                nop_node_chip_idx=nop_chip_dict[nop_position]
                nop_node = G_sys.nodes[nop_node_chip_idx]
                #import pdb; pdb.set_trace()
                nop_node=update_dict(nop_node, "NoP Router Latency", router_latency_2_5d)
        if connection_type in ["3d", "3_5d"]:
            hops3d=edge_attrs.get("hops3d", 0)
            Q_3d=edge_attrs.get("Q", 0)
            bus_width_3d=edge_attrs.get("bus_width", 1)
            latency_3d=hops3d*(trc_cycles_noc+tva_cycles_noc+tsa_cycles_noc+tst_cycles_noc+tl_cycles_noc)+tenq_cycles_noc*Q_3d/bus_width_3d
            noc_latency_breakdown_3d[(source_chip_tile_idx, dest_chip_tile_idx)]={"3d latency": latency_3d/f_noc,
                                    "act_fac_channel": (hops3d*(tl_cycles_noc)+Q_3d/bus_width_3d)/ latency_3d if latency_3d!=0 else 0,
                                    "act_fac_inputbuffer": tenq_cycles_noc*Q_3d/bus_width_3d/ latency_3d if latency_3d!=0 else 0,
                                    "act_fac_switch": hops3d*(trc_cycles_noc+tva_cycles_noc+tsa_cycles_noc)/ latency_3d if latency_3d!=0 else 0,
                                    "act_fac_switchctrl": (hops3d*(trc_cycles_noc+tva_cycles_noc+tsa_cycles_noc)+Q_3d/bus_width_3d)/ latency_3d if latency_3d!=0 else 0,
                                    "act_fac_output": (hops3d*(tst_cycles_noc)+Q_3d/bus_width_3d)/ latency_3d if latency_3d!=0 else 0,
                                    "act_fac_outputclk": (hops3d*(tst_cycles_noc))/ latency_3d if latency_3d!=0 else 0
            }
            latency_3d/=f_noc

        #import pdb; pdb.set_trace()    
        noc_latency_3d[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)] = latency_3d
        
        for router_node_stack_idx, noc_positions in edge_attrs.get("router_list_3d", []).items():
            for noc_position in noc_positions:
                #import pdb; pdb.set_trace()
                router_node_chip_tile_idx=noc_tile_dict[router_node_stack_idx][noc_position]
                router_node = G_chip.nodes[router_node_chip_tile_idx]
                router_node["3D link Latency"] += latency_3d

    return noc_latency_2d, noc_latency_3d, nop_latency, noc_latency_breakdown_2d, noc_latency_breakdown_3d, nop_latency_breakdown, G_chip, G_stack, G_sys
#print(tile_map)

def determine_network_energy(G_chip, G_stack, G_sys,noc_tile_dict, nop_chip_dict, network_json_data, nw_df, noc_latency_breakdown_2d, noc_latency_breakdown_3d, nop_latency_breakdown):
    noc_energy_2d={}
    noc_energy_3d={}
    nop_energy={}
    for edge in G_chip.edges():
        source_chip_tile_idx, dest_chip_tile_idx=edge
        edge_attrs=G_chip.edges[edge]
        source_chip_node=G_chip.nodes[source_chip_tile_idx]
        dest_chip_node=G_chip.nodes[dest_chip_tile_idx]
        connection_type=edge_attrs.get("connection_type", 0)
        bus_width_2d=edge_attrs.get("bus_width", 1)
        _, channel_power_2d,router_power_2d =power_router_single(bus_width_2d, "2D_NoC", noc_latency_breakdown_2d[(source_chip_tile_idx, dest_chip_tile_idx)])
        channel_power_2d*=1e-3*f_noc/1e9 # converting to W and scaling to respective frequency
        router_power_2d*=1e-3*f_noc/1e9
        #import pdb; pdb.set_trace()
        energy_2d=(channel_power_2d+router_power_2d)*noc_latency_breakdown_2d[(source_chip_tile_idx, dest_chip_tile_idx)]["2d latency"]
        router_list_len_2d = len(edge_attrs.get("router_list_2d", []))
        for router_node_stack_idx, noc_positions in edge_attrs.get("router_list_2d", []).items():
            for noc_position in noc_positions:
                #import pdb; pdb.set_trace()
                chip_tile_idx=noc_tile_dict[router_node_stack_idx][noc_position]
                router_node = G_chip.nodes[chip_tile_idx]
                router_node["router Energy"] += energy_2d/router_list_len_2d if router_list_len_2d>0 else 0

        energy_3d=0
        if connection_type in ["2_5d", "3_5d"]:
            hops2_5d=edge_attrs.get("hops2_5d", 0)
            Q_2_5d=edge_attrs.get("Q", 0)
            bus_width_2_5d=edge_attrs.get("bus_width", 1)
            no_2_5d_channels_active=edge_attrs.get("no_2_5d_channels_active", 1)
            aib_out=aib(Q_2_5d*1e-6/8, None, 0, 0.8, n_ch=no_2_5d_channels_active)
            nop_interface_energy_2_5d=aib_out[1]*1e-12 #converting to J
            nop_position_id_s = source_chip_node["Stack ID"]+"_"+edge_attrs.get("nop_location_s", None)[0]
            G_stack.nodes[nop_position_id_s]["2.5d link Energy"]+=nop_interface_energy_2_5d/2
            nop_position_id_d = dest_chip_node["Stack ID"]+"_"+edge_attrs.get("nop_location_d", None)[0]
            G_stack.nodes[nop_position_id_d]["2.5d link Energy"]+=nop_interface_energy_2_5d/2
            #import pdb; pdb.set_trace()

            _, channel_power_2_5d,router_power_2_5d=power_router_single(bus_width_2_5d, "2D_NoC", nop_latency_breakdown[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)])
            channel_power_2_5d*=1e-3*f_nop/1e9
            router_power_2_5d*=1e-3*f_nop/1e9
            router_energy_2_5d=(channel_power_2_5d+router_power_2_5d)*nop_latency_breakdown[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)]["nop router latency"]
            nop_energy=update_dict(nop_energy, (source_chip_tile_idx.split('_')[0], dest_chip_tile_idx.split('_')[0], connection_type), router_energy_2_5d+nop_interface_energy_2_5d)
            router_list_len_2_5d = len(edge_attrs.get("nop_router_list", []))
            for nop_position in edge_attrs.get("nop_router_list", []):
                nop_node_chip_idx=nop_chip_dict[nop_position]
                nop_node = G_sys.nodes[nop_node_chip_idx]
                nop_node = update_dict(nop_node, "NoP Router Energy", router_energy_2_5d/router_list_len_2_5d if router_list_len_2_5d>0 else 0)
            #nop_position_id_s = source_chip_node["Stack ID"]+"_"+edge_attrs.get("nop_location_s", None)[0]
            #G_stack.nodes[nop_position_id_s]["2.5d link Energy"]+=energy_2_5d
        if connection_type in ["3d", "3_5d"]:
            bus_width_3d=edge_attrs.get("bus_width", 1)
            _, channel_power_3d,router_power_3d=power_router_single(bus_width_3d, "TSV", noc_latency_breakdown_3d[(source_chip_tile_idx, dest_chip_tile_idx)])
            channel_power_3d*=1e-3*f_noc/1e9 # converting to W and scaling to respective frequency
            router_power_3d*=1e-3*f_noc/1e9
            energy_3d=(channel_power_3d+router_power_3d)*noc_latency_breakdown_3d[(source_chip_tile_idx, dest_chip_tile_idx)]["3d latency"]
        noc_energy_2d[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)] = energy_2d
        noc_energy_3d[(source_chip_tile_idx, dest_chip_tile_idx, connection_type)] = energy_3d
        router_list_len_3d = len(edge_attrs.get("router_list_3d", []))
        for router_node_stack_idx, noc_positions in edge_attrs.get("router_list_3d", []).items():
            for noc_position in noc_positions:
                router_node_chip_tile_idx=noc_tile_dict[router_node_stack_idx][noc_position]
                router_node = G_chip.nodes[router_node_chip_tile_idx]
                router_node["3D link Energy"] += energy_3d/router_list_len_3d if router_list_len_3d>0 else 0
        #source_chip_node["router Energy"]+= energy_2d
        #source_chip_node["3D link Energy"]+= energy_3d
    
    return noc_energy_2d, noc_energy_3d, nop_energy, G_chip, G_stack, G_sys

def network_main_fn(G_ai_model, G_chip, G_sys, G_stack, noc_tile_dict, nop_chip_dict, tile_map, mem_req, mesh_size, mesh_size_nop, stack_ids):
    file_path = current_dir+'/Network.json'
    with open(file_path, 'r') as f:
        network_json_data  = json.load(f)
    nw_df= pd.read_csv(f"{target_file_path}/Network_Spec_{aimodel}.csv")
     
    nop_json_path = os.path.join(parent_dir, 'aib_params.json')
    with open(nop_json_path, 'r') as f:
        nop_json_data  = json.load(f)
    #import pdb; pdb.set_trace()
    G_chip, noc_tile_dict, G_stack, nop_chip_dict, G_sys=network_map(G_chip, G_sys, G_ai_model, G_stack, noc_tile_dict, nop_chip_dict, nw_df, tile_map, mem_req, mesh_size, mesh_size_nop, nop_json_data)

    G_chip,G_sys,G_stack, beachhead, link_feasibility, router_area, beachhead_chiplets, n_signal_IO=determine_network_area(network_json_data, G_chip, G_sys, G_stack, nw_df, stack_ids, nop_json_data)
    
    #latency
    noc_latency_2d, noc_latency_3d,nop_latency, noc_latency_breakdown_2d, noc_latency_breakdown_3d, nop_latency_breakdown,G_chip, G_stack, G_sys=determine_network_latency(G_chip, G_stack, G_sys, noc_tile_dict, nop_chip_dict,network_json_data, nw_df)
    
    #import pdb; pdb.set_trace()

    #power function update
    #calculate power
    noc_energy_2d, noc_energy_3d,nop_energy,G_chip, G_stack, G_sys=determine_network_energy(G_chip, G_stack, G_sys, noc_tile_dict, nop_chip_dict, network_json_data, nw_df, noc_latency_breakdown_2d, noc_latency_breakdown_3d, nop_latency_breakdown)

    for node in G_chip.nodes():
        G_chip.nodes[node]["router Power"]= G_chip.nodes[node]["router Energy"]/G_chip.nodes[node]["router Latency"] if G_chip.nodes[node]["router Latency"]>0 else 0
        G_chip.nodes[node]["3D link Power"]= G_chip.nodes[node]["3D link Energy"]/G_chip.nodes[node]["3D link Latency"] if G_chip.nodes[node]["3D link Latency"]>0 else 0
    
    for node in G_stack.nodes():
        G_stack.nodes[node]["2.5d link Power"]= G_stack.nodes[node]["2.5d link Energy"]/G_stack.nodes[node]["2.5d link Latency"] if G_stack.nodes[node]["2.5d link Latency"]>0 else 0
    
    print("----------Network Summary---------------")
    print("Total NoC Area (in mm^2)", sum(router_area.values()))
    print("Total NoP Interface Area (in mm^2)", sum([G_stack.nodes[stack_id]["2.5d link Area"] for stack_id in G_stack.nodes()]))
    print("Total NoP Router Area (in mm^2)", sum([G_sys.nodes[chiplet_idx]["NoP Router Area"] for chiplet_idx in G_sys.nodes()]))
    print("Total 2D Network NoC latency (in s)", sum(noc_latency_2d.values()))
    print("Total 2D Network NoC energy (in J)", sum(noc_energy_2d.values()))
    print("Total 2.5D Network NoP latency (in s)", sum(nop_latency.values()))
    print("Total 2.5D Network NoP energy (in J)", sum(nop_energy.values()))
    print("Total 3D Network NoC latency (in s)", sum(noc_latency_3d.values()))
    print("Total 3D Network NoC energy (in J)", sum(noc_energy_3d.values()))
    print("----------------------------------------")

    print("----------Combined Summary (Compute+Noc+NoP+Memory)---------------")
    stack_area={G_sys.nodes[chip_idx]["Stack ID"] : G_sys.nodes[chip_idx]["Chiplet Actual Area"] for chip_idx in G_sys.nodes()}
    #import pdb; pdb.set_trace()
    print("Total Chip Area (in mm^2)", sum(stack_area.values())+sum([G_sys.nodes[chiplet_idx]["NoP Router Area"] for chiplet_idx in G_sys.nodes()]))
    #import pdb; pdb.set_trace()
    print("Total Chip Latency (in s)", sum([G_chip.nodes[node]["Tile_Latency"] for node in G_chip.nodes()])+sum(noc_latency_2d.values())+sum(nop_latency.values())+ sum(noc_latency_3d.values()))
    print("Total Chip Energy (in J)", sum([G_chip.nodes[node]["Tile_Energy"] for node in G_chip.nodes()])+sum(noc_energy_2d.values())+sum(nop_energy.values())+ sum(noc_energy_3d.values()))
    print("----------------------------------------")
    if DEBUG:
        print_summary(G_chip, G_sys, G_stack)
        #plot breakdown of compute, noc and memory area in single pie chart
        labels = ['Compute', 'Memory', 'NoC', 'NoP Interface', 'NoP Router']
        compute_area=sum([G_chip.nodes[node]["Tile_Area"] for node in G_chip.nodes() if G_chip.nodes[node]["HW Type"].endswith("SA") or G_chip.nodes[node]["HW Type"].endswith("CPU")])
        memory_area=sum([G_chip.nodes[node]["Tile_Area"] for node in G_chip.nodes() if G_chip.nodes[node]["HW Type"].endswith("Mem") or G_chip.nodes[node]["HW Type"].startswith("Mem")])
        noc_area=sum(router_area.values())
        nop_interface_area=sum([G_stack.nodes[stack_id]["2.5d link Area"] for stack_id in G_stack.nodes()]) 
        nop_router_area=sum([G_sys.nodes[chiplet_idx]["NoP Router Area"] for chiplet_idx in G_sys.nodes()])
        area_values = [compute_area, memory_area, noc_area, nop_interface_area, nop_router_area]
        #print(area_values)
        plt.figure(figsize=(10, 6))
        #make text larger in pie chart and place it outside the pie chart
        plt.rcParams['font.size'] = 12
        wedges, texts, autotexts = plt.pie(area_values, labels=labels, autopct='%1.1f%%', startangle=140, colors=['blue', 'orange', 'green', 'red', 'purple'])
        plt.setp(autotexts, size=10, weight="bold", color="white")
        plt.title('Chip Area Breakdown')
        os.makedirs(f"{main_dir}/Results", exist_ok=True)
        plt.savefig(f"{main_dir}/Results/chip_area_breakdown.png", dpi=100, bbox_inches='tight')
        plt.close()
       

        #plot latency and energy breakdown of compute, noc and memory in single pie chart
        labels = ['Compute', 'Memory', 'Network', 'DDR']
        compute_latency=sum([G_chip.nodes[node]["Tile_Latency"] for node in G_chip.nodes() if G_chip.nodes[node]["HW Type"].endswith("SA") or G_chip.nodes[node]["HW Type"].endswith("CPU")])
        memory_latency=sum([G_chip.nodes[node]["Tile_Latency"] for node in G_chip.nodes() if G_chip.nodes[node]["HW Type"].endswith("Mem") or G_chip.nodes[node]["HW Type"].startswith("Mem")])
        noc_latency=sum(noc_latency_2d.values())+sum(nop_latency.values())+ sum(noc_latency_3d.values())
        ddr_latency=sum([G_chip.nodes[node]["DDR_Latency"] for node in G_chip.nodes()])
        latency_values = [compute_latency, memory_latency, noc_latency, ddr_latency]
        compute_energy=sum([G_chip.nodes[node]["Tile_Energy"] for node in G_chip.nodes() if G_chip.nodes[node]["HW Type"].endswith("SA") or G_chip.nodes[node]["HW Type"].endswith("CPU")])
        memory_energy=sum([G_chip.nodes[node]["Tile_Energy"] for node in G_chip.nodes() if G_chip.nodes[node]["HW Type"].endswith("Mem") or G_chip.nodes[node]["HW Type"].startswith("Mem")])
        noc_energy=sum(noc_energy_2d.values())+sum(nop_energy.values())+ sum(noc_energy_3d.values())
        ddr_energy=sum([G_chip.nodes[node]["DDR_Energy"] for node in G_chip.nodes()])
        energy_values = [compute_energy, memory_energy, noc_energy, ddr_energy]
        #import pdb; pdb.set_trace()
        plt.figure(figsize=(12, 6))
        plt.rcParams['font.size'] = 12
        # Latency subplot
        plt.subplot(1, 2, 1)
        wedges, texts, autotexts = plt.pie(latency_values, labels=labels, autopct='%1.1f%%', startangle=140, colors=['blue', 'orange', 'green', 'red'])
        plt.setp(autotexts, size=10, weight="bold", color="white")
        plt.title('Chip Latency Breakdown')
        # Energy subplot
        plt.subplot(1, 2, 2)
        wedges, texts, autotexts = plt.pie(energy_values, labels=labels, autopct='%1.1f%%', startangle=140, colors=['blue', 'orange', 'green', 'red'])
        plt.setp(autotexts, size=10, weight="bold", color="white")
        plt.title('Chip Energy Breakdown')
        plt.tight_layout()
        os.makedirs(f"{main_dir}/Results", exist_ok=True)
        plt.savefig(f"{main_dir}/Results/chip_latency_energy_breakdown.png", dpi=100, bbox_inches='tight')
        plt.close()

        #plot network breakdown of 2d, 2.5d and 3d latency and energy
        ddr_chip_tile_idx=tile_map["DDR_Mem"]

        labels = ['2D NoC', '2.5D NoP', '3D NoC']

        # Example values: each list has [without DDR, with DDR]
        latency_values = [
            [sum(v for k, v in noc_latency_2d.items() if ddr_chip_tile_idx not in k),
            sum(v for k, v in noc_latency_2d.items() if ddr_chip_tile_idx in k)],
            [sum(v for k, v in nop_latency.items() if ddr_chip_tile_idx.split('_')[0] not in k),
            sum(v for k, v in nop_latency.items() if ddr_chip_tile_idx.split('_')[0] in k)],
            [sum(v for k, v in noc_latency_3d.items() if ddr_chip_tile_idx not in k),
            sum(v for k, v in noc_latency_3d.items() if ddr_chip_tile_idx in k)]
        ]

        energy_values = [
            [sum(v for k, v in noc_energy_2d.items() if ddr_chip_tile_idx not in k),
            sum(v for k, v in noc_energy_2d.items() if ddr_chip_tile_idx in k)],
            [sum(v for k, v in nop_energy.items() if ddr_chip_tile_idx.split('_')[0] not in k),
            sum(v for k, v in nop_energy.items() if ddr_chip_tile_idx.split('_')[0] in k)],
            [sum(v for k, v in noc_energy_3d.items() if ddr_chip_tile_idx not in k),
            sum(v for k, v in noc_energy_3d.items() if ddr_chip_tile_idx in k)]
        ]

        #import pdb; pdb.set_trace()

        x = np.arange(len(labels))
        width = 0.35  # width of each bar

        plt.figure(figsize=(12, 6))

        # Latency subplot
        plt.subplot(1, 2, 1)
        plt.bar(x - width/2, [v[0] for v in latency_values], width, label='Network excluding DDR', color='blue')
        plt.bar(x + width/2, [v[1] for v in latency_values], width, label='Network - DDR only', color='orange')
        plt.xticks(x, labels)
        plt.ylabel('Total Latency (s)')
        plt.yscale('log')
        #plot values on top of bars
        plt.title('Network Latency Breakdown')
        plt.legend()

        # Energy subplot
        plt.subplot(1, 2, 2)
        plt.bar(x - width/2, [v[0] for v in energy_values], width, label='Network excluding DDR', color='blue')
        plt.bar(x + width/2, [v[1] for v in energy_values], width, label='Network - DDR only', color='orange')
        plt.xticks(x, labels)
        plt.ylabel('Total Energy (J)')
        plt.yscale('log')
        
        plt.title('Network Energy Breakdown')
        plt.legend()

        plt.tight_layout()
        os.makedirs(f"{main_dir}/Results", exist_ok=True)
        plt.savefig(f"{main_dir}/Results/network_breakdown.png", dpi=100, bbox_inches='tight')

        #import pdb; pdb.set_trace()

        totalhops2d=sum([G_chip.edges[edge]["hops2d"] for edge in G_chip.edges])
        totalhops2_5d=sum([G_chip.edges[edge]["hops2_5d"] for edge in G_chip.edges])
        totalhops3d=sum([G_chip.edges[edge]["hops3d"] for edge in G_chip.edges])
        #BW_3d_list=[G_chip.edges[edge]["BW"] for edge in G_chip.edges if G_chip.edges[edge]["hops3d"]>0]
        #BW_2d_list=[G_chip.edges[edge]["BW"] for edge in G_chip.edges]

        #print("Total 2D hops: ", totalhops2d, " Total 2.5D hops: ", totalhops2_5d, " Total 3D hops: ", totalhops3d)
        #print("Effective BW of communicating tiles - IntraTier: ", BW_2d_list)
        #print("Effective BW of communicating tiles - InterTier: ", BW_3d_list)
        scale_factor={}
        for chip_idx in G_sys.nodes():
            if chip_idx!="Empty":
                scale_factor[chip_idx]=math.ceil(max([G_chip.nodes[chip_tile_idx]["Tile_Area"]**0.5 for chip_tile_idx in G_chip.nodes() if chip_tile_idx.split('_')[0]==chip_idx]+[G_sys.nodes[chip_idx]["2.5D link Area"]**0.5]))

        #import pdb; pdb.set_trace()
        coords={}
        areas, latencies, energies = {}, {}, {}
        for chip_tile_idx in G_chip.nodes():
            chip_idx=chip_tile_idx.split('_')[0]
            node=G_chip.nodes[chip_tile_idx]
            noc_position= tuple(map(int, node["NoC Position"].split(',')))
            area=node.get("Tile_Area", 0)
            edge_tile=math.sqrt(area)
            router_coords = (noc_position[0], noc_position[1]-node.get("router Area", 0)**0.5/scale_factor[chip_idx])
            link_3d_coords =(noc_position[0]+edge_tile/scale_factor[chip_idx] if edge_tile>0 else noc_position[0]+0.5, noc_position[1])
            if chip_idx not in coords:
                coords[chip_idx], areas[chip_idx], latencies[chip_idx], energies[chip_idx] = {}, {}, {}, {}
            coords[chip_idx].update({chip_tile_idx: noc_position, chip_tile_idx+"_router": router_coords, chip_tile_idx+"_3dlink": link_3d_coords})
            areas[chip_idx].update({chip_tile_idx: node.get("Tile_Area", 0), chip_tile_idx+"_router": node.get("router Area", 0), chip_tile_idx+"_3dlink": node.get("3D link Area", 0)})
            latencies[chip_idx].update({chip_tile_idx: node.get("Tile_Latency", 0), chip_tile_idx+"_router": node.get("router Latency", 0), chip_tile_idx+"_3dlink": node.get("3D link Latency", 0)})
            energies[chip_idx].update({chip_tile_idx: node.get("Tile_Energy", 0), chip_tile_idx+"_router": node.get("router Energy", 0), chip_tile_idx+"_3dlink": node.get("3D link Energy", 0)})
        
        for stack_id in G_stack.nodes():
            #import pdb; pdb.set_trace()
            stack_node=G_stack.nodes[stack_id]
            chip_idx=stack_node["chip_idx"]
            coords[chip_idx].update({"nop"+stack_id: stack_node["nop_noc_router_location"]})
            areas[chip_idx].update({"nop"+stack_id: stack_node["2.5d link Area"]})
            latencies[chip_idx].update({"nop"+stack_id: stack_node["2.5d link Latency"]})
            energies[chip_idx].update({"nop"+stack_id: stack_node["2.5d link Energy"]})

        #plot tiles, routers, 3d links for each chiplet in square based on area
        for chip_idx, coord_dict in coords.items():
            #print(chip_idx)
            area_dict=areas[chip_idx]
            latency_dict=latencies[chip_idx]
            energy_dict=energies[chip_idx]
            colors={k: 'red' if k.endswith('_3dlink') else 'grey' if k.endswith('_router') else 'skyblue' for k in area_dict.keys()}
            #import pdb; pdb.set_trace()
            #plot_component(area_dict, latency_dict, energy_dict, coord_dict, f"{main_dir}/Results/Network_Map/Network_Map_{chip_idx}.png", title=f"Network of Chip {chip_idx}", scale_factor=scale_factor[chip_idx], colors=colors)
            plot_component(area_dict, {}, {}, coord_dict, f"{main_dir}/Results/Network_Map/Network_Map_{chip_idx}.png", title=f"Network of Chip {chip_idx}", scale_factor=scale_factor[chip_idx], colors=colors)
            #plot latency heatmap of all tiles, routers, and 3D links - no predefined function

        scale_factor_sys=math.ceil(max([v**0.5 for v in stack_area.values()]+[G_sys.nodes[chip_idx]["NoP Router Area"]**0.5 for chip_idx in G_sys.nodes()]))
        coords={}
        areas = {}
        for chip_idx in G_sys.nodes():
            #import pdb; pdb.set_trace()
            sys_node=G_sys.nodes[chip_idx]
            nop_pos=tuple(map(int, sys_node["NoP Position"].split(',')))
            if nop_pos[2]==0:
                area_chip=sys_node["Chiplet Actual Area"]
                router_coords=(nop_pos[0], nop_pos[1]-sys_node["NoP Router Area"]**0.5/scale_factor_sys)
                stack_idx=sys_node["Stack ID"]
                coords.update({stack_idx: (nop_pos[0], nop_pos[1]), stack_idx+"_router": router_coords})
                areas.update({stack_idx: area_chip, stack_idx+"_router":sys_node["NoP Router Area"]})


        colors={k: 'grey' if k.endswith('_router') else 'skyblue' for k in areas.keys()}
        plot_component(areas, {}, {}, coords, f"{main_dir}/Results/Network_Map/Network_Map_System.png", title=f"Network on Package", scale_factor=scale_factor_sys, colors=colors)


        # Extract keys and values
        router_area = {nodes: G_chip.nodes[nodes]["router Area"] for nodes in G_chip.nodes() if G_chip.nodes[nodes]["router Area"]>0}
        
        plt.figure(figsize=(12, 6))
        plt.bar(list(router_area.keys()), list(router_area.values()))

        plt.xticks(rotation=90)  # Rotate labels for readability
        plt.ylabel("Router Area in mm2")
        plt.title("Area of each NoC Router")

        plt.tight_layout()
        os.makedirs(f"{target_file_path}", exist_ok=True)
        plt.savefig(f"{target_file_path}/noc_router_area_bar_graph.png", dpi=300, bbox_inches="tight")

        # Extract keys and values
        router_latency = {nodes: G_chip.nodes[nodes]["router Latency"] for nodes in G_chip.nodes() if G_chip.nodes[nodes]["router Latency"]>0}
        #import pdb; pdb.set_trace()
        plt.figure(figsize=(12, 6))
        plt.bar(list(router_latency.keys()), list(router_latency.values()))

        plt.xticks(rotation=90)  # Rotate labels for readability
        plt.ylabel("Router Latency in seconds")
        plt.title("Latency of each NoC Router")
        plt.yscale("log")
        plt.text(0, max(list(router_latency.values()))*1.1, "Note: Y-axis is in log scale", fontsize=10, color="red")
        plt.tight_layout()
        os.makedirs(f"{target_file_path}", exist_ok=True)
        plt.savefig(f"{target_file_path}/noc_router_latency_bar_graph.png", dpi=300, bbox_inches="tight")

        # Extract keys and values
        router_energy = {nodes: G_chip.nodes[nodes]["router Energy"] for nodes in G_chip.nodes() if G_chip.nodes[nodes]["router Energy"]>0}

        plt.figure(figsize=(12, 6))
        plt.bar(list(router_energy.keys()), list(router_energy.values()))

        plt.xticks(rotation=90)  # Rotate labels for readability
        plt.ylabel("Router Energy in Joules")
        plt.title("Energy Consumption of each NoC Router")
        plt.yscale("log")
        plt.text(0, max(list(router_energy.values()))*1.1, "Note: Y-axis is in log scale", fontsize=10, color="red")
        plt.tight_layout()
        os.makedirs(f"{target_file_path}", exist_ok=True)
        plt.savefig(f"{target_file_path}/noc_router_energy_bar_graph.png", dpi=300, bbox_inches="tight")

        outs=(G_chip, G_sys)
        with open('G_chip_G_stack'+'.pkl', 'wb') as file:
            pickle.dump(outs, file)   
    #import pdb; pdb.set_trace()

    return G_chip, G_sys, G_stack, nw_df, stack_area, n_signal_IO