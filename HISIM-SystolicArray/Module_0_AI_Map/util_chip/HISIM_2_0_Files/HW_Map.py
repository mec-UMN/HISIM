import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import math
import time
import config
import csv

aimodel=config.aimodel
main_dir = config.main_dir
DEBUG=config.DEBUG
CREATE_DEFAULT_FILES=config.CREATE_DEFAULT_FILES
TYPE_DEFAULT_FILES=config.TYPE_DEFAULT_FILES
SET_SUFF_BANKS=config.SET_SUFF_BANKS
current_dir = os.path.dirname(__file__)
parent_dir_inter = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir_inter)
target_file_path = os.path.join(parent_dir, 'HISIM_2_0_AI_layer_information')
compute_config_file_path = os.path.join(main_dir, 'Module_1_Compute', 'HISIM_2_0_Files', 'HW_configs')
network_config_file_path = os.path.join(main_dir, 'Module_2_Network', 'HISIM_2_0_Files', 'Network_configs')
#print("target: ", target_file_path)
#print("attention network.csv: ", f'{target_file_path}/{aimodel}/Network.csv')
#print("attention edge.csv: ", f'{target_file_path}/{aimodel}/Edge.csv')

def calculate_required_banks(dim_values, node_prec, def_NW, def_NB, def_prec):
    mem_req= math.prod([int(v) if not math.isnan(v) else 1 for v in dim_values])*node_prec
    Nbank=math.ceil(mem_req/def_NW/def_NB/def_prec)
    return Nbank

def create_default_files(G, compute_config_file_path, network_config_file_path, TYPE_DEFAULT_FILES):
    chip_map, mem_spec, sa_spec = {}, {}, {}
    def_Nbank, def_NW, def_NB, def_CM, def_prec, def_clk_hz =1, 1024, 320, 4, 8, 1e9
    def_SA_size_x, def_SA_size_y, def_n_SA = 16, 16, 2
    chip_idx, tile_idx=1, 1
    if TYPE_DEFAULT_FILES=="2_5D_Mesh_Scaled" or TYPE_DEFAULT_FILES=="3_5D_Mesh_Scaled":
        chip_increment=1
        tile_increment=0
    elif TYPE_DEFAULT_FILES=="2D_Mesh":
        chip_increment=0
        tile_increment=1
    

    #Add DDR Mem Tile
    tile_id="T"+str(tile_idx)
    chip_map["C"+str(chip_idx)+"_"+tile_id]=["C"+str(chip_idx), tile_id, "Mem Tile", "0,0", "DDR", "DDR"]
    mem_spec["C"+str(chip_idx)+"_"+tile_id]=["C"+str(chip_idx), tile_id, "DDR", 16, def_NW, def_NB, def_CM, def_prec, def_clk_hz]
    if TYPE_DEFAULT_FILES!="2_5D_Mesh" and TYPE_DEFAULT_FILES!="3_5D_Mesh":
        chip_idx+=chip_increment
        tile_idx+=tile_increment
    else:
        chip_count=3
        tile_increment=1
    for layer_idx, node in G.nodes(data=True):
        if layer_idx!="In":
            #print(layer_idx, node)
            if TYPE_DEFAULT_FILES=="2_5D_Mesh" or TYPE_DEFAULT_FILES=="3_5D_Mesh":
                chip_idx=2 if node["Type"]=="Conv" or node["Type"]=="Matmul" else 3
            if layer_idx=="L1":
                chiplet_id="C"+str(chip_idx)
                tile_id="T"+str(tile_idx)
                chip_map[chiplet_id+"_"+tile_id]=[chiplet_id, tile_id, "Mem Tile", "0,0", layer_idx, "Input"]
                Nbank=calculate_required_banks([node.get(f"indim{i}", 1) for i in range(1, 5)], node.get("prec", 1), def_NW, def_NB, def_prec) if SET_SUFF_BANKS else def_Nbank
                mem_spec[chiplet_id+"_"+tile_id]=[chiplet_id, tile_id, "Mem Tile", Nbank , def_NW, def_NB, def_CM, def_prec, def_clk_hz]
                if TYPE_DEFAULT_FILES!="2_5D_Mesh" and TYPE_DEFAULT_FILES!="3_5D_Mesh":
                    chip_idx+=chip_increment
                tile_idx+=tile_increment
            if node["Type"]=="Conv" or node["Type"]=="Matmul":
                if len(G.in_edges(layer_idx))==1:
                    chiplet_id="C"+str(chip_idx)
                    tile_id="T"+str(tile_idx)
                    chip_map[chiplet_id+"_"+tile_id]=[chiplet_id, tile_id, "Mem Tile", "0,0", layer_idx, "Weight"]
                    Nbank=calculate_required_banks([node.get(f"wdim{i}", 1) for i in range(1, 5)], node.get("prec", 1), def_NW, def_NB, def_prec) if SET_SUFF_BANKS else def_Nbank
                    mem_spec[chiplet_id+"_"+tile_id]=[chiplet_id, tile_id, "Mem Tile", Nbank, def_NW, def_NB, def_CM, def_prec, def_clk_hz]
                    if TYPE_DEFAULT_FILES!="2_5D_Mesh" and TYPE_DEFAULT_FILES!="3_5D_Mesh":
                        chip_idx+=chip_increment
                    tile_idx+=tile_increment
                chiplet_id="C"+str(chip_idx)
                tile_id="T"+str(tile_idx)
                chip_map[chiplet_id+"_"+tile_id]=[chiplet_id, tile_id, "SA", "0,0", layer_idx, node["Type"]]
                sa_spec[chiplet_id+"_"+tile_id]=[chiplet_id, tile_id, "SA", def_SA_size_x, def_SA_size_y, def_n_SA, def_prec, def_clk_hz]
            else:
                chiplet_id="C"+str(chip_idx)
                tile_id="T"+str(tile_idx)
                chip_map[chiplet_id+"_"+tile_id]=[chiplet_id, tile_id, "CPU", "0,0", layer_idx, node["Type"]]
            if TYPE_DEFAULT_FILES!="2_5D_Mesh" and TYPE_DEFAULT_FILES!="3_5D_Mesh":
                chip_idx+=chip_increment
            tile_idx+=tile_increment
            chiplet_id="C"+str(chip_idx)
            tile_id="T"+str(tile_idx)
            chip_map[chiplet_id+"_"+tile_id]=[chiplet_id, tile_id, "Mem Tile", "0,0", layer_idx, "Output"]
            Nbank=calculate_required_banks([node.get(f"outdim{i}", 1) for i in range(1, 5)], node.get("prec", 1), def_NW, def_NB, def_prec) if SET_SUFF_BANKS else def_Nbank
            mem_spec[chiplet_id+"_"+tile_id]=[chiplet_id, tile_id, "Mem Tile", Nbank, def_NW, def_NB, def_CM, def_prec, def_clk_hz]
            if TYPE_DEFAULT_FILES!="2_5D_Mesh" and TYPE_DEFAULT_FILES!="3_5D_Mesh":
                chip_count=chip_idx
                chip_idx+=chip_increment
            tile_idx+=tile_increment
    
    if TYPE_DEFAULT_FILES=="2_5D_Mesh_Scaled":
        no_stacks=len(chip_map.keys())
    elif TYPE_DEFAULT_FILES=="3_5D_Mesh_Scaled":
        no_stacks=len(G.nodes()) #one stack per layer
    elif TYPE_DEFAULT_FILES=="2D_Mesh":
        no_stacks=1
        noc_mesh_size=math.ceil(math.sqrt(len(chip_map.keys())))
        for key, value in chip_map.items():
            # Calculate NoC position - snake pattern
            if (int(value[1][1:])-1)//noc_mesh_size % 2 == 0:
                value[3]= f"{(int(value[1][1:])-1)%noc_mesh_size},{(int(value[1][1:])-1)//noc_mesh_size}"
            else:
                value[3]= f"{noc_mesh_size-1-(int(value[1][1:])-1)%noc_mesh_size},{(int(value[1][1:])-1)//noc_mesh_size}"
    elif TYPE_DEFAULT_FILES=="2_5D_Mesh" or TYPE_DEFAULT_FILES=="3_5D_Mesh":
        noc_mesh_size, noc_mesh={}, 0
        no_stacks=chip_count
        for chip_idx in range(1, chip_count+1):
            no_tiles=sum(1 for key in chip_map.keys() if key.startswith("C"+str(chip_idx)+"_"))
            if TYPE_DEFAULT_FILES=="3_5D_Mesh" and chip_idx!=1:
                noc_mesh=max(noc_mesh, math.ceil(math.sqrt(no_tiles))) 
            else:
                noc_mesh_size[chip_idx]=math.ceil(math.sqrt(no_tiles))
            chiplet_id="C"+str(chip_idx)

        for chip_idx in range(1, chip_count+1):
            if chip_idx not in noc_mesh_size:
                noc_mesh_size[chip_idx]=noc_mesh

        tile_idx={i:0 for i in range(1, chip_count+1)}
        #print(tile_idx)
        for key, value in chip_map.items():
            chiplet_id=key.split('_')[0]
            chip_idx=int(chiplet_id[1:])
            # Calculate NoC position - snake pattern
            tile_idx[chip_idx]+=1
            value[1]= "T"+str(tile_idx[chip_idx])
            if (tile_idx[chip_idx]-1)//noc_mesh_size[chip_idx] % 2 == 0:
                value[3]= f"{(tile_idx[chip_idx]-1)%noc_mesh_size[chip_idx]},{(tile_idx[chip_idx]-1)//noc_mesh_size[chip_idx]}"
            else:
                value[3]= f"{noc_mesh_size[chip_idx]-1-(tile_idx[chip_idx]-1)%noc_mesh_size[chip_idx]},{(tile_idx[chip_idx]-1)//noc_mesh_size[chip_idx]}"
            if key in mem_spec:
                mem_spec[key][1]=value[1]
            if key in sa_spec:
                sa_spec[key][1]=value[1]
    # create a Chip_Map_<aimodel>.csv file in main_dir
    chip_map_file = os.path.join(main_dir, f'Chip_Map_{aimodel}.csv')

    # optional: define headers for readability
    headers = ["Chiplet ID", "Tile ID", "HW Type", "NoC Position", "AI Layer", "Comment"]

    with open(chip_map_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # write header row
        writer.writerows(chip_map.values())  # write only the values, no keys
    mem_spec_file = os.path.join(compute_config_file_path, f'Mem_Spec_{aimodel}.csv')
    # optional: define headers for readability
    headers = ["Chiplet ID", "Tile ID", "HW Type", "Nbank", "NW", "NB", "CM", "prec", "Clock Frequency (Hz)"]
    with open(mem_spec_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # write header row
        writer.writerows(mem_spec.values())  # write only the values, no keys
    
    sa_spec_file = os.path.join(compute_config_file_path, f'SA_Spec_{aimodel}.csv')
    # optional: define headers for readability
    headers = ["Chiplet ID", "Tile ID", "HW Type", "SA_size_x", "SA_size_y", "n_SA", "prec", "Clock Frequency (Hz)"]
    with open(sa_spec_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # write header row
        writer.writerows(sa_spec.values())  # write only the values, no keys

    nop_mesh_size=math.ceil(math.sqrt(no_stacks))
    sys_map, net_spec = {}, {}
    def_n_2d_links_per_tile, def_n_3d_links_per_tile, def_n_2_5d_links_per_chiplet_edge = 80, 500, 1
    for chiplet_idx in range(1, chip_count+1):
        chiplet_id="C"+str(chiplet_idx)
        if TYPE_DEFAULT_FILES=="3_5D_Mesh_Scaled":
            layer_idx=chip_map[chiplet_id+"_T1"][4]
            chiplet_role=chip_map[chiplet_id+"_T1"][5]
            stack_idx=int(layer_idx[1:])+2 if layer_idx!="DDR" and chiplet_role!="Input" else 2 if chiplet_role=="Input" else 1
            tier_pos_idx=1 if chiplet_role=="Output" else 2 if chiplet_role=="Weight" else 0
        elif TYPE_DEFAULT_FILES=="3_5D_Mesh":
            stack_idx=2 if chiplet_idx>1 else 1
            tier_pos_idx= 1 if chiplet_idx==3 else 0
        elif TYPE_DEFAULT_FILES=="2_5D_Mesh_Scaled" or TYPE_DEFAULT_FILES=="2_5D_Mesh":
            stack_idx=chiplet_idx
            tier_pos_idx=0
        elif TYPE_DEFAULT_FILES=="2D_Mesh":
            stack_idx=1
            tier_pos_idx=0
        stack_id="S"+str(stack_idx)
        # Calculate NoP position - snake pattern
        if (stack_idx-1)//nop_mesh_size % 2 == 0:
            nop_pos= (f"{(stack_idx-1)%nop_mesh_size},{(stack_idx-1)//nop_mesh_size},{tier_pos_idx}")
        else:
            nop_pos= (f"{nop_mesh_size-1-(stack_idx-1)%nop_mesh_size},{(stack_idx-1)//nop_mesh_size},{tier_pos_idx}")
        sys_map[chiplet_id]=[chiplet_id, stack_id, "TR0", nop_pos]
        net_spec[stack_id]=[stack_id, def_n_2d_links_per_tile, def_n_3d_links_per_tile, def_n_2_5d_links_per_chiplet_edge]
    # create a Sys_Map_<aimodel>.csv file in main_dir
    sys_map_file = os.path.join(main_dir, f'Sys_Map_{aimodel}.csv')
    # optional: define headers for readability
    headers = ["Chiplet ID", "Stack ID", "Tier ID", "NoP Position"]
    with open(sys_map_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # write header row
        writer.writerows(sys_map.values())  # write only the values, no keys        
    
    net_spec_file = os.path.join(network_config_file_path, f'Network_Spec_{aimodel}.csv')
    # optional: define headers for readability
    headers = ["Stack ID", "N_2D_Links_per_tile", "N_3D_Links_per_tile", "N_2.5D_channels_per_chiplet_edge"]
    with open(net_spec_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # write header row
        writer.writerows(net_spec.values())  # write only the values, no keys
    
    #import pdb; pdb.set_trace()
    
def load_ai_network(aimodel):
    #Load AI network parameters from the network csv files
    #print(aimodel)
    network_df = pd.read_csv(f"{target_file_path}/{aimodel}/Network.csv")
    edge_df = pd.read_csv(f"{target_file_path}/{aimodel}/Edge.csv")

    # Build graph
    G = nx.DiGraph()  

    # Add nodes with attributes
    for _, row in network_df.iterrows():
        node_id = row.iloc[0]  
        attrs = row.to_dict()
        G.add_node(node_id, **attrs)
    G.add_node("In")
    # Add edges
    for _, row in edge_df.iterrows():
        src, dst = row.iloc[0], row.iloc[1] 
        G.add_edge(src, dst, **attrs)
    #print("Nodes:", G.nodes(data=True))
    #print("Edges:", G.edges(data=True))
    if CREATE_DEFAULT_FILES:
        create_default_files(G, compute_config_file_path, network_config_file_path, TYPE_DEFAULT_FILES)
    if DEBUG:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  

        # Node labels
        labels = {}
        labels["In"]="Input\n"  
        for _, row in network_df.iterrows():
            node_id = str(row.iloc[0])           
            attr1   = str(row.iloc[1])           
            labels[node_id] = f"{node_id}\n{attr1}"  

        # Plot
        plt.figure(figsize=(len(labels), 0.5*len(labels)))
        nx.draw(G, pos, labels=labels,
                node_size=1500, node_color="lightblue",
                font_size=10, font_weight="bold",
                edge_color="gray", arrows=True)

        plt.title(f"{aimodel} AI Network Graph", fontsize=14)
        os.makedirs(f"{target_file_path}/{aimodel}", exist_ok=True)
        plt.savefig(f"{target_file_path}/{aimodel}/AI_network_graph.png", dpi=300, bbox_inches="tight")
        #import pdb; pdb.set_trace()
    return G

def load_ai_chip(chip_config, sys_config):
    #Load AI network parameters from the network csv files
    #print(chip_config)
    chip_df = pd.read_csv(chip_config)
    sys_df = pd.read_csv(sys_config) 
    stack_ids={}
    tier_ids=set()
    nop_chip_dict={}
    #import pdb; pdb.set_trace()
    G_sys =nx.DiGraph()
    # Add nodes with attributes
    for _, row in sys_df.iterrows():
        node_id = str(row.iloc[0]) 
        attrs = row.to_dict()
        G_sys.add_node(node_id, **attrs)
        if row.iloc[1] not in stack_ids:
           stack_ids[row.iloc[1]]=set()
        stack_ids[row.iloc[1]].add(row.iloc[0])
        tier_ids.add(int(row.iloc[2][2:]))
        nop_chip_dict[tuple(map(int, attrs["NoP Position"].split(',')))] = node_id

    # Build graph
    G_chip = nx.DiGraph()  
    tile_ids ={}
    noc_tile_dict={}
    # Add nodes with attributes
    for _, row in chip_df.iterrows():
        node_id = str(row.iloc[0])+"_"+str(row.iloc[1])  
        attrs = row.to_dict()
        G_chip.add_node(node_id, **attrs)
        stack_id=G_sys.nodes[row.iloc[0]]["Stack ID"]   
        G_chip.nodes[node_id]["Stack ID"] = stack_id
        nop_pos = G_sys.nodes[row.iloc[0]]["NoP Position"]
        G_chip.nodes[node_id]["NoP Position"] = nop_pos
        _,_,z_nop=map(int, nop_pos.split(','))
        #import pdb; pdb.set_trace()
        if row.iloc[0] not in tile_ids:
           tile_ids[row.iloc[0]]=set()
        tile_ids[row.iloc[0]].add(int(row.iloc[1][1:]))
        if stack_id not in noc_tile_dict:
            noc_tile_dict[stack_id]={}
        noc_tile_dict[stack_id][tuple(map(int, attrs["NoC Position"].split(',')))+(z_nop,)]=node_id
    
    mesh_size={}
    mesh_increment={}
    for chip_idx in tile_ids:
        noc_positions = chip_df.loc[chip_df.iloc[:, 0] == chip_idx, "NoC Position"]

        # Convert "x,y" -> (int(x), int(y))
        coords = [tuple(map(int, pos.split(','))) for pos in noc_positions]

        # mesh size is (max_x+1, max_y+1)
        max_x = max(x for x, _ in coords)
        max_y = max(y for _, y in coords)

        # Take the larger of the two, +1 because coords are 0-based
        mesh_size[chip_idx] = max(max_x, max_y) + 1

        if chip_idx == "C1":
            mesh_increment[chip_idx]=mesh_size[chip_idx]
        else:
            mesh_increment[chip_idx]=mesh_size[chip_idx]+mesh_increment[chip_idx[0]+str(int(chip_idx[1:])-1)]
    mesh_increment_last=mesh_increment[chip_idx[0]+str(len(tile_ids))]
    #import pdb; pdb.set_trace()

    G_stack = nx.DiGraph()
    mesh_size_nop=0
    for k, chip_ids in stack_ids.items():
        #import pdb; pdb.set_trace()
        chip_id=[chip_id for chip_id in chip_ids if G_sys.nodes[chip_id]["Tier ID"]=="TR0"][0]
        #import pdb; pdb.set_trace()
        nop_pos = G_sys.nodes[chip_id]["NoP Position"]
        x, y, z = map(int, nop_pos.split(','))
        # Take the larger of the two, +1 because coords are 0-based
        mesh_size_nop = max(mesh_size_nop, max(x, y) + 1)
        mesh_size_s=mesh_size[chip_id]
        nodes_with_attrs = [
            (k + "_N", {"NoP Position": nop_pos, "2.5d link position": "N", "nop_noc_router_location": (math.ceil(mesh_size_s/2-1), mesh_size_s, 0), "chip_idx": chip_id}),
            (k + "_S", {"NoP Position": nop_pos, "2.5d link position": "S", "nop_noc_router_location": (math.ceil(mesh_size_s/2-1), -1, 0), "chip_idx": chip_id}),
            (k + "_E", {"NoP Position": nop_pos, "2.5d link position": "E", "nop_noc_router_location": (mesh_size_s, math.ceil(mesh_size_s/2-1), 0), "chip_idx": chip_id}),
            (k + "_W", {"NoP Position": nop_pos, "2.5d link position": "W", "nop_noc_router_location": (-1, math.ceil(mesh_size_s/2-1), 0), "chip_idx": chip_id}),
        ]

        G_stack.add_nodes_from(nodes_with_attrs)
    

    for stack_id in stack_ids:
        all_equal = len({mesh_size[c] for c in stack_ids[stack_id]}) == 1
        if not all_equal:
            print(f"Error: In stack {stack_id}, all chips must have the same mesh size.")
            raise ValueError("Inconsistent mesh sizes in stack.")
    #import pdb; pdb.set_trace()
    #print("Nodes:", G.nodes(data=True))
    if DEBUG:
        pos = nx.nx_agraph.graphviz_layout(G_chip, prog="dot")  

        # Node labels
        labels = {}
        pos={}
        for _, row in chip_df.iterrows():
            node_id = row.iloc[0]+"_"+row.iloc[1]          
            attr1   = row['HW Type']          
            noc_pos_str = row['NoC Position']
            chiplet_id=row.iloc[0]   
            #import pdb; pdb.set_trace()
            nop_pos_str = G_sys.nodes[chiplet_id]["NoP Position"]          
            x_noc, y_noc = map(int, noc_pos_str.split(','))
            x_nop, y_nop, z_nop = map(int, nop_pos_str.split(','))
            labels[node_id] = f"{node_id}\n{attr1}"  
            #import pdb; pdb.set_trace()
            pos[node_id] = (x_noc+x_nop*max(mesh_size.values())+ x_nop*3, y_noc+y_nop*max(mesh_size.values())+y_nop*3, z_nop*3) #3 to have enough spacing between each tier
        #print(pos)
        fig = plt.figure(figsize=(3*mesh_increment_last*len(tier_ids),2*mesh_increment_last*len(tier_ids)))
        ax = fig.add_subplot(111, projection='3d')

        # Draw nodes
        for node, (x, y, z) in pos.items():
            ax.scatter(x, y, z, s=300, c='lightblue')
            ax.text(x, y, z, labels[node].split('\n')[0]+'\n'+labels[node].split('\n')[1][0:4]+'\n', fontsize=11, ha='center', va='bottom')
        
        ax.set_title("Hardware Plot")
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        #ax.set_zlim(top=(len(tier_ids)-1)*3+0.01) #3 to have enough spacing between each tier
        ax.set_zlim(bottom=0)
        #add note: grid lines are not to scale
        ax.text2D(0.05, 0.95, "Note: Grid lines are not to scale", transform=ax.transAxes, fontsize=10, verticalalignment='top')
        os.makedirs(f"{main_dir}/Results", exist_ok=True)
        plt.savefig(f"{main_dir}/Results/HW_config_plot.png", dpi=300, bbox_inches="tight")
    #import pdb; pdb.set_trace()

    return G_sys, G_chip, G_stack, noc_tile_dict, nop_chip_dict, tier_ids, stack_ids, tile_ids, mesh_size, mesh_size_nop