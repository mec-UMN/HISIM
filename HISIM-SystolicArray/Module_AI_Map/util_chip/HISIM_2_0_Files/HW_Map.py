import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import math
import time
import config

aimodel=config.aimodel
main_dir = config.main_dir
DEBUG=config.DEBUG
current_dir = os.path.dirname(__file__)
parent_dir_inter = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir_inter)
target_file_path = os.path.join(parent_dir, 'HISIM_2_0_AI_layer_information')

#print("target: ", target_file_path)
#print("attention network.csv: ", f'{target_file_path}/{aimodel}/Network.csv')
#print("attention edge.csv: ", f'{target_file_path}/{aimodel}/Edge.csv')


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

    # Build graph
    G_chip = nx.DiGraph()  
    tile_ids ={}
    # Add nodes with attributes
    for _, row in chip_df.iterrows():
        node_id = str(row.iloc[0])+"_"+str(row.iloc[1])  
        attrs = row.to_dict()
        G_chip.add_node(node_id, **attrs)
        G_chip.nodes[node_id]["Stack ID"] = G_sys.nodes[row.iloc[0]]["Stack ID"]
        G_chip.nodes[node_id]["NoP Position"] = G_sys.nodes[row.iloc[0]]["NoP Position"]
        if row.iloc[0] not in tile_ids:
           tile_ids[row.iloc[0]]=set()
        tile_ids[row.iloc[0]].add(int(row.iloc[1][1:]))
    
    G_stack = nx.DiGraph()  

    for k, chip_ids in stack_ids.items():
        #import pdb; pdb.set_trace()
        chip_id=next(iter(chip_ids))
        tile_id=next(iter(tile_ids[chip_id]))
        #import pdb; pdb.set_trace()
        nop_pos = G_chip.nodes[chip_id+'_T'+str(tile_id)]["NoP Position"]

        nodes_with_attrs = [
            (k + "_N", {"NoP Position": nop_pos, "2.5d link position": "N"}),
            (k + "_S", {"NoP Position": nop_pos, "2.5d link position": "S"}),
            (k + "_E", {"NoP Position": nop_pos, "2.5d link position": "E"}),
            (k + "_W", {"NoP Position": nop_pos, "2.5d link position": "W"}),
        ]

        G_stack.add_nodes_from(nodes_with_attrs)
    

    mesh_size={}
    mesh_increment={}
    for chip_idx in tile_ids:
        noc_positions = chip_df.loc[chip_df.iloc[:, 0] == chip_idx, "NoC Position"]

        # Convert "x,y" -> (int(x), int(y))
        coords = [tuple(map(int, pos.split(","))) for pos in noc_positions]

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
        ax.set_zlim(top=(len(tier_ids)-1)*3+0.01) #3 to have enough spacing between each tier
        ax.set_zlim(bottom=0)


        plt.savefig(f"{main_dir}/Results/HW_config_plot.png", dpi=300, bbox_inches="tight")
        #import pdb; pdb.set_trace()

    return G_sys, G_chip, G_stack, tier_ids, stack_ids, tile_ids, mesh_size