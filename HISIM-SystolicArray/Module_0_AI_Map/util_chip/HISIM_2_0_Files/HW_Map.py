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
DEFAULT_FILES_GENERIC=config.DEFAULT_FILES_GENERIC
TYPE_DEFAULT_FILES=config.TYPE_DEFAULT_FILES if CREATE_DEFAULT_FILES and DEFAULT_FILES_GENERIC else "User-Defined"
SET_SUFF_BANKS=config.SET_SUFF_BANKS
stack_count=config.stack_count
chip_count=config.chip_count
tile_count_dict=config.tile_count_dict
def_Nbank = getattr(config, 'def_Nbank', 1)
def_NW=getattr(config, 'def_NW', 1024)
def_NB=getattr(config, 'def_NB', 320)
def_CM=getattr(config, 'def_CM', 4)
def_clk_hz = getattr(config, 'def_clk_hz', 1e9)
def_SA_size_x = getattr(config, 'def_SA_size_x', 16)
def_SA_size_y = getattr(config, 'def_SA_size_y', 16)
def_n_SA= getattr(config, 'def_n_SA', 2)
def_prec = getattr(config, 'def_prec', 8)
def_n_2d_links_per_tile=getattr(config, 'def_n_2d_links_per_tile', 80)
def_n_3d_links_per_tile=getattr(config, 'def_n_3d_links_per_tile', 80)
def_n_2_5d_channels_per_chiplet_edge =getattr(config, 'def_n_2_5d_channels_per_chiplet_edge', 1)


current_dir = os.path.dirname(__file__)
parent_dir_inter = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir_inter)
target_file_path = os.path.join(parent_dir, 'HISIM_2_0_AI_layer_information')
compute_config_file_path = os.path.join(main_dir, 'Module_1_Compute', 'HISIM_2_0_Files', 'HW_configs')
network_config_file_path = os.path.join(main_dir, 'Module_2_Network', 'HISIM_2_0_Files', 'Network_configs')
#print("target: ", target_file_path)
#print("attention network.csv: ", f'{target_file_path}/{aimodel}/Network.csv')
#print("attention edge.csv: ", f'{target_file_path}/{aimodel}/Edge.csv')

def calculate_required_banks(dim_values, node_prec, def_NW, def_NB):
    mem_req= math.prod([int(v) if not math.isnan(v) else 1 for v in dim_values])*node_prec
    Nbank=math.ceil(mem_req/def_NW/def_NB)
    return Nbank

def create_layer_spec(node, layer_idx):
    if node["Type"]=="Conv":
        #import pdb; pdb.set_trace()
        if int(node["Groups"].split(":")[0])==1: #Standard Conv
            layer_spec_idx = [layer_idx, node["Type"], "AxB mul BxC", "Temporal", "B1", "C1-Unroll-out1_dim3", "C2-Unroll-out1_dim4", "A", "B1", "B2", "B3", "NA", "Auto", "SA_size_x", "n_SA*SA_size_y"]
        else:
            layer_spec_idx = [layer_idx, node["Type"], "AxB mul BxC", "Temporal", "Parallel",  "C1-Unroll-out1_dim3", "C2-Unroll-out1_dim4", "Parallel", "B1", "B2", "B3", "n_SA", "Auto", "SA_size_x", "SA_size_y"]
    else:
        if math.isnan(node["in1_dim4"]):
            if math.isnan(node["in1_dim3"]):
                if math.isnan(node["in2_dim3"]):
                    if node["in1_dim2"]==node["in2_dim1"]:
                        layer_spec_idx= [layer_idx, node["Type"], "AxB mul BxC",  "A", "B", "NA", "NA", "B", "C", "NA", "NA", "NA", "n_SA*SA_size_y", "SA_size_x", "Auto"]  if node["in1_dim1"]>node["in2_dim2"] else [layer_idx, node["Type"], "AxB mul BxC",  "A", "B", "NA", "NA", "B", "C", "NA", "NA", "NA",  "Auto", "SA_size_x", "n_SA*SA_size_y"]
                    else:
                        layer_spec_idx = [layer_idx, node["Type"], "AxB mul BxC",  "A", "B", "NA", "NA", "C", "B", "NA", "NA", "NA", "Auto", "SA_size_x", "n_SA*SA_size_y"] 
                else:
                    layer_spec_idx = [layer_idx, node["Type"], "AxB mul BxC",  "A", "B", "NA", "NA", "Temporal", "B", "C", "NA", "NA", "Auto", "SA_size_x", "n_SA*SA_size_y"]
            elif math.isnan(node["in2_dim3"]):
                layer_spec_idx = [layer_idx, node["Type"], "AxB mul BxC", "Temporal", "A", "B", "NA", "B", "C", "NA", "NA", "NA", "SA_size_y", "n_SA*SA_size_x", "Auto"] 
            else:
                layer_spec_idx = [layer_idx, node["Type"], "AxB mul BxC", "Temporal", "A", "B", "NA", "Temporal", "B", "C", "NA", "NA", 	"SA_size_y","n_SA*SA_size_x","Auto"]
        else:
            #print(node["in2_dim3"])
            layer_spec_idx = [layer_idx, node["Type"], "AxB mul BxC", "Temporal", "Parallel", "A", "B", "Temporal", "Parallel", "B", "C", "n_SA", "SA_size_y", "SA_size_x", "Auto"]
    return layer_spec_idx

def create_ddr_spec_initial(chip_idx, tile_idx, def_NW, def_NB, def_CM, def_clk_hz):
    tile_id="T"+str(tile_idx)
    chip_map_idx=["C"+str(chip_idx), tile_id, "Mem Tile", "0,0", ["DDR"], ["DDR"]]
    mem_spec_idx=["C"+str(chip_idx), tile_id, "DDR", 16, def_NW, def_NB, def_CM, def_clk_hz]
    key_name="C"+str(chip_idx)+"_"+tile_id
    return chip_map_idx, mem_spec_idx, key_name

def create_mem_tile_spec_initial(chip_idx, tile_idx, layer_idxes, dim_info_list,prec, mem_type, def_Nbank, def_NW, def_NB, def_CM, def_clk_hz):
    chiplet_id="C"+str(chip_idx)
    tile_id="T"+str(tile_idx)
    chip_map_idx=[chiplet_id, tile_id, "Mem Tile", "0,0", layer_idxes, mem_type] 
    Nbank=0
    if SET_SUFF_BANKS:
        for idx, dim_info in enumerate(dim_info_list):
            Nbank+=calculate_required_banks(dim_info, prec[idx], def_NW, def_NB)
    else:
        Nbank=def_Nbank
    mem_spec_idx=[chiplet_id, tile_id, "Mem Tile", Nbank , def_NW, def_NB, def_CM, def_clk_hz]
    key_name=chiplet_id+"_"+tile_id
    return chip_map_idx, mem_spec_idx, key_name

def create_compute_tile_spec_initial(chip_idx, tile_idx, layer_idxes, compute_type, node_types, def_SA_size_x, def_SA_size_y, def_n_SA, def_prec, def_clk_hz):
    chiplet_id="C"+str(chip_idx)
    tile_id="T"+str(tile_idx)
    chip_map_idx=[chiplet_id, tile_id, compute_type, "0,0", layer_idxes, node_types]
    sa_spec_idx=None
    if compute_type=="SA":
        sa_spec_idx=[chiplet_id, tile_id, compute_type, def_SA_size_x, def_SA_size_y, def_n_SA, def_prec, def_clk_hz]
    key_name=chiplet_id+"_"+tile_id
    return chip_map_idx, sa_spec_idx, key_name

def update_noc_pos_chip_map(tile_id, noc_mesh_size):
    # Calculate NoC position - snake pattern
    if (tile_id-1)//noc_mesh_size % 2 == 0:
        noc_pos= f"{(tile_id-1)%noc_mesh_size},{(tile_id-1)//noc_mesh_size}"
    else:
        noc_pos= f"{noc_mesh_size-1-(tile_id-1)%noc_mesh_size},{(tile_id-1)//noc_mesh_size}"

    return noc_pos
def create_default_files(G, TYPE_DEFAULT_FILES):
    chip_map, mem_spec, sa_spec, layer_spec = {}, {}, {}, {}
    chip_idx, tile_idx=1, 1
    if TYPE_DEFAULT_FILES=="2_5D_Mesh_Scaled" or TYPE_DEFAULT_FILES=="3_5D_Mesh_Scaled":
        chip_increment=1
        tile_increment=0
    elif TYPE_DEFAULT_FILES=="2D_Mesh":
        chip_increment=0
        tile_increment=1
    

    #Add DDR Mem Tile
    chip_map_idx, mem_spec_idx, key_name = create_ddr_spec_initial(chip_idx, tile_idx, def_NW, def_NB, def_CM, def_clk_hz)
    chip_map[key_name]=chip_map_idx
    mem_spec[key_name]=mem_spec_idx
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
                chip_idx=3 if node["Type"]=="Conv" or node["Type"].startswith("MatMul") or node["Type"]=="Gemm" or node["Type"].endswith("Attention") else 2
            if layer_idx=="L1":
                chip_map_idx, mem_spec_idx, key_name = create_mem_tile_spec_initial(chip_idx, tile_idx, [layer_idx], [[node.get(f"in1_dim{i}", 1) for i in range(1, 5)]], [node.get("prec", 1)], ["Input"], def_Nbank, def_NW, def_NB, def_CM, def_clk_hz)
                chip_map[key_name]=chip_map_idx
                mem_spec[key_name]=mem_spec_idx
                if TYPE_DEFAULT_FILES!="2_5D_Mesh" and TYPE_DEFAULT_FILES!="3_5D_Mesh":
                    chip_idx+=chip_increment
                tile_idx+=tile_increment
            if node["Type"]=="Conv" or node["Type"].startswith("MatMul") or node["Type"]=="Gemm" or node["Type"].endswith("Attention"):
                #import pdb; pdb.set_trace()
                if len(G.in_edges(layer_idx))<=1: # to  ensure that QKV layers dont have weight mem tile
                    chip_map_idx, mem_spec_idx, key_name = create_mem_tile_spec_initial(chip_idx, tile_idx, [layer_idx], [[node.get(f"in2_dim{i}", 1) for i in range(1, 5)]], [node.get("prec", 1)], ["Weight"], def_Nbank, def_NW, def_NB, def_CM, def_clk_hz)
                    chip_map[key_name]=chip_map_idx
                    mem_spec[key_name]=mem_spec_idx
                    if TYPE_DEFAULT_FILES!="2_5D_Mesh" and TYPE_DEFAULT_FILES!="3_5D_Mesh":
                        chip_idx+=chip_increment
                    tile_idx+=tile_increment
                chip_map_idx, sa_spec_idx, key_name = create_compute_tile_spec_initial(chip_idx, tile_idx, [layer_idx], "SA", [node["Type"]], def_SA_size_x, def_SA_size_y, def_n_SA, def_prec, def_clk_hz)
                chip_map[key_name]=chip_map_idx
                sa_spec[key_name]=sa_spec_idx
                layer_spec[layer_idx] = create_layer_spec(node, layer_idx)    
            else:
                chip_map_idx, _, key_name = create_compute_tile_spec_initial(chip_idx, tile_idx, [layer_idx], "CPU", [node["Type"]], None, None, None, None, None)
                chip_map[key_name]=chip_map_idx
            if TYPE_DEFAULT_FILES!="2_5D_Mesh" and TYPE_DEFAULT_FILES!="3_5D_Mesh":
                chip_idx+=chip_increment
            tile_idx+=tile_increment
            chip_map_idx, mem_spec_idx, key_name = create_mem_tile_spec_initial(chip_idx, tile_idx, [layer_idx], [[node.get(f"out1_dim{i}", 1) for i in range(1, 5)]], [node.get("prec", 1)], ["Output"], def_Nbank, def_NW, def_NB, def_CM, def_clk_hz)
            chip_map[key_name]=chip_map_idx
            mem_spec[key_name]=mem_spec_idx
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
            value[3]=update_noc_pos_chip_map(int(value[1][1:]), noc_mesh_size)
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
            value[3]=update_noc_pos_chip_map(tile_idx[chip_idx], noc_mesh_size[chip_idx])
            if key in mem_spec:
                mem_spec[key][1]=value[1]
            if key in sa_spec:
                sa_spec[key][1]=value[1]

    nop_mesh_size=math.ceil(math.sqrt(no_stacks))
    sys_map, net_spec = {}, {}
    for chiplet_idx in range(1, chip_count+1):
        chiplet_id="C"+str(chiplet_idx)
        if TYPE_DEFAULT_FILES=="3_5D_Mesh_Scaled":
            layer_idx=chip_map[chiplet_id+"_T1"][4][0]
            chiplet_role=chip_map[chiplet_id+"_T1"][5][0]
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
        sys_map[chiplet_id]=[chiplet_id, stack_id, "TR"+str(tier_pos_idx), nop_pos]
        net_spec[stack_id]=[stack_id, def_n_2d_links_per_tile, def_n_3d_links_per_tile, def_n_2_5d_channels_per_chiplet_edge]    
    
    #import pdb; pdb.set_trace()
    return chip_map, mem_spec, sa_spec, layer_spec, sys_map, net_spec

def create_files_from_user_input(G, no_stacks, no_chip_stack, tile_count):
    #Print the type of configuration based on user input
    if no_stacks==1 and no_chip_stack==1:
        print("Creating files for 2D configuration based on user input")
    elif no_stacks==1 and no_chip_stack>1:
        print("Creating files for 3D configuration based on user input")
    elif no_stacks>1 and no_chip_stack==1:
        print("Creating files for 2.5D configuration based on user input")
    elif no_stacks>1 and no_chip_stack>1:
        print("Creating files for 3.5D configuration based on user input")
    
    chip_count=no_stacks*no_chip_stack

    layer_spec={}
    #Layer Spec
    for layer_idx, node in G.nodes(data=True):
        if layer_idx!="In":
            if node["Type"]=="Conv" or node["Type"].startswith("MatMul") or node["Type"]=="Gemm" or node["Type"].endswith("Attention"):
                layer_spec[layer_idx] = create_layer_spec(node, layer_idx)
    
    #import pdb; pdb.set_trace()
    

    no_layer_chiplet=math.ceil(len(G.nodes())/chip_count)

    #Add DDR Mem Tile to first chiplet
    chip_map, mem_spec, sa_spec = {}, {}, {}
    tile_idx, chip_idx=1, 1
    chip_map_idx, mem_spec_idx, key_name = create_ddr_spec_initial(chip_idx, tile_idx, def_NW, def_NB, def_CM, def_clk_hz)
    chip_map[key_name]=chip_map_idx
    mem_spec[key_name]=mem_spec_idx
    tile_idx+=1

    for chiplet_idx in range(1, chip_count+1):
        #assign the next no_layer_chiplet layers to the current chiplet
        layer_idxes = ["L"+str(i+(chiplet_idx-1)*no_layer_chiplet) for i in range(1, no_layer_chiplet+1)] 
        #remvoe layer idx from layer idxes if it exceeds the total number of layers in the graph
        layer_idxes = [layer_idx for layer_idx in layer_idxes if layer_idx in G.nodes()]
        layer_idxes_sa = [layer_idx for layer_idx in layer_idxes if layer_idx in layer_spec]
        layer_idxes_cpu = [layer_idx for layer_idx in layer_idxes if layer_idx not in layer_spec]
        no_layers_tiles_sa = math.ceil(len(layer_idxes_sa)/tile_count_dict["SA"]) if tile_count_dict["SA"]>0 else print("Error: Number of SA tiles cannot be zero")
        no_layers_tiles_cpu = math.ceil(len(layer_idxes_cpu)/tile_count_dict["CPU"]) if tile_count_dict["CPU"]>0 else print("Error: Number of CPU tiles cannot be zero")
        no_layers_tile_mem ={"Mem_I": math.ceil(len(layer_idxes)/tile_count_dict["Mem_I"]) if tile_count_dict["Mem_I"]>0 else print("Error: Number of Memory Input tiles cannot be zero"),
                            "Mem_W": math.ceil(len(layer_idxes_sa)/tile_count_dict["Mem_W"]) if tile_count_dict["Mem_W"]>0 else print("Error: Number of Memory Weight tiles cannot be zero"),
                            "Mem_O": math.ceil(len(layer_idxes)/tile_count_dict["Mem_O"]) if tile_count_dict["Mem_O"]>0 else print("Error: Number of Memory Output tiles cannot be zero")}
        prec=[G.nodes()[layer_idx].get("prec", 1) for layer_idx in layer_idxes]

        sa_tile_idx, cpu_tile_idx=1, 1
        mem_tile_idx={"Mem_I":1, "Mem_W":1, "Mem_O":1}
        for tile_type, tile_count in tile_count_dict.items():
            for t in range(tile_count):
                if tile_type=="SA":
                    layer_idxes_tile = layer_idxes_sa[sa_tile_idx-1: sa_tile_idx-1+no_layers_tiles_sa] if len(layer_idxes_sa)>=sa_tile_idx else [] 
                    node_types_tile = [G.nodes()[layer_idx]["Type"] for layer_idx in layer_idxes_tile]
                    if len(layer_idxes_tile)>0:
                        chip_map_idx, sa_spec_idx, key_name = create_compute_tile_spec_initial(chiplet_idx, tile_idx, layer_idxes_tile, "SA", node_types_tile, def_SA_size_x, def_SA_size_y, def_n_SA, def_prec, def_clk_hz)
                        chip_map[key_name]=chip_map_idx
                        sa_spec[key_name]=sa_spec_idx
                        sa_tile_idx+=no_layers_tiles_sa
                elif tile_type.startswith("Mem"):
                    if tile_type=="Mem_W":
                        layer_idxes_tile=layer_idxes_sa[mem_tile_idx[tile_type]-1: mem_tile_idx[tile_type]-1+no_layers_tile_mem[tile_type]] if len(layer_idxes_sa)>=mem_tile_idx[tile_type] else []
                        layer_idxes_tile = [layer_idx for layer_idx in layer_idxes_tile if len(G.in_edges(layer_idx))<=1] #Done in order to not assign layers that have dunamic weights such as in QKV
                        tile_role="Weight"
                        dim_infos = [[G.nodes()[layer_idx].get(f"in2_dim{i}", 1) for i in range(1, 5)] for layer_idx in layer_idxes_tile]
                    elif tile_type=="Mem_I":
                        layer_idxes_tile=layer_idxes[mem_tile_idx[tile_type]-1: mem_tile_idx[tile_type]-1+no_layers_tile_mem[tile_type]] if len(layer_idxes)>=mem_tile_idx[tile_type] else []
                        #check if the layers in layer_indxes have incoming edges from the graph to remove redudnacy of output tile of previous layer and input tile of current layer
                        layer_idxes_tile = [layer_idx for layer_idx in layer_idxes_tile if len(G.in_edges(layer_idx))==0] 
                        tile_role="Input"
                        dim_infos = [[G.nodes()[layer_idx].get(f"in1_dim{i}", 1) for i in range(1, 5)] for layer_idx in layer_idxes_tile]
                    elif tile_type=="Mem_O":
                        layer_idxes_tile=layer_idxes[mem_tile_idx[tile_type]-1: mem_tile_idx[tile_type]-1+no_layers_tile_mem[tile_type]] if len(layer_idxes)>=mem_tile_idx[tile_type] else []
                        tile_role="Output"
                        dim_infos = [[G.nodes()[layer_idx].get(f"out1_dim{i}", 1) for i in range(1, 5)] for layer_idx in layer_idxes_tile]
                    if len(layer_idxes_tile)>0:
                        chip_map_idx, mem_spec_idx, key_name = create_mem_tile_spec_initial(chiplet_idx, tile_idx, layer_idxes_tile, dim_infos, prec, [tile_role], def_Nbank, def_NW, def_NB, def_CM, def_clk_hz)
                        chip_map[key_name]=chip_map_idx
                        mem_spec[key_name]=mem_spec_idx
                        mem_tile_idx[tile_type]=mem_tile_idx.get(tile_type, 0)+no_layers_tile_mem[tile_type]
                else:
                    layer_idxes_tile = layer_idxes_cpu[cpu_tile_idx-1: cpu_tile_idx-1+no_layers_tiles_cpu] if len(layer_idxes_cpu)>=cpu_tile_idx else []
                    node_types_tile = [G.nodes()[layer_idx]["Type"] for layer_idx in layer_idxes_tile]
                    if len(layer_idxes_tile)>0:
                        chip_map_idx, _, key_name = create_compute_tile_spec_initial(chiplet_idx, tile_idx, layer_idxes_tile, "CPU", node_types_tile, None, None, None, None, None)
                        chip_map[key_name]=chip_map_idx
                        cpu_tile_idx+=no_layers_tiles_cpu

                tile_idx+=1 #increment tile idx for each tile added to the chiplet
        #import pdb; pdb.set_trace()
        tile_idx=1 # reset tile idx for next chiplet
      
    #Update NoC position in Chip Map
    noc_mesh_size=math.ceil(math.sqrt(sum(tile_count_dict.values())))
    for key, value in chip_map.items():
        value[3]=update_noc_pos_chip_map(int(value[1][1:]), noc_mesh_size)

    nop_mesh_size=math.ceil(math.sqrt(no_stacks))
    sys_map, net_spec = {}, {}
    for chiplet_idx in range(1, chip_count+1):
        #check if the chiplet is part of chip_map, if not continue to next chiplet
        if not any(key.startswith("C"+str(chiplet_idx)+"_") for key in chip_map.keys()):
            continue
        chiplet_id="C"+str(chiplet_idx)
        stack_idx=math.ceil(chiplet_idx/no_chip_stack)
        stack_id="S"+str(stack_idx)
        tier_pos_idx= (chiplet_idx-1)%no_chip_stack if no_chip_stack>1 else 0
        # Calculate NoP position - snake pattern
        if (stack_idx-1)//nop_mesh_size % 2 == 0:
            nop_pos= (f"{(stack_idx-1)%nop_mesh_size},{(stack_idx-1)//nop_mesh_size},{tier_pos_idx}")
        else:
            nop_pos= (f"{nop_mesh_size-1-(stack_idx-1)%nop_mesh_size},{(stack_idx-1)//nop_mesh_size},{tier_pos_idx}")
        sys_map[chiplet_id]=[chiplet_id, stack_id, "TR"+str(tier_pos_idx), nop_pos]
        net_spec[stack_id]=[stack_id, def_n_2d_links_per_tile, def_n_3d_links_per_tile, def_n_2_5d_channels_per_chiplet_edge]    
 
    #import pdb; pdb.set_trace()
    return chip_map, mem_spec, sa_spec, layer_spec, sys_map, net_spec

def write_default_files(chip_map, mem_spec, sa_spec, layer_spec, sys_map, net_spec, compute_config_file_path, network_config_file_path):
    # create a Chip_Map_<aimodel>.csv file in main_dir
    chip_map_file = os.path.join(main_dir, f'Chip_Map_{aimodel}.csv')
    # optional: define headers for readability
    headers = ["Chiplet ID", "Tile ID", "HW Type", "NoC Position", "AI Layer", "NodeName"]
    with open(chip_map_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # write header row
        writer.writerows(chip_map.values())  # write only the values, no keys
    
    mem_spec_file = os.path.join(compute_config_file_path, f'Mem_Spec_{aimodel}.csv')
    # optional: define headers for readability
    headers = ["Chiplet ID", "Tile ID", "HW Type", "Nbank", "NW", "NB", "CM",  "Clock Frequency (Hz)"]
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

    #Create Layer_Mapping_<aimodel>.csv file in main_dir
    layer_mapping_file = os.path.join(main_dir, f'Layer_Mapping_{aimodel}.csv')
    # optional: define headers for readability
    headers = ["Layer ID", "Type", "Function", "in1_dim1", "in1_dim2", "in1_dim3", "in1_dim4", "in2_dim1", "in2_dim2", "in2_dim3", "in2_dim4", "Parallel", "A", "B", "C"]
    with open(layer_mapping_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # write header row
        writer.writerows(layer_spec.values())  # write only the values, no keys

    # create a Sys_Map_<aimodel>.csv file in main_dir
    sys_map_file = os.path.join(main_dir, f'Sys_Map_{aimodel}.csv')
    # optional: define headers for readability
    headers = ["Chiplet ID", "Stack ID", "Tier ID", "NoP Position"]
    with open(sys_map_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # write header row
        writer.writerows(sys_map.values())  # write only the values, no keys       

    #Write Network_Spec_<aimodel>.csv file in network_config_file_path
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
    #get column id whose column named as Layer_ID in network_df
    layer_id_col = network_df.columns.get_loc("Layer_ID")
    layer_type_col = network_df.columns.get_loc("Type")

    # Add nodes with attributes
    for _, row in network_df.iterrows():
        node_id = row.iloc[layer_id_col]
        attrs = row.to_dict()
        if math.isnan(attrs["in2_dim2"]) and attrs["Type"]=="Mul":
            attrs["Type"]="ScalarMul"
        G.add_node(node_id, **attrs)
    G.add_node("In")
    #import pdb; pdb.set_trace()
    # Add edges
    for _, row in edge_df.iterrows():
        src, dst = row.iloc[0], row.iloc[1] 
        G.add_edge(src, dst, **attrs)
    #print("Nodes:", G.nodes(data=True))
    #print("Edges:", G.edges(data=True))
    if CREATE_DEFAULT_FILES:
        if DEFAULT_FILES_GENERIC:
            chip_map, mem_spec, sa_spec, layer_spec, sys_map, net_spec = create_default_files(G, TYPE_DEFAULT_FILES)
        else:
            chip_map, mem_spec, sa_spec, layer_spec, sys_map, net_spec = create_files_from_user_input(G, stack_count, chip_count, tile_count_dict)
        write_default_files(chip_map, mem_spec, sa_spec, layer_spec, sys_map, net_spec, compute_config_file_path, network_config_file_path)
    
    if DEBUG:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  

        # Node labels
        labels = {}
        labels["In"]="Input\n"  
        for _, row in network_df.iterrows():
            node_id = str(row.iloc[layer_id_col])           
            attr1   = str(row.iloc[layer_type_col])           
            labels[node_id] = f"{node_id}\n{attr1}"  

        # Plot
        plt.figure(figsize=(len(labels), 0.5*len(labels)))
        nx.draw(G, pos, labels=labels,
                node_size=1500, node_color="lightblue",
                font_size=10, font_weight="bold",
                edge_color="gray", arrows=True)

        plt.title(f"{aimodel} AI Network Graph", fontsize=14)
        os.makedirs(f"{target_file_path}/{aimodel}", exist_ok=True)
        plt.savefig(f"{target_file_path}/{aimodel}/AI_network_graph.png", dpi=100, bbox_inches="tight")
        plt.close()
        #import pdb; pdb.set_trace()
    return G

def natural_key(key_str):
    prefix = key_str[0]  # 'c' (assumes consistent single-letter prefix)
    num_str = key_str[1:]  # e.g., "1", "10", "111"
    return (prefix, int(num_str))  # Tuple: sorts by prefix first, then number

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
    #import pdb; pdb.set_trace()
    for chip_idx in sorted(tile_ids.keys(), key=natural_key): #ascending order of chip idx
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
        #ax.set_zlim(top=(len(tier_ids)-1)*3+0.01) #3 to have enough spacing between each tier
        ax.set_zlim(bottom=0)
        #add note: grid lines are not to scale
        ax.text2D(0.05, 0.95, "Note: Grid lines are not to scale", transform=ax.transAxes, fontsize=10, verticalalignment='top')
        os.makedirs(f"{main_dir}/Results", exist_ok=True)
        #plt.savefig(f"{main_dir}/Results/HW_config_plot.png", dpi=100, bbox_inches="tight")
    #import pdb; pdb.set_trace()

    return G_sys, G_chip, G_stack, noc_tile_dict, nop_chip_dict, tier_ids, stack_ids, tile_ids, mesh_size, mesh_size_nop