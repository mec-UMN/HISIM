import pickle
#Inputs to the Thermal Module
#G_chip:Graph representing the tiles, layer connections and their PPA
#G_stack: Graph representing the chiplet, stacks and their connections
#Device_map.csv file: Information regarding the sublayers in the z direction for each stack
#Thermal.csv file: Thermal properties of the tile type
#T_ambient: Ambient temperature -298K
#mesh_soze_resolution: Resolution of the mesh grid for thermal solver - 0.1mm - Talk to Ziyao how to update it

#Note:
#Parameters of graph G_chip:
#Node_name: Chiplet ID_Tile ID
#Node attributes: 'Chiplet ID': 'C1', 'Tile ID': 'T1', 'HW Type', 'NoC Position', 'NoP Position', 'Stack ID', 'Tile Area', 'Tile Power', 'Tile Latency', 'router Area', 'router Power', 'router Latency', '3D link Area', '3D link latency', '3D link Power'

#Note:
#Parameters of graph G_stack:
#Node_name: 'Stack ID'_'2.5d link position'
#Node attributes:'Stack ID', 'NoP Position', '2.5d link Area', '2.5d link latency', '2.5d link power', '2.5d link position'
with open('G_chip_G_stack'+'.pkl', 'rb') as file:
    G_chip, G_stack = pickle.load(file)
    #to read graphs
    #[G_chip.nodes[node] for node in G_chip]
    #[[node, G_stack.nodes[node]] for node in G_stack]
import pdb; pdb.set_trace()
#For HW type: take the last word from G_chip node attribute 'HW Type' - e.g., 'C1 Out Mem' -> 'Mem'
#Area units=mm2 - take sqrt for length/width
#Power units=W
#Latency units=ms

#Steps: import from csvs - device map and thermal info

def thermal_analysis(G_chip, G_stack, Device_map, Thermal_info, T_ambient, mesh_size_resolution):

    #Step 1: Build dicitonaries required for thermal - Each tile, router, chiplet can be of diffrent size -2D/2.5D/3D/3.5D
    #Step 2: Construct G matrix for thermal solver -cube_g_dict
    #Step 3: Run the thermal solver -solver function - sparse_algebra.spsolve - line 1060 in util.py
    #Step 4: Convert the 2D temperature matrix to a set of temperatures for each sublayer, plot thermal maps - part of solver code - line 1066 to 1096 in util.py

    return min_temperature, avg_temperature,max_temperature

#outputs - min_temperature, avg_temperature,max_temperature, thermal maps 