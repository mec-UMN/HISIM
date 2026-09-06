import os
DEBUG=False
CREATE_DEFAULT_FILES=True
DEFAULT_FILES_GENERIC=True
#If DEFAULT_FILES_GENERIC is True, the default files will be created based on the TYPE_DEFAULT_FILES and SET_SUFF_BANKS parameters.

#If generic files are enabled, number of tiles are determined by number of AI model layers, 1 layer is assigned for each tile type.
#Options: "2D_Mesh", "3D_Mesh", "2_5D_Mesh", "3_5D_Mesh", "2_5D_Mesh_Scaled", "3_5D_Mesh_Scaled"
TYPE_DEFAULT_FILES="2D_Mesh" #2D configuration
#TYPE_DEFAULT_FILES="2_5D_Mesh" #Functionality based chiplet assignment, a 3-chiplet configuration - DDR memory chiplet, chiplet with SA and memory tiles, and chiplet with CPU and memory tiles. 
#TYPE_DEFAULT_FILES="3_5D_Mesh" #3.5D version of the above 2.5D configuration
#TYPE_DEFAULT_FILES="2_5D_Mesh_Scaled" #Layer-wise chiplet assignment, number of chiplets is equal to number of tiles required in 2D_Mesh configuration, and the tiles are assigned to chiplets based on the number of layers in the AI model.
#TYPE_DEFAULT_FILES="3_5D_Mesh_Scaled" #3.5D version of the above 2.5D configuration
SET_SUFF_BANKS=True

#If DEFAULT_FILES_GENERIC is False, the files will be created based on following input parameters:
#Please note the files produced here are homogeneous chiplets. For heterogeneous chiplets, the files needs to be modified as per required.
stack_count=4 #Number of stacks in the system - >1 for 2.5D and 3.5D architectures, =1 for 2D and 3D architectures
chip_count=4  #Number of chips per stack - =1 for 2D and 2.5D architectures, >1 for 3D and 3.5D architectures
tile_count_dict={"SA":2, "CPU":2,"Mem_I":1, "Mem_W":1, "Mem_O":1}  #Number of tiles of each type within a chiplet. Minimum 1 tile of each type. 

aimodel="Attention"
#aimodel='mobilenetv2'
#aimodel='vitbase'
aimodel='gpt2'
#aimodel='resnet50'
#aimodel='vgg16'
#aimodel='gemma1b'
#aimodel='llama'
#aimodel='qwen0.6b'
main_dir = os.path.dirname(__file__)
parse_mlir_output=False
