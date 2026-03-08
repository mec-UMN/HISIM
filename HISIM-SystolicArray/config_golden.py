import os
DEBUG=False
CREATE_DEFAULT_FILES=True
DEFAULT_FILES_GENERIC=True
#If DEFAULT_FILES_GENERIC is True, the default files will be created based on the TYPE_DEFAULT_FILES and SET_SUFF_BANKS parameters.
TYPE_DEFAULT_FILES="2D_Mesh"
#TYPE_DEFAULT_FILES="2_5D_Mesh"
#TYPE_DEFAULT_FILES="3_5D_Mesh"
#TYPE_DEFAULT_FILES="2_5D_Mesh_Scaled"
#TYPE_DEFAULT_FILES="3_5D_Mesh_Scaled"
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
main_dir = os.path.dirname(__file__)
parse_mlir_output=False
