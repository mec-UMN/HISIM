import os
#If DEFAULT_FILES_GENERIC is True, the default files will be created based on the TYPE_DEFAULT_FILES and SET_SUFF_BANKS parameters.

#If generic files are enabled, number of tiles are determined by number of AI model layers, 1 layer is assigned for each tile type.
#Options: "2D_Mesh", "3D_Mesh", "2_5D_Mesh", "3_5D_Mesh", "2_5D_Mesh_Scaled", "3_5D_Mesh_Scaled"
#TYPE_DEFAULT_FILES="2_5D_Mesh" #Functionality based chiplet assignment, a 3-chiplet configuration - DDR memory chiplet, chiplet with SA and memory tiles, and chiplet with CPU and memory tiles. 
#TYPE_DEFAULT_FILES="3_5D_Mesh" #3.5D version of the above 2.5D configuration
#TYPE_DEFAULT_FILES="2_5D_Mesh_Scaled" #Layer-wise chiplet assignment, number of chiplets is equal to number of tiles required in 2D_Mesh configuration, and the tiles are assigned to chiplets based on the number of layers in the AI model.
#TYPE_DEFAULT_FILES="3_5D_Mesh_Scaled" #3.5D version of the above 2.5D configuration

#If DEFAULT_FILES_GENERIC is False, the files will be created based on following input parameters:
#Please note the files produced here are homogeneous chiplets. For heterogeneous chiplets, the files needs to be modified as per required.

#aimodel='mobilenetv2'
#aimodel='vitbase'
#aimodel='resnet50'
#aimodel='vgg16'
#aimodel='gemma1b'
#aimodel='llama'
#aimodel='qwen0.6b'
main_dir = os.path.dirname(__file__)

# ---- written by the HISIM GUI ----
DEBUG=False
CREATE_DEFAULT_FILES=True
DEFAULT_FILES_GENERIC=True
TYPE_DEFAULT_FILES='2D_Mesh'
SET_SUFF_BANKS=True
stack_count=4
chip_count=4
tile_count_dict={'SA': 2, 'CPU': 2, 'Mem_I': 1, 'Mem_W': 1, 'Mem_O': 1}
aimodel='qwen0.6b'
parse_mlir_output=False
# Keep custom maps/specs in ../uploaded_files/<aimodel> and validate the exact
# six files that HISIM is about to consume.
USE_USER_FILES=True
VALIDATE_MAPS=True
VALIDATE_STRICT=False
VALIDATE_VERBOSE=False
def_SA_size_x=16
def_SA_size_y=16
def_n_SA=2
def_prec=8
def_clk_hz=1000000000.0
def_Nbank=1
def_NW=1024
def_NB=320
def_CM=4
def_n_2d_links_per_tile=80
def_n_3d_links_per_tile=80
def_n_2_5d_channels_per_chiplet_edge=1
