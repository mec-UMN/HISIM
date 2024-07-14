import csv,math,sys
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import os
import sys

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../HiSim'))
# print(parent_dir)
# p = sys.path.insert(0, parent_dir)

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
target_file_path = os.path.join(parent_dir, 'AI_Networks')

print("target: ", target_file_path)
print("vit: ", f'{target_file_path}/GCN/VIT_base.csv')

def smallest_square_greater_than(n):
    square_root = math.ceil(math.sqrt(n))
    return square_root ** 2

def load_ai_network(aimodel):
    #Load AI network parameters from the network csv files
    if aimodel =='vit':
        network_params = np.loadtxt(f'{target_file_path}/Transformer/VIT_base.csv', dtype=int, delimiter=',')
    elif aimodel =='gcn':
        network_params = np.loadtxt(f'{target_file_path}/GCN/NetWork.csv', dtype=int, delimiter=',')
    elif aimodel=='resnet50':
        network_params = np.loadtxt(f'{target_file_path}/ResNet/50/NetWork.csv', dtype=int, delimiter=',')
    elif aimodel=='resnet110':
        network_params = np.loadtxt(f'{target_file_path}/ResNet/110/NetWork.csv', dtype=int, delimiter=',')
    elif aimodel=='densenet121':    
        network_params = np.loadtxt(f'{target_file_path}/DenseNet_IMG/NetWork_121.csv', dtype=int, delimiter=',')
    elif aimodel=='vgg16':    
        network_params = np.loadtxt(f'{target_file_path}/VGG/VGG16_IMG/NetWork.csv', dtype=int, delimiter=',')
    elif aimodel=='test':    
        network_params = np.loadtxt(f'{target_file_path}/Testing/NetWork.csv', dtype=int, delimiter=',')
    elif aimodel=='roofline':    
        network_params = np.loadtxt(f'{target_file_path}/Testing/NetWork_roofline_3.csv', dtype=int, delimiter=',')
    
    return network_params

def model_mapping(filename,placement_method,network_params,quant_act,xbar_size,N_crossbar,N_pe,quant_weight,N_tile,N_tier,N_stack):
    #---------------------------------------------------------------------#

    #         configuration of the AI models mapped to architecture

    #---------------------------------------------------------------------#
    #Initialize variables
    numComputation=0

    total_number_layers=network_params.shape[0]
    tile_x_bit=xbar_size*math.sqrt(N_crossbar)*math.sqrt(N_pe)   #Total number of bits in a tile in x-dimension
    tile_y_bit=xbar_size*math.sqrt(N_crossbar)*math.sqrt(N_pe)   #Total number of bits in a tile in y-dimension
    util_map_fn=util_map(N_tile=N_tile, placement_method=placement_method, N_tier=N_tier, N_stack=N_stack)

    # write the layer information data to csv file-HISIM Default Mapping
    with open(filename, 'w') as csvfile: 
        writer = csv.writer(csvfile) 
        for layer_idx in range(0, total_number_layers):      

            #Read the parameters from the network file      
            in_x=network_params[layer_idx][0]                           #Size of the input of the Layer in x-dimension
            in_y=network_params[layer_idx][1]                           #Size of the input of the layer in y-dimension
            in_channel=network_params[layer_idx][2]                     #Number of input channels of the layer    
            k_x=network_params[layer_idx][3]                            #Kernel size in x-dimension of the layer
            k_y=network_params[layer_idx][4]                            #Kernel size in y-dimension of the layer
            out_channel=network_params[layer_idx][5]                    #Number of output channels of the layer 
            enable_pooling=network_params[layer_idx][6]                 #Parameter indicating if the layer is followed by pooling or not
            sparsity=1-network_params[layer_idx][7]                     #Total Sparsity of the layer

            #Calculate parameters of the layer
            ip_activation=in_x*in_y*in_channel*quant_act                #Total number of input activations for the layer  
            input_cycle=(in_x-k_x+1)*(in_y-k_y+1)*quant_act             #Number of input cycles for the layer

            #Number of FLOPS of the layer
            numComputation_layer=2*(in_x*in_y*in_channel*k_x*k_y*out_channel)
            numComputation+=numComputation_layer
            
            # Mapping according to HISIM Default Mapping
            layer_num_tile=math.ceil(in_channel*k_x*k_y/tile_x_bit)*math.ceil(out_channel*quant_weight/tile_y_bit)  #Number of tiles required for the layer
            layer_num_crossbar=math.ceil(in_channel*k_x*k_y/xbar_size)*math.ceil(out_channel*quant_weight/xbar_size)#Number of PEs required for the layer
            n_c_x=math.ceil(in_channel*k_x*k_y/xbar_size)               #Number of rows of PEs for the layer
            n_c_y=math.ceil(out_channel*quant_weight/xbar_size)         #Number of columns of PEs for the layer

            err_result = util_map_fn.forward(layer_num_tile, layer_idx)

            # None means didn't error out
            if err_result is not None:
                return err_result

            #Hardware utilization related parameters for the layer
            total_bit=n_c_x*n_c_y*xbar_size*xbar_size                   #Total number of bits in the mapped PEs for the layer
            total_bit_real=in_channel*k_x*k_y*out_channel*quant_weight  #Total number of weight bits for the layer
            utilization=total_bit_real/total_bit                        #Cell Bit Utilization for the layer
            util_row=out_channel*quant_weight/(n_c_y*xbar_size)         #Average Utilization of a row for the layer
            util_col=in_channel*k_x*k_y/(n_c_x*xbar_size)               #Average Utilization of a column for the layer

            # CSV file is written in the following format:
            #0-layer index, 
            #1-Number of tiles required for the layer, 
            #2-Number of PEs required for the layer , 
            #3-Number of rows of PEs for the layer,
            #4-Number of columns of PEs for the layer, 
            #5-Number of input cycles for the layer, 
            #6-pooling,
            #7-Number of tiles mapped uptill this layer, 
            #8-Total number of input activations for the layer, 
            #9-Tier/chiplet index that the layer is mapped to for this layer,
            #10-Cell Bit Utilization for the layer,
            #11-Average Utilization of a row for the layer,
            #12-Total number of weight bits for the layer,
            #13-Average Utilization of a column for the layer,
            #14-Number of FLOPS of the layer 
            csvfile.write(str(layer_idx)+","+str(layer_num_tile)+","+str(layer_num_crossbar)+","+str(n_c_x)+","+str(n_c_y)+","+str(input_cycle)+","+str(enable_pooling)+","+str(util_map_fn.total_tiles_real)+","+str(ip_activation)+","+str(util_map_fn.tier_index)+","+str(utilization)+","+str(util_row)+","+str(total_bit_real)+","+str(util_col)+","+str(numComputation_layer)+","+str(util_map_fn.stack_index))
            csvfile.write('\n')
    #import pdb;pdb.set_trace()

    return util_map_fn.total_tiles_real, util_map_fn.tiles_each_tier, util_map_fn.tiles_each_stack

class util_map():
    def __init__(self, N_tile,placement_method, N_tier, N_stack):
        super(util_map, self).__init__()
        self.N_tile = N_tile                                            
        self.placement_method=placement_method                          
        self.N_tier=N_tier     
        self.N_stack=N_stack                                              
        self.tiles_each_tier=[[0]*N_tier]  
        self.tiles_each_stack=[0]                            

        #Initialize variables
        self.total_tiles_required=0
        self.total_tiles_real=0                                              
        self.tier_index=0
        self.tier_index=0
        self.stack_index=0
        self.tiles_cumm_stack=0
        self.cumm_layer_stack=0
    
    def forward(self, layer_num_tile, layer_idx):

        #Check if the required number of tiles for this layer are greater than the user-defined number of tiles in a tier/chiplet
        if layer_num_tile>self.N_tile:
            print("Alert!!!","layer",layer_idx,"mapped to multiple chiplet/tier")
            print("please increase crossbar size, PE number, or tile number")
            # sys.exit()
            return [f"Alert!!! layer {layer_idx} mapped to multiple chiplet/tier: please increase crossbar size, PE number, or tile number"]
        

        self.total_tiles_required+=layer_num_tile
        if self.placement_method==5:
            #Placement method 5: tile-to-tile 3D connection. 
            #Example mapping for 2 tier architecture
            #layer 0 in tier 0 -> layer 1 in tier 1 -> layer 2 in tier 1 -> layer 3 in tier 0 -> layer 4 in tier 0 -> layer 5 in tier 1 
            tier_index_fac=(layer_idx-self.cumm_layer_stack)%(2*self.N_tier)
            if tier_index_fac > self.N_tier-1:
                #Top tier to bottom tier mapping
                self.tier_index=2*self.N_tier-1-tier_index_fac
            else:
                #Bottom to top tier mapping
                self.tier_index=tier_index_fac

            while True:
                if self.tier_index==0:
                    break
                elif self.tiles_each_tier[self.stack_index][self.tier_index]+layer_num_tile>self.N_tile:
                #print(self.tier_index)
                    self.tier_index-=1
                else:
                    break
            #print(self.tiles_each_tier[self.stack_index][self.tier_index]+layer_num_tile>self.N_tile, self.tiles_each_tier, self.N_tile, layer_num_tile)
            #Check if the number of tiles of a layer cannot fit on the remaining tiles on the corresponding tier/chiplet
            if self.tiles_each_tier[self.stack_index][self.tier_index]+layer_num_tile>self.N_tile:
                self.tiles_cumm_stack+=self.tiles_each_stack[self.stack_index]
                self.cumm_layer_stack=layer_idx
                if self.stack_index<self.N_stack-1:
                    self.stack_index+=1
                    self.tiles_each_stack.append(0)
                    self.tiles_each_tier.append([0]*self.N_tier)
                    self.tier_index=0
                else:
                    print("Alert!!!","No available stacks/tile/tiers")
                    print("please increase number of stacks, tier or tiles")
                    #import pdb;pdb.set_trace()
                    #sys.exit()
                    return["Alert!!! No available tile/tiers: please increase Tiers/tile number"]
            self.tiles_each_tier[self.stack_index][self.tier_index]+=layer_num_tile           #Assign the tiles of the layer to the corresponding tier/chiplet
            self.tiles_each_stack[self.stack_index]+=layer_num_tile
            self.total_tiles_real=self.total_tiles_required                 
            #print("layer_idx", layer_idx, "layer_num_tile", layer_num_tile, "stack_index", self.stack_index, "tier_index", self.tier_index, "tiles_each_tier", self.tiles_each_tier)
                          
        else:
            #Placement method 1: Tier/Chiplet Edge to Tier/Chiplet Edge connection
            total_tiles_required_layer=layer_num_tile
            #Map top tier/chiplet completely before proceeding to next tier/chiplet 
            if self.tiles_each_tier[self.stack_index][self.tier_index]+layer_num_tile>self.N_tile:
                total_tiles_required_layer=self.N_tile-self.tiles_each_tier[self.stack_index][self.tier_index]+layer_num_tile              #Count the total number of tiles required uptil this layer
                if self.tier_index==self.N_tier-1:
                    self.stack_index+=1
                    self.tiles_each_stack.append(0)
                    self.tiles_each_tier.append([0]*self.N_tier)
                    self.tier_index=0
                else:
                    self.tier_index+=1                                    #Increment tier/chiplet index
            self.tiles_each_tier[self.stack_index][self.tier_index]+=layer_num_tile     #Assign the tiles of the layer to the corresponding tier/chiplet
            self.tiles_each_stack[self.stack_index]+= total_tiles_required_layer
            self.total_tiles_real+=total_tiles_required_layer         
            #print("layer_idx", layer_idx, "layer_num_tile", layer_num_tile, "stack_index", self.stack_index, "tier_index", self.tier_index, "tiles_each_tier", self.tiles_each_tier)

def create_tile(ax, x, y, z, dx, dy, dz, color, label):
    xx = [x, x, x+dx, x+dx, x]
    yy = [y, y+dy, y+dy, y, y]
    kwargs = {'alpha': 0.6, 'color': color}
    ax.plot3D(xx, yy, [z]*5, **kwargs)
    ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
    ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
    ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
    ax.text(x + dx/2, y + dy/2, z + dz/2, label, color='black', ha='center', va='center', fontsize=7)

    
