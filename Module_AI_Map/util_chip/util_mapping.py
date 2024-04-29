import csv,math,sys

def smallest_square_greater_than(n):
    square_root = math.ceil(math.sqrt(n))
    return square_root ** 2


def model_mapping(filename,placement_method,total_number_layers,network_params,quant_act,xbar_size,N_crossbar,N_pe,quant_weight,N_tile,N_tier,tiles_each_tier):
    #---------------------------------------------------------------------#

    #         configuration of the AI models mapped to architecture

    #---------------------------------------------------------------------#

    numComputation=0
    total_tiles_required=0
    total_tiles_real=0 # some tiles are jumped since these are the last several tiles in the tier
    tier_index=0
    with open(filename, 'w') as csvfile: 
        writer = csv.writer(csvfile) 
        for layer_idx in range(0, total_number_layers):            
            params_row = network_params[layer_idx]
            in_x=network_params[layer_idx][0]
            in_y=network_params[layer_idx][1]
            in_channel=network_params[layer_idx][2]
            k_x=network_params[layer_idx][3]
            k_y=network_params[layer_idx][4]
            out_channel=network_params[layer_idx][5]
            enable_pooling=network_params[layer_idx][6]
            sparsity=1-network_params[layer_idx][7]

            ip_activation=in_x*in_y*in_channel*quant_act
            input_cycle=(in_x-k_x+1)*(in_y-k_y+1)*quant_act
            numComputation_layer=2*(network_params[layer_idx][0] * network_params[layer_idx][1] * network_params[layer_idx][2] * network_params[layer_idx][3] * network_params[layer_idx][4] * network_params[layer_idx][5])
            numComputation+=numComputation_layer
            # for this layer, calculate how many crossbar/tiles to map the weight
            tile_x_bit=xbar_size*math.sqrt(N_crossbar)*math.sqrt(N_pe)
            tile_y_bit=xbar_size*math.sqrt(N_crossbar)*math.sqrt(N_pe)
            layer_num_tile=math.ceil(in_channel*k_x*k_y/tile_x_bit)*math.ceil(out_channel*quant_weight/tile_y_bit)
            layer_num_crossbar=math.ceil(in_channel*k_x*k_y/xbar_size)*math.ceil(out_channel*quant_weight/xbar_size)

            
            #print(in_channel*k_x*k_y/xbar_size,out_channel*quant_weight/xbar_size)
            #print(layer_num_tile)

            n_c_x=math.ceil(in_channel*k_x*k_y/xbar_size) #number of crossbar r
            n_c_y=math.ceil(out_channel*quant_weight/xbar_size)# number of crossbar c

            total_bit=n_c_x*n_c_y*xbar_size*xbar_size
            total_bit_real=in_channel*k_x*k_y*out_channel*quant_weight
            utilization=total_bit_real/total_bit
            util_row=out_channel*quant_weight/(n_c_y*xbar_size)
            util_col=in_channel*k_x*k_y/(n_c_x*xbar_size)
            if layer_num_tile>N_tile:
                print("Alert!!!","layer",layer_idx,"mapped to multiple chiplet/tier")
                print("please increase crossbar size, PE number, or tile number")
                sys.exit()
                # how many extra chiplet
            if placement_method==5:
                tier_index_fac=layer_idx%(2*N_tier)
                if tier_index_fac > N_tier-1:
                    tier_index=2*N_tier-1-tier_index_fac
                else:
                    tier_index=tier_index_fac
                tiles_each_tier[tier_index]+=layer_num_tile
                total_tiles_required+=layer_num_tile
                if total_tiles_required>N_tile*N_tier or layer_num_tile>N_tile-tiles_each_tier[tier_index]:
                    #import pdb;pdb.set_trace()
                    print("Alert!!!","No available tile/tiers")
                    print("please increase Tiers/tile number")
                    sys.exit()
                total_tiles_real=total_tiles_required
                
            else:
                if total_tiles_required%N_tile==0 and total_tiles_required//N_tile!=0:
                    tier_index+=1
                total_tiles_required+=layer_num_tile
                
                # the layer tile won't be across two chiplet(tier)
                if total_tiles_required%(N_tile*(tier_index+1))<layer_num_tile :
                    if total_tiles_required%(N_tile*(tier_index+1))==0:
                        total_tiles_real=total_tiles_required   
                    else:
                        total_tiles_real=N_tile*(tier_index+1)+layer_num_tile
                        total_tiles_required=total_tiles_real
                        tier_index+=1
                else:
                    total_tiles_real=total_tiles_required

                # creating a csv writer object  
            #import pdb;pdb.set_trace()
                
            csvfile.write(str(layer_idx)+","+str(layer_num_tile)+","+str(layer_num_crossbar)+","+str(n_c_x)+","+str(n_c_y)+","+str(input_cycle)+","+str(enable_pooling)+","+str(total_tiles_real)+","+str(ip_activation)+","+str(tier_index)+","+str(utilization)+","+str(util_row)+","+str(total_bit_real)+","+str(util_col)+","+str(numComputation_layer))
            csvfile.write('\n')
        
            # for computing unit power/latency computing
            # area easy based on the tile size
            # latency :each layer ->one tile-> one PE ->subarray latency *vector
    #print(numComputation)
    return total_tiles_real