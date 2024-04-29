# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import math
import os
import shutil
import csv
import time
import re
import matplotlib.pyplot as plt
import pickle
import sys
import scipy.sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra
import torch
import collections
from Module_Thermal.util_thermal import *
if not os.path.exists('./Debug/to_interconnect_analy'):
    os.makedirs('./Debug/to_interconnect_analy')
if not os.path.exists('./Results/result_thermal'):
    os.makedirs('./Results/result_thermal')
if os.path.exists('./Results/result_thermal/1stacks'):
    shutil.rmtree('.//Results/result_thermal/1stacks')
os.makedirs('.//Results/result_thermal/1stacks')
#Take all below parameters as argument

xbar_size = 64 # 64,128,256,512,1024
N_tile=25 # 4,9,16,25,36,49 # how many tile in tier (chiplet)
N_tier=2 # 2,3,4,5,6,7,8,9,10 
N_pe=4 # 4,9,16,25 # how many PE in tile
N_crossbar=4 # 4, 9, 16 # how many crossbar in PE
quant_weight=8 # weight quantization bi
quant_act=8 # activation quantization bit
bus_width=64 # in PE and in tile bus width
relu=True
sigmoid=False

freq_computing=1 #GHz
freq_noc=1
#chiplet_size = 9 
#num_chiplets = 144
#type = 'Homogeneous Design'
#scale = 100

## load the network
## here is the ViT as example:
start = time.time()
#network_params = np.loadtxt('./Module_AI_Map/AI_Networks/Transformer/VIT_base.csv', dtype=int, delimiter=',')
#network_params = np.loadtxt('./Module_AI_Map/AI_Networks/GCN/NetWork.csv', dtype=int, delimiter=',')
#network_params = np.loadtxt('./Module_AI_Map/AI_Networks/ResNet/50/NetWork.csv', dtype=int, delimiter=',')
sim_name="GCN_1stack_placement_1"
network_params = np.loadtxt('./Module_AI_Map/AI_Networks/Testing/NetWork.csv', dtype=int, delimiter=',')
total_number_layers=network_params.shape[0]
total_tiles_required=0
total_tiles_real=0 # some tiles are jumped since these are the last several tiles in the tier
tier_index=0
numComputation=0
filename = "./Debug/to_interconnect_analy/layer_inform.csv"

#---------------------------------------------------------------------#

#         configuration of the AI models mapped to architecture

#---------------------------------------------------------------------#
placement_method=1 # 1:from the top to the bottom tier
                    # 2: from the bottom to top tier
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
        numComputation+=2*(network_params[layer_idx][0] * network_params[layer_idx][1] * network_params[layer_idx][2] * network_params[layer_idx][3] * network_params[layer_idx] * network_params[layer_idx])
        # for this layer, calculate how many crossbar/tiles to map the weight
        tile_x_bit=xbar_size*math.sqrt(N_crossbar)*math.sqrt(N_pe)
        tile_y_bit=xbar_size*math.sqrt(N_crossbar)*math.sqrt(N_pe)
        layer_num_tile=math.ceil(in_channel*k_x*k_y/tile_x_bit)*math.ceil(out_channel*quant_weight/tile_y_bit)
        layer_num_crossbar=math.ceil(in_channel*k_x*k_y/xbar_size)*math.ceil(out_channel*quant_weight/xbar_size)

        
        #print(in_channel*k_x*k_y/xbar_size,out_channel*quant_weight/xbar_size)
        #print(layer_num_tile)

        n_c_x=math.ceil(in_channel*k_x*k_y/xbar_size) #number of crossbar r
        n_c_y=math.ceil(out_channel*quant_weight/xbar_size)# number of crossbar c

        if layer_num_tile>N_tile:
            print("Alert!!!","layer",layer_idx,"mapped to multiple chiplet/tier")
            print("please increase crossbar size, PE number, or tile number")
            sys.exit()
            # how many extra chiplet

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
            
        csvfile.write(str(layer_idx)+","+str(layer_num_tile)+","+str(layer_num_crossbar)+","+str(n_c_x)+","+str(n_c_y)+","+str(input_cycle)+","+str(enable_pooling)+","+str(total_tiles_real)+","+str(ip_activation)+","+str(tier_index))
        csvfile.write('\n')
       
        # for computing unit power/latency computing
        # area easy based on the tile size
        # latency :each layer ->one tile-> one PE ->subarray latency *vector



#---------------------------------------------------------------------#

#                         IMC computing units

#---------------------------------------------------------------------#

#
# area,latency,power
# every layer-> tile number, input vector

#---------------------------------------------------------------------#
computing_inform = "./Debug/to_interconnect_analy/layer_inform.csv"
computing_data = pd.read_csv(computing_inform, header=None)
computing_data = computing_data.to_numpy()
total_model_L=0
total_model_E_dynamic=0
total_leakage=0

layer_idx=0

filename = "./Debug/to_interconnect_analy/layer_performance.csv"
    # writing to csv file  
with open(filename, 'w') as csvfile1: 
    writer_performance = csv.writer(csvfile1) 
    for layer_idx in range(len(computing_data)):
        # single crossbar
        L_single_crossbar=(7e-14*xbar_size*xbar_size-7e-13*xbar_size+1E-9)*computing_data[layer_idx][5] # unit=s
        #L_single_crossbar=(0.2345*math.sqrt(N_pe)*math.sqrt(N_pe) - 0.0329*math.sqrt(N_pe) + 0.1268)*L_single_crossbar
        E_single_crossbar=(5e-17*xbar_size*xbar_size+9e-14*xbar_size+2e-14)*computing_data[layer_idx][5] # unit=j
        #import pdb;pdb.set_trace()
        # single PE
        L_bus=(6E-9+7.56E-8)*xbar_size/64*computing_data[layer_idx][5]/100*64/bus_width*math.sqrt(N_crossbar)
        L_PE=L_single_crossbar+7.36e-08*computing_data[layer_idx][5]/100*xbar_size/64*math.sqrt(N_crossbar)+(1.85E-7+1.25e-8)*computing_data[layer_idx][5]*math.sqrt(N_crossbar)*xbar_size/64/100+L_bus
        print(L_single_crossbar)
        print(E_single_crossbar)

        L_accum=2.74e-09*math.sqrt(N_pe)/2*quant_act*computing_data[layer_idx][5]/100 #unit=s
        if relu==True:
            L_activation=4e-9
        else:
            L_activation=1.64e-8
        L_hTree=2.82e-9*math.sqrt(N_pe)/2*quant_act/8*computing_data[layer_idx][5]/100
        L_buffer=(8.34e-9+9.33e-10)*computing_data[layer_idx][5]/100*quant_act/8
        L_tile=L_PE+L_activation+L_hTree+L_accum+L_buffer

        # chip level
        # if pooling
        L_maxpool=5.812E-09*network_params[layer_idx][0]*network_params[layer_idx][1]/49
        L_accum_chip=3.97e-8*network_params[layer_idx][0]*network_params[layer_idx][1]/30
        if computing_data[layer_idx][1]>1:
            if computing_data[layer_idx][6]==1:
                L_chip=L_tile+L_maxpool+L_accum_chip
            else:
                L_chip=L_tile+L_accum_chip
        else:
            if computing_data[layer_idx][6]==1:
                L_chip=L_tile+L_maxpool
            else:
                L_chip=L_tile
        L_chip=L_chip/freq_computing
        total_model_L+=L_chip
        
        #------------------------------------#
        #          dynamic energy            #
        #------------------------------------#
        
        E_total_crossbar=E_single_crossbar*computing_data[layer_idx][2]
        
        E_PE=computing_data[layer_idx][1]*N_pe*(0.86E-7*computing_data[layer_idx][5]/200+(8.39E-11+2.02E-10)*computing_data[layer_idx][5]/200/64*xbar_size+(7.8E-10+9.8E-9)*computing_data[layer_idx][5]/200*xbar_size/64)
        
        E_tile=computing_data[layer_idx][1]*(5.18E-12*math.sqrt(N_pe)/2*xbar_size/64+(1.39E-09+2.12E-10)*computing_data[layer_idx][5]/200+1.14E-10+2.13E-09*computing_data[layer_idx][5]/200*math.sqrt(N_pe)/2)
        
        E_single_tile=N_pe*E_single_crossbar+N_pe*(0.86E-7*computing_data[layer_idx][5]/200+(8.39E-11+2.02E-10)*computing_data[layer_idx][5]/200/64*xbar_size+(7.8E-10+9.8E-9)*computing_data[layer_idx][5]/200*xbar_size/64)

        
        if computing_data[layer_idx][1]>1:
            if computing_data[layer_idx][6]==1:
                E_dynamic=E_total_crossbar+E_PE+E_tile+2.48e-10*network_params[layer_idx][0]*network_params[layer_idx][1]/49+1.42E-10*computing_data[layer_idx][1]/6
            else:
                E_dynamic=E_total_crossbar+E_PE+E_tile+1.42E-10*computing_data[layer_idx][1]/6
        else:
            if computing_data[layer_idx][6]==1:
                E_dynamic=E_total_crossbar+E_PE+E_tile+2.48e-10*network_params[layer_idx][0]*network_params[layer_idx][1]/49
            else:
                E_dynamic=E_total_crossbar+E_PE+E_tile

        total_model_E_dynamic+=E_dynamic

        #--------------------------#
        #          leakage         #
        #--------------------------#
        leak_single_xbar=4e-7*xbar_size+3e-7
        leak_addtree=1.22016e-05*quant_act/8*N_crossbar/4
        leak_buffer=(2.59739e-05+5.28E-06)*quant_act/8*N_crossbar/4
        leak_PE=(4e-7*xbar_size+3e-7)*N_crossbar+1.22016e-05*quant_act/8*N_crossbar/4+(2.59739e-05+5.28E-06)*quant_act/8*N_crossbar/4
        leak_accum=1.31e-5*math.sqrt(N_pe)/2*xbar_size/64
        leak_buffer_tile=4.63e-5*quant_act/8*N_pe/4*xbar_size/64
        leak_tile=(leak_PE*N_pe+leak_accum+leak_buffer_tile)
        total_leakage+=leak_tile*L_chip*computing_data[layer_idx][1]

        # layer index, number of tiles per layer, latency of this layer, E of this layer, leak of this layer, power of this layer each tile
        csvfile1.write(str(layer_idx)+","+str(computing_data[layer_idx][1])+","+str(L_chip)+","+str(E_dynamic)+","+str(leak_tile)+","+str('%.3f'% (E_dynamic/L_chip*1000/computing_data[layer_idx][1])))
        csvfile1.write('\n')


print("----------computing performance---------------------")
print("latency",total_model_L*pow(10,9),"ns")
print("dynamic energy",total_model_E_dynamic*pow(10,12),"pJ")
print("leakage energy",total_leakage*pow(10,12),"pJ")

        #-----------------------------------#
        #         Computing Area            #
        #-----------------------------------#
# single tile area
area_single_tile=33638.9/3*xbar_size/64*N_pe*10e-6 #mm2
if total_tiles_real%N_tile!=0:
    N_tier_real=int(total_tiles_real//N_tile)+1 # num of chiplet
else:
    N_tier_real=int(total_tiles_real//N_tile)

if N_tier_real>4:
    print("Alert!!! too many number of tiers")
    sys.exit()

total_tiles_area=N_tier_real*N_tile*area_single_tile
print("total_tiles_area",total_tiles_area,"mm2")
print("every tier tiles total area,",total_tiles_area/N_tier_real,"mm2")

# total tier(chiplet) number
end_computing = time.time()
print("The computing unit sim time is:", (end_computing - start))
print("----------------------------------------------------")
print('\n')
#
#---------------------------------------------------------------------#

#                         Network:NoC,3D NoC

#---------------------------------------------------------------------#

# Network,3D NoC
# area,latency,power

#---------------------------------------------------------------------#
chiplet_num=N_tier_real
mesh_edge=int(math.sqrt(N_tile))

total_tile=0
layer_start_tile=0
tile_total=[]

for layer_index in range(len(computing_data)):
    if computing_data[layer_index][9]>1 and computing_data[layer_index-1][9]!=computing_data[layer_index][9]:
        layer_start_tile=0
    # get this layer information 
    
    layer_end_tile=layer_start_tile+computing_data[layer_index][1]-1

    tile_index = np.array([[0,0,0]])

    for layer_tile_number in range(layer_start_tile,layer_end_tile+1):
        
        x_idx= int((layer_tile_number)//(mesh_edge))
        #print("x_idx",x_idx)
        y_idx= int((layer_tile_number)%(mesh_edge))
        #print("y_idx",y_idx)

        tile_index = np.append(tile_index, [[x_idx, y_idx, computing_data[layer_index][9]]],axis=0)
    
    tile_index=tile_index[1:]

    each_tile_activation_Q=int(computing_data[layer_index][8]/computing_data[layer_index][1])
    
    tile_index= np.append(tile_index,[[each_tile_activation_Q,each_tile_activation_Q,each_tile_activation_Q]],axis=0)

    tile_total.append(tile_index)
    layer_start_tile=layer_end_tile+1

empty_tile_total=[]
for tier_index in range(chiplet_num):
    tile_index = np.array([[0,0,0]])
    for x in range(int(math.sqrt(N_tile))):
        for y in range(int(math.sqrt(N_tile))):
            tile_index = np.append(tile_index, [[x, y, tier_index]],axis=0)
    tile_index=tile_index[1:]
    empty_tile_total.append(tile_index)

hop2d=0
hop3d=0
Q_3d=0
Q_2d=0
layer_Q=[]
layer_HOP_2d=[]
layer_HOP_3d=[]
# counting total 2d hop and 3d hop
routing_method=1   #: local routing-> only use the routers and tsvs nearby
#routing method 2: global routing-> in the global routing, data will try to use all the routers to transport to next tier
for i in range(len(tile_total)-1):
    #print(tile_total[i])
    #print(tile_total[i+1])
    num_tiles_this_layer=len(tile_total[i]-1)
    num_tiles_left_this_layer=N_tile-num_tiles_this_layer
    Q_3d_scatter=tile_total[i][-1][2]*num_tiles_this_layer/N_tile
    
    layer_2d_hop=hop2d
    layer_3d_hop=hop3d
    
    if routing_method==1:
        for x in range(len(tile_total[i])-1):
            for y in range(len(tile_total[i+1])-1):
                hop2d+=abs(tile_total[i][x][0]-tile_total[i+1][y][0])+abs(tile_total[i][x][1]-tile_total[i+1][y][1])+1
                hop3d+=abs(tile_total[i][x][2]-tile_total[i+1][y][2])
        layer_HOP_2d.append(hop2d-layer_2d_hop)
        layer_HOP_3d.append(hop3d-layer_3d_hop)
        if tile_total[i][0][2]!=tile_total[i+1][0][2]:
            Q_3d+=(tile_total[i][-1][2])*(len(tile_total[i+1])-1)
            layer_Q.append((tile_total[i][-1][2])*(len(tile_total[i])-1))
            Q_2d+=tile_total[i][-1][2]*(len(tile_total[i+1])-1)
        else:
            Q_2d+=tile_total[i][-1][2]*(len(tile_total[i+1])-1)
            layer_Q.append(tile_total[i][-1][2]*(len(tile_total[i])-1))
    
    elif routing_method==2:
        for x in range(len(tile_total[i])-1):
            for y in range(len(tile_total[i+1])-1):
                hop2d+=abs(tile_total[i][x][0]-tile_total[i+1][y][0])+abs(tile_total[i][x][1]-tile_total[i+1][y][1])+1
                hop3d+=(abs(tile_total[i][x][2]-tile_total[i+1][y][2]))*N_tile
        
        layer_HOP_2d.append(hop2d-layer_2d_hop)
        layer_HOP_3d.append(hop3d-layer_3d_hop)

        if tile_total[i][0][2]!=tile_total[i+1][0][2]:
            for x in range(len(tile_total[i])-1):
                for y in range(len(empty_tile_total[tile_total[i][x][2]])):
                    hop2d+=(abs(tile_total[i][x][0]-empty_tile_total[tile_total[i][x][2]][y][0])+abs(tile_total[i][x][1]-empty_tile_total[tile_total[i][x][2]][y][1])+1)*2
            Q_3d+=int((tile_total[i][-1][2])*(len(tile_total[i+1])-1)/N_tile)
            layer_Q.append((tile_total[i][-1][2])*(len(tile_total[i])-1))
            Q_2d+=tile_total[i][-1][2]*(len(tile_total[i+1])-1)
        else:
            Q_2d+=tile_total[i][-1][2]*(len(tile_total[i+1])-1)
            layer_Q.append(tile_total[i][-1][2]*(len(tile_total[i])-1))

print("Total Q bits for 2d communication:", Q_2d)
print("Total Q bits for 3d communication:", Q_3d)
print("Total HOP for 2d communication:", hop2d)
print("Total HOP for 3d communication:", hop3d)


#------------bandwidth-------------------#
# 2D noc
# fix as 32
# 3D tsv
W2d=32 # this is the bandwidth of 2d
W3d_assume=32
tsvPitch = 10; #um
single_router_area=(1e-6*(4/5*W2d+1/5*W3d_assume)*(4/5*W2d+1/5*W3d_assume)+5e-5*(4/5*W2d+1/5*W3d_assume)+0.0005)
edge_single_router=math.sqrt(single_router_area)
edge_single_tile=math.sqrt(area_single_tile)

print("--------------area report---------------")
print("single tile area",area_single_tile,"mm2")
print("single router area",single_router_area,"mm2")

print("edge_single_router",edge_single_router) #mm
print("edge_single_tile",edge_single_tile) #mm
print("----------------------------------------")
num_tsv_io=int(edge_single_router/tsvPitch*1000)*int(edge_single_router/tsvPitch*1000)*2
W3d=num_tsv_io
# Router technology delay 
trc=1
tva=1
tsa=1
tst=1
tl=1
tenq=1
fclk_noc=1
working_channel=2 # last layer source to dst paths
#3d noc edges links
links_topology_2d=(math.sqrt(N_tile)-1)*math.sqrt(N_tile)*2*chiplet_num
links_topology_3d=N_tile*(chiplet_num-1)
# 2.1 latency of booksim
L_booksim=hop2d*(trc+tva+tsa+tst+tl)+(tenq+2/3+0.3334)*(Q_2d/W2d)+hop3d*(trc+tva+tsa+tst+tl)+(tenq+2/3+0.3334)*(Q_3d/W3d)/fclk_noc

# 2.2 power of booksim
layer_index=0
tier_2d_hop_list=[]
tier_3d_hop_list=[]
tier_total_2d_hop=0
tier_total_3d_hop=0
num_layer=0
num_layer_list=[]
for i in range(chiplet_num):
    for layer_index in range(len(tile_total)-1):
        if computing_data[layer_index][9]==i :
            tier_total_2d_hop+=layer_HOP_2d[layer_index]
            tier_total_3d_hop+=layer_HOP_3d[layer_index]
        num_layer+=1
    tier_2d_hop_list.append(tier_total_2d_hop/(num_layer)/N_tile)
    tier_3d_hop_list.append(tier_total_3d_hop)
    tier_total_2d_hop=0
    tier_total_3d_hop=0
    num_layer=0

single_tsv_channel_power=1.3558*W3d/32
single_tsv_router_power=1.5657*W3d/32
single_2d_channel_power=4.7312*W2d/32
single_2d_router_power=2.6883*W2d/32
tier_2d_hop_list[-1]=tier_2d_hop_list[-2]
tier_3d_hop_list[-1]=tier_3d_hop_list[-2]
tier_2d_hop_list_power=[i * single_2d_router_power/i*0 for i in tier_2d_hop_list]
tier_3d_hop_list_power=[i * single_tsv_channel_power/i*0 for i in tier_3d_hop_list]

print(tier_2d_hop_list_power)
print(tier_3d_hop_list_power)
print("booksim latency", L_booksim)

# 2.3 area of booksim
wire_length_2d=1 #unit=mm\
wire_pitch_2d=0.0045 #unit=mm
Num_routers=N_tile*chiplet_num

Total_area_routers=(single_router_area)*Num_routers
Total_channel_area=wire_length_2d*wire_pitch_2d*W2d
#single_TSV_area=math.sqrt(area_single_tile)*math.sqrt((1e-6*(4/5*W2d+1/5*W3d)*(4/5*W2d+1/5*W3d)+5e-5*(4/5*W2d+1/5*W3d)+0.0005))
print("computing latency",total_model_L*pow(10,9),"ns")
print("NoC latency",L_booksim,"ns")
print("total system latency", L_booksim+total_model_L*pow(10,9))
end_noc = time.time()
print("The noc sim time is:", (end_noc - end_computing))


#import pdb;pdb.set_trace()


#---------------------------------------------------------------------#

#                 Thermal results (based on power,area)

#---------------------------------------------------------------------#


# np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=50_000)
# cu =   398 w/mk
# air = 24.3 mw/mk
# sink 4mm,2mm, 6mm 2 times chip  3times chip
#
tiers_of_stacks=chiplet_num
tiles_edges_in_tier=int(math.sqrt(N_tile))
#====================================================================================================
# mw ,later in load_power will be converted to w
#====================================================================================================
case_=4
alpha=1.2

#for i in range(chiplet_num):
##
power_router={}
power_tsv={}
for i in range(len(tier_2d_hop_list_power)):
    if placement_method==1:
        power_router[i] = tier_2d_hop_list_power[i]
    elif placement_method==2:
        power_router[len(tier_2d_hop_list_power)-i-1]=tier_2d_hop_list_power[i]
for i in range(len(tier_3d_hop_list_power)):
    if placement_method==1:
        power_tsv[i] = tier_3d_hop_list_power[i]
    elif placement_method==2:
        power_tsv[len(tier_3d_hop_list_power)-i-1] = tier_3d_hop_list_power[i]

#import pdb;pdb.set_trace()

#====================================================================================================
# w/mk
#====================================================================================================
dict_k = dict()
dict_k['k_imc_0']  = 110/alpha
dict_k['k_imc_1']= 142.8/alpha
dict_k['k_imc_2']    = 4/alpha
dict_k['k_r_0']    = 110/alpha
dict_k['k_r_1']  = 142.8/alpha
dict_k['k_r_2']      = 4/alpha
dict_k['k_tsv_0']= 110/alpha
dict_k['k_tsv_1']  = 142.8/alpha
dict_k['k_tsv_2']  = 4/alpha
dict_k['cu']       = 398/alpha
dict_k['air']   = 0.0243/alpha
dict_k['subs']   = 142.8/alpha

#====================================================================================================
# m
#====================================================================================================
imc_size = math.sqrt(0.0001)/1000
r_size   = imc_size


#====================================================================================================
# imc_size = 0.00085
# r_size = 0.00085
#====================================================================================================
#====================================================================================================

tsv0_length = r_size
tsv0_width  = imc_size
tsv1_length = imc_size
tsv1_width  = r_size


dict_z=dict()
#====================================================================================================
# set to mm
#====================================================================================================
# it will be later converted to m
#====================================================================================================
dict_z['heatsink']=40
dict_z['heatspread']=20
dict_z['device']=(0.002, 0.1, 0.02)
# die thickness (0.1-0.5)
# tim thickness (0.01-0.05)
dict_z['subs']=1
dict_z['air']=50
heatsinkair_resoluation=0.5
#====================================================================================================
dict_size = dict()
dict_size["imc"]  = (imc_size,imc_size)
dict_size["r"]    = (r_size,r_size)
dict_size["tsv0"] = (tsv0_length,tsv0_width)
dict_size["tsv1"] = (tsv1_length,tsv1_width)

#====================================================================================================
devicemap = collections.defaultdict(list)

#====================================================================================================

#====================================================================================================
#devicemap['3stacks'].append([(0, 5, 'heatsink')])
#devicemap['3stacks'].append([(0, 5, 'heatspread')])
#devicemap['3stacks'].append([(0, 1,'device'), (1,2,'air'), (2, 3, 'device' ),(3,4,'air'), (4, 5, 'device' )])
###devicemap['3stacks'].append([(0, 1,'device'), (1,2,'air'), (2, 3, 'device' ),(3,4,'air'), (4, 5, 'device' )])
#devicemap['3stacks'].append([(0, 5, 'subs')])
#devicemap['3stacks'].append([(0, 5, 'air')])
#====================================================================================================
#devicemap['2stacks'].append([(0, 3, 'heatsink')])
#devicemap['2stacks'].append([(0, 3, 'heatspread')])
#devicemap['2stacks'].append([(0, 1,'device'), (1,2,'air'), (2, 3, 'device' )])
#devicemap['2stacks'].append([(0, 1,'device'), (1,2,'air'), (2, 3, 'device' )])
##devicemap['2stacks'].append([(0, 1,'device'), (1,2,'air'), (2, 3, 'device' )])
##devicemap['2stacks'].append([(0, 3, 'subs')])
#devicemap['2stacks'].append([(0, 3, 'air')])
###====================================================================================================
devicemap['1stacks'].append([(0, 1, 'heatsink')])
devicemap['1stacks'].append([(0, 1, 'heatspread')])
for i in range(chiplet_num):
    devicemap['1stacks'].append([(0, 1,'device')])
devicemap['1stacks'].append([(0, 1, 'subs')])
devicemap['1stacks'].append([(0, 1, 'air')])
###====================================================================================================
#devicemap['0stacks'].append([(0, 6, 'heatsink')])
#devicemap['0stacks'].append([(0, 6, 'heatspread' )])
#devicemap['0stacks'].append([(0, 1, 'device' ),(1, 2, 'device' ),(2, 3, 'device' ),(3, 4, 'device' ),(4, 5, 'device' ),(5, 6, 'device' ),  ])
#devicemap['0stacks'].append([(0, 6, 'subs' )])
#devicemap['0stacks'].append([(0, 6, 'air' )])
#====================================================================================================
numofdevicelayer = dict()
#numofdevicelayer['0stacks']= 1
numofdevicelayer['1stacks']= chiplet_num
#numofdevicelayer['2stacks']= 3
##numofdevicelayer['3stacks']= 2
#====================================================================================================




devicemap_sanitycheck(devicemap)
xdim,_                                                                   = get_unitsize(dict_size,tiles_edges_in_tier)
cube_geo_dict, cube_k_dict, cube_z_dict, cube_n_dict,cube_layertype_dict = create_cube(dict_size, dict_z, dict_k,  xdim , devicemap,heatsinkair_resoluation,tiles_edges_in_tier)
cube_power_dict                                                          = load_power(case_,dict_z, devicemap, cube_n_dict, power_tsv, power_router,numofdevicelayer,cube_layertype_dict,tiles_edges_in_tier,chiplet_num,placement_method)
cube_G_dict                                                              = get_conductance_G_new(cube_geo_dict, cube_k_dict, cube_z_dict)
cube_T_dict                                                              = solver(cube_G_dict, cube_n_dict,cube_power_dict,cube_layertype_dict,xdim,sim_name )
end_thermal = time.time()
print("The noc sim time is:", (end_thermal - end_noc))
print("whole sim time",(end_thermal-start))