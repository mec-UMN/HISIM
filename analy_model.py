# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import math
import os
import shutil
import csv
import time
import argparse
import re
import matplotlib.pyplot as plt
import pickle
import sys
import scipy.sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra
import torch
import collections
from Module_Compute.functions import imc_analy
from Module_Thermal.util import *
from Module_Thermal.H2_5D_thermal import *
from Module_Network.orion_power_area import power_summary_router
from Module_AI_Map.util_chip.util_mapping import model_mapping, smallest_square_greater_than
from Module_Network.aib_2_5d import  aib
from itertools import chain
if not os.path.exists('./Debug/to_interconnect_analy'):
    os.makedirs('./Debug/to_interconnect_analy')
if not os.path.exists('./Results/result_thermal'):
    os.makedirs('./Results/result_thermal')
if os.path.exists('./Results/result_thermal/1stacks'):
    shutil.rmtree('.//Results/result_thermal/1stacks')
if not os.path.exists('./Results'):
    os.makedirs('./Results')
os.makedirs('.//Results/result_thermal/1stacks')

#---------------------------------------------------------------------#

#         results lists

#---------------------------------------------------------------------#
result_list=[]
parser = argparse.ArgumentParser(description='Design Space Search',
								 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--chip_architect', type=str, default="M3D",help='hardware architecture:M3D,M2D,H2_5D,H2_5D_3D')
parser.add_argument('--xbar_size', type=int, default=512,help='crossbar size')
parser.add_argument('--N_tile', type=int, default=324,help='how many tiles in tier')
parser.add_argument('--N_pe', type=int, default=16,help='how many PEs in tile')
parser.add_argument('--freq_computing', type=float, default=1,help='Computing unit operation frequency')
parser.add_argument('--fclk_noc', type=float, default=1,help='network data communication operation frequency')
parser.add_argument('--tsvPitch', type=float, default=10,help='TSV pitch um')
parser.add_argument('--N_tier', type=int, default=4,help='how many tiers')
parser.add_argument('--volt', type=int, default=0.5,help='Operating Voltage in volt')
parser.add_argument('--placement_method', type=int, default=5,help='computing tile placement method')
parser.add_argument('--percent_router', type=float, default=0.5,help='when data route from one tier to next tier, the system will choose how much percent routers for 3D communication')
parser.add_argument('--no_compute_validate', action='store_false',help='mode to valiate the compute model with neurosim')
parser.add_argument('--W2d', type=int, default=32,help='Number of links of 2D NoC')
parser.add_argument('--router_times_scale', type=int, default=1,help='Scaling factor for time components of router: trc, tva, tsa, tst,tl, tenq')

#Take all below parameters as argument
args = parser.parse_args()

xbar_size = args.xbar_size # 64,128,256,512,1024
N_tile=args.N_tile # 4,9,16,25,36,49 # how many tile in tier (chiplet)
N_tier=args.N_tier # 2,3,4,5,6,7,8,9,10 
N_pe=args.N_pe # 4,9,16,25 # how many PE in tile
N_crossbar=1 # 4, 9, 16 # how many crossbar in PE
quant_weight=8 # weight quantization bi
quant_act=8 # activation quantization bit
bus_width=64 # in PE and in tile bus width
chip_architect=args.chip_architect 
COMPUTE_VALIDATE=args.no_compute_validate
placement_method=args.placement_method  # 1: from the top to the bottom tier
                                        # 2: from the bottom to top tier1
                                        # 3: the hotspot far from each other
                                        # 4: worse case:put all hotspot in the same place
                                        # 5: tile-to-tile connection
if chip_architect=="H2_5D":
    placement_method=1
percent_router=args.percent_router
relu=True
sigmoid=False
temp=0 #Temperature to be evaluated
freq_computing=args.freq_computing #GHz
fclk_noc=args.fclk_noc
W2d=args.W2d
volt=args.volt
scale_factor=args.router_times_scale
result_list.append(freq_computing)
result_list.append(fclk_noc)
result_list.append(xbar_size)

result_list.append(N_tile)
result_list.append(N_pe)

#---------------------------------------------------------------------#

#                               Load AI model

#---------------------------------------------------------------------#
start = time.time()
network_params = np.loadtxt('./Module_AI_Map/AI_Networks/Transformer/VIT_base.csv', dtype=int, delimiter=',')
#network_params = np.loadtxt('./Module_AI_Map/AI_Networks/GCN/NetWork.csv', dtype=int, delimiter=',')
#network_params = np.loadtxt('./Module_AI_Map/AI_Networks/ResNet/50/NetWork.csv', dtype=int, delimiter=',')
#network_params = np.loadtxt('./Module_AI_Map/AI_Networks/ResNet/110/NetWork.csv', dtype=int, delimiter=',')
#network_params = np.loadtxt('./Module_AI_Map/AI_Networks/DenseNet_IMG/NetWork_121.csv', dtype=int, delimiter=',')
#network_params = np.loadtxt('./Module_AI_Map/AI_Networks/VGG/VGG16_IMG/NetWork.csv', dtype=int, delimiter=',')
#network_params = np.loadtxt('./Module_AI_Map/AI_Networks/Testing/NetWork.csv', dtype=int, delimiter=',')
#network_params = np.loadtxt('./Module_AI_Map/AI_Networks/Testing/NetWork_roofline_3.csv', dtype=int, delimiter=',')
sim_name="GCN_1stack_placement_1"

total_number_layers=network_params.shape[0]
filename_results = "./Results/PPA.csv"


#---------------------------------------------------------------------#

#         configuration of the AI models mapped to architecture

#---------------------------------------------------------------------#
filename = "./Debug/to_interconnect_analy/layer_inform.csv"
tiles_each_tier = [0]*N_tier
total_tiles_real=model_mapping(filename,placement_method,total_number_layers,network_params,quant_act,xbar_size,N_crossbar,N_pe,quant_weight,N_tile,N_tier,tiles_each_tier)
#import pdb;pdb.set_trace()

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
out_peripherial=[]

layer_idx=0

filename = "./Debug/to_interconnect_analy/layer_performance.csv"
if COMPUTE_VALIDATE:
    freq_adc=0.005
else:
    freq_adc=freq_computing
imc_analy_fn=imc_analy(xbar_size=xbar_size, volt=volt, freq=freq_computing, freq_adc=freq_adc, compute_ref=COMPUTE_VALIDATE, quant_bits=[quant_weight,quant_act])
    # writing to csv file  
with open(filename, 'w') as csvfile1: 
    writer_performance = csv.writer(csvfile1) 
    for layer_idx in range(len(computing_data)):
        A_pe, L_layer, E_layer, peripherials, A_peri = imc_analy_fn.forward(computing_data, layer_idx)
        #------------------------------------#
        #             latency                #
        #------------------------------------#
        #print("layer",layer_idx,L_layer)
        total_model_L+=L_layer
        if COMPUTE_VALIDATE:
            if len(out_peripherial)==0:
                out_peripherial.append(peripherials)
                out_peripherial=list(chain.from_iterable(out_peripherial))
            else:
                for i in range(len(peripherials)):
                    out_peripherial[i]+=peripherials[i]
 

        #------------------------------------#
        #          dynamic energy            #
        #------------------------------------#
        total_model_E_dynamic+=E_layer
        #print("layer",layer_idx,total_model_L,total_model_E_dynamic)

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
        total_leakage+=leak_tile*L_layer*computing_data[layer_idx][1]

        # layer index, number of tiles per layer, latency of this layer, E of this layer, leak of this layer, power of this layer each tile
        csvfile1.write(str(layer_idx)+","+str(computing_data[layer_idx][1])+","+str(L_layer)+","+str(E_layer)+","+str(leak_tile)+","+str('%.3f'% (E_layer/L_layer*1000/computing_data[layer_idx][1])))
        #csvfile1.write(str(layer_idx)+","+str(computing_data[layer_idx][1])+","+str(L_layer)+","+str(E_layer)+","+str(leak_tile)+","+str(pk_power)+","+str(peak_density)+","+str('%.3f'% (E_layer/L_layer*1000/computing_data[layer_idx][1])))
        csvfile1.write('\n')


print("----------computing performance---------------------")
print("latency",total_model_L*pow(10,9),"ns")
print("dynamic energy",total_model_E_dynamic*pow(10,12),"pJ")
print("Overall Compute Power",total_model_E_dynamic/(total_model_L),"W")
print("leakage energy",total_leakage*pow(10,12),"pJ")

        #-----------------------------------#
        #         Computing Area            #
        #-----------------------------------#
# single tile area
if placement_method==5:
   N_tier_real=N_tier
   N_tile_real=smallest_square_greater_than(max(tiles_each_tier))
else:
    N_tile_real=N_tile
    if total_tiles_real%N_tile==0:
        N_tier_real=int(total_tiles_real//N_tile) # num of chiplet
    else:
        N_tier_real=int(total_tiles_real//N_tile)+1 # num of chiplet
result_list.append(N_tile_real)

if N_tier_real>4:
    print("Alert!!! too many number of tiers")
    sys.exit()
#import pdb;pdb.set_trace()
result_list.append(N_tier_real)
result_list.append(total_model_L*pow(10,9))
result_list.append(total_model_E_dynamic*pow(10,12))

area_single_tile=A_pe*N_pe*N_crossbar
total_tiles_area=N_tier_real*N_tile*area_single_tile
print("total_tiles_area",total_tiles_area,"mm2")
print("every tier tiles total area,",total_tiles_area/N_tier_real,"mm2")
result_list.append(total_tiles_area*pow(10,6))
# total tier(chiplet) number
end_computing = time.time()
print("The computing unit sim time is:", (end_computing - start))
print("----------------------------------------------------")
print('\n')

#remapping
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
layer_start_tile_tier=[0]*N_tier
tile_total=[]
# for decide (x,y)
for layer_index in range(len(computing_data)):
    if placement_method ==5:
        if layer_index<N_tier:
            layer_start_tile_tier[int(computing_data[layer_index][9])]=0
        layer_start_tile=layer_start_tile_tier[int(computing_data[layer_index][9])]
    else:
        if computing_data[layer_index][9]>1 and computing_data[layer_index-1][9]!=computing_data[layer_index][9]:
            layer_start_tile=0
    # get this layer information 

    layer_end_tile=layer_start_tile+int(computing_data[layer_index][1])-1

    tile_index = np.array([[0,0,0]])
    #import pdb;pdb.set_trace()
    for layer_tile_number in range(layer_start_tile,layer_end_tile+1):
        
        x_idx= int((layer_tile_number)//(mesh_edge))
        #print("x_idx",x_idx)
        y_idx= int((layer_tile_number)%(mesh_edge))
        #print("y_idx",y_idx)

        tile_index = np.append(tile_index, [[x_idx, y_idx, computing_data[layer_index][9]]],axis=0)
    
    tile_index=tile_index[1:]

    each_tile_activation_Q=int(computing_data[layer_index][8]/computing_data[layer_index][1])
    
    tile_index= np.append(tile_index,[[each_tile_activation_Q,each_tile_activation_Q,each_tile_activation_Q]],axis=0)
    if placement_method==5:
        tile_total.append(tile_index)
        layer_start_tile_tier[int(computing_data[layer_index][9])]=layer_end_tile+1
        #import pdb;pdb.set_trace()
    else:    
        tile_total.append(tile_index)
        layer_start_tile=layer_end_tile+1

empty_tile_total=[]
if placement_method==5:
    for tier_index in range(chiplet_num):
        tile_index = np.array([[0,0,0]])
        for x in range(mesh_edge):
            for y in range(mesh_edge):
                tile_index = np.append(tile_index, [[x, y, tier_index]],axis=0)
        tile_index=tile_index[1:]
        empty_tile_total.append(tile_index)

else:
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
Q_2_5d=0
layer_Q=[]
layer_Q_2_5d=[]
layer_HOP_2d=[]
layer_HOP_3d=[]
# counting total 2d hop and 3d hop
routing_method=2  #: local routing-> only use the routers and tsvs nearby
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
            #Q_2d+=tile_total[i][-1][2]*(len(tile_total[i+1])-1)
        else:
            Q_2d+=tile_total[i][-1][2]*(len(tile_total[i+1])-1)
            layer_Q.append(tile_total[i][-1][2]*(len(tile_total[i])-1))
    
    elif routing_method==2:
        for x in range(len(tile_total[i])-1):
            for y in range(len(tile_total[i+1])-1):
                hop2d+=abs(tile_total[i][x][0]-tile_total[i+1][y][0])+abs(tile_total[i][x][1]-tile_total[i+1][y][1])+1
                hop3d+=(abs(tile_total[i][x][2]-tile_total[i+1][y][2]))*N_tile*percent_router
        
        layer_HOP_2d.append(hop2d-layer_2d_hop)
        layer_HOP_3d.append(hop3d-layer_3d_hop)

        if tile_total[i][0][2]!=tile_total[i+1][0][2]:
            for x in range(len(tile_total[i])-1):
                #import pdb;pdb.set_trace()
                for y in range(int(len(empty_tile_total[int(tile_total[i][x][2])])*percent_router)):
                    hop2d+=(abs(tile_total[i][x][0]-empty_tile_total[int(tile_total[i][x][2])][y][0])+abs(tile_total[i][x][1]-empty_tile_total[int(tile_total[i][x][2])][y][1])+1)*2
            Q_3d+=int((tile_total[i][-1][2])*(len(tile_total[i+1])-1)/(N_tile*percent_router))
            #import pdb;pdb.set_trace()
            layer_Q.append((tile_total[i][-1][2])*(len(tile_total[i])-1))
            Q_2d+=int((tile_total[i][-1][2]*(len(tile_total[i])-1))/(N_tile*percent_router))
            #import pdb;pdb.set_trace()
        else:
            Q_2d+=tile_total[i][-1][2]*(len(tile_total[i+1])-1)
            layer_Q.append(tile_total[i][-1][2]*(len(tile_total[i])-1))

    if tile_total[i][0][2]!=tile_total[i+1][0][2]:
        Q_2_5d+=(tile_total[i][-1][2])*(len(tile_total[i+1])-1)
        layer_Q_2_5d.append((tile_total[i][-1][2])*(len(tile_total[i+1])-1))
print("----------computing performance done--------------------")
print("\n")
print("----------network model--------------------")
print("Total Q bits for 2d communication:", Q_2d)
print("Total HOP for 2d communication:", hop2d)
if chip_architect=="M3D":
    print("Total Q bits for 3d communication:", Q_3d)
    print("Total HOP for 3d communication:", hop3d)
elif chip_architect=='H2_5D':
    print("Total Q bits for 2.5d communication:", Q_2_5d)
print("\n")
#import pdb;pdb.set_trace()

#------------bandwidth-------------------#
# 2D noc
# fix as 32
# 3D tsv
#W2d=32 # this is the bandwidth of 2d
W3d_assume=32
tsvPitch = args.tsvPitch; #um
trc=1*scale_factor
tva=1*scale_factor
tsa=1*scale_factor
tst=1*scale_factor
tl=1*scale_factor
tenq=2*scale_factor

if chip_architect=="M3D" and N_tier_real!=1:
    channel_width=4/5*W2d+1/5*W3d_assume # mix 2d and 3d
    total_router_area,_,_=power_summary_router(channel_width,6,6,hop3d,trc,tva,tsa,tst,tl,tenq,Q_3d,int(chiplet_num),int(mesh_edge))
elif chip_architect=="M2D" or chip_architect=="H2_5D" or N_tier_real==1:
    channel_width=W2d # mix 2d and 3d
    total_router_area,_,_=power_summary_router(channel_width,5,5,hop3d,trc,tva,tsa,tst,tl,tenq,Q_3d,int(chiplet_num),int(mesh_edge))  
single_router_area=total_router_area/(mesh_edge*mesh_edge*chiplet_num)
edge_single_router=math.sqrt(single_router_area)
edge_single_tile=math.sqrt(area_single_tile)
#import pdb;pdb.set_trace()

num_tsv_io=int(edge_single_router/tsvPitch*1000)*int(edge_single_router/tsvPitch*1000)*2
W3d=num_tsv_io
if chip_architect=="M3D" and N_tier_real!=1:
    channel_width=4/5*W2d+1/5*W3d # mix 2d and 3d
    total_router_area,_,_=power_summary_router(channel_width,6,6,hop3d,trc,tva,tsa,tst,tl,tenq,Q_3d,int(chiplet_num),int(mesh_edge))
single_router_area=total_router_area/(mesh_edge*mesh_edge*chiplet_num)
edge_single_router=math.sqrt(single_router_area)
layer_aib_list=[]
if chip_architect=="H2_5D" and N_tier_real!=1:
    aib_out=[0,0,0]
    for i in range(len(layer_Q_2_5d)):
        layer_aib=aib(layer_Q_2_5d[i]*1e-6/8, (edge_single_router+edge_single_tile)*mesh_edge, 1, volt)
        layer_aib_list.append(layer_aib)
        #area- layer_aib_list[idx][0] -mm2, energy- layer_aib_list[idx][1] -pJ, latency-layer_aib_list[idx][2]-ns
        for i in range(len(aib_out)):
            aib_out[i] += layer_aib[i]
    area_2_5d=aib_out[0]
else:
    area_2_5d=0
#import pdb;pdb.set_trace()

print("--------------network area report---------------")
print("single tile area",area_single_tile,"mm2")
print("single router area",single_router_area,"mm2")
print("edge_single_router",edge_single_router) #mm
print("edge_single_tile",edge_single_tile) #mm
print("total 3d stack area",(edge_single_router+edge_single_tile)*(edge_single_router+edge_single_tile)*N_tile)
print("2.5d area", area_2_5d)
print("----------------------------------------")
result_list.append((edge_single_router+edge_single_tile)*(edge_single_router+edge_single_tile)*N_tile+area_2_5d)
result_list.insert(7,W2d)
result_list.insert(8,W3d)
# Router technology delay 

working_channel=2 # last layer source to dst paths
#3d noc edges links
links_topology_2d=(math.sqrt(N_tile)-1)*math.sqrt(N_tile)*2*chiplet_num
links_topology_3d=N_tile*(chiplet_num-1)

# 2.1 latency of booksim
if chip_architect=="M3D" or chip_architect=="M2D" or N_tier_real==1:
    L_booksim=(hop2d*(trc+tva+tsa+tst+tl)+(tenq)*(Q_2d/W2d)+hop3d*(trc+tva+tsa+tst+tl)+(tenq)*(Q_3d/W3d))/fclk_noc
    L_2_5d=0
    L_booksim_2d=(hop2d*(trc+tva+tsa+tst+tl)+(tenq)*(Q_2d/W2d))/fclk_noc
    L_booksim_3d=(hop3d*(trc+tva+tsa+tst+tl)+(tenq)*(Q_3d/W3d))/fclk_noc
   
elif chip_architect=="H2_5D" and  N_tier_real!=1:
    L_booksim_2d=(hop2d*(trc+tva+tsa+tst+tl)+(tenq)*(Q_2d/W2d))/fclk_noc
    L_2_5d=aib_out[2]
    L_booksim_3d=0
    L_booksim=L_booksim_2d+L_2_5d
result_list.append(chip_architect)
result_list.append(L_booksim_2d)
result_list.append(L_booksim_3d)
result_list.append(L_2_5d)
#import pdb;pdb.set_trace()
# 2.2 power of booksim


# ADD VOLTAGE
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

chiplet_num=N_tier_real

#mesh_edge=int(math.sqrt(N_tile))
if chip_architect=="M3D" and N_tier_real!=1:
    _,total_tsv_channel_power,total_3d_router_power=power_summary_router(W3d,6,6,hop3d,trc,tva,tsa,tst,tl,tenq,Q_3d,int(chiplet_num),int(mesh_edge))
    _,total_2d_channel_power,total_2d_router_power=power_summary_router(W2d,5,5,hop2d,trc,tva,tsa,tst,tl,tenq,Q_2d,int(chiplet_num),int(mesh_edge))
elif chip_architect=="M2D" or chip_architect=="H2_5D" or N_tier_real==1:
    _,total_tsv_channel_power,total_3d_router_power=0,0,0
    _,total_2d_channel_power,total_2d_router_power=power_summary_router(W2d,5,5,hop2d,trc,tva,tsa,tst,tl,tenq,Q_2d,int(chiplet_num),int(mesh_edge))
#import pdb;pdb.set_trace()

if chip_architect=="H2_5D" and N_tier_real!=1:
    total_2_5d_channel_power=aib_out[1]/aib_out[2]
else:
    total_2_5d_channel_power=0
# total area of router channel= single_channel_area*(channel number*2+2*number_router)
# total switch+input+output=(switch+input+output)*number_router
#import pdb;pdb.set_trace()
total_router_power=total_3d_router_power+total_2d_router_power+total_2d_channel_power

print("2d",total_router_power)
print("tsv",total_tsv_channel_power)
energy_2d=(total_2d_channel_power+total_2d_router_power)*L_booksim_2d*fclk_noc

energy_3d=(total_tsv_channel_power+total_3d_router_power)*L_booksim_3d*fclk_noc

energy_2_5d=total_2_5d_channel_power*L_2_5d*fclk_noc 

total_energy=energy_2d+energy_2_5d+energy_3d
if len(tier_2d_hop_list)!=1:
    tier_2d_hop_list[-1]=tier_2d_hop_list[-2]
    tier_3d_hop_list[-1]=tier_3d_hop_list[-2]
else:
    tier_3d_hop_list[-1]=total_router_power/chiplet_num
#import pdb;pdb.set_trace()
tier_2d_hop_list_power=[i * total_router_power/chiplet_num/i*fclk_noc for i in tier_2d_hop_list]
if chip_architect=="M3D" and N_tier_real!=1:
    tier_3d_hop_list_power=[i * total_tsv_channel_power/(chiplet_num-1)/i*fclk_noc for i in tier_3d_hop_list]
elif chip_architect=="M2D" or chip_architect=="H2_5D" or N_tier_real==1:
    tier_3d_hop_list_power=[i * 0 for i in tier_3d_hop_list]

print(tier_2d_hop_list_power)
print(tier_3d_hop_list_power)
print("2D NoC W2d",W2d)
print("3D TSV W3d",W3d)
print("network total energy",total_energy,"pJ")
print("network power",(total_router_power+total_tsv_channel_power)*fclk_noc,"mW")
print("NoC latency", L_booksim,"ns")

result_list.append(L_booksim)
result_list.append(energy_2d)
result_list.append(energy_3d)
result_list.append(energy_2_5d) # Pj
result_list.append(total_energy)
# 2.3 area of booksim
wire_length_2d=2 #unit=mm\
wire_pitch_2d=0.0045 #unit=mm
Num_routers=N_tile*chiplet_num

Total_area_routers=(single_router_area)*Num_routers
Total_channel_area=wire_length_2d*wire_pitch_2d*W2d
#single_TSV_area=math.sqrt(area_single_tile)*math.sqrt((1e-6*(4/5*W2d+1/5*W3d)*(4/5*W2d+1/5*W3d)+5e-5*(4/5*W2d+1/5*W3d)+0.0005))
print("computing latency",total_model_L*pow(10,9),"ns")
print("total system latency", L_booksim+total_model_L*pow(10,9))
result_list.append(total_model_L*pow(10,9)/L_booksim)
flops=0
for j in range(len(computing_data)):
    flops+=computing_data[j][14]
result_list.append(flops*pow(10,-3)/(L_booksim+total_model_L*pow(10,9)))
result_list.append(total_model_E_dynamic/total_model_L)
result_list.append((total_router_power+total_tsv_channel_power)*fclk_noc*pow(10,-3))
if area_2_5d!=0:
    result_list.append(total_2_5d_channel_power*pow(10,-3))
else:
    result_list.append(0)
result_list.append(Total_area_routers+Total_channel_area)
end_noc = time.time()
print("\n")
print("-------------------time report--------------------")
print("The noc sim time is:", (end_noc - end_computing))
print("The total sim time is:", (end_noc - start))

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
alpha=3.5
#1.2
#for i in range(chiplet_num):
##
power_router={}
power_tsv={}
for i in range(len(tier_2d_hop_list_power)):
    if placement_method==1 or placement_method==3 or placement_method==4 or placement_method==5:
        power_router[i] = tier_2d_hop_list_power[i]
    elif placement_method==2:
        power_router[len(tier_2d_hop_list_power)-i-1]=tier_2d_hop_list_power[i]
for i in range(len(tier_3d_hop_list_power)):
    if placement_method==1 or placement_method==3 or placement_method==4 or placement_method==5:
        power_tsv[i] = tier_3d_hop_list_power[i]
    elif placement_method==2:
        power_tsv[len(tier_3d_hop_list_power)-i-1] = tier_3d_hop_list_power[i]

#import pdb;pdb.set_trace()

#====================================================================================================
# w/mk
#====================================================================================================
if chip_architect=="M3D":
    dict_k = dict()
    dict_k['k_imc_0']  = 110/alpha
    dict_k['k_imc_1']= 142.8/alpha
    dict_k['k_imc_2']    = 4/alpha
    dict_k['k_r_0']    = 110/alpha
    dict_k['k_r_1']  = 142.8/alpha
    dict_k['k_r_2']      = 4/alpha
    dict_k['k_tsv_0']= 142.8/alpha
    dict_k['k_tsv_1']  = 200/alpha
    dict_k['k_tsv_2']  = 7.9/alpha
    dict_k['cu']       = 398/alpha
    dict_k['air']   = 0.0243/alpha
    dict_k['subs']   = 142.8/alpha
elif chip_architect=="M2D":
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
imc_size = math.sqrt(area_single_tile)/1000
r_size   = math.sqrt(single_router_area)/1000


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

if temp and chip_architect!="H2_5D":


    devicemap_sanitycheck(devicemap)
    xdim,_                                                                   = get_unitsize(dict_size,mesh_edge)
    cube_geo_dict, cube_k_dict, cube_z_dict, cube_n_dict,cube_layertype_dict = create_cube(dict_size, dict_z, dict_k,  xdim , devicemap,heatsinkair_resoluation,mesh_edge)
    cube_power_dict                                                          = load_power(case_,dict_z, devicemap, cube_n_dict, power_tsv, power_router,numofdevicelayer,cube_layertype_dict,mesh_edge,chiplet_num,placement_method,chip_architect)
    cube_G_dict                                                              = get_conductance_G_new(cube_geo_dict, cube_k_dict, cube_z_dict)
    peak_temp                                                              = solver(cube_G_dict, cube_n_dict,cube_power_dict,cube_layertype_dict,xdim,sim_name )

    result_list.append(peak_temp)
    end_thermal = time.time()

elif temp and chip_architect=="H2_5D":

    power_aib_l=[]
    power_emib_l=[]
    area_aib_l=[]
    area_emib_l=[]
    #power_Tx, power_Rx, power_wire
    for i in range(len(layer_aib_list)):
        power_aib_l+=[layer_aib_list[i][6]/layer_aib_list[i][9]+layer_aib_list[i][8], layer_aib_list[i][7]/layer_aib_list[i][10]]
        #power_emib_l+=[layer_aib_list[i][8]]
        area_aib_l+=[layer_aib_list[i][3], layer_aib_list[i][4]]
        area_emib_l+=[layer_aib_list[i][5]]
    # average area of single aib
    if chiplet_num==1:
        area_aib=0
        area_emib=0
    else:
        area_aib=sum(area_aib_l)/len(area_aib_l)
        area_emib=sum(area_emib_l)/len(area_emib_l)

    
    #import pdb;pdb.set_trace()
    case_H2_5D=H2_5D(area_single_tile=area_single_tile,single_router_area=single_router_area,chiplet_num=chiplet_num,mesh_edge=mesh_edge,area_aib=area_aib,area_emib=area_emib,resolution=2)
    
    power_tier_l=power_tile_reorg(mesh_edge)
    
    #import pdb;pdb.set_trace()
    #power_tier_l   = [20]*case_H2_5D.Nstructure*case_H2_5D.N*case_H2_5D.N
    power_router_l = tier_2d_hop_list_power#[5]*case_H2_5D.Nstructure
    if case_H2_5D.Nstructure   == 2: numofaib = 2;  numofemib = 1
    elif case_H2_5D.Nstructure == 3: numofaib = 4;  numofemib = 2
    elif case_H2_5D.Nstructure == 4: numofaib = 6;  numofemib = 4
    elif case_H2_5D.Nstructure == 1: numofaib = 0;  numofemib = 0

    
    #area- layer_aib_list[idx][0] -mm2, energy- layer_aib_list[idx][1] -pJ, latency-layer_aib_list[idx][2]-ns
    #power_aib_l    = [5]*numofaib
    power_emib_l   = [0.0]*numofemib
    case_H2_5D.input_sanity_check()
    #import pdb;pdb.set_trace()
    full_k, full_p, grid_size, all_height_l, all_z_count_l= case_H2_5D.create_global_structure(power_tier_l=power_tier_l,power_router_l=power_router_l,power_aib_l=power_aib_l,power_emib_l=power_emib_l)
    
    currfull_k, currfull_p, curr_grid_size = case_H2_5D.subdivide(full_k, full_p, grid_size)
    G_sparse = case_H2_5D.get_conductance_G_new(currfull_k, curr_grid_size, all_height_l, all_z_count_l)
    peak_temp=case_H2_5D.solver(G_sparse, currfull_p,  all_z_count_l,curr_grid_size)
    result_list.append(peak_temp)
    end_thermal = time.time()

else:
    result_list.append('NA')
result_list.append(placement_method)
result_list.append(percent_router)



if COMPUTE_VALIDATE:
    for i in range(len(out_peripherial)-4):
        result_list.append(out_peripherial[i])

    for i in range(len(A_peri)-1):
        result_list.append(A_peri[i]*1e+6*N_pe*N_crossbar*N_tier_real*N_tile)

    for i in range(len(out_peripherial)-4,len(out_peripherial)-2):
        result_list.append(out_peripherial[i])
    result_list.append(A_peri[-1]*1e+6*N_pe*N_crossbar*N_tier_real*N_tile)

    for i in range(len(out_peripherial)-2,len(out_peripherial)):
        result_list.append(out_peripherial[i])


with open(filename_results, 'a', newline='') as csvfile:
    # Create a csv writer object
    writer = csv.writer(csvfile)
    # Write the header row (optional)
    #writer.writerow(["freq_core","freq_noc","Xbar_size","N_tile","N_pe","N_tier(chiplet)","W2d","W3d","Computing_latency", "Computing_energy","chip_area","network_latency","network_energy","peak_temperature"])
    # Write each row of data from the list
    writer.writerow(result_list)
if temp:

    print("The noc sim time is:", (end_thermal - end_noc))
    print("whole sim time",(end_thermal-start))


