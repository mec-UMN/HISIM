# -*- coding: utf-8 -*-
# *******************************************************************************
# Copyright (c)
# School of Electrical, Computer and Energy Engineering, Arizona State University
# Department of Electrical and Computer Engineering, University of Minnesota

# PI: Prof.Yu(Kevin) Cao
# All rights reserved.

# This source code is for HISIM: Analytical Performance Modeling and Design Exploration 
# of 2.5D/3D Heterogeneous Integration for AI Computing

# Copyright of the model is maintained by the developers, and the model is distributed under 
# the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
# http://creativecommons.org/licenses/by-nc/4.0/legalcode.
# The source code is free and you can redistribute and/or modify it
# by providing that the following conditions are met:
# 
#  1) Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
# 
#  2) Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# Developer list: 
#   Zhenyu Wang	    Email: zwang586@asu.edu                
#   Pragnya Nalla   Email: nalla052@umn.edu
#   Jingbo Sun      Email: jsun127@asu.edu
# *******************************************************************************/

import pandas as pd
import numpy as np
import os
import shutil
import csv
import time
import argparse
import matplotlib.pyplot as plt
import sys
import scipy.sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra
from Module_Compute.functions import imc_analy
from Module_Thermal.thermal_model import thermal_model
from Module_Network.network_model import network_model
from Module_Compute.compute_IMC_model import compute_IMC_model
from Module_AI_Map.util_chip.util_mapping import model_mapping, load_ai_network,smallest_square_greater_than
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

#results lists for showing final electro-thermal results
result_list=[]
parser = argparse.ArgumentParser(description='Design Space Search',
								 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--chip_architect', type=str, default="M3D",help='hardware architecture:M3D,M2D,H2_5D,H2_5D_3D')
parser.add_argument('--xbar_size', type=int, default=512,help='crossbar size')
parser.add_argument('--N_tile', type=int, default=324,help='how many tiles in tier')
parser.add_argument('--N_crossbar', type=int, default=1,help='how many crossbar in one PE')
parser.add_argument('--N_pe', type=int, default=16,help='how many PEs in tile')
parser.add_argument('--quant_weight', type=int, default=8,help='Precision of quantized weight of AI model')
parser.add_argument('--quant_act', type=int, default=8,help='Precision of quantized activation of AI model')
parser.add_argument('--freq_computing', type=float, default=1,help='Computing unit operation frequency')
parser.add_argument('--fclk_noc', type=float, default=1,help='network data communication operation frequency')
parser.add_argument('--tsvPitch', type=float, default=10,help='TSV pitch um')
parser.add_argument('--bus_width', type=int, default=64,help='Inside PE or Tile, the number of links of the bus width')
parser.add_argument('--N_tier', type=int, default=4,help='how many tiers')
parser.add_argument('--volt', type=int, default=0.5,help='Operating Voltage in volt')
parser.add_argument('--placement_method', type=int, default=5,help='computing tile placement method')
parser.add_argument('--percent_router', type=float, default=0.5,help='when data route from one tier to next tier, the system will choose how much percent routers for 3D communication')
parser.add_argument('--compute_validate', action='store_true',help='mode to validate the compute model with neurosim')
parser.add_argument('--W2d', type=int, default=32,help='Number of links of 2D NoC')
parser.add_argument('--router_times_scale', type=int, default=1,help='Scaling factor for time components of router: trc, tva, tsa, tst,tl, tenq')
parser.add_argument('--ai_model', type=str, default="vit",help='AI models:vit, gcn, resnet50, resnet110, vgg16, densenet121, test, roofline')
parser.add_argument('--thermal', action='store_true', help='Run thermal simulation or not')


#Take all below parameters as argument
args = parser.parse_args()

xbar_size = args.xbar_size # 64,128,256,512,1024
N_tile=args.N_tile # 4,9,16,25,36,49 # how many tile in tier (chiplet)
N_tier=args.N_tier # 2,3,4,5,6,7,8,9,10 
N_pe=args.N_pe # 4,9,16,25 # how many PE in tile
N_crossbar=args.N_crossbar # 4, 9, 16 # how many crossbar in PE
quant_weight=args.quant_weight # weight quantization bi
quant_act=args.quant_act # activation quantization bit
bus_width=args.bus_width # in PE and in tile bus width
tsvPitch = args.tsvPitch
chip_architect=args.chip_architect 
if args.thermal:
    thermal=args.thermal
else:
    thermal = False
if args.compute_validate:
    COMPUTE_VALIDATE = True
else:
    COMPUTE_VALIDATE = False
placement_method=args.placement_method  # 1: Tier/Chiplet Edge to Tier/Chiplet Edge connection 
                                        # 2: from the bottom to top tier1
                                        # 3: the hotspot far from each other
                                        # 4: worse case:put all hotspot in the same place
                                        # 5: tile-to-tile 3D connection
if chip_architect=="H2_5D":
    placement_method=1
percent_router=args.percent_router
relu=True
sigmoid=False
freq_computing=args.freq_computing #GHz
fclk_noc=args.fclk_noc
W2d=args.W2d
volt=args.volt
scale_factor=args.router_times_scale
aimodel=args.ai_model
result_list.append(freq_computing)
result_list.append(fclk_noc)
result_list.append(xbar_size)
result_list.append(N_tile)
result_list.append(N_pe)

print("start HISIM simulation ","\n")
#---------------------------------------------------------------------#
#                                                                     #
#     Mapping: from AI model -> hardware mapping                      #
#                                                                     #
#---------------------------------------------------------------------#
start = time.time()
network_params=load_ai_network(aimodel)                 #Load AI network parameters from the network csv file
sim_name="VIT_placement_1"
filename_results = "./Results/PPA.csv"                  #Location to store PPA results
print("----------------------------------------------------","\n")
print("start mapping ",args.ai_model,"\n")


#---------------------------------------------------------------------#
#                                                                     #
#         configuration of the AI models mapped to architecture       #
#                                                                     #
#---------------------------------------------------------------------#
filename = "./Debug/to_interconnect_analy/layer_inform.csv"
tiles_each_tier = [0]*N_tier
total_tiles_real=model_mapping(filename,placement_method,network_params,quant_act,xbar_size,N_crossbar,N_pe,quant_weight,N_tile,N_tier,tiles_each_tier)

#Placement Method 1: Number of tiers are determined based on the mapping and user defined number of tiles per tier
#Placement Method 5: Number of tiles per tier are determined based on the mapping and user defined number of tiers
if placement_method==5:
   N_tier_real=N_tier
   #Average of tiles mapped per tier                    
   N_tile_real=smallest_square_greater_than(max(tiles_each_tier))
else:
    N_tile_real=N_tile 
    #Total number of tiers or chiplets                          
    if total_tiles_real%N_tile==0:
        N_tier_real=int(total_tiles_real//N_tile)       
    else:
        N_tier_real=int(total_tiles_real//N_tile)+1     
result_list.append(N_tile_real)                         

if N_tier_real>4:
    print("Alert!!! too many number of tiers")
    sys.exit()

result_list.append(N_tier_real)


#---------------------------------------------------------------------#
#                                                                     #
#     Computing: generate PPA for IMC/GPU/CPU/ASIC computing units    #
#                                                                     #
#---------------------------------------------------------------------#
N_tier_real,computing_data,area_single_tile,volt,total_model_L,result_list,out_peripherial,A_peri=compute_IMC_model(COMPUTE_VALIDATE,xbar_size,volt, freq_computing,quant_act,quant_weight,N_crossbar,N_pe,N_tier_real,N_tile,result_list)
end_computing = time.time()
print("Computing model sim time is:", (end_computing - start),"s")
print("----------------------------------------------------")
print('\n')

#---------------------------------------------------------------------#
#                                                                     #
#     Network: generate PPA for NoC/NoP (2D/2.5D/3D/heterogeneous)    #
#                                                                     #
#---------------------------------------------------------------------#

chiplet_num,tier_2d_hop_list_power,tier_3d_hop_list_power,single_router_area,mesh_edge,layer_aib_list,result_list=network_model(N_tier_real,
                                                                                                                    N_tile,N_tier,computing_data,placement_method,percent_router,chip_architect,tsvPitch,
                  area_single_tile,result_list,volt,fclk_noc,total_model_L)

end_noc = time.time()
print("\n")
print("-------------------time report--------------------")
print("The noc sim time is:", (end_noc - end_computing))
print("The total sim time is:", (end_noc - start))



#---------------------------------------------------------------------------------#
#                                                                                 #
#     Thermal: generate temperature of the chip (based on power,area)             #
#                                                                                 #
#---------------------------------------------------------------------------------#

# thermal_model function will start the simulation for generating the temperature results based on the power and area of the different blocks of the chip

peak_temp=thermal_model(thermal,chip_architect,chiplet_num,N_tile,placement_method,tier_2d_hop_list_power,tier_3d_hop_list_power,area_single_tile,single_router_area
                  ,mesh_edge,sim_name,layer_aib_list)

end_thermal = time.time()
result_list.append(peak_temp)
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
if thermal:

    print("The noc sim time is:", (end_thermal - end_noc))
    print("whole sim time",(end_thermal-start))


