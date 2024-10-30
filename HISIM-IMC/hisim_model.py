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
#   Jennifer Zhou   Email: 
# *******************************************************************************/

import os
import shutil
import csv
import time
import argparse
import sys
from Module_Compute.functions import imc_analy
from Module_Thermal.thermal_model import thermal_model
from Module_Network.network_model import network_model
from Module_Compute.compute_IMC_model import compute_IMC_model
from Module_AI_Map.util_chip.util_mapping import model_mapping, load_ai_network,smallest_square_greater_than
from Module_Network.aib_2_5d import  aib
from itertools import chain
import pandas as pd

if not os.path.exists('./Debug/to_interconnect_analy'):
    os.makedirs('./Debug/to_interconnect_analy')
if not os.path.exists('./Results/result_thermal'):
    os.makedirs('./Results/result_thermal')
if os.path.exists('./Results/result_thermal/1stacks'):
    shutil.rmtree('.//Results/result_thermal/1stacks')
if not os.path.exists('./Results'):
    os.makedirs('./Results')
os.makedirs('.//Results/result_thermal/1stacks')

class HiSimModel:

    def __init__(
        self,
        chip_architect = "M3D", # string | hardware architecture                        - M3D,M2D,H2_5D,M3_5D,H2_5D_3D
        xbar_size = 512,        # int    | crossbar size                                - 64,128,256,512,1024
        N_tile = 324,           # int    | number of tiles in tier                      - 4,9,16,25,36,49                     
        N_crossbar = 1,         # int    | number of crossbars in one PE                - 4, 9, 16
        N_pe = 16,              # int    | number of PEs in one tile                    - 4,9,16,25,36
        quant_weight = 8,       # int    | precision of quantized weight of AI model
        quant_act = 8,          # int    | precision of quantized activation of AI model
        freq_computing = 1,     # float  | computing unit operation frequency
        fclk_noc = 1,           # float  | network data communication operation frequency
        tsv_pitch = 10,         # float  | TSV pitch (um)
        N_tier = 4,             # int    | number of tiers                              - 2,3,4,5,6,7,8,9,10 
        volt = 0.5,             # float  | operating voltage
        placement_method = 5,   # int    | computing tile placement method                 1: Tier/Chiplet Edge to Tier/Chiplet Edge connection
                                #                                                          5: tile-to-tile 3D connection
        percent_router = 0.5,   # float  | percentage of router dedicated to 3D commmunication in routing method 2
        routing_method = 2,      #int     | 3D routing method                        1- local routing-only uses nearby routers and tsvs 
                                #        |                                           2-global routing-data will try to use all the routers to transport to next tier
        compute_validate = False,       # | validate compute model with neurosim 
        W2d = 32,               # int    | number of links of 2D NoC                    
        router_times_scale = 1, # int    | scaling factor for time of: trc, tva, tsa, tst, tl, tenq
        ai_model = "vit",       # string | AI model                                     -vit, gcn, resnet50, resnet110, vgg16, densenet121, test, roofline
        thermal = True,                 # | run thermal simulation
        N_stack=1,               #int      |Number of 3D stacks in 3.5D design           -1, 2,3,4,5,6,7,8,9,10 
        ppa_filepath = "./Results/PPA.csv"  
    ):
        if chip_architect == "H2_5D":
            self.placement_method = 1
            N_tier=1
        elif chip_architect=="M3D":
            N_stack=1
        elif chip_architect=="M2D":
            N_stack=1
            N_tier=1
        else:
            self.placement_method = placement_method

        self.relu=True
        self.sigmoid=False

        self.chip_architect = chip_architect
        self.xbar_size = xbar_size
        self.N_tile = N_tile
        self.N_stack=N_stack
        self.N_crossbar = N_crossbar
        self.N_pe = N_pe
        self.quant_weight = quant_weight
        self.quant_act = quant_act
        self.freq_computing = freq_computing
        self.fclk_noc = fclk_noc
        self.tsv_pitch = tsv_pitch
        self.N_tier = N_tier
        self.volt = volt
        self.placement_method = placement_method
        self.percent_router = percent_router
        self.routing_method=routing_method 
        self.compute_validate = compute_validate
        self.W2d = W2d
        self.router_times_scale = router_times_scale
        self.ai_model = ai_model
        self.thermal = thermal
        self.filename_results = ppa_filepath
        
        self.csv_header = [
                                'freq_core (GHz)',
                                'freq_noc (GHz)',
                                'Xbar_size',
                                'N_tile', 
                                'N_pe',
                                'N_tile(real)',
                                'N_tier(real)',
                                "N_stack(real)",
                                'W2d',
                                'W3d',
                                'Computing_latency (ns)',
                                'Computing_energy (pJ)',
                                'compute_area (um2)',
                                'chip area (mm2)',
                                'chip_Architecture',
    
                                '2d NoC latency (ns)',
                                '3d NoC latency (ns)',
                                '2.5d NoC latency (ns)',
                                'network_latency (ns)',
                                '2d NoC energy (pJ)',
                                '3d NoC energy (pJ)',
                                '2.5d NoC energy (pJ)',
                                'network_energy (pJ)',

                                'rcc (compute latency/communciation latency)',
                                'Throughput(TFLOPS/s)',
                                '2D_3D_NoC_power (W)',
                                '2_5D_power (W)',
                                '2d_3d_router_area (mm2)',

                                'placement_method',
                                'percent_router',

                                'peak_temperature (C)',

                                'thermal simulation time (s)',
                                'networking simulation time (s)',
                                'computing simulation time (s)',
                                'total simulation time (s)'
                            ]

        with open(self.filename_results, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.csv_header)

        with open("./Results/PPA_new.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.csv_header)
    
    ################################
    ############ Inputs ############
    ################################

    def set_num_pe(self, N_pe):
        self.N_pe = N_pe
    
    def set_chip_architecture(self, chip_architect):
        self.chip_architect = chip_architect

    def set_xbar_size(self, xbar_size):
        self.xbar_size = xbar_size

    def set_N_tile(self, N_tile):
        self.N_tile = N_tile
    
    def set_N_crossbar(self, N_crossbar):
        self.N_crossbar = N_crossbar

    def set_quant_weight(self, quant_weight):
        self.quant_weight = quant_weight

    def set_quant_act(self, quant_act):
        self.quant_act = quant_act

    def set_freq_computing(self, freq_computing):
        self.freq_computing = freq_computing

    def set_fclk_noc(self, fclk_noc):
        self.fclk_noc = fclk_noc

    def set_tsv_pitch(self, tsv_pitch):
        self.tsv_pitch = tsv_pitch

    def set_N_tier(self, N_tier):
        self.N_tier = N_tier

    def set_N_stack(self, N_stack):
        self.N_stack = N_stack

    def set_volt(self, volt):
        self.volt = volt
    
    def set_placement(self, placement_method): 
        self.placement_method = placement_method
    
    def set_router(self, route_method): 
        self.route_method = route_method

    def set_percent_router(self, percent_router):
        self.percent_router = percent_router

    def set_W2d(self, W2d):
        self.W2d = W2d
    
    
    ################################
    ######### Meta-Inputs ##########
    ################################

    def set_compute_validate(self, compute_validate):
        self.compute_validate = compute_validate

    def set_router_times_scale(self, router_times_scale):
        self.router_times_scale = router_times_scale

    def set_ai_model(self, ai_model):
        self.ai_model = ai_model

    def set_thermal(self, thermal):
        self.thermal = thermal

    def set_ppa_filepath(self, ppa_filepath):
        self.filename_results = ppa_filepath

    def run_model(self):
        if os.path.exists('./Results/result_thermal/'):
            shutil.rmtree('./Results/result_thermal/')
        os.makedirs('./Results/result_thermal/')
        os.makedirs('./Results/result_thermal/1stacks')

        result_list=[]

        result_list.append(self.freq_computing)
        result_list.append(self.fclk_noc)
        result_list.append(self.xbar_size)
        result_list.append(self.N_tile)
        result_list.append(self.N_pe)

        result_dictionary = {}
        for label in self.csv_header:
            result_dictionary[label] = 'NaN'

        # these are all the inputs to be printed in PPA
        result_dictionary['freq_core (GHz)'] = self.freq_computing
        result_dictionary['freq_noc (GHz)'] = self.fclk_noc
        result_dictionary['Xbar_size'] = self.xbar_size
        result_dictionary['N_tile'] = self.N_tile
        result_dictionary['N_pe'] = self.N_pe

        result_dictionary['placement_method'] = self.placement_method
        result_dictionary['percent_router'] = self.percent_router

        print("=========================================start HISIM simulation ========================================= ","\n")
        start = time.time()       
        #---------------------------------------------------------------------#
        #                                                                     #
        #     Mapping: from AI model -> hardware mapping                      #
        #                                                                     #
        #---------------------------------------------------------------------#
        network_params = load_ai_network(self.ai_model)                 #Load AI network parameters from the network csv file
        sim_name="Densenet_placement_1"
        filename_results = "./Results/PPA.csv"                          #Location to store PPA results

        print("----------------------------------------------------","\n")
        print("start mapping ",self.ai_model,"\n")

        #---------------------------------------------------------------------#
        #                                                                     #
        #     Configuration of the AI models mapped to architecture           # 
        #                                                                     #
        #---------------------------------------------------------------------#
        filename = "./Debug/to_interconnect_analy/layer_inform.csv"
        mapping_results= model_mapping(
            filename,
            self.placement_method,
            network_params,
            self.quant_act,
            self.xbar_size,
            self.N_crossbar,
            self.N_pe,
            self.quant_weight,
            self.N_tile,
            self.N_tier,
            self.N_stack)


        #print("total_tiles_real: ", mapping_results[0])
        if isinstance(mapping_results, list):
            # error!!
            with open("./Results/PPA_new.csv", 'a', newline='') as csvfile:
                values = list(result_dictionary.values())
                writer = csv.writer(csvfile)
                writer.writerow(values)
            return mapping_results
        total_tiles_real ,tiles_each_tier, tiles_each_stack=mapping_results

        #Placement Method 1: Number of tiers are determined based on the mapping and user defined number of tiles per tier
        #Placement Method 5: Number of tiles per tier are determined based on the mapping and user defined number of tiers
        N_stack_real=len(tiles_each_stack)
        if self.placement_method == 5:
            N_tier_real = self.N_tier
            #Average of tiles mapped per tier                    
            N_tile_real=smallest_square_greater_than(max(max(tiles_each_tier)))
        else:
            N_tile_real = self.N_tile 
            if N_stack_real>1:
                N_tier_real=self.N_tier
            else:
                #Total number of tiers or chiplets                          
                if total_tiles_real % self.N_tile==0:
                    N_tier_real=int(total_tiles_real//self.N_tile)       
                else:
                    N_tier_real=int(total_tiles_real//self.N_tile)+1     
        result_list.append(N_tile_real)          
        result_dictionary['N_tile(real)'] = N_tile_real               

        if N_tier_real>4:
            print("Alert!!! too many number of tiers")
            # sys.exit()
            with open("./Results/PPA_new.csv", 'a', newline='') as csvfile:
                values = list(result_dictionary.values())
                writer = csv.writer(csvfile)
                writer.writerow(values)
            return ["Alert!!! too many number of tiers"]

        result_list.append(N_tier_real)
        result_dictionary['N_tier(real)'] = N_tier_real
    
        if N_stack_real==1:
            self.chip_architect = "M3D" if N_tier_real>1 else "M2D"
        elif N_tier_real==1:
            self.chip_architect = "H2_5D"
        result_list.append(N_stack_real)
        result_dictionary['N_stack(real)'] = N_stack_real
        #---------------------------------------------------------------------#
        #                                                                     #
        #     Computing: generate PPA for IMC/GPU/CPU/ASIC computing units    #
        #                                                                     #
        #---------------------------------------------------------------------#
        compute_results = compute_IMC_model(
                                            self.compute_validate,
                                            self.xbar_size,
                                            self.volt, 
                                            self.freq_computing,
                                            self.quant_act,
                                            self.quant_weight,
                                            self.N_crossbar,
                                            self.N_pe,
                                            N_tier_real,
                                            N_stack_real,
                                            self.N_tile,
                                            result_list, 
                                            result_dictionary,
                                            network_params,
                                            self.relu
                                        )

        N_tier_real,computing_data,area_single_tile,volt,total_model_L,result_list,out_peripherial,A_peri = compute_results
        end_computing = time.time()
        print("--------------------------------------------------------")
        print("----------computing performance done--------------------")


        #---------------------------------------------------------------------#
        #                                                                     #
        #     Network: generate PPA for NoC/NoP (2D/2.5D/3D/heterogeneous)    #
        #                                                                     #
        #---------------------------------------------------------------------#

        network_results = network_model(
                                        N_tier_real,
                                        N_stack_real,
                                        self.N_tile,
                                        self.N_tier,
                                        computing_data,
                                        self.placement_method,
                                        self.percent_router,
                                        self.chip_architect,
                                        self.tsv_pitch,
                                        area_single_tile,
                                        result_list,
                                        result_dictionary,
                                        self.volt,
                                        self.fclk_noc,
                                        total_model_L,
                                        self.router_times_scale,
                                        tiles_each_tier, 
                                        self.routing_method, 
                                        self.W2d
                                    )
        chiplet_num,tier_2d_hop_list_power,tier_3d_hop_list_power,single_router_area,mesh_edge,layer_aib_list,result_list = network_results

        end_noc = time.time()
        print("\n")
        result_list.append(self.placement_method)
        result_list.append(self.percent_router)

        #---------------------------------------------------------------------------------#
        #                                                                                 #
        #     Thermal: generate temperature of the chip (based on power,area)             #
        #                                                                                 #
        #---------------------------------------------------------------------------------#
        sim_name="Densenet_placement_1"
        # thermal_model function will start the simulation for generating the temperature results based on the power and area of the different blocks of the chip
        if self.thermal and self.chip_architect!="M3_5D":
            print("----------thermal analysis start--------------------")

            peak_temp = thermal_model(
                            self.thermal,
                            self.chip_architect,
                            chiplet_num,
                            self.N_tile,
                            self.placement_method,
                            tier_2d_hop_list_power,
                            tier_3d_hop_list_power,
                            area_single_tile,
                            single_router_area,
                            mesh_edge,
                            sim_name,
                            layer_aib_list)
        else:
            peak_temp = 'NaN'

        end_thermal = time.time()
            
        result_list.append(peak_temp)
        result_list.append(end_thermal - end_noc) # thermal time

        result_list.append(end_noc - end_computing) # networking time
        result_list.append(end_computing - start) # computing time
        result_list.append(end_thermal - start) # whole sim

        result_dictionary['peak_temperature (C)'] = peak_temp
        result_dictionary['thermal simulation time (s)'] = end_thermal - end_noc
        result_dictionary['networking simulation time (s)'] = end_noc - end_computing
        result_dictionary['computing simulation time (s)'] = end_computing - start
        result_dictionary['total simulation time (s)'] = end_thermal - start

        with open(self.filename_results, 'a', newline='') as csvfile:
            # Create a csv writer object
            writer = csv.writer(csvfile)
            writer.writerow(result_list)

        with open("./Results/PPA_new.csv", 'a', newline='') as csvfile:
            values = list(result_dictionary.values())
            writer = csv.writer(csvfile)
            writer.writerow(values)

        return result_list

    def results_dict(self, results_list):
        return dict(zip(self.csv_header, results_list))
"""
hisim = HiSimModel(
            chip_architect = "M3_5D",
            xbar_size = 1024,
            N_tile = 100,
            N_pe = 9,
            N_tier = 2,
            freq_computing = 1,
            fclk_noc = 1,
            placement_method = 5,
            router_times_scale = 1,
            percent_router = 1,
            tsv_pitch = 5,
            W2d = 32,
            ai_model = 'vit',
            thermal = False,
            N_stack =2,
        )
hisim.run_model()

df = pd.read_csv(f'inputs.csv')

num_pe = df['N_pe']
chip_architect = df['chip_Architecture']

# for i in range(len(df)):
for i in range(9, 50):

    # hisim.set_num_pe(num_pe[i])
    hisim.set_num_pe(i)
    hisim.set_chip_architecture(chip_architect[i])
    _ = hisim.run_model()
# print("results: ")
# for each in results:
#     print(each)
"""