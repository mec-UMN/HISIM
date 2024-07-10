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
        chip_architect = "M3D", # string | hardware architecture                        - M3D,M2D,H2_5D,H2_5D_3D
        xbar_size = 512,        # int    | crossbar size                                - 64,128,256,512,1024
        N_tile = 324,           # int    | number of tiles in tier                      - 4,9,16,25,36,49                     
        N_crossbar = 1,         # int    | number of crossbars in one PE                - 4, 9, 16
        N_pe = 16,              # int    | number of PEs in one tile - 4,9,16,25,36
        quant_weight = 8,       # int    | precision of quantized weight of AI model
        quant_act = 8,          # int    | precision fo quantized activation of AI model
        freq_computing = 1,     # float  | computing unit operation frequency
        fclk_noc = 1,           # float  | network data communication operation frequency
        tsv_pitch = 10,          # float  | TSV pitch (um)
        N_tier = 4,             # int    | number of tiers                              - 2,3,4,5,6,7,8,9,10 
        volt = 0.5,             # float  | operating voltage
        placement_method = 5,   # int    | computing tile placement
        percent_router = 0.5,   # float  | percentage of router dedicated to 3D comm LOOK INTO THIS WHAT
        compute_validate = False,       # | validate compute model with neurosim 
        W2d = 32,               # int    | number of links of 2D NoC
        router_times_scale = 1, # int    | scaling factor for time of: trc, tva, tsa, tst, tl, tenq
        ai_model = "vit",       # string | AI model
        thermal = True,                 # | run thermal simulation
        ppa_filepath = "./Results/PPA.csv"  
    ):
        if chip_architect == "H2_5D":
            self.placement_method = 1
        else:
            self.placement_method = placement_method

        # never used 
        self.relu=True
        self.sigmoid=False

        self.chip_architect = chip_architect
        self.xbar_size = xbar_size
        self.N_tile = N_tile
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
                                'W2d',
                                'W3d',
                                'Computing_latency (ns)',
                                'Computing_energy (pJ)',
                                'compute_area (um2)',
                                'compute_area per tier (mm2)',
                                'chip_Architecture',
    
                                '2d NoC latency (ns)',
                                '3d NoC latency (ns)',
                                '2.5d NoC latency (ns)',
                                'network_latency (ns)',
                                '2d NoC energy (pJ)',
                                '3d NoC energy (pJ)',
                                '2.5d NoC energy (pJ)',
                                'network_energy (pJ)',

                                'rcc (??)',
                                'compute_power (W)',
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
        self.result_dictionary = {}
        for label in self.csv_header:
            self.result_dictionary[label] = 'NaN'

        with open(self.filename_results, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.csv_header)

        with open("./Results/PPA_new.csv", 'w', newline='') as csvfile:
            header = list(self.result_dictionary.keys())
            writer = csv.writer(csvfile)
            writer.writerow(header)
    
    def set_num_pe(self, N_pe):
        self.N_pe = N_pe
    
    def set_chip_architecture(self, chip_architect):
        self.chip_architect = chip_architect

    def run_model(self):

        result_list=[]

        result_list.append(self.freq_computing)
        result_list.append(self.fclk_noc)
        result_list.append(self.xbar_size)
        result_list.append(self.N_tile)
        result_list.append(self.N_pe)

        # these are all the inputs to be printed in PPA
        self.result_dictionary['freq_core (GHz)'] = self.freq_computing
        self.result_dictionary['freq_noc (GHz)'] = self.fclk_noc
        self.result_dictionary['Xbar_size'] = self.xbar_size
        self.result_dictionary['N_tile'] = self.N_tile
        self.result_dictionary['N_pe'] = self.N_pe

        self.result_dictionary['placement_method'] = self.placement_method
        self.result_dictionary['percent_router'] = self.percent_router

        print("=========================================start HISIM simulation ========================================= ","\n")
        start = time.time()       
        #---------------------------------------------------------------------#
        #                                                                     #
        #     Mapping: from AI model -> hardware mapping                      #
        #                                                                     #
        #---------------------------------------------------------------------#
        network_params = load_ai_network(self.ai_model)                 #Load AI network parameters from the network csv file
        print("----------------------------------------------------","\n")
        print("start mapping ",self.ai_model,"\n")

        #---------------------------------------------------------------------#
        #                                                                     #
        #     Configuration of the AI models mapped to architecture           # 
        #                                                                     #
        #---------------------------------------------------------------------#
        filename = "./Debug/to_interconnect_analy/layer_inform.csv"
        tiles_each_tier = [0] * self.N_tier
        total_tiles_real = model_mapping(
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
            tiles_each_tier)


        print("total_tiles_real: ", total_tiles_real)
        if isinstance(total_tiles_real, list):
            # error!!
            with open("./Results/PPA_new.csv", 'a', newline='') as csvfile:
                values = list(self.result_dictionary.values())
                writer = csv.writer(csvfile)
                writer.writerow(values)
            return total_tiles_real

        #Placement Method 1: Number of tiers are determined based on the mapping and user defined number of tiles per tier
        #Placement Method 5: Number of tiles per tier are determined based on the mapping and user defined number of tiers
        if self.placement_method == 5:
            N_tier_real = self.N_tier
            #Average of tiles mapped per tier                    
            N_tile_real=smallest_square_greater_than(max(tiles_each_tier))
        else:
            N_tile_real = self.N_tile 
            #Total number of tiers or chiplets                          
            if total_tiles_real % self.N_tile==0:
                N_tier_real=int(total_tiles_real//self.N_tile)       
            else:
                N_tier_real=int(total_tiles_real//self.N_tile)+1     
        result_list.append(N_tile_real)          
        self.result_dictionary['N_tile(real)'] = N_tile_real               

        if N_tier_real>4:
            print("Alert!!! too many number of tiers")
            # sys.exit()
            with open("./Results/PPA_new.csv", 'a', newline='') as csvfile:
                values = list(self.result_dictionary.values())
                writer = csv.writer(csvfile)
                writer.writerow(values)
            return ["Alert!!! too many number of tiers"]

        result_list.append(N_tier_real)
        self.result_dictionary['N_tier(real)'] = N_tier_real

        #---------------------------------------------------------------------#
        #                                                                     #
        #     Computing: generate PPA for IMC/GPU/CPU/ASIC computing units    #
        #                                                                     #
        #---------------------------------------------------------------------#
        N_tier_real,computing_data,area_single_tile,volt,total_model_L,result_list,out_peripherial,A_peri = compute_IMC_model(
                                                                                                                self.compute_validate,
                                                                                                                self.xbar_size,
                                                                                                                self.volt, 
                                                                                                                self.freq_computing,
                                                                                                                self.quant_act,
                                                                                                                self.quant_weight,
                                                                                                                self.N_crossbar,
                                                                                                                self.N_pe,
                                                                                                                N_tier_real,
                                                                                                                self.N_tile,
                                                                                                                result_list, 
                                                                                                                self.result_dictionary,
                                                                                                                network_params)
        end_computing = time.time()
        print("--------------------------------------------------------")
        print("----------computing performance done--------------------")


        #---------------------------------------------------------------------#
        #                                                                     #
        #     Network: generate PPA for NoC/NoP (2D/2.5D/3D/heterogeneous)    #
        #                                                                     #
        #---------------------------------------------------------------------#

        chiplet_num,tier_2d_hop_list_power,tier_3d_hop_list_power,single_router_area,mesh_edge,layer_aib_list,result_list = network_model(
                                                                                                                                N_tier_real,
                                                                                                                                self.N_tile,
                                                                                                                                self.N_tier,
                                                                                                                                computing_data,
                                                                                                                                self.placement_method,
                                                                                                                                self.percent_router,
                                                                                                                                self.chip_architect,
                                                                                                                                self.tsv_pitch,
                                                                                                                                area_single_tile,
                                                                                                                                result_list,
                                                                                                                                self.result_dictionary,
                                                                                                                                self.volt,
                                                                                                                                self.fclk_noc,
                                                                                                                                total_model_L,
                                                                                                                                self.router_times_scale)
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
        if self.thermal:
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

        self.result_dictionary['peak_temperature (C)'] = peak_temp
        self.result_dictionary['thermal simulation time (s)'] = end_thermal - end_noc
        self.result_dictionary['networking simulation time (s)'] = end_noc - end_computing
        self.result_dictionary['computing simulation time (s)'] = end_computing - start
        self.result_dictionary['total simulation time (s)'] = end_thermal - start

        with open(self.filename_results, 'a', newline='') as csvfile:
            # Create a csv writer object
            writer = csv.writer(csvfile)
            writer.writerow(result_list)

        with open("./Results/PPA_new.csv", 'a', newline='') as csvfile:
            values = list(self.result_dictionary.values())
            writer = csv.writer(csvfile)
            writer.writerow(values)


        return result_list

    def results_dict(self, results_list):
        return dict(zip(self.csv_header, results_list))

hisim = HiSimModel(
            chip_architect = "M3D",
            xbar_size = 1024,
            N_tile = 100,
            N_pe = 9,
            N_tier = 3,
            freq_computing = 1,
            fclk_noc = 1,
            placement_method = 5,
            router_times_scale = 1,
            percent_router = 1,
            tsv_pitch = 5,
            W2d = 32,
            ai_model = 'vit',
            thermal = False
        )


df = pd.read_csv(f'inputs.csv')

num_pe = df['N_pe']
chip_architect = df['chip_Architecture']

# for i in range(len(df)):
for i in range(1, 50):

    # hisim.set_num_pe(num_pe[i])
    hisim.set_num_pe(i)
    hisim.set_chip_architecture(chip_architect[i])
    _ = hisim.run_model()
# print("results: ")
# for each in results:
#     print(each)
