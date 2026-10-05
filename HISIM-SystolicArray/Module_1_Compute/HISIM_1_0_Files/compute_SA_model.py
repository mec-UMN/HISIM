import pandas as pd
from Module_Compute.functions_SA import SA_calc
import csv
from itertools import chain
import math
import sys


def compute_SA_model(SA_size, freq_computing, N_arr, N_pe, N_tier_real, N_stack_real, N_tile, volt, bit_width, result_list, result_dictionary, network_params):
    
    #Initialize variables
    total_model_L = 0
    total_model_E_dynamic = 0
    layer_idx = 0

    #Obtain layer information from the csv file
    computing_inform = "./Debug/to_interconnect_analy/layer_inform.csv"
    computing_data = pd.read_csv(computing_inform, header=None)
    computing_data = computing_data.to_numpy()

    filename = "./Debug/to_interconnect_analy/layer_performance.csv"
    
    SA_function = SA_calc(SA_size, freq_computing, N_arr, N_pe, N_tile, bit_width)

    A_tile = 0

    # write the layer performance data to csv file  
    with open(filename, 'w') as csvfile1: 
        
        for layer_idx in range(len(computing_data)):
            A_curr_tile, L_layer, E_layer = SA_function.forward(layer_idx, network_params, computing_data)          
            total_model_L += L_layer
            total_model_E_dynamic += E_layer

            A_tile = max(A_curr_tile, A_tile)

            # CSV file is written in the following format:
            # layer index, number of tiles required for this layer, latency of the layer, Energy of the layer, (TODO) leakage energy of the layer, average power consumption of each tile for the layer
            csvfile1.write(str(layer_idx)+","+str(computing_data[layer_idx][1])+","+str(L_layer)+","+str(E_layer)+","+str('%.3f'% (E_layer/L_layer*1000/computing_data[layer_idx][1])))
            csvfile1.write('\n')


    print("----------computing performance results-----------------")
    print("--------------------------------------------------------")
    print("Total compute latency", round(total_model_L*pow(10, 9), 5), "ns")
    print("Total dynamic energy", round(total_model_E_dynamic*pow(10, 3), 5), "mJ")
    print("Overall compute Power", round(total_model_E_dynamic/(total_model_L), 5), "W")
    # print("Total Leakage energy", round(total_leakage*pow(10, 12), 5), "pJ")
    result_list.append(total_model_L*pow(10, 9))
    result_list.append(total_model_E_dynamic*pow(10, 12))
    
            #-----------------------------------#
            #         Computing Area            #
            #-----------------------------------#
    total_tiles_area = N_stack_real*N_tier_real*N_tile*A_tile
    print("Total tiles area", round(total_tiles_area, 5), "mm2")
    print("Total tiles area each tier,", round(total_tiles_area/N_stack_real/N_tier_real, 5), "mm2")
    result_list.append(total_tiles_area*pow(10, 6))


    result_dictionary['Computing_latency (ns)'] = total_model_L*pow(10,9)
    result_dictionary['Computing_energy (pJ)'] = total_model_E_dynamic*pow(10,12)
    result_dictionary['compute_area (um2)'] = total_tiles_area*pow(10,6)

    return N_tier_real, computing_data, A_tile, volt, total_model_L, result_list