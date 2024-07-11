import pandas as pd
from Module_Compute.functions import imc_analy
import csv
from itertools import chain
import math
import sys


def compute_IMC_model(COMPUTE_VALIDATE,xbar_size,volt, freq_computing,quant_act, quant_weight, N_crossbar,N_pe,N_tier_real,N_stack_real,N_tile,result_list, network_params):
    #Initialize variables
    total_model_L=0
    total_model_E_dynamic=0
    total_leakage=0
    out_peripherial=[]
    layer_idx=0

    #Obtain layer information from the csv file
    computing_inform = "./Debug/to_interconnect_analy/layer_inform.csv"
    computing_data = pd.read_csv(computing_inform, header=None)
    computing_data = computing_data.to_numpy()

    filename = "./Debug/to_interconnect_analy/layer_performance.csv"
    if COMPUTE_VALIDATE:
        freq_adc=0.005
    else:
        freq_adc=freq_computing
    imc_analy_fn=imc_analy(xbar_size=xbar_size, volt=volt, freq=freq_computing, freq_adc=freq_adc, compute_ref=COMPUTE_VALIDATE, quant_bits=[quant_weight,quant_act])

    # write the layer performance data to csv file  
    with open(filename, 'w') as csvfile1: 
        writer_performance = csv.writer(csvfile1) 
        for layer_idx in range(len(computing_data)):
            A_pe, L_layer, E_layer, peripherials, A_peri = imc_analy_fn.forward(computing_data, layer_idx, network_params)          
            total_model_L+=L_layer
            total_model_E_dynamic+=E_layer
            leak_tile=imc_analy_fn.leakage(N_crossbar,N_pe)
            total_leakage+=leak_tile*L_layer*computing_data[layer_idx][1]

            # CSV file is written in the following format:
            #layer index, number of tiles required for this layer, latency of the layer, Energy of the layer, leakage energy of the layer, average power consumption of each tile for the layer
            csvfile1.write(str(layer_idx)+","+str(computing_data[layer_idx][1])+","+str(L_layer)+","+str(E_layer)+","+str(leak_tile)+","+str('%.3f'% (E_layer/L_layer*1000/computing_data[layer_idx][1])))
            csvfile1.write('\n')

            #Save performance data of peripherials for each layer
            if COMPUTE_VALIDATE:
                if len(out_peripherial)==0:
                    out_peripherial.append(peripherials)
                    out_peripherial=list(chain.from_iterable(out_peripherial))
                else:
                    for i in range(len(peripherials)):
                        out_peripherial[i]+=peripherials[i]


    print("----------computing performance results-----------------")
    print("--------------------------------------------------------")
    print("Total compute latency",round(total_model_L*pow(10,9),5),"ns")
    print("Total dynamic energy",round(total_model_E_dynamic*pow(10,12),5),"pJ")
    print("Overall compute Power",round(total_model_E_dynamic/(total_model_L),5),"W")
    print("Total Leakage energy",round(total_leakage*pow(10,12),5),"pJ")
    result_list.append(total_model_L*pow(10,9))
    result_list.append(total_model_E_dynamic*pow(10,12))
    
            #-----------------------------------#
            #         Computing Area            #
            #-----------------------------------#
    area_single_tile=A_pe*N_pe*N_crossbar
    total_tiles_area=N_stack_real*N_tier_real*N_tile*area_single_tile
    print("Total tiles area",round(total_tiles_area,5),"mm2")
    print("Total tiles area each tier,",round(total_tiles_area/N_stack_real/N_tier_real,5),"mm2")
    result_list.append(total_tiles_area*pow(10,6))
    return computing_data,area_single_tile,volt,total_model_L,result_list,out_peripherial,A_peri