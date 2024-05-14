import pandas as pd
from Module_Compute.functions import imc_analy,power_density
import csv
from itertools import chain
import math
from Module_AI_Map.util_chip.util_mapping import model_mapping, smallest_square_greater_than, aib
import sys
def compute_IMC_model(COMPUTE_VALIDATE,xbar_size,freq_computing,quant_act,N_crossbar,N_pe,placement_method,N_tier,tiles_each_tier,N_tile,total_tiles_real,result_list):
    #
    # area,latency,power
    # every layer-> tile number, input vector

    #---------------------------------------------------------------------#
    computing_inform = "./to_interconnect_analy/layer_inform.csv"
    computing_data = pd.read_csv(computing_inform, header=None)
    computing_data = computing_data.to_numpy()
    total_model_L=0
    total_model_E_dynamic=0
    total_leakage=0
    volt=0.5
    out_peripherial=[]

    layer_idx=0

    filename = "./to_interconnect_analy/layer_performance.csv"
        # writing to csv file  
    with open(filename, 'w') as csvfile1: 
        writer_performance = csv.writer(csvfile1) 
        for layer_idx in range(len(computing_data)):
            # single crossbar

            # LUT table

            mode = 1
            if COMPUTE_VALIDATE:
                freq_adc=0.005
                A_pe, L_layer, E_layer, peripherials, A_peri = imc_analy(computing_data, mode, xbar_size, layer_idx, volt, freq_computing, freq_adc)
            else:
                A_pe, L_layer, E_layer, peripherials, A_peri = imc_analy(computing_data, mode, xbar_size, layer_idx, volt, freq_computing, freq_computing)
            L_layer=L_layer/freq_computing
            #print("layer",layer_idx,L_chip)
            total_model_L+=L_layer
            if COMPUTE_VALIDATE:
                if len(out_peripherial)==0:
                    out_peripherial.append(peripherials)
                    out_peripherial=list(chain.from_iterable(out_peripherial))
                else:
                    for i in range(len(peripherials)):
                        out_peripherial[i]+=peripherials[i]
            pk_power,peak_density =power_density( A_pe, L_layer, E_layer,peripherials, A_peri,computing_data,layer_idx)

            #------------------------------------#
            #          dynamic energy            #
            #------------------------------------#
            # Volatage user setup

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
            csvfile1.write(str(layer_idx)+","+str(computing_data[layer_idx][1])+","+str(L_layer)+","+str(E_layer)+","+str(leak_tile)+","+str('%.3f'% (E_layer/L_layer*1000/computing_data[layer_idx][1]))+","+str(pk_power)+","+str(peak_density))
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
    #area_single_tile=33638.9/3*xbar_size/64*N_pe*0.000001 #mm2
    if placement_method==5:
        N_tier_real=N_tier
        N_tile_real=smallest_square_greater_than(max(tiles_each_tier))
        #N_tile=N_tile_real
        #result_list.insert(3,N_tile)
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
    return N_tier_real,computing_data,area_single_tile,volt,total_model_L,result_list,out_peripherial,A_peri