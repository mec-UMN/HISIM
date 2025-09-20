import math
from Module_Compute.HISIM_2_0_Files.Mem import area_mem_tile, latency_buf, determine_buffer_energy, nearest_ceil_value
import pandas as pd
import numpy as np
from functools import reduce
import operator

def area_sa_tile(sa_df,compute_json_data,tile_area, lut_df):
    tile_buf_area={}
    tile_buf_df={}
    for _, row in sa_df.iterrows():
        #print(row)
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        SA_size_x=row["SA_size_x"]
        SA_size_y=row["SA_size_y"]
        n_SA=row["n_SA"]
        #PE
        area_SA=(compute_json_data["A_pe"]*(SA_size_x*SA_size_y-SA_size_y)+compute_json_data["A_pe_toprow"]*SA_size_y)*n_SA
        
        #Ctl
        area_ctl=compute_json_data["A_ctl_a"]*SA_size_x+compute_json_data["A_ctl_b"]

        #Buff
        rows = []
        NW_inbuf_real = nearest_ceil_value(lut_df["NW"], SA_size_x + SA_size_y + 1)
        NB_inbuf_real = nearest_ceil_value(lut_df["NB"], SA_size_x * row["prec"])
        rows.append({
            "Chiplet ID": row["Chiplet ID"],
            "Tile ID": row["Tile ID"] + "_inbuf",
            "HW Type": "Tile Input Buffer",
            "Nbank": 1,
            "NW": NW_inbuf_real,
            "NB": NB_inbuf_real,
            "CM": 4,
            "F_actbitlines": SA_size_x * row["prec"]/NB_inbuf_real,
            "F_non_zeros_r": (SA_size_x + SA_size_y - 1)/(2*SA_size_x+2*SA_size_y-1),
            "F_non_zeros_w": 1,
        })

        tile_buf_df[chip_tile_idx] = pd.DataFrame(rows)
        chip_tile_inbuf_idx= row["Chiplet ID"]+"_"+row["Tile ID"]+"_inbuf"
        NW_wgbuf_real = nearest_ceil_value(lut_df["NW"], SA_size_x+1)
        NB_wgbuf_real = nearest_ceil_value(lut_df["NB"], SA_size_y*row["prec"])
        new_row=pd.DataFrame([{
            "Chiplet ID": row["Chiplet ID"],
            "Tile ID": row["Tile ID"]+"_wgbuf",
            "HW Type": "Tile Weight Buffer",
            "Nbank": 1,
            "NW": NW_wgbuf_real,
            "NB": NB_wgbuf_real,
            "CM": 4,
            "F_actbitlines": SA_size_y*row["prec"]/(NB_wgbuf_real),
            "F_non_zeros_r": SA_size_x/(2*SA_size_x+2*SA_size_y-1),
            "F_non_zeros_w": 1,
        }])
        tile_buf_df[chip_tile_idx] = pd.concat([tile_buf_df[chip_tile_idx], new_row], ignore_index=True)
        chip_tile_wgbuf_idx= row["Chiplet ID"]+"_"+row["Tile ID"]+"_wgbuf"
        NW_outbuf_real = nearest_ceil_value(lut_df["NW"], SA_size_x + SA_size_y - 1)
        NB_outbuf_real = nearest_ceil_value(lut_df["NB"], SA_size_y * row["prec"])
        new_row=pd.DataFrame([{
            "Chiplet ID": row["Chiplet ID"],
            "Tile ID": row["Tile ID"]+"_outbuf",
            "HW Type": "Tile Output Buffer",
            "Nbank": 1,
            "NW": NW_outbuf_real,
            "NB": NB_outbuf_real,
            "CM": 4,
            "F_actbitlines": SA_size_y*row["prec"]/(NB_outbuf_real),
            "F_non_zeros_r": 1,
            "F_non_zeros_w": 1,
        }])
        tile_buf_df[chip_tile_idx] = pd.concat([tile_buf_df[chip_tile_idx], new_row], ignore_index=True)
        chip_tile_outbuf_idx= row["Chiplet ID"]+"_"+row["Tile ID"]+"_outbuf"
        tile_buf_area=area_mem_tile(tile_buf_df[chip_tile_idx], compute_json_data, tile_buf_area, lut_df)
        area_buf=tile_buf_area[chip_tile_inbuf_idx]+tile_buf_area[chip_tile_wgbuf_idx]+tile_buf_area[chip_tile_outbuf_idx]
        #import pdb; pdb.set_trace()
        # Total number of adders in Accumulation Module
        area_adder=compute_json_data["A_accum"]*row["num_adders_tile"]
        tile_area[chip_tile_idx]=area_SA+area_ctl+area_buf+area_adder
        #print("chip_tile_idx, area_SA, area_ctl, area_buf, area_adder, tile_area[chip_tile_idx]", chip_tile_idx, area_SA, area_ctl, area_buf, area_adder, tile_area[chip_tile_idx])
    return tile_area, tile_buf_df


    tile_latency_breakdown={}
def latency_sa_tile(sa_df,compute_json_data, tile_latency, tile_latency_breakdown):
    compute_latency={}
    for _, row in sa_df.iterrows():
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        SA_size_x=row["SA_size_x"]
        SA_size_y=row["SA_size_y"]
        n_SA=row["n_SA"]
        util={}
        no_inp_cycles=1
        for ele in ["SA_size_x", "SA_size_y", "n_SA"]:
            no_inp_cycles*=math.ceil(row["wrap_"+ele]/row[ele])
            util[ele]= row["wrap_"+ele]/row[ele] if row[ele]>row["wrap_"+ele] else 1

        no_hw_macs = no_inp_cycles*reduce(operator.mul, [util[i] for i in util], 1)* n_SA * SA_size_x * SA_size_y* SA_size_x
        #import pdb; pdb.set_trace()

        no_inp_cycles*=math.ceil(row["layer_macs"]/no_hw_macs) #to incorporate dimension assigned to Auto
        no_inp_cycles*= row["wrap_Temporal"]
        
        #Assuming PEs whtin a tile operate at same clock frequency
        #PEs
        latency_pe=(SA_size_x+SA_size_y-1)*compute_json_data["latency_sa_per_pe_op"]

        #Buff
        latency_buf_rd={}
        for buf_type in ["inbuf", "wgbuf", "outbuf"]:
            if buf_type=="outbuf":
                latency_buf_rd[buf_type]=latency_buf(compute_json_data, SA_size_x+SA_size_y-1)
            else:
                latency_buf_rd[buf_type]=latency_buf(compute_json_data, 2*SA_size_x+2*SA_size_y-1)

        #Accu
        latency_accum = compute_json_data["latency_accum"]* row["num_adder_stages"]
        tile_latency[chip_tile_idx]=max(latency_pe,max(latency_buf_rd.values()), latency_accum)*no_inp_cycles/row["Clock Frequency (Hz)"]
        compute_latency[chip_tile_idx]=tile_latency[chip_tile_idx]
        tile_latency_breakdown[chip_tile_idx]={"latency_pe": latency_pe*no_inp_cycles/row["Clock Frequency (Hz)"],
                                               "latency_buf_rd": {k: v*no_inp_cycles/row["Clock Frequency (Hz)"] for k, v in latency_buf_rd.items()},
                                               "latency_accum": latency_accum*no_inp_cycles/row["Clock Frequency (Hz)"],
                                               "no_inp_cycles": no_inp_cycles,
                                               "util": util,
                                               "F_actpes": SA_size_x/(SA_size_x+SA_size_y-1)}
    return tile_latency, tile_latency_breakdown, compute_latency


def energy_sa_tile(sa_df,compute_json_data,tile_energy,tile_latency_breakdown, tile_latency,tile_buf_df, lut_df):
    compute_energy={}
    tile_energy_breakdown={}
    for _ , row in sa_df.iterrows():
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        SA_size_x=row["SA_size_x"]
        SA_size_y=row["SA_size_y"]
        n_SA=row["n_SA"]

        #PE
        power_pe=compute_json_data["Power_sa_per_pe"]*(SA_size_x*SA_size_y-SA_size_y)+compute_json_data["Power_sa_per_pe_toprow"]*SA_size_y
        power_pe*= n_SA
        energy_pe = power_pe*tile_latency_breakdown[chip_tile_idx]["latency_pe"]*tile_latency_breakdown[chip_tile_idx]["F_actpes"]*tile_latency_breakdown[chip_tile_idx]["util"]["SA_size_x"]*tile_latency_breakdown[chip_tile_idx]["util"]["SA_size_y"]*tile_latency_breakdown[chip_tile_idx]["util"]["n_SA"]
        #Ctl
        power_ctl=(compute_json_data["Power_ctl_a"]*SA_size_y+compute_json_data["Power_ctl_b"])*n_SA
        energy_clt=power_ctl*tile_latency[chip_tile_idx]

        #Buff
        energy_buf=0
        for buf_type in ["inbuf", "wgbuf", "outbuf"]:
            row_buf_1=tile_buf_df[chip_tile_idx]
            row_buf = row_buf_1[(row_buf_1["Tile ID"] == row["Tile ID"]+"_"+buf_type)].iloc[0]
            NW=row_buf["NW"]
            NB=row_buf["NB"]
            CM=row_buf["CM"]
            lut_row = lut_df[(lut_df["NW"] == NW) & (lut_df["NB"] == NB) & (lut_df["CM"] == CM)]
            energy_buf += determine_buffer_energy(lut_row["P_freq_uW_MHz_read"], lut_row["P_freq_uW_MHz_write"],row["Clock Frequency (Hz)"], lut_row["K_lut_read"], lut_row["K_lut_write"], row_buf["F_actbitlines"], row_buf["F_non_zeros_r"],row_buf["F_non_zeros_w"], tile_latency_breakdown[chip_tile_idx]["latency_buf_rd"][buf_type], 0)

        #Accu
        energy_accum = compute_json_data["energy_per_adder"]*row["prec"]*row["num_adders_tile"]*tile_latency_breakdown[chip_tile_idx]["latency_accum"]
        tile_energy[chip_tile_idx]=energy_pe+energy_clt+energy_buf+energy_accum
        compute_energy[chip_tile_idx]=tile_energy[chip_tile_idx]
        tile_energy_breakdown[chip_tile_idx]={"energy_pe": energy_pe,
                                              "energy_clt": energy_clt,
                                              "energy_buf": energy_buf,
                                              "energy_accum": energy_accum}
    return tile_energy, tile_energy_breakdown, compute_energy
