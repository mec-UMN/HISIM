import math
import numpy as np
def nearest_ceil_value(arr, val):
    arr = np.array(arr)
    if np.any(arr >= val):
        return np.min(arr[arr >= val])
    else:
        return np.max(arr)

def area_mem_tile(mem_df, compute_json_data, tile_area, lut_df):
    #print(mem_df)
    for idx, row in mem_df.iterrows():
        Nbank=row["Nbank"]
        CM=row["CM"]
        if "F_actbitlines" not in row:
            NW=nearest_ceil_value(lut_df["NW"], row["NW"])
            NB=nearest_ceil_value(lut_df["NB"], row["NB"])
            mem_df.loc[idx, "NW_real"] = NW
            mem_df.loc[idx, "NB_real"] = NB
            mem_df.loc[idx, "F_actbitlines"] = row["NB"]/NB
        else:
            NW=row["NW"]
            NB=row["NB"]
        #print(NW, NB, Nbank, CM)
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        #print(chip_tile_idx)
        width = compute_json_data["W_other"] + compute_json_data["W_cell"] * CM * NB + math.floor((NB - 1) / (compute_json_data["MP1"]  / CM)) * compute_json_data["CenDec"]  * compute_json_data["W_dri"]
        height = compute_json_data["H_other"] + (compute_json_data["H_cell"]  * NW) / CM
        if chip_tile_idx in tile_area:
            print("ERROR: Duplicate Data Detected. Please check your tile assignment in input files")
        tile_area[chip_tile_idx] = width * height *Nbank #area in mm2
    #print(tile_area, mem_df)
    return tile_area

def latency_buf(compute_json_data, total_accesses):
    latency=compute_json_data["Latency_per_mem_access"]+total_accesses*compute_json_data["Latency_per_mem_access_to_access"]
    return latency

def load_ddr_config():
    ddr_config={
        "DDR3_1600H_2Gb_x8": {
            "ddr_clock_frequency": 0.8e9, #in Hz
            "ddr_bus_width_burst": 8*64, #in bits
            "ddr_row_latency_burst": 4, #in number of bursts
            "ddr_no_cols_per_row": 1024, #in number of columns
            "row activation_latency": 9, #in cycles
            "refresh_latency_read": 152, #in cycles
            "refresh_latency_write": 170, #in cycles
            "ddr_refresh_window": 6240, #in Hz
            "ddr_energy_per_bit": 8.75e-12 #in Joules
        }
    }
    return ddr_config

def latency_ddr(compute_json_data, row ):
    ddr_config_name = compute_json_data["ddr_config_name"]
    ddr_config = load_ddr_config()
    if ddr_config_name not in ddr_config:
        print("ERROR: Unsupported DDR configuration. Please check your DDR configuration name in input files")
        sys.exit(1)
    next_layer_type=row["Next Layer Type"]
    #read
    #print(row)
    ddr_wrap_size = row["mem_req"]
    N_raccess_rd= math.ceil(ddr_wrap_size/ddr_config[ddr_config_name]["ddr_bus_width_burst"])*row["ddr_reuse_mul_r"]
    latency_rd= N_raccess_rd*ddr_config[ddr_config_name]["ddr_row_latency_burst"]
    latency_rd = latency_rd + math.ceil(latency_rd/ddr_config[ddr_config_name]["ddr_row_latency_burst"]/ddr_config[ddr_config_name]["ddr_no_cols_per_row"])* ddr_config[ddr_config_name]["row activation_latency"]
    latency_rd = latency_rd + math.floor(latency_rd/ddr_config[ddr_config_name]["ddr_refresh_window"])*(ddr_config[ddr_config_name]["refresh_latency_read"]-2*ddr_config[ddr_config_name]["row activation_latency"])
    latency_rd = latency_rd/ ddr_config[ddr_config_name]["ddr_clock_frequency"]
    #write
    ddr_wrap_size = row["mem_req"]
    N_raccess_wr=math.ceil(ddr_wrap_size/ddr_config[ddr_config_name]["ddr_bus_width_burst"])*row["ddr_reuse_mul_w"]
    latency_wr=N_raccess_wr*ddr_config[ddr_config_name]["ddr_row_latency_burst"]
    latency_wr = latency_wr + math.ceil(latency_wr/ddr_config[ddr_config_name]["ddr_row_latency_burst"]/ddr_config[ddr_config_name]["ddr_no_cols_per_row"])* ddr_config[ddr_config_name]["row activation_latency"]
    latency_wr = latency_wr + math.floor(latency_wr/ddr_config[ddr_config_name]["ddr_refresh_window"])*(ddr_config[ddr_config_name]["refresh_latency_write"]-2*ddr_config[ddr_config_name]["row activation_latency"])
    latency_wr = latency_wr/ ddr_config[ddr_config_name]["ddr_clock_frequency"]
    return latency_rd, latency_wr, N_raccess_rd+N_raccess_wr, ddr_config[ddr_config_name]["ddr_bus_width_burst"]

def latency_mem_tile(mem_df, compute_json_data, tile_latency,tile_latency_breakdown):
    mem_latency, ddr_latency = {}, {}
    for _, row in mem_df.iterrows():
        if row["HW Type"]!="DDR":
            chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
            #print(NW, NB, Nbank, CM)
            SRAM_read_latency=latency_buf(compute_json_data,  row["Accesses_r"])/row["Clock Frequency (Hz)"]
            SRAM_write_latency=latency_buf(compute_json_data,  row["Accesses_w"])/row["Clock Frequency (Hz)"]
            tile_latency_breakdown[chip_tile_idx]={"SRAM_read_latency": SRAM_read_latency, 
                                                    "SRAM_write_latency": SRAM_write_latency}     
            #print(row)
            #print(chip_tile_idx,row["Chiplet ID"],row["Tile ID"])
            tile_latency[chip_tile_idx]=SRAM_read_latency+SRAM_write_latency
            mem_latency[chip_tile_idx]=tile_latency[chip_tile_idx]
            if row["DDR_access"]:
                latency_rd_ddr, latency_wr_ddr, N_accesses_ddr,bus_width_ddr=latency_ddr(compute_json_data, row)
                tile_latency_breakdown[chip_tile_idx].update({"DDR_read_latency": latency_rd_ddr, "DDR_write_latency": latency_wr_ddr, "DDR_accesses": N_accesses_ddr, "DDR_bus_width": bus_width_ddr})
                ddr_latency[chip_tile_idx]=latency_rd_ddr+latency_wr_ddr
            else:
                tile_latency_breakdown[chip_tile_idx].update({"DDR_read_latency": 0, "DDR_write_latency": 0, "DDR_accesses": 0, "DDR_bus_width": 0})
    return tile_latency, tile_latency_breakdown, mem_latency, ddr_latency

def energy_mem_tile(mem_df, compute_json_data, tile_energy, tile_latency_breakdown, lut_df):
    mem_energy, ddr_energy = {}, {}
    ddr_config_name = compute_json_data["ddr_config_name"]
    ddr_config = load_ddr_config()
    #import pdb; pdb.set_trace()
    for _, row in mem_df.iterrows():
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        #print(row)
        lut_row = lut_df[(lut_df["NW"] == row["NW_real"]) & (lut_df["NB"] == row["NB_real"]) & (lut_df["CM"] == row["CM"])]
        if lut_row.empty:
            print("ERROR: No LUT entry found for the given memory configuration. Please check your memory configuration.")
            sys.exit(1)
        #print(NW, NB, Nbank, CM)
        if row["HW Type"]!="DDR":# to avoid duplicate calculation for DDR
            N_ddr_accesses=tile_latency_breakdown[chip_tile_idx]["DDR_accesses"]
            ddr_energy[chip_tile_idx]= (ddr_config[ddr_config_name]["ddr_energy_per_bit"]*N_ddr_accesses*row["prec"])
            #print(lut_row)
        tile_energy[chip_tile_idx]=determine_buffer_energy(lut_row["P_freq_uW_MHz_read"], lut_row["P_freq_uW_MHz_write"],row["Clock Frequency (Hz)"], lut_row["K_lut_read"], lut_row["K_lut_write"], row["F_actbitlines"], 0.5, 0.5, tile_latency_breakdown[chip_tile_idx]["SRAM_read_latency"], tile_latency_breakdown[chip_tile_idx]["SRAM_write_latency"])
        mem_energy[chip_tile_idx]=tile_energy[chip_tile_idx]
    return tile_energy, mem_energy, ddr_energy

def determine_buffer_energy(P_freq_uW_MHz_read, P_freq_uW_MHz_write, Clock_Frequency_Hz, K_lut_read, K_lut_write, F_actbitlines, F_non_zeros_r, F_non_zeros_w, SRAM_read_latency, SRAM_write_latency):
    if K_lut_read.values[0]=="TBC" or K_lut_write.values[0]=="TBC":
        print("Note: Memory configuration yet to be calibration. Default calibration parameters are being used.")
        K_lut_read.values[0]=0.5018
        K_lut_write.values[0]=0.4501
    energy_read = (P_freq_uW_MHz_read.values[0] * Clock_Frequency_Hz / 1e12 * float(K_lut_read.values[0]) * F_actbitlines * F_non_zeros_r) * SRAM_read_latency
    energy_write = (P_freq_uW_MHz_write.values[0] * Clock_Frequency_Hz / 1e12 * float(K_lut_write.values[0]) * F_actbitlines * F_non_zeros_w) * SRAM_write_latency
    return (energy_read + energy_write) 

def update_ddr_mem_latency(mem_df, compute_json_data, tile_latency, tile_latency_breakdown, mem_latency, tile_map, DDR_access_table, mem_req, mem_size):
    ddr_mem_req = sum([(v["DDR_accesses"])*v["DDR_bus_width"] for k, v in tile_latency_breakdown.items() if "DDR_accesses" in v])
    chip_tile_idx=tile_map["DDR_Mem"]
    mask = (mem_df["Chiplet ID"] == chip_tile_idx.split('_')[0]) & (mem_df["Tile ID"]   == chip_tile_idx.split('_')[1])
    mem_req[chip_tile_idx]={k:v["DDR_accesses"]*v["DDR_bus_width"] for k, v in tile_latency_breakdown.items() if "DDR_accesses" in v}
    DDR_access_table.add_row(["DDR", "Mem", chip_tile_idx.split('_')[0], chip_tile_idx.split('_')[1], mem_size["DDR_Mem"]*1E-6/8, ddr_mem_req*1E-6/8, True])
    accesses = math.ceil(ddr_mem_req/mem_df.loc[mask, "NB"].iloc[0])
    mem_df.loc[mask, "Accesses_r"]= accesses
    mem_df.loc[mask, "Accesses_w"]= accesses
    mem_df.loc[mask, "mem_req"]=ddr_mem_req
    SRAM_read_latency=latency_buf(compute_json_data,  accesses)/mem_df.loc[mask, "Clock Frequency (Hz)"].iloc[0]
    SRAM_write_latency=latency_buf(compute_json_data,  accesses)/mem_df.loc[mask, "Clock Frequency (Hz)"].iloc[0]
    tile_latency_breakdown[chip_tile_idx]={"SRAM_read_latency": SRAM_read_latency, 
                                            "SRAM_write_latency": SRAM_write_latency}     
    #print(chip_tile_idx,row["Chiplet ID"],row["Tile ID"])
    tile_latency[chip_tile_idx]=SRAM_read_latency+SRAM_write_latency
    mem_latency[chip_tile_idx]=tile_latency[chip_tile_idx]
    #import pdb; pdb.set_trace()
    return tile_latency, tile_latency_breakdown, mem_latency, DDR_access_table, mem_req, mem_df