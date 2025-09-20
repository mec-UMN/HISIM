import math

def area_cpu_tile(cpu_df,compute_json_data, tile_area):
    #source from json
    #print(cpu_df)
    for _, row in cpu_df.iterrows():
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        if chip_tile_idx in tile_area:
            print("ERROR: Duplicate Data Detected. Please check your tile assignment in input files")
        tile_area[chip_tile_idx]=compute_json_data["Area_CPU"]
    return tile_area

def latency_cpu_tile(cpu_df,compute_json_data, tile_latency, tile_latency_breakdown, compute_latency):
    for _, row in cpu_df.iterrows():
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        layer_type=row["Type"]
        values=[row[f"indim{i}"]for i in range(1, 5)]
        total_instructions=3*math.prod([int(v) if not math.isnan(v) else 1 for v in values])
        if layer_type=="Relu":
            total_instructions+=1
        elif layer_type=="Batchnorm":
            total_instructions+=5*math.prod([int(v) if not math.isnan(v) else 1 for v in values[0:2]])
        elif layer_type=="Add":
            total_instructions+=2*math.prod([int(v) if not math.isnan(v) else 1 for v in values[0:2]])
        elif layer_type=="ScalarMul":
            total_instructions+=1
        elif layer_type=="Softmax":
            total_instructions+=6*math.prod([int(v) if not math.isnan(v) else 1 for v in values])+1
        else:
            print("Unsupported layer type for CPU:", layer_type)
            sys.exit(1)
        tile_latency[chip_tile_idx]=total_instructions*compute_json_data["Average_CPI_"+layer_type]/row["Clock Frequency (Hz)"]
        compute_latency[chip_tile_idx]=tile_latency[chip_tile_idx]
        tile_latency_breakdown[chip_tile_idx]={"total_instructions": total_instructions,
                                               "CPI": compute_json_data["Average_CPI_"+layer_type],
                                               "Clock Frequency (Hz)": row["Clock Frequency (Hz)"]}

    return tile_latency, tile_latency_breakdown, compute_latency

def energy_cpu_tile(cpu_df,compute_json_data,tile_energy, tile_latency, compute_energy):
    for _, row in cpu_df.iterrows():
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        tile_energy[chip_tile_idx]=compute_json_data["Power_CPU"]*tile_latency[chip_tile_idx]
        compute_energy[chip_tile_idx]=tile_energy[chip_tile_idx]
    return tile_energy, compute_energy