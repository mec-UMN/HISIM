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
    #import pdb; pdb.set_trace()
    for _, row in cpu_df.iterrows():
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        tile_latency_breakdown[chip_tile_idx]={}
        tile_latency[chip_tile_idx]=0
        for layer_idx, layer_id in enumerate(row["Layer_ID"]):
            average_cpi_term = None
            layer_type=row["Type"][layer_idx]
            values=[row[f"in1_dim{i}"][layer_idx]for i in range(1, 5)]
            total_instructions=3*math.prod([int(v) if not math.isnan(v) else 1 for v in values])
            if layer_type=="Relu":
                total_instructions+=1
            elif layer_type=="Batchnorm":
                total_instructions+=5*math.prod([int(v) if not math.isnan(v) else 1 for v in values[0:2]])
            elif layer_type=="Add": # to be verified
                total_instructions+=2*math.prod([int(v) if not math.isnan(v) else 1 for v in values[0:2]])
            elif layer_type=="ScalarMul": # to be verified
                total_instructions+=1
            elif layer_type=="Softmax":
                total_instructions+=10*math.prod([int(v) if not math.isnan(v) else 1 for v in values])+1
            elif layer_type.endswith("LayerNormalization") or layer_type.endswith("LayerNorm"): 
                total_instructions+=5*math.prod([int(v) if not math.isnan(v) else 1 for v in values[0:2]])
                average_cpi_term="LayerNormalization"
            elif layer_type=="Tanh": # To be Verified
                #pade approximation tanh(x)= x*(27+ x^2)/(27+9*x^2) 
                #=> total additional instructions other than iniitally considered 3 operations:
                # 2 to load 27 and 9, 2 for multiplication and addition (fmadd operation) and 1 for division
                total_instructions+=2 # 2 to load 27 and 9
                total_instructions+=3*math.prod([int(v) if not math.isnan(v) else 1 for v in values]) # 2 for multiplication and addition (fmadd operation) and 1 for division
            elif layer_type.endswith("Gelu"): # To be Verified
                #fastgelu approximation with 2 terms gelu(x)=0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) 
                #=> total additional instructions other than iniitally considered 3 operations: 
                #3 to load 0, sqrt(2/pi) and 0.044715, 
                #4 for multiplication and addition (fmadd operation - x*x+0, x^2*x+0, x^3*0.044715+x, (x^3*0.044715+x)*sqrt(2/pi)) 
                #then operations required for tanh pade approximation with 2 terms tanh(x)= x*(27+ x^2)/(27+9*x^2) 
                #1 for addition (1+tanh term), 1 for multiplication with x
                total_instructions+=3 # 3 to load 0, sqrt(2/pi) and 0.044715
                total_instructions+=4*math.prod([int(v) if not math.isnan(v) else 1 for v in values]) # 4 for multiplication and addition (fmadd operation - x*x+0, x^2*x+0, x^3*0.044715+x, (x^3*0.044715+x)*sqrt(2/pi))
                total_instructions+=3*math.prod([int(v) if not math.isnan(v) else 1 for v in values])+2 # then operations required for tanh pade approximation - see above
                total_instructions+=2*math.prod([int(v) if not math.isnan(v) else 1 for v in values]) #  1 for addition (1+tanh term), 1 for multiplication with x
                average_cpi_term="Gelu"
            elif layer_type=="Clip": # To be Verified
                total_instructions+=2
            elif layer_type=="Sigmoid": # To be Verified
                #taylor series expansion with 2 terms e^x=1+x+x^2/2 and sigmoid(x)=e^x/(e^x+1)= 2+2x+x^2/(4+2x+x^2)
                #=> total additional instructions other than iniitally considered 3 operations = 2 to load numbers 2 and 4, 2 for multiplication and addition (fmadd operation) and 1 for division
                total_instructions+=2 # 2 to load numbers 2 and 4
                total_instructions+=3*math.prod([int(v) if not math.isnan(v) else 1 for v in values]) # 2 for multiplication and addition (fmadd operation) and 1 for division
            else:
                print("Unsupported layer type for CPU:", layer_type)
                exit(1)
            average_cpi= compute_json_data["Average_CPI_"+layer_type] if average_cpi_term is None else compute_json_data["Average_CPI_"+average_cpi_term]
            tile_latency[chip_tile_idx]+=total_instructions*average_cpi/row["Clock Frequency (Hz)"]
            compute_latency[chip_tile_idx]=tile_latency[chip_tile_idx]
            tile_latency_breakdown[chip_tile_idx][layer_idx]={"total_instructions": total_instructions,
                                                "CPI": average_cpi,
                                                "Clock Frequency (Hz)": row["Clock Frequency (Hz)"]}
    #import pdb; pdb.set_trace()
    return tile_latency, tile_latency_breakdown, compute_latency

def energy_cpu_tile(cpu_df,compute_json_data,tile_energy, tile_latency, compute_energy):
    for _, row in cpu_df.iterrows():
        chip_tile_idx=row["Chiplet ID"]+"_"+row["Tile ID"]
        tile_energy[chip_tile_idx]=compute_json_data["Power_CPU"]*tile_latency[chip_tile_idx]
        compute_energy[chip_tile_idx]=tile_energy[chip_tile_idx]
    return tile_energy, compute_energy