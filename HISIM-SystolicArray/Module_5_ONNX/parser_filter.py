
### #####################################################3
### Parser Script to parse the tmp files and generate csv 
### ###################################################

import re
import pandas as pd
from pathlib import Path
import os

## Define the dimension size and no of inputs/outputs
dimension_size = 4
no_of_inputs = 3
no_of_outputs = 1


# Define the regular expression pattern to match the different column values
reg_exp = re.compile(
    r'^\s*(?P<layer_id>%[\w\:]+(?:\s*,\s*%\w+)*)\s*=\s*'
    r'(?:"(?P<operation>[^"]+)"|(?P<operation_wo_quotes>[A-Za-z_][\w.\d]+))'
    r'(?:\(\s*(?P<edges>.*?)\s*\))?\s*'
    r'(?P<node_detail>\{.*?\})?.*\s*:\s*(?P<tensor_attr>.+?)\s*$'
)

current_dir = os.path.dirname(__file__)
main_dir = os.path.dirname(current_dir)

# Function to generate the columns names based on the number of inputs, outputs, and dimension size
def generate_column_names(no_of_inputs, no_of_outputs, dimension_size):
    cols = ['Layer_ID', 'Type']
    
    for i in range(1,no_of_inputs+1):
        for d in range(1, dimension_size+1):
            cols.append(f'in{i}_dim{d}')
    
    for o in range(1,no_of_outputs+1):
        for d in range(1, dimension_size+1):
            #print(no_of_outputs)
            #print(o,d)
            cols.append(f'out{o}_dim{d}')
    
    cols.append('NodeName')
    cols.append('Dilations')
    cols.append('Groups')
    cols.append('KernelShape')
    cols.append('Pads')
    cols.append('Strides')

    return cols


# Function to parse the input and output tensors
def parse_tensors(tensor_attr):
    t = tensor_attr.strip()
    list_tensors = re.match(r'^\((.*)\)\s* ->\s*(.*)$',t)
    if list_tensors:
        inputs,outputs =list_tensors.groups()
        inputs = [i.strip() for i in inputs.split(',')]
        outputs = [o.strip() for o in outputs.split(',')]
    else:  # Matches the case with single tensor<>
        inputs = []
        outputs = [t]
    return inputs, outputs

#Function to extract dimensions from the tensor list
def extract_dimensions(tensor_list,count, op_type):
    dimension_list =[]
    for i in range(count):
            if i < len(tensor_list):
                tensor = tensor_list[i]  
                dimensions = re.search(r'tensor<([^>]+)>', tensor)
                if dimensions:
                    dimension = dimensions.group(1)
                    #print(i_dims)
                    i_dims = dimension.split('x')
                    #print(i_dims)
                    dim_tokens = i_dims[:-1]
                    #print(dim_tokens)
                    for d in range(dimension_size):
                        if d < len(dim_tokens):
                            dimension_list.append(dim_tokens[d])
                            #print(dim_tokens[d])
                        else:
                            dimension_list.append('NA')
                            #print('NA')
                else:
                    for d in range(dimension_size):
                        dimension_list.append('NA')
                        #print('NA')
            else:
                for d in range(dimension_size):
                    dimension_list.append('NA')
                    #print('NA')

    return dimension_list


# Function to filter out edge roews based on the filtered dataframe
def filter_edge_rows(df_edge, filter_df):
    filtered_edge_rows = []
    valid_layers = set(filter_df['Original_ID'])
    #import pdb; pdb.set_trace()
    for start_layer, end_layer in df_edge:
        if start_layer in valid_layers and end_layer in valid_layers:
            new_start_layer = filter_df.loc[filter_df['Original_ID'] == start_layer, 'Layer_ID'].values[0]
            new_end_layer = filter_df.loc[filter_df['Original_ID'] == end_layer, 'Layer_ID'].values[0]
            filtered_edge_rows.append((new_start_layer, new_end_layer))
        # If only the start_layer is in valid_layers, map it and keep end_layer as next layer in valid_layers
        elif start_layer in valid_layers:
            set_idx_start = filter_df.index[filter_df['Original_ID'] == start_layer].tolist()[0]
            if set_idx_start+1 < len(filter_df):
                end_layer_valid= filter_df.loc[set_idx_start+1, 'Original_ID'] 
                filtered_edge_rows.append((filter_df.loc[filter_df['Original_ID'] == start_layer, 'Layer_ID'].values[0], filter_df.loc[filter_df['Original_ID'] == end_layer_valid, 'Layer_ID'].values[0]))
            # If only the end_layer is in valid_layers, map it and keep start_layer as is
        elif end_layer in valid_layers:
            set_idx_end = filter_df.index[filter_df['Original_ID'] == end_layer].tolist()[0]
            if set_idx_end-1 >= 0:
                start_layer_valid= filter_df.loc[set_idx_end-1, 'Original_ID']
                filtered_edge_rows.append((filter_df.loc[filter_df['Original_ID'] == start_layer_valid, 'Layer_ID'].values[0], filter_df.loc[filter_df['Original_ID'] == end_layer, 'Layer_ID'].values[0]))    
    #remove duplicates from filtered_edge_rows without changing the order
    filtered_edge_rows = list(dict.fromkeys(filtered_edge_rows))
    return filtered_edge_rows

## Function to parse lines and extract the row data
def parse_lines_to_rows(lines):
    rows = []
    edge_rows = []
    for row in lines:
        line = row.strip()
        #print("Line :",line)
        if not line or not line.startswith("%"): # Skip the lines that do not start with '%'
            continue
        
        match = reg_exp.match(line)
        #print(match)
        if not match:  # If no match found skip the line.
            #print("No match found, skipping line.")
            continue
        
        # Extract the layer_id
        layer_ids = match.group('layer_id') or ''
        #print("Layer IDs :", layer_ids)

        # Extract only the first layer_id if multiple are present
        layer_id_list = layer_ids.split(',')
        layer_id = layer_id_list[0].strip()
        #print("Layer ID:", layer_id)

       
        # Extract the edge details 
        edges = match.group('edges') or  ''
        if edges:
            #print("Edges:" ,edges)
            # Extract only the first edge if multiple are present
            edge_list = edges.split(',')
            edge = edge_list[0]
            #print("Edge:",edge)

            ## Extract the data for Edge.csv
            for e in edge_list:
                #for l in layer_id_list:
                    #edge_rows.append((e.strip(), l.strip()))
                edge_rows.append((e.strip(), layer_id))  ## Only consider the first layer_id if there are multiple layer_ids

        # Extract the operation type
        operation = match.group('operation') or  match.group('operation_wo_quotes') or ''
        #print("Operation:", operation)
        op_match  = re.match(r'onnx\.([A-Za-z_][\w.\d]+)', operation)
        op_type = op_match.group(1)
        ## For Normalization ops, add suffix (first edge) to the operation type
        #if op_type == 'LayerNormalization':
        #    op_type = 'Normalization_' + edge       
        #print("Operation Type :",op_type)

        # Extract the node details 
        node_detail = match.group('node_detail') or ''
        #print("Node Details :",node_detail)
       
        #Extract node_name from the node_detail
        node_match = re.search(r'onnx_node_name\s*=\s*"([^"]+)"', node_detail)
        if node_match:
            node_name = node_match.group(1)
        else:
            node_name = "NA"
        #print("Node Name :", node_name)
        if op_type == 'Custom':
            op_type = node_name.split('/')[-1]  
        

        # Extract the dilations, groups, kernel_shape, pads, strides from node_detail
        dilations_match = re.search(r'dilations\s*=\s*(\[[^\]]+\])', node_detail)
        
        dilations = dilations_match.group(1) if dilations_match else 'NA'
        #print("Dilations :", dilations)

        groups_match = re.search(r'group\s*=\s*([^,]+)', node_detail)
        groups = groups_match.group(1) if groups_match else 'NA'
        #print("Groups :", groups)

        kernel_shape_match = re.search(r'kernel_shape\s*=\s*(\[[^\]]+\])', node_detail)
        kernel_shape = kernel_shape_match.group(1) if kernel_shape_match else 'NA'
        #print("Kernel Shape :", kernel_shape)

        pads_match = re.search(r'pads\s*=\s*(\[[^\]]+\])', node_detail)
        pads = pads_match.group(1) if pads_match else 'NA'
        #print("Pads :", pads)

        strides_match = re.search(r'strides\s*=\s*(\[[^\]]+\])', node_detail)
        strides = strides_match.group(1) if strides_match else 'NA'
        #print("Strides :", strides)

        # Extract the tensor attributes
        tensor_attr = match.group('tensor_attr') or ''
        input_tensor_list, output_tensor_list = parse_tensors(tensor_attr)
        #print("Input Tensor list :" ,input_tensor_list)
        #print("Output Tensor List :",output_tensor_list)
        
        
        # Extract the dimension values of input and output tensors
        input_dim_list = extract_dimensions(input_tensor_list,no_of_inputs, op_type)
        output_dim_list = extract_dimensions(output_tensor_list,no_of_outputs, op_type)
        #print("Input dimension list" ,input_dim_list)
        #print("Output dimension list", output_dim_list)
        #input dim 0 to dim_size-1 and dim_size to 2*dim_size-1 if the layer type is 'Gather'
        #if op_type == 'Gather':
            #input_dim_list = input_dim_list[dimension_size: dimension_size*2]+ input_dim_list[0:dimension_size] + input_dim_list[dimension_size*2:]
        ## Create the row with extracted values
        rows.append([layer_id, op_type] + input_dim_list + output_dim_list + [node_name] + [dilations] + [groups] + [kernel_shape] + [pads] + [strides])
     
    return rows, edge_rows
    
## Function to filter the dataframe based on the supported operations
def filter_dataframe(df):
    #print set of unique operation types in the dataframe and their counts
    set_op_types = set(df['Type'].unique())
    op_type_counts = df['Type'].value_counts()
    #print("Unique operation types in the dataframe:", set_op_types)
    #print("Operation type counts in the dataframe:", op_type_counts)
    filter_network=True
    # Define the list of operations to keep
    if filter_network:
        list = [ 'MatMul', 'Add', 'Relu', 'Softmax', 'Gelu' ,'Constant', 'LayerNormalization', 'Clip', 'Custom', 'MatMulInteger', 'Sigmoid', 'MatMul_Q4', 'GroupQueryAttention', 'Conv', 'Gemm', 'FastGelu', 'SimplifiedLayerNormalization', 'LayerNorm', 'SkipLayerNorm']
        filter_ops = ['MatMul', 'Add', 'Relu', 'Softmax', 'Gelu' ,'LayerNormalization', 'Clip', 'Custom', 'MatMulInteger', 'Sigmoid',  'MatMul_Q4', 'GroupQueryAttention', 'Conv', 'Gemm', 'FastGelu', 'SimplifiedLayerNormalization', 'LayerNorm', 'SkipLayerNorm']

        df_new = df[df['Type'].isin(list)].reset_index(drop=True)
    else:
        df_new=df
    df_new.insert(1, 'Original_ID',df_new['Layer_ID'])  # Insert a new column 'Original_ID' after 'Layer_ID')
    
    const_df = df_new[df_new['Type'] == 'Constant'].reset_index(drop=True)

    for index, row in const_df.iterrows():
        df_new.loc[df_new['Layer_ID'] == row['Layer_ID'], 'Layer_ID'] = f"C_{index+1}"
        const_df.loc[const_df['Layer_ID'] == row['Layer_ID'], 'Layer_ID'] = f"C_{index+1}"
        
    if filter_network:
        filter_df = df_new[df_new['Type'].isin(filter_ops)].reset_index(drop=True)
    else:
        filter_df=df_new

    for index, row in filter_df.iterrows():
        df_new.loc[df_new['Layer_ID'] == row['Layer_ID'], 'Layer_ID'] = f"L_{index+1}"
        filter_df.loc[filter_df['Layer_ID'] == row['Layer_ID'], 'Layer_ID'] = f"L_{index+1}"

    #normalization_df = df_new[df_new['Type'] == 'LayerNormalization'].reset_index(drop=True)

    #for index,row in normalization_df.iterrows():
     #   df_new.loc[df_new['Layer_ID'] == row['Layer_ID'], 'Layer_ID'] = f"N_{index+1}"
    
    #print operations in the filter_df and their counts
    set_filter_op_types = set(filter_df['Type'].unique())
    print("Unique operation types in the filtered dataframe:", set_filter_op_types)
    #print("Operation type counts in the filtered dataframe:", filter_df['Type'].value_counts())

    #print set of operations not in the filter_df but in the original dataframe and their counts
    not_in_filter = set_op_types - set_filter_op_types
    #print("Operation types in the original dataframe but not in the filtered dataframe:", not_in_filter)
    op_count ={op: op_type_counts[op] for op in not_in_filter}   
    #sort it by count in descending order
    #print the sorted dictionary
    sorted_op_count = dict(sorted(op_count.items(), key=lambda item: item[1], reverse=True))
    print("Operation types in the original dataframe but not in the filtered dataframe sorted by count in descending order:", sorted_op_count) 
    
    #import pdb; pdb.set_trace()
    return filter_df, const_df


        
   

def parse(model_name):

    ## Example cases for testing
    #example = '%225 = "onnx.Softmax"(%224) {axis = -1 : si64, onnx_node_name = "/vit/encoder/layer.0/attention/attention/Softmax"} : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>'
    #example = '%Y_15, %Mean_16, %InvStdDev_17 = "onnx.LayerNormalization"(%410, %226, %225) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, onnx_node_name = "Add_675_89", stash_type = 1 : si64} : (tensor<1x8x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x8x768xf32>, none, none)'
    #example = '%171 = onnx.Constant dense_resource<__elided__> : tensor<768x768xf32>'
    #example = '%678:3 = "onnx.Split"(%677, %159) {axis = 2 : si64, onnx_node_name = "Split_2381_49"} : (tensor<1x8x2304xf32>, tensor<3xi64>) -> (tensor<1x8x768xf32>, tensor<1x8x768xf32>, tensor<1x8x768xf32>)'
    #example  = '%79 = onnx.Constant dense<[[[[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]]]]> : tensor<1x1x8x8xf32>'
    #example =' %235 = "onnx.Conv"(%arg0, %127, %128) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "/mobilenet_v2/conv_stem/first_conv/convolution/Conv", pads = [0, 0, 1, 1], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x32x112x112xf32>'
    #print(example)
    #rows,edge_rows = parse_lines_to_rows([example])
    #df = pd.DataFrame(edge_rows, columns=['Start_Layer','End_Layer'])
    #df = df_edge.drop_duplicates().reset_index(drop=True)
    #print(df)
    model = model_name.split('-')[0]
    #print("Model Name :", model_name, " Model :", model)
    ## Read in the tmp file
    #filename =current_dir + "onnx-mlir_models/" + model_name + ".tmp" if os.path.exists(current_dir + "onnx-mlir_models/" + model_name + ".tmp") else current_dir + "onnx-mlir_models/" + model_name + ".mlir"
    filename =current_dir  + '/' + model_name + ".tmp" if os.path.exists(current_dir  + '/' + model_name + ".tmp") else current_dir + '/' + model_name + ".mlir"
    if not os.path.exists(filename):
        print(f"ONNX-MLIR output file not found for model: {model_name} at {filename}")
        exit(1)

    data = Path(filename).read_text().splitlines()  # Specify the path of the tmp file
    rows, edge_rows = parse_lines_to_rows(data)
    df_edge = pd.DataFrame(edge_rows, columns=['Start_Layer', 'End_Layer'])
    df_edge = df_edge.drop_duplicates().reset_index(drop=True)
    
    ## Generate the column names based on inputs, outputs and dimension size
    cols = generate_column_names(no_of_inputs, no_of_outputs, dimension_size)

    ## Initial Data frame creation
    df = pd.DataFrame(rows, columns=cols) ## Dataframe creation

    ###  Filtering the operations supported by the tool
    final_df,const_df = filter_dataframe(df)
   ## Generating the edge.csv for the filtered operations
    filtered_edge_rows = filter_edge_rows(df_edge.values, final_df)
    df_edge_filtered = pd.DataFrame(filtered_edge_rows, columns=['Start_Layer','End_Layer'])
    #df_edge = df_edge_filtered.drop_duplicates().reset_index(drop=True)
    #rename L_1 to L1, L_2 to L2 etc., in final_df and df_edge_filtered
    final_df['Layer_ID'] = final_df['Layer_ID'].str.replace('L_', 'L')
    const_df['Layer_ID'] = const_df['Layer_ID'].str.replace('C_', 'C')
    df_edge_filtered['Start_Layer'] = df_edge_filtered['Start_Layer'].str.replace('L_', 'L')
    df_edge_filtered['End_Layer'] = df_edge_filtered['End_Layer'].str.replace('L_', 'L')
    df_edge_filtered['Start_Layer'] = df_edge_filtered['Start_Layer'].str.replace('C_', 'C')
    df_edge_filtered['End_Layer'] = df_edge_filtered['End_Layer'].str.replace('C_', 'C')

    #insert new colum 'prec' to final_df with default value 8
    final_df.insert(15, 'prec', 8)
    ## Save the dataframes to csv files
    #network_csv = "network.csv" # Specify the  network csv file 
    #edge_csv = "edge.csv" # Specify the edge csv file
    target_dir = main_dir+"/Module_0_AI_Map/HISIM_2_0_AI_layer_information/" + model + "/"
    os.makedirs(target_dir, exist_ok=True)
    final_network_csv = target_dir + "Network.csv"
    final_const_csv = target_dir + "Const.csv"
    final_edge_csv = target_dir + "Edge.csv"

    #df.to_csv(network_csv, index=False) # Write the DataFrame to a CSV file
    #df_edge.to_csv(edge_csv, index=False) # Write the edge DataFrame to a CSV file
    final_df.to_csv(final_network_csv, index=False)
    const_df.to_csv(final_const_csv, index=False)
    df_edge_filtered.to_csv(final_edge_csv, index=False)

    ## Print the lengths of the dataframes
    #print( "Length of the Network DataFrame ", len(df))
    #print( "Length of the Edge DataFrame ", len(df_edge))
    #print( "Length of the Final DataFrame ", len(final_df))
    #print( "Length of the Const DataFrame ", len(const_df))
    #print(f"DataFrame is saved to the csv files : {final_network_csv} and {final_edge_csv} and {final_const_csv}")
    #print( "Length of the Final Edge DataFrame ", len(df_edge_filtered))
    #import pdb; pdb.set_trace()
    return final_network_csv, final_edge_csv, final_const_csv
    