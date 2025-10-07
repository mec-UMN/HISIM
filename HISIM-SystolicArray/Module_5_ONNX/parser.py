
### #####################################################3
### Parser Script to parse the tmp files and generate csv 
### ###################################################

import re
import pandas as pd
from pathlib import Path

## Define the dimension size and no of inputs/outputs
dimension_size = 4
no_of_inputs = 2
no_of_outputs = 1


# Define the regular expression pattern to match the different column values
reg_exp = re.compile(
    r'^\s*(?P<layer_id>%[\w\:]+(?:\s*,\s*%\w+)*)\s*=\s*'
    r'(?:"(?P<operation>[^"]+)"|(?P<operation_wo_quotes>[A-Za-z_][\w.\d]+))'
    r'(?:\(\s*(?P<edges>.*?)\s*\))?\s*'
    r'(?P<node_detail>\{.*?\})?.*\s*:\s*(?P<tensor_attr>.+?)\s*$'
)

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
    
    cols.append('Node_Name')
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
def extract_dimensions(tensor_list,count):
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
        print(match)
        if not match:  # If no match found skip the line.
            print("No match found, skipping line.")
            continue
        
        # Extract the layer_id
        layer_ids = match.group('layer_id') or ''
        #print("Layer IDs :", layer_ids)

        # Extract only the first layer_id if multiple are present
        layer_id_list = layer_ids.split(',')
        layer_id = layer_id_list[0].strip()
        print("Layer ID:", layer_id)

       
        # Extract the edge details 
        edges = match.group('edges') or  ''
        if edges:
            print("Edges:" ,edges)
            # Extract only the first edge if multiple are present
            edge_list = edges.split(',')
            edge = edge_list[0]
            #print("Edge:",edge)

            ## Extract the data for Edge.csv
            for e in edge_list:
                for l in layer_id_list:
                    edge_rows.append((e.strip(), l.strip()))

        # Extract the operation type
        operation = match.group('operation') or  match.group('operation_wo_quotes') or ''
        #print("Operation:", operation)
        op_match  = re.match(r'onnx\.([A-Za-z_][\w.\d]+)', operation)
        op_type = op_match.group(1)
        ## For Normalization ops, add suffix (first edge) to the operation type
        if op_type == 'LayerNormalization':
            op_type = 'Normalization_' + edge       
        print("Operation Type :",op_type)

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
        
        # Extract the tensor attributes
        tensor_attr = match.group('tensor_attr') or ''
        input_tensor_list, output_tensor_list = parse_tensors(tensor_attr)
        #print("Input Tensor list :" ,input_tensor_list)
        #print("Output Tensor List :",output_tensor_list)
        
        
        # Extract the dimension values of input and output tensors
        input_dim_list = extract_dimensions(input_tensor_list,no_of_inputs)
        output_dim_list = extract_dimensions(output_tensor_list,no_of_outputs)
        print("Input dimension list" ,input_dim_list)
        print("Output dimension list", output_dim_list)
        
       
        ## Create the row with extracted values
        rows.append([layer_id, op_type] + input_dim_list + output_dim_list + [node_name])
     
    return rows, edge_rows
    

if __name__ == "__main__":

    ## Example cases for testing
    #example = '%225 = "onnx.Softmax"(%224) {axis = -1 : si64, onnx_node_name = "/vit/encoder/layer.0/attention/attention/Softmax"} : (tensor<1x12x197x197xf32>) -> tensor<1x12x197x197xf32>'
    #example = '%Y_15, %Mean_16, %InvStdDev_17 = "onnx.LayerNormalization"(%410, %226, %225) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, onnx_node_name = "Add_675_89", stash_type = 1 : si64} : (tensor<1x8x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x8x768xf32>, none, none)'
    #example = '%171 = onnx.Constant dense_resource<__elided__> : tensor<768x768xf32>'
    #example = '%678:3 = "onnx.Split"(%677, %159) {axis = 2 : si64, onnx_node_name = "Split_2381_49"} : (tensor<1x8x2304xf32>, tensor<3xi64>) -> (tensor<1x8x768xf32>, tensor<1x8x768xf32>, tensor<1x8x768xf32>)'
    #example  = '%79 = onnx.Constant dense<[[[[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00], [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]]]]> : tensor<1x1x8x8xf32>'
    #print(example)
    #rows,edge_rows = parse_lines_to_rows([example])
    #df = pd.DataFrame(edge_rows, columns=['Start_Layer','End_Layer'])
    #df = df_edge.drop_duplicates().reset_index(drop=True)
    #print(df)

    
    ## Read in the tmp file 
    data = Path("mobilenet.tmp").read_text().splitlines() #Specify the path of the tmp file 
    rows,edge_rows = parse_lines_to_rows(data)
    df_edge = pd.DataFrame(edge_rows, columns=['Start_Layer','End_Layer'])
    df_edge = df_edge.drop_duplicates().reset_index(drop=True)
    
    ## Generate the column names based on inputs, outputs and dimension size
    cols = generate_column_names(no_of_inputs, no_of_outputs, dimension_size)

    df = pd.DataFrame(rows, columns=cols) ## Dataframe creation
    network_csv = "network.csv" # Specify the  network csv file 
    edge_csv = "edge.csv" # Specify the edge csv file
    df.to_csv(network_csv, index=False) # Write the DataFrame to a CSV file
    df_edge.to_csv(edge_csv, index=False) # Write the edge DataFrame to a CSV file
    print( "Length of the Network DataFrame ", len(df))
    print( "Length of the Edge DataFrame ", len(df_edge))
    print(f"DataFrame is saved to the csv files : {network_csv} and {edge_csv}")
    