from Module_0_AI_Map.util_chip.HISIM_2_0_Files.HW_Map import load_ai_chip, load_ai_network
from Module_1_Compute.HISIM_2_0_Files.Compute import compute_main_fn
from Module_2_Network.HISIM_2_0_Files.Network import network_main_fn
from Module_3_Cost.Cost import cost_main_fn
from Module_5_ONNX.parser_filter import parse
import time
import os
import config
import pickle
def del_files_folder(folder_path, exts):
    #delete folders and files in the given folder path
    
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    if exts=='all':
                        os.remove(file_path)
                    else:
                        if any([file_path.endswith(ext) for ext in exts]):
                            os.remove(file_path)
                #elif os.path.isdir(file_path):
                #    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

aimodel=config.aimodel
main_dir = config.main_dir
parse_mlir_output = config.parse_mlir_output
create_dfiles = config.CREATE_DEFAULT_FILES
#delete old files in Results folder
results_dir = os.path.join(main_dir, "Results")
del_files_folder(results_dir, exts ='all')
del_files_folder(os.path.join(results_dir, "Chip_Map"), exts =['.png'])
del_files_folder(os.path.join(results_dir, "Network_Map"), exts =['.png'])
#delete .png and .txt files in Network_configs folder
network_configs_dir = os.path.join(main_dir, "Module_2_Network", "HISIM_2_0_Files", "Network_configs")
del_files_folder(network_configs_dir, exts=['.png', '.txt'])

#delete .png and .txt files in HW_configs folder
HW_configs_dir = os.path.join(main_dir, "Module_1_Compute", "HISIM_2_0_Files", "HW_configs")
del_files_folder(HW_configs_dir, exts=['.png', '.txt'])

#delete .png and .txt files in HISIM_2_0_AI_layer_information folder
AI_layer_info_dir = os.path.join(main_dir, "Module_0_AI_Map", "HISIM_2_0_AI_layer_information",f"{aimodel}")
del_files_folder(AI_layer_info_dir, exts=['.png', '.txt'])

if create_dfiles:
    #delete old files starting with "Chip_Map_", "Sys_Map_", "Layer_Mapping_" in main_dir
    for filename in os.listdir(main_dir):
        if filename.startswith("Chip_Map_") or filename.startswith("Sys_Map_") or filename.startswith("Layer_Mapping_"):
            file_path = os.path.join(main_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

if parse_mlir_output:
    parse(aimodel)

#import pdb; pdb.set_trace()
start_initial = time.time()
G_ai_model=load_ai_network(aimodel)
G_sys, G_chip, G_stack, noc_tile_dict, nop_chip_dict, tier_ids, stack_ids, tile_ids, mesh_size, mesh_size_nop=load_ai_chip(f"{main_dir}/Chip_Map_{aimodel}.csv", f"{main_dir}/Sys_Map_{aimodel}.csv")
end_mapping = time.time()
print("AI mapping sim time is:", (end_mapping - start_initial),"s")

#import pdb; pdb.set_trace()
start_compute = time.time()
G_chip,tile_map, mem_req, ip_list= compute_main_fn(G_ai_model, G_chip, tile_ids)
end_computing = time.time()
print("Computing model sim time is:", (end_computing - start_compute),"s")
# import pdb; pdb.set_trace()

#import pdb; pdb.set_trace()

start_network = time.time()
G_chip, G_sys, G_stack, nw_df, stack_area, n_signal_IO =network_main_fn(G_ai_model, G_chip, G_sys, G_stack, noc_tile_dict, nop_chip_dict, tile_map, mem_req, mesh_size, mesh_size_nop, stack_ids)
end_network = time.time()
print("Network model sim time is:", (end_network - start_network),"s")
#import pdb; pdb.set_trace()

outs=(G_chip, G_sys, nw_df, stack_area, stack_ids, mesh_size, ip_list, n_signal_IO)
with open('input_params_to_cost'+'.pkl', 'wb') as file:
    pickle.dump(outs, file)

start_cost = time.time()
total_cost, cost_breakdown, cost_per_die = cost_main_fn(G_chip, G_sys, nw_df, stack_area, stack_ids, mesh_size, ip_list, n_signal_IO)
end_cost = time.time()
print("Cost model sim time is:", (end_cost - start_cost),"s")

end = time.time()
print("Total sim time is:", (end - start_initial),"s")

