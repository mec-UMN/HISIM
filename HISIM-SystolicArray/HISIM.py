from Module_AI_Map.util_chip.HISIM_2_0_Files.HW_Map import load_ai_chip, load_ai_network
from Module_Compute.HISIM_2_0_Files.Compute import compute_main_fn
from Module_Network.HISIM_2_0_Files.Network import network_main_fn
import time
import os
import config

def del_files_folder(folder_path, exts):
    #delete folders and files in the given folder path
    
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

#delete old files in Results folder
results_dir = os.path.join(main_dir, "Results")
del_files_folder(results_dir, exts ='all')

#delete .png and .txt files in Network_configs folder
network_configs_dir = os.path.join(main_dir, "Module_Network", "HISIM_2_0_Files", "Network_configs")
del_files_folder(network_configs_dir, exts=['.png', '.txt'])

#delete .png and .txt files in HW_configs folder
HW_configs_dir = os.path.join(main_dir, "Module_Compute", "HISIM_2_0_Files", "HW_configs")
del_files_folder(HW_configs_dir, exts=['.png', '.txt'])

#delete .png and .txt files in HISIM_2_0_AI_layer_information folder
AI_layer_info_dir = os.path.join(main_dir, "Module_AI_Map", "HISIM_2_0_AI_layer_information",f"{aimodel}")
del_files_folder(AI_layer_info_dir, exts=['.png', '.txt'])

#import pdb; pdb.set_trace()
start = time.time()
G_ai_model=load_ai_network(aimodel)
G_sys, G_chip, G_stack, tier_ids, stack_ids, tile_ids, mesh_size=load_ai_chip(f"{main_dir}/Chip_Map_{aimodel}.csv", f"{main_dir}/Sys_Map_{aimodel}.csv")
end_computing = time.time()
print("AI mapping sim time is:", (end_computing - start),"s")
#import pdb; pdb.set_trace()

start = time.time()
G_chip,tile_map, mem_req= compute_main_fn(G_ai_model, G_chip, tile_ids)
end_computing = time.time()
print("Computing model sim time is:", (end_computing - start),"s")

#import pdb; pdb.set_trace()

start = time.time()
G_chip=network_main_fn(G_ai_model, G_chip, G_sys, G_stack, tile_map, mem_req, mesh_size, stack_ids)
end_computing = time.time()
print("Network model sim time is:", (end_computing - start),"s")
#import pdb; pdb.set_trace()

