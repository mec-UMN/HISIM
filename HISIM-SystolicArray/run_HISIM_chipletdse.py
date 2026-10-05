import os
import math
#update config.py file with the desired aimodel and type of default files to be generated
aimodels=['mobilenetv2', 'vitbase','gpt2','resnet50','vgg16', 'gemma1b', 'llama', 'qwen0.6b']
TYPE_DEFAULT_FILES=["2D_Mesh", "2_5D_Mesh","3_5D_Mesh", "2_5D_Mesh_Scaled", "3_5D_Mesh_Scaled"]
#DEFAULT_FILES_GENERIC =["True", "False"]
DEFAULT_FILES_GENERIC =["False"]

n_chiplets=64 #powers of two
n_log2chiplets=int(math.log2(n_chiplets))
stack_count=[2**i for i in range(0, n_log2chiplets+1) ] #number of stacks in the system
chip_count=[1 for _ in range(0, n_log2chiplets+1)] #no of chips per stack
tile_split = [math.ceil(n_chiplets/2**i) for i in range(0, n_log2chiplets+1)] #no of tiles per split

#read initial config.py file for default values of parameters
with open('config_golden.py', 'r') as f:
    lines_inital = f.readlines()
#write the initial config.py file to reset it to initial state after all the runs are completed
with open('config.py', 'w') as f:
    f.writelines(lines_inital)

#TYPE_DEFAULT_FILES=["2D_Mesh"]
#create a file named "output_summary.txt"
with open('output_summary.txt', 'w') as f:
    f.write("Model, Type of Default Files, Output\n")
for model in aimodels:
    for create in DEFAULT_FILES_GENERIC:
        if create=="True":
            for type in TYPE_DEFAULT_FILES:
                #write the configurations in the "output_summary.txt" file
                with open('output_summary.txt', 'a') as f:
                    f.write(f"\n")
                    f.write(f"\n")
                    f.write(f"---------------------Model: {model}, Type: {type}, -------------------------\n") 
                #read config.py file and append lines to it
                with open('config.py', 'a') as f:
                    f.write(f"\naimodel='{model}'\n")
                    f.write(f'DEFAULT_FILES_GENERIC={create}\n')
                    f.write(f'TYPE_DEFAULT_FILES="{type}"\n')
                
                #run HISIM.py file and append the output in a file named "output_summary.txt"
                os.system(f"python HISIM.py >> output_summary.txt")

                #import pdb; pdb.set_trace()
                #remove the added lines "aimodel='{model}' and 'TYPE_DEFAULT_FILES="{type}" from config.py file and also the blank lines
                with open('config.py', 'r') as f:
                    lines = f.readlines()
                with open('config.py', 'w') as f:
                    for line in lines:
                        if line.strip() != f"aimodel='{model}'" and line.strip() != f'TYPE_DEFAULT_FILES="{type}"' and line.strip() != f"DEFAULT_FILES_GENERIC={create}" and line.strip() != "":
                            f.write(line)
            #import pdb; pdb.set_trace()
        else:
            for idx, _ in enumerate(stack_count):
                #write the configurations in the "output_summary.txt" file
                with open('output_summary.txt', 'a') as f:
                    f.write(f"\n")
                    f.write(f"\n")
                    f.write(f"---------------------Model: {model}, Type: User-Defined, Stack Count: {stack_count[idx]}, Chip Count: {chip_count[idx]} -------------------------\n") 
                #read config.py file and append lines to it
                with open('config.py', 'a') as f:
                    f.write(f"\naimodel='{model}'\n")
                    f.write(f'DEFAULT_FILES_GENERIC={create}\n')
                    f.write(f'stack_count={stack_count[idx]}\n')
                    f.write(f'chip_count={chip_count[idx]}\n')
                    f.write(f'tile_count_dict={{"SA":{tile_split[idx]}, "CPU":{tile_split[idx]},"Mem_I":{tile_split[idx]}, "Mem_W":{tile_split[idx]}, "Mem_O":{tile_split[idx]}}}\n')

                #run HISIM.py file and append the output in a file named "output_summary.txt"
                os.system(f"python HISIM.py >> output_summary.txt")

                #remove the added lines "aimodel='{model}' and 'TYPE_DEFAULT_FILES="{type}" from config.py file and also the blank lines
                with open('config.py', 'r') as f:
                    lines = f.readlines()
                with open('config.py', 'w') as f:
                    for line in lines:
                        if line.strip() != f"aimodel='{model}'" and line.strip() != f'DEFAULT_FILES_GENERIC={create}' and line.strip() != "" and line.strip() != f'stack_count={stack_count[idx]}' and line.strip() != f'chip_count={chip_count[idx]}' and line.strip() != f'tile_count_dict={{"SA":{tile_split[idx]}, "CPU":{tile_split[idx]},"Mem_I":{tile_split[idx]}, "Mem_W":{tile_split[idx]}, "Mem_O":{tile_split[idx]}}}':
                            f.write(line)
                #import pdb; pdb.set_trace()

#store the lines_initial in config.py file to reset it to initial state after all the runs are completed
with open('config.py', 'w') as f:
    f.writelines(lines_inital)