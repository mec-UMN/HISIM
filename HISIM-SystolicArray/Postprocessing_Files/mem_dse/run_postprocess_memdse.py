#read ourput_summary.txt file and print the output in a csv file named "output_summary.csv"
#Columns in the csv file should be "Model", "Type", "Total sim time", , "Total Cost per Part", "Total Chip Area"
import csv
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(curr_dir)
parent_dir = os.path.dirname(parent_dir)
with open(parent_dir+'/output_summary.txt', 'r') as f:
    lines = f.readlines()
output_data = []
for line in lines:
    if line.startswith("---------------------Model:"):
        model = line.split(",")[0].split(":")[1].strip()
        type_run = line.split(",")[1:]
        #strip spaces and --- from type_run and join them to make a single string
        type_run = [t.strip().strip("-").strip() for t in type_run]
        type_run = " ".join(type_run)
    if line.startswith("Total Chip Area"):
        total_chip_area = line.split(")")[1].strip().split(" ")[0]
    if line.startswith("Total Cost"):
        total_cost = line.split(":")[1].strip().split(" ")[0]
    if line.startswith("Memory Latency"):
        total_memory_latency = line.split(":")[1].strip().split(" ")[0]
    if line.startswith("DDR Latency"):
        total_ddr_latency = line.split(":")[1].strip().split(" ")[0]
    if line.startswith("Total sim time"):
        total_sim_time = line.split(":")[1].strip().split(" ")[0]
        output_data.append([model, type_run, total_sim_time, total_cost, total_chip_area, total_memory_latency, total_ddr_latency])
#print(output_data)
#import pdb; pdb.set_trace()
with open(curr_dir+'/output_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Type", "Total sim time", "Total Cost per Part", "Total Chip Area", "Total Memory Latency", "Total DDR Latency"])
    writer.writerows(output_data)

#output another csv file with model names as row headers and type of default files as column headers and total memory latency as values
type_set = set()
for data in output_data:
    type_set.add(data[1])
type_list = list(type_set)
type_list.sort()
model_set = set()
for data in output_data:
    model_set.add(data[0])
model_list = list(model_set)
#sort model_list based on sim time
model_list.sort(key=lambda x: float([data[5] for data in output_data if data[0] == x][0]))
mem_latency_dict = {}
for data in output_data:
    mem_latency_dict[(data[0], data[1])] = data[5]
with open(curr_dir+'/output_summary_mem_latency.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model"] + type_list)
    for model in model_list:
        row = [model]
        for type_str in type_list:
            row.append(mem_latency_dict.get((model, type_str), ""))
        writer.writerow(row)

#output another csv file with model names as row headers and type of default files as column headers and total ddr latency as values
ddr_latency_dict = {}
for data in output_data:
    ddr_latency_dict[(data[0], data[1])] = data[6]
with open(curr_dir+'/output_summary_ddr_latency.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model"] + type_list)
    for model in model_list:
        row = [model]
        for type_str in type_list:
            row.append(ddr_latency_dict.get((model, type_str), ""))
        writer.writerow(row)

rows = []

for key in ddr_latency_dict:
    model = key[0]
    type_str = key[1]             
    ddr_latency = float(ddr_latency_dict[key])
    mem_latency = float(mem_latency_dict[key])
    nbanks = type_str.split(" ")[9]   
    nbanks = int(nbanks)

    rows.append((model, nbanks, ddr_latency,mem_latency))

# Now build the DataFrame from the list of tuples
df = pd.DataFrame(rows, columns=["Model", "Number of Banks", "DDR Latency", "Memory Latency"])

models = sorted(df["Model"].unique(), key=lambda x: df[df["Model"] == x]["DDR Latency"].max()) #sort based on cost
n_models = len(models)

# choose grid, e.g., up to 3 columns
n_cols = 3
n_rows = math.ceil(n_models / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)

for i, model in enumerate(models):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]

    sub = df[df["Model"] == model].sort_values("Number of Banks")

    ax.plot(sub["Number of Banks"], sub["DDR Latency"], marker="o")
    ax.set_title(str(model.capitalize()), fontsize=15, weight='bold')
    ax.set_xlabel("Number of Banks Per Tile", fontsize=15, weight='bold')
    ax.set_ylabel("DDR Latency", fontsize=15, weight='bold')
    #adjust y-axis ticks to font size 15 and bold
    ax.tick_params(axis='y', labelsize=15)
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    #x-axis ticks to font size 15 and bold
    ax.tick_params(axis='x', labelsize=15)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')

# Hide any unused axes if models don't fill the grid
for j in range(i+1, n_rows*n_cols):
    r = j // n_cols
    c = j % n_cols
    fig.delaxes(axes[r, c])

fig.suptitle("DDR Latency vs Number of Banks (per model)\n"+ str(len(sub["Number of Banks"]))+" Configurations & "+ str(len(models))+" Models", fontsize=16, weight='bold')
fig.tight_layout()

# show and/or save
plt.show()
fig.savefig(curr_dir+"/chiplet_ddr_latency.png", dpi=300, bbox_inches="tight")
