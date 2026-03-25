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
    if line.startswith("Total sim time"):
        total_sim_time = line.split(":")[1].strip().split(" ")[0]
        output_data.append([model, type_run, total_sim_time, total_cost, total_chip_area])
#print(output_data)
#import pdb; pdb.set_trace()
with open(curr_dir+'/output_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Type", "Total sim time", "Total Cost per Part", "Total Chip Area"])
    writer.writerows(output_data)

#output another csv file with model names as row headers and type of default files as column headers and total sim time as values
type_set = set()
for data in output_data:
    type_set.add(data[1])
type_list = list(type_set)
model_set = set()
for data in output_data:
    model_set.add(data[0])
model_list = list(model_set)
#sort model_list based on sim time
model_list.sort(key=lambda x: float([data[2] for data in output_data if data[0] == x][0]))
sim_time_dict = {}
for data in output_data:
    sim_time_dict[(data[0], data[1])] = data[2]
with open(curr_dir+'/output_summary_sim_time.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model"] + type_list)
    for model in model_list:
        row = [model]
        for type_str in type_list:
            row.append(sim_time_dict.get((model, type_str), ""))
        writer.writerow(row)

total_sim_time=sum([float(data) for data in sim_time_dict.values() if data!=""])

#plot simtimes as a 2D heatmap with x-axis as type of default files and y-axis as model names, sort models as per sim time
sim_time_matrix = []
for model in model_list:
    row = []
    for type_str in type_list:
        sim_time = sim_time_dict.get((model, type_str), "")
        if sim_time!="":
            sim_time = float(sim_time)
        row.append(sim_time)
    #import pdb; pdb.set_trace()
    #sort row based on int(type_str.split(" ")[4]) which is the chiplet count
    row = [x for _, x in sorted(zip(type_list, row), key=lambda pair: int(pair[0].split(" ")[4]))]
    chiplet_counts_updated = sorted([int(type_str.split(" ")[4]) for type_str in type_list])
    sim_time_matrix.append(row)
plt.figure(figsize=(10, 8))
plt.imshow(sim_time_matrix, cmap='viridis', aspect='auto')
cbar = plt.colorbar()                 # create colorbar
cbar.set_label('Total Sim Time (s)',  # label text
               size=15, weight='bold')               # label font size
cbar.ax.tick_params(labelsize=15)  # tick font size

for tick in cbar.ax.get_yticklabels():
    tick.set_fontweight('bold')    # tick font weight

plt.xticks(ticks=range(len(type_list)), labels=chiplet_counts_updated, rotation=45, ha='right', fontsize=15, weight='bold')
#model_list's first letter in capatilized and rest in small letters
model_list = [model.capitalize() for model in model_list]
plt.yticks(ticks=range(len(model_list)), labels=model_list, fontsize=15, weight='bold') 
plt.xlabel("Chiplet count", fontsize=15, weight='bold')
plt.title("Simulation Time Heatmap \n"+ str(len(type_list))+" Types & "+ str(len(model_list))+" Models", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig(curr_dir+"/sim_time_heatmap.png", dpi=300, bbox_inches="tight")

#output another csv file with model names as row headers and type of default files as column headers and total cost per part as values
cost_dict = {}
for data in output_data:
    cost_dict[(data[0], data[1])] = data[3]
with open(curr_dir+'/output_summary_cost.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model"] + type_list)
    for model in model_list:
        row = [model]
        for type_str in type_list:
            row.append(cost_dict.get((model.lower(), type_str), ""))
        writer.writerow(row)

#plot a line graph with x-axis as number of chiplet count and y-axis as costperpart, spearate sub-figures for each model
rows = []

for key in cost_dict:
    model = key[0]
    type_str = key[1]             
    cost = float(cost_dict[key])
    chiplet_count = type_str.split(" ")[4]   
    chiplet_count = int(chiplet_count)

    rows.append((model, chiplet_count, cost))

# Now build the DataFrame from the list of tuples
df = pd.DataFrame(rows, columns=["Model", "Chiplet Count", "Cost per Part"])

models = sorted(df["Model"].unique(), key=lambda x: df[df["Model"] == x]["Cost per Part"].max()) #sort based on cost
n_models = len(models)

# choose grid, e.g., up to 3 columns
n_cols = 3
n_rows = math.ceil(n_models / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)

for i, model in enumerate(models):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]

    sub = df[df["Model"] == model].sort_values("Chiplet Count")
    #import pdb; pdb.set_trace()
    ax.plot(sub["Chiplet Count"], sub["Cost per Part"], marker="o")
    ax.set_title(str(model.capitalize()), fontsize=15, weight='bold')
    ax.set_xlabel("Chiplet count", fontsize=15, weight='bold')
    ax.set_ylabel("Cost per part", fontsize=15, weight='bold')
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

fig.suptitle("Cost per part vs chiplet count (per model)\n Total sim time: "+str(round(total_sim_time, 3))+" seconds \n"+ str(len(sub["Chiplet Count"]))+" Configurations & "+ str(len(models))+" Models", fontsize=16, weight='bold')
fig.tight_layout()

# show and/or save
plt.show()
fig.savefig(curr_dir+"/chiplet_cost_per_part.png", dpi=300, bbox_inches="tight")


#output another csv file with model names as row headers and type of default files as column headers and total chip area as values
chip_area_dict = {}
for data in output_data:
    chip_area_dict[(data[0], data[1])] = data[4]
#import pdb; pdb.set_trace()
with open(curr_dir+'/output_summary_chip_area.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model"] + type_list)
    for model in model_list:
        row = [model]
        for type_str in type_list:
            row.append(chip_area_dict.get((model.lower(), type_str), ""))
            #import pdb; pdb.set_trace()
        writer.writerow(row)