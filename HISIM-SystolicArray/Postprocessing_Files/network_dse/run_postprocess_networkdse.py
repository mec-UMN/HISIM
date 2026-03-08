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
        type_run = line.split(",")[4:]
        #strip spaces and --- from type_run and join them to make a single string
        type_run = [t.strip().strip("-").strip() for t in type_run]
        type_run = " ".join(type_run)
    if line.startswith("Total 2D Network NoC latency "):
        total_network_latency = float(line.split(")")[1].strip())
    if line.startswith("Total 2.5D Network NoP latency"):
        total_network_latency += float(line.split(")")[1].strip())
    if line.startswith("Total 3D Network NoC latency"):
        total_network_latency += float(line.split(")")[1].strip())
    if line.startswith("Total sim time"):
        total_sim_time = line.split(":")[1].strip().split(" ")[0]
        output_data.append([model, type_run, total_sim_time, total_network_latency])
#print(output_data)
#import pdb; pdb.set_trace()
with open(curr_dir+'/output_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Type", "Total sim time","Total Network latency"])
    writer.writerows(output_data)

#output another csv file with model names as row headers and type of default files as column headers and total network latency as values
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
model_list.sort(key=lambda x: float([data[3] for data in output_data if data[0] == x][0]))
network_latency_dict = {}
for data in output_data:
    network_latency_dict[(data[0], data[1])] = data[3]
with open(curr_dir+'/output_summary_network_latency.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model"] + type_list)
    for model in model_list:
        row = [model]
        for type_str in type_list:
            row.append(network_latency_dict.get((model, type_str), ""))
        writer.writerow(row)

rows = []

for key in network_latency_dict:
    model = key[0]
    type_str = key[1]             
    network_latency = float(network_latency_dict[key])
    n_2_5d_channels = type_str.split(":")[1]   
    #import pdb; pdb.set_trace()
    n_2_5d_channels = int(n_2_5d_channels)

    rows.append((model, n_2_5d_channels, network_latency))

# Now build the DataFrame from the list of tuples
df = pd.DataFrame(rows, columns=["Model", "Number of 2D Links", "Network Latency"])

models = sorted(df["Model"].unique(), key=lambda x: df[df["Model"] == x]["Network Latency"].max()) #sort based on cost
n_models = len(models)

# choose grid, e.g., up to 3 columns
n_cols = 3
n_rows = math.ceil(n_models / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)

for i, model in enumerate(models):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]

    sub = df[df["Model"] == model].sort_values("Number of 2D Links")

    ax.plot(sub["Number of 2D Links"], sub["Network Latency"], marker="o")
    ax.set_title(str(model))
    ax.set_xlabel("Number of 2D Links per Tile")
    ax.set_ylabel("Network Latency")

# Hide any unused axes if models don't fill the grid
for j in range(i+1, n_rows*n_cols):
    r = j // n_cols
    c = j % n_cols
    fig.delaxes(axes[r, c])

fig.suptitle("Network Latency vs Number of 2D Links per tile\n"+ str(len(sub["Number of 2D Links"]))+" Configurations & "+ str(len(models))+" Models", fontsize=16)
fig.tight_layout()

# show and/or save
plt.show()
fig.savefig(curr_dir+"/chiplet_network_latency.png", dpi=300, bbox_inches="tight")
