import os
import pandas as pd
import itertools
import time

current_dir = os.path.dirname(__file__)
parent_dir_inter = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir_inter)
golden_filename = parent_dir+'/Chip_Map_Attention_Golden.csv'
new_filename = parent_dir+'/Chip_Map_Attention.csv'

# Load data
chip_df_golden = pd.read_csv(golden_filename)

# Get coordinates only for C2
coords_c2 = chip_df_golden.loc[chip_df_golden["Chiplet ID"] == "C2", "NoC Position"].tolist()

# Generate all permutations of those coords
permutations = itertools.permutations(coords_c2, len(coords_c2))

# Iterate
for i, perm in enumerate(permutations, 1):
    print(f"Permutation {i}: {perm}")

    chip_df = chip_df_golden.copy()

    # Assign permuted coords back only to rows with Chiplet ID C2
    chip_df.loc[chip_df["Chiplet ID"] == "C2", "NoC Position"] = list(perm)

    # Now chip_df has permuted positions for C2, others unchanged
    chip_df.to_csv(new_filename, index=False)

    start_time_init = time.time()
    # Run external script
    os.system(f"python Network.py")
    end_time_last = time.time()
    print(f"Time taken for permutation {i}: {end_time_last - start_time_init} seconds")
    #import pdb; pdb.set_trace()
