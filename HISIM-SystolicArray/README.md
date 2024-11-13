# HISIM_V1.0_Systolic Array
HISIM introduces a suite of analytical models at the system level to speed up performance prediction for AI models, covering logic-on-logic architectures across 2D, 2.5D, 3D and 3.5D integration.

![HISIM Overview](https://github.com/pragnyan948/HISIM/blob/main/HISIM-SystolicArray/HISIM_Overview.png "HISIM Overview")

## File Lists
The main directory structure of the repository is shown below. The file `SA_run_tb.py` contains example use cases for running the tool.
```
├── Debug                   -- Folder containing layer mapping and performance information obtained from simulations
├── Demos                   -- Folder containing demo videos of the tools
├── Module_AI_Map
    ├── AI_Networks         -- Folder containing configuration files for AI network models
    ├── util_chip           -- Folder containing code to map the AI model onto the device using HISIM's default mapping
├── Module_Compute          -- Folder containing code for PPA evaluation of the Systolic Array compute core
├── Module_Network          -- Folder containing code for PPA evaluation of routers and the AIB interface
├── Module_Thermal          -- Folder containing thermal simulation code for 2D, 2.5D, and 3D architectures
├── Results
    ├── result_thermal      -- Folder containing thermal maps of the device obtained from simulations
    ├── PPA.csv             -- CSV file containing PPA and thermal values in list format
    ├── PPA_new.csv         -- CSV file containing PPA and thermal values in dictionary format
├── hisim_model.py          -- Main file used by SA_run_tb.py; implements setter functions
├── SA_run_tb.py               -- Run file to execute example runs using the HiSimModel defined in hisim_model.py

```

## Network File
To facilitate extensibility in evaluating the performance of various AI algorithms, AI network files for different DNNs, GNNs, and transformers are provided at [this link](https://github.com/mec-UMN/HISIM/tree/main/Module_AI_Map/AI_Networks). The structure of the `network.csv` file is as follows: IFM_size_x, IFM_size_y, N_IFM, Kx, Ky, NOFM, pool, layer-wise sparsity.
```
IFM_Size_x:            Size of the input of the Layer in the x-dimension
IFM_Size_y:            Size of the input of the Layer in the y-dimension
N_IFM:                 Number of input channels of the layer   
Kx:                    Kernel size in x-dimension of the layer
Ky:                    Kernel size in y-dimension of the layer
NOFM:                  Number of output channels of the layer  
pool:                  Parameter indicating if the layer is followed by pooling or not: 0 if not followed by pooling and 1 if followed by pooling
layer-wise sparsity:   Total Sparsity of the layer
```
For a fully connected (FC) or linear layer, `Kx` and `Ky` are both set to 1. For an example, refer to the ViT network [here](https://github.com/pragnyan948/HISIM/blob/main/Module_AI_Map/AI_Networks/Transformer/VIT_base.csv).

## Installation and Usage

To run example use-cases, run the following command
```
python SA_run_tb.py
```
### Package Dependencies

run the following command to install dependencies
``` 
pip install -r requirements.txt
```
* Python 3.8.5
* pandas 1.1.3
* numpy  1.19.2
* torch  2.2.2
* matplotlib 3.3.2
* scipy  1.5.2

### HiSimModel Input Parameters 
Input parameters of HiSimModel and their corresponding setter functions and options are as follows:
```
Parameter           Setter Function           Parameter Options
--chip_architect    set_chip_architecture     Chip Architecture options are 2D chip (M2D), 3D chip (M3D), 2.5D chip (H2_5D), and 3.5D (M3_5D).
--SA_size           set_SA_size               Systolic Array size 
--N_tile            set_N_tile                Number of tiles per tier, with options 4, 9, 16, 25, 36, and 49.
--N_pe              set_num_pe                Number of processing elements (PEs) per tile, with options 4, 9, 16, 25, and 36.
--N_arr             set_N_arr                 Number of Arrays per PE, with options 4, 9, and 16.
--freq_computing    set_freq_computing        Clock frequency of the compute core in GHz.
--fclk_noc          set_fclk_noc              Clock frequency of the network communication unit in GHz.
--tsvPitch          set_tsv_pitch             3D TSV (Through-Silicon Via) pitch in micrometers (µm).
--N_tier            set_N_tier                Number of tiers in the chip for 3D architecture.
--volt              set_volt                  Operating voltage in volts
--placement_method  set_placement             Placement method options:
                                              1: Tier/Chiplet Edge to Tier/Chiplet Edge connection
                                              5: Tile-to-Tile 3D connection
--routing_method    set_router                Routing method options:
                                              1: Local routing—uses only nearby routers and TSVs.
                                              2: Global routing—data will attempt to use all available routers to reach the next tier.
--percent_router    set_percent_router        Percentage of routers used for 3D communication when data is routed from one tier to the next.
--W2d               set_W2d                   2D NoC (Network on Chip) bandwidth.
--router_times_scale  set_router_times_scale  Scaling factor for time components of the router: trc, tva, tsa, tst, tl, tenq.
--ai_model          set_ai_model              AI models, including vit, gcn, resnet50, resnet110, vgg16, and densenet121.
--thermal           set_thermal               Set to True to run a thermal simulation; set to False otherwise.
--N_stack,          set_N_stack               Number of 3D stacks in a 3.5D design or number of chiplets in a 2.5D design.
```

### Running the 2D/2.5D/3D/3.5D simulations
#### 2D Simulation
To run a 2D simulation, apply the following settings:
```
-- Use HiSimModel.set_chip_architecture("M2D") to set the chip architecture to 2D
-- Use HiSimModel.set_N_tier(1) and HiSimModel.set_N_stack(1) to set the number of tiers and stacks to 1
-- Set the remaining input parameters based on the required hardware configuration
-- AI models can be specified as one of the following: vit, gcn, resnet50, resnet110, vgg16, densenet121
-- Use HiSimModel.run_model() to evaluate performance
-- Check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information
```

#### 2.5D Simulation
To run a 2.5D simulation, apply the following settings:
```
-- Use HiSimModel.set_chip_architecture("H2_5D") to set the chip architecture to H2_5D
-- Use HiSimModel.set_N_tier(1) and HiSimModel.set_placement(1) to set the number of tiers and placement_method to 1 
-- Set the remaining input parameters based on the required hardware configuration
-- AI models can be specified as one of the following: vit, gcn, resnet50, resnet110, vgg16, densenet121
-- Use HiSimModel.run_model() to evaluate performance
-- Check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information
```

#### 3D simulation
To run a 3D simulation, apply the following settings:
```
-- Use HiSimModel.set_chip_architecture("M3D") to set the chip architecture to M3D
-- Use HiSimModel.set_N_stack(1) and HiSimModel.set_placement(5) to set the number of stacks to 1 and placement_method to 5
-- Set the remaining input parameters based on the required hardware configuration
-- AI models can be specified as one of the following: vit, gcn, resnet50, resnet110, vgg16, densenet121
-- Use HiSimModel.run_model() to evaluate performance
-- Check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information
```

#### 3.5D simulation
To run a 3_5D simulation, apply the following settings:
```
-- Use HiSimModel.set_chip_architecture("M3_5D") to set the chip architecture to M3_5D
-- Use HiSimModel.set_placement(5) to set the placement_method to 5
-- Set the remaining input parameters based on the required hardware configuration
-- AI models can be specified as one of the following: vit, gcn, resnet50, resnet110, vgg16, densenet121
-- Use HiSimModel.run_model() to evaluate performance
-- Check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information
```

### Workflow
The workflow of the codes is as follows: The AI model is first mapped onto the architecture using the default mapping in util_mapping.py located in the Module_AI_Map folder. This process outputs layer_information.csv in the following format:
```
layer index, Number of tiles required for the layer, Number of PEs required for the layer, Number of rows of PEs for the layer, Number of columns of PEs for the layer, Number of input cycles for the layer, pooling, Number of tiles mapped until this layer, Total number of input activations for the layer, Tier index that the layer is mapped to for this layer, Bit Utilization for the layer, Average Utilization of a row for the layer, Total number of weights for the layer, Average Utilization of a column for the layer, Number of FLOPS of the layer,Stack index that the layer is mapped to for this layer.
```
The performance of each layer is estimated based on the layer mapping, assuming a Systolic Array PE. The layer performance is output in the following format:
```
layer index, number of tiles required for this layer, latency of the layer, Energy of the layer, average power consumption of each tile for the layer
```
The performance of the network and interconnect is then estimated based on the number of tiles, placement method, and the percentage of routers. The thermal simulation is performed using power and area maps. It outputs the peak temperature, average temperature, and thermal maps. Lastly, All the results are stored in the output file PPA.csv.

### Outputs
#### PPA File
The structure of output PPA.csv file is as follows: 
```
freq_core (GHz),freq_noc (GHz),Xbar_size,N_tile,N_pe,N_tile(real),N_tier(real),N_stack(real),W2d,W3d,Computing_latency (ns),Computing_energy (pJ),compute_area (um2),chip area (mm2),chip_Architecture,2d NoC latency (ns),3d NoC latency (ns),2.5d NoC latency (ns),network_latency (ns),2d NoC energy (pJ),3d NoC energy (pJ),2.5d NoC energy (pJ),network_energy (pJ),rcc (compute latency/communciation latency),Throughput(TFLOPS/s),2D_3D_NoC_power (W),2_5D_power (W),2d_3d_router_area (mm2),placement_method,percent_router,peak_temperature (C),thermal simulation time (s),networking simulation time (s),computing simulation time (s),total simulation time (s)
```
The parameters freq_core (GHz),freq_noc (GHz),Xbar_size,N_tile,N_pe,N_tile(real),N_tier(real),N_stack(real),W2d,placement_method,percent_router are input parameters of the simulation performed

Outputs from the simulation:
```
W3d:                   3D TSV Bandwidth                     
N_tile(real):          Number of real tiles mapped in a tier
Computing_latency(ns): Total Latency of the computing core
Computing_energy(pJ):  Total Energy of the computing core
compute_area(um2):     Total Area of the computing core
chip_area(mm2):        Total Chip Area
2d NoC latency(ns):    Total Latency of the 2D NoC
3d NoC latency(ns):    Total Latency of the 3D TSV
2.5d NoC latency(ns):  Total Latency of the 2.5D AIB interface
network_latency(ns):   Total Network Latency consisting of 2D NoC, 3D TSV, 2.5D AIB interface latencies
2d NoC energy(pJ):     Total Energy of the 2D NoC
3d NoC energy(pJ):     Total Energy of the 3D TSV
2.5d NoC energy(pJ):   Total Energy of the 2.5D AIB interface
network_energy(pJ):    Total Network Energy consisting of 2D NoC, 3D TSV, 2.5D AIB interface energies
rcc:                   Ratio between computation and communication latencies
hroughput(TFLOPS/s)    Total number of FLOPS divided by latency in TFLOPS/s
2D_3D_NoC_power(W):    Total power of the 2D NoC and 3D TSV        
2_5D_power(W):         Total power of the 2.5D AIB interface   
2d_3d_router_area(mm2):Total area of the 2D and 3D router   
peak_temperature (K):  Peak temperature of the chip
```
#### Tile Maps
The Tile Map is a visual representation of the default mapping implemented in HISIM. It displays the stacks, tiers, and tiles present in the architecture, and it links each tile to the AI layer number to which it is mapped. Note that multiple tiles can be mapped to a single AI layer. This tile map can be found at `Results/tile_map.png`.

#### Thermal Maps
The Thermal Map is a visual representation of the temperature distribution across the components of each tier. It shows the temperature profile of tiles and routers on each tier, including the 2.5D connections. These temperature maps can be found in the folder `Results/result_thermal/1stacks` for 3D runs and in folder `Results/result_thermal/` for 2.5 runs.

## Examples
A demo video has been added to the repository to help users get started, showcasing a few examples using SA_run_tb.py. The test cases, their respective outputs, AI networks, hardware configuration, and DSE parameters inside SA_run_tb.py are as follows:
```
Test Case      Output        AI Network      HW configuration                   DSE parameter
                                             (Narr-Npe-Ntile-Ntier-Nstack-arch)   
Test Case 1    PPA           ViT             128-36-64-2-2-3.5D                 NA - Single run
Test Case 2    PPA           densenet121     128-36-144-2-2-3.5D                NA - Single run
Test Case 3    PPA           densenet121     128-36-256-2-1-3D                  tsv_pitch: [2,3,4,5,10,20]
Test Case 4    PPA           densenet121     128-36-256-2-1-3D                  noc_width(W2d): [i for i in range(1,32, 5)]
```

## Citing this work
If you found this tool useful, please use the following bibtex to cite us
```
@INPROCEEDINGS{10396377,
  author={Wang, Zhenyu and Sun, Jingbo and Goksoy, Alper and Mandal, Sumit K. and Seo, Jae-Sun and Chakrabarti, Chaitali and Ogras, Umit Y. and Chhabria, Vidya and Cao, Yu},
  booktitle={2023 IEEE 15th International Conference on ASIC (ASICON)}, 
  title={Benchmarking Heterogeneous Integration with 2.5D/3D Interconnect Modeling}, 
  year={2023},
  volume={},
  number={},
  pages={1-4},
  keywords={Analytical models;Three-dimensional displays;Computational modeling;Multichip modules;Benchmark testing;Data models;Artificial intelligence;Heterogeneous Integration;2.5D;3D;Chiplet;ML accelerators;Electro-thermal Co-design},
  doi={10.1109/ASICON58565.2023.10396377}}

@INPROCEEDINGS{10473875,
  author={Wang, Zhenyu and Sun, Jingbo and Goksoy, Alper and Mandal, Sumit K. and Liu, Yaotian and Seo, Jae-Sun and Chakrabarti, Chaitali and Ogras, Umit Y. and Chhabria, Vidya and Zhang, Jeff and Cao, Yu},
  booktitle={2024 29th Asia and South Pacific Design Automation Conference (ASP-DAC)}, 
  title={Exploiting 2.5D/3D Heterogeneous Integration for AI Computing}, 
  year={2024},
  volume={},
  number={},
  pages={758-764},
  keywords={Analytical models;Three-dimensional displays;Computational modeling;Wires;Multichip modules;Benchmark testing;Transformers;Heterogeneous Integration;2.5D;3D;Chiplet;ML accelerators;Performance Analysis},
  doi={10.1109/ASP-DAC58780.2024.10473875}}

```
## Developers
Main devs:
* Zhenyu Wang 
* Pragnya Sudershan Nalla
* Jingbo Sun
* Emad Haque

Contributers:
* A. Alper Goksoy
  
Maintainers and Advisors
* Sumit K.Mandal
* Jae-sun Seo
* Vidya A. Chhabria
* Jeff Zhang
* Chaitali Chakrabarti
* Umit Y. Ogras
* Yu Cao
