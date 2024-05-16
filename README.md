# HISIM_V1.0
HISIM introduces a suite of analytical models at the system level to speed up performance prediction for AI models, covering logic-on-logic architectures across 2D, 2.5D and 3D integration.

![HISIM Overview](https://github.com/pragnyan948/HISIM/blob/main/HISIM_Overview.png "HISIM Overview")
## File Lists
Main directory structure of the repository is below
```
├── Debug                   --Folder containing layer mapping and layer performance information obtained from simulation
├── Module_AI_Map
    ├── AI_Networks         --Folder containing network configuration files of AI models
    ├── util_chip           --Folder containing codes to map the AI model onto the device as per HISIM default mapping
├── Module_Compute          --Folder containing codes for PPA evaluation of IMC compute core
├── Module_Network          --Folder containing codes for PPA evaluation of routers and AIB interface
├── Module_Thermal          --Folder containing thermal codes for simulations of 2D, 2.5D, 3D architectures
├── Results
    ├── result_thermal      --Folder containing device thermal maps obtained from the simulations
    ├── PPA.csv             --CSV file containing PPA and thermal values obtained from the simulation 
├── analy_model.py          --main file
├── run.py                  --Run file to execute multiple runs of analy_model.py with different configurations
```

## Network File
The structure of network.csv file is as follows:
IFM_size_x, IFM_size_y, N_IFM, Kx, Ky, NOFM, pool, layer-wise sparsity.
```
IFM_Size_x:            Size of the input of the Layer in x-dimension
IFM_Size_y:            Size of the input of the Layer in y-dimension
N_IFM:                 Number of input channels of the layer   
Kx:                    Kernel size in x-dimension of the layer
Ky:                    Kernel size in y-dimension of the layer
NOFM:                  Number of output channels of the layer  
pool:                  Parameter indicating if the layer is followed by pooling or not: 0 if not followed by pooling and 1 if followed by pooling
layer-wise sparsity:   Total Sparsity of the layer
```

## Installation and Usage

### Dependencies
* Python
* pandas
* numpy
* torch
* matplotlib
* scipy
* csv
* collections
* json

### Running analy_model.py file
Usage to run the python file analy_model.py
```
--chip_architect        Chip Archiecture: 2D chip (M2D), 3D chip(M3D), 2.5D chip (H2_5D)
--xbar_size             RRAM crossbar size 
--N_tile                Number of tiles in a tier
--N_pe                  Number of PE in a tile
--N_crossbar            Number of crossbars in a PE
--quant_weight          Precision of quantized weight of AI model
--quant_act             Precision of quantized activation of AI model
--freq_computing        Clock frequency of compute core in GHz
--fclk_noc              Clock frequency of network communication unit in GHz
--tsvPitch              TSV pitch um
--N_tier                Number of tiers in the chip for 3D archiecture or Number of chiplets for 2D archiecture 
--volt                  Operating Voltage in volt
--placement_method      1: Tier/Chiplet Edge to Tier/Chiplet Edge connection
                        5: tile-to-tile 3D connection
--percent_router        The percentage of routers to be used for 3D communication when data is routed from one tier to the next 
--W2d                   2D NoC Bandwidth
--router_times_scale    Scaling factor for time components of router: trc, tva, tsa, tst,tl, tenq
--ai_model              AI models:vit, gcn, resnet50, resnet110, vgg16, densenet121
--thermal               Run thermal simulation
         
```
#### 2D Simulation
To run a 2D simulation, apply the following settings:
```
python analy_model.py --chip_architect M2D --N_tile  <Input the total number of tiles required> --ai_model <Input ai model> --thermal
```

#### 2.5d Simulation
To run a 2.5D simulation, apply the following settings:
```
python analy_model.py --chip_architect H2_5D --placement_method 1 --N_tile <Input the number of tiles per chiplet> --ai_model <Input ai model> --thermal
```

#### 3d simulation
To run a 3D simulation with placement method 1 (Tier/Chiplet Edge to Tier/Chiplet Edge connection), apply the following settings:
```
python analy_model.py --chip_architect M3D --placement_method 1 --N_tile <Input the number of tiles per tier> --ai_model <Input ai model> --thermal
```

To run a 3D simulation with placement method 5 (tile-to-tile 3D connection), apply the following settings:
```
python analy_model.py --chip_architect M3D --placement_method 5 --N_tier <Input the number of tiers to be tested for> --ai_model <Input ai model> --thermal
```

### Running run.py file
Additionally, to run design space exploration, the run.py script is provided. Each of the required parameters for the design space can be configured as an array.To include thermal simulations in the design space exploration, add the --thermal flag to the python command for the run.py file.

## Workflow
The workflow of the codes is as follows: The AI model is first mapped onto the architecture using the default mapping in util_mapping.py located in the Module_AI_Map folder. This process outputs layer_information.csv in the following format:
```
layer index, Number of tiles required for the layer, Number of PEs required for the layer, Number of rows of PEs for the layer, Number of columns of PEs for the layer, Number of input cycles for the layer, pooling, Number of tiles mapped until this layer, Total number of input activations for the layer, Tier/chiplet index that the layer is mapped to for this layer, Cell Bit Utilization for the layer, Average Utilization of a row for the layer, Total number of weight bits for the layer, Average Utilization of a column for the layer, Number of FLOPS of the layer.
```
The performance of each layer is estimated based on the layer mapping, assuming an Analog IMC PE. The layer performance is output in the following format:
```
layer index, number of tiles required for this layer, latency of the layer, Energy of the layer, leakage energy of the layer, average power consumption of each tile for the layer
```
The performance of the network and interconnect is then estimated based on the number of tiles, placement method, and the percentage of routers. The thermal simulation is performed using power and area maps. It outputs the peak temperature, average temperature, and thermal maps. Lastly, All the results are stored in the output file PPA.csv.

## PPA File
The structure of output PPA.csv file is as follows: 
```
freq_core,freq_noc,Xbar_size,N_tile,N_pe,N_tile(real),N_tier(chiplet),W2d,W3d,Computing_latency, Computing_energy,compute_area,chip_area,chip_Architecture,2d NoC latency,3d NoC latency,2.5d NoC latency, network_latency,2d NoC energy,3d NoC energy,2.5d NoC energy,network_energy,rcc,TFLOPS,compute_power, 2D_3D_NoC_power,2_5D_power,2d_3d_router_area,peak_temperature,placement_method,percent_router
```
The parameters freq_core,freq_noc,Xbar_size,N_tile,N_pe,N_tile(real),N_tier(chiplet),W2d,placement_method,percent_router are input parameters of the simulation performed

Outputs from the simulation:
```
W3d:                   3D TSV Bandwidth                     
N_tile(real):          Number of real tiles mapped in a tier
Computing_latency:     Total Latency of the computing core
Computing_energy:      Total Energy of the computing core
compute_area:          Total Area of the computing core
chip_area:             Total Chip Area
2d NoC latency:        Total Latency of the 2D NoC
3d NoC latency:        Total Latency of the 3D TSV
2.5d NoC latency:      Total Latency of the 2.5D AIB interface
network_latency:       Total Network Latency consisting of 2D NoC, 3D TSV, 2.5D AIB interface latencies
2d NoC energy:         Total Energy of the 2D NoC
3d NoC energy:         Total Energy of the 3D TSV
2.5d NoC energy:       Total Energy of the 2.5D AIB interface
network_energy:        Total Network Energy consisting of 2D NoC, 3D TSV, 2.5D AIB interface energies
rcc:                   Ratio between computation and communication latencies
TFLOPS:                Total number of FLOPS divided by latency in TFLOPS/s
2D_3D_NoC_power:       Total power of the 2D NoC and 3D TSV        
2_5D_power:            Total power of the 2.5D AIB interface   
2d_3d_router_area:     Total area of the 2D and 3D router   
peak_temperature:      Peak temperature of the chip
```
## Examples
A demo video has been added to the repository to help users get started, demonstrating a few examples.

#### Thermal Maps
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
