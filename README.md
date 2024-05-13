# HISIM_V1.0

## File Lists
Main directory structure is below
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
├── analy_model_thermal.py
├── run.py                  --Run file to execute multiple runs of analy_model.py with different configurations
```

## CSV Files

### Network File
The structure of network.csv file is as follows:
IFM_size_x, IFM_size_y, N_IFM, Kx, Ky, NOFM, pool, layer-wise sparsity.
```
IFM_Size_x: Size of the input of the Layer in x-dimension
IFM_Size_y: Size of the input of the Layer in y-dimension
N_IFM: Number of input channels of the layer   
Kx: Kernel size in x-dimension of the layer
Ky: Kernel size in y-dimension of the layer
NOFM:  number of output feature map
pool: Parameter indicating if the layer is followed by pooling or not: 0 if not followed by pooling and 1 if followed by pooling
layer-wise sparsity: Total Sparsity of the layer
```

### PPA File
The structure of PPA.csv file is as follows: 
```
freq_core,freq_noc,Xbar_size,N_tile,N_pe,N_tile(real),N_tier(chiplet),W2d,W3d,Computing_latency, Computing_energy,compute_area,chip_area,chip_Architecture,2d NoC latency,3d NoC latency,2.5d NoC latency, network_latency,2d NoC energy,3d NoC energy,2.5d NoC energy,network_energy,rcc,TFLOPS,compute_power, 2D_3D_NoC_power,2_5D_power,2d_3d_router_area,peak_temperature,placement_method,percent_router
```

Input Parameters of the Simulation performed
```
freq_core:
freq_noc:
Xbar_size:
N_tile:
N_pe:
N_tile(real):
N_tier(chiplet):
W2d:
W3d:
chip_Architecture:
placement_method:
percent_router:
```
Outputs from the simulation:
```
Computing_latency:
Computing_energy:
compute_area:
chip_area:
2d NoC latency:
3d NoC latency:
2.5d NoC latency:
network_latency:
2d NoC energy:
3d NoC energy:
2.5d NoC energy:
network_energy:
rcc:
TFLOPS:
compute_power:
2D_3D_NoC_power:
2_5D_power:
2d_3d_router_area:
peak_temperature:
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

#### 2D Simulation


#### 2.5d Simulation


#### 3d simulation
Placement mode 1:

Placement mode 5:

### Running run.py file

## Examples

### Configuration

### Commands

### Outputs

#### PPA

#### Terminal

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
