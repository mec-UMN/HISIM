# HISIM_V2.0_Systolic Array
HISIM introduces a suite of analytical models for compute, memory, ddr, network to speed up performance prediction for AI models, covering logic-on-logic architectures across 2D, 2.5D, 3D and 3.5D integration.

## File Lists
The main directory structure of the repository is shown below. Please run 'python HISIM.py' to run the tool. Input files for the AI layer, compute modules, network, and their interactions can be found in the respective folders below.
```
├── Module_0_AI_Map
    ├── HISIM_2_0_AI_layer_information         -- Folder containing input configuration files that describe the AI models
    ├── util_chip                              
        ├── HISIM_2_0_Files                    -- Input Folder containing codes to load AI network, chiplet and tile mapping files
├── Module_1_Compute                              
    ├── HISIM_2_0_Files                        
        ├── HW_configs                         -- Input Folder containing the specification files that describe the memory, systolic array specifications for each tile 
        ├──Compute.json                        -- Input File containing the parameter values used within the HISIM compute and memory analytical models
        ├──Compute.py                          -- Code performing PPA evaluation of the Compute, Memory and DDR of tiles
        ├──CPU.py                              -- Code containing analytical models of CPU to execute non-linear activation units
        ├──SA.py                               -- Code containing analytical models of systolic array to execute matrix multiplication and convolution
        ├──Mem.py                              -- Code containing analytical models of Memory and DDR communication
├── Module_2_Network          
    ├── HISIM_2_0_Files  
        ├── Network_configs                    -- Folder containing the input specification file that describes the network specifications for each stack 
        ├── Network.json                       -- Input File containing the parameter values used within the HISIM network analytical models
        ├──Compute.py                          -- Code performing PPA evaluation of the noc, nop and 3d links within the system
├── Module_3_Cost                              -- Folder containing code for cost evaluation of the system
├── Module_4_Thermal                           -- Folder containing thermal simulation code for 2D, 2.5D, and 3D architectures
├── Results                                    -- Folder containing the visualization outputs to guide the user in DEBUG mode
├── Z_Map_Files_Sample                         -- Sample 2.5D/3D files to run attention layer
├── Chip_Map.csv                               -- Input file describing the HW tile to AI layer mapping
├── config.py                                  -- Code to set global variables across all files. Please set DEBUG = TRUE to get intermediate output plots
├── HISIM.py                                   -- Main file to run the hisim code
├── Layer_Map.csv                              -- Input file describing how the user wants to map the AI layer onto the systolic array
├── Sys_Map.csv                                -- Input file describing the chiplet placement within the stack

```
All Power, Temperature, Performance, Area, and Cost, (PTPAC) outputs are printed in the terminal output and the intermediatory outputs are saved in HW_configs, Network_configs and Results folder in DEBUG mode.

## Configuration File (config.py)

To generate the default input files, set the following parameter in config.py:

```

CREATE_DEFAULT_FILES = True

```
Then, choose the desired system configuration by setting TYPE_DEFAULT_FILES:

```
TYPE_DEFAULT_FILES = "2_5D_Mesh"  # 1 tile per AI layer’s compute and memory type, 1 chip, 1 tier
TYPE_DEFAULT_FILES = "2D_Mesh"  # 1 tile per layer’s compute and memory type, 3-chiplet system: SRAM memory connected to DDR, SAs+Mem, and CPUs+Mem; 1 tier
TYPE_DEFAULT_FILES = "3_5D_Mesh"  # 1 tile per layer’s compute and memory type, 3-chiplet system: SRAM memory connected to DDR, SAs+Mem, and CPUs+Mem; 2 tiers
TYPE_DEFAULT_FILES = "2_5D_Mesh_Scaled"  # 1 component (CPU, Mem, SA) per chiplet, 1 chiplet per layer’s compute and memory type, 1 tier
TYPE_DEFAULT_FILES = "3_5D_Mesh_Scaled"  # 1 component (CPU, Mem, SA) per chiplet; 1 chiplet per layer’s compute and memory type, 1 stack per AI layer, 3 tiers per stack: Tier 1: Compute, Tier 2: Output Memory, Tier 3: Weight Memory
```
To generate default files with sufficient on-package memory, set:

```

SET_SUFF_BANK = True
```

## Output Files in DEBUG Mode

When DEBUG mode is enabled, the following output files and visualizations are generated:

```
Chip_Map Folder:
  Chip_map_Ci
    - Displays the schematic of tiles placed within chiplet Ci.
    - Shows the relative position of tiles according to their NoC coordinates.

Network_Map Folder:
  Network_Map_Ci
    - Illustrates the schematic of tiles, NoC routers, 3D links, and NoP interfaces within chiplet Ci.
    - Uses color coding:
        • Blue blocks – Tiles  
        • Grey blocks – NoC routers  
        • Red blocks – 3D links  

  Network_Map_System
    - Shows the schematic of all chiplets and NoP routers placed within the system package.
    - Layout reflects relative positions based on NoP coordinates.

Generated Plots and Reports:

chip_area_breakdown.png
    - Area pie chart breakdown of compute, memory, NoC, NoP router, and interface components.

chip_latency_energy_breakdown.png
    - Latency and energy pie chart breakdown across compute, network, DDR, and memory.

Cost_Breakdown_<aimodel>.png
    - Cost bar graph covering die, assembly, interposer, substrate, testing, and NRE contributions.

HW_config_plot.png
    - 3D visualization of the system showing chiplets and their corresponding TILES.

network_breakdown.png
    - Network latency and energy breakdown for 2D NoC, 2.5D NoP, and 3D NoC/NoP.

network_graph.png
    - Graphical representation of required inter-chip/tile connections to execute the AI model.

Yield_Breakdown_<aimodel>.png
    - Yield bar graph detailing die, assembly, interposer, and substrate yields.

```
All intermediate outputs can be found in the HW_configs and Network_configs folders.

## Citing this work
If you found this tool useful, please use the following bibtex to cite us
```
@ARTICLE{10844846,
  author={Wang, Zhenyu and Nalla, Pragnya Sudershan and Sun, Jingbo and Goksoy, A. Alper and Mandal, Sumit K. and Seo, Jae-sun and Chhabria, Vidya A. and Zhang, Jeff and Chakrabarti, Chaitali and Ogras, Umit Y. and Cao, Yu},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={HISIM: Analytical Performance Modeling and Design Space Exploration of 2.5D/3D Integration for AI Computing}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Artificial intelligence;Integrated circuit modeling;Three-dimensional displays;Chiplets;Computer architecture;Computational modeling;Benchmark testing;Integrated circuit interconnections;Analytical models;Data models;heterogeneous integration;2.5D/3D;chiplet;in-memory computing;network-on-package;thermal simulation},
  doi={10.1109/TCAD.2025.3531348}}

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
* Pragnya Sudershan Nalla 
* Nikhil Kumar Cherukuri
* Hanpei Liu
* Ashish Kumar Kola
* Zhenyu Wang 
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
