# HISIM_V2.0_Systolic Array
HISIM introduces a suite of analytical models for compute, memory, ddr, network to speed up performance prediction for AI models, covering logic-on-logic architectures across 2D, 2.5D, 3D and 3.5D integration.

## File Lists
The main directory structure of the repository is shown below. Please run 'python HISIM.py' to run the tool. Input files for the AI layer, compute modules, network, and their interactions can be found in the respective folders below.
```
├── Module_AI_Map
    ├── HISIM_2_0_AI_layer_information         -- Folder containing input configuration files that describe the AI models
    ├── util_chip                              
        ├── HISIM_2_0_Files                    -- Input Folder containing codes to load AI network, chiplet and tile mapping files
├── Module_Compute                              
    ├── HISIM_2_0_Files                        
        ├── HW_configs                         -- Input Folder containing the specification files that describe the memory, systolic array specifications for each tile 
        ├──Compute.json                        -- Input File containing the parameter values used within the HISIM compute and memory analytical models
        ├──Compute.py                          -- Code performing PPA evaluation of the Compute, Memory and DDR of tiles
        ├──CPU.py                              -- Code containing analytical models of CPU to execute non-linear activation units
        ├──SA.py                               -- Code containing analytical models of systolic array to execute matrix multiplication and convolution
        ├──Mem.py                              -- Code containing analytical models of Memory and DDR communication
├── Module_Cost                                -- Folder containing code for cost evaluation of the system
├── Module_Network          
    ├── HISIM_2_0_Files  
        ├── Network_configs                    -- Folder containing the input specification file that describes the network specifications for each stack 
        ├── Network.json                       -- Input File containing the parameter values used within the HISIM network analytical models
        ├──Compute.py                          -- Code performing PPA evaluation of the noc, nop and 3d links within the system
├── Module_Thermal                             -- Folder containing thermal simulation code for 2D, 2.5D, and 3D architectures
├── Results                                    -- Folder containing the visualization outputs to gudie the user in DEBUG mode
├── Z_Map_Files_Sample                         -- Sample 2.5D/3D files to run attention layer
├── Chip_Map.csv                               -- Input file describing the HW tile to AI layer mapping
├── config.py                                  -- Code to set global variables across all files. Please set DEBUG = TRUE to get intermediate output plots 
├── HISIM.py                                   -- Main file to run the hisim code
├── Layer_Map.csv                              -- Input file describing how the user wants to map the AI layer onto the systolic array
├── Sys_Map.csv                                -- Input file describing the chiplet placement within the stack


All Power, Temperature, Performance, Area, and Cost, (PTPAC) outputs are printed in the terminal output and the intermediatory outputs are saved in HW_configs, Network_configs and Results fodler in DEBUG mode.

```


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
