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

### PPA File

## Installation and Usage

### Dependencies

### Running analy_model.py file

### Running run.py file

## Examples

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
