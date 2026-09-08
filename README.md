# HISIM 2.0

HISIM 2.0 is a fast analytical design-space exploration framework for
heterogeneous integration and chiplet-based AI systems. It supports
system modeling across **2D, 2.5D, 3D, and 3.5D integration** and
provides estimates for **Power, Temperature, Performance, Area, and Cost
(PTPAC)**.

HISIM combines analytical models for compute, memory, DDR,
network/interconnect, thermal behavior, and cost. Users can define
hardware architectures, system and chiplet organization, AI layer
mappings, and custom heterogeneous designs either through the HISIM
Workbench GUI or directly through the underlying HISIM source code.

![HISIM
Overview](https://github.com/mec-UMN/HISIM/blob/main/HISIM-SystolicArray/HISIM_2_0_Overview.jpg "HISIM Overview")

## Repository Structure

The top-level repository is organized as follows:

``` text
HISIM/
├── HISIM-IMC/                 # HISIM implementation for IMC-based architectures
├── HISIM-SystolicArray/       # HISIM 2.0 systolic-array analytical engine
├── backend/                   # Backend API and GUI simulation orchestration
├── examples/                  # Example input files and configurations
├── generated_files/           # Runtime-generated HISIM input files
├── runs/                      # Simulation run history, logs, and results
├── uploaded_files/            # User-uploaded custom input files
├── webui/                     # Browser-based HISIM GUI
├── .gitignore
├── README.md                  # Main repository documentation
└── requirements.txt           # Required Python dependencies
```

The main HISIM systolic-array analytical engine is organized as follows:

``` text
HISIM-SystolicArray/
├── Module_0_AI_Map/
│   ├── HISIM_2_0_AI_layer_information/   # AI model input configurations
│   └── util_chip/
│       └── HISIM_2_0_Files/              # AI network, chiplet, tile, and mapping utilities
├── Module_1_Compute/
│   └── HISIM_2_0_Files/
│       ├── HW_configs/                    # Compute and memory specification files
│       ├── Compute.json                   # Compute/memory analytical-model parameters
│       ├── Compute.py                     # Compute, memory, and DDR PPA evaluation
│       ├── CPU.py                         # CPU analytical models for non-linear operations
│       ├── SA.py                          # Systolic-array analytical models
│       └── Mem.py                         # Memory and DDR communication models
├── Module_2_Network/
│   └── HISIM_2_0_Files/
│       ├── Network_configs/               # Network specification files
│       ├── Network.json                   # Network analytical-model parameters
│       └── Compute.py                     # NoC, NoP, and 3D-link PPA evaluation
├── Module_3_Cost/                         # System cost evaluation
├── Module_4_Thermal/                      # Thermal simulation
├── Chip_Map.csv                           # Tile/chiplet organization
├── Layer_Map.csv                          # AI layer-to-hardware mapping
├── Sys_Map.csv                            # Chiplet placement in the system
├── config.py                              # Global HISIM configuration
└── HISIM.py                               # Main HISIM executable
```

## Quick Start

### 1. Clone the repository

Clone the HISIM repository and move into the repository directory. Then
switch to the `essc_2026_demo` branch:

``` bash
git clone https://github.com/mec-UMN/HISIM.git
cd HISIM
git checkout essc_2026_demo
```

### 2. Create and activate a virtual environment

Linux/macOS:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

``` powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.9 or later is required.

`pygraphviz` is optional and is required only when HISIM debug figure
generation is enabled.

### 4. Start the HISIM Workbench

``` bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Open the following address in a browser:

``` text
http://127.0.0.1:8000/
```

The GUI acts as a wrapper around the HISIM analytical engine.
Researchers can therefore modify the underlying HISIM models while
continuing to use the GUI, provided that the required input and output
structures remain unchanged.

## Running HISIM through the GUI

The HISIM Workbench provides an interactive interface for configuring
and evaluating architectures.

### Single Simulation

1.  Select the desired AI model.
2.  Configure the required hardware and system design knobs.
3.  Select **Run simulation**.
4.  Inspect the generated PTPAC results, breakdowns, system
    visualizations, and logs.

The GUI exposes parameters for compute, memory, network, packaging, and
system organization, including parameters such as systolic-array
dimensions, memory-bank configuration, on-chip/off-chip memory
organization, and network bandwidth.

### Design-Space Sweep

1.  Enable **Design-space sweep**.
2.  Select the hardware parameter to sweep.
3.  Specify the sweep values or range.
4.  Select the required AI workload(s).
5.  Select **Launch sweep**.
6.  Open **Runs & Compare** to inspect sweep responses, trade-offs, and
    scorecards.

Enable **Compare multiple models** when the selected sweep should be
evaluated for multiple AI models at every sweep point.

### Custom Input Files

For custom heterogeneous designs:

1.  Open **HISIM input files** in the Design Knobs panel.
2.  Download the current CSV files or templates if required.
3.  Upload the customized CSV files.
4.  Select **Run uploaded files**.

The input validator checks all six required files before invoking HISIM:

-   `Chip_Map.csv`
-   `Sys_Map.csv`
-   `Layer_Map.csv`
-   `sa_spec`
-   `mem_spec`
-   `network_spec`

## Running HISIM Directly

HISIM can also be executed without the GUI.

Move to the analytical-engine directory:

``` bash
cd HISIM-SystolicArray
```

Configure the desired architecture in `config.py` and, if required,
modify the mapping/specification CSV files.

Then run:

``` bash
python3 HISIM.py
```

All PTPAC outputs are printed to the terminal. Intermediate files and
visualizations are generated when debug mode is enabled.

## HISIM Input Files

HISIM uses six primary files to define the system architecture, design
partitioning, AI mapping, and hardware specifications.

### Mapping and Partitioning Files

#### `Sys_Map.csv`

Defines the positions of chiplets within the system/3D space.

#### `Chip_Map.csv`

Defines the position and hardware type of tiles within each chiplet.

#### `Layer_Map.csv`

Defines the mapping of AI layers onto hardware resources, including the
corresponding hardware dimensions and positions.

Together, these three files define the **system partitioning and AI
algorithm-to-hardware mapping**.

### Hardware Specification Files

#### `sa_spec`

Defines systolic-array specifications and dimensions for the compute
tiles.

Default location:

``` text
HISIM-SystolicArray/Module_1_Compute/HISIM_2_0_Files/HW_configs/
```

#### `mem_spec`

Defines memory parameters such as the number of banks, rows, columns,
and related memory configuration information.

Default location:

``` text
HISIM-SystolicArray/Module_1_Compute/HISIM_2_0_Files/HW_configs/
```

#### `network_spec`

Defines network-related specifications such as link bandwidths.

Default location:

``` text
HISIM-SystolicArray/Module_2_Network/HISIM_2_0_Files/Network_configs/
```

These files can be automatically generated for supported default
configurations or supplied by the user to represent custom heterogeneous
architectures.

## Configuration File

The `config.py` file defines the major global hardware and simulation
parameters used by HISIM.

### Generate Default Input Files

``` python
CREATE_DEFAULT_FILES = True
```

### Select Default Architecture

``` python
TYPE_DEFAULT_FILES = "2_5D_Mesh"
TYPE_DEFAULT_FILES = "2D_Mesh"
TYPE_DEFAULT_FILES = "3_5D_Mesh"
TYPE_DEFAULT_FILES = "2_5D_Mesh_Scaled"
TYPE_DEFAULT_FILES = "3_5D_Mesh_Scaled"
```

The configurations represent:

  ---------------------------------------------------------------------
  Configuration                      Description
  ---------------------------------- ----------------------------------
  `2_5D_Mesh`                        One tile per AI layer's compute
                                     and memory type, one chip, one
                                     tier

  `2D_Mesh`                          Three-chiplet system containing
                                     SRAM/DDR, SA+memory, and
                                     CPU+memory resources; one tier

  `3_5D_Mesh`                        Three-chiplet system containing
                                     SRAM/DDR, SA+memory, and
                                     CPU+memory resources; two tiers

  `2_5D_Mesh_Scaled`                 One component (CPU, memory, or SA)
                                     per chiplet; one tier

  `3_5D_Mesh_Scaled`                 One component per chiplet and one
                                     stack per AI layer, with compute,
                                     output-memory, and weight-memory
                                     tiers
  ---------------------------------------------------------------------

### Size Memory for Sufficient On-Package Capacity

``` python
SET_SUFF_BANKS = True
```

For experiments where the memory-bank count should be controlled
manually:

``` python
SET_SUFF_BANKS = False
```

## Design-Space Exploration Examples

The following examples reproduce representative HISIM design-space
exploration workflows.

### Example 1: DDR Latency vs. On-Chip Memory

This experiment studies how the number of on-chip SRAM banks affects DDR
latency.

#### Using HISIM Code

In `HISIM-SystolicArray/config.py`, set:

``` python
CREATE_DEFAULT_FILES = True
DEFAULT_FILES_GENERIC = True
TYPE_DEFAULT_FILES = "2D_Mesh"
SET_SUFF_BANKS = False
```

Select the desired AI workload.

Then open:

``` text
HISIM-SystolicArray/Module_0_AI_Map/util_chip/HISIM_2_0_Files/HW_Map.py
```

Change the number of on-chip SRAM banks through:

``` python
def_Nbank = getattr(config, "def_Nbank", 1)
```

Run HISIM:

``` bash
cd HISIM-SystolicArray
python3 HISIM.py
```

Repeat the experiment for the required values of `def_Nbank` and compare
the resulting DDR latency.

#### Using the GUI

1.  Turn off **Size banks for full on-package memory**.
2.  Turn on **Design-space sweep**.
3.  Under the sweep knob selection, choose **Memory & DDR → Banks per
    memory tile**.
4.  Select the AI workload(s).
5.  Define the required sweep range.
6.  Select **Launch sweep**.
7.  Open **Runs & Compare**.
8.  Under **Sweep Response**, select **DDR Latency** as the metric.
9.  Select the required models for comparison.

### Example 2: NoC Bandwidth Effect

The NoC bandwidth experiment follows the same general procedure as the
DDR experiment, but sweeps the corresponding **NoC link bandwidth**
parameter.

#### Using HISIM Code

Configure HISIM for the required workload and architecture, vary the NoC
link bandwidth in the relevant network specification/configuration, and
run:

``` bash
python3 HISIM.py
```

Repeat for each bandwidth value and compare the resulting network
metrics.

#### Using the GUI

1.  Enable **Design-space sweep**.
2.  Select the corresponding NoC link/network-bandwidth knob.
3.  Define the sweep range.
4.  Select the required AI workload(s).
5.  Launch the sweep.
6.  Use **Runs & Compare** to inspect the resulting network latency,
    energy, and other required metrics.

## Custom Heterogeneous Designs

### Example 3: Qwen Heterogeneous Tile Definition

This example evaluates user-defined heterogeneous tile configurations
using the six HISIM input files.

#### Using HISIM Code

In `config.py`, set:

``` python
CREATE_DEFAULT_FILES = False
DEFAULT_FILES_GENERIC = True
TYPE_DEFAULT_FILES = "2D_Mesh"
SET_SUFF_BANKS = False
```

From the root-level `examples/` folder, copy the corresponding set of
six CSV files into the following locations:

  -------------------------------------------------------------------------------------------
  File              Destination
  ----------------- -------------------------------------------------------------------------
  `sa_spec`         `HISIM-SystolicArray/Module_1_Compute/HISIM_2_0_Files/HW_configs/`

  `mem_spec`        `HISIM-SystolicArray/Module_1_Compute/HISIM_2_0_Files/HW_configs/`

  `network_spec`    `HISIM-SystolicArray/Module_2_Network/HISIM_2_0_Files/Network_configs/`

  `Chip_Map.csv`    `HISIM-SystolicArray/`

  `Sys_Map.csv`     `HISIM-SystolicArray/`

  `Layer_Map.csv`   `HISIM-SystolicArray/`
  -------------------------------------------------------------------------------------------

Replace the previously existing files where applicable and save the new
files.

Run:

``` bash
cd HISIM-SystolicArray
python3 HISIM.py
```

Repeat for each required configuration and compare the resulting metrics
as described in the report.

#### Using the GUI

1.  Turn off **Regenerate mapping files**.
2.  Turn off **Size banks for full on-package memory**.
3.  Open the HISIM input-file interface.
4.  Upload the corresponding six files one by one.
5.  Run the simulation.
6.  Repeat for each heterogeneous configuration and compare the
    resulting metrics.

## Notes

-   Runtime GUI data are written into `runs/`, `uploaded_files/`, and
    `generated_files/`. These directories are intentionally empty in a
    fresh clone and excluded from Git.
-   Simulation runs are isolated. A failed sweep point is recorded as
    failed and does not prevent the remaining sweep points from running.
-   The web interface loads Plotly, Chart.js, D3, and Three.js from
    public CDNs. Internet access is therefore required when the browser
    first loads the application unless these assets are vendored
    locally.
-   Graphviz and `pygraphviz` are required only for HISIM debug figure
    generation and are not required for normal GUI simulations.
-   Example CSV configurations are provided for testing user-defined
    mappings and heterogeneous system partitioning.

## Citing HISIM

If you find HISIM useful, please cite the following works.

``` bibtex
@ARTICLE{10844846,
  author={Wang, Zhenyu and Nalla, Pragnya Sudershan and Sun, Jingbo and Goksoy, A. Alper and Mandal, Sumit K. and Seo, Jae-sun and Chhabria, Vidya A. and Zhang, Jeff and Chakrabarti, Chaitali and Ogras, Umit Y. and Cao, Yu},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  title={HISIM: Analytical Performance Modeling and Design Space Exploration of 2.5D/3D Integration for AI Computing},
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Artificial intelligence;Integrated circuit modeling;Three-dimensional displays;Chiplets;Computer architecture;Computational modeling;Benchmark testing;Integrated circuit interconnections;Analytical models;Data models;heterogeneous integration;2.5D/3D;chiplet;in-memory computing;network-on-package;thermal simulation},
  doi={10.1109/TCAD.2025.3531348}
}

@INPROCEEDINGS{10396377,
  author={Wang, Zhenyu and Sun, Jingbo and Goksoy, Alper and Mandal, Sumit K. and Seo, Jae-Sun and Chakrabarti, Chaitali and Ogras, Umit Y. and Chhabria, Vidya and Cao, Yu},
  booktitle={2023 IEEE 15th International Conference on ASIC (ASICON)},
  title={Benchmarking Heterogeneous Integration with 2.5D/3D Interconnect Modeling},
  year={2023},
  volume={},
  number={},
  pages={1-4},
  keywords={Analytical models;Three-dimensional displays;Computational modeling;Multichip modules;Benchmark testing;Data models;Artificial intelligence;Heterogeneous Integration;2.5D;3D;Chiplet;ML accelerators;Electro-thermal Co-design},
  doi={10.1109/ASICON58565.2023.10396377}
}

@INPROCEEDINGS{10473875,
  author={Wang, Zhenyu and Sun, Jingbo and Goksoy, Alper and Mandal, Sumit K. and Liu, Yaotian and Seo, Jae-Sun and Chakrabarti, Chaitali and Ogras, Umit Y. and Chhabria, Vidya and Zhang, Jeff and Cao, Yu},
  booktitle={2024 29th Asia and South Pacific Design Automation Conference (ASP-DAC)},
  title={Exploiting 2.5D/3D Heterogeneous Integration for AI Computing},
  year={2024},
  volume={},
  number={},
  pages={758-764},
  keywords={Analytical models;Three-dimensional displays;Computational modeling;Wires;Multichip modules;Benchmark testing;Transformers;Heterogeneous Integration;2.5D;3D;Chiplet;ML accelerators;Performance Analysis},
  doi={10.1109/ASP-DAC58780.2024.10473875}
}
```

## Developers and Maintainers

### Main Developers

-   Pragnya Sudershan Nalla
-   Nikhil Kumar Cherukuri
-   Hanpei Liu
-   Ashish Kumar Kola
-   Zhenyu Wang
-   Jingbo Sun
-   Emad Haque
-   Tanishq Kekre
-   Ganap Ashit Tewary

### Contributors

-   A. Alper Goksoy

### Maintainers and Advisors

-   Sumit K. Mandal
-   Jae-sun Seo
-   Vidya A. Chhabria
-   Jeff Zhang
-   Chaitali Chakrabarti
-   Umit Y. Ogras
-   Yu Cao
