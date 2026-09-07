# -*- coding: utf-8 -*-
# *******************************************************************************
# Copyright (c)
# School of Electrical, Computer and Energy Engineering, Arizona State University
# Department of Electrical and Computer Engineering, University of Minnesota

# PI: Prof.Yu(Kevin) Cao
# All rights reserved.

# This source code is for HISIM: Analytical Performance Modeling and Design Exploration 
# of 2.5D/3D Heterogeneous Integration for AI Computing

# Copyright of the model is maintained by the developers, and the model is distributed under 
# the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
# http://creativecommons.org/licenses/by-nc/4.0/legalcode.
# The source code is free and you can redistribute and/or modify it
# by providing that the following conditions are met:
# 
#  1) Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
# 
#  2) Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# Developer list: 
#   Zhenyu Wang	    Email: zwang586@asu.edu                
#   Pragnya Nalla   Email: nalla052@umn.edu
#   Jingbo Sun      Email: jsun127@asu.edu
#   Jennifer Zhou   Email: 
# *******************************************************************************/
from hisim_model import HiSimModel
import pandas as pd 

#Test Case 1
print("Test Case 1: Running HISIM to obtain PPA")
print("AI Network: ViT")
print("HW configuration (Xbar-Npe-Ntile-Ntier-Nstack-chip_arch):1024-9-100-2-2-3.5D")
hisim = HiSimModel(
            chip_architect = "M3_5D",
            xbar_size = 1024,
            N_tile = 100,
            N_pe = 9,
            N_tier = 2,
            freq_computing = 1,
            fclk_noc = 1,
            placement_method = 5,
            router_times_scale = 1,
            percent_router = 1,
            tsv_pitch = 5,
            W2d = 32,
            ai_model = 'vit',
            thermal = False,
            N_stack =2,
        )
hisim.run_model()
print("Test case 1 done. Please check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information")
print("")
input("Press Enter to execute next test case")


#Test Case 2
print("Test Case 2: Running HISIM to obtain PPA")
print("AI Network: densenet121")
print("HW configuration (Xbar-Npe-Ntile-Ntier-Nstack-chip_arch):1024-36-64-2-2")
hisim.set_N_tile(64)
hisim.set_num_pe(36)
hisim.set_ai_model("densenet121")
hisim.run_model()
print("Test case 2 done. Please check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information")
print("")
input("Press Enter to execute next test case")
#Test Case 3
print("Test Case 3: Running HISIM to obtain PPA for different TSV pitches")
print("AI Network: densenet121")
print("HW configuration ( Xbar-Npe-Ntile-Ntier-Nstack-chip_arch):1024-36-81-2-1-3D")
hisim.set_N_tile(81)
hisim.set_N_stack(1)
hisim.set_chip_architecture("M3D")
tsv_pitch=[2,3,4,5,10,20] # um
for i in range(len(tsv_pitch)):
    print("TSV_pitch:", tsv_pitch[i], "um")
    hisim.set_tsv_pitch(tsv_pitch[i])
    _ = hisim.run_model()
hisim.set_tsv_pitch(5)
print("Test case 3 done. Please check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information")
print("")
input("Press Enter to execute next test case")

#Test Case 4
print("Test Case 4: Running HISIM to obtain PPA for different NoC bandwidths")
print("AI Network: densenet121")
print("HW configuration ( Xbar-Npe-Ntile-Ntier-Nstack-chip_arch):1024-36-81-2-1-3D")
noc_width=[i for i in range(1,32, 5)] # um
for i in range(1, len(noc_width)):
    print("number of links of 2D NoC:", noc_width[i])
    hisim.set_W2d(noc_width[i])
    _ = hisim.run_model()

hisim.set_W2d(32)
print("Test case 4 done. Please check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information")
print("")
input("Press Enter to execute next test case")

#Test Case 5
print("Test Case 5: Running HISIM to obtain PPA and thermal for different Ntier")
print("AI Network: densenet121")
print("HW configuration (Xbar-Npe-Ntile-Ntier-Nstack-chip_arch):1024-36-169-Depends on input config-1")
hisim.set_N_tile(169)
hisim.set_thermal("True")
N_tier=[i for i in range(4)] # um
for i in range(1, len(N_tier)):
    print("number of tiers:", N_tier[i])
    hisim.set_N_tier(N_tier[i])
    _ = hisim.run_model()
hisim.set_N_tier(2)
hisim.set_N_tile(81)
print("Test case 5 done. Please check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information")
print("")
input("Press Enter to execute next test case")

#Test Case 6
print("Test Case 6: Running HISIM to obtain PPA and thermal")
print("AI Network: densenet121")
print("HW configuration (Xbar-Npe-Ntile-Ntier-Nstack-chip_arch):1024-36-81-2-1-3D")
hisim.run_model()

print("Test case 6 done. Please check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information")
print("")
input("Press Enter to execute next test case")

#Test Case 7
print("Test Case 7: Running HISIM to obtain PPA and thermal")
print("AI Network: ViT")
print("HW configuration (Xbar-Npe-Ntile-Ntier-Nstack-chip_arch):1024-9-169-2-1-3D")
hisim.set_N_tile(169)
hisim.set_num_pe(9)
hisim.set_ai_model("vit")
hisim.run_model()

print("Test case 7 done. Please check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information")
print("")
input("Press Enter to execute next test case")

#Test Case 8
print("Test Case 8: Running HISIM to obtain PPA and thermal")
print("AI Network: densenet121")
print("HW configuration (Xbar-Npe-Ntile-Ntier-Nstack-chip_arch):1024-36-81-1-2-2.5D")
hisim.set_N_tier(1)
hisim.set_placement(1)
hisim.set_N_stack(2)
hisim.set_num_pe(36)
hisim.set_N_tile(81)
hisim.set_ai_model("densenet121")
#hisim.set_thermal(True)
hisim.run_model()
hisim.set_thermal(False)
hisim.set_placement(5)

print("Test case 8 done. Please check Results/PPA.csv for PPA information and Results/tile_map.png for tile mapping information")
print("")


