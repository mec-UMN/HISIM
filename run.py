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
# *******************************************************************************/
import os
import csv
filename_results = "./Results/PPA.csv"
COMPUTE_VALIDATE=False
with open(filename_results, 'a', newline='') as csvfile:
    # Create a csv writer object
    writer = csv.writer(csvfile)
    # Write the header row (optional)
    if COMPUTE_VALIDATE:
        writer.writerow(["freq_core","freq_noc","Xbar_size","N_tile","N_pe","N_tile(real)","N_tier(chiplet)","W2d","W3d","Computing_latency", "Computing_energy","compute_area","chip_area", "chip_Architecture","2d NoC latency","3d NoC latency","2.5d NoC latency", "network_latency","2d NoC energy","3d NoC energy","2.5d NoC energy","network_energy","rcc","TFLOPS","compute_power", "2D_3D_NoC_power","2_5D_power","2d_3d_router_area", "peak_temperature", "placement_method","percent_router", "array latency", "adc latency",  "array energy","adc energy", "shiftadd latency", "accum latency", "shiftadd energy","accum energy", "control latency","control energy","array area","adc area", "shiftadd area","accum area", "control area" , "routing area", "routing latency","routing energy","buffer area", "buffer latency","buffer energy"])
    else:
        writer.writerow(["freq_core","freq_noc","Xbar_size","N_tile","N_pe","N_tile(real)","N_tier(chiplet)","W2d","W3d","Computing_latency", "Computing_energy","compute_area","chip_area","chip_Architecture","2d NoC latency","3d NoC latency","2.5d NoC latency", "network_latency","2d NoC energy","3d NoC energy","2.5d NoC energy","network_energy","rcc","TFLOPS", "2D_3D_NoC_power","2_5D_power","2d_3d_router_area","peak_temperature","placement_method","percent_router"])

mode=0 #single corner case
#mode=1#DESIGN SPACE
#mode=2#customize 
#mode=3#l/tsv
#mode=4#t/l
if mode==0:
    crossbar_size=[1024] 
    N_tile=[100]
    N_pe=[9]
    N_tier=[3]   
    f_core=[1]
    f_noc=[1]
    method=[5]
    router_times_scale=[1]
    percent_router=[1]
    tsv_pitch=[5]
    W2d=[32]
    chip_arch=["M3D"]
    ai_model=['vit']
elif mode==1:
    crossbar_size=[1024] 
    N_tile=[16,25,36,49,64,81,100,121,144]
    N_pe=[36]
    N_tier=[2]   
    f_core=[1]
    f_noc=[1]
    method=[1]
    router_times_scale=[1]
    percent_router=[1]
    tsv_pitch=[2,3,4,5,10,20] # um
    W2d=[32]
    chip_arch=["H2_5D"]
elif mode==2:
    crossbar_size=[] 
    N_tile=[16,25,36,49,64,81,100,121,144]
    f_core=[0.75]
    f_noc=[0.75]
    tsv_pitch=[5] # um
elif mode==3:
    crossbar_size=[] 
    N_tile=[]
    f_core=[0.75]
    f_noc=[0.75]
    tsv_pitch=[2, 3, 4, 5,10, 15] # um
elif mode==4:
    crossbar_size=[256] 
    N_tile=[36]
    N_tier=[4]
    f_core=[0.75]
    f_noc=[0.75]
    tsv_pitch=[5] # um

# For design space search
# HISIM will generate all results for different configurations
for i in crossbar_size:
    for i_tile in N_tile:
        for i_tier in N_tier:
            for pe in N_pe:
                for fcore in f_core:    
                    for fnoc in f_noc:
                        for i_scale in router_times_scale:
                            for i_w2d in W2d:
                                for tsvpitch in tsv_pitch:
                                    for placement in method:
                                        for p_router in percent_router:
                                            for i_arch in chip_arch:
                                                for i_model in ai_model:
                                                    os.system('python analy_model.py --xbar_size %d \
                                                        --N_tile %d \
                                                        --N_tier %d \
                                                        --N_pe %d \
                                                        --freq_computing %f \
                                                        --fclk_noc %f \
                                                        --placement_method %d \
                                                        --percent_router %f\
                                                        --tsvPitch %f \
                                                        --chip_architect %s\
                                                        --W2d %d\
                                                        --router_times_scale %d\
                                                        --ai_model %s ' %(int(i),int(i_tile),int(i_tier),int(pe),float(fcore),float(fnoc),float(placement),float(p_router),float(tsvpitch), str(i_arch), int(i_w2d),int(i_scale), str(i_model)))
                                        

