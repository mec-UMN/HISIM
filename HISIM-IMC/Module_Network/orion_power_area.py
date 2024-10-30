
import numpy as np
import scipy.sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra
import matplotlib.pyplot as plt

import pandas as pd

# tech parameters

# area areaCrossbar+areaInputModule+areaOutputModule

# 2007 ITRS predictions for a 32nm high-performance library
H_INVD2  = 8#int
W_INVD2  = 3#int
H_DFQD1  = 8#int
W_DFQD1  = 16#int
H_ND2D1  = 8#int
W_ND2D1  = 3#int
H_SRAM  = 8#int
W_SRAM  = 6#int
Vdd  = 0.9#float
R  = 606.321#float
IoffP  = 0.00000102#float
IoffN  = 0.00000102#float
IoffSRAM  = 0.00000032
Cg_pwr  = 0.000000000000000534#float
Cd_pwr  = 0.000000000000000267#float
Cgdl  = 0.0000000000000001068#float
Cg  = 0.000000000000000534#float
Cd  = 0.000000000000000267#float
LAMBDA  = 0.016#float
MetalPitch  = 0.000080#float
Rw  = 0.0435644#float

Ci_delay=3*(Cg+Cgdl)
Co_delay=3*Cd
Ci = (1.0 + 2.0) * Cg_pwr 
Co = (1.0 + 2.0) * Cd_pwr 
FO4    = R * ( 3.0 * Cd + 12 * Cg + 12 * Cgdl)		     
tCLK   = 20 * FO4
fCLK   = 1.0 / tCLK       
ChannelPitch = 2.0 * MetalPitch 
CrossbarPitch = 2.0 * MetalPitch        
#channel_width=32
numVC=3
buf_size = 10
depthVC = 10
output_buffer_size = 1
input_switch=6
output_switch=6




def interconnect_type(type):
    # TSV wire optimization
    if type=="TSV":
        Cw_gnd  = 0.00000000000001509
        Cw_cpl  = 0.00000000000001509
        K=1
        M=1
        N=1
        Cw=2.0*Cw_cpl+2.0*Cw_gnd
        wire_length  = 0.05

    if type=="2D_NoC":
    # 2D NoC optimization
        Cw_gnd  = 0.000000000000267339
        Cw_cpl  = 0.000000000000267339
        K = 8.1
        M = 2 
        N = 1
        Cw=2.0*Cw_cpl+2.0*Cw_gnd
        wire_length  = 2.0
    return K,M,N,wire_length,Cw

#---------------------------------------------------------------------#

#         channel power and area

#---------------------------------------------------------------------#
def Power_Module_powerRepeatedWire(L,  K,  M, N,Cw):
    segments = 1.0 * M * N
    Ca = K * (Ci + Co) + Cw * (L/segments) 
    Pa = 0.5 * Ca * Vdd * Vdd * fCLK
    return Pa * M * N  
def Power_Module_powerWireClk (M, W,Cw):
    #number of clock wires running down one repeater bank
    columns = H_DFQD1 * MetalPitch /  ChannelPitch 

    #length of clock wire
    clockLength = W * ChannelPitch 
    Cclk = (1 + 5.0/16.0 * (1+Co_delay/Ci_delay)) * (clockLength * Cw * columns +W * Ci_delay)

    return M * Cclk * (Vdd * Vdd) * fCLK 

def Power_Module_powerRepeatedWireLeak (K, M, N):
    Pl = K * 0.5 * ( IoffN + 2.0 * IoffP ) * Vdd  
    return Pl * M * N 

def Power_Module_powerWireDFF(M, W, alpha=1):
    Cdin = 2 * 0.8 * (Ci + Co) + 2 * ( 2.0/3.0 * 0.8 * Co )  
    Cclk = 2 * 0.8 * (Ci + Co) + 2 * ( 2.0/3.0 * 0.8 * Cg_pwr) 
    Cint = (alpha * 0.5) * Cdin + alpha * Cclk 
  
    return Cint * M * W * (Vdd*Vdd) * fCLK 

def Power_Module_calcChannel(channel_width,K,M,N,wire_length,Cw):
    channel_wire_power=Power_Module_powerRepeatedWire(wire_length,K,M,N,Cw)*channel_width
    channel_clk_power=Power_Module_powerWireClk(M,W=channel_width,Cw=Cw)
    channel_DFFPower=Power_Module_powerWireDFF(M,W=channel_width,alpha=1)
    channelLeakPower= Power_Module_powerRepeatedWireLeak(K,M,N)*channel_width
    channelArea=Power_Module_areaChannel(K,M,N,channel_width)
    dynamic_power=channel_wire_power+channel_clk_power+channel_DFFPower
    return channel_wire_power,channel_clk_power,channel_DFFPower,channelLeakPower,channelArea

def Power_Module_areaChannel (K, N, M,channel_width):

    Adff = M * W_DFQD1 * H_DFQD1 
    Ainv = M * N * ( W_INVD2 + 3 * K) * H_INVD2 

    return channel_width * (Adff + Ainv) * MetalPitch * MetalPitch 

#---------------------------------------------------------------------#

#         Memory power and area

#---------------------------------------------------------------------#

def Power_Module_calcBuffer(channel_width,input_switch,Cw):
    depth = numVC * depthVC
    Pleak=depth*IoffSRAM*Vdd
    inputArea=Power_Module_areaInputModule(depth,channel_width)*input_switch
    Pwl =  Power_Module_powerWordLine( channel_width, depth,channel_width,Cw) 
    Prd = Power_Module_powerMemoryBitRead( depth,Cw ) * channel_width
    Pwr = Power_Module_powerMemoryBitWrite( depth,Cw ) * channel_width
    inputReadPower  = ( Pwl + Prd ) 
    inputWritePower  =( Pwl + Pwr )
    return inputReadPower,inputWritePower,Pleak,inputArea

def Power_Module_areaInputModule(Words,channel_width):
    Asram =  ( channel_width * H_SRAM ) * (Words * W_SRAM) 
    return Asram * (MetalPitch * MetalPitch) 

def Power_Module_powerWordLine(memoryWidth, memoryDepth,channel_width,Cw):
  # wordline capacitance
    Ccell = 2 * ( 4.0 * LAMBDA ) * Cg_pwr +  6 * MetalPitch * Cw      
    Cwl = memoryWidth * Ccell  

    # wordline circuits
    Warray = 8 * MetalPitch + memoryDepth 
    x = 1.0 + (5.0/16.0) * (1 + Co/Ci)  
    Cpredecode = x * (Cw * Warray  * Ci) 
    Cdecode    = x * Cwl 

    # bitline circuits
    Harray =  6 * memoryWidth * MetalPitch 
    y = (1 + 0.25) * (1 + Co/Ci) 
    Cprecharge = y * ( Cw * Harray + 3 * channel_width * Ci ) 
    Cwren      = y * ( Cw * Harray + 2 * channel_width * Ci ) 

    Cbd = Cprecharge + Cwren 
    Cwd = 2 * Cpredecode + Cdecode 

    return ( Cbd + Cwd ) * Vdd * Vdd * fCLK 
def Power_Module_powerMemoryBitRead(memoryDepth,Cw):
    # bitline capacitance
    Ccell  = 4.0 * LAMBDA * Cd_pwr + 8 * MetalPitch * Cw 
    Cbl    = memoryDepth * Ccell 
    Vswing = Vdd  
    return ( Cbl ) * ( Vdd * Vswing ) * fCLK 
def Power_Module_powerMemoryBitWrite(memoryDepth,Cw):
    # bitline capacitance
    Ccell  = 4.0 * LAMBDA * Cd_pwr + 8 * MetalPitch * Cw  
    Cbl    = memoryDepth * Ccell 

    # internal capacitance
    Ccc    = 2 * (Co + Ci) 

    return (0.5 * Ccc * (Vdd*Vdd)) + ( Cbl ) * ( Vdd * Vdd ) * fCLK 


#---------------------------------------------------------------------#

#         switch power and area

#---------------------------------------------------------------------#
def Power_Module_calcSwitch(channel_width,Cw):
    switchArea = Power_Module_areaCrossbar(input_switch, output_switch,channel_width)
    outputArea=Power_Module_areaOutputModule(output_switch,channel_width)
    switchPowerLeak=Power_Module_powerCrossbarLeak(channel_width,input_switch,output_switch,Cw)

    switchPower=channel_width*Power_Module_powerCrossbar(channel_width,input_switch,output_switch,Cw)
    swicthPowerCtrl=Power_Module_powerCrossbarCtrl(channel_width,  input_switch,input_switch,Cw)
    outputPowerClk = Power_Module_powerWireClk( 1, channel_width,Cw )
    outputPower =  Power_Module_powerWireDFF( 1, channel_width, 1.0 )
    outputCtrlPower = Power_Module_powerOutputCtrl(channel_width ,Cw)
    return switchPower,swicthPowerCtrl,switchPowerLeak,switchArea,outputPower,outputPowerClk,outputCtrlPower,outputArea

def Power_Module_areaCrossbar(Inputs, Outputs,channel_width) :
    return (Inputs * channel_width * CrossbarPitch) * (Outputs * channel_width * CrossbarPitch) 
def Power_Module_areaOutputModule(Outputs,channel_width):
    Adff = Outputs * W_DFQD1 * H_DFQD1 
    return channel_width * Adff * MetalPitch * MetalPitch 

def Power_Module_powerCrossbarLeak (width, inputs, outputs,Cw):
    
    Wxbar = width * outputs * CrossbarPitch 
    Hxbar = width * inputs  * CrossbarPitch 
    #wires
    CwIn  = Wxbar * Cw 
    CwOut = Hxbar * Cw 
    # cross-points
    Cxi = (1.0/16.0) * CwOut 
    # driver
    Cti  = (1.0/16.0) * CwIn 

    return 0.5 * (IoffN + 2 * IoffP)*width*(inputs*outputs*Cxi+inputs*Cti+outputs*Cti)/Ci

def Power_Module_powerCrossbar( width,  inputs,  outputs,Cw):
  # datapath traversal power
    Wxbar = width * outputs * CrossbarPitch 
    Hxbar = width * inputs  * CrossbarPitch 

    # wires
    CwIn  = Wxbar * Cw 
    CwOut = Hxbar * Cw 

    # cross-points
    Cxi = (1.0/16.0) * CwOut 
    Cxo = 4.0 * Cxi * (Co_delay/Ci_delay) 

    # drivers
    Cti = (1.0/16.0) * CwIn 
    Cto = 4.0 * Cti * (Co_delay/Ci_delay) 

    CinputDriver = 5.0/16.0 * (1 + Co_delay/Ci_delay) * (0.5 * Cw * Wxbar + Cti) 

    # total switched capacitance
    
    #this maybe missing +Cto
    Cin  = CinputDriver + CwIn + Cti + (outputs * Cxi) 

    #this maybe missing +cti
    Cout = CwOut + Cto + (inputs * Cxo) 

    return 0.5 * (Cin + Cout) * (Vdd * Vdd * fCLK) 

def Power_Module_powerCrossbarCtrl( width, inputs, outputs,Cw):
 
    # datapath traversal power
    Wxbar = width * outputs * CrossbarPitch 
    Hxbar = width * inputs  * CrossbarPitch 

    # wires
    CwIn  = Wxbar * Cw 

    # drivers
    Cti  = (5.0/16.0) * CwIn 

    # need some estimate of how many control wires are required
    Cctrl  = width * Cti + (Wxbar + Hxbar) * Cw   
    Cdrive = (5.0/16.0) * (1 + Co_delay/Ci_delay) * Cctrl 

    return (Cdrive + Cctrl) * (Vdd*Vdd) * fCLK 

def Power_Module_powerOutputCtrl(width,Cw):

    Woutmod = width * ChannelPitch 
    Cen     = Ci 

    Cenable = (1 + 5.0/16.0)*(1.0+Co/Ci)*(Woutmod* Cw + width* Cen) 

    return Cenable * (Vdd*Vdd) * fCLK 
        
def power_summary_router(channel_width,input_switch,output_switch,hop,trc,tva,tsa,tst,tl,tenq,Q,N_chiplet,mesh_edge):
    if input_switch==6:
        type="TSV"
    else:
        type="2D_NoC"

    K,M,N,wire_length,Cw=interconnect_type(type)
    channel_wire_power,channel_clk_power,channel_DFFPower,channelLeakPower,channelArea=Power_Module_calcChannel(channel_width,wire_length,K,M,N,Cw)
    inputReadPower,inputWritePower,Pleak,inputArea=Power_Module_calcBuffer(channel_width=channel_width,input_switch=input_switch,Cw=Cw)
    switchPower,switchPowerCtrl,switchPowerLeak,switchArea,outputPower,outputPowerClk,outputCtrlPower,outputArea=Power_Module_calcSwitch(channel_width=channel_width,Cw=Cw)
    #total_power=channel_wire_power+channel_clk_power+channel_DFFPower+channelLeakPower+inputReadPower+inputWritePower+Pleak+switchPower+switchPowerCtrl+outputPower+outputPowerClk+outputCtrlPower
    
    # total noc cycle
    Latency_cycle=hop*(trc+tva+tsa+tst+tl)+(tenq)*(Q/channel_width)
    if input_switch==6:
        channel_wire_power=channel_wire_power*(hop*tl+Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet-1)
        channel_DFFPower=channel_DFFPower*(hop*tl+Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet-1)
        inputReadPower=inputReadPower*(tenq)*(Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
        inputWritePower=inputWritePower*(tenq)*(Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
        switchPower=switchPower*hop*(trc+tva+tsa)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
        switchPowerCtrl=switchPowerCtrl*(hop*(trc+tva+tsa)+Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
        outputPower=outputPower*(hop*(tst)+Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
        outputCtrlPower=outputCtrlPower*(hop*(tst)+Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
    if input_switch==5:
        channel_wire_power=channel_wire_power*(hop*tl+Q/channel_width)/Latency_cycle*mesh_edge*(mesh_edge*2-2)*N_chiplet
        channel_DFFPower=channel_DFFPower*(hop*tl+Q/channel_width)/Latency_cycle*mesh_edge*(mesh_edge*2-2)*N_chiplet
        inputReadPower=inputReadPower*(tenq)*(Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
        inputWritePower=inputWritePower*(tenq)*(Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
        switchPower=switchPower*hop*(trc+tva+tsa)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
        switchPowerCtrl=switchPowerCtrl*(hop*(trc+tva+tsa)+Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
        outputPower=outputPower*(hop*(tst)+Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)
        outputCtrlPower=outputCtrlPower*(hop*(tst)+Q/channel_width)/Latency_cycle*mesh_edge*mesh_edge*(N_chiplet)

    channel_clk_power=channel_clk_power*(2*mesh_edge*mesh_edge*N_chiplet+2*mesh_edge*mesh_edge*(N_chiplet-1))
    outputPowerClk=outputPowerClk*hop*(tst)/Latency_cycle*(2*mesh_edge*mesh_edge*(N_chiplet-1)+2*mesh_edge*mesh_edge*N_chiplet)
    channelArea=channelArea*(mesh_edge*mesh_edge*(N_chiplet)*2+ 2*mesh_edge*(mesh_edge*2-2)*N_chiplet+2*mesh_edge*mesh_edge*(N_chiplet-1))
    switchArea=switchArea*mesh_edge*mesh_edge*(N_chiplet)
    inputArea=inputArea*mesh_edge*mesh_edge*(N_chiplet)
    outputArea=outputArea*mesh_edge*mesh_edge*(N_chiplet)
    total_area_router=channelArea+switchArea+inputArea+outputArea


    return total_area_router, channel_wire_power+channel_clk_power+channel_DFFPower,inputReadPower+inputWritePower+switchPower+switchPowerCtrl+outputPower+outputPowerClk+outputCtrlPower

