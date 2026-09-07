import csv,math,sys
import json 
import os

# Load parameters from JSON file
current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, 'params.json')

with open(relative_path) as f:
    params = json.load(f)

#Clocks on the chip - device side
noc_rd_clk= params["noc_rd_clk_GHz"]        #Read clock for Tx module in AIB adapter
noc_wr_clk= params["noc_wr_clk_GHz"]        #Write clock for Rx module in AIB adapter

#AIB Specifications on the wire side 
n_ch= params["n_ch"]                        #Number of AIB channels
n_IO= params["n_IO"]                        #Number of AIB IOs per channel (Rx IOs +Tx IOs + Other)
n_Rx_config= params["n_Rx_config"]          #Number of Rx data lines per channel
n_Tx_config= params["n_Tx_config"]          #Number of Tx data lines per channel
aib_ns_fwd_clk= params["aib_ns_fwd_clk_GHz"]    #Near side clock TX AIB IO Module 
aib_fs_fwd_clk= params["aib_fs_fwd_clk_GHz"]    #Far side clock recieved by RX AIB IO Module
aib_Volt= params["aib_Volt"]                #Voltage on wire side connecting two AIB modules 

#Power activity factors
alpha_aib= params["alpha_aib"]              #Fraction of the duration for which the AIB is switched on
alpha_in= params["alpha_in"]                #1-input sparsity

#Specifications of wire interface connecting two AIB modules 
W_wire= params["W_wire_um"]                 #Width of EMIB wire
Len_wire= params["L_wire_mm"]               #Length of EMIB wire
C_in= params["C_in_pF"]                     #Input capacitance of the wire
C_out= params["C_out_pF"]                   #Output capacitance of the wire
Cg_wire= params["Cg_wire_pf"]               #Ground capacitance of the wire
R_on= params["R_on_ohm"]                    #Source on resistance on the IO side
R_wire= params["R_wire_ohm"]                #Resistance of the wire

#Microbump Dimensions
p_rw= params["p_rw_um"]                     #Aligned-row bump-to-bump pitch
p_col= params["p_col_um"]                   #Aligned-column bump-to-bump pitch

#Assign number of IOs per AIB column based on AIB 2.0 layout recommendation and microbump dimensions
if p_rw>=55:
    n_IOcl = 3
elif p_rw>=52:
    n_IOcl = 6
elif p_rw>=20:
    n_IOcl = 12

#Assign the FIFO mode based on clock: 1x, 2x, 4x mode 
if (aib_fs_fwd_clk==noc_rd_clk*2) and (aib_ns_fwd_clk==noc_wr_clk*2):
    fifo_mode=1
elif (aib_fs_fwd_clk==noc_rd_clk*4) and (aib_ns_fwd_clk==noc_wr_clk*4):
    fifo_mode=2
else:
    fifo_mode=4

#Area coefficients is in um^2
β1= params["Other Area _β1"]                
β2= params["Other Adapter Area_β2"]             
A_IO= params["Tx or Rx Area _A_IO_buffer"]
δ1= params["Tx Adapter Area_δ1"]
δ2= params["Other Tx Adapter Area_δ2"]
δ3= params["Rx Adapter Area_δ3"]
δ4= params["Other Rx Adapter Area_δ4"]
validate =False

#Power coefficients is in uW
γ1= params["γ2_wr_clk"]
γ2= params["γ2_rd_clk"]
γ3= params["γ2_ns_fwd_clk"]
γ4= params["γ2_fs_fwd_clk"]

γ5_1= params["γ1_in_tx_clk"]
γ5_2= params["γ1_wr_clk"]
γ5_3= params["γ1_ns_fwd_clk"]

γ6_1= params["γ1_in_rx_clk"]
γ6_2= params["γ1_rd_clk"]
γ6_3= params["γ1_fs_fwd_clk"]

# Latency coefficients is in ns
#1x FIFO mode:
rx_1x_fifo= params["rx_1x_fifo"]
rx_1x_io= params["rx_1x_io"]
tx_1x_fifo= params["tx_1x_fifo"]
tx_1x_io= params["tx_1x_io"]
#2x FIFO mode:
rx_2x_fifo= params["rx_2x_fifo"]
rx_2x_io= params["rx_2x_io"]
tx_2x_fifo= params["tx_2x_fifo"]
tx_2x_io= params["tx_2x_io"]
#4x FIFO mode:
rx_4x_fifo= params["rx_4x_fifo"]
rx_4x_io= params["rx_4x_io"]
tx_4x_fifo= params["tx_4x_fifo"]
tx_4x_io= params["tx_4x_io"]


def aib(Q, Len_chip, mode, volt):
    #Q: data volume is in MegaBytes

    #AIB Tx area, energy and latency (40Tx, 0Rx)
    area_Tx, BW =area_aib(Len_chip, mode, n_Tx_config*2, 0)        
    latency_Tx, energy_Tx,energy_eff_Tx =performance_aib(volt,n_Tx_config*2, 0)
    
    #area, energy, power and latency of wire interface connecting AIB modules of two chiplets
    A_wire, L_wire,E_wire, P_wire = area_performance_wire(n_Tx_config*2, 0)

    #AIB Rx area, energy and latency(0Tx, 40Rx)
    area_Rx, BW =area_aib(Len_chip, mode, 0, n_Rx_config*2)
    latency_Rx, energy_Rx,energy_eff_Rx =performance_aib(volt,0, n_Rx_config*2)

    #Number of AIB transmissions required to send the total data volume Q
    N_tr=math.ceil(Q*1e+6*8/(max(n_Tx_config*2, n_Rx_config*2)*8*n_ch))

    #total area, energy and latency
    area=area_Tx+A_wire+area_Rx
    energy=energy_Tx+E_wire+energy_Rx
    latency=latency_Tx+L_wire+latency_Rx
    energy*=N_tr
    latency*=N_tr

    if validate:
        #To validate PPA compared to the AIB tapeout (40Tx, 40 Rx) - https://ieeexplore.ieee.org/document/10374406
        area_Tx, BW =area_aib(Len_chip, mode, n_Tx_config, n_Rx_config)
        latency_Tx, energy_Tx,energy_eff_Tx =performance_aib(volt,n_Tx_config, n_Rx_config)

    #print("AIB:",round(area_Tx,2), "mm2 ",round(latency_Tx,2), "ns ", round(energy_Tx,2), "pJ ", round(energy_eff_Tx,2), "pJ/bit",round(BW,2), "Tbps" , round(energy_Tx/latency_Tx/area_Tx,2), "mW/mm2" )
    #print("EMIB:", round(A_wire,2), "mm2 ",round(L_wire,2), "ns ", round(E_wire,2), "pJ ")
    return [area, energy, latency, area_Tx, area_Rx, A_wire, energy_Tx, energy_Rx, P_wire, latency_Tx, latency_Rx]

def area_aib(Len, mode, n_Tx, n_Rx):
    #IO Area
    if mode ==0:
        #assuming equal length and width dimensions for a IO cell 
        Len_IO = math.sqrt(A_IO)*n_IO           #Total length of AIB IO cell in one channel 
        W_IO = math.sqrt(A_IO)                  #Total width of AIB IO cell in one channel
    else:
        #If total length of the AIB Module needs to be fixed due to external constraints
        Len_IO = Len*1e+3/n_ch                  #Total Length of AIB IO cell in one channel
        W_IO = A_IO/(Len_IO/n_IO)               #Total width of AIB IO cell in one channel         

    #Adapter Area
    A_adpt_inital= δ1*n_Tx+δ2+δ3*n_Rx+δ4+β2     
    A_other = β1
    W_adapter = math.sqrt(A_adpt_inital+A_other) #Assuming uniform dimensions for an adapter

    #Calculate total area with every I/O cell positioned along the same edge
    area = Len_IO*(W_IO+W_adapter)*n_ch*1e-6    #Total area of the AIB module in mm2

    BW=2*aib_ns_fwd_clk*(n_Tx+n_Rx)*n_ch*1e-3   #Aggregrated AIB Bandwidth in Tbps

    return area, BW

def performance_aib(volt, n_Tx, n_Rx):
    #Latency
    if (fifo_mode==1):
        #FIFO 1x mode
        L_Rx_adpt=(2+rx_1x_fifo)/noc_rd_clk     #Latency of AIB Adapter Rx module in ns
        L_Rx_IO = rx_1x_io/aib_fs_fwd_clk       #Latency of AIB Rx IO module in ns
        L_Tx_adpt=(1+tx_1x_fifo)/noc_wr_clk     #Latency of AIB Adapter Tx module in ns
        L_Tx_IO = (1+tx_1x_io)/aib_ns_fwd_clk   #Latency of AIB Tx IO module in ns
    elif (fifo_mode==2):
        L_Rx_adpt=(2+rx_2x_fifo)/noc_rd_clk      
        L_Rx_IO = rx_2x_io/aib_fs_fwd_clk
        L_Tx_adpt=(1+tx_2x_fifo)/noc_wr_clk
        L_Tx_IO = (1+tx_2x_io)/aib_ns_fwd_clk
    else:
        L_Rx_adpt=(2+rx_4x_fifo)/noc_rd_clk       
        L_Rx_IO = rx_4x_io/aib_fs_fwd_clk
        L_Tx_adpt=(1+tx_4x_fifo)/noc_wr_clk
        L_Tx_IO = (1+tx_4x_io)/aib_ns_fwd_clk
    
    #Adapter power in mW
    aib_adapt_max_txclk = noc_wr_clk
    aib_adapt_max_rxclk = noc_rd_clk
    P_Tx_adpt=γ1*noc_wr_clk+(γ3)*aib_ns_fwd_clk
    P_Rx_adpt=(γ2)*noc_rd_clk+(γ4)*aib_fs_fwd_clk
    P_Tx_adpt+=(γ5_1*n_Tx*8)*aib_adapt_max_txclk+(γ5_2*n_Tx*8)*noc_wr_clk+(γ5_3*n_Tx*2)*aib_ns_fwd_clk
    P_Rx_adpt+=(γ6_1*n_Rx*8)*aib_adapt_max_rxclk+(γ6_2*n_Rx*8)*noc_rd_clk+(γ6_3*n_Rx*2)*aib_fs_fwd_clk

    #IO cell power in mW
    P_Rx_IO=0.1*80*2/1.5*n_Tx*2/160*aib_ns_fwd_clk/2
    P_Tx_IO=0.1*80*2/1.5*n_Rx*2/160*aib_fs_fwd_clk/2

    #Adapter Energy in pJ
    energy= (P_Rx_adpt)*L_Rx_adpt+(P_Tx_adpt)*L_Tx_adpt
    energy *=volt**2*alpha_in*n_ch*1e-3                         #Adapter AIB Energy in pJ
    adapt_energy_eff=energy/(n_ch*max(n_Tx, n_Rx)*8)            #Adapter AIB Energy Efficency in pJ/b        

    #Total AIB Energy in pJ and Latency in ns
    energy+=(P_Rx_IO*L_Rx_IO+P_Tx_IO*L_Tx_IO)*aib_Volt**2/0.4**2*alpha_in*n_ch 
    latency = max(L_Rx_adpt+L_Rx_IO,L_Tx_adpt+L_Tx_IO)

    return latency, energy, adapt_energy_eff

def area_performance_wire(n_Tx, n_Rx):
    #Area encompassing the wire interface in mm2
    A_wire= (p_rw/(2*n_IOcl)*n_IO+W_wire)*Len_wire*1e-3     
    A_wire*=n_ch

    L_wire= 0.69*(R_on*C_in+(R_on+R_wire/2)*Cg_wire+(R_on+R_wire)*C_out)*1e-3 #Latency of the wire interface in ns

    #Power of wire interface in mW
    P_wire= (alpha_aib*aib_ns_fwd_clk+ alpha_aib*aib_fs_fwd_clk+alpha_in*aib_fs_fwd_clk*n_Rx+alpha_in*aib_fs_fwd_clk*n_Tx)*(Cg_wire)*aib_Volt**2 
    P_wire*=n_ch

    return A_wire, L_wire, P_wire*L_wire, P_wire