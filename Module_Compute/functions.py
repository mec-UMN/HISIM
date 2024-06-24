import math 

class imc_analy():
    def __init__(self, xbar_size, volt, freq, freq_adc, compute_ref, quant_bits):
        super(imc_analy, self).__init__()
        self.Qact=quant_bits[1]                             #Bitwidth of activations
        self.Qw=quant_bits[0]                               #Bitwidth of weights
        self.volt=volt                                      #Operating voltage
        self.Qadc=3                                         #Bitwidth of ADC
        self.xbar_y = xbar_size                             #Number of rows inside a crossbar
        self.C_d=1                                          #2: implies storage of signed weights using dual row approach         
        if compute_ref==1:
            #To set the array and adc configurations similar to the reference data - https://ieeexplore.ieee.org/document/8993641/
            self.xbar_x= 64                                 #Number of columns inside a crossbar
            self.xbar_y= 256                                #Number of rows inside a crossbar
            self.volt=0.3 
            self.N_adc =5                                   #Number of ADCs inside a PE
            self.ADC_factor=64/5                            #Number of columns shared across a ADC
            self.scaling_factor_1=1                         #PPA is evaulated at same tech node as reference data
            self.scaling_factor_2=1                         #PPA is evaulated at same tech node as reference data
        else:
            #To set the array and adc configurations as per default HISIM setting
            self.xbar_x =self.xbar_y                        #Number of columns inside a crossbar
            self.ADC_factor=8                               #Number of columns shared across a ADC
            self.N_adc = self.xbar_y/self.ADC_factor        #Number of ADCs inside a PE
            self.scaling_factor_1=496.58/120.53             #PPA is scaled to 28nm tech node as reference data - obtained from https://ieeexplore.ieee.org/document/9484090
            self.scaling_factor_2=3.85/1.14                 #PPA is scaled to 28nm tech node as reference data - obtained from https://ieeexplore.ieee.org/document/9484090
            self.relu=True
        #To set the array parameters similar to the reference data - https://ieeexplore.ieee.org/document/8993641/
        self.A_cell = 0.167*1e-7/self.scaling_factor_1      #Area of a RRAM cell in mm^2 scaled to 28nm
        self.Lmax= 0.24*1e-9                                #Latency of RRAM crossbar configured as per reference data in seconds
        self.Icell=3*1e-6/self.scaling_factor_2              #Current of a RRAM cell in Amperes at 28nm
        
        #To set the adc parameters similar to the reference data - https://ieeexplore.ieee.org/document/8993641/
        self.A_adc=3000*1e-6/self.scaling_factor_1           #Area of single ADC in mm^2 scaled to 28nm
        self.P_adc=2*1e-4/self.scaling_factor_2*freq_adc/0.005 # Power of single ADC scaled wrt operating frequency in Watts at 28nm
        self.L_single_adc=50.48*1e-9*5/64*0.005/freq_adc #Latency of single ADC scaled wrt operating frequency
        
        #import pdb;pdb.set_trace()
        self.A_shiftadd = 2.678*1e-3*0.49                    #Area of shiftadd inside a single PE in mm^2 scaled to 28nm - obtained from https://ieeexplore.ieee.org/document/9458501 
        self.E_shiftadd = 1.73*1e-12*0.49                    #Energy of shiftadd inside a single PE in J at 28nm - obtained from https://ieeexplore.ieee.org/document/9458501
        self.L_shiftadd = 1*1e-9*1/freq                 #Latency of shiftadd inside a single PE in ns 

        self.A_accum =14.794*1e-3*0.49                       #Area of accummulation in mm^2 scaled to 28nm - obtained from https://ieeexplore.ieee.org/document/9458501 
        self.E_accum = 9.77*1e-12*0.49                       #Energy of accummulation in J at 28nm - obtained from https://ieeexplore.ieee.org/document/9458501
        self.L_accum = 1*1e-9*1/freq                    #Latency of accummulation in ns

        self.A_control = 13.892*1e-3*0.49                    #Area of control modules in mm^2 scaled to 28nm - obtained from https://ieeexplore.ieee.org/document/9458501 
        self.E_control = 2.24214e-14                         #Energy of control modules for single operation in J - Calibrated from Neurosim Simulations
        self.L_control = 1e-9/8/freq                    #Latency of control modules for single operation in ns

        self.A_matrix=31.589*1e-3*0.49                       #Area of switchmatrix in mm^2 scaled to 28nm - obtained from https://ieeexplore.ieee.org/document/9458501 
        self.L_matrix=5.00958e-10                            #Latency of control modules for single operation in ns
        self.E_matrix=2.22787e-13                            #Energy of control modules for single operation - Calibrated from Neurosim Simulations

        self.L_bus = 2.55099e-07                             #Latency of bus in ns - Calibrated from Neurosim Simulations
        self.W_bus = 1.7024e-03                              #Width of bus in mm  - Calibrated from Neurosim Simulations
        self.E_bus = 2.58654e-10/freq                   #Energy of bus in J - Calibrated from Neurosim Simulations

        self.A_buffer=2.43737e-03                            #Area of buffer in mm2 - Calibrated from Neurosim Simulations
        self.E_buffer=2.4061e-11/7200/27/12                  #Energy of buffer in J - Calibrated from Neurosim Simulations
        self.L_buffer=4.05e-06/8100/freq                #Latency of buffer in ns - Calibrated from Neurosim Simulations

        #shiftadder
        self.Qadd=math.ceil(math.log2(self.Qadc))+self.Qact+1     #Bitwidth of shiftadder module  
        
        #accummulation 1 for accumulating outputs corresponding to columns where weights were split 
        self.num_stages_accum=math.ceil(math.log2(self.Qw))       #number of stages in the accumulation module 1
        self.Qavg_accum=self.Qadd+math.ceil(self.num_stages_accum/2)  #average bitwidth of accumulation module 1
        self.Qaccum=self.Qadd+math.ceil(math.log2(self.Qw))       #Bitwidth at the output of accumulation module 1
       

    def forward(self, data, layer_idx, network_params):
        self.st = 1                                         #stride of the layer 
        self.C_i=1                                          #Number of Parallel IFM computes for the layer
        self.C_w=1                                          #Weight duplication factor within a crossbar for the layer #
        self.ADC_factor*=self.C_w                           #Multiply weight duplication factor to #

        input_cycle=data[layer_idx][5]                      #Number of input cycles for the layer : (in_x-k_x+1)*(in_y-k_y+1)*quant_act for HISIM generic mapping
        total_bit_real =data[layer_idx][12]                 #Total number of weight bits for the layer: n_channel*k_x*k_y*out_channel*quant_weight for HISIM generic mapping
        n_c_x=data[layer_idx][3]                            #Number of rows of PEs for the layer: math.ceil(in_channel*k_x*k_y/xbar_size) for HISIM generic mapping
        n_c_y=data[layer_idx][4]                            #Number of columns of PEs for the layer: math.ceil(out_channel*quant_weight/xbar_size) for HISIM generic mapping
        util_row=data[layer_idx][11]                        #Average Utilization of a row for the layer: out_channel*quant_weight/(n_c_y*xbar_size)
        util_col=data[layer_idx][13]                        #Average Utilization of a column for the layer: in_channel*k_x*k_y/(n_c_x*xbar_size)
        
        #Array area and latency
        A_arr=self.A_cell*self.xbar_x*self.xbar_y           #Array area for single PE                            
        L_xbar= self.Lmax*self.xbar_y/256                   #Array latency for the crossbar
        L_arr=L_xbar*input_cycle/(self.C_i*self.C_w*self.st**2)

        #ADC latency for the layer
        adc_col_res_factor=(1/self.xbar_y)                  #column resistance and column capacitance is depdendent on xbar size
        factor_lat=1.29 if self.xbar_y==64 else 1           #calibration factor for adc latency for xbar size of 64
        L_adc_xbar=self.L_single_adc*(adc_col_res_factor**2/(1/256**2))*self.ADC_factor*factor_lat         #ADC latency inside a PE - ADC is operated ADC_factor times and its latency varies wrt to xbar size based on both column resitance and column capacitance
        L_adc_t=L_adc_xbar*input_cycle/(self.C_i*self.C_w*self.st**2)

        #Array Energy for the layer
        Emax= self.Icell*self.volt*self.xbar_x*self.xbar_y*(L_adc_xbar+L_xbar)*(self.N_adc/self.xbar_x)     #Maximum Energy of a crossbar under 100% utilization
        E_arr=Emax*(input_cycle/(self.st**2))*self.C_d*(total_bit_real/(self.xbar_x*self.xbar_y*self.st**2))#Array Energy of the layer dependent on input cycles and cell utilization ratio

        #ADC Area and Energy 
        A_adc_t=self.A_adc*self.N_adc                                                                       #Area of ADCs for single PE
        P_adc_t=self.P_adc*(self.N_adc*n_c_x*n_c_y)*self.xbar_y**2/(256**2)                                 #Total Power of ADCs for the layer- dependent on number of ADCs and xbar size. 
        E_adc_t=P_adc_t*(L_adc_xbar+L_xbar)*(input_cycle/(self.C_i*self.C_w*self.st**2))                    #Energy of ADCs for the layer 

        
        ADC_cycles=self.ADC_factor*input_cycle/(self.C_i*self.C_w*self.st**2)                               #Total number of times of exexution of modules before shiftadd
        
        #Shiftadd area, latency and energy
        A_shadd_t=self.A_shiftadd*self.N_adc/16                                                             #Area of shiftadders for single PE
        L_shadd_t=self.L_shiftadd*ADC_cycles/8                                                            #Latency of shiftadders for the layer
        E_shadd_t=self.E_shiftadd*(self.N_adc/16*n_c_x*n_c_y*self.Qadd)*ADC_cycles/64                     #Energy of shiftadders for the layer

        #area, latency and energy of accummulation 1 for accumulating outputs corresponding to columns where weights were split 
        num_adders=self.Qw/2*self.num_stages_accum*self.xbar_y/self.Qw*n_c_y*n_c_x                                    #number of adders of accumulation module 1
        Cycles_accum=ADC_cycles/self.Qact/self.ADC_factor
        factor_lat=3.5 if self.xbar_y==64 else 1                                                            #calibration factor for accumulation latency for xbar size of 64
        
        A_accum_t=self.A_accum*num_adders/12/16                                                             # Area of accumulation module 1
        L_accum_t=self.L_accum*self.num_stages_accum/3*Cycles_accum*util_row*util_col*factor_lat            #Latency of accumulation modules 1 for the layer
        E_accum_t=self.E_accum*self.Qavg_accum/14*Cycles_accum*num_adders/12/16                             #Energy of accumulation modules 1 for the layer

        #area, latency and energy of accumulation 2 to accumulate outputs across PEs
        num_stages=math.ceil(math.log2(n_c_x))                                                              #number of stages in the accumulation module 2                                                              
        Qavg=self.Qaccum+math.ceil(num_stages/2)                                                            #number of stages in the accumulation module 2                                                           
        if n_c_x==1:
            num_adders=0
        else:
            num_adders=n_c_x/2*num_stages*self.xbar_y/self.Qw*n_c_y                                              #Number of adders of accumulation module 2
        A_accum_t+=self.A_accum*num_adders                                                                  # Area of accumulation module
        factor_lat=3.5 if self.xbar_y==64 else 1                                                            #calibration factor for accumulation latency for xbar size of 64
        L_accum_t+=self.L_accum*num_stages/3*Cycles_accum*util_row*factor_lat                               #Latency of accumulation modules for the layer
        E_accum_t+=self.E_accum*Qavg/14*Cycles_accum*num_adders/12/16                                       #Energy of accumulation modules for the layer
        #import pdb;pdb.set_trace()

        A=A_arr+A_accum_t+A_adc_t+A_shadd_t+self.A_buffer                                                   

        #area, latency and energy of switch matrix and control modules such as mux decoder
        A_con=(math.sqrt(self.A_control*self.N_adc/16+self.A_matrix*self.xbar_y/128)/math.sqrt(A)+math.sqrt(A))*math.sqrt(A)  #Fit decoder and switch matrix to dimensions of crossbar and its peripherals
        A_con_t=A_con-A
        A=A_con
        factor_lat=1.5 if self.xbar_y==64 else 1                                                            #calibration factor for switch matrix and contol module latency for xbar size of 64
        L_con_t=max(self.L_matrix*util_col,self.L_control*self.N_adc**2/256*util_row)*ADC_cycles*factor_lat #latency estimated based on maximum latency of switch matrix or control modules 
        E_con_t=self.E_control*ADC_cycles*n_c_x*n_c_y+self.E_matrix*self.xbar_y**2/(128**2)*ADC_cycles*n_c_x*n_c_y  #energy of switch matrix and control modules 

        #latency and energy of buffer
        num_bits = input_cycle/(self.C_i*self.C_w*self.st**2)*(1+self.Qaccum/self.Qact)                          #Total number of bits saved in the buffer
        L_buffer_t=self.L_buffer*input_cycle/(self.C_i*self.C_w*self.st**2)                                 #Latency of the buffers for the layer
        E_buffer_t=self.E_buffer*num_bits*input_cycle/(self.C_i*self.C_w*self.st**2)*n_c_x*n_c_y*self.Qaccum    #Energy of the buffers for the layer
        
        #area, latency and energy of bus
        #Note: bus widths are self.xbar_y/10 for input bus and self.xbar_x/10 for output bus i.e. assume 1 inputbus per 10 rows and 1 outputbus per 10 cols
        bus_len=math.sqrt(A)                                                                                #length of the bus                                                                              
        A_bus_t=self.W_bus*(self.xbar_x+self.xbar_y)*bus_len/10                                             #area of input bus and output bus
        L_bus_t=self.L_bus*bus_len*(util_col/self.xbar_y+self.Qaccum/self.Qact*util_row/self.xbar_x)*input_cycle/(self.C_i*self.C_w*self.st**2)*10 #Latency of the bus for the layer 
        E_bus_t=self.E_bus*bus_len*(n_c_x+n_c_y*self.Qaccum/self.Qact)*input_cycle/(self.C_i*self.C_w*self.st**2)/self.xbar_x                      #Energy of the bus for the layer 
        #import pdb;pdb.set_trace()
        if self.relu==True:
            L_activation=4e-9
        else:
            L_activation=1.64e-8

        if network_params[layer_idx][6]:
            #import pdb;pdb.set_trace()
            L_maxpool=5.812E-09*network_params[layer_idx][0]*network_params[layer_idx][1]/49
        else:
            L_maxpool=0
        
        A+=A_bus_t                                          
        L=L_arr+L_adc_t+L_accum_t+L_con_t+L_shadd_t+L_buffer_t+L_bus_t+L_maxpool+L_activation
        E=E_arr+E_adc_t+E_accum_t+E_con_t+E_shadd_t+E_buffer_t+E_bus_t
        #import pdb;pdb.set_trace()

        peripherials=[L_arr*1e+9,L_adc_t*1e+9,E_arr*1e+12, E_adc_t*1e+12, L_shadd_t*1e+9, L_accum_t*1e+9, E_shadd_t*1e+12, E_accum_t*1e+12, L_con_t*1e+9, E_con_t*1e+12,L_bus_t*1e+9, E_bus_t*1e+12,L_buffer_t*1e+9, E_buffer_t*1e+12]
        A_peripherial=[A_arr, A_adc_t, A_shadd_t, A_accum_t, A_con_t, A_bus_t,self.A_buffer]

        return A, L, E, peripherials, A_peripherial

    def leakage(self,N_crossbar,N_pe):
        
        leak_single_xbar=4e-7*self.xbar_y+3e-7
        leak_addtree=1.22016e-05*self.Qact/8*N_crossbar/4
        leak_buffer=(2.59739e-05+5.28E-06)*self.Qact/8*N_crossbar/4
        leak_PE=(4e-7*self.xbar_y+3e-7)*N_crossbar+1.22016e-05*self.Qact/8*N_crossbar/4+(2.59739e-05+5.28E-06)*self.Qact/8*N_crossbar/4
        leak_accum=1.31e-5*math.sqrt(N_pe)/2*self.xbar_y/64
        leak_buffer_tile=4.63e-5*self.Qact/8*N_pe/4*self.xbar_y/64
        leak_tile=(leak_PE*N_pe+leak_accum+leak_buffer_tile)

        return leak_tile