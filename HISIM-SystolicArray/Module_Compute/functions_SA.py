import math

class SA_calc():

    def __init__(self, SA_size, freq, N_arr, N_pe, N_tile, bit_width):

        # Architecture Parameters
        self.N_arr = N_arr                                      # Number of arrays in one PE
        self.N_pe = N_pe                                        # Number of PEs in a tile
        self.N_tile = N_tile                                    # Number of tiles in a tier

        # Hardware Parameters
        self.SA_size = SA_size
        self.bit_width = bit_width
        self.freq = freq
        self.clk_period = 1/freq

        # Model Parameters
        self.C_i = 1                                            # TODO: Number of parallel IFM computes
        self.C_w = 1                                            # TODO: Weight Duplication Factor

        # Processing Element == MAC unit (From synthesis data)
        self.L_pe = 1.0                                         # Latency of single MAC unit in clock cycles
        self.A_pe = 674.73                                      # Area of MAC unit in um^2
        self.A_pe_toprow = 419.20                               # Area of first (top) row MAC unit in um^2
        self.P_pe = 0.929                                       # Power of MAC unit in mW
        self.P_pe_toprow = 0.0551                               # Power of first (top) row MAC unit in mW
        self.P_pe_cal = 0.0326                                  # Power calibration value in mW
        self.Leak_pe = 0.000060438                              # Leakage Power in mW

        # Bias + ReLu Unit (From synthesis data)
        self.L_bias = 1.0                                       # Latency of single compute of Bias+Relu unit in clock cycles
        self.A_bias = 988.218                                   # Area of Bias+ReLu unit in um^2, calibrated from synthesis values
        self.P_bias = 0.08105                                   # Power of Bias+ReLu unit in mW, calibrated from synthesis values
        self.Leak_bias = 0.000080035                            # Leakage Power in mW

        # Control Logic (From synthesis data)
        self.A_cont = 8548.484 + 234.101*SA_size                # Area of Control circuit in um^2, calibrated from synthesis values
        self.P_cont = 1.614 + 0.04557*SA_size                   # Power of Control circuit in mW, calibrated from synthesis values
        self.Leak_cont = 0.00085633 + 0.000022476*SA_size       # Leakage Power in mW

        # Partial Sum Accumulator
        self.L_accum = 1.0                                      # Latency of Accumulator in clock cycles - obtained from https://ieeexplore.ieee.org/document/9458501
        self.A_accum = (36.766+39.179)*self.bit_width*0.49/8    # Area of accummulation in um^2 scaled to 28nm - obtained from https://ieeexplore.ieee.org/document/9458501 
        self.E_accum = 8.97*1e-12*0.49                          # Energy of accummulation in J at 28nm - obtained from https://ieeexplore.ieee.org/document/9458501

        # Data Bus
        self.L_bus = 2.55099e-07                                # Latency of 1 bit repeater module inside the bus in ns - Calibrated from SIAM Simulations
        self.W_bus = 1.7024                                     # Width of 1 bit inverter module inside the bus in um - Calibrated from SIAM Simulations
        self.E_bus = 2.58654e-10/freq                           # Energy of 1 bit repeater module inside the bus in J - Calibrated from SIAM Simulations
        self.bus_share = 10                                     # Number of MAC units within SA that share a bus line

        # Output Activation Buffer
        self.L_buff = 1.0                                       # 1 additional cycle to write to output buffer
        self.A_buff = 20.665                                    # Approx. area in um^2 for 1 kB of memory, calibrated from synthesis values
        self.P_buff = 0.0014                                    # Approx. power in mW for 1 kB of memory, calibrated from synthesis values
        self.P_buff_cal = 0.18403                               # Approx. power calibration value for memory in mW, from synthesis report

    def forward(self, layer_idx, network_params, computing_data):

        # Input Feature Mask Dimensions
        in_x = network_params[layer_idx][0]
        in_y = network_params[layer_idx][1]
        in_channel = network_params[layer_idx][2]

        # Kernel Dimensions    
        k_x = network_params[layer_idx][3]
        k_y = network_params[layer_idx][4]
        out_channel = network_params[layer_idx][5]              # Number of filters (output channels) of the layer

        # TODO: Include Pooling and Sparsity
        enable_pooling = network_params[layer_idx][6]           # Parameter indicating if the layer is followed by pooling or not
        sparsity = 1 - network_params[layer_idx][7]             # Total Sparsity of the layer

        ### Calculations
        inp_cycles = computing_data[layer_idx][5]/(self.C_i*self.C_w)   # Number of Input Cycles for the layer (Unrolled Conv. Windows)

        total_compute_cycles = inp_cycles + 2*self.SA_size - 1  # Total compute cycles needed by SA to generate all outputs

        n_c_x = computing_data[layer_idx][3]                    # Number of SAs in x direction for the layer
        n_c_y = computing_data[layer_idx][4]                    # Number of SAs in y direction for the layer

        tile_cols = math.sqrt(self.N_pe)*math.sqrt(self.N_arr)*self.SA_size  # Maximum MAC unit columns in a tile
        num_cols = out_channel                                  # Total columns of MAC units needed in this layer for weight stationary mapping

        tiles_needed = computing_data[layer_idx][1]             # Number of tiles required for this layer

        util_row = computing_data[layer_idx][11]                # Average Utilization of a row for the layer
        util_col = computing_data[layer_idx][13]                # Average Utilization of a column for the layer
        

        # Systolic Array MAC unit calculations
        L_arr_t = total_compute_cycles*self.clk_period          # Latency of the SA for the layer in ns

        A_arr = self.A_pe * (self.SA_size**2 - self.SA_size)    # Area of all but first (top) row of MAC units in um^2
        A_arr += self.A_pe_toprow * self.SA_size                # Area of first (top) row of MAC units in um^2
        A_arr_pe = A_arr * self.N_arr                           # Total Array Area in um^2 for one PE
        
        P_arr = self.P_pe * (self.SA_size**2 - self.SA_size)    # Power of all but first (top) row of MAC units in mW
        P_arr += self.P_pe_toprow*self.SA_size + self.P_pe_cal  # Power of first (top) row of MAC units in mW
        P_arr_t = P_arr * n_c_x * n_c_y                         # Total Power of Systolic Arrays in mW
        E_arr_t = P_arr_t/1e3*inp_cycles*self.clk_period/1e9    # Total Energy in J

        # Systolic Array Control Logic calculations
        A_cont_pe = self.A_cont * self.N_arr                    # Area of control logic for one PE
        P_cont_t = self.P_cont * n_c_x * n_c_y                  # Total power of control logic in mW
        E_cont_t = P_cont_t/1e3*total_compute_cycles*self.clk_period/1e9   # Total Energy of control logic in J

        # Systolic Array Bias+ReLu unit calculations
        L_bias_t = self.L_bias*self.clk_period                  # Bias+ReLu unit takes 1 additional cycle
        A_bias_pe = self.A_bias*self.SA_size*self.N_arr         # Area of Bias+ReLu units in um^2 for one PE
        P_bias_t = self.P_bias*self.SA_size*n_c_x*n_c_y         # Total power of Bias+Relu units for the layer in mW
        E_bias_t = P_bias_t/1e3*inp_cycles*self.clk_period/1e9  # Total Energy in J

        # Partial Sum Accumulator calculations
        num_stages_accum = math.ceil(math.log2(n_c_x))          # Number of stages in the Accumulation Module (Binary Adder Tree structure)                                                              

        # Total number of adders in Accumulation Module
        if n_c_x == 1:
            num_adders = 0
            num_adders_tile = 0
        else:
            num_adders = (n_c_x - 1) * num_cols                         # Total number of adders needed for this layer (out_channel = total num_cols of MAC units)
            num_adders_tile = (n_c_x - 1) * min(num_cols, tile_cols)    # Total number of adders within one tile for this layer

        L_accum_t = self.L_accum * num_stages_accum * self.clk_period       # Latency of Accumulation Module (additional cycles = length of critical path of tree)
        A_accum_t = self.A_accum * num_adders_tile                          # Area of accumulation module in um^2 for one tile
        E_accum_t = self.E_accum*self.bit_width*num_adders*inp_cycles*self.clk_period/1e9   # Energy in J of Accumulation module

        # Output Buffer
        buff_size = self.get_outbuff_size(inp_cycles, out_channel)      # Output buffer size in kB
        buff_size_per_col = buff_size/num_cols                          # Buffer size needed per column of MAC units
        buff_size_tile = buff_size_per_col * min(num_cols, tile_cols)   # Largest buffer size for a tile in this layer (Buffer is split across tiles for the layer)
        
        # Buffer takes one additional clock cycle to write outputs, ret of the latency is masked by latency of the array
        L_buff = self.L_buff*self.clk_period                            # Buffer latency in ns
        A_buff_t = self.A_buff*buff_size_tile                           # Area of the buffer for this tile in um^2
        E_buff_t = (self.P_buff*buff_size+self.P_buff_cal)/1e3*inp_cycles*self.clk_period/1e9 # Buffer Power in J

        # Data Bus Calculations (N rows of MAC units share one data bus line, where N = bus_share)
        curr_area = (A_arr_pe + A_bias_pe + A_cont_pe)*self.N_pe + A_accum_t + A_buff_t     # Current tile area
        bus_len = math.sqrt(curr_area)                          # Length of the bus (assuming it runs edge to edge within a tile)

        A_bus_t = self.W_bus*(self.SA_size*2)*bus_len*self.bit_width/self.bus_share    # Total Bus Area 
        L_bus_t = self.L_bus*bus_len*(util_col/self.SA_size+util_row/self.SA_size)*self.bus_share # TODO: Latency of the bus for the layer
        E_bus_t = self.E_bus*bus_len*(n_c_x+n_c_y)*inp_cycles/self.SA_size*self.bit_width*self.clk_period/1e9

        L = L_arr_t + L_bias_t + L_accum_t + L_buff                             # Total Layer Latency in ns
        A = (curr_area + A_bus_t)*1e-6                                          # Total Tile Area in mm^2
        E = E_arr_t + E_bias_t + E_cont_t + E_accum_t + E_bus_t + E_buff_t # Total Layer Energy in J

        return A, L*1e-9, E


    # Output buffer size required to store all activations from this layer for a single image
    def get_outbuff_size(self, inp_cycles, out_channel):

        buff_size = inp_cycles * out_channel * self.bit_width / (1024 * 8)  # Output Buffer size in kB

        return buff_size