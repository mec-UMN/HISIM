

import math 
def imc_analy(data, mode, xbar, layer_idx, volt, freq, freq_adc):
    neurosim=2
    Qw=8
    Qact=8
    Qadc=3
    st = 1
    xbar_y = xbar
    C_d=1
    ref_data=0
    if ref_data==1:
        xbar_x= 64
        volt=0.3
        N_adc =5
        ADC_factor=64/5
    else:
        xbar_x =xbar_y
        ADC_factor =8
        N_adc = xbar_y/ADC_factor
    C_i=1
    C_w=1
    ADC_factor*=C_w
    if mode==1:
        input_cycle=data[layer_idx][5]
        #input_cycle = (IFM_x-k_x+1)*(IFM_y-k_y+1)*Qact
        total_bit_real =data[layer_idx][12]
        n_c_x=data[layer_idx][3]
        n_c_y=data[layer_idx][4]
        util_row=data[layer_idx][11]
        util_col=data[layer_idx][13]
        #print(input_cycle, total_bit_real)
        #total_bit_real = k_x*k_y*N_IFM*N_OFM
    elif mode == 2:
        input_cycle = (IFM_x-k_x+1)*(IFM_y-k_y+1)*Qact
        total_bit_real = k_x*k_y*N_IFM
   
    if neurosim==1:
        #40nm neurosim tapeout
        A_cell=0.12*1e-6*0.49#mm2
        Lmax= 0.19*1e-9
        Icell=4.99*1e-12/(Lmax*256*256*0.9*1e-3)*0.49
        A_adc=5400*1e-6/(16)*0.49
        E_adc=85.8*1e-12/128*0.49
        L_adc=5*1e-9*(16/128)
    else:
        if ref_data==0:
            scaling_factor_1=496.58/120.53
            scaling_factor_2=3.85/1.14
        else:
            scaling_factor_1=1
            scaling_factor_2=1
        A_cell = 0.167*1e-7/scaling_factor_1
        Lmax= 0.24*1e-9 #256x64
        Icell=3*1e-6/scaling_factor_2
        A_adc=3000*1e-6/scaling_factor_1
        P_adc=2*1e-4/scaling_factor_2*freq_adc/0.005#W
        L_single_adc=50.48*1e-9*5/64*0.005/freq_adc
        #scaling factor
    #import pdb;pdb.set_trace()
    A_shiftadd = 2.678*1e-3*0.49
    E_shiftadd = 1.73*1e-12*0.49
    L_shiftadd = 1*1e-9*1/freq

    A_accum =14.794*1e-3*0.49
    E_accum = 9.77*1e-12*0.49
    L_accum = 1*1e-9*1/freq

    A_control = 13.892*1e-3*0.49
    E_control = 2.24214e-14
    L_control = 1e-9/8/freq

    A_matrix=31.589*1e-3*0.49
    L_matrix=5.00958e-10
    E_matrix=2.22787e-13

    L_bus = 2.55099e-07
    W_bus = 1.7024e-03
    E_bus = 2.58654e-10/freq

    A_buffer=2.43737e-03
    E_buffer=2.4061e-11/7200/27/12
    L_buffer=4.05e-06/8100/freq
    """
    A_other = 31.38*1e-3/(128)*0.49
    E_other = 15.26*1e-12/128/8*0.49
    L_other = 10*1e-9/8
    """

    A_cell*=xbar_x*xbar_y
    Lmax*= 1*xbar_y/256 
    L_arr= Lmax

    #col res and cap charging factor depdendent on xbar size
    adc_col_res_factor=(1/xbar_y)
    A_adc_t=A_adc*N_adc
    factor_lat=1.29 if xbar_y==64 else 1
    #L=(latency of one adc*col res and cap charging factor)* no of cycles  
    L_adc_t=L_single_adc*(adc_col_res_factor**2/(1/256**2))*ADC_factor*factor_lat #Col resistance dependency on xbar 


    Emax= Icell*volt*(L_adc_t+Lmax)*xbar_x*xbar_y*(N_adc/xbar_x) #crossbar 
    E_arr=Emax*(input_cycle/(st**2))*C_d*(total_bit_real/(xbar_x*xbar_y*st**2))
    A_arr=A_cell

    #P=power of single adc* no of instances*col current factor
    P_adc_t=P_adc*(N_adc*n_c_x*n_c_y)*xbar_y**2/(256**2)
    E_adc_t=P_adc_t*(L_adc_t+Lmax)*(input_cycle/(st**2))
    #import pdb;pdb.set_trace()
    #other -shiftadd
    A=A_arr+A_adc_t
    E=E_arr+E_adc_t
    L_arr*=input_cycle/(C_i*C_w*st**2)
    L_adc_t*=input_cycle/(C_i*C_w*st**2)
    L=L_arr+L_adc_t
     
    #ADC
    ADC_cycles=ADC_factor*input_cycle/(C_i*C_w*st**2)
    Qadd=math.ceil(math.log2(Qadc))+Qact+1
    E_shadd_t=E_shiftadd*(N_adc/16*n_c_x*n_c_y*Qadd)*ADC_cycles/64
    A_shadd_t=A_shiftadd*N_adc/16
    L_shadd_t=L_shiftadd*ADC_cycles/8

    #accum_1
    num_stages=math.ceil(math.log2(Qw))
    Qavg=Qadd+math.ceil(num_stages/2)
    num_adders=Qw/2*num_stages*xbar_y/Qw*n_c_y*n_c_x
    E_accum_t=E_accum*Qavg/14*ADC_cycles/Qact/ADC_factor*num_adders/12/16
    A_accum_t=A_accum*num_adders/12/16
    factor_lat=3.5 if xbar_y==64 else 1
    L_accum_t=L_accum*num_stages/3*ADC_cycles/Qact/ADC_factor*util_row*util_col*factor_lat
    Qadd+=math.ceil(math.log2(Qw))
    #accum_2
    num_stages=math.ceil(math.log2(n_c_x))
    Qavg=Qadd+math.ceil(num_stages/2)
    if n_c_x==1:
        num_adders=0
    else:
        num_adders=n_c_x/2*num_stages*xbar_y/Qw*n_c_y
    E_accum_t+=E_accum*Qavg/14*ADC_cycles/Qact/ADC_factor*num_adders/12/16
    A_accum_t+=A_accum*num_adders
    factor_lat=3.5 if xbar_y==64 else 1
    L_accum_t+=L_accum*num_stages/3*ADC_cycles/Qact/ADC_factor*util_row*factor_lat
    #import pdb;pdb.set_trace()

    A=A_arr+A_accum_t+A_adc_t+A_shadd_t+A_buffer
    #other- Switch Matrix +Mux decoder
    E_con_t=E_control*ADC_cycles*n_c_x*n_c_y+E_matrix*xbar_y**2/(128**2)*ADC_cycles*n_c_x*n_c_y
    A_con=(math.sqrt(A_control*N_adc/16+A_matrix*xbar_y/128)/math.sqrt(A)+math.sqrt(A))*math.sqrt(A)
    factor_lat=1.5 if xbar_y==64 else 1
    L_con_t=max(L_matrix*util_col,L_control*N_adc**2/256*util_row)*ADC_cycles*factor_lat
    #fixed buffer size and bus width-64
    num_bits = input_cycle/(C_i*C_w*st**2)*(1+Qadd/Qact)
    E_buffer_t=E_buffer*num_bits*input_cycle/(C_i*C_w*st**2)*n_c_x*n_c_y*Qadd
    L_buffer_t=L_buffer*input_cycle/(C_i*C_w*st**2)
    A_con_t=A_con-A
    A=A_con
    E+=E_accum_t+E_con_t+E_shadd_t+E_buffer_t
    L+=L_accum_t+L_con_t+L_shadd_t+L_buffer_t
    
    bus_len=math.sqrt(A)
    #bus widths are xbar_y for input bus and xbar_x for output bus
    E_bus_t=E_bus*bus_len*(n_c_x+n_c_y*Qadd/Qact)*input_cycle/(C_i*C_w*st**2)/xbar_x
    A_bus_t=W_bus*(xbar_x+xbar_y)*bus_len/10 #input bus and output bus
    L_bus_t=L_bus*bus_len*(util_col/xbar_y+Qadd/Qact*util_row/xbar_x)*input_cycle/(C_i*C_w*st**2)*10# Assumption: 1 inputbus per row and 1 outputbus per col
    #import pdb;pdb.set_trace()
    
    E+=E_bus_t
    A+=A_bus_t
    L+=L_bus_t

    #print(A, L,E*1e-6)result_list.insert(19,L_adc_t)
    peripherials=[L_arr*1e+9,L_adc_t*1e+9,E_arr*1e+12, E_adc_t*1e+12, L_shadd_t*1e+9, L_accum_t*1e+9, E_shadd_t*1e+12, E_accum_t*1e+12, L_con_t*1e+9, E_con_t*1e+12,L_bus_t*1e+9, E_bus_t*1e+12,L_buffer_t*1e+9, E_buffer_t*1e+12]
    A_peripherial=[A_arr, A_adc_t, A_shadd_t, A_accum_t, A_con_t, A_bus_t,A_buffer]

    #import pdb;pdb.set_trace()
    return A, L, E, peripherials,A_peripherial

#data =[[0,1,1,1,1,394272,0,1,24576,0,0.052734375,0.5, 13824]]
#mode =1
#imc_analy(data,mode,256,0)

def power_density(A,L,E,peripherials, A_peri, data, layer_idx):
    #peripherials=[L_arr*1e+9,L_adc_t*1e+9,E_arr*1e+12, E_adc_t*1e+12, L_shadd_t*1e+9, L_accum_t*1e+9, E_shadd_t*1e+12, E_accum_t*1e+12, L_con_t*1e+9, E_con_t*1e+12,L_bus_t*1e+9, E_bus_t*1e+12,L_buffer_t*1e+9, E_buffer_t*1e+12]
    #A_peripherial=[A_arr, A_adc_t, A_shadd_t, A_accum_t, A_con_t, A_bus_t,A_buffer]
    n_c_x=data[layer_idx][3]
    n_c_y=data[layer_idx][4]
    p_arr=peripherials[2]/(peripherials[1]+peripherials[0])
    p_adc=peripherials[3]/(peripherials[1]+peripherials[0])
    p_shadd=peripherials[6]/(peripherials[4])
    p_accum=peripherials[7]/(peripherials[5])
    p_cont= peripherials[9]/(peripherials[8])
    p_bus=peripherials[11]/(peripherials[10])
    p_buffer=peripherials[13]/(peripherials[12])
    pow = [p_arr, p_adc,p_shadd, p_accum, p_cont, p_bus, p_buffer] #mW
    density_peri=[]
    for i in range(len(pow)):
        density_peri.append(pow[i]/(A_peri[i]*n_c_x*n_c_y)) #density
    #print("pdavg",E/L*1000/A,"pdpk",sum(pow),"pddis",max(density_peri))
    #import pdb;pdb.set_trace()
    return sum(pow), max(density_peri)
