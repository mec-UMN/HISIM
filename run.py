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
        writer.writerow(["freq_core","freq_noc","Xbar_size","N_tile","N_pe","N_tile(real)","N_tier(chiplet)","W2d","W3d","Computing_latency", "Computing_energy","compute_area","chip_area","chip_Architecture","2d NoC latency","3d NoC latency","2.5d NoC latency", "network_latency","2d NoC energy","3d NoC energy","2.5d NoC energy","network_energy","rcc","TFLOPS","compute_power", "2D_3D_NoC_power","2_5D_power","2d_3d_router_area","peak_temperature","placement_method","percent_router"])

    #writer.writerow(["freq_core","freq_noc","Xbar_size","N_tile","N_pe","N_tier(chiplet)","W2d","W3d","Computing_latency", "Computing_energy","compute_area","chip_area","2d NoC latency","3d NoC latency", "network_latency","network_energy","peak_temperature"])

mode=0 #crossbar
#mode=1#DESIGN SPACE
#mode=2#2dn3dlatency
#mode=3#l/tsv
#mode=4#t/l
if mode==0:
    crossbar_size=[1024] 
    #N_tile=[16,25,36,49,64,81,100,121,144]
    N_tile=[100]
    N_pe=[9]
    N_tier=[3]    ##f_core=[0.75,1] # Ghz
    #f_noc=[0.75,1] # Ghz
    f_core=[1]
    f_noc=[1]
    method=[5]
    router_times_scale=[1]
    percent_router=[1]
    tsv_pitch=[5]
    #tsv_pitch=[2,3,4,5,10,20] # um
    #W2d=[i for i in range(1,50,5)]
    W2d=[32]
    chip_arch=["M3D"]
    ai_model=['vit']
elif mode==1:
    crossbar_size=[1024] 
    #N_tile=[16,25,36,49,64,81,100,121,144]
    N_tile=[100]
    N_pe=[36]
    N_tier=[2]    ##f_core=[0.75,1] # Ghz
    #f_noc=[0.75,1] # Ghz
    f_core=[1]
    f_noc=[1]
    method=[1]
    router_times_scale=[1]
    percent_router=[1]
    tsv_pitch=[5]
    #tsv_pitch=[2,3,4,5,10,20] # um
    #W2d=[i for i in range(1,50,5)]
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
                                                        --ai_model %s\
                                                        --no_compute_validate' %(int(i),int(i_tile),int(i_tier),int(pe),float(fcore),float(fnoc),float(placement),float(p_router),float(tsvpitch), str(i_arch), int(i_w2d),int(i_scale), str(i_model)))
                                        

