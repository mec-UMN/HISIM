import math
import numpy as np
from Module_Network.orion_power_area import power_summary_router
from Module_Network.aib_2_5d import aib

def network_model(N_tier_real,N_tile,N_tier,computing_data,placement_method,percent_router,chip_architect,tsvPitch,
                  area_single_tile,result_list,volt,fclk_noc,total_model_L):
    # Network,3D NoC
    # area,latency,power

    #---------------------------------------------------------------------#
    chiplet_num=N_tier_real
    mesh_edge=int(math.sqrt(N_tile))

    total_tile=0
    layer_start_tile=0
    layer_start_tile_tier=[0]*N_tier
    tile_total=[]
    # for decide (x,y)
    for layer_index in range(len(computing_data)):
        if placement_method ==5:
            if layer_index<N_tier:
                layer_start_tile_tier[int(computing_data[layer_index][9])]=0
            layer_start_tile=layer_start_tile_tier[int(computing_data[layer_index][9])]
        else:
            if computing_data[layer_index][9]>1 and computing_data[layer_index-1][9]!=computing_data[layer_index][9]:
                layer_start_tile=0
        # get this layer information 

        layer_end_tile=layer_start_tile+int(computing_data[layer_index][1])-1

        tile_index = np.array([[0,0,0]])
        #import pdb;pdb.set_trace()
        for layer_tile_number in range(layer_start_tile,layer_end_tile+1):
            
            x_idx= int((layer_tile_number)//(mesh_edge))
            #print("x_idx",x_idx)
            y_idx= int((layer_tile_number)%(mesh_edge))
            #print("y_idx",y_idx)

            tile_index = np.append(tile_index, [[x_idx, y_idx, computing_data[layer_index][9]]],axis=0)
        
        tile_index=tile_index[1:]

        each_tile_activation_Q=int(computing_data[layer_index][8]/computing_data[layer_index][1])
        
        tile_index= np.append(tile_index,[[each_tile_activation_Q,each_tile_activation_Q,each_tile_activation_Q]],axis=0)
        if placement_method==5:
            tile_total.append(tile_index)
            layer_start_tile_tier[int(computing_data[layer_index][9])]=layer_end_tile+1
            #import pdb;pdb.set_trace()
        else:    
            tile_total.append(tile_index)
            layer_start_tile=layer_end_tile+1

    empty_tile_total=[]
    if placement_method==5:
        for tier_index in range(chiplet_num):
            tile_index = np.array([[0,0,0]])
            for x in range(mesh_edge):
                for y in range(mesh_edge):
                    tile_index = np.append(tile_index, [[x, y, tier_index]],axis=0)
            tile_index=tile_index[1:]
            empty_tile_total.append(tile_index)

    else:
        for tier_index in range(chiplet_num):
            tile_index = np.array([[0,0,0]])
            for x in range(int(math.sqrt(N_tile))):
                for y in range(int(math.sqrt(N_tile))):
                    tile_index = np.append(tile_index, [[x, y, tier_index]],axis=0)
            tile_index=tile_index[1:]
            empty_tile_total.append(tile_index)



    hop2d=0
    hop3d=0
    Q_3d=0
    Q_2d=0
    Q_2_5d=0
    layer_Q=[]
    layer_Q_2_5d=[]
    layer_HOP_2d=[]
    layer_HOP_3d=[]
    # counting total 2d hop and 3d hop
    routing_method=2  #: local routing-> only use the routers and tsvs nearby
    #routing method 2: global routing-> in the global routing, data will try to use all the routers to transport to next tier
    for i in range(len(tile_total)-1):
        #print(tile_total[i])
        #print(tile_total[i+1])
        num_tiles_this_layer=len(tile_total[i]-1)
        num_tiles_left_this_layer=N_tile-num_tiles_this_layer
        Q_3d_scatter=tile_total[i][-1][2]*num_tiles_this_layer/N_tile
        
        layer_2d_hop=hop2d
        layer_3d_hop=hop3d
        
        if routing_method==1:
            for x in range(len(tile_total[i])-1):
                for y in range(len(tile_total[i+1])-1):
                    hop2d+=abs(tile_total[i][x][0]-tile_total[i+1][y][0])+abs(tile_total[i][x][1]-tile_total[i+1][y][1])+1
                    hop3d+=abs(tile_total[i][x][2]-tile_total[i+1][y][2])
            layer_HOP_2d.append(hop2d-layer_2d_hop)
            layer_HOP_3d.append(hop3d-layer_3d_hop)
            if tile_total[i][0][2]!=tile_total[i+1][0][2]:
                Q_3d+=(tile_total[i][-1][2])*(len(tile_total[i+1])-1)
                layer_Q.append((tile_total[i][-1][2])*(len(tile_total[i])-1))
                #Q_2d+=tile_total[i][-1][2]*(len(tile_total[i+1])-1)
            else:
                Q_2d+=tile_total[i][-1][2]*(len(tile_total[i+1])-1)
                layer_Q.append(tile_total[i][-1][2]*(len(tile_total[i])-1))
        
        elif routing_method==2:
            for x in range(len(tile_total[i])-1):
                for y in range(len(tile_total[i+1])-1):
                    hop2d+=abs(tile_total[i][x][0]-tile_total[i+1][y][0])+abs(tile_total[i][x][1]-tile_total[i+1][y][1])+1
                    hop3d+=(abs(tile_total[i][x][2]-tile_total[i+1][y][2]))*N_tile*percent_router
            
            layer_HOP_2d.append(hop2d-layer_2d_hop)
            layer_HOP_3d.append(hop3d-layer_3d_hop)

            if tile_total[i][0][2]!=tile_total[i+1][0][2]:
                for x in range(len(tile_total[i])-1):
                    #import pdb;pdb.set_trace()
                    for y in range(int(len(empty_tile_total[int(tile_total[i][x][2])])*percent_router)):
                        hop2d+=(abs(tile_total[i][x][0]-empty_tile_total[int(tile_total[i][x][2])][y][0])+abs(tile_total[i][x][1]-empty_tile_total[int(tile_total[i][x][2])][y][1])+1)*2
                Q_3d+=int((tile_total[i][-1][2])*(len(tile_total[i+1])-1)/(N_tile*percent_router))
                #import pdb;pdb.set_trace()
                layer_Q.append((tile_total[i][-1][2])*(len(tile_total[i])-1))
                Q_2d+=int((tile_total[i][-1][2]*(len(tile_total[i])-1))/(N_tile*percent_router))
                #import pdb;pdb.set_trace()
            else:
                Q_2d+=tile_total[i][-1][2]*(len(tile_total[i+1])-1)
                layer_Q.append(tile_total[i][-1][2]*(len(tile_total[i])-1))

        if tile_total[i][0][2]!=tile_total[i+1][0][2]:
            Q_2_5d+=(tile_total[i][-1][2])*(len(tile_total[i+1])-1)
            layer_Q_2_5d.append((tile_total[i][-1][2])*(len(tile_total[i+1])-1))
    
    print("\n")
    print("----------network performance results--------------------")
    print("----------network data information-----------------------")
    print("Total Q bits for 2d communication:", Q_2d)
    print("Total HOP for 2d communication:", hop2d)
    if chip_architect=="M3D":
        print("Total Q bits for 3d communication:", Q_3d)
        print("Total HOP for 3d communication:", hop3d)
    elif chip_architect=='H2_5D':
        print("Total Q bits for 2.5d communication:", Q_2_5d)

    #------------bandwidth-------------------#
    # 2D noc
    # fix as 32
    # 3D tsv
    W2d=32 # this is the bandwidth of 2d
    W3d_assume=32
    
    trc=1
    tva=1
    tsa=1
    tst=1
    tl=1
    tenq=2

    if chip_architect=="M3D" and N_tier_real!=1:
        channel_width=4/5*W2d+1/5*W3d_assume # mix 2d and 3d
        total_router_area,_,_=power_summary_router(channel_width,6,6,hop3d,trc,tva,tsa,tst,tl,tenq,Q_3d,int(chiplet_num),int(mesh_edge))
    elif chip_architect=="M2D" or chip_architect=="H2_5D" or N_tier_real==1:
        channel_width=W2d # mix 2d and 3d
        total_router_area,_,_=power_summary_router(channel_width,5,5,hop3d,trc,tva,tsa,tst,tl,tenq,Q_3d,int(chiplet_num),int(mesh_edge))  
    single_router_area=total_router_area/(mesh_edge*mesh_edge*chiplet_num)
    edge_single_router=math.sqrt(single_router_area)
    edge_single_tile=math.sqrt(area_single_tile)

    num_tsv_io=int(edge_single_router/tsvPitch*1000)*int(edge_single_router/tsvPitch*1000)*2
    W3d=num_tsv_io
    if chip_architect=="M3D" and N_tier_real!=1:
        channel_width=4/5*W2d+1/5*W3d # mix 2d and 3d
        total_router_area,_,_=power_summary_router(channel_width,6,6,hop3d,trc,tva,tsa,tst,tl,tenq,Q_3d,int(chiplet_num),int(mesh_edge))
    single_router_area=total_router_area/(mesh_edge*mesh_edge*chiplet_num)
    edge_single_router=math.sqrt(single_router_area)
    layer_aib_list=[]
    if chip_architect=="H2_5D" and N_tier_real!=1:
        aib_out=[0,0,0]
        for i in range(len(layer_Q_2_5d)):
            layer_aib=aib(layer_Q_2_5d[i]*1e-6/8, (edge_single_router+edge_single_tile)*mesh_edge, 1, volt)
            layer_aib_list.append(layer_aib)
            #area- layer_aib_list[idx][0] -mm2, energy- layer_aib_list[idx][1] -pJ, latency-layer_aib_list[idx][2]-ns
            for i in range(len(aib_out)):
                aib_out[i] += layer_aib[i]
        area_2_5d=aib_out[0]
    else:
        area_2_5d=0

    print("--------------network area report------------------------")
    print("single tile area",round(area_single_tile,5),"mm2")
    print("single router area",round(single_router_area,5),"mm2")
    print("edge length single router",round(edge_single_router,5),"mm") #mm
    print("edge length single tile",round(edge_single_tile,5),"mm") #mm
    print("total 3d stack area",round((edge_single_router+edge_single_tile)*(edge_single_router+edge_single_tile)*N_tile,5),"mm2")
    print("2.5d area", round(area_2_5d,5))
    print("---------------------------------------------------------")
    result_list.append((edge_single_router+edge_single_tile)*(edge_single_router+edge_single_tile)*N_tile+area_2_5d)
    result_list.insert(7,W2d)
    result_list.insert(8,W3d)
    # Router technology delay 

    working_channel=2 # last layer source to dst paths
    #3d noc edges links
    links_topology_2d=(math.sqrt(N_tile)-1)*math.sqrt(N_tile)*2*chiplet_num
    links_topology_3d=N_tile*(chiplet_num-1)

    # 2.1 latency of booksim
    if chip_architect=="M3D" or chip_architect=="M2D" or N_tier_real==1:
        L_booksim=(hop2d*(trc+tva+tsa+tst+tl)+(tenq)*(Q_2d/W2d)+hop3d*(trc+tva+tsa+tst+tl)+(tenq)*(Q_3d/W3d))/fclk_noc
        L_2_5d=0
        L_booksim_2d=(hop2d*(trc+tva+tsa+tst+tl)+(tenq)*(Q_2d/W2d))/fclk_noc
        L_booksim_3d=(hop3d*(trc+tva+tsa+tst+tl)+(tenq)*(Q_3d/W3d))/fclk_noc
    
    elif chip_architect=="H2_5D" and  N_tier_real!=1:
        L_booksim_2d=(hop2d*(trc+tva+tsa+tst+tl)+(tenq)*(Q_2d/W2d))/fclk_noc
        L_2_5d=aib_out[2]
        L_booksim_3d=0
        L_booksim=L_booksim_2d+L_2_5d
    result_list.append(chip_architect)
    result_list.append(L_booksim_2d)
    result_list.append(L_booksim_3d)
    result_list.append(L_2_5d)
    # 2.2 power of booksim


    # ADD VOLTAGE
    layer_index=0
    tier_2d_hop_list=[]
    tier_3d_hop_list=[]
    tier_total_2d_hop=0
    tier_total_3d_hop=0
    num_layer=0
    num_layer_list=[]
    for i in range(chiplet_num):
        for layer_index in range(len(tile_total)-1):
            if computing_data[layer_index][9]==i :
                tier_total_2d_hop+=layer_HOP_2d[layer_index]
                tier_total_3d_hop+=layer_HOP_3d[layer_index]
            num_layer+=1
        tier_2d_hop_list.append(tier_total_2d_hop/(num_layer)/N_tile)
        tier_3d_hop_list.append(tier_total_3d_hop)
        tier_total_2d_hop=0
        tier_total_3d_hop=0
        num_layer=0

    chiplet_num=N_tier_real

    #mesh_edge=int(math.sqrt(N_tile))
    if chip_architect=="M3D" and N_tier_real!=1:
        _,total_tsv_channel_power,total_3d_router_power=power_summary_router(W3d,6,6,hop3d,trc,tva,tsa,tst,tl,tenq,Q_3d,int(chiplet_num),int(mesh_edge))
        _,total_2d_channel_power,total_2d_router_power=power_summary_router(W2d,5,5,hop2d,trc,tva,tsa,tst,tl,tenq,Q_2d,int(chiplet_num),int(mesh_edge))
    elif chip_architect=="M2D" or chip_architect=="H2_5D" or N_tier_real==1:
        _,total_tsv_channel_power,total_3d_router_power=0,0,0
        _,total_2d_channel_power,total_2d_router_power=power_summary_router(W2d,5,5,hop2d,trc,tva,tsa,tst,tl,tenq,Q_2d,int(chiplet_num),int(mesh_edge))

    if chip_architect=="H2_5D" and N_tier_real!=1:
        total_2_5d_channel_power=aib_out[1]/aib_out[2]
    else:
        total_2_5d_channel_power=0
    # total area of router channel= single_channel_area*(channel number*2+2*number_router)
    # total switch+input+output=(switch+input+output)*number_router
    total_router_power=total_3d_router_power+total_2d_router_power+total_2d_channel_power

    #print("2d",total_router_power)
    #print("tsv",total_tsv_channel_power)
    energy_2d=(total_2d_channel_power+total_2d_router_power)*L_booksim_2d*fclk_noc

    energy_3d=(total_tsv_channel_power+total_3d_router_power)*L_booksim_3d*fclk_noc

    energy_2_5d=total_2_5d_channel_power*L_2_5d*fclk_noc 

    total_energy=energy_2d+energy_2_5d+energy_3d
    if len(tier_2d_hop_list)!=1:
        tier_2d_hop_list[-1]=tier_2d_hop_list[-2]
        tier_3d_hop_list[-1]=tier_3d_hop_list[-2]
    else:
        tier_3d_hop_list[-1]=total_router_power/chiplet_num
    tier_2d_hop_list_power=[i * total_router_power/chiplet_num/i*fclk_noc for i in tier_2d_hop_list]
    if chip_architect=="M3D" and N_tier_real!=1:
        tier_3d_hop_list_power=[i * total_tsv_channel_power/(chiplet_num-1)/i*fclk_noc for i in tier_3d_hop_list]
    elif chip_architect=="M2D" or chip_architect=="H2_5D" or N_tier_real==1:
        tier_3d_hop_list_power=[i * 0 for i in tier_3d_hop_list]

    #print("each tier average 2d NoC router power",tier_2d_hop_list_power,"mW")
    #print("each tier average 3D NoC router power",tier_3d_hop_list_power,"mW")
    print("2D NoC W2d",W2d)
    print("3D TSV W3d",W3d)
    print("network total energy",round(total_energy,5),"pJ")
    print("network power",round((total_router_power+total_tsv_channel_power)*fclk_noc,5),"mW")
    print("total NoC latency", round(L_booksim,5),"ns")

    result_list.append(L_booksim)
    result_list.append(energy_2d)
    result_list.append(energy_3d)
    result_list.append(energy_2_5d) # Pj
    result_list.append(total_energy)

    # 2.3 area of booksim
    wire_length_2d=2 #unit=mm\
    wire_pitch_2d=0.0045 #unit=mm
    Num_routers=N_tile*chiplet_num

    Total_area_routers=(single_router_area)*Num_routers
    Total_channel_area=wire_length_2d*wire_pitch_2d*W2d
    #single_TSV_area=math.sqrt(area_single_tile)*math.sqrt((1e-6*(4/5*W2d+1/5*W3d)*(4/5*W2d+1/5*W3d)+5e-5*(4/5*W2d+1/5*W3d)+0.0005))
    print("computing latency",round(total_model_L*pow(10,9),5),"ns")
    print("total system latency", round(L_booksim+total_model_L*pow(10,9),5),"ns")
    print("----------network performance done--------------------")
    result_list.append(total_model_L*pow(10,9)/L_booksim)

    flops=0
    for j in range(len(computing_data)):
        flops+=computing_data[j][14]
    result_list.append(flops*pow(10,-3)/(L_booksim+total_model_L*pow(10,9)))
    result_list.append((total_router_power+total_tsv_channel_power)*fclk_noc*pow(10,-3))
    if area_2_5d!=0:
        result_list.append(total_2_5d_channel_power*pow(10,-3))
    else:
        result_list.append(0)
    result_list.append(Total_area_routers+Total_channel_area)
    
    return chiplet_num,tier_2d_hop_list_power,tier_3d_hop_list_power,single_router_area,mesh_edge,layer_aib_list,result_list