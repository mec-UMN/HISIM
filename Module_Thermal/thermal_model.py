import torch
import math
import collections
from Module_Thermal.util import *
from Module_Thermal.H2_5D_thermal import *


def thermal_model(temp,chip_architect,chiplet_num,N_tile,placement_method,tier_2d_hop_list_power,tier_3d_hop_list_power,area_single_tile,single_router_area
                  ,mesh_edge,sim_name,layer_aib_list):
    #---------------------------------------------------------------------#

    #                 Thermal simulation (based on power,area)

    #---------------------------------------------------------------------#

    if temp and chip_architect!="H2_5D":

        # np.set_printoptions(threshold=sys.maxsize)
        torch.set_printoptions(threshold=50_000)
        # cu =   398 w/mk
        # air = 24.3 mw/mk
        # sink 4mm,2mm, 6mm 2 times chip  3times chip
        #
        tiers_of_stacks=chiplet_num
        tiles_edges_in_tier=int(math.sqrt(N_tile))
        #====================================================================================================
        # mw ,later in load_power will be converted to w
        #====================================================================================================
        case_=4
        alpha=3.5
        #1.2
        #for i in range(chiplet_num):
        ##
        power_router={}
        power_tsv={}
        for i in range(len(tier_2d_hop_list_power)):
            if placement_method==1 or placement_method==3 or placement_method==4 or placement_method==5:
                power_router[i] = tier_2d_hop_list_power[i]
            elif placement_method==2:
                power_router[len(tier_2d_hop_list_power)-i-1]=tier_2d_hop_list_power[i]
        for i in range(len(tier_3d_hop_list_power)):
            if placement_method==1 or placement_method==3 or placement_method==4 or placement_method==5:
                power_tsv[i] = tier_3d_hop_list_power[i]
            elif placement_method==2:
                power_tsv[len(tier_3d_hop_list_power)-i-1] = tier_3d_hop_list_power[i]

        #import pdb;pdb.set_trace()

        #====================================================================================================
        # w/mk
        #====================================================================================================
        if chip_architect=="M3D":
            dict_k = dict()
            dict_k['k_imc_0']  = 110/alpha
            dict_k['k_imc_1']= 142.8/alpha
            dict_k['k_imc_2']    = 4/alpha
            dict_k['k_r_0']    = 110/alpha
            dict_k['k_r_1']  = 142.8/alpha
            dict_k['k_r_2']      = 4/alpha
            dict_k['k_tsv_0']= 142.8/alpha
            dict_k['k_tsv_1']  = 200/alpha
            dict_k['k_tsv_2']  = 7.9/alpha
            dict_k['cu']       = 398/alpha
            dict_k['air']   = 0.0243/alpha
            dict_k['subs']   = 142.8/alpha
        elif chip_architect=="M2D":
            dict_k = dict()
            dict_k['k_imc_0']  = 110/alpha
            dict_k['k_imc_1']= 142.8/alpha
            dict_k['k_imc_2']    = 4/alpha
            dict_k['k_r_0']    = 110/alpha
            dict_k['k_r_1']  = 142.8/alpha
            dict_k['k_r_2']      = 4/alpha
            dict_k['k_tsv_0']= 110/alpha
            dict_k['k_tsv_1']  = 142.8/alpha
            dict_k['k_tsv_2']  = 4/alpha
            dict_k['cu']       = 398/alpha
            dict_k['air']   = 0.0243/alpha
            dict_k['subs']   = 142.8/alpha

        #====================================================================================================
        # m
        #====================================================================================================
        imc_size = math.sqrt(area_single_tile)/1000
        r_size   = math.sqrt(single_router_area)/1000


        #====================================================================================================
        # imc_size = 0.00085
        # r_size = 0.00085
        #====================================================================================================
        #====================================================================================================

        tsv0_length = r_size
        tsv0_width  = imc_size
        tsv1_length = imc_size
        tsv1_width  = r_size


        dict_z=dict()
        #====================================================================================================
        # set to mm
        #====================================================================================================
        # it will be later converted to m
        #====================================================================================================
        dict_z['heatsink']=40
        dict_z['heatspread']=20
        dict_z['device']=(0.002, 0.1, 0.02)
        # die thickness (0.1-0.5)
        # tim thickness (0.01-0.05)
        dict_z['subs']=1
        dict_z['air']=50
        heatsinkair_resoluation=0.5
        #====================================================================================================
        dict_size = dict()
        dict_size["imc"]  = (imc_size,imc_size)
        dict_size["r"]    = (r_size,r_size)
        dict_size["tsv0"] = (tsv0_length,tsv0_width)
        dict_size["tsv1"] = (tsv1_length,tsv1_width)

        #====================================================================================================
        devicemap = collections.defaultdict(list)

        #====================================================================================================

        #====================================================================================================
        #devicemap['3stacks'].append([(0, 5, 'heatsink')])
        #devicemap['3stacks'].append([(0, 5, 'heatspread')])
        #devicemap['3stacks'].append([(0, 1,'device'), (1,2,'air'), (2, 3, 'device' ),(3,4,'air'), (4, 5, 'device' )])
        ###devicemap['3stacks'].append([(0, 1,'device'), (1,2,'air'), (2, 3, 'device' ),(3,4,'air'), (4, 5, 'device' )])
        #devicemap['3stacks'].append([(0, 5, 'subs')])
        #devicemap['3stacks'].append([(0, 5, 'air')])
        #====================================================================================================
        #devicemap['2stacks'].append([(0, 3, 'heatsink')])
        #devicemap['2stacks'].append([(0, 3, 'heatspread')])
        #devicemap['2stacks'].append([(0, 1,'device'), (1,2,'air'), (2, 3, 'device' )])
        #devicemap['2stacks'].append([(0, 1,'device'), (1,2,'air'), (2, 3, 'device' )])
        ##devicemap['2stacks'].append([(0, 1,'device'), (1,2,'air'), (2, 3, 'device' )])
        ##devicemap['2stacks'].append([(0, 3, 'subs')])
        #devicemap['2stacks'].append([(0, 3, 'air')])
        ###====================================================================================================
        devicemap['1stacks'].append([(0, 1, 'heatsink')])
        devicemap['1stacks'].append([(0, 1, 'heatspread')])
        for i in range(chiplet_num):
            devicemap['1stacks'].append([(0, 1,'device')])
        devicemap['1stacks'].append([(0, 1, 'subs')])
        devicemap['1stacks'].append([(0, 1, 'air')])
        ###====================================================================================================
        #devicemap['0stacks'].append([(0, 6, 'heatsink')])
        #devicemap['0stacks'].append([(0, 6, 'heatspread' )])
        #devicemap['0stacks'].append([(0, 1, 'device' ),(1, 2, 'device' ),(2, 3, 'device' ),(3, 4, 'device' ),(4, 5, 'device' ),(5, 6, 'device' ),  ])
        #devicemap['0stacks'].append([(0, 6, 'subs' )])
        #devicemap['0stacks'].append([(0, 6, 'air' )])
        #====================================================================================================
        numofdevicelayer = dict()
        #numofdevicelayer['0stacks']= 1
        numofdevicelayer['1stacks']= chiplet_num
        #numofdevicelayer['2stacks']= 3
        ##numofdevicelayer['3stacks']= 2
        #====================================================================================================
        devicemap_sanitycheck(devicemap)
        xdim,_                                                                   = get_unitsize(dict_size,mesh_edge)
        cube_geo_dict, cube_k_dict, cube_z_dict, cube_n_dict,cube_layertype_dict = create_cube(dict_size, dict_z, dict_k,  xdim , devicemap,heatsinkair_resoluation,mesh_edge)
        cube_power_dict                                                          = load_power(case_,dict_z, devicemap, cube_n_dict, power_tsv, power_router,numofdevicelayer,cube_layertype_dict,mesh_edge,chiplet_num,placement_method,chip_architect)
        cube_G_dict                                                              = get_conductance_G_new(cube_geo_dict, cube_k_dict, cube_z_dict)
        peak_temp                                                              = solver(cube_G_dict, cube_n_dict,cube_power_dict,cube_layertype_dict,xdim,sim_name )

        #result_list.append(peak_temp)
        #end_thermal = time.time()

    elif temp and chip_architect=="H2_5D":

        power_aib_l=[]
        power_emib_l=[]
        area_aib_l=[]
        area_emib_l=[]
        #power_Tx, power_Rx, power_wire
        for i in range(len(layer_aib_list)):
            power_aib_l+=[layer_aib_list[i][6]/layer_aib_list[i][9]+layer_aib_list[i][8], layer_aib_list[i][7]/layer_aib_list[i][10]]
            #power_emib_l+=[layer_aib_list[i][8]]
            area_aib_l+=[layer_aib_list[i][3], layer_aib_list[i][4]]
            area_emib_l+=[layer_aib_list[i][5]]
        
        # average area of single aib
        if chiplet_num==1:
            area_aib=0
            area_emib=0
        else:
            area_aib=sum(area_aib_l)/len(area_aib_l)
            area_emib=sum(area_emib_l)/len(area_emib_l)

        
        #import pdb;pdb.set_trace()
        case_H2_5D=H2_5D(area_single_tile=area_single_tile,single_router_area=single_router_area,chiplet_num=chiplet_num,mesh_edge=mesh_edge,area_aib=area_aib,area_emib=area_emib,resolution=1)
        
        power_tier_l=power_tile_reorg(mesh_edge)
        
        #import pdb;pdb.set_trace()
        #power_tier_l   = [20]*case_H2_5D.Nstructure*case_H2_5D.N*case_H2_5D.N
        power_router_l = tier_2d_hop_list_power#[5]*case_H2_5D.Nstructure
        if case_H2_5D.Nstructure   == 2: numofaib = 2;  numofemib = 1
        elif case_H2_5D.Nstructure == 3: numofaib = 4;  numofemib = 2
        elif case_H2_5D.Nstructure == 4: numofaib = 6;  numofemib = 3 #;power_aib_l+=[0,0]
        elif case_H2_5D.Nstructure == 1: numofaib = 0;  numofemib = 0

        
        #area- layer_aib_list[idx][0] -mm2, energy- layer_aib_list[idx][1] -pJ, latency-layer_aib_list[idx][2]-ns
        #power_aib_l    = [5]*numofaib
        power_emib_l   = [0.0]*numofemib
        case_H2_5D.input_sanity_check()
        #import pdb;pdb.set_trace()
        full_k, full_p, grid_size, all_height_l, all_z_count_l= case_H2_5D.create_global_structure(power_tier_l=power_tier_l,power_router_l=power_router_l,power_aib_l=power_aib_l,power_emib_l=power_emib_l)
        
        currfull_k, currfull_p, curr_grid_size = case_H2_5D.subdivide(full_k, full_p, grid_size)
        G_sparse = case_H2_5D.get_conductance_G_new(currfull_k, curr_grid_size, all_height_l, all_z_count_l)
        peak_temp=case_H2_5D.solver(G_sparse, currfull_p,  all_z_count_l,curr_grid_size)
        #result_list.append(peak_temp)
        #end_thermal = time.time()

    else:
        peak_temp='NA'

    return peak_temp