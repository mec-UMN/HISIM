
import numpy as np
import scipy.sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra
import matplotlib.pyplot as plt

import pandas as pd
#====================================================================================================
def plot_im(plot_data, title, save_name, vmin, vmax):

    ## change?
    fig,ax = plt.subplots(figsize=(15,5))
    width_plot = 100
    im = ax.imshow(plot_data , cmap = 'jet',vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=20)
    fig.colorbar(im)
    # fig.set_size_inches(len_plot*0.09,width_plot*0.09) # convert to inches, 100->4 inches
    # fig.figure(figsize=(100,50))
    fig.savefig(save_name, bbox_inches='tight',dpi=100)
    plt.close()



def checkneighbours(idxx,idxy,idxz,numx,numy,numz):
    if idxx<0 or idxx>=numx:
        return False
    if idxy<0 or idxy>=numy:
        return False
    if idxz<0 or idxz>=numz:
        return False
    return True
def id_(idxx, idxy, idxz, numx, numy, numz):
    return idxz*(numx*numy)+ idxx*(numy)+ idxy
#====================================================================================================


def devicemap_sanitycheck(devicemap):
    def checkadded(layer,totalwidth):
        pre = layer[0]
        total = pre[1]-pre[0]
        for idx, part in enumerate(layer):
            if idx==0:
                continue
            assert pre[1]==part[0]
            assert part[1]>part[0]
            assert (part[1]-part[0])%0.5==0
            total+= part[1]-part[0]
            pre = part
        assert total == totalwidth



    for design,layerlist in devicemap.items():
        toplayer = layerlist[0]
        assert toplayer[0][0]==0 
        totalwidth = toplayer[0][1]
        for layer in layerlist:
            checkadded(layer,totalwidth)



def get_unitsize(dict_size,tiles_edges_in_tier):
    tmp      = np.array(["imc","tsv0"])
    tmp_one  = np.concatenate([tmp]*tiles_edges_in_tier)
    tmp      = np.array(["tsv1","r"])
    tmp_two  = np.concatenate([tmp]*tiles_edges_in_tier)
    k_row    = np.stack((tmp_one,tmp_two))
    oneplane = np.concatenate([k_row]*tiles_edges_in_tier,axis=0)  #10,10
    xdim = ydim = 0
    for x in range(len(oneplane)):
        celltype = oneplane[x,0]
        ydim+=dict_size[celltype][1]
    for y in range(len(oneplane[0])):
        celltype = oneplane[0,y]
        xdim+=dict_size[celltype][0]
    oneplane_xdim = xdim
    oneplane_ydim = ydim
    return oneplane_xdim, oneplane_ydim




def basicblock(dict_size, dict_k,xdim,tiles_edges_in_tier):

    tallair  = dict_size["imc"][0]
    shortair = dict_size["r"][0]
    #====================================================================================================
    # size  
    #====================================================================================================
    tallunit  = str(xdim/(2*tiles_edges_in_tier))+','+ str(tallair)
    shortunit  = str(xdim/(2*tiles_edges_in_tier))+','+ str(shortair)
    onenormalcol=[]
    for i in range(tiles_edges_in_tier):
        onenormalcol.append(tallunit)
        onenormalcol.append(shortunit)
    onenormalcol=np.array(onenormalcol).reshape(tiles_edges_in_tier*2,1)
    #onenormalcol = np.array([tallunit,shortunit,tallunit,shortunit,tallunit,shortunit,tallunit,shortunit,tallunit,shortunit]).reshape(tiles_edges_in_tier*2,1)


    planeunit0_0 =  str(tallair)+','+ str(tallair)
    planeunit0_1 =  str(tallair)+','+ str(shortair)

    planeunit1_0 =  str(shortair)+','+ str(tallair)
    planeunit1_1 =  str(shortair)+','+ str(shortair)
    onedevicecol0=[]
    onedevicecol1=[]

    for i in range(tiles_edges_in_tier):
        onedevicecol0.append(planeunit0_0)
        onedevicecol0.append(planeunit0_1)
    onedevicecol0=np.array(onedevicecol0).reshape(tiles_edges_in_tier*2,1)

    for y in range(tiles_edges_in_tier):
        onedevicecol1.append(planeunit1_0)
        onedevicecol1.append(planeunit1_1)
    onedevicecol1=np.array(onedevicecol1).reshape(tiles_edges_in_tier*2,1)

    #onedevicecol0 = np.array([planeunit0_0,planeunit0_1,planeunit0_0,planeunit0_1,planeunit0_0,planeunit0_1,planeunit0_0,planeunit0_1,planeunit0_0,planeunit0_1,]).reshape(10,1)
    #onedevicecol1 = np.array([planeunit1_0,planeunit1_1,planeunit1_0,planeunit1_1,planeunit1_0,planeunit1_1,planeunit1_0,planeunit1_1,planeunit1_0,planeunit1_1,]).reshape(10,1)

    #====================================================================================================
    # conductivity
    #====================================================================================================
    # device layer 0 k
    onedevice_k_col0_layer0=[]
    onedevice_k_col1_layer0=[]
    onedevice_k_col0_layer1=[]
    onedevice_k_col1_layer1=[]
    onedevice_k_col0_layer2=[]
    onedevice_k_col1_layer2=[]
    for i in range(tiles_edges_in_tier):
        onedevice_k_col0_layer0.append(dict_k['k_imc_0'])
        onedevice_k_col0_layer0.append(dict_k['k_tsv_0'])
        onedevice_k_col1_layer0.append(dict_k['k_tsv_0'])
        onedevice_k_col1_layer0.append(dict_k['k_r_0'])
        onedevice_k_col0_layer1.append(dict_k['k_imc_1'])
        onedevice_k_col0_layer1.append(dict_k['k_tsv_1'])
        onedevice_k_col1_layer1.append(dict_k['k_tsv_1'])
        onedevice_k_col1_layer1.append(dict_k['k_r_1'])
        onedevice_k_col0_layer2.append(dict_k['k_imc_2'])
        onedevice_k_col0_layer2.append(dict_k['k_tsv_2'])
        onedevice_k_col1_layer2.append(dict_k['k_tsv_2'])
        onedevice_k_col1_layer2.append(dict_k['k_r_2'])
    onedevice_k_col0_layer0=np.array(onedevice_k_col0_layer0).reshape(tiles_edges_in_tier*2,1)
    onedevice_k_col1_layer0=np.array(onedevice_k_col1_layer0).reshape(tiles_edges_in_tier*2,1)
    onedevice_k_col0_layer1=np.array(onedevice_k_col0_layer1).reshape(tiles_edges_in_tier*2,1)
    onedevice_k_col1_layer1=np.array(onedevice_k_col1_layer1).reshape(tiles_edges_in_tier*2,1)
    onedevice_k_col0_layer2=np.array(onedevice_k_col0_layer2).reshape(tiles_edges_in_tier*2,1)
    onedevice_k_col1_layer2=np.array(onedevice_k_col1_layer2).reshape(tiles_edges_in_tier*2,1)

    #onedevice_k_col0_layer0= np.array([dict_k['k_imc_0'],dict_k['k_tsv_0'],dict_k['k_imc_0'],dict_k['k_tsv_0'],dict_k['k_imc_0'],dict_k['k_tsv_0'],dict_k['k_imc_0'],dict_k['k_tsv_0'],dict_k['k_imc_0'],dict_k['k_tsv_0']  ]).reshape(10,1)
    #onedevice_k_col1_layer0= np.array([dict_k['k_tsv_0'],dict_k['k_r_0'],  dict_k['k_tsv_0'],dict_k['k_r_0'],dict_k['k_tsv_0'],dict_k['k_r_0'],dict_k['k_tsv_0'],dict_k['k_r_0'],dict_k['k_tsv_0'],dict_k['k_r_0']  ]).reshape(10,1)
    # ddevice_k_yer _layer0 k
    #onedevice_k_col0_layer1= np.array([dict_k['k_imc_1'],dict_k['k_tsv_1'],dict_k['k_imc_1'],dict_k['k_tsv_1'],dict_k['k_imc_1'],dict_k['k_tsv_1'],dict_k['k_imc_1'],dict_k['k_tsv_1'],dict_k['k_imc_1'],dict_k['k_tsv_1']  ]).reshape(10,1)
    #onedevice_k_col1_layer1= np.array([dict_k['k_tsv_1'],dict_k['k_r_1'],dict_k['k_tsv_1'],dict_k['k_r_1'],dict_k['k_tsv_1'],dict_k['k_r_1'],dict_k['k_tsv_1'],dict_k['k_r_1'],dict_k['k_tsv_1'],dict_k['k_r_1']  ]).reshape(10,1)
    # ddevice_k_yer _layer0 k
    #onedevice_k_col0_layer2= np.array([dict_k['k_imc_2'],dict_k['k_tsv_2'],dict_k['k_imc_2'],dict_k['k_tsv_2'],dict_k['k_imc_2'],dict_k['k_tsv_2'],dict_k['k_imc_2'],dict_k['k_tsv_2'],dict_k['k_imc_2'],dict_k['k_tsv_2']  ]).reshape(10,1)
    #onedevice_k_col1_layer2= np.array([dict_k['k_tsv_2'],dict_k['k_r_2'],dict_k['k_tsv_2'],dict_k['k_r_2'],dict_k['k_tsv_2'],dict_k['k_r_2'],dict_k['k_tsv_2'],dict_k['k_r_2'],dict_k['k_tsv_2'],dict_k['k_r_2']  ]).reshape(10,1)
    #====================================================================================================
    # heatsink k
    oneheatsinkcol = np.array([dict_k['cu']]*tiles_edges_in_tier*2).reshape(tiles_edges_in_tier*2,1)
    # heatspread k
    oneheatspreadcol = np.array([dict_k['cu']]*tiles_edges_in_tier*2).reshape(tiles_edges_in_tier*2,1)
    # subs k
    onesubscol = np.array([dict_k['subs']]*tiles_edges_in_tier*2).reshape(tiles_edges_in_tier*2,1)
    # air k
    oneaircol = np.array([dict_k['air']]*tiles_edges_in_tier*2).reshape(tiles_edges_in_tier*2,1)

    #====================================================================================================
    dict_colk = dict()
    dict_colk['onesubs_k_col'] = onesubscol
    dict_colk['oneheatspread_k_col'] = oneheatspreadcol
    dict_colk['oneheatsink_k_col'] = oneheatsinkcol
    dict_colk['oneair_k_col'] = oneaircol
    dict_colk['onedevice_k_col0_layer0'] = onedevice_k_col0_layer0
    dict_colk['onedevice_k_col1_layer0'] = onedevice_k_col1_layer0
    dict_colk['onedevice_k_col0_layer1'] = onedevice_k_col0_layer1
    dict_colk['onedevice_k_col1_layer1'] = onedevice_k_col1_layer1
    dict_colk['onedevice_k_col0_layer2'] = onedevice_k_col0_layer2
    dict_colk['onedevice_k_col1_layer2'] = onedevice_k_col1_layer2
    #====================================================================================================
    onedevice_n_col0_layer0=[]
    onedevice_n_col1_layer0= []
    onedevice_n_col0_layer1= []
    onedevice_n_col1_layer1= []
    onedevice_n_col0_layer2= []
    onedevice_n_col1_layer2=[]
    for i in range(tiles_edges_in_tier):
        onedevice_n_col0_layer0.append('imc0')
        onedevice_n_col0_layer0.append('tsv')
        onedevice_n_col1_layer0.append('tsv')
        onedevice_n_col1_layer0.append('r')
        onedevice_n_col0_layer1.append('imc1')
        onedevice_n_col0_layer1.append('tsv')
        onedevice_n_col1_layer1.append('tsv')
        onedevice_n_col1_layer1.append('r')
        onedevice_n_col0_layer2.append('imc2')
        onedevice_n_col0_layer2.append('tsv')
        onedevice_n_col1_layer2.append('tsv')
        onedevice_n_col1_layer2.append('r')
    onedevice_n_col0_layer0=np.array(onedevice_n_col0_layer0).reshape(tiles_edges_in_tier*2,1)
    onedevice_n_col1_layer0=np.array(onedevice_n_col1_layer0).reshape(tiles_edges_in_tier*2,1)
    onedevice_n_col0_layer1=np.array(onedevice_n_col0_layer1).reshape(tiles_edges_in_tier*2,1)
    onedevice_n_col1_layer1=np.array(onedevice_n_col1_layer1).reshape(tiles_edges_in_tier*2,1)
    onedevice_n_col0_layer2=np.array(onedevice_n_col0_layer2).reshape(tiles_edges_in_tier*2,1)
    onedevice_n_col1_layer2=np.array(onedevice_n_col1_layer2).reshape(tiles_edges_in_tier*2,1)
    #onedevice_n_col0_layer0= np.array(['imc0','tsv','imc0','tsv','imc0','tsv','imc0','tsv','imc0','tsv'] ).reshape(10,1)
    #onedevice_n_col1_layer0= np.array(['tsv','r','tsv','r','tsv','r','tsv','r','tsv','r'] ).reshape(10,1)
    #onedevice_n_col0_layer1= np.array(['imc1','tsv','imc1','tsv','imc1','tsv','imc1','tsv','imc1','tsv'] ).reshape(10,1)
    #onedevice_n_col1_layer1= np.array(['tsv','r','tsv','r','tsv','r','tsv','r','tsv','r'] ).reshape(10,1)
    #onedevice_n_col0_layer2= np.array(['imc2','tsv','imc2','tsv','imc2','tsv','imc2','tsv','imc2','tsv'] ).reshape(10,1)
    #onedevice_n_col1_layer2= np.array(['tsv','r','tsv','r','tsv','r','tsv','r','tsv','r'] ).reshape(10,1)
    oneheatsink_n_col   = np.array(['cu']*tiles_edges_in_tier*2).reshape(tiles_edges_in_tier*2,1)
    oneheatspread_n_col = np.array(['cu']*tiles_edges_in_tier*2).reshape(tiles_edges_in_tier*2,1)
    onesubs_n_col       = np.array(['subs']*tiles_edges_in_tier*2).reshape(tiles_edges_in_tier*2,1)
    oneair_n_col        = np.array(['air']*tiles_edges_in_tier*2).reshape(tiles_edges_in_tier*2,1)
    #====================================================================================================
    dict_coln = dict()
    dict_coln['onesubs_n_col'] = onesubs_n_col
    dict_coln['oneheatspread_n_col'] = oneheatspread_n_col
    dict_coln['oneheatsink_n_col'] = oneheatsink_n_col
    dict_coln['oneair_n_col'] = oneair_n_col
    dict_coln['onedevice_n_col0_layer0'] = onedevice_n_col0_layer0
    dict_coln['onedevice_n_col1_layer0'] = onedevice_n_col1_layer0
    dict_coln['onedevice_n_col0_layer1'] = onedevice_n_col0_layer1
    dict_coln['onedevice_n_col1_layer1'] = onedevice_n_col1_layer1
    dict_coln['onedevice_n_col0_layer2'] = onedevice_n_col0_layer2
    dict_coln['onedevice_n_col1_layer2'] = onedevice_n_col1_layer2
    #====================================================================================================

    return onenormalcol, onedevicecol0, onedevicecol1, dict_colk, dict_coln

    



def iffallintodevicerange(xpos, devicelayer,tiles_edges_in_tier):
    for part in devicelayer:
        if part[2]=='device':
            partstart = part[0]
            partend = part[1]
            if partstart*tiles_edges_in_tier*2<=xpos<partend*tiles_edges_in_tier*2:
                return True
    return False
            
def findparttype(xpos, layer,tiles_edges_in_tier):
    for part in layer:
        partstart = part[0]
        partend = part[1]
        if partstart*tiles_edges_in_tier*2<=xpos<partend*tiles_edges_in_tier*2:
            return part[2]
            




def create_cube(dict_size,dict_z, dict_k ,xdim , devicemap,   heatsinkair_resoluation,tiles_edges_in_tier ):


    onebasiccol, onedevicecol0, onedevicecol1,dict_colk, dict_coln = basicblock(dict_size, dict_k, xdim,tiles_edges_in_tier)
    cube_geo_dict = dict()
    cube_k_dict   = dict()
    cube_z_dict   = dict()
    cube_n_dict   = dict()
    cube_layertype_dict   = dict()

    for designname, layerlist in devicemap.items():
        layer_geo_l = []
        layer_k_l = []
        layer_z_l = []
        layer_n_l = []
        layer_type_l = []

        devicelayer     = layerlist[2]
        heatsinklayer   = layerlist[0]
        heatspreadlayer = layerlist[1]



        totalx = heatsinklayer[0][1]

        for idx, layer in enumerate(layerlist):
            layertype = 'device'
            if idx==0:
                layertype = 'heatsink'
            elif idx==1:
                layertype = 'heatspread'
            elif idx==len(layerlist)-2:
                layertype = 'subs'
            elif idx==len(layerlist)-1:
                layertype = 'air'


            #====================================================================================================
            if layertype=='device':
                numofsublayer=3
            else :
                numofsublayer = (int)(dict_z[layertype]//heatsinkair_resoluation)
            #====================================================================================================


            for idx1 in range(numofsublayer):
                layer_type_l.append(layertype)
                #====================================================================================================
                if layertype != 'device':
                    layer_z_l.append(heatsinkair_resoluation/1000)
                else:
                    layer_z_l.append(dict_z[layertype][idx1]/1000)
                #====================================================================================================

                xpos = 0
                devicetrigger = False
                #====================================================================================================
                currlayer_dim = None
                currlayer_k = None
                currlayer_n = None
                #====================================================================================================

                while xpos<= totalx*tiles_edges_in_tier*2-1:

                    parttype = findparttype(xpos,layer,tiles_edges_in_tier )
                    layer2dictk = 'one'+parttype+'_k_col'
                    layer2dictn = 'one'+parttype+'_n_col'
                    if iffallintodevicerange(xpos,devicelayer,tiles_edges_in_tier):

                        if not devicetrigger:
                            if parttype =='device':
                                layer2dictk = 'one'+parttype+'_k_col0_layer{}'.format(idx1)
                                layer2dictn = 'one'+parttype+'_n_col0_layer{}'.format(idx1)
                            devicetrigger = True
                            if currlayer_dim is None:
                                currlayer_dim = onedevicecol0
                                currlayer_k = dict_colk[layer2dictk]
                                currlayer_n = dict_coln[layer2dictn]
                            else:
                                currlayer_dim = np.hstack([currlayer_dim, onedevicecol0]) 
                                currlayer_k = np.hstack([currlayer_k, dict_colk[layer2dictk]]) 
                                currlayer_n = np.hstack([currlayer_n, dict_coln[layer2dictn]]) 
                        else:
                            if parttype =='device':
                                layer2dictk = 'one'+parttype+'_k_col1_layer{}'.format(idx1)
                                layer2dictn = 'one'+parttype+'_n_col1_layer{}'.format(idx1)
                            devicetrigger = False
                            if currlayer_dim is None:
                                currlayer_dim = onedevicecol1
                                currlayer_k = dict_colk[layer2dictk]
                                currlayer_n = dict_coln[layer2dictn]
                            else :
                                currlayer_dim = np.hstack([currlayer_dim, onedevicecol1]) 
                                currlayer_k = np.hstack([currlayer_k, dict_colk[layer2dictk]]) 
                                currlayer_n = np.hstack([currlayer_n, dict_coln[layer2dictn]]) 
                    else:
                        if currlayer_dim is None:
                            currlayer_dim = onebasiccol
                            currlayer_k = dict_colk[layer2dictk]
                            currlayer_n = dict_coln[layer2dictn]
                        else:
                            currlayer_dim = np.hstack([currlayer_dim, onebasiccol]) 
                            currlayer_k = np.hstack([currlayer_k, dict_colk[layer2dictk]]) 
                            currlayer_n = np.hstack([currlayer_n, dict_coln[layer2dictn]]) 

                    xpos+=1

                #====================================================================================================
                layer_geo_l.append(currlayer_dim)
                layer_k_l.append(currlayer_k)
                layer_n_l.append(currlayer_n)
                #====================================================================================================

        layer_geo = np.stack(layer_geo_l)
        layer_k   = np.stack(layer_k_l)
        layer_n   = np.stack(layer_n_l)
        assert layer_geo.shape==layer_k.shape==layer_n.shape
        cube_geo_dict[designname] = layer_geo
        cube_k_dict[designname]   = layer_k
        cube_z_dict[designname]   = layer_z_l
        cube_n_dict[designname]   = layer_n
        cube_layertype_dict[designname]   = layer_type_l


    return cube_geo_dict, cube_k_dict, cube_z_dict,cube_n_dict, cube_layertype_dict




def snakewalk(numofnodex, numofnodey, start_pos, direction0, direction1):

    totalnode = numofnodex*numofnodey
    count = 0
    x, y = start_pos
    order_idx = []
    cood_set = set()
    prevx=x
    prevy=y


    while True:
        order_idx.append((x,y))
        assert (x,y) not in cood_set
        assert prevx==x or prevy==y
        cood_set.add((x,y))
        prevx=x
        prevy=y
        #====================================================================================================
        if direction0== 'up':
            x-=1
            if x<0:
                direction0='down'
                x=0
                if direction1 == 'right':
                    y+=2
                    if y==start_pos[1]+numofnodey:
                        y-=2
                        break
                elif direction1=='left':
                    y-=2
                    if y<start_pos[1]:
                        y+=2
                        break
                else:
                    print('error')
                    exit()
        #====================================================================================================
        elif direction0 == 'down':
            x+=1
            if x==numofnodex:
                direction0 = 'up'
                x=numofnodex-1
                if direction1 == 'right':
                    y+=2
                    if y==start_pos[1]+numofnodey:
                        y-=2
                        break
                elif direction1=='left':
                    y-=2
                    if y<start_pos[1]:
                        y+=2
                        break
                else:
                    print('error')
                    exit()
        #====================================================================================================
        elif direction0=='left':
            y-=2
            if y<start_pos[1]:
                direction0 ='right'
                y=start_pos[1]
                if direction1 == 'up':
                    x-=1
                    if x<0:
                        x+=1
                        break
                elif direction1=='down':
                    x+=1
                    if x==numofnodex:
                        x-=1
                        break
                else:
                    print('error')
                    exit()
        #====================================================================================================
        elif direction0=='right' :
            y+=2
            if y==start_pos[1]+numofnodey:
                direction0 ='left'
                y=start_pos[1]+numofnodey-1
                if direction1 == 'up':
                    x-=1
                    if x<0:
                        x+=1
                        break
                elif direction1=='down':
                    x+=1
                    if x==numofnodex:
                        x-=1
                        break
                else:
                    print('error')
                    exit()
        else:
            print('error')
            exit()


    # assert  totalnode==len(order_idx)
    return order_idx, (x,y)







def load_power(case_,dict_z, devicemap, cube_n_dict, power_tsv, power_router, numofdevicelayer_dict,cube_layertype_dict,tiles_edges_in_tier,chiplet_num,placement_method):

    power_inform = "./to_interconnect_analy_v1/layer_performance.csv"
    power_inform = pd.read_csv(power_inform, header=None)
    power_inform = power_inform.to_numpy()

    computing_inform = "./to_interconnect_analy_v1/layer_inform.csv"
    computing_data = pd.read_csv(computing_inform, header=None)
    computing_data = computing_data.to_numpy()
    power_l=[]
    for i in range(len(computing_data)-1):
        for y in range(computing_data[i][1]):
            power_l.append(float(power_inform[i][5]))
        if (computing_data[i][9]+1)*(tiles_edges_in_tier)*(tiles_edges_in_tier)-computing_data[i][7]!=0 and (computing_data[i][9]+1)*(tiles_edges_in_tier)*(tiles_edges_in_tier)-computing_data[i][7]<computing_data[i+1][1]:
            for c in range((computing_data[i][9]+1)*(tiles_edges_in_tier)*(tiles_edges_in_tier)-computing_data[i][7]):
                power_l.append(0)
    for i in range(computing_data[-1][1]):
        power_l.append(float(power_inform[-1][5]))
    if (computing_data[-1][9]+1)*(tiles_edges_in_tier)*(tiles_edges_in_tier)-computing_data[-1][7]!=0:
        for c in range((computing_data[-1][9]+1)*(tiles_edges_in_tier)*(tiles_edges_in_tier)-computing_data[-1][7]):
            power_l.append(0)
    
    new_power_list=[]
    if placement_method==1:
        power_l=power_l
    elif placement_method==2:
        for i in range(len(power_l)):
            new_power_list.append(power_l[len(power_l)-i-1])
        power_l=new_power_list
    
    for i in range(len(power_l)):
        if i ==int(len(power_l)/4) or i==int(len(power_l)*3/4):
            power_l[i]=1000/25
        else:
            power_l[i]=1000/25
    print(power_l)
    #====================================================================================================
    """
    if case_==1:
        power_empty_loc= [74,99,124,149]
        with open('tmp.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row["Power"]!="":
                    # power_l.append(np.float64(row['each tile power']))
                    power_l.append(float(row['each tile power']))
                    if row['tile number']=='2':
                        power_l.append(float(row['each tile power']))
    else:
    #====================================================================================================
        power_empty_loc= []
        with open('tmp{}.csv'.format(case_-1), mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row["Power(mW)"]!="":
                    power_l.append(float(row["Power(mW)"]))
        
    #====================================================================================================
    count =0
    power_p = 0
    power_real_l = []
    while count<25*6:
        if count in set(power_empty_loc):
            power_real_l.append(0)
        else:
            power_real_l.append(power_l[power_p])
            power_p+=1
        count+=1
    """
    #====================================================================================================
    power_l = np.array(power_l)
   
    assert len(power_l)==chiplet_num*tiles_edges_in_tier*tiles_edges_in_tier
    #====================================================================================================

    dict_power_container = dict()
    for design, layer_n in  cube_n_dict.items():


        #====================================================================================================
        curr_chip_zstart_copy = None
        layertype_l = cube_layertype_dict[design]
        for idx,layertype  in enumerate(layertype_l):
            if layertype == 'device':
                curr_chip_zstart_copy=idx
                break
        #====================================================================================================




        layer_l = devicemap[design]
        devicelayer = layer_l[2]
        numofdevicelayer = numofdevicelayer_dict[design]
        # numofdevicelayer = len(layer_l)-3
        device_range = []


        power_container= np.zeros(layer_n.shape)
        power_count=0
        tsv_idx=0
        router_idx=0
        for partidx, part in enumerate(devicelayer):
            if part[2]!='device':
                continue
                # device_range.append((part[0]*10, part[1]*10))


            curr_chip_xystart  = (0,part[0]*tiles_edges_in_tier*2)
            # curr_chip_zstart   = dict_z['heatsink']+dict_z['heatspread']
            curr_chip_zstart   = curr_chip_zstart_copy
            order_idx, end_pos = snakewalk(tiles_edges_in_tier*2, tiles_edges_in_tier*2, curr_chip_xystart, 'down','right')

            if design=='0stacks' and partidx%2==1:
                curr_chip_xystart  = (0,part[0]*tiles_edges_in_tier*2)
                order_idx, end_pos = snakewalk(tiles_edges_in_tier*2, tiles_edges_in_tier*2, curr_chip_xystart, 'down','right')
            elif design=='0stacks' and partidx%2==0:
                curr_chip_xystart  = (9,part[0]*tiles_edges_in_tier*2)
                order_idx, end_pos = snakewalk(tiles_edges_in_tier*2, tiles_edges_in_tier*2, curr_chip_xystart, 'up','right')
            # print(design,  '        ',   order_idx)
            # print()



            for devicelayer in range(numofdevicelayer) :
                #import pdb;pdb.set_trace()
                for x,y in order_idx:
                    for sublayer in range(3):
                        actual_layer = curr_chip_zstart+sublayer
                        tiletype = layer_n[actual_layer, x,y]
                        tiletype_next = layer_n[actual_layer, x,y+1]
                        assert tiletype!='subs' and tiletype!='air' and tiletype!='cu'
                        assert tiletype_next!='subs' and tiletype_next!='air' and tiletype_next!='cu'

                        tilepower = 0
                        if tiletype=='tsv' :
                            tilepower = power_tsv[tsv_idx]
                        elif tiletype=='imc0':
                            tilepower = power_l[power_count]
                            power_count+=1
                        elif tiletype=='r' and sublayer ==0:
                            tilepower = power_router[router_idx]


                        tilepower_next = 0
                        if tiletype_next=='tsv' :
                            tilepower_next = power_tsv[tsv_idx]
                        elif tiletype_next=='imc0':
                            tilepower_next = power_l[power_count]
                            power_count+=1
                        elif tiletype_next=='r' and sublayer ==0:
                            tilepower_next = power_router[router_idx]
                        #====================================================================================================
                        # tilepower = 0.008
                        # tilepower_next = 0.008
                        #====================================================================================================
                        power_container[actual_layer,x,y ]  = tilepower/1000
                        power_container[actual_layer,x,y+1 ]=tilepower_next/1000


                tsv_idx+=1
                router_idx+=1
                curr_chip_zstart = curr_chip_zstart+3
        
        assert tsv_idx==chiplet_num
        assert router_idx==chiplet_num
        #import pdb;pdb.set_trace()
        assert power_count==len(power_l)
        dict_power_container[design]=power_container 



    return dict_power_container






def get_conductance_G(cube_geo_dict, cube_k_dict, cube_z_dict):


    direction_map = dict()
    direction_map[0]=(0,1,0)  #east,0
    direction_map[1]=(0,-1,0) #west,1
    direction_map[2]=(1,0,0)  #north,2
    direction_map[3]=(-1,0,0) #south,3
    direction_map[4]=(0,0,1)  #top,4
    direction_map[5]=(0,0,-1) #bottom,5


    cube_G_dict = dict()
    for design, geo_info_np in cube_geo_dict.items():
        k_info_np = cube_k_dict[design]
        z_info_np = cube_z_dict[design]
        assert geo_info_np.shape==k_info_np.shape
        numtotalnode =(geo_info_np.shape[0]+1)*geo_info_np.shape[1]*geo_info_np.shape[2]
        numoflayer = geo_info_np.shape[0]
        numofnodex = geo_info_np.shape[1]
        numofnodey = geo_info_np.shape[2]

        #====================================================================================================
        G_sparse = sparse_mat.dok_matrix((numtotalnode, numtotalnode))
        #====================================================================================================
        nodeid = 0
        for z_idx in range(numoflayer):
            for x_idx in range(numofnodex):
                for y_idx in range(numofnodey):



                    dim_str = geo_info_np[z_idx, x_idx, y_idx].split(',')
                    centernode_length, centernode_width = (float)(dim_str[0]),(float)(dim_str[1])   
                    centerlayer_height = z_info_np[z_idx]

                    centernodeid = id_(x_idx, y_idx, z_idx, numofnodex, numofnodey,numoflayer )
                    assert centernodeid==nodeid
                    center_k = k_info_np[z_idx, x_idx, y_idx].item()




                    #====================================================================================================
                    if z_idx== numoflayer-1:
                        neighbour_id_downbelow = id_(x_idx, y_idx, z_idx+1, numofnodex, numofnodey,numoflayer )
                        G_sparse[centernodeid, neighbour_id_downbelow ]=1
                        G_sparse[neighbour_id_downbelow, centernodeid ]=1
                    #====================================================================================================




                    G_sparse[centernodeid, centernodeid]=0
                    #====================================================================================================
                    # check all neighbours and calculate conductance 
                    #====================================================================================================
                    for direction in range(6):
                        neighbour_x_idx= (direction_map[direction][0]+x_idx)
                        neighbour_y_idx= (direction_map[direction][1]+y_idx)
                        neighbour_z_idx= (direction_map[direction][2]+z_idx)
                        if checkneighbours(neighbour_x_idx,neighbour_y_idx,neighbour_z_idx,numofnodex, numofnodey,numoflayer):
                            neighbour_id = id_(neighbour_x_idx, neighbour_y_idx,neighbour_z_idx, numofnodex, numofnodey,numoflayer)
                            #====================================================================================================
                            # calcualte the avg k
                            #====================================================================================================
                            neighbour_k =   k_info_np[neighbour_z_idx,neighbour_x_idx,neighbour_y_idx]
                            dim_str     = geo_info_np[neighbour_z_idx,neighbour_x_idx,neighbour_y_idx].split(',')
                            neighbour_node_length, neighbour_node_width = (float)(dim_str[0]),(float)(dim_str[1])   

                            #====================================================================================================
                            if direction==0 or direction==1:
                                assert centernode_width==neighbour_node_width
                                d1 = centernode_length/2
                                d2 = neighbour_node_length/2
                                A = centernode_width*centerlayer_height
                            elif direction==2 or direction==3:
                                assert centernode_length==neighbour_node_length
                                d1 = centernode_width/2
                                d2 = neighbour_node_width/2
                                A = centernode_length*centerlayer_height
                            else:
                                assert centernode_width==neighbour_node_width
                                assert centernode_length==neighbour_node_length

                                #====================================================================================================
                                neighbourlayer_height = z_info_np[neighbour_z_idx]
                                #====================================================================================================
                                d1 = centerlayer_height/2
                                d2 = neighbourlayer_height/2
                                A = centernode_width*centernode_length




                            dd = d1+d2
                            #====================================================================================================
                            k_avg = dd/(d1/center_k+ d2/neighbour_k)
                            G_ = (k_avg*A)/dd
                            G_sparse[centernodeid, centernodeid]+=G_
                            G_sparse[centernodeid, neighbour_id]=-G_


                    nodeid+=1
        #====================================================================================================
        cube_G_dict[design]= G_sparse
        print('INFO:Done generating G for ',design)
    return cube_G_dict





def get_conductance_G_new(cube_geo_dict, cube_k_dict, cube_z_dict):


    direction_map = dict()
    direction_map[0]=(0,1,0)  #east,0
    direction_map[1]=(0,-1,0) #west,1
    direction_map[2]=(1,0,0)  #north,2
    direction_map[3]=(-1,0,0) #south,3
    direction_map[4]=(0,0,1)  #top,4
    direction_map[5]=(0,0,-1) #bottom,5


    cube_G_dict = dict()
    for design, geo_info_np in cube_geo_dict.items():
        k_info_np = cube_k_dict[design]
        z_info_np = cube_z_dict[design]
        assert geo_info_np.shape==k_info_np.shape
        numtotalnode =(geo_info_np.shape[0]+2)*geo_info_np.shape[1]*geo_info_np.shape[2]
        numoflayer = geo_info_np.shape[0]
        numofnodex = geo_info_np.shape[1]
        numofnodey = geo_info_np.shape[2]

        #====================================================================================================
        G_sparse = sparse_mat.dok_matrix((numtotalnode, numtotalnode))
        #====================================================================================================
        nodeid = 0
        for z_idx in range(1,numoflayer+1):
            for x_idx in range(numofnodex):
                for y_idx in range(numofnodey):



                    dim_str = geo_info_np[z_idx-1, x_idx, y_idx].split(',')
                    centernode_length, centernode_width = (float)(dim_str[0]),(float)(dim_str[1])   
                    centerlayer_height = z_info_np[z_idx-1]

                    centernodeid = id_(x_idx, y_idx, z_idx, numofnodex, numofnodey,numoflayer )
                    # assert centernodeid==nodeid
                    center_k = k_info_np[z_idx-1, x_idx, y_idx].item()




                    #====================================================================================================
                    if z_idx== numoflayer:
                        neighbour_id_downbelow = id_(x_idx, y_idx, z_idx+1, numofnodex, numofnodey,numoflayer )
                        G_sparse[centernodeid, neighbour_id_downbelow ]=1
                        G_sparse[neighbour_id_downbelow, centernodeid ]=1

                    if z_idx== 1:
                        neighbour_id_downbelow = id_(x_idx, y_idx, z_idx-1, numofnodex, numofnodey,numoflayer )
                        G_sparse[centernodeid, neighbour_id_downbelow ]=1
                        G_sparse[neighbour_id_downbelow, centernodeid ]=1
                    #====================================================================================================




                    G_sparse[centernodeid, centernodeid]=0
                    #====================================================================================================
                    # check all neighbours and calculate conductance 
                    #====================================================================================================
                    for direction in range(6):
                        neighbour_x_idx= (direction_map[direction][0]+x_idx)
                        neighbour_y_idx= (direction_map[direction][1]+y_idx)
                        neighbour_z_idx= (direction_map[direction][2]+z_idx)
                        if checkneighbours(neighbour_x_idx,neighbour_y_idx,neighbour_z_idx-1,numofnodex, numofnodey,numoflayer):
                            neighbour_id = id_(neighbour_x_idx, neighbour_y_idx,neighbour_z_idx, numofnodex, numofnodey,numoflayer)
                            #====================================================================================================
                            # calcualte the avg k
                            #====================================================================================================
                            neighbour_k =   k_info_np[neighbour_z_idx-1,neighbour_x_idx,neighbour_y_idx]
                            dim_str     = geo_info_np[neighbour_z_idx-1,neighbour_x_idx,neighbour_y_idx].split(',')
                            neighbour_node_length, neighbour_node_width = (float)(dim_str[0]),(float)(dim_str[1])   

                            #====================================================================================================
                            if direction==0 or direction==1:
                                assert centernode_width==neighbour_node_width
                                d1 = centernode_length/2
                                d2 = neighbour_node_length/2
                                A = centernode_width*centerlayer_height
                            elif direction==2 or direction==3:
                                assert centernode_length==neighbour_node_length
                                d1 = centernode_width/2
                                d2 = neighbour_node_width/2
                                A = centernode_length*centerlayer_height
                            else:
                                assert centernode_width==neighbour_node_width
                                assert centernode_length==neighbour_node_length

                                #====================================================================================================
                                neighbourlayer_height = z_info_np[neighbour_z_idx-1]
                                #====================================================================================================
                                d1 = centerlayer_height/2
                                d2 = neighbourlayer_height/2
                                A = centernode_width*centernode_length




                            dd = d1+d2
                            #====================================================================================================
                            k_avg = dd/(d1/center_k+ d2/neighbour_k)
                            G_ = (k_avg*A)/dd
                            G_sparse[centernodeid, centernodeid]+=G_
                            G_sparse[centernodeid, neighbour_id]=-G_


                    nodeid+=1
        #====================================================================================================
        cube_G_dict[design]= G_sparse
        print('INFO:Done generating G for ',design)
    return cube_G_dict



def convert2realratio(t,namemap,xdim):
    assert t.shape==namemap.shape

    airlength = xdim/10
    #====================================================================================================
    outt = None
    for layer in range(len(t)):
        currlayer = None
        for row in range(t.shape[1]):
            currrow = None
            for col in range(t.shape[2]):

                value = t[layer,row,col]
                celltype = namemap[layer,row,col]
                #====================================================================================================
                numofwidthblock = 0
                numoflengthblock = 0
                if row %2==0:
                    numofwidthblock=2
                    # two block height per row, 
                    if celltype=='imc0' or celltype=='imc1' or celltype=='imc2':
                        # two block length per col
                        numoflengthblock = 2
                    else:
                        # one  block length per col
                        numoflengthblock = 1
                else:
                    numofwidthblock=1
                    # one block height per row, 
                    if celltype=='tsv':
                        # two block length per col
                        numoflengthblock = 2
                    else:
                        # one  block length per col
                        numoflengthblock = 1
                assert numofwidthblock!=0 and numoflengthblock!=0
                #====================================================================================================
                if currrow is None:
                    currrow = np.ones((numofwidthblock, numoflengthblock  ))*value
                    continue
                newcell = np.ones((numofwidthblock, numoflengthblock  ))*value
                currrow = np.hstack([currrow, newcell])
            if currlayer is None:
                currlayer = currrow
                continue
            currlayer = np.vstack([currlayer, currrow])
            

        x = currlayer.shape[0]
        y = currlayer.shape[1]
        currlayer= currlayer.reshape((1,x,y))
        if outt is None:
            outt= currlayer
            continue
        #====================================================================================================
        outt = np.vstack([outt, currlayer])
    return outt
            







    #====================================================================================================




def solver(cube_G_dict, cube_n_dict,cube_power_dict,cube_layertype_dict,xdim,sim_name ):
    for design,G_sparse in cube_G_dict.items():

        namemap = cube_n_dict[design]
        p = cube_power_dict[design]
        numoflayer = p.shape[0]
        numofnodex = p.shape[1]
        numofnodey = p.shape[2]

        p= p.reshape(numoflayer*numofnodex*numofnodey,1)
        newp = np.ones(((numoflayer+2)*numofnodex*numofnodey,1))*298
        #====================================================================================================
        # newp = np.ones(((numoflayer+1)*numofnodex*numofnodey,1))*298
        # newp[:numoflayer*numofnodex*numofnodey,:]= p
        #====================================================================================================
        newp[numofnodex*numofnodey:(numoflayer+1)*numofnodex*numofnodey,:]= p
        p=newp

        #====================================================================================================



        # print(p.min())
        # print(p.max())

        G_sparse = G_sparse.tocsc()

        p = sparse_mat.csc_matrix(p)
        # I = sparse_mat.identity(G_sparse.shape[0]) * 1e-2
        # G_sparse = G_sparse + I
        t = sparse_algebra.spsolve(G_sparse,p,permc_spec=None,use_umfpack=True)

        #====================================================================================================
        # t = t.reshape(numoflayer+1,numofnodex,numofnodey)
        # t = t[:numoflayer, :,:]
        #====================================================================================================
        t = t.reshape(numoflayer+2,numofnodex,numofnodey)
        t = t[1:numoflayer+1, :,:]
        #====================================================================================================


        layertype_l = cube_layertype_dict[design]
        devicestart = None
        deviceend = None
        start = False
        for i,layername in enumerate(layertype_l):
            if devicestart is None and layername == 'device':
                devicestart = i
                start = True
                continue
            if deviceend is None and  start and layername != 'device':
                deviceend = i
                break

        t = t[devicestart:deviceend,:,:]
        namemap = namemap[devicestart:deviceend,:,:]
        realratiot = convert2realratio(t, namemap,xdim)



        vmin  = t.min()
        vmax  = t.max()

        print(design,'    Min t: ', round(vmin, 2),  '\t\tPeak t:', round(vmax,2),'\t\tAverage t:', round(t.mean(),2 ))
        for i in range(len(t)):
            # layername = layertype_l[i]+' '+str(i)
            plot_im(plot_data=realratiot[i,:,:], title='device {}'.format(i), save_name='./result_thermal/{}/power_map{}.png'.format(design, i), vmin=vmin, vmax=vmax)








