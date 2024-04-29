import re
import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
import sys
import scipy.sparse as sparse_mat
import scipy.sparse.linalg as sparse_algebra
from scipy.sparse.linalg import inv
import torch
import collections
import pandas as pd
# from util import *
import time
#import cupy as cp
#from cupyx.scipy.sparse import csr_matrix as csr_gpu



# np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=50_000)



class H2_5D(object):
    def __init__(self,area_single_tile,single_router_area,chiplet_num,mesh_edge,area_aib,area_emib,resolution =2):
        #====================================================================================================
        # parameters
        #====================================================================================================
        self.alpha=3.5
        # self.alpha=20
        self.N = mesh_edge         # the single tier stucture NxN #todo
        self.tier_sublayer = 1  #todo
        self.emib_sublayer = 1  #todo
        self.resolution = resolution
        #====================================================================================================
        # 2:   2 chiplet
        # 3:   L shape   
        # 4:   4 tiers
        self.Nstructure = chiplet_num  #todo
        #====================================================================================================
        self.assign_power_start = 'tl'  # tl: topleft  tr: topright   bl bottomleft  br bottomright
        self.initial_direction = 'b'    # l: go left   r : go right   t: go top    b : go bottom
        #====================================================================================================
        # size of router, imc
        self.imc_size = (math.sqrt(area_single_tile+1)-1)/1000
        self.r_size   = math.sqrt(single_router_area)/1000
        #self.imc_size = math.sqrt(0.336536)/1000
        #self.r_size   = math.sqrt(0.019626)/1000

        self.edge_length_chiplet=(self.imc_size+self.r_size)*self.N*1000 #mm 
        self.area_aib=area_aib #mm2
        self.area_emib=area_emib
        self.aib_width=self.area_aib/self.edge_length_chiplet/1000
        self.emib_width=self.area_emib/self.edge_length_chiplet/1000
        #import pdb;pdb.set_trace()
        #====================================================================================================
        # size of aib, emib
        self.aib_width_n  = math.ceil(self.aib_width/self.r_size)  #todo  # real_width/ unit_width
        self.emib_width_n = math.ceil(self.emib_width/self.r_size)
        #import pdb;pdb.set_trace()
        self.aib_size     = self.r_size
        self.emib_size    = self.r_size
        #====================================================================================================
        # vertical z 
        self.dict_z = dict()
        self.dict_z['tier_z_0'] = 0.002/1000  #todo
        self.dict_z['tier_z_1'] = 0.1/1000    #todo
        self.dict_z['emib_z_0'] = 0.008/1000   #todo change to the thickness of
        self.dict_z['emib_z_1'] = 0.008/1000   #todo


        self.dict_z['heatsink']=40/1000 # jingbo
        self.dict_z['heatspread']=20/1000 # jingbo
        # self.dict_z['heatsink']=3/1000
        # self.dict_z['heatspread']=1/1000

        self.dict_z['subs']=1/1000
        self.dict_z['pcb']=1.6/1000

        self.dict_z['air']=50/1000  # jingbo
        # self.dict_z['air']=1/1000
        self.heatsinkair_resoluation=1/1000

        #====================================================================================================


        self.dict_k = dict()
        self.dict_k['k_imc_0']  = 110/self.alpha  #todo
        self.dict_k['k_imc_1']= 142.8/self.alpha  #todo

        self.dict_k['k_aib_0']  = 110/self.alpha # todo
        self.dict_k['k_aib_1']= 142.8/self.alpha # todo
        self.dict_k['k_emib_0']  =110/self.alpha # todo
        self.dict_k['k_emib_1']  = 110/self.alpha # todo



        self.dict_k['k_tsv_0']= 110/self.alpha
        self.dict_k['k_tsv_1']  = 142.8/self.alpha
        # self.dict_k['k_tsv_0']= 142.8/self.alpha
        # self.dict_k['k_tsv_1']  = 200/self.alpha

        # self.dict_k['k_imc_2']    = 4/self.alpha
        self.dict_k['k_r_0']    = 110/self.alpha
        self.dict_k['k_r_1']  = 142.8/self.alpha
        # self.dict_k['k_r_2']      = 4/self.alpha
        self.dict_k['cu']       = 398/self.alpha
        self.dict_k['air']   = 0.0243/self.alpha
        self.dict_k['subs']   = 142.8/self.alpha
        self.dict_k['pcb']   = 0.6/self.alpha  #todo




        #====================================================================================================
        # debug
        # self.alpha=1
        # self.dict_k['k_imc_0']  = 10/self.alpha  #todo
        # self.dict_k['k_imc_1']  = 11/self.alpha  #todo
        # self.dict_k['k_aib_0']  = 20/self.alpha # todo
        # self.dict_k['k_aib_1']  = 21/self.alpha # todo
        # self.dict_k['k_emib_0'] = 30/self.alpha # todo
        # self.dict_k['k_r_0']    = 70/self.alpha
        # self.dict_k['k_r_1']    = 71/self.alpha
        # self.dict_k['cu']       = 398/self.alpha
        # self.dict_k['air']      = 77/self.alpha
        # self.dict_k['subs']     = 33/self.alpha
        # self.dict_k['pcb']      = 55/self.alpha  #todo
        # self.emib_size =1
        # self.imc_size =2
        # self.r_size =1
        # self.aib_size     =  4 #todo
        # self.emib_size    = 5  #todo
        #====================================================================================================


    def plot_im(self,plot_data, title, save_name, vmin, vmax):
        fig,ax = plt.subplots(figsize=(15,5))
        width_plot = 100
        im = ax.imshow(plot_data , cmap = 'jet',vmin=vmin, vmax=vmax)
        ax.set_title(title, pad=20)
        fig.colorbar(im)
        # fig.set_size_inches(len_plot*0.09,width_plot*0.09) # convert to inches, 100->4 inches
        # fig.figure(figsize=(100,50))
        fig.savefig(save_name, bbox_inches='tight',dpi=100)
        plt.close()

    def checkneighbours(self,idxx,idxy,idxz,numx,numy,numz):
        if idxx<0 or idxx>=numx:
            return False
        if idxy<0 or idxy>=numy:
            return False
        if idxz<0 or idxz>=numz:
            return False
        return True
    def id_(self,idxx, idxy, idxz, numx, numy, numz):
        return idxz*(numx*numy)+ idxx*(numy)+ idxy


    def input_sanity_check(self):
        if self.assign_power_start=='tl':
            assert self.initial_direction=='r' or self.initial_direction=='b'
        elif self.assign_power_start=='tr':
            assert self.initial_direction=='l' or self.initial_direction=='b'
        elif self.assign_power_start=='bl':
            assert self.initial_direction=='r' or self.initial_direction=='t'
        elif self.assign_power_start=='br':
            assert self.initial_direction=='l' or self.initial_direction=='t'
        else:
            print('unknown self.assign_power_start, should be tl/tr/bl/br')
            exit()

    def snake_walk_new(self,shape,power_id):

        ids = np.zeros(shape)
        power_id_tocood = dict()
        #====================================================================================================
        x = None
        y = None
        nextx = None
        nexty = None

        if self.assign_power_start=='tl':
            x = 0
            y = 0
        elif self.assign_power_start=='tr':
            x = 0
            y = self.N-1
        elif self.assign_power_start=='bl':
            x = self.N-1
            y = 0
        elif self.assign_power_start=='br':
            x = self.N-1
            y = self.N-1
        else:
            print('unknown self.assign_power_start, should be tl/tr/bl/br')
            exit()
        #====================================================================================================
        totalnodes = self.N*self.N
        count = 0
        direction = self.initial_direction
        visited = set()


        while count< totalnodes:
            count+=1
            row_s = x*2
            col_s = y*2
            cood  = [(row_s,col_s),(row_s+1,col_s),(row_s,col_s+1),(row_s+1,col_s+1)]
            power_id_tocood[power_id] = cood
            visited.add((x,y))

            if direction=='b':
                if x==self.N-1:
                    direction = 't'
                    nextx = x
                    if (x,y+1) in visited or y+1==self.N: nexty=y-1
                    else:                  nexty=y+1
                else:
                    nextx = x+1
                    nexty = y

                    
            elif direction=='t':
                if x==0:
                    direction = 'b'
                    nextx = x
                    if (x,y+1) in visited or y+1==self.N: nexty=y-1
                    else:                  nexty=y+1
                else:
                    nextx = x-1
                    nexty = y

            elif direction=='l':
                if y==0:
                    direction = 'r'
                    nexty = y
                    if (x+1,y) in visited or x+1==self.N: nextx=x-1
                    else:                  nextx=x+1
                else:
                    nextx = x
                    nexty = y-1

            elif direction=='r':
                if y==self.N-1:
                    direction = 'l'
                    nexty = y
                    if (x+1,y) in visited or x+1==self.N: nextx=x-1
                    else:                  nextx=x+1
                else:
                    nextx = x
                    nexty = y+1

            for (tmpx, tmpy) in cood: 
                ids[tmpx,tmpy]=power_id

            x=nextx
            y=nexty
            power_id+=1
        return power_id,power_id_tocood
    
    def create_tier_structure(self,power_id):

 

        # debug
        # self.imc_size =2
        # self.aib_size =1
        # self.r_size =1
        # self.dict_z['tier_z_0']=0.1
        # self.dict_z['tier_z_1']=0.2
        #====================================================================================================
        basic_k    = []
        basic_n    = []
        basic_size = []

        z_height_l   = []
        z_name_l   = []


        for k in range(self.tier_sublayer):
            #====================================================================================================
            name_imc  = 'k_imc_'+str(k)
            name_r    = 'k_r_'+str(k)
            name_tsv  = 'k_tsv_'+str(k)


            assert name_imc in self.dict_k.keys()
            assert name_r in self.dict_k.keys()
            if self.Nstructure==1: #hardcode
                basic_k.append(np.array([[self.dict_k[name_imc],self.dict_k[name_tsv] ], [self.dict_k[name_tsv], self.dict_k[name_r]]]))
            else:
                basic_k.append(np.array([[self.dict_k[name_imc],self.dict_k[name_imc] ], [self.dict_k[name_imc], self.dict_k[name_r]]]))
            #====================================================================================================
            basic_n.append(np.array([['imc_n_'+(str)(k),'imc_v_'+(str)(k) ], ['imc_h_'+(str)(k), 'route_'+(str)(k)]]))
            #====================================================================================================
            zname_imc = 'tier_z_'+str(k)
            assert zname_imc in self.dict_z.keys()
            z_height_l.append(self.dict_z[zname_imc])
            z_name_l.append(zname_imc)

            basic_size.append(np.array([[[self.imc_size, self.imc_size],[self.imc_size, self.r_size] ], [[self.r_size, self.imc_size],[self.r_size, self.r_size]]]))
        #====================================================================================================



        single_tier_name = np.array(['none___']*self.N*self.N*2*2*self.tier_sublayer).reshape(self.tier_sublayer,self.N*2,self.N*2)
        single_tier_size = np.zeros((self.tier_sublayer, self.N*2,self.N*2,2))
        single_tier_k = np.zeros((self.tier_sublayer, self.N*2,self.N*2))


        power_id, power_id_tocood = self.snake_walk_new(np.ones((self.N*2,self.N*2)).shape, power_id=power_id)
        for m in range(self.N):
            for n in range(self.N):
                row_s= m*2
                row_e= (m+1)*2
                col_s= n*2
                col_e= (n+1)*2

                for k in range(self.tier_sublayer):
                    single_tier_k[k,row_s:row_e, col_s:col_e] =basic_k[k]
                    single_tier_name[k,row_s:row_e, col_s:col_e] =basic_n[k]
                    single_tier_size[k,row_s:row_e, col_s:col_e] =basic_size[k]


        # print(single_tier_name[0,:,:])
        # print(single_tieself.r_size[0,:,:])
        # exit()
        # print(single_tier_k[1,:,:])
        # print(single_tier_k[0,:,:])
        # exit()
        # print(single_tier_name.shape)
        # print(single_tieself.r_size.shape)
        # print(single_tier_k.shape)


        return  single_tier_name, single_tier_size, single_tier_k, z_height_l, z_name_l,power_id, power_id_tocood
    #====================================================================================================






    def create_aib_structure(self):

        # debug
        # self.imc_size =2
        # self.aib_size =1
        # self.r_size =1
        # self.dict_z['tier_z_0']=0.1
        # self.dict_z['tier_z_1']=0.2



        basic_k    = []
        basic_n    = []
        basic_size = []

        z_height_l   = []
        z_name_l   = []

        for k in range(self.tier_sublayer):
            #====================================================================================================
            name_aib  = 'k_aib_'+str(k)
            assert name_aib in self.dict_k.keys()
            basic_k.append(np.array([self.dict_k[name_aib]]*2).reshape(2,1))
            #====================================================================================================
            basic_n.append(np.array(['aib___'+str(k)]*2).reshape(2,1))
            #====================================================================================================
            zname_aib = 'tier_z_'+str(k)
            assert zname_aib in self.dict_z.keys()
            z_height_l.append(self.dict_z[zname_aib])
            z_name_l.append(zname_aib)
            basic_size.append(np.array([[[self.imc_size, self.aib_size] ], [[self.r_size, self.aib_size]  ]]))


        single_aib_name = np.array(['none___']*self.N*self.aib_width_n*2*self.tier_sublayer).reshape(self.tier_sublayer,self.N*2,self.aib_width_n)
        single_aib_size = np.zeros((self.tier_sublayer, self.N*2,self.aib_width_n,2))
        single_aib_k    = np.zeros((self.tier_sublayer, self.N*2,self.aib_width_n))



        for m in range(self.N):
            for n in range(self.aib_width_n):
                row_s= m*2
                row_e= (m+1)*2
                col_s= n
                col_e= (n+1)
                for k in range(self.tier_sublayer):
                    single_aib_k[k,row_s:row_e, col_s:col_e]    = basic_k[k]
                    single_aib_name[k,row_s:row_e, col_s:col_e] = basic_n[k]
                    single_aib_size[k,row_s:row_e, col_s:col_e] = basic_size[k]


        # print(single_tier_k[1])
        # print(single_tier_name[1])
        # print(single_tieself.r_size[1])
        # exit()

        return single_aib_name, single_aib_size, single_aib_k, z_height_l, z_name_l
        



    #====================================================================================================

    def create_emib_structure(self):
        # debug
        # self.emib_size =1
        # self.imc_size =2
        # self.r_size =1



        basic_k    = []
        basic_n    = []
        basic_size = []
        z_height_l   = []
        z_name_l   = []


        for k in range(self.emib_sublayer):
            #====================================================================================================
            name_emib  = 'k_emib_'+str(k)
            assert name_emib in self.dict_k.keys()
            basic_k.append(np.array([self.dict_k[name_emib]]*2).reshape(2,1))
            #====================================================================================================
            basic_n.append(np.array(['emib__'+str(k)]*2).reshape(2,1))
            #====================================================================================================
            zname_emib = 'emib_z_'+str(k)
            assert zname_emib in self.dict_z.keys()
            z_height_l.append(self.dict_z[zname_emib])
            z_name_l.append(zname_emib)

            basic_size.append(np.array([[[self.imc_size, self.emib_size] ], [[self.r_size, self.emib_size]  ]]))


        single_emib_name = np.array(['none___']*self.N*self.emib_width_n*2*self.emib_sublayer).reshape(self.emib_sublayer,self.N*2,self.emib_width_n)
        single_emib_size = np.zeros((self.emib_sublayer, self.N*2,self.emib_width_n,2))
        single_emib_k    = np.zeros((self.emib_sublayer, self.N*2,self.emib_width_n))



        for m in range(self.N):
            for n in range(self.emib_width_n):
                row_s= m*2
                row_e= (m+1)*2
                col_s= n
                col_e= (n+1)
                for k in range(self.emib_sublayer):
                    single_emib_k[k,row_s:row_e, col_s:col_e]    = basic_k[k]
                    single_emib_name[k,row_s:row_e, col_s:col_e] = basic_n[k]
                    single_emib_size[k,row_s:row_e, col_s:col_e] = basic_size[k]


        # print(single_emib_k[0])
        # print(single_emib_name[0])
        # print(single_self.emib_size[0])
        # exit()

        return single_emib_name, single_emib_size, single_emib_k,z_height_l, z_name_l
        



    #====================================================================================================
    def load_power_tier(self,power_l,router_l, tier_shape,power_id_tocood,tsv_l=None  ):

        numofimc             = self.N*self.N
        power_ptr            = 0
        power_container      = []
        #import pdb;pdb.set_trace()
        assert len(power_l) <= self.Nstructure*self.N*self.N
        assert len(router_l) == self.Nstructure


        for i in range(self.Nstructure):
            single_tier_p = np.zeros( tier_shape )
            for m in range(self.N*self.N):
                coods  = power_id_tocood[m]

                if power_ptr>=len(power_l):
                    currp=0
                else:
                    currp  = power_l[power_ptr]/1000


                currpr = router_l[i]/(self.N*self.N)/1000
                area0  = self.imc_size*self.imc_size
                area1  = self.imc_size*self.r_size
                area2  = self.imc_size*self.r_size
                area3  = self.r_size*self.r_size
                imc_p0 = currp*area0/(area0+area1+area2)

                if tsv_l is not None:
                    currptsv = tsv_l[i]/(self.N*self.N)/1000/2


                imc_p1 = currp*area1/(area0+area1+area2)
                imc_p2 = currp*area2/(area0+area1+area2)
                # todo did not add the router power

                fullh= 0
                for h in range(self.tier_sublayer):
                    nameh = 'tier_z_'+str(h)
                    assert nameh in self.dict_z
                    fullh+=self.dict_z[nameh]

                for h in range(self.tier_sublayer):
                    nameh = 'tier_z_'+str(h)
                    if tsv_l is not None:
                        single_tier_p[h, coods[0][0], coods[0][1]]= currp*self.dict_z[nameh]/fullh
                        single_tier_p[h, coods[1][0], coods[1][1]]= currptsv*self.dict_z[nameh]/fullh
                        single_tier_p[h, coods[2][0], coods[2][1]]= currptsv*self.dict_z[nameh]/fullh
                    else:

                        single_tier_p[h, coods[0][0], coods[0][1]]= imc_p0*self.dict_z[nameh]/fullh
                        single_tier_p[h, coods[1][0], coods[1][1]]= imc_p1*self.dict_z[nameh]/fullh
                        single_tier_p[h, coods[2][0], coods[2][1]]= imc_p2*self.dict_z[nameh]/fullh
                         
                    single_tier_p[h, coods[3][0], coods[3][1]]= currpr*self.dict_z[nameh]/fullh

                power_ptr+=1

            power_container.append(single_tier_p)
        return power_container
        
    #====================================================================================================
    def load_power_aib(self,power_aib_l,aib_shape, single_aib_size):

            
        if self.Nstructure==1: numofaib = 0
        elif self.Nstructure==2: numofaib = 2
        elif self.Nstructure==3: numofaib = 4
        elif self.Nstructure==4: numofaib = 8
        else: 
            print('error')
            exit()
        assert len(power_aib_l)==numofaib
        power_ptr = 0
        power_container= []

        areas = np.zeros((aib_shape[1], aib_shape[2]))
        for m in range(aib_shape[1]):
            for n in range(aib_shape[2]):
                x = single_aib_size[0,m,n,0]
                y = single_aib_size[0,m,n,1]
                areas[m,n]=x*y

        totalarea = np.sum(areas).item()
        for i in range(numofaib):

            single_aib_p = np.zeros( aib_shape )
            if power_ptr>=len(power_aib_l):
                totalp=0
            else:
                totalp        = power_aib_l[power_ptr]/1000
            power_ptr+=1


            for m in range(aib_shape[1]):
                for n in range(aib_shape[2]):
                    currarea = areas[m,n].item()
                    currp=totalp*currarea/totalarea

                    fullh= 0
                    for h in range(aib_shape[0]):
                        nameh = 'tier_z_'+str(h)
                        assert nameh in self.dict_z
                        fullh+=self.dict_z[nameh]

                    for h in range(aib_shape[0]):
                        nameh = 'tier_z_'+str(h)
                        single_aib_p[h,m,n]=currp*self.dict_z[nameh]/fullh

            power_container.append(single_aib_p)
        return power_container









    #====================================================================================================
    def load_power_emib(self,power_emib_l,emib_shape, single_emib_size, aib_shape, single_aib_size):
        if self.Nstructure==1: numofemib = 0
        elif self.Nstructure==2: numofemib = 1
        elif self.Nstructure==3: numofemib = 2
        elif self.Nstructure==4: numofemib = 4
        else: 
            print('error')
            exit()
        assert len(power_emib_l)==numofemib

        power_ptr = 0
        power_container_center= []
        power_container_aibunder= []

        areas_center = np.zeros((emib_shape[1], emib_shape[2]))
        for m in range(emib_shape[1]):
            for n in range(emib_shape[2]):
                x = single_emib_size[0,m,n,0]
                y = single_emib_size[0,m,n,1]
                areas_center[m,n]=x*y


        areas_underaib = np.zeros((aib_shape[1], aib_shape[2]))
        for m in range(aib_shape[1]):
            for n in range(aib_shape[2]):
                x = single_aib_size[0,m,n,0]
                y = single_aib_size[0,m,n,1]
                areas_underaib[m,n]=x*y
        areas = np.concatenate([areas_underaib, areas_center, areas_underaib], axis=1)
        totalarea = np.sum(areas).item()



        #====================================================================================================
        fullh= 0
        for h in range(emib_shape[0]):
            nameh = 'emib_z_'+str(h)
            assert nameh in self.dict_z
            fullh+=self.dict_z[nameh]
        #====================================================================================================

        for i in range(numofemib):

            single_emib_p = np.zeros((emib_shape[0], areas.shape[0], areas.shape[1] )  )
            if power_ptr>=len(power_emib_l):
                totalp=0
            else:
                totalp        = power_emib_l[power_ptr]/1000
            power_ptr+=1

            for m in range(areas.shape[0]):
                for n in range(areas.shape[1]):
                    currarea = areas[m,n].item()
                    currp=totalp*currarea/totalarea

                    for h in range(emib_shape[0]):
                        nameh = 'emib_z_'+str(h)
                        single_emib_p[h,m,n]=currp*self.dict_z[nameh]/fullh


            power_underaib_l = single_emib_p[:,:,:aib_shape[2] ]
            power_underaib_r = single_emib_p[:,:,-aib_shape[2]: ]
            power_center     = single_emib_p[:,:,aib_shape[2]:-aib_shape[2] ]
            
            power_container_center.append(power_center)
            power_container_aibunder.append(power_underaib_l)
            power_container_aibunder.append(power_underaib_r)
        return power_container_center, power_container_aibunder



    #====================================================================================================





    #====================================================================================================
    def create_global_structure(self,power_tier_l,power_router_l,power_aib_l, power_emib_l,power_tsv_l=None):
        #====================================================================================================
        # tier_container_l = []
        # power_id=0
        # for i in range(self.Nstructure):
        #     previous_power_id = power_id
        #     single_tier_name, single_tieself.r_size, single_tier_k,z_height_l, z_name_l, power_id, power_id_tocood=  create_tier_structure(power_id=power_id)
        #     tier_container_l.append((single_tier_name, single_tieself.r_size,single_tier_k, z_height_l, z_name_l,previous_power_id, power_id_tocood))
        #import pdb;pdb.set_trace()
        
        single_tier_name, single_tier_size, single_tier_k,z_height_l, z_name_l, power_id, power_id_tocood = self.create_tier_structure(power_id=0)
        single_aib_name,  single_aib_size, single_aib_k, z_height_aib_l, z_name_aib_l                     = self.create_aib_structure()
        single_emib_name, single_emib_size, single_emib_k, z_height_emib_l, z_name_emib_l                 = self.create_emib_structure()
        print('Done creating basic blocks')
        #====================================================================================================
        # load_power
        #====================================================================================================
        tier_p_container_l                    = self.load_power_tier(power_tier_l,power_router_l,single_tier_k.shape, power_id_tocood,power_tsv_l)
        aib_p_container_l                     = self.load_power_aib(power_aib_l,single_aib_k.shape, single_aib_size)
        emib_p_container_l, emib_p_underaib_l = self.load_power_emib(power_emib_l,single_emib_k.shape, single_emib_size,single_aib_k.shape, single_aib_size)
        #====================================================================================================


        all_height_l = z_height_aib_l+z_height_emib_l
        all_z_name_l = ['tier']*len(z_height_aib_l)+['emib']*len(z_height_emib_l)




        pcb_n_tier = np.array(['pcb___']*self.N*self.N*2*2*self.emib_sublayer).reshape(self.emib_sublayer,self.N*2,self.N*2)
        emib_n_aib = np.array(['emib__']*self.N*2*self.aib_width_n*self.emib_sublayer).reshape(self.emib_sublayer,self.N*2,self.aib_width_n)
        air_n_emib = np.array(['air___']*self.N*2*self.emib_width_n*self.tier_sublayer).reshape(self.tier_sublayer,self.N*2,self.emib_width_n)

        pcb_k_tier = np.array([self.dict_k['pcb']]*self.N*self.N*2*2*self.emib_sublayer).reshape(self.emib_sublayer,self.N*2,self.N*2)
        emib_k_aib  = np.array([self.dict_k['k_emib_0']]*self.N*2*self.aib_width_n*self.emib_sublayer).reshape(self.emib_sublayer,self.N*2,self.aib_width_n)
        air_k_emib = np.array([self.dict_k['air']]*self.N*2*self.emib_width_n*self.tier_sublayer).reshape(self.tier_sublayer,self.N*2,self.emib_width_n)

        pcb_size_tier = np.repeat(single_tier_size[0][None,:,:,:], self.emib_sublayer,axis=0)
        self.emib_size_aib = np.repeat(single_aib_size[0][None,:,:,:], self.emib_sublayer,axis=0)
        air_size_emib = np.repeat(single_emib_size[0][None,:,:,:],self.tier_sublayer,axis=0)


        pcb_p_tier = np.array([0]*self.N*self.N*2*2*self.emib_sublayer).reshape(self.emib_sublayer,self.N*2,self.N*2)
        emib_p_aib = np.array([0]*self.N*2*self.aib_width_n*self.emib_sublayer).reshape(self.emib_sublayer,self.N*2,self.aib_width_n)
        air_p_emib = np.array([0]*self.N*2*self.emib_width_n*self.tier_sublayer).reshape(self.tier_sublayer,self.N*2,self.emib_width_n)
        #====================================================================================================
        full_k_tier   = np.vstack([single_tier_k, pcb_k_tier])
        full_k_aib    = np.vstack([single_aib_k, emib_k_aib])
        full_k_emib   = np.vstack([air_k_emib,single_emib_k])

        full_size_tier = np.vstack([single_tier_size, pcb_size_tier])
        full_size_aib  = np.vstack([single_aib_size, self.emib_size_aib])
        full_size_emib = np.vstack([air_size_emib,single_emib_size])
        #====================================================================================================
        def transpose_size(input):
            assert len(input.shape)==4
            tmpnew = np.transpose(input, (0,2,1,3))
            for m in range(tmpnew.shape[0]):
                for n in range(tmpnew.shape[1]):
                    for k in range(tmpnew.shape[2]):
                        pair = tmpnew[m,n,k].copy()
                        tmpnew[m,n,k,0] = pair[1]
                        tmpnew[m,n,k,1] = pair[0]
            return tmpnew

        def transpose_k(input):
            assert len(input.shape)==3
            tmpnew = np.transpose(input, (0,2,1))
            return tmpnew
        #====================================================================================================
        full_size_aib_t  = transpose_size(full_size_aib.copy())
        full_size_emib_t = transpose_size(full_size_emib.copy())
        full_k_aib_t     = transpose_k(full_k_aib.copy())
        full_k_emib_t    = transpose_k(full_k_emib.copy())
        #====================================================================================================
        row0 = np.concatenate([full_size_tier, full_size_aib, full_size_emib, full_size_aib, full_size_tier ],axis=2)
        row0_k = np.concatenate([full_k_tier, full_k_aib, full_k_emib, full_k_aib, full_k_tier ],axis=2)


        if self.Nstructure==1:
            row0 = np.concatenate([full_size_tier],axis=2)
            row0_k = np.concatenate([full_k_tier],axis=2)
            col0   = np.concatenate([full_size_tier],axis=1)
            col0_k = np.concatenate([full_k_tier],axis=1)
        elif self.Nstructure==2:
            col0   = np.concatenate([full_size_tier],axis=1)
            col0_k = np.concatenate([full_k_tier],axis=1)
        else :
            col0   = np.concatenate([full_size_tier, full_size_aib_t, full_size_emib_t, full_size_aib_t, full_size_tier ],axis=1)
            col0_k = np.concatenate([full_k_tier, full_k_aib_t, full_k_emib_t, full_k_aib_t, full_k_tier ],axis=1)
        #====================================================================================================
        numofnodesx = col0.shape[1]
        numofnodesy = row0.shape[2]

        full_k            = np.vstack([np.ones((self.tier_sublayer, numofnodesx, numofnodesy ))*self.dict_k['air'], np.ones((self.emib_sublayer, numofnodesx, numofnodesy ))*self.dict_k['pcb']])
        full_k[:,:2*self.N,:]  = row0_k
        full_k[:,:,-2*self.N:] = col0_k
        if self.Nstructure==4:
            full_k[:,-2*self.N:,:] = row0_k
            full_k[:,:,:2*self.N]  = col0_k

        grid_size = np.zeros((numofnodesx, numofnodesy,2))
        for m in range(numofnodesx):
            for n in range(numofnodesy):
                cellx = col0[0,m,0,0]
                celly = row0[0,0,n,1]
                grid_size[m,n,0]=cellx
                grid_size[m,n,1]=celly

        #====================================================================================================
        full_p = np.zeros(full_k.shape)

        if self.Nstructure==1:
            row0_p_top = np.concatenate([tier_p_container_l[0]],axis=2)
            row0_p_bot = np.concatenate([pcb_p_tier           ],axis=2)
            full_p[:self.tier_sublayer,:,:] = row0_p_top
            full_p[self.tier_sublayer:,:,:] = row0_p_bot

        else:
            row0_p_top = np.concatenate([tier_p_container_l[0], aib_p_container_l[0], air_p_emib, aib_p_container_l[1], tier_p_container_l[1] ],axis=2)
            row0_p_bot = np.concatenate([pcb_p_tier           , emib_p_underaib_l[0], emib_p_container_l[0], emib_p_underaib_l[1], pcb_p_tier ],axis=2)
             

            if self.Nstructure==2:
                full_p[:self.tier_sublayer,:,:] = row0_p_top
                full_p[self.tier_sublayer:,:,:] = row0_p_bot

            if self.Nstructure>=3:
                full_p[:self.tier_sublayer,:row0_p_top.shape[1],:] = row0_p_top
                full_p[self.tier_sublayer:,:row0_p_top.shape[1],:] = row0_p_bot
                col0_p_top = np.concatenate([tier_p_container_l[1], transpose_k(aib_p_container_l[2]), transpose_k(air_p_emib), transpose_k(aib_p_container_l[3]), tier_p_container_l[2] ],axis=1)
                col0_p_bot = np.concatenate([pcb_p_tier, transpose_k(emib_p_underaib_l[2]), transpose_k(emib_p_container_l[1]), transpose_k(emib_p_underaib_l[3]), pcb_p_tier ],axis=1)
                full_p[:self.tier_sublayer,:,-col0_p_top.shape[2]:] = col0_p_top
                full_p[self.tier_sublayer:,:,-col0_p_top.shape[2]:] = col0_p_bot

            if self.Nstructure==4:
                row1_p_top = np.concatenate([tier_p_container_l[3], aib_p_container_l[5], air_p_emib, aib_p_container_l[4], tier_p_container_l[2] ],axis=2)
                row1_p_bot = np.concatenate([pcb_p_tier           , emib_p_underaib_l[5], emib_p_container_l[2], emib_p_underaib_l[4], pcb_p_tier ],axis=2)
                full_p[:self.tier_sublayer,-row1_p_top.shape[1]:,:] = row1_p_top
                full_p[self.tier_sublayer:,-row1_p_top.shape[1]:,:] = row1_p_bot

                col1_p_top = np.concatenate([tier_p_container_l[0], transpose_k(aib_p_container_l[7]), transpose_k(air_p_emib), transpose_k(aib_p_container_l[6]), tier_p_container_l[3] ],axis=1)
                col1_p_bot = np.concatenate([pcb_p_tier, transpose_k(emib_p_underaib_l[7]), transpose_k(emib_p_container_l[3]), transpose_k(emib_p_underaib_l[6]), pcb_p_tier ],axis=1)
                full_p[:self.tier_sublayer,:,:col1_p_top.shape[2]] = col1_p_top
                full_p[self.tier_sublayer:,:,:col1_p_top.shape[2]] = col1_p_bot

        #====================================================================================================
        full_grid_size = np.repeat(grid_size[None,:,:,:], self.emib_sublayer+self.tier_sublayer,axis=0)
        #====================================================================================================
        # print(grid_size.shape)
        # print(full_k.shape)
        # print(full_p.shape)
        # print(full_grid_size.shape)
        #====================================================================================================

        all_height_l = z_height_aib_l+z_height_emib_l
        all_z_name_l = ['tier']*len(z_height_aib_l)+['emib']*len(z_height_emib_l)

        heat_height_l = [self.heatsinkair_resoluation]*int((self.dict_z['heatsink']+self.dict_z['heatspread'])//self.heatsinkair_resoluation)
        pcb_height_l  = [self.heatsinkair_resoluation]*int((self.dict_z['pcb']))
        air_height_l  = [self.heatsinkair_resoluation]*int((self.dict_z['air'])//self.heatsinkair_resoluation)

        heat_name_l = ['heatsink']*int((self.dict_z['heatsink']+self.dict_z['heatspread'])//self.heatsinkair_resoluation)
        pcb_name_l  = ['pcb']*int((self.dict_z['pcb']))
        air_name_l  = ['air']*int((self.dict_z['air'])//self.heatsinkair_resoluation)

        all_height_l = heat_height_l+all_height_l+pcb_height_l+air_height_l
        all_z_name_l = heat_name_l+all_z_name_l+pcb_name_l+air_name_l
        all_z_count_l = [len(heat_height_l), self.emib_sublayer+self.tier_sublayer, len(pcb_height_l),len(air_height_l) ]
        assert len(all_height_l)==len(all_z_name_l)
        #====================================================================================================
        heatspread_k = np.ones((len(heat_height_l), full_k.shape[1], full_k.shape[2] ))*self.dict_k['cu']
        pcb_k        = np.ones((len(pcb_height_l), full_k.shape[1], full_k.shape[2] ))*self.dict_k['pcb']
        air_k        = np.ones((len(air_height_l), full_k.shape[1], full_k.shape[2] ))*self.dict_k['air']

        heatspread_p = np.zeros((len(heat_height_l), full_k.shape[1], full_k.shape[2] ))
        pcb_p        = np.zeros((len(pcb_height_l), full_k.shape[1], full_k.shape[2] ))
        air_p        = np.zeros((len(air_height_l), full_k.shape[1], full_k.shape[2] ))
        #====================================================================================================
        full_k = np.vstack([heatspread_k,full_k,pcb_k,air_k  ])
        full_p = np.vstack([heatspread_p,full_p,pcb_p,air_p  ])
        #====================================================================================================
        assert full_k.shape[0]==len(all_height_l)
        assert full_p.shape[0]==len(all_height_l)
        #====================================================================================================
        return full_k, full_p, grid_size, all_height_l, all_z_count_l






        

        





    def get_conductance_G_new(self,full_k, grid_size, all_height_l, all_z_count_l):
    # def get_conductance_G_new(cube_geo_dict, cube_k_dict, cube_z_dict):


        direction_map = dict()
        direction_map[0]=(0,1,0)  #east,0
        direction_map[1]=(0,-1,0) #west,1
        direction_map[2]=(1,0,0)  #north,2
        direction_map[3]=(-1,0,0) #south,3
        direction_map[4]=(0,0,1)  #top,4
        direction_map[5]=(0,0,-1) #bottom,5


        k_info_np = full_k
        z_info_np = all_height_l

        # assert geo_info_np.shape==k_info_np.shape
        numtotalnode =(k_info_np.shape[0]+2)*k_info_np.shape[1]*k_info_np.shape[2]
        numoflayer = k_info_np.shape[0]
        numofnodex = k_info_np.shape[1]
        numofnodey = k_info_np.shape[2]

        #====================================================================================================
        G_sparse = sparse_mat.dok_matrix((numtotalnode, numtotalnode))
        #====================================================================================================
        nodeid = 0
        for z_idx in range(1,numoflayer+1):
            for x_idx in range(numofnodex):
                for y_idx in range(numofnodey):



                    # dim_str = geo_info_np[z_idx-1, x_idx, y_idx].split(',')

                    dim_str                             = [ grid_size[x_idx, y_idx,0],   grid_size[x_idx, y_idx,1]   ]
                    centernode_width, centernode_length = (float)(dim_str[0]),(float)(dim_str[1])
                    centerlayer_height                  = z_info_np[z_idx-1]

                    centernodeid = self.id_(x_idx, y_idx, z_idx, numofnodex, numofnodey,numoflayer )
                    center_k = k_info_np[z_idx-1, x_idx, y_idx].item()




                    #====================================================================================================
                    tmp =1
                    if z_idx== numoflayer:
                        neighbour_id_downbelow = self.id_(x_idx, y_idx, z_idx+1, numofnodex, numofnodey,numoflayer )
                        G_sparse[centernodeid, neighbour_id_downbelow ]=tmp
                        G_sparse[neighbour_id_downbelow, centernodeid ]=tmp

                    if z_idx== 1:
                        neighbour_id_downbelow = self.id_(x_idx, y_idx, z_idx-1, numofnodex, numofnodey,numoflayer )
                        G_sparse[centernodeid, neighbour_id_downbelow ]=tmp
                        G_sparse[neighbour_id_downbelow, centernodeid ]=tmp
                    #====================================================================================================




                    G_sparse[centernodeid, centernodeid]=0
                    #====================================================================================================
                    # check all neighbours and calculate conductance 
                    #====================================================================================================
                    for direction in range(6):
                        neighbour_x_idx= (direction_map[direction][0]+x_idx)
                        neighbour_y_idx= (direction_map[direction][1]+y_idx)
                        neighbour_z_idx= (direction_map[direction][2]+z_idx)
                        if self.checkneighbours(neighbour_x_idx,neighbour_y_idx,neighbour_z_idx-1,numofnodex, numofnodey,numoflayer):
                            neighbour_id = self.id_(neighbour_x_idx, neighbour_y_idx,neighbour_z_idx, numofnodex, numofnodey,numoflayer)
                            #====================================================================================================
                            # calcualte the avg k
                            #====================================================================================================
                            neighbour_k =   k_info_np[neighbour_z_idx-1,neighbour_x_idx,neighbour_y_idx]
                            # dim_str     = geo_info_np[neighbour_z_idx-1,neighbour_x_idx,neighbour_y_idx].split(',')

                            dim_str_neigh                             = [ grid_size[neighbour_x_idx, neighbour_y_idx,0],   grid_size[neighbour_x_idx, neighbour_y_idx,1]   ]
                            neighbour_node_width, neighbour_node_length = (float)(dim_str_neigh[0]),(float)(dim_str_neigh[1])   

                            # print(dim_str_neigh, neighbour_x_idx, neighbour_y_idx)

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
        print('INFO:Done generating G ')


        return G_sparse



    
    def subdivide(self, full_k, full_p, grid_size):
        if self.resolution ==1:
            return full_k, full_p ,grid_size

        numz = full_k.shape[0]
        numx = full_k.shape[1]
        numy = full_k.shape[2]

        newfull_k = np.zeros((numz,  numx*self.resolution, numy*self.resolution))
        newfull_p = np.zeros((numz,  numx*self.resolution, numy*self.resolution))
        newgrid_size = np.zeros(( numx*self.resolution, numy*self.resolution,2))

        for z in range(numz):
            for x in range(numx):
                for y in range(numy):
                    sx = x*self.resolution
                    sy = y*self.resolution

                    fullk = full_k[z, x, y]
                    fullp = full_p[z, x, y]
                    sizex  = grid_size[x,y,0]
                    sizey  = grid_size[x,y,1]

                    subp = fullp/(self.resolution*self.resolution)
                    sub_x = sizex/(self.resolution)
                    sub_y = sizey/(self.resolution)

                    for subx in range(self.resolution):
                        for suby in range(self.resolution):
                            coodx = sx+subx
                            coody = sy+suby
                            newfull_k[z,coodx,coody ]= fullk
                            newfull_p[z,coodx,coody ]= subp
                            newgrid_size[coodx, coody, 0 ] = sub_x
                            newgrid_size[coodx, coody, 1 ] = sub_y

        return newfull_k, newfull_p, newgrid_size




    def subdivide_for_mape(self, inputt, subdivide ):
        numz = inputt.shape[0]
        numx = inputt.shape[1]
        numy = inputt.shape[2]

        newfull_x = np.zeros((numz,  numx*subdivide, numy*subdivide))

        for z in range(numz):
            for x in range(numx):
                for y in range(numy):
                    sx = x*subdivide
                    sy = y*subdivide

                    t = inputt[z, x, y]
                    for subx in range(subdivide):
                        for suby in range(subdivide):
                            coodx = sx+subx
                            coody = sy+suby
                            newfull_x[z,coodx,coody ]=t 
        return newfull_x  









    def convert2realratio(self,t, grid_size):

        #====================================================================================================
        outt = None
        for layer in range(len(t)):
            currlayer = None
            for row in range(t.shape[1]):
                currrow = None
                for col in range(t.shape[2]):

                    value = t[layer,row,col]
                    # celltype = namemap[layer,row,col]
                    boxx = grid_size[row,col,0]
                    boxy = grid_size[row,col,1]
                    #====================================================================================================
                    numofwidthblock = 0
                    numoflengthblock = 0

                    if   boxx==self.imc_size/self.resolution and boxy==self.imc_size/self.resolution:   numofwidthblock = 2 ; numoflengthblock = 2
                    elif boxx==self.imc_size/self.resolution and boxy==self.r_size/self.resolution:     numofwidthblock = 2 ; numoflengthblock = 1
                    elif boxx==self.r_size/self.resolution and boxy==self.imc_size/self.resolution:     numofwidthblock = 1 ; numoflengthblock = 2
                    elif boxx==self.r_size/self.resolution and boxy==self.r_size/self.resolution:       numofwidthblock = 1 ; numoflengthblock = 1
                    # elif boxx==self.imc_size and boxy==self.aib_size:   numofwidthblock = 2 ; numoflengthblock = 2



                    # if row %2==0:
                    #     numofwidthblock=2
                    #     # two block height per row, 
                    #     if celltype=='imc0' or celltype=='imc1' or celltype=='imc2':
                    #         # two block length per col
                    #         numoflengthblock = 2
                    #     else:
                    #         # one  block length per col
                    #         numoflengthblock = 1
                    # else:
                    #     numofwidthblock=1
                    #     # one block height per row, 
                    #     if celltype=='tsv':
                    #         # two block length per col
                    #         numoflengthblock = 2
                    #     else:
                    #         # one  block length per col
                    #         numoflengthblock = 1

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




        

    def solver(self,G_sparse, full_p,  all_z_count_l,grid_size):

        # namemap = cube_n_dict[design]


        p = full_p
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

        print('Starting solving...')
        start_time = time.time()


    
        #====================================================================================================
        # G_sparse_inv = inv(G_sparse)
        # # A = some_sparse_matrix #(scipy.sparse.csr_matrix)
        # # x = some_dense_vector  #(numpy.ndarray)
        # print('Sent to GPU...')
        # G_sparse_inv_gpu = csr_gpu(G_sparse_inv)  #moving A to the gpu
        # p_gpu = cp.array(p)  #moving A to the gpu


        # print('Sent to GPU...')
        # t_gpu = G_sparse_inv_gpu.dot(p_gpu)

        # print('Transfer back to CPU...')
        # t = cp.asnumpy(t_gpu) #back to numpy object for fast indexing

        # # print('here')
        # print(t.shape)
        # exit()
        #====================================================================================================



        t = sparse_algebra.spsolve(G_sparse,p,permc_spec=None,use_umfpack=True)
        print("Running time --- %s seconds ---" % (time.time() - start_time))

        #====================================================================================================
        # t = t.reshape(numoflayer+1,numofnodex,numofnodey)
        # t = t[:numoflayer, :,:]
        #====================================================================================================
        t = t.reshape(numoflayer+2,numofnodex,numofnodey)
        t = t[1:numoflayer+1, :,:]
        #====================================================================================================


        # layertype_l = cube_layertype_dict[design]
        t = t[all_z_count_l[0]:all_z_count_l[0]+all_z_count_l[1],:,:]
        assert t.shape[0]==self.tier_sublayer+self.emib_sublayer
        # namemap = namemap[devicestart:deviceend,:,:]
        realratiot = self.convert2realratio(t, grid_size)



        vmin  = t.min()
        vmax  = t.max()
        print('Resolution:{}    Min t: '.format(self.resolution), round(vmin, 2),  '\t\tPeak t:', round(vmax,2),'\t\tAverage t:', round(t.mean(),2 ))
        # vmin  = 298.5
        # vmin =298
        # vmax =298+3
        for i in range(len(t)):
            # layername = layertype_l[i]+' '+str(i)
            self.plot_im(plot_data=realratiot[i,:,:], title='resolution {}'.format(i), save_name='./Results/result_thermal/thermal_map_res{}_{}.png'.format(self.resolution, i), vmin=vmin, vmax=vmax)

        return realratiot
    
        # return vmax


def power_tile_reorg(tiles_edges_in_tier):
    power_inform = "./Debug/to_interconnect_analy/layer_performance.csv"
    power_inform = pd.read_csv(power_inform, header=None)
    power_inform = power_inform.to_numpy()

    computing_inform = "./Debug/to_interconnect_analy/layer_inform.csv"
    computing_data = pd.read_csv(computing_inform, header=None)
    computing_data = computing_data.to_numpy()
    power_l=[]
    

    for i in range(len(computing_data)-1):

        for y in range(int(computing_data[i][1])):
            power_l.append(float(power_inform[i][5]))
        if (computing_data[i][9]+1)*(tiles_edges_in_tier)*(tiles_edges_in_tier)-computing_data[i][7]!=0 and (computing_data[i][9]+1)*(tiles_edges_in_tier)*(tiles_edges_in_tier)-computing_data[i][7]<computing_data[i+1][1]:
            for c in range(int((computing_data[i][9]+1)*(tiles_edges_in_tier)*(tiles_edges_in_tier)-computing_data[i][7])):
                power_l.append(0)
    for i in range(int(computing_data[-1][1])):
        power_l.append(float(power_inform[-1][5]))
    if (computing_data[-1][9]+1)*(tiles_edges_in_tier)*(tiles_edges_in_tier)-computing_data[-1][7]!=0:
        for c in range (int(((computing_data[-1][9]+1)*(tiles_edges_in_tier)*(tiles_edges_in_tier)-computing_data[-1][7]))):
            power_l.append(0)
    return power_l
    #import pdb;pdb.set_trace()

#====================================================================================================
##if self.Nstructure   == 2: numofaib = 2;  numofemib = 1
#elif self.Nstructure == 3: numofaib = 4;  numofemib = 2
#elif self.Nstructure == 4: numofaib = 8;  numofemib = 4
#====================================================================================================


# power_tier_l   = np.arange(self.Nstructure*N*N)+10
# power_router_l = np.arange(self.Nstructure)+5
# power_aib_l    = np.arange(numofaib)+100
# power_emib_l   = np.arange(numofemib)+200

# power_tier_l   = power_tier_l.tolist()
# power_router_l = power_router_l.tolist()
# power_aib_l    = power_aib_l.tolist()
# power_emib_l   = power_emib_l.tolist()

"""
Nstructure =2
resolution=2
resolution_l = [8,6,4,2,1]
resolution_l = [1]
# resolution_l = [4,2,1]
case_H2_5D=H2_5D(area_single_tile=0.336536,single_router_area=0.019626,chiplet_num=Nstructure,mesh_edge=4,resolution=resolution)
#====================================================================================================
if case_H2_5D.Nstructure   == 2: numofaib = 2;  numofemib = 1
elif case_H2_5D.Nstructure == 3: numofaib = 4;  numofemib = 2
elif case_H2_5D.Nstructure == 4: numofaib = 8;  numofemib = 4
elif case_H2_5D.Nstructure == 1: numofaib = 0;  numofemib = 0

#====================================================================================================
power_tier_l   = [20]*case_H2_5D.Nstructure*case_H2_5D.N*case_H2_5D.N
# power_tier_l[0]=20
# power_tier_l[1]=20
# power_tier_l[6]=20
# power_tier_l[7]=20

power_router_l = [5]*case_H2_5D.Nstructure
power_aib_l    = [100]*numofaib
power_emib_l   = [1]*numofemib
#====================================================================================================
# power_tsv_l   = [2.5*case_H2_5D.N*case_H2_5D.N*2]*case_H2_5D.Nstructure
# power_router_l = [5*case_H2_5D.N*case_H2_5D.N]*case_H2_5D.Nstructure
# power_tsv_l   = None
#====================================================================================================


case_H2_5D.input_sanity_check()
full_k, full_p, grid_size, all_height_l, all_z_count_l= case_H2_5D.create_global_structure(power_tier_l,power_router_l,power_aib_l,power_emib_l)
currfull_k, currfull_p, curr_grid_size = case_H2_5D.subdivide(full_k, full_p, grid_size)
G_sparse = case_H2_5D.get_conductance_G_new(currfull_k, curr_grid_size, all_height_l, all_z_count_l)
t = case_H2_5D.solver(G_sparse, currfull_p,  all_z_count_l,curr_grid_size)
"""





