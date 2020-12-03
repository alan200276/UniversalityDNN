#/bin/python3

import time
import pyjet
import numpy as np

from BranchClass import *

def Event_List(GenParticle):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    ticks_1 = time.time()
    
    k = 0
    event_list = []
    
    for N in range(GenParticle.length):
        
        event_list_tmp = []
        event_list_tmp_tmp = []


        event_list_tmp.append([GenParticle.PT_At(N),  GenParticle.Eta_At(N), \
                               GenParticle.Phi_At(N), GenParticle.Mass_At(N), \
                               GenParticle.PID_At(N), GenParticle.Status_At(N)
                              ])

        event_list_tmp = np.array(event_list_tmp)
        event_list_tmp = event_list_tmp[0][:,event_list_tmp[0][5,:]==1]
        event_list_tmp = event_list_tmp[:,np.abs(event_list_tmp[4,:])!=12]
        event_list_tmp = event_list_tmp[:,np.abs(event_list_tmp[4,:])!=14]
        event_list_tmp = event_list_tmp[:,np.abs(event_list_tmp[4,:])!=16]
        
        event_list.append(event_list_tmp)

    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
    
    return event_list