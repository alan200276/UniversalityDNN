#/bin/python3

import uproot
import pyjet
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import importlib
import time
import re

from BranchClass import *

import Event_List 
import jet_trimming 
import JSS 

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
from matplotlib import cm


###################################################################################
"""
Input Check and Setting
"""
###################################################################################
print("Input Check and Setting")

try:
    data_path = str(sys.argv[1])
    
    index = int(sys.argv[2])
    
    MCData = uproot.open(data_path)["Delphes;1"]
    
    findlast = []

    for m in re.finditer("/",data_path):
        findlast.append(m.start())

    l = findlast[-1]    
    
    print("File is loaded!")
    print("Generator (GEN) : {}".format(data_path[l+6:l+12]))
    print("Showering (SHO) : {}".format(data_path[l+13:l+16]))
    print("Process (PRO) : {}".format(data_path[l+1:l+5]))
    
    if data_path[l+6:l+12] == str("pythia"):
        GEN = str("pythia")
    elif data_path[l+6:l+12] == str("sherpa"):
        GEN = str("sherpa")
    elif data_path[l+6:l+12] == str("herwig"):
        GEN = str("herwig")
        
    if data_path[l+13:l+16] == str("def"):
        SHO = str("def")
    elif data_path[l+13:l+16] == str("dip"):
        SHO = str("dip")
    elif data_path[l+13:l+16] == str("ang"):
        SHO = str("ang")
    elif data_path[l+13:l+16] == str("vin"):
        SHO = str("vin")
        
    if data_path[l+1:l+5] == str("ppwz"):
        PRO = str("W")
    elif data_path[l+1:l+5] == str("ppjj"):
        PRO = str("QCD")
    
except:
    print("********* Please Check Input Argunment *********")
    print("********* Usage: python3 preprocess.py <path-of-file>/XXXX.root index *********")
    sys.exit(1)
    

    
###################################################################################
"""
Read Data and Jet Clustering 
"""
###################################################################################

print("Read Data and Jet Clustering ")
print("\n")

GenParticle = BrachGenParticles(MCData)
EventList = Event_List.Event_List(GenParticle)



event_list_clustered = []

print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
ticks_1 = time.time()

for j in range(len(EventList)):
    to_cluster = np.core.records.fromarrays(EventList[j], 
                                             names="pT, eta, phi, mass, PID, Status",
                                             formats = "f8, f8, f8, f8, f8, f8")
    pt_min = 300
    sequence_cluster = pyjet.cluster(to_cluster, R = 1.2, p = -1) # p = -1: anti-kt , 0: Cambridge-Aachen(C/A), 1: kt
    jets_cluster = sequence_cluster.inclusive_jets(pt_min)
    event_list_clustered.append(jets_cluster)

#     if j == 5000:
#         break

ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
print("\n")

print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
ticks_1 = time.time()


    
###################################################################################
"""
Read Data
"""
###################################################################################

print("Read Data")
print("\n")

M_J = []
M_J_trimmed = []
T21 = []
T21_trimmed = []
D21, D22, C21, C22 = [], [], [], []
D21_trimmed, D22_trimmed, C21_trimmed, C22_trimmed = [], [], [], []


###################################################################################
"""
Create Pandas DataFrame
"""
###################################################################################

print("Create Pandas DataFrame")
print("\n")

HOMEPATH = "/home/u5/Universality/"
path =  HOMEPATH + "Data_High_Level_Features/"


for j, filename in enumerate(os.listdir(path)):
    if filename == str(GEN) + "_" + str(SHO) + "_" + str(PRO) + ".csv":
        index = 1 

if index == 0:
    dataframe = pd.DataFrame()
if index == 1:
    dataframe = pd.DataFrame()
    save_to_csvdata = pd.read_csv(path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + ".csv")

###################################################################################
    
print("Selection and Trimming")
print("\n")    
    
features = ["GEN","SHO","PRO","MJ","PTJ","t12","D21","D22","C21","C22","eventindex"]


k = 0
for N in range(len(event_list_clustered)):
    if len(event_list_clustered[N]) >= 1: # at least two jets in this event.
        jet_1 = event_list_clustered[N][0] #leading jet's information

#         if jet_1.pt < 300 or jet_1.pt > 400: 
#             continue
#         M_J[i].append(jet_1.mass)

#         t1 = tn(jet_1, n=1)
#         t2 = tn(jet_1, n=2)
#         t21 = t2 / t1 if t1 > 0.0 else 0.0

#         ee2 = CalcEECorr(jet_1, n=2, beta=1.0)
#         ee3 = CalcEECorr(jet_1, n=3, beta=1.0)
#         d21 = ee3/(ee2**3) if ee2>0 else 0
#         d22 = ee3**2/((ee2**2)**3) if ee2>0 else 0
#         c21 = ee3/(ee2**2) if ee2>0 else 0
#         c22 = ee3**2/((ee2**2)**2) if ee2>0 else 0

#         T21[i].append(t21)
#         D21[i].append(d21)
#         D22[i].append(d22)
#         C21[i].append(c21)
#         C22[i].append(c22)


        jet_1 = jet_trimming.jet_trim(jet_1)[0]  #trimming jet's information

        if jet_1.pt < 300 or jet_1.pt > 500: 
            continue


        t1 = JSS.tn(jet_1, n=1)
        t2 = JSS.tn(jet_1, n=2)
        t21 = t2 / t1 if t1 > 0.0 else 0.0

        ee2 = JSS.CalcEECorr(jet_1, n=2, beta=1.0)
        ee3 = JSS.CalcEECorr(jet_1, n=3, beta=1.0)
        d21 = ee3/(ee2**3) if ee2>0 else 0
        d22 = ee3**2/((ee2**2)**3) if ee2>0 else 0
        c21 = ee3/(ee2**2) if ee2>0 else 0
        c22 = ee3**2/((ee2**2)**2) if ee2>0 else 0


        M_J_trimmed.append(jet_1.mass)
        T21_trimmed.append(t21)
        D21_trimmed.append(d21)
        D22_trimmed.append(d22)
        C21_trimmed.append(c21)
        C22_trimmed.append(c22)

        var = []
    
        var.append(GEN)
        var.append(SHO)
        var.append(PRO)
        var.append(jet_1.mass)
        var.append(jet_1.pt)
        var.append(t21)
        var.append(d21)
        var.append(d22)
        var.append(c21)
        var.append(c22)

        var.append(k)

        dataframe_tmp = pd.DataFrame([var],columns=features)
        dataframe = dataframe.append(dataframe_tmp, ignore_index = True)

        k += 1

        if k >= 10:
            break
            


if index == 0:
    dataframe.to_csv( path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + ".csv", index = 0)
elif index == 1:
    DATA = pd.concat([save_to_csvdata, dataframe], ignore_index=True, axis=0,join='inner')
    DATA.to_csv(path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + ".csv", index = 0)

ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))

