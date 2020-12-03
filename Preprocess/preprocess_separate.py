#!/bin/python3

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


# Returns the difference in phi between phi, and phi_center
# as a float between (-PI, PI)
def dphi(phi,phi_c):

    dphi_temp = phi - phi_c
    while dphi_temp > np.pi:
        dphi_temp = dphi_temp - 2*np.pi
    while dphi_temp < -np.pi:
        dphi_temp = dphi_temp + 2*np.pi
    return (dphi_temp)


###################################################################################
"""
Input Check and Setting
"""
###################################################################################
print("Input Check and Setting")

try:
    data_path = str(sys.argv[1])
    
    index = int(sys.argv[2])
    
    file_number = int(sys.argv[3])
    
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

GenParticle = BranchGenParticles(MCData)
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
imagespath =  HOMEPATH + "Jet_Images/"


for j, filename in enumerate(os.listdir(path)):
    if filename == str(GEN) + "_" + str(SHO) + "_" + str(PRO) + "_" + str(file_number) + "_trimmed.csv":
        index = 1 

if index == 0:
    dataframe = pd.DataFrame()
if index == 1:
    dataframe = pd.DataFrame()
    save_to_csvdata = pd.read_csv(path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + "_" + str(file_number) + "_trimmed.csv")

###################################################################################
    
print("Selection and Trimming")
print("\n")    
print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
ticks_1 = time.time()
    
features = ["GEN","SHO","PRO","MJ_0","PTJ_0","t21_0","D21_0","D22_0","C21_0","C22_0","MJ","PTJ","t21","D21","D22","C21","C22","eventindex"]

# features = ["GEN","SHO","PRO","MJ_0","PTJ_0","t21_0","D21_0","D22_0","C21_0","C22_0","eventindex"]

trimmed_jet = []
k = 0
for N in range(len(event_list_clustered)):
    if len(event_list_clustered[N]) >= 1: # at least two jets in this event.
        jet_1 = event_list_clustered[N][0] #leading jet's information
        
        var = []
        
        if jet_1.pt < 300 or jet_1.pt > 500: 
            continue
            
        M_J.append(jet_1.mass)

        t1 = JSS.tn(jet_1, n=1)
        t2 = JSS.tn(jet_1, n=2)
        t21 = t2 / t1 if t1 > 0.0 else 0.0

        ee2 = JSS.CalcEECorr(jet_1, n=2, beta=1.0)
        ee3 = JSS.CalcEECorr(jet_1, n=3, beta=1.0)
        d21 = ee3/(ee2**3) if ee2>0 else 0
        d22 = ee3**2/((ee2**2)**3) if ee2>0 else 0
        c21 = ee3/(ee2**2) if ee2>0 else 0
        c22 = ee3**2/((ee2**2)**2) if ee2>0 else 0

        T21.append(t21)
        D21.append(d21)
        D22.append(d22)
        C21.append(c21)
        C22.append(c22)

        
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
        
        
#         trimmed_jet.append(jet_1)
        
        ## Jet Trimming
        jet_1 = jet_trimming.jet_trim(jet_1)[0]  #trimming jet's information

        if jet_1.pt < 300 or jet_1.pt > 500: 
            continue
            
        trimmed_jet.append(jet_1)

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

#         if k >= 10000:
#             break
            

print("There are {} jets.".format(len(dataframe)))
    
if index == 0:
    dataframe.to_csv( path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + "_" + str(file_number) + "_trimmed_.csv", index = 0)
elif index == 1:
    DATA = pd.concat([save_to_csvdata, dataframe], ignore_index=True, axis=0,join='inner')
    DATA.to_csv(path + str(GEN) + "_" + str(SHO) + "_" + str(PRO) + "_" + str(file_number) + "_trimmed.csv", index = 0)

ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))


###################################################################################
print("Make Jet Images")
print("\n")    
print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
ticks_1 = time.time()

jetimage_list = []

for i, jet in enumerate(trimmed_jet):
    
    width,height = 40,40
    image = np.zeros((width,height))  
    isReflection = 1
    x_hat = np.array([1,0]) 
    y_hat = np.array([0,1])
    
    subjets = pyjet.cluster(jet.constituents_array(), R=0.2, p=1)
    subjet_array = subjets.inclusive_jets()
    
        
    if len(subjet_array) > 1:
            #First, let's find the direction of the second-hardest jet relative to the first-hardest jet
            phi_dir = -(dphi(subjet_array[1].phi,subjet_array[0].phi))
            eta_dir = -(subjet_array[1].eta - subjet_array[0].eta)
            #Norm difference:
            norm_dir = np.linalg.norm([phi_dir,eta_dir])
            #This is now the y-hat direction. so we can actually find the unit vector:
            y_hat = np.divide([phi_dir,eta_dir],np.linalg.norm([phi_dir,eta_dir]))
            #and we can find the x_hat direction as well
            x_hat = np.array([y_hat[1],-y_hat[0]]) 
    
    if len(subjet_array) > 2:
        phi_dir_3 = -(dphi(subjet_array[2].phi,subjet_array[0].phi))
        eta_dir_3 = -(subjet_array[2].eta - subjet_array[0].eta)

        isReflection = np.cross(np.array([phi_dir,eta_dir,0]),np.array([phi_dir_3,eta_dir_3,0]))[2]
        
            

    R = 1.2
    for constituent in jet:
        if (len(subjet_array) == 1):
            #In the case that the reclustering only found one hard jet (that seems kind of bad, but hey)
            #no_two = no_two+1
            new_coord = [dphi(constituent.phi,subjet_array[0].phi),constituent.eta-subjet_array[0].eta]
            indxs = [math.floor(width*new_coord[0]/R)+width//2,math.floor(height*(new_coord[1])/(R*1.5))+height//2]
            
        else:
            #Now, we want to express an incoming particle in this new basis:
            part_coord = [dphi(constituent.phi,subjet_array[0].phi),constituent.eta-subjet_array[0].eta]
            new_coord = np.dot(np.array([x_hat,y_hat]),part_coord)
            
            #put third-leading subjet on the right-hand side
            if isReflection < 0: 
                new_coord = [-new_coord[0],new_coord[1]]
            elif isReflection > 0:
                new_coord = [new_coord[0],new_coord[1]]
            #Now, we want to cast these new coordinates into our array
            #(row,column)
    #         indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2,math.floor(height*(new_coord[1]+norm_dir/1.5)/(R*1.5))+height//2]
    #         indxs = [math.floor(width*new_coord[0]/(R*1.5))+width//2,math.floor(height*new_coord[1]/(R*1.5))+height//2] #(phi,eta) and the leading subjet at the origin
#             indxs = [math.floor(height*new_coord[1]/(R*1.5))+height//2,math.floor(width*new_coord[0]/(R*1.5))+width//2] #(eta,phi) and the leading subjet at the origin
            indxs = [math.floor(height*new_coord[1]/R)+height//2,math.floor(width*new_coord[0]/R)+width//2] #(eta,phi) and the leading subjet at the origin


        if indxs[0] >= width or indxs[1] >= height or indxs[0] <= 0 or indxs[1] <= 0:
            continue
        phi_index = int(indxs[0]); eta_index = int(indxs[1])

        #finally, lets fill

        image[phi_index,eta_index] = image[phi_index,eta_index] + constituent.e/np.cosh(constituent.eta)
#         image[phi_index,eta_index] = image[phi_index,eta_index] + constituent.pt

    image = np.divide(image,np.sqrt(np.sum(image*image)))

    jetimage_list.append(image)
    
jetimage_list = np.array(jetimage_list)


print("There are {} jet images.".format(len(jetimage_list)))

np.savez(imagespath +str(PRO)+"_"+str(GEN)+"_"+str(SHO)+"_"+str(file_number)+"_trimmed.npz", jetimage_list)

ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))


