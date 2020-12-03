#!/usr/bin/env python
# encoding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
# Install TensorFlow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten , Convolution2D, MaxPooling2D , Lambda, Conv2D, Activation,Concatenate
from tensorflow.keras.layers import ActivityRegularization
from tensorflow.keras.optimizers import Adam , SGD , Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers , initializers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import NumpyArrayIterator

gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
except RuntimeError as e:
# Visible devices must be set before GPUs have been initialized
    print(e)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
# from xgboost import XGBClassifier
import tensorflow.keras.backend as K
from sklearn import metrics

# Import local libraries
import numpy as np
import time
import pandas as pd
import os
from Model_Scaner import DNN_Model_Scan

print("Tensorflow Version is {}".format(tf.__version__))
print("Keras Version is {}".format(tf.keras.__version__))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tf.device('/device:XLA_GPU:0')


##############################################################################################################
# preprocess = "trimmed"
preprocess = "untrimmed"

HOMEPATH = "/dicos_ui_home/alanchung/UniversalityDNN/"
Data_High_Level_Features_path =  HOMEPATH + "Data_High_Level_Features/"


total_list = ["GEN","SHO","PRO",
              "MJ_0","PTJ_0","t21_0","D21_0","D22_0","C21_0","C22_0",
              "MJ","PTJ","t21","D21","D22","C21","C22",
              "eventindex","index"]

if os.path.exists(HOMEPATH + "Data_ML/DNN_Scan" + "_" + str(preprocess)) == 0:
    os.mkdir(HOMEPATH + "Data_ML/DNN_Scan" + "_" + str(preprocess))
    datapath = HOMEPATH + "Data_ML/"
#     savepath = HOMEPATH + "Data_ML/DNN_Scan" + "_" + str(preprocess) + "/"
else: 
    datapath = HOMEPATH + "Data_ML/"
#     savepath = HOMEPATH + "Data_ML/DNN_Scan" + "_" + str(preprocess) + "/"

herwig_ang_train = pd.read_csv(datapath + "herwig_ang_train" + "_" + str(preprocess) + ".csv")
herwig_ang_test = pd.read_csv(datapath + "herwig_ang_test" + "_" + str(preprocess) + ".csv")
herwig_ang_val = pd.read_csv(datapath + "herwig_ang_val" + "_" + str(preprocess) + ".csv")


pythia_def_train = pd.read_csv(datapath + "pythia_def_train" + "_" + str(preprocess) + ".csv")
pythia_def_test = pd.read_csv(datapath + "pythia_def_test" + "_" + str(preprocess) + ".csv")
pythia_def_val = pd.read_csv(datapath + "pythia_def_val" + "_" + str(preprocess) + ".csv")


pythia_vin_train = pd.read_csv(datapath + "pythia_vin_train" + "_" + str(preprocess) + ".csv")
pythia_vin_test = pd.read_csv(datapath + "pythia_vin_test" + "_" + str(preprocess) + ".csv")
pythia_vin_val = pd.read_csv(datapath + "pythia_vin_val" + "_" + str(preprocess) + ".csv")


pythia_dip_train = pd.read_csv(datapath + "pythia_dip_train" + "_" + str(preprocess) + ".csv")
pythia_dip_test = pd.read_csv(datapath + "pythia_dip_test" + "_" + str(preprocess) + ".csv")
pythia_dip_val = pd.read_csv(datapath + "pythia_dip_val" + "_" + str(preprocess) + ".csv")


sherpa_def_train = pd.read_csv(datapath + "sherpa_def_train" + "_" + str(preprocess) + ".csv")
sherpa_def_test = pd.read_csv(datapath + "sherpa_def_test" + "_" + str(preprocess) + ".csv")
sherpa_def_val = pd.read_csv(datapath + "sherpa_def_val" + "_" + str(preprocess) + ".csv")


if preprocess == "trimmed":
    features = ["MJ","t21","D21","D22","C21","C22"]
    
if preprocess == "untrimmed":   
    features = ["MJ_0","t21_0","D21_0","D22_0","C21_0","C22_0"]


Generator_Name = ["herwig_ang", "pythia_def", "pythia_vin", "pythia_dip", "sherpa_def"]
DNN_model_name = ["Herwig Angular", "Pythia Default", "Pythia Vincia", "Pythia Dipole", "Sherpa Default"]

XTRAIN = [herwig_ang_train[features],pythia_def_train[features],
         pythia_vin_train[features],pythia_dip_train[features],
         sherpa_def_train[features]]

YTRAIN = [herwig_ang_train["target"],pythia_def_train["target"],
         pythia_vin_train["target"],pythia_dip_train["target"],
         sherpa_def_train["target"]]


XVAL = [herwig_ang_val[features],pythia_def_val[features],
         pythia_vin_val[features],pythia_dip_val[features],
         sherpa_def_val[features]]

YVAL = [herwig_ang_val["target"],pythia_def_val["target"],
         pythia_vin_val["target"],pythia_dip_val["target"],
         sherpa_def_val["target"]]


XTEST = [herwig_ang_test[features],pythia_def_test[features],
         pythia_vin_test[features],pythia_dip_test[features],
         sherpa_def_test[features]]

YTEST = [herwig_ang_test["target"],pythia_def_test["target"],
         pythia_vin_test["target"],pythia_dip_test["target"],
         sherpa_def_test["target"]]

print("W jet : QCD jet = 1 : 1")
print("\r")
print("{:^8}{:^15}{:^15}{:^15}{:^15}{:^15}".format("","Herwig Angular","Pythia Default","Pythia Vincia","Pythia Dipole","Sherpa Default"))
print("{:^8}{:^15}{:^15}{:^15}{:^15}{:^15}".format("Train #",len(herwig_ang_train),len(pythia_def_train),len(pythia_vin_train),len(pythia_dip_train),len(sherpa_def_train)))
print("{:^8}{:^15}{:^15}{:^15}{:^15}{:^15}".format("Test #",len(herwig_ang_test),len(pythia_def_test),len(pythia_vin_test),len(pythia_dip_test),len(sherpa_def_test)))
print("{:^8}{:^15}{:^15}{:^15}{:^15}{:^15}".format("Val. #",len(herwig_ang_val),len(pythia_def_val),len(pythia_vin_val),len(pythia_dip_val),len(sherpa_def_val)))

##############################################################################################################
Maxlayer = [1,2,3,4,5,6]
dense_unit = [32, 64, 128, 256]
BATCHSize = [64, 128, 256]  


for i, element in enumerate(Generator_Name):
    for maxlayer in Maxlayer:
        for denseunit in dense_unit:
            for batchsize in BATCHSize:
                print("\r")
                print(element)

                DNN_Model_Scan(np.asarray(XTRAIN[i]), np.asarray(YTRAIN[i]), 
                          np.asarray(XVAL[i]), np.asarray(YVAL[i]),
                          np.asarray(XTEST[0]), np.asarray(YTEST[0]),
                          maxlayers=maxlayer, 
                          IsSameSize=1, 
                          max_dense_unit=denseunit, 
                          min_dense_unit=32, 
                          model_name=str(Generator_Name[i]), 
                          featurelength=len(features),
                          model_opt=keras.optimizers.Adam(),
                          BatchSize = batchsize, 
                          Epochs=500, 
                          savendir="./DNN_Scan_"+str(Generator_Name[i])+"_"+str(maxlayer)+"_"+str(denseunit)+"_"+str(batchsize), Verbose=1)

                print("\r")
