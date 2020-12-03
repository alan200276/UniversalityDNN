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
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
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
from Model_Scaner import CNN_2CNN_Model_Scan

print("Tensorflow Version is {}".format(tf.__version__))
print("Keras Version is {}".format(tf.keras.__version__))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tf.device('/device:XLA_GPU:0')


####################################################################
# Take in a tuple of image lists, normalze and zero zenter all of them.
def Get_average_var(image_lists):
    
    image_lists = image_lists.reshape(image_lists.shape[0], 40 ,40)
    tmp_av = np.average(image_lists, axis=0)
    tmp_var = np.var(image_lists, axis=0)
    
    return tmp_av, tmp_var

def zero_center_and_normalize(image_lists, average, var):
    image_lists = image_lists.reshape(image_lists.shape[0], 40 ,40)
    for i, element in enumerate(image_lists):
        image_lists[i] = np.divide((element - average), (np.sqrt(var)+1e-5)) #perhaps add some r to temp_sd to suppress noise
    image_lists = image_lists.reshape(image_lists.shape[0], 40 ,40, 1)
    
    return image_lists

######################################################################################
# preprocess = "trimmed"
preprocess = "untrimmed"
standardlization = 1 #1 for True #0 for False 

######################################################################################

HOMEPATH = "/dicos_ui_home/alanchung/UniversalityDNN/"

if standardlization:
    
    if os.path.exists(HOMEPATH + "Model_Scan/CNN_2CNN_Scan_std") == 0:
        os.mkdir(HOMEPATH + "Model_Scan/CNN_2CNN_Scan_std")
        datapath = HOMEPATH + "Data_ML/"
        savepath = HOMEPATH + "Model_Scan/CNN_2CNN_Scan_std/"
    else: 
        datapath = HOMEPATH + "Data_ML/"
        savepath = HOMEPATH + "Model_Scan/CNN_2CNN_Scan_std/"
        
    av = np.load(HOMEPATH + "average" + "_" + str(preprocess) + ".npy")
    var = np.load(HOMEPATH + "variance" + "_" + str(preprocess) + ".npy")

           
else:
        
    if os.path.exists(HOMEPATH + "Model_Scan/CNN_2CNN_Scan") == 0:
        os.mkdir(HOMEPATH + "Model_Scan/CNN_2CNN_Scan")
        datapath = HOMEPATH + "Data_ML/"
        savepath = HOMEPATH + "Model_Scan/CNN_2CNN_Scan/"
    else: 
        datapath = HOMEPATH + "Data_ML/"
        savepath = HOMEPATH + "Model_Scan/CNN_2CNN_Scan/"
           

W = ["W_herwig_ang","W_pythia_def","W_pythia_vin","W_pythia_dip","W_sherpa_def"]
QCD = [ "QCD_herwig_ang","QCD_pythia_def","QCD_pythia_vin","QCD_pythia_dip","QCD_sherpa_def"]  

herwig_ang_train = np.load(datapath + "herwig_ang_train" + "_" + str(preprocess) + ".npz")["arr_0"]
herwig_ang_test = np.load(datapath + "herwig_ang_test" + "_" + str(preprocess) + ".npz")["arr_0"]
herwig_ang_val = np.load(datapath + "herwig_ang_val" + "_" + str(preprocess) + ".npz")["arr_0"]

pythia_def_train = np.load(datapath + "pythia_def_train" + "_" + str(preprocess) + ".npz")["arr_0"]
pythia_def_test = np.load(datapath + "pythia_def_test" + "_" + str(preprocess) + ".npz")["arr_0"]
pythia_def_val = np.load(datapath + "pythia_def_val" + "_" + str(preprocess) + ".npz")["arr_0"]

pythia_vin_train = np.load(datapath + "pythia_vin_train" + "_" + str(preprocess) + ".npz")["arr_0"]
pythia_vin_test = np.load(datapath + "pythia_vin_test" + "_" + str(preprocess) + ".npz")["arr_0"]
pythia_vin_val = np.load(datapath + "pythia_vin_val" + "_" + str(preprocess) + ".npz")["arr_0"]

pythia_dip_train = np.load(datapath + "pythia_dip_train" + "_" + str(preprocess) + ".npz")["arr_0"]
pythia_dip_test = np.load(datapath + "pythia_dip_test" + "_" + str(preprocess) + ".npz")["arr_0"]
pythia_dip_val = np.load(datapath + "pythia_dip_val" + "_" + str(preprocess) + ".npz")["arr_0"]

sherpa_def_train = np.load(datapath + "sherpa_def_train" + "_" + str(preprocess) + ".npz")["arr_0"]
sherpa_def_test = np.load(datapath + "sherpa_def_test" + "_" + str(preprocess) + ".npz")["arr_0"]
sherpa_def_val = np.load(datapath + "sherpa_def_val" + "_" + str(preprocess) + ".npz")["arr_0"]

ntrain = int(len(herwig_ang_train)/2.)
ntest = int(len(herwig_ang_test)/2.)
nval = int(len(herwig_ang_val)/2.)
y_train = np.concatenate((np.full(ntrain, 1), np.full(ntrain, 0)))
y_test = np.concatenate((np.full(ntest, 1), np.full(ntest, 0)))
y_val = np.concatenate((np.full(nval, 1), np.full(nval, 0)))

if standardlization:
    herwig_ang_train = zero_center_and_normalize(herwig_ang_train, av, var)
    herwig_ang_test = zero_center_and_normalize(herwig_ang_test, av, var)
    herwig_ang_val = zero_center_and_normalize(herwig_ang_val, av, var)

    pythia_def_train = zero_center_and_normalize(pythia_def_train, av, var)
    pythia_def_test = zero_center_and_normalize(pythia_def_test, av, var)
    pythia_def_val = zero_center_and_normalize(pythia_def_val, av, var)

    pythia_vin_train = zero_center_and_normalize(pythia_vin_train, av, var)
    pythia_vin_test = zero_center_and_normalize(pythia_vin_test, av, var)
    pythia_vin_val = zero_center_and_normalize(pythia_vin_val, av, var)

    pythia_dip_train = zero_center_and_normalize(pythia_dip_train, av, var)
    pythia_dip_test = zero_center_and_normalize(pythia_dip_test, av, var)
    pythia_dip_val = zero_center_and_normalize(pythia_dip_val, av, var)

    sherpa_def_train = zero_center_and_normalize(sherpa_def_train, av, var)
    sherpa_def_test = zero_center_and_normalize(sherpa_def_test, av, var)
    sherpa_def_val = zero_center_and_normalize(sherpa_def_val, av, var)


Generator_Name = ["herwig_ang", "pythia_def", "pythia_vin", "pythia_dip", "sherpa_def"]
CNN_model_name = ["Herwig Angular", "Pythia Default", "Pythia Vincia", "Pythia Dipole", "Sherpa Default"]


XTRAIN = [herwig_ang_train,pythia_def_train,
         pythia_vin_train,pythia_dip_train,
         sherpa_def_train]

YTRAIN = [y_train,y_train,y_train,y_train,y_train]


XVAL = [herwig_ang_val,pythia_def_val,
         pythia_vin_val,pythia_dip_val,
         sherpa_def_val]

YVAL = [y_val,y_val,y_val,y_val,y_val]


XTEST = [herwig_ang_test,pythia_def_test,
         pythia_vin_test,pythia_dip_test,
         sherpa_def_test]

YTEST = [y_test,y_test,y_test,y_test,y_test]


print("W jet : QCD jet = 1 : 1")
print("\r")
print("{:^8}{:^15}{:^15}{:^15}{:^15}{:^15}".format("","Herwig Angular","Pythia Default","Pythia Vincia","Pythia Dipole","Sherpa Default"))
print("{:^8}{:^15}{:^15}{:^15}{:^15}{:^15}".format("Train #",len(herwig_ang_train),len(pythia_def_train),len(pythia_vin_train),len(pythia_dip_train),len(sherpa_def_train)))
print("{:^8}{:^15}{:^15}{:^15}{:^15}{:^15}".format("Test #",len(herwig_ang_test),len(pythia_def_test),len(pythia_vin_test),len(pythia_dip_test),len(sherpa_def_test)))
print("{:^8}{:^15}{:^15}{:^15}{:^15}{:^15}".format("Val. #",len(herwig_ang_val),len(pythia_def_val),len(pythia_vin_val),len(pythia_dip_val),len(sherpa_def_val)))

######################################################################################
# time counter
print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
ticks_1 = time.time()
######################################################################################


Dense_Unit = [50,100,200,300]
First_Filters = [32,64]
Second_Filters = [32,64]
Dropout_Rate = [0.1, 0.01, 0.001]
BATCHSize = [64, 128, 256, 512]  


for i, element in enumerate(Generator_Name):
    for dense_unit in Dense_Unit:
        for first_filters in First_Filters:
            for second_filters in Second_Filters:
                for dropout_rate in Dropout_Rate:
                    for batchsize in BATCHSize:
                        if os.path.isdir("./CNN_2CNN_Scan_std/CNN_2CNN_Scan_"
                                         +str(Generator_Name[i])
                                         +"_"+str(dense_unit)
                                         +"_"+str(first_filters)
                                         +"_"+str(second_filters)
                                         +"_"+str(int(1/dropout_rate))
                                         +"_"+str(batchsize)):
                            continue

                        print("\r")
                        print(element)

                        CNN_2CNN_Model_Scan(np.asarray(XTRAIN[i]), np.asarray(YTRAIN[i]), 
                                                  np.asarray(XVAL[i]), np.asarray(YVAL[i]),
                                                  np.asarray(XTEST[i]), np.asarray(YTEST[i]),
                                                  dense_unit=dense_unit, 
                                                  first_filters = first_filters,
                                                  second_filters = second_filters,
                                                  dropout_rate=dropout_rate,
                                                  model_name=str(Generator_Name[i]), 
                                                  model_opt=keras.optimizers.Adadelta(),
                                                  BatchSize=BATCHSize[i], Epochs=500,
                                                  savendir="./CNN_2CNN_Scan_std/CNN_2CNN_Scan_"
                                            +str(Generator_Name[i])
                                            +"_"+str(dense_unit)
                                            +"_"+str(first_filters)
                                            +"_"+str(second_filters)
                                            +"_"+str(int(1/dropout_rate))
                                            +"_"+str(batchsize), Verbose=1 )

                        print("\r")
