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


print("Tensorflow Version is {}".format(tf.__version__))
print("Keras Version is {}".format(tf.keras.__version__))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tf.device('/device:XLA_GPU:0')
######################################################################################
"""
DNN_Model_Scan
"""
def DNN_Model_Scan(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest,
              maxlayers=3, 
              IsSameSize=1, 
              max_dense_unit=128, 
              min_dense_unit=32, 
              model_name="Model", 
              featurelength=6,
              model_opt=keras.optimizers.Adam(),
              BatchSize=64, Epochs=100,
              savendir="./DNN_Scan", Verbose=1 ):
    
    
    if os.path.exists(savendir) == 0:
        os.mkdir(savendir)
        savepath = savendir + "/" 
    else: 
        savepath = savendir + "/" 
        
    ######################################################################################
    # time counter
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    ticks_1 = time.time()
    ######################################################################################


    model_DNN = Sequential(name = "Sequential_"+str(model_name))
    model_DNN.add(keras.Input(shape=(featurelength,), name = 'input'))
    
    if not IsSameSize:
        for i in range(int(max(maxlayers, 1))):
#             dense_unit = np.linspace(min_dense_unit, max_dense_unit, maxlayers)
#             model_DNN.add(Dense(int(dense_unit[i]), activation='relu', name = "dense_"+str(i+1)))
            
            max_dense_unit = int(max_dense_unit/2)
            model_DNN_1.add(Dense(max_dense_unit, activation='relu', name = "dense_"+str(i+1)))
            
    if IsSameSize:
        for i in range(int(max(maxlayers, 1))):
            model_DNN.add(Dense(max_dense_unit, activation='relu', name = "dense_"+str(i+1)))
    
    model_DNN.add(Dense(1, activation='sigmoid', name = 'output'))
    model_DNN.add(Dropout(0.00001))


    model_DNN.summary()

    model_DNN.compile(loss="binary_crossentropy",#keras.losses.binary_crossentropy
                          optimizer=model_opt,
                          metrics=['accuracy'])
    check_list=[]
    csv_logger = CSVLogger(savepath + model_name + "_training_log.csv")
    checkpoint = ModelCheckpoint(
                        filepath= savepath + model_name + "_checkmodel.h5",
                        save_best_only=True,
                        verbose=0)
    
    earlystopping = EarlyStopping(
                        monitor="loss",
                        min_delta=0.01,
                        patience=50,
                        verbose=Verbose,
                        mode="auto",
                        baseline=None,
                        restore_best_weights=False,
                    )

    check_list.append(checkpoint)
    check_list.append(csv_logger)
    check_list.append(earlystopping)
    model_DNN.fit(Xtrain, Ytrain,
                    validation_data = (Xval, Yval),
                    batch_size=BatchSize,
                    epochs=Epochs,
                    callbacks=check_list,
                    verbose=Verbose)

    model_DNN.save(savepath + model_name + "_DNN" + "_"  + ".h5")
    
    Learning_curve = pd.read_csv(savepath + model_name + "_training_log.csv")
    loss = model_DNN.evaluate(Xtest, Ytest,verbose=0)
    ######################################################################################
    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
    ######################################################################################
    
    f = open(savepath + model_name + "_outinfo.txt","w")
    f.write("\n")
    f.write("File Name: {}\n".format(model_name))
    f.write("Dense_layers: {}\n".format(maxlayers))
    f.write("MaxNNode: {}\n".format(max_dense_unit))
    f.write("MinNNode: {}\n".format(min_dense_unit))
#     f.write("learningrate: {}\n".format(learingrate))
    f.write("epoch: {}\n".format(len(Learning_curve)))
    f.write("batch_s: {}\n".format(BatchSize))
    f.write("optimizer: {}\n".format(str(model_opt)))
    f.write("loss: {:.3f}\n".format(loss[0]))
    f.write("acc: {:.3f} \n".format(loss[1]))
#     f.write("mse: {} \n".format(loss[2]))
#     f.write("mae: {} \n".format(loss[3]))
#     f.write("mape: {} \n".format(loss[4]))
    f.write("Time consumption: {:.3f} min\n".format(totaltime/60.))
    f.close()
    ######################################################################################
    
######################################################################################
"""
CNN_2CNN_Model_Scan
"""

def CNN_2CNN_Model_Scan(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest,
                      dense_unit=128, 
                      first_filters = 32,
                      second_filters = 64,
                      dropout_rate=0.1,
                      model_name="Model", 
                      model_opt=keras.optimizers.Adadelta(),
                      BatchSize=256, Epochs=500,
                      savendir="./CNN_2CNN_Scan", Verbose=1 ):

    if os.path.exists(savendir) == 0:
        os.mkdir(savendir)
        savepath = savendir + "/" 
    else: 
        savepath = savendir + "/" 
        
    ######################################################################################
    # time counter
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    ticks_1 = time.time()
    ######################################################################################

    
    model_jet = Sequential(name = "Sequential_for_CNN_2CNN_"+str(model_name))
    input_shape = (40, 40, 1)
    model_jet.add(Conv2D( first_filters , kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
    #                 data_format='channels_first',
                    input_shape=input_shape, 
#                     input_shape=(40, 40, 1),
                    name = 'jet'))
    model_jet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
    #                            data_format='channels_first', 
                               name = 'jet_MaxPooling_1'))
    model_jet.add(Conv2D(second_filters, (5, 5), activation='relu',
    #                      data_format='channels_first', 
                         name = 'jet_2D_1'))
    model_jet.add(MaxPooling2D(pool_size=(2, 2),
    #                            data_format='channels_first', 
                               name = 'jet_MaxPooling_2'))
    model_jet.add(Flatten(name = 'jet_flatten'))
    model_jet.add(Dense(dense_unit, activation='relu', name = 'jet_dense_1'))

    model_jet.add(Dropout(dropout_rate))
    model_jet.add(Dense(1, activation='sigmoid', name = 'jet_dense_2'))

#     model_opt = keras.optimizers.Adadelta()

    model_jet.compile(loss="binary_crossentropy",#keras.losses.
                  optimizer=model_opt,
                  metrics=['accuracy'])
    model_jet.summary()
    

    check_list=[]
    csv_logger = CSVLogger(savepath + model_name + "_training_log.csv")
    checkpoint = ModelCheckpoint(
                        filepath= savepath + model_name + "_checkmodel.h5",
                        save_best_only=True,
                        verbose=0)
    
    earlystopping = EarlyStopping(
                        monitor="loss",
                        min_delta=0.01,
                        patience=50,
                        verbose=Verbose,
                        mode="auto",
                        baseline=None,
                        restore_best_weights=False,
                    )

    check_list.append(checkpoint)
    check_list.append(csv_logger)
    check_list.append(earlystopping)
    model_jet.fit(Xtrain, Ytrain,
                    validation_data = (Xval, Yval),
                    batch_size=BatchSize,
                    epochs=Epochs,
                    shuffle=True,
                    callbacks=check_list,
                    verbose=Verbose)

    model_jet.save(savepath + model_name + "_CNN_2CNN" + "_"  + ".h5")
    
    Learning_curve = pd.read_csv(savepath + model_name + "_training_log.csv")
    loss = model_jet.evaluate(Xtest, Ytest,verbose=0)
    ######################################################################################
    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
    ######################################################################################
    
    f = open(savepath + model_name + "_outinfo.txt","w")
    f.write("\n")
    f.write("File Name: {}\n".format(model_name))
    f.write("First_filters: {}\n".format(first_filters))
    f.write("Second_filters: {}\n".format(second_filters))
    f.write("Dense_nodes: {}\n".format(dense_unit))
    f.write("dropout_rate: {}\n".format(dropout_rate))
    f.write("epoch: {}\n".format(len(Learning_curve)))
    f.write("batch_s: {}\n".format(BatchSize))
    f.write("optimizer: {}\n".format(str(model_opt)))
    f.write("loss: {:.3f}\n".format(loss[0]))
    f.write("acc: {:.3f} \n".format(loss[1]))
#     f.write("mse: {} \n".format(loss[2]))
#     f.write("mae: {} \n".format(loss[3]))
#     f.write("mape: {} \n".format(loss[4]))
    f.write("Time consumption: {:.3f} min\n".format(totaltime/60.))
    f.close()
    ######################################################################################
    
    

######################################################################################
"""
CNN_MaxOut_Model_Scan
"""

def CNN_MaxOut_Model_Scan(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest,
                      frist_dense_unit=256, 
                      frist_nb_feature = 5,
                      second_dense_unit=128, 
                      second_nb_feature = 5,
                      first_dense = 64,
                      second_dense = 32,
                      dropout_rate=0.1,
                      model_name="Model", 
                      model_opt=keras.optimizers.Adadelta(),
                      BatchSize=256, Epochs=500,
                      savendir="./CNN_MaxOut_Scan", Verbose=1 ):
    
    
    import tensorflow.keras as keras
    from keras.layers import MaxoutDense
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout, Flatten, Concatenate
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
    
    
    if os.path.exists(savendir) == 0:
        os.mkdir(savendir)
        savepath = savendir + "/" 
    else: 
        savepath = savendir + "/" 
        
    ######################################################################################
    # time counter
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    ticks_1 = time.time()
    ######################################################################################

    
    
    model_jet = Sequential(name = "Sequential_for_CNN_MaxOut_"+str(model_name))
    input_shape = (40, 40, 1)
    model_jet.add(Flatten(input_shape=input_shape,name = 'jet_flatten'))
    model_jet.add(MaxoutDense(frist_dense_unit, input_dim=1600, nb_feature=frist_nb_feature, init='he_uniform'))
    model_jet.add(MaxoutDense(second_dense_unit, nb_feature=second_nb_feature))

    model_jet.add(Dense(first_dense, activation='relu', name = 'jet_dense_1'))
    model_jet.add(Dense(second_dense, activation='relu', name = 'jet_dense_2'))
    
    model_jet.add(Dropout(dropout_rate))
    model_jet.add(Dense(1, activation='sigmoid', name = 'jet_dense_3'))
    model_opt = keras.optimizers.Adadelta()

    model_jet.compile(loss="binary_crossentropy",#keras.losses.
                  optimizer=model_opt,
                  metrics=['accuracy'])
    model_jet.summary()
    

    check_list=[]
    csv_logger = CSVLogger(savepath + model_name + "_training_log.csv")
    checkpoint = ModelCheckpoint(
                        filepath= savepath + model_name + "_checkmodel.h5",
                        save_best_only=True,
                        verbose=0)
    
    earlystopping = EarlyStopping(
                        monitor="loss",
                        min_delta=0.01,
                        patience=50,
                        verbose=Verbose,
                        mode="auto",
                        baseline=None,
                        restore_best_weights=False,
                    )

    check_list.append(checkpoint)
    check_list.append(csv_logger)
    check_list.append(earlystopping)
    model_jet.fit(Xtrain, Ytrain,
                    validation_data = (Xval, Yval),
                    batch_size=BatchSize,
                    epochs=Epochs,
                    shuffle=True,
                    callbacks=check_list,
                    verbose=Verbose)

    model_jet.save(savepath + model_name + "_CNN_MaxOut" + "_"  + ".h5")
    
    Learning_curve = pd.read_csv(savepath + model_name + "_training_log.csv")
    loss = model_jet.evaluate(Xtest, Ytest,verbose=0)
    ######################################################################################
    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))
    ######################################################################################
    
    f = open(savepath + model_name + "_outinfo.txt","w")
    f.write("\n")
    f.write("File Name: {}\n".format(model_name))
    f.write("Frist_Dense_Unit: {}\n".format(frist_dense_unit))
    f.write("Frist_nb_feature: {}\n".format(frist_nb_feature))
    f.write("Second_Dense_Unit: {}\n".format(second_dense_unit))
    f.write("Second_nb_feature: {}\n".format(second_nb_feature))
    f.write("First_Dense: {}\n".format(first_dense))
    f.write("Second_Dense: {}\n".format(second_dense))
    f.write("dropout_rate: {}\n".format(dropout_rate))
    f.write("epoch: {}\n".format(len(Learning_curve)))
    f.write("batch_s: {}\n".format(BatchSize))
    f.write("optimizer: {}\n".format(str(model_opt)))
    f.write("loss: {:.3f}\n".format(loss[0]))
    f.write("acc: {:.3f} \n".format(loss[1]))
#     f.write("mse: {} \n".format(loss[2]))
#     f.write("mae: {} \n".format(loss[3]))
#     f.write("mape: {} \n".format(loss[4]))
    f.write("Time consumption: {:.3f} min\n".format(totaltime/60.))
    f.close()
    ######################################################################################
    
    