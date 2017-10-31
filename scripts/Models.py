import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, InputLayer
from keras.layers.convolutional import Conv2D, MaxPooling2D
import sys
import os
pwd = os.getcwd()
sys.path.append("/root/Gan/jidian/MLexperiments")
sys.path.append("/root/Gan/jidian")
sys.path.append(pwd)
sys.path.append(os.path.dirname(os.getcwd()))
import MLexperiments.classes
from MLexperiments.classes import ReadAutoLabeledData
import tensorflow as tf
import MLexperiments.config.parameters
from keras.layers import LSTM, Reshape
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import utils
import keras
from keras.callbacks import ModelCheckpoint
import os
from keras.callbacks import TensorBoard
from time import time
from keras.models import Model
from keras.layers import Input, Dense
import datetime

BATCH_SIZE=128
def getMODEL(flag = 0):
    if flag == 1:
        return get_RNNMODEL()
    else:
        return get_CNNMODEL()
def get_RNNMODEL():
    RNNMODEL = Sequential()
    #model.add(Input(shape=(MLexperiments.config.parameters.SAMPLE_LEN, MLexperiments.config.parameters.SAMPLE_HEIGHT, 1)))
    # model.add(
    #     Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),input_shape=(MLexperiments.config.parameters.SAMPLE_LEN,\
    #                                                                        MLexperiments.config.parameters.SAMPLE_HEIGHT, 1)\
    #            , padding='valid', activation='relu',name="conv1"))

    RNNMODEL.add(
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),batch_input_shape=(None, MLexperiments.config.parameters.SAMPLE_LEN,\
                                                                           MLexperiments.config.parameters.SAMPLE_HEIGHT, 1)\
               , padding='valid', activation='relu',name="conv1"))

    #model.add(MaxPooling2D(pool_size=(2, 2)))
    RNNMODEL.add(Dropout(0.5, name="dropout1"))
    RNNMODEL.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', name="conv2"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    RNNMODEL.add(Dropout(0.5, name="dropout2"))
    RNNMODEL.add(Conv2D(32, kernel_size=(10, MLexperiments.config.parameters.SAMPLE_HEIGHT - 4), name="conv3", strides=(1, 1), padding='valid', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    RNNMODEL.add(Dropout(0.5, name="dropout3"))

    RNNMODEL.add(Reshape((RNNMODEL.get_layer("dropout3").output_shape[1], RNNMODEL.get_layer("dropout3").output_shape[3]), name ="reshape"))

    RNNMODEL.add(LSTM(128, name ="LSTM1"))
    RNNMODEL.add(Dropout(0.5))
    RNNMODEL.add(Dense(64, activation='relu', name ="dense1"))
    RNNMODEL.add(Dropout(0.5))
    RNNMODEL.add(Dense(32, activation='relu', name ="dense2"))
    RNNMODEL.add(Dropout(0.5))
    RNNMODEL.add(Dense(MLexperiments.config.parameters.OUTPUTNUM, activation='softmax', name ="dense3"))
    return RNNMODEL

def get_CNNMODEL():
    model = Sequential()
    model.add(
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', input_shape=\
               (MLexperiments.config.parameters.SAMPLE_LEN, MLexperiments.config.parameters.SAMPLE_HEIGHT, 1),\
               activation='relu'))

    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(MLexperiments.config.parameters.OUTPUTNUM, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model


#todo[][] refactoring model
# BATCH_SIZE=128

# a = Input(shape=(MLexperiments.config.parameters.SAMPLE_LEN,\
#                                                                        MLexperiments.config.parameters.SAMPLE_HEIGHT, 1))
# #model.add(Input(shape=(MLexperiments.config.parameters.SAMPLE_LEN, MLexperiments.config.parameters.SAMPLE_HEIGHT, 1)))
# # model.add(
# #     Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),input_shape=(MLexperiments.config.parameters.SAMPLE_LEN,\
# #                                                                        MLexperiments.config.parameters.SAMPLE_HEIGHT, 1)\
# #            , padding='valid', activation='relu',name="conv1"))

# # x =    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1)\
# #            , padding='valid', activation='relu',name="conv1")(a)
# x = Flatten()(a)

# x = Dense(2, activation='softmax', name = "dense3")(x)

# model = Model(inputs = [a], outputs = [x])
