# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 19:10:29 2018

@author: Marina
"""



from keras.layers import *
from keras import initializers, regularizers, constraints, optimizers
from keras.models import load_model, Model
from keras.applications import resnet50
from keras.layers import Dropout, Flatten, Dense, Input
import keras.backend as K
import numpy as np


def Resnet50_NinaProDB5(input_shape,classes,dropout_rate1=0,dropout_rate2=0, batch_norm=False):

#   

    res_model= resnet50.ResNet50(weights='imagenet', include_top=False)
#    res_model.summary()
    
    res_model = Model(res_model.input, res_model.layers[48].output) 

    x_input = Input(input_shape)

    output_res= res_model(x_input)
    for layer in res_model.layers:
        layer.trainable = False
        
    x = BatchNormalization()(output_res)
    x = Flatten()(x)   
    x = Dropout(dropout_rate1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate2)(x)
    if batch_norm is True:
        x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x) # Number of classes
    new_model = Model(inputs=x_input,outputs=x ,name='Resnet50_NinaProDB5')
    new_model.summary()
    print('Model loaded.')
    
    return new_model
   


def getNetwork(network):

    new_model = Resnet50_NinaProDB5
    return new_model
