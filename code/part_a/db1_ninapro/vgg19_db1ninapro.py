# -*- coding: utf-8 -*-
"""Modify pretrained VGG19"""


from keras.layers import *
from keras import initializers, regularizers, constraints, optimizers
from keras.models import load_model, Model
from keras.applications import vgg19
from keras.layers import Dropout, Flatten, Dense, Input
import keras.backend as K



def VGG19_DB1_NinaPro(input_shape, classes,dropout_rate=0.,dense1=0,dense2=0, batch_norm=False):

     #Load pretrained vgg19 model
    vgg19_model= vgg19.VGG19(weights='imagenet', include_top=False)
   
    #Keep 10/26 layers of original vgg19 model
    vgg19_model = Model(vgg19_model.input, vgg19_model.get_layer('block3_pool').output) 

    
    layers = [l for l in vgg19_model.layers]
    
    
    layers = [l for l in vgg19_model.layers]
    freeze_index, keep_index = 0, 0
    for i in range(0, len(layers)):
        if 'block1' in layers[i].name:
            keep_index = i
        if 'block3' in layers[i].name:
            keep_index = i

    assert(keep_index >= freeze_index), '{} {}'.format(keep_index, freeze_index)

    # Define input
    x_input = Input(input_shape)
    x=x_input
    for i in range(1,len(layers)):
		# Freeze layers
        if i <= freeze_index:
           layers[i].trainable = False
        # Fine-tune layers
        elif i > freeze_index and i <= keep_index:
           layers[i].trainable = True
        else:
        	break
        x=layers[i](x)

    #Classifier  
    if batch_norm is True:    
        x = BatchNormalization()(x)       
    x = Flatten()(x) 
    x = Dropout(dropout_rate)(x)
    x = Dense(dense1, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense2, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs=x_input,outputs=x ,name='VGG19_DB1_NinaPro')
    model.summary()
    
    print('The model is fine tuning')
    return model
   

   


def getNetwork(network):

    new_model = VGG19_DB1_NinaPro
    return new_model
