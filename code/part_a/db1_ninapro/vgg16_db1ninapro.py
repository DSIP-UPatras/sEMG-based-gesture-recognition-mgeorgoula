
# -*- coding: utf-8 -*-

from keras.layers import *
from keras import initializers, regularizers, constraints, optimizers
from keras.models import load_model, Model
from keras.applications import vgg16
from keras.layers import Dropout, Flatten, Dense, Input



def VGG16_DB1_NinaPro(input_shape,classes, dropout_rate=0,dense1=0,
                  dense2=0, dense3=0, batch_norm=False):
    
    
    #Load pretrained vgg16 model
    vgg16_model= vgg16.VGG16(weights='imagenet', include_top=False)
   
    #Keep 10/23 layers of original vgg16 model
    vgg16_model = Model(vgg16_model.input, vgg16_model.get_layer('block3_pool').output) 

    
    layers = [l for l in vgg16_model.layers]
    
    
    layers = [l for l in vgg16_model.layers]
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
    x = Dense(dense1, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense3, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs=x_input,outputs=x ,name='VGG16_DB1_NinaPro')
    model.summary()
    
    print('The model is fine tuning')
    return model
   


def getNetwork(network):

    new_model = VGG16_DB1_NinaPro
    return new_model
