# -*- coding: utf-8 -*-


from keras.layers import *
from keras import initializers, regularizers, constraints, optimizers
from keras.models import load_model, Model
from keras.applications import inception_v3
from keras.layers import Dropout, Flatten, Dense, Input
import keras.backend as K
import keras.utils




def InceptionV3_Capgmyo(input_shape, classes, dropout_rate=0., dense1=0,dense2=0, batch_norm=False):

 
 
    #Load pretrained InceptionV3 model
    inc_model= inception_v3.InceptionV3(weights='imagenet', include_top=False)
    #Keep 9/311 layers of original InceptionV3
    inc_model = Model(inc_model.input, inc_model.layers[9].output)

    # Define input
    x_input = Input(input_shape)
    output_inc= inc_model(x_input)

    #Classifier    
    if batch_norm is True:    
        x = BatchNormalization()(output_inc)
    x = Flatten()(x)
    x = Dense(dense1, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense2, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x) # Number of classes
    model = Model(x_input, x ,name='Inceptionv3_Capgmyo')
    
    #Freeze first convolutional layer
    for layer in model.layers[:1]:
        layer.trainable = False
    model.summary()
    print('Model is fine tuning')    
    return model
   


def getNetwork(network):

    model = InceptionV3_Capgmyo
    return model