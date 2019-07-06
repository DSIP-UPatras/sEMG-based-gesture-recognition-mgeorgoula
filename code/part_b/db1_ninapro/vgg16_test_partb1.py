# -*- coding: utf-8 -*-
#!/usr/bin/env python


"""Make prediction and compute confusion matrix for modified input data"""
"""PART B1 : Zero electrode column of emg data"""

import numpy as np
import tensorflow as tf

import random

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(1234)

random.seed(12345)

# 
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

from keras import backend as K

#
tf.set_random_seed(1234)


sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
sess.run(tf.global_variables_initializer())

K.set_session(sess)
##############################################################################
import sys
import matplotlib.pyplot as plt
from keras import optimizers, initializers, regularizers, constraints
from tensorflow.keras.callbacks import TensorBoard
from keras.utils import plot_model
from utils import *
from datagenerator_b2 import *
import preprocessing
import json
from sklearn import metrics
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix

with open('DB1_vgg16_b1.json') as json_file:
    config_data = json.load(json_file)
    
#Load model saved from training     

MODEL_WEIGHTS_SAVE_FILE = os.path.abspath(
        'models_vgg16') + '/'+'_DB1_vgg16'+'_{}.h5'

MODEL_SAVE_FILE = os.path.abspath(
            'models_vgg16') + '/'+'_DB1_vgg16'+'_{}.json'

PARAMS_MODEL = config_data['model']
PARAMS_DATASET = config_data['dataset']

                                  
PARAMS_TEST_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = PARAMS_DATASET.get('test_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_TEST_GENERATOR[key] = params_gen[key]


#input_directory = r'drive/Thesis_emg/Ninapro-DB1_Preprocessed'

input_directory = r'C:\Users\Marina\Desktop\HMTY\ΔΙΠΛΩΜΑΤΙΚΗ\EMG datasets\DB1-NINAPRO\Ninapro-DB1_Preprocessed'


PARAMS_TEST_GENERATOR['preprocess_function'] = [preprocessing.lpf]
PARAMS_TEST_GENERATOR['preprocess_function_extra'] = [{'fs':100}]
PARAMS_TEST_GENERATOR['data_type'] = 'rms'
PARAMS_TEST_GENERATOR['classes'] = [i for i in range(13)]


PARAMS_TEST_GENERATOR.pop('input_directory', '')


test_generator = DataGeneratorB(input_directory=input_directory,**PARAMS_TEST_GENERATOR)

X_test, Y_test, test_reps = test_generator.get_data()
y_test = np.argmax(Y_test, axis=1)


# load json and create model
with open(MODEL_SAVE_FILE,'r') as f:
    json = f.read()
loaded_model = model_from_json(json)


loaded_model.load_weights(MODEL_WEIGHTS_SAVE_FILE)
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


Y_pred = loaded_model.predict(X_test)

y_pred = np.argmax(Y_pred, axis=1)

#Print Confusion Matrix
print(confusion_matrix(y_test,y_pred))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.imshow(confusion_matrix(y_test,y_pred))

