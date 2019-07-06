# -*- coding: utf-8 -*-
"""Train customized VGG16 - DB5 NINAPRO - A EXERCISE"""


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
from generator_db5 import *
from vgg16_ninaprodb5 import *
import preprocessing_db5
import json
import re
import datetime
from sklearn import metrics
import scipy.io
import matplotlib.pyplot as plt
import time
from keras.models import model_from_json
# 1. Logging
TIMESTAMP = '{}'.format(
    re.sub('[^A-Za-a0-9]+', '', '{}'.format(datetime.datetime.now())))

with open('DB5_vgg16.json') as json_file:
    config_data = json.load(json_file)



MODEL_SAVE_FILE = os.path.abspath(
        'models_vgg16_db5') + os.sep +'_' + config_data['model']['save_file'] + '_{}.json'
MODEL_WEIGHTS_SAVE_FILE = os.path.abspath(
        'models_vgg16_db5') + os.sep + '_' + config_data['model']['save_file'] + '_{}.h5'


LOG_FILE = os.path.abspath(
        'logs_vgg16_db5') + os.sep + TIMESTAMP + '_' + config_data['logging']['log_file'] + '.log'
LOG_TESNORBOARD_FILE = os.path.abspath(
 'tensorboardlogs_vgg16_db5') + os.sep + TIMESTAMP + '_' + config_data['logging']['log_file']

METRICS_SAVE_FILE = os.path.abspath(
    'metrics_vgg16_db5') + os.sep + TIMESTAMP + '_' + config_data['logging']['log_file'] + '.mat'


# 2. Config params
PARAMS_TRAINING = config_data['training']
PARAMS_MODEL = config_data['model']
PARAMS_DATASET = config_data['dataset']

PARAMS_TRAIN_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = PARAMS_DATASET.get('train_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_TRAIN_GENERATOR[key] = params_gen[key]

PARAMS_VALID_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = PARAMS_DATASET.get('valid_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_VALID_GENERATOR[key] = params_gen[key]

PARAMS_TEST_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = PARAMS_DATASET.get('test_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_TEST_GENERATOR[key] = params_gen[key]


SUBJECTS = config_data.get('subjects', [i for i in range(1, 11)])

# 3. Initialization

input_directory = r'C:\Users\Marina\Desktop\ninapro-db5\Ninapro-DB5_Preprocessed'
#input_directory = r'drive/Thesis_emg/Ninapro-DB5_Preprocessed'
PARAMS_TRAIN_GENERATOR['preprocess_function_1'] = [preprocessing_db5.lpf]
PARAMS_TRAIN_GENERATOR['preprocess_function_1_extra'] = [{'fs':200}]
PARAMS_TRAIN_GENERATOR['data_type'] = 'rms'
PARAMS_TRAIN_GENERATOR['classes'] = [i for i in range(13)]

PARAMS_VALID_GENERATOR['preprocess_function_1'] = [preprocessing_db5.lpf]
PARAMS_VALID_GENERATOR['preprocess_function_1_extra'] = [{'fs':200}]
PARAMS_VALID_GENERATOR['data_type'] = 'rms'
PARAMS_VALID_GENERATOR['classes'] = [i for i in range(13)]

PARAMS_TEST_GENERATOR['preprocess_function_1'] = [preprocessing_db5.lpf]
PARAMS_TEST_GENERATOR['preprocess_function_1_extra'] = [{'fs':200}]
PARAMS_TEST_GENERATOR['data_type'] = 'rms'
PARAMS_TEST_GENERATOR['classes'] = [i for i in range(13)]



PARAMS_TRAIN_GENERATOR.pop('input_directory', '')
PARAMS_VALID_GENERATOR.pop('input_directory', '')
PARAMS_TEST_GENERATOR.pop('input_directory', '')

MODEL = VGG16_NinaProDB5

mean_train, mean_validation = [], []
mean_cm = []
mean_train_loss, mean_validation_loss = [], []

open(LOG_FILE,'a')
with open(LOG_FILE, "w") as f:
        f.write(
            'TIMESTAMP: {}\n'
            'DATASET: {}\n'
            'TRAIN_GENERATOR: {}\n'
            'VALID_GENERATOR: {}\n'
            'TEST_GENERATOR: {}\n'
            'MODEL: {}\n'
            'MODEL_PARAMS: {}\n'
            'TRAIN_PARAMS: {}\n'.format(
                TIMESTAMP, PARAMS_DATASET['name'], PARAMS_TRAIN_GENERATOR,
                PARAMS_VALID_GENERATOR, PARAMS_TEST_GENERATOR,
                PARAMS_MODEL['name'], PARAMS_MODEL['extra'],
                PARAMS_TRAINING)
        )


train_generator = DataGenerator(input_directory=input_directory, **PARAMS_TRAIN_GENERATOR)
        
valid_generator = DataGenerator(input_directory=input_directory, **PARAMS_VALID_GENERATOR)

X_valid, Y_valid, valid_reps = valid_generator.get_data()

y_valid = np.argmax(Y_valid, axis=1)


#Load modified pretrained CNN model
model = MODEL(
        input_shape=train_generator.dim,
        classes=train_generator.n_classes,
        **PARAMS_MODEL['extra'])

#Define optimizer     
optimizer = optimizers.SGD(lr=PARAMS_TRAINING['fine_tuning_lrate'], momentum=0.9)
opt='SGD'
    
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[
                  'accuracy'])
    
#Plot training and validation acc/loss on the same graph 
train_callbacks = []
tensorboardCallback = TrainValTensorBoard(log_dir=LOG_TESNORBOARD_FILE)
train_callbacks.append(tensorboardCallback)

#Fine tune the model
model.fit_generator(train_generator, epochs=PARAMS_TRAINING['fine_tuning_epochs'],
                                  validation_data = (X_valid,Y_valid),callbacks=train_callbacks,verbose=1,shuffle=True)



train_generator = DataGenerator(input_directory=input_directory, **PARAMS_TRAIN_GENERATOR)
valid_generator = DataGenerator(input_directory=input_directory, **PARAMS_VALID_GENERATOR)
X_valid, Y_valid, valid_reps = valid_generator.get_data() 
   
optimizer = optimizers.SGD(lr=PARAMS_TRAINING['training_lrate'],momentum=0.9)


#Begin normal training

#Freeze network - Train only classifier
for i in range(0,11):                   #Freeze 9 first layers for VGG16
        model.layers[i].trainable = False
model.summary()  
print('Model is training normally')      

model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=[
                  'accuracy'])

history = model.fit_generator(train_generator, epochs=PARAMS_TRAINING['training_epochs'],
                                  validation_data = (X_valid,Y_valid)
                               , callbacks=train_callbacks,verbose=1,shuffle=True)


# serialize model to JSON
model_json = model.to_json()
with open(MODEL_SAVE_FILE, "w+") as json_file:
            json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(MODEL_WEIGHTS_SAVE_FILE)
print("Saved model to disk")


test_generator = DataGenerator(input_directory=input_directory, **PARAMS_TEST_GENERATOR)
X_test, Y_test, test_reps = test_generator.get_data()
y_test = np.argmax(Y_test, axis=1)


# load json and create model
with open(MODEL_SAVE_FILE,'r') as f:
    json = f.read()
loaded_model = model_from_json(json)

loaded_model.load_weights(MODEL_WEIGHTS_SAVE_FILE)
print("Loaded model from disk")
 
## evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_acc = score[0]
test_loss = score[1]

Y_pred = model.predict(X_test)

y_pred = np.argmax(Y_pred, axis=1)

cnf_matrix_frame = metrics.confusion_matrix(y_test, y_pred)
if np.array(mean_cm).shape != cnf_matrix_frame.shape:
            mean_cm = cnf_matrix_frame
else:
            mean_cm += cnf_matrix_frame

mean_train.append(np.mean(history.history['acc'][-5:]))
mean_validation.append(np.mean(history.history['val_acc'][-5:]))
mean_train_loss.append(np.mean(history.history['loss'][-5:]))
mean_validation_loss.append(np.mean(history.history['val_loss'][-5:]))
K.clear_session()

#Writ results to log file
with open(LOG_FILE, 'a+') as f:
        f.write('Train loss: {} +- {}\n'.format(np.mean(mean_train_loss),
                                                np.std(mean_train_loss)))
        f.write(
            'Train accuracy: {} +- {}\n'.format(np.mean(mean_train), np.std(mean_train)))
        f.write('Validation loss: {} +- {}\n'.format(np.mean(mean_validation_loss),
                                               np.std(mean_validation_loss)))
        f.write(
            'Validation accuracy: {} +- {}\n'.format(np.mean(mean_validation), np.std(mean_validation)))
        f.write('Test loss: {} +- {}\n'.format(np.mean(test_loss),
                                               np.std(test_loss)))
        f.write(
            'Test accuracy: {} +- {}\n'.format(np.mean(test_acc), np.std(test_acc)))
           
           
print('Train accuracy: {} +- {}\n'.format(np.mean(mean_train), np.std(mean_train)))
print('Train loss:{}\n'.format(np.mean(mean_train_loss)))
print('Validation accuracy: {} +- {}\n'.format(np.mean(mean_validation), np.std(mean_validation)))
print('Validation loss:{}\n'.format(np.mean(mean_validation_loss)))
print('Test accuracy: {} +- {}\n'.format(test_acc, np.std(test_acc)))
print('Test loss:{}\n'.format(test_loss))


metrics_dict = {
    'mean_cm': mean_cm,
    'mean_validation': mean_validation,
    'mean_train': mean_train,
    'mean_train_loss': mean_train_loss,
    'mean_validation_loss': mean_validation_loss
}
scipy.io.savemat(METRICS_SAVE_FILE, metrics_dict)

## list all data in history


# summarize history for accuracy
fig = plt.figure(figsize=(12,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig('/accuracy-{}-model.jpg'.format(MODEL))
plt.show()
plt.close(fig)

## summarize history for loss
fig = plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig('/loss-{}-model.jpg'.format(MODEL))
plt.show()
plt.close(fig)

