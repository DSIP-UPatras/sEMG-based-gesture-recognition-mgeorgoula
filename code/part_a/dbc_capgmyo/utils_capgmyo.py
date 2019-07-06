# -*- coding: utf-8 -*-

#Based on  https://github.com/BrambleXu/nlp-beginner-guide-keras/blob/master/sentiment-comparison/char-level/data_helpers.py


import numpy as np
from sklearn import metrics
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import TensorBoard, Callback
import keras.backend as K
import tensorflow as tf
import os


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir=r'drive/Thesis_emg/tensorboardlogs_combo', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
        
DEFAULT_GENERATOR_PARAMS = {
    "trials": [],
    "input_directory": '',
    "batch_size": 32,
    "sample_weight": False,
    "dim": [None,],
    "classes": 8,
    "shuffle": False,
    "snr_db": 0,
    "window_size": 0,
    "window_step": 0,
    "size_factor": 0,
    "preprocess_function": None,
    "min_max_norm": False,
    "update_after_epoch": True
}