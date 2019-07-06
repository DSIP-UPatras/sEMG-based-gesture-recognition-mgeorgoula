# -*- coding: utf-8 -*-

# Based on https://github.com/DSIP-UPatras/PhyCS2018_paper/blob/master/generator.py

"""Image data generator for DB5 NinaPro"""



import numpy as np
import keras
import scipy.io, scipy.signal
import data_augmentation as da


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, repetitions, input_directory, batch_size=32, sample_weight=False, dim=(52,16,3),
                 classes=2, shuffle=True, 
                 snr_db=0, window_size=52,scale_sigma=0,window_step=1,rotation=0, rotation_mask=None, time_warping=0, mag_warping=0, permutation=0,
                 data_type='rms',
                 preprocess_function=None,preprocess_function_extra=None,
                 size_factor=1, min_max_norm=True, update_after_epoch=True):
        self.repetitions = repetitions
        self.input_directory = input_directory if isinstance(input_directory, list) else [input_directory]
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.dim = tuple(dim)
        if isinstance(classes, int):
            self.n_classes = classes
            self.classes = [i for i in range(classes)]
        elif isinstance(classes, list):
            self.n_classes = len(classes)
            self.classes = classes
        self.__make_class_index()
        self.n_reps = len(repetitions)
        self.shuffle = shuffle
        self.snr_db = snr_db
        self.time_warping = time_warping
        self.scale_sigma = scale_sigma
    
        self.rotation = rotation
        self.rotation_mask = rotation_mask
        self.time_warping = time_warping
        self.mag_warping = mag_warping
        self.permutation = permutation
        self.window_size = window_size
        self.window_step = window_step
        self.data_type = 'rms' if data_type=='rms' else 'raw'
        self.preprocess_function_= preprocess_function
        self.preprocess_function_extra = preprocess_function_extra
        self.size_factor = size_factor
        self.min_max_norm = min_max_norm
        self.update_after_epoch = update_after_epoch
        self.__load_dataset()
        self.__generate()
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
        
    def __str__(self):
        return  'Classes: {}\n'.format(self.n_classes) + \
                'Class weights: {}\n'.format(self.class_weights) + \
                'Original dataset: {}\n'.format(len(self.X)) + \
                'Augmented dataset: {}\n'.format(len(self.X_aug)) + \
                'Number of sliding windows: {}\n'.format(len(self.x_offsets)) + \
                'Batch size: {}\n'.format(self.batch_size) + \
                'Number of iterations: {}\n'.format(self.__len__()) + \
                'Window size: {}\n'.format(self.window_size) + \
                'Window step: {}\n'.format(self.window_step) + \
                'Output shape: {}\n'.format(self.dim)

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        output = self.__data_generation(indexes)

        return output

    def __generate(self):
        self.__augment()
        self.__make_segments()
        self.indexes = np.arange(len(self.x_offsets))       
        if self.batch_size > len(self.x_offsets):
            self.batch_size = len(self.x_offsets)

        self.class_weights = []
        if self.sample_weight:
            self.__make_sample_weights()

    def on_epoch_end(self):
        '''Applies augmentation and updates indexes after each epoch'''
        if self.update_after_epoch:
            self.__generate()

        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        '''Generates data containing batch_size samples'''
        # Initialization
        X = np.empty((self.batch_size, *self.dim))           
        y = np.empty((self.batch_size), dtype=int)
        if self.sample_weight:
            w = np.empty((self.batch_size), dtype=float)

        # Generate data
        for k, index in enumerate(indexes):
            
            i, j = self.x_offsets[index]
            # Store sample
            if self.window_size != 0:

                im1 = np.copy(self.X_aug[i][j:j + self.window_size])

                j=j+1

                im2 = np.copy(self.X_aug[i][j:j + self.window_size])

                j=j+1

                im3 = np.copy(self.X_aug[i][j:j + self.window_size])

                j=j+1

                #Compensate for mismatch
                if im1.shape != (52,16):

                    a=np.zeros(16)
                    im1=np.insert(im1,51,a,0)
                if im3.shape != (52,16):
                    a=np.zeros(16)
                    im3=np.insert(im3,51,a,0)

                image =(im1,im2,im3)

                x_aug = np.stack(image, axis=-1)

            else:
                im= np.copy(self.X_aug[i])         

                x_aug = np.stack(im,axis=-1)
            if self.min_max_norm is True:
                max_x = x_aug.max()
                min_x = x_aug.min()
                x_aug = (x_aug - min_x) / (max_x - min_x)

            if np.prod(x_aug.shape) == np.prod(self.dim):
                x_aug = np.reshape(x_aug,(52,16,3))
            else:
                raise Exception('Generated sample dimension mismatch. Found {}, expected {}.'.format(x_aug.shape, self.dim))

            X[k, ] = x_aug
            # Store class
            y[k] = self.class_index[int(self.y_aug[i])]

            if self.sample_weight:
                w[k] = self.class_weights[(y[k])]

        
        output = (X, keras.utils.to_categorical(y, num_classes=self.n_classes))
        if self.sample_weight:
            output += (w,)

        return output

    def __augment(self):
        '''Applies augmentation incrementally'''
        self.X_aug, self.y_aug, self.r_aug = [], [], []
        for i in range(len(self.X)):
            for _ in range(self.size_factor):
                x = np.copy(self.X[i])
                if self.snr_db != 0:
                    x = da.jitter(x, snr_db=self.snr_db)
                if self.time_warping != 0:
                    x = da.time_warp(x, sigma=self.time_warping)    

                if self.permutation or self.rotation or self.time_warping or self.scale_sigma or self.mag_warping or self.snr_db:
                    self.X_aug.append(x)
                    self.y_aug.append(self.y[i])
                    self.r_aug.append(self.r[i])
                
            
            self.X_aug.append(self.X[i])
            self.y_aug.append(self.y[i])
            self.r_aug.append(self.r[i])

    def __load_dataset(self):
        '''Loads data and applies preprocess_function_1'''
        X, y, r = [], [], []
        self._max_len = 0
        if 0 in self.classes:
            rest_rep_groups = list(
                zip(
                    np.random.choice(self.repetitions, (self.n_reps), replace=False),
                    np.random.choice([i for i in self.classes if i != 0], (self.n_reps), replace=False)
                    )
                )
        for subject in range(1, 11):
            for d in range(len(self.input_directory)):
                for label in [i for i in self.classes if i != 0]:
                    
                    for rep in self.repetitions:
                        file = '{}/subject-{:02d}/gesture-{:02d}/{}/rep-{:02d}.mat'.format(self.input_directory[d],int(subject), int(label), self.data_type, int(rep))
                        data = scipy.io.loadmat(file)
                        x = data['emg'].copy()
                        if self.preprocess_function is not None:
                            if isinstance(self.preprocess_function, list):
                                for params, func in zip(self.preprocess_function_extra, self.preprocess_function):
                                    x = func(x, **params)

                            else:
                                x = self.preprocess_function(x, **self.preprocess_function_extra)

                        if len(x) > self._max_len:
                            self._max_len = len(x)

                        X.append(x)
                        y.append(int(np.squeeze(data['stimulus'])[0]))
                        r.append(int(np.squeeze(data['repetition'])[0]))

                if 0 in self.classes:
                    for rep, label in rest_rep_groups:
                        file = '{}/subject-{:02d}/gesture-00/{}/rep-{:02d}_{:02d}.mat'.format(self.input_directory[d],int(subject), self.data_type, int(rep), int(label))
                        data = scipy.io.loadmat(file)
                        x = data['emg'].copy()

                        if self.preprocess_function is not None:
                            if isinstance(self.preprocess_function, list):
                                for params, func in zip(self.preprocess_function_extra, self.preprocess_function):
                                    x = func(x, **params)

                            else:
                                x = self.preprocess_function(x, **self.preprocess_function_extra)
                        if len(x) > self._max_len:
                            self._max_len = len(x)

                        X.append(x)
                        y.append(int(np.squeeze(data['stimulus'])[0]))
                        r.append(int(np.squeeze(data['repetition'])[0]))
 
                    
        self.X = X
        self.y = y
        self.r = r

    def __make_segments(self):
        '''Creates segments either using predefined step'''
        x_offsets = []

        if self.window_size != 0:
            for i in range(len(self.X_aug)):
                for j in range(0, len(self.X_aug[i]) - self.window_size, self.window_step):
                    x_offsets.append((i, j))
        else:
            x_offsets = [(i, 0) for i in range(len(self.X_aug))]

        self.x_offsets = x_offsets
        

    def __make_sample_weights(self):
        '''Computes weights for samples'''
        self.class_weights = np.zeros(self.n_classes)
        for index in self.indexes:
            i, j = self.x_offsets[index]
            self.class_weights[self.class_index[int(self.y_aug[i])]] += 1
        self.class_weights = 1 / self.class_weights
        self.class_weights /= np.max(self.class_weights)

    def __make_class_index(self):
        '''Maps class label to 0...len(classes)'''
        self.classes.sort()
        self.class_index = np.zeros(np.max(self.classes) + 1, dtype=int)
        for i, j in enumerate(self.classes):
            self.class_index[j] = i

    def get_data(self):
        '''Retrieves all data of the epoch'''
        X = np.zeros((self.__len__() * self.batch_size, *self.dim))
        y = np.zeros((self.__len__() * self.batch_size, self.n_classes))
        r = np.zeros((self.__len__() * self.batch_size))
        if self.sample_weight:
            w = np.zeros((self.__len__() * self.batch_size))
        for i in range(self.__len__()):
            if self.sample_weight:
                x_, y_, w_ = self.__getitem__(i)
                w[i * self.batch_size:(i + 1) * self.batch_size] = w_
            else:
                x_, y_ = self.__getitem__(i)
            X[i * self.batch_size:(i + 1) * self.batch_size] = x_
            y[i * self.batch_size:(i + 1) * self.batch_size] = y_

        for k, index in enumerate(self.indexes):
            i, j = self.x_offsets[index]
            if k >= len(r):
                break
            r[k] = self.r_aug[i]
        if self.sample_weight:
            return X, y, r, w
        return X, y, r