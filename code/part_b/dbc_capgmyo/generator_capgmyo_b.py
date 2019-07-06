# -*- coding: utf-8 -*-

""" Image data generator for CapgMyo PartB """

import numpy as np
import keras
import scipy.io, scipy.signal
import data_augmentation_capgmyo as da


class DataGeneratorB(keras.utils.Sequence):
    
    def __init__(self, trials, input_directory, batch_size=32, sample_weight=False, dim=(8,16,3),
                 classes=2, shuffle=True,
                 preprocess_function=None,preprocess_function_extra=None,
                 partb_zero =False,partb_noise=False,
                 zero_column1=0,zero_column2=0,noise_column1=0,noise_column2=0,
                 snr_db=0, window_size=3,scale_sigma=0,window_step=1,rotation=0, rotation_mask=None, time_warping=0, mag_warping=0, permutation=0,
                 size_factor=1, min_max_norm=False, update_after_epoch=True):
       
        self.trials = trials
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
        self.n_reps = len(trials)
        self.shuffle = shuffle
        self.snr_db = snr_db
        self.time_warping = time_warping
        self.scale_sigma = scale_sigma
        self.partb_zero = partb_zero
        self.partb_noise = partb_noise
        self.zero_column1 = zero_column1
        self.zero_column2 = zero_column2
        self.noise_column2 = noise_column2
        self.noise_column2 = noise_column2
        self.rotation = rotation
        self.rotation_mask = rotation_mask
        self.time_warping = time_warping
        self.mag_warping = mag_warping
        self.permutation = permutation
        self.window_size = window_size
        self.window_step = window_step

        self.preprocess_function = preprocess_function
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

    
    def __generate(self):
        self.__augment()
        self.__make_segments()
        self.indexes = np.arange(len(self.x_offsets)) 

        if self.batch_size > len(self.x_offsets):
            self.batch_size = len(self.x_offsets)
        
        self.class_weights = []
        if self.sample_weight:
            self.__make_sample_weights()
            
    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        output = self.__data_generation(indexes)

        return output
        
      
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

                im= np.copy(self.X_aug[i][j:j + self.window_size])
                
                
                #PartB1 - Zero electrode
                if self.partb_zero is True :
                    im[:,self.zero_column1:self.zero_column2] = 0                            
                                        
                    #PartB2 - Input random noise to electrode    
            
                if self.partb_noise is True:
                    scale=0.001                   
                    im[:,self.noise_column1:self.noise_column2]=scale*np.random.randn()  
                
                
                im = np.reshape(im,(8,16,3))
              

            else:
                im= np.copy(self.X_aug[i])         

                x_aug = np.stack(im,axis=-1)
            if self.min_max_norm is True:
                max_x = x_aug.max()

                min_x = x_aug.min()
                x_aug = (x_aug - min_x) / (max_x - min_x)
               

            
            if np.prod(x_aug.shape) == np.prod(self.dim):

                x_aug = np.reshape(x_aug,self.dim)
            else:
                raise Exception('Generated sample dimension mismatch. Found {}, expected {}.'.format(x_aug.shape, self.dim))        
            
            X[k, ] = x_aug
            # Store class
            y[k] = self.class_index[int(self.y_aug[i])-1]

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

        for subject in range(1, 11):
            for d in range(len(self.input_directory)):
                for gesture in [i for i in self.classes]:
                    
                    for trial in self.trials:

                        file = '{}/{:03d}-{:03d}-{:03d}.mat'.format(self.input_directory[d],int(subject), int(gesture+1), int(trial))

                        emg_data = scipy.io.loadmat(file)
                      
                        x_emg = emg_data['data'].copy()
                       
                        x=x_emg


                        if self.preprocess_function is not None:
                            if isinstance(self.preprocess_function, list):
                                for params, func in zip(self.preprocess_function_extra, self.preprocess_function):
                                    x = func(x, **params)

                            else:
                                x = self.preprocess_function(x, **self.preprocess_function_extra)

                        if len(x) > self._max_len:
                            self._max_len = len(x)

                        X.append(x)

                        y.append(int(np.squeeze(emg_data['gesture'])))
                        r.append(int(np.squeeze(emg_data['trial'])))


         
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

            self.class_weights[self.class_index[int(self.y_aug[i])-1]] += 1
 

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
        return X, y,r
