# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:11:21 2019

@author: Marina
"""

# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import preprocessing_capgmyo 

dataDir = r'C:\Users\Marina\Desktop\CAPGMYO\Capgmyo_dbc_preprocessed'

X_data=[]
Y_data=[]
e=0
for subject in range(1, 10):
    for g in range(1,13):
#         for d in range(len(dataDir)):       
            for rep in range(1,11):
                        file = dataDir+'/{:03d}-{:03d}-{:03d}.mat'.format(int(subject),int(g),int(rep))
                        emg_data = scipy.io.loadmat(file)
                        x_electrode_arr = np.zeros((1000,8))
                        x = emg_data['data'].copy()
                        x = preprocessing_capgmyo.lpf(x)
#                        print(x[:,16:32].mean(axis=-1).shape)
#                        for t in range(8):
#                            x_electrode_arr[:][t:t] = np.mean(x[:,e:e+16],axis=-1)
#                            e=e+16
                        
                        x_electrode_arr[:,0] = np.mean(x[:,0:15],axis=-1)
                        x_electrode_arr[:,1] = np.mean(x[:,16:31],axis=-1)
                        x_electrode_arr[:,2] = np.mean(x[:,32:47],axis=-1)
                        x_electrode_arr[:,3] = np.mean(x[:,48:63],axis=-1)
                        x_electrode_arr[:,4] = np.mean(x[:,64:79],axis=-1)
                        x_electrode_arr[:,5] = np.mean(x[:,80:95],axis=-1)
                        x_electrode_arr[:,6] = np.mean(x[:,96:111],axis=-1)
                        x_electrode_arr[:,7] = np.mean(x[:,112:127],axis=-1)
                            
                        X_data.append(x_electrode_arr)
                                              
                        Y_data.append(int(np.squeeze(emg_data['gesture'])))
 


                         
# 1. Use middle segment of signals to compute the mean
                      
X_segm = np.zeros((len(X_data), 340, 8))
Y_segm = np.zeros((len(Y_data),340,1))
for i in range(len(X_data)):
    m = len(X_data[i]) // 2
    X_segm[i,:,:] = X_data[i][m-170: m+170, :]

plt.figure(figsize=(20,20))
# 2. For every gesture compute and plot mean



for label in range(1,13):
    l=[]
    for k in range(len(Y_data)):
      if Y_data[k]==label:
    
        l.append(k) 
        
    x_mean = [X_segm[j] for j in l]  

    xm = np.mean(x_mean,axis=0)
    plt.subplot(4,3,label)
    plt.plot(xm)
    plt.title('gesture {}'.format(label))
plt.legend(['electrode-{}'.format(i+1) for i in range(8)], bbox_to_anchor=(1.1, 1.05))
plt.savefig('ground_truth_capgmyo.png')
plt.show()



    