# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import preprocessing_db5

dataDir = r'C:\Users\Marina\Desktop\ninapro-db5\Ninapro-DB5_Preprocessed'

X_data=[]
Y_data=[]
for subject in range(1, 11):
    for g in range(1,13):
#         for d in range(len(dataDir)):       
            for rep in range(1,7):
                        file = dataDir+'/subject-{:02d}/gesture-{:02d}/rms/rep-{:02d}.mat'.format(int(subject),int(g),int(rep))
                        data = scipy.io.loadmat(file)
                        x = data['emg'].copy()
                        x = preprocessing_db5.lpf(x)
                        X_data.append(x)
                        Y_data.append(int(np.squeeze(data['stimulus'][0])))
                          
# 1. Use middle segment of signals to compute the mean
                      
X_segm = np.zeros((len(X_data), 260, 10))
Y_segm = np.zeros((len(Y_data),260,1))
for i in range(len(X_data)):
    m = len(X_data[i]) // 2
    X_segm[i,:,:] = X_data[i][m-130: m+130,0:10]

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
plt.legend(['electrode-{}'.format(i+1) for i in range(10)], bbox_to_anchor=(1.1, 1.05))
plt.savefig('ground_truth_db5_mpeeee.png')
plt.show()



    