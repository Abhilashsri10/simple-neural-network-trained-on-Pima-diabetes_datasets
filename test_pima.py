# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:01:24 2018

@author: Abhilash Srivastava
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#stochastic process,so that we can use the same random number again anad again
np.random.seed(7)
#loading the dataset(pima dataset)
filename='pima-indians-diabetes.data.csv'
data=np.loadtxt(filename,delimiter=",")
#splitting the data into i/p and o/p
x=data[:,0:8]
y=data[:,8]
#for certain amount of epochs
#creating the model
model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.load_weights('weights.best.hdf5')

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

score=model.evaluate(x,y)
print('%.2f%%'%(score[1]*100))