# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 00:16:45 2018

@author: Abhilash Srivastava
"""
"""
this project is built on the adam neural
network model which trains on the pima
diabetes datasets and is done on CPU
"""
from keras.callbacks import ModelCheckpoint
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
#filepath="weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
#to checkpoint for the improvements in acc
filepath='weights.best.hdf5'
cb=ModelCheckpoint(filepath,monitor='acc',verbose=1,save_best_only=True,mode='max',period=3)
#creating the model
model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
#model.load_weights('weights.best.hdf5')
#compile the model
#binary_crossentropy for the loss function
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#train the model
model.fit(x,y,epochs=150,callbacks=[cb],batch_size=10)
'''
#evaluate the model
scores=model.evaluate(x,y)
print("%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))
'''