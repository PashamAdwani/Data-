# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 01:00:53 2018

@author: ADWANI
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
import json
import scipy as sc
import numpy as np
import keras
import matplotlib.pylab as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten


train_file=pd.read_json(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\train.json', orient='records')
test_file=pd.read_csv(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\sample_solution.csv', header=0)    
train_inp=pd.read_excel(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\Train_Input.xlsx',index_col=None,header=None)
test_inp=pd.read_excel(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\Test_Input.xlsx',index_col=None,header=None)
train_l=train_file['cuisine']
test_l=test_file['cuisine']
train_label=train_l[0:4000]
test_label=test_l[0:1000]

inpshape=(6714,)
y_labels=np.zeros((4000,1))
map_to_int={ele:cnt for cnt,ele in enumerate(train_label.unique())}
y_labels=train_label.replace(map_to_int)  
y_test=np.zeros((1000,1))
y_test=test_label.replace(map_to_int)  


classifier=Sequential()
classifier.add(Dense(64,input_shape=inpshape,activation='relu'))
classifier.add(Dense(32,input_shape=inpshape,activation='relu'))
classifier.add(Dense(20,input_shape=inpshape,activation='softmax'))
classifier.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
classifier_fit=classifier.fit(train_inp, y_labels,epochs=20,validation_data=(test_inp,y_test))
#z=classifier.predict(test_inp)
#classifier.evaluate(z,y_test)

epochs=20
train_loss=classifier_fit.history['loss']
val_loss=classifier_fit.history['val_loss']
train_acc=classifier_fit.history['acc']
val_acc=classifier_fit.history['val_acc']
xc=range(epochs)


plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])