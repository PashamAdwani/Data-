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
classifier=Sequential()
classifier.add(Dense(64,input_shape=inpshape,activation='relu'))
classifier.add(Dense(32,input_shape=inpshape,activation='relu'))
classifier.add(Dense(16,input_shape=inpshape,activation='softmax'))
classifier.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
classifier.fit(train_inp, y_labels,epochs=10)
z=classifier.predict(test_inp)
#classifier.score(z,test_label)