# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:27:51 2018

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
import xlsxwriter as xs


train_file=pd.read_json(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\train.json', orient='records')
test_file=pd.read_json(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\test.json', orient='records')    
test_output=pd.read_csv(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\sample_solution.csv',header=0)
##================
#Features
##================
test_labels=test_output['cuisine']
train_input=train_file['ingredients']
#train_input=train_input.astype('float32',(20,))
train_output=train_file['cuisine']
test_input=test_file['ingredients']

t=[]
for i in train_input:
    t.extend(i)

ing=set(t)
print(len(ing),' is the length of the ingreduents in train_input')

te=[]
for i in test_input:
    te.extend(i)

ingte=set(te)
print(len(ingte),' is the length of the ingredients in test input')    

element=list(ing)
train=np.zeros((39775,6714))
test=np.zeros((9945,6714))

##============== Train File
print("It will take time")
x=0
y=0

for i in train_input[0:39774]:
    for j in i:
        y=0
        for k in element:
            if(j==k):
                train[x,y]=1
        y=y+1 
    if(x<=39773):   
        x=x+1
print("over")

##============== Writing Train  xlsx File
print('Writing to the xlsx training file')
workbook_train=xs.Workbook('D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\Train_Input.xlsx')
worksheet_train=workbook_train.add_worksheet()
i=0
j=0
for i in range(39774):
    for j in range(6714):
        worksheet_train.write(i,j,train[i,j])
workbook_train.close()

##============== Test File=======
print("It will take time")
x=0
y=0

for i in test_input[0:9944]:
    for j in i:
        y=0
        for k in element:
            if(j==k):
                test[x,y]=1
        y=y+1 
    if(x<=9943):   
        x=x+1
print("over")

##==============Writing  Test File
i=0
j=0
print('Writing to the xlsx testing file')
workbook_test=xs.Workbook('D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\Test_Input.xlsx')
worksheet_test=workbook_test.add_worksheet()
for i in range(9944):
    print(i)
    for j in range(6714):
        worksheet_test.write(i,j,test[i,j])
workbook_test.close()
