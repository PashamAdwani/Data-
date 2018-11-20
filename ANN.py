import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
import json
import scipy as sc
import numpy as np
impoert keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten

train_file=pd.read_json(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\train.json', orient='records')
test_file=pd.read_json(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\test.json', orient='records')    
test_output=pd.read_csv(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\sample_solution.csv',header=0)
##================
#Features
##================

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

print("It will take time")
x=0
y=0
for i in train_input:
	for j in i:
		y=0
		for k in element:
			if(j==k):
				train[x,y]=1
			y=y+1
	x=x+1
print("over")
print("It will take time")
x=0
y=0
for i in test_input:
	for j in i:
		y=0
		for k in element:
			if(j==k):
				test[x,y]=1
			y=y+1
	x=x+1

print("Over")

y_labels=np.zeros((39774,1))
map_to_int={ele:cnt for cnt,ele in enumerate(train_output.unique())}
y_labels=train_output.replace(map_to_int)     
        

inpshape=(6714,)
y_labels1=keras.utils.to_categorical(y_labels, num_classes=20)
classifier7=Sequential()
classifier7.add(Dense(64,input_shape=inpshape,activation='relu'))
classifier7.add(Dense(32,input_shape=inpshape,activation='relu'))
classifier7.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
classifier7.fit(train[0:39774,:],y_labels)
classifier7.predict(test)
