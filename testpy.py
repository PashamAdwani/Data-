import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
import json
import scipy as sc

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

##t=train_input
##for i in t:
##	for z in range(len(i),65):
##		i.extend('X')


#test_input=test_input.astype('float32',(20,))



best=0
accuracy=0
hl=[1]
act=['logistic', 'tanh', 'relu']
sol=['lbfgs','sgd','adam']
al=[0.0001,0.0005]
bs=[64,128]
lr=['constant','invscaling','adaptive']
best_params = [0,0,0,0,0,0]
for h in hl:
    for a in act:
        for s in sol:
            for a1 in al:
                for b in bs:
                    for l in lr:
                        classifier=MLPClassifier(hidden_layer_sizes=h,activation=a,solver=s,alpha=a1,batch_size=b,learning_rate=l)
                        classifier.fit(train_input,train_output)
                        ypred=classifier.predict(test_input)
                        



print(ypred)
