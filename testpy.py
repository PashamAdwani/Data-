import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
import json

train_file=pd.read_json(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\train.json', orient='records')
test_file=pd.read_json(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\test.json', orient='records')    
test_output=pd.read_csv(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\sample_solution.csv',header=0)

test_labels=test_output['cuisine']

train_input=train_file['ingredients']
train_input=train_input.astype('str',(10,))
train_output=train_file['cuisine']

test_input=test_file['ingredients']
test_input=test_input.astype('str',(10,))



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
