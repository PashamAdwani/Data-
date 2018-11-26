import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
import json
import scipy as sc
import numpy as np

train_file=pd.read_json(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\train.json', orient='records')
test_file=pd.read_csv(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\sample_solution.csv', header=0)    
train_inp=pd.read_excel(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\Train_Input.xlsx',index_col=None,header=None)
test_inp=pd.read_excel(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\Test_Input.xlsx',index_col=None,header=None)
train_l=train_file['cuisine']
test_l=test_file['cuisine']
train_label=train_l[0:4000]
test_label=test_l[0:1000]




best=0
accuracy=0
hl=[1,3]
act=['logistic', 'tanh', 'relu']
sol=['lbfgs','sgd','adam']
al=[0.0001,0.0005]
bs=[64,128]
lr=['constant','invscaling','adaptive']
best_params = [0,0,0,0,0,0]
params = [0,0,0,0,0,0]
for h in hl:
    for a in act:
        for s in sol:
            for a1 in al:
                for b in bs:
                    for l in lr:
                        classifier=MLPClassifier(hidden_layer_sizes=h,activation=a,solver=s,alpha=a1,batch_size=b,learning_rate=l)
                        classifier.fit(train_inp,train_label)
                        ypred=classifier.predict(test_inp)
                        #acc=classifier.score(ypred,test_label)
                        x=0
                        score=0
                        for i in ypred:
                            if(i==test_label[x]):
                                score=score+1
                        accuracy=score/10
                        params=[h,a,s,a1,b,l]
                        print("=========")
                        print('Accuracy:',accuracy)
                        print('Params:',params)
                        print("=========")
                        if(best<=accuracy):
                            best=accuracy
                            best_params=[h,a,s,a1,b,l]

print('Best Parameters:',best_params)
print('Best Accuracy:',best)