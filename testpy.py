import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
import json

train_file=pd.read_json(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\train.json', orient='records')
test_file=pd.read_json(r'D:\Data Analysis\Project2\Project2\Data-Analysis-Project2\test.json', orient='records')    

train_input=train_file['ingredients']
train_output=train_file['cuisine']
test_input=test_file['ingredients']
test_output=test_file['cuisine']


MLP=MLPClassifier()
