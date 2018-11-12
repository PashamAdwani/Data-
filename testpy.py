import pandas as pd

df.to_json(orient='index')
dt=pd.read_json('D:\Data Analysis\Project2\Project2\train.json',orient='split')

