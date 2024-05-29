import pandas as pd

data = pd.read_csv('data/dataset.csv',sep=';')

print('============== Dataframe Head ==============')
print(data.head(10))
print('============== Dataframe Shape ==============')
print(f'Rows: {data.shape[0]} \nColumns: {data.shape[1]}')
print('============== Dataframe Describe ==============')
print(data.describe().round(2))
print('============== Dataframe Corr ==============')
print(data.corr().round(4))