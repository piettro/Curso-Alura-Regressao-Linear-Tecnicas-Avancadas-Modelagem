import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

data = pd.read_csv('data/dataset.csv',sep=';')
data['log_value'] = np.log(data['Valor'])
data['log_area'] = np.log(data['Area'])
data['log_dist_beach'] = np.log(data['Dist_Praia'] + 1)
data['log_dist_pharmacy'] = np.log(data['Dist_Farmacia'] + 1)

y = data['log_value']
x = data[['log_area','log_dist_beach']]

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=2811)

model =  LinearRegression() 
model.fit(x_train, y_train)
print('R² = {}'.format(model.score(x_train, y_train)))

y_predict = model.predict(x_test)
print('R² = %s' % metrics.r2_score(y_test, y_predict))

area = 250
dist_beach = 1
entry = [[np.log(area), np.log(dist_beach + 1)]]

print('R$ {0: 2f}'.format(np.exp(model.predict(entry)[0])))