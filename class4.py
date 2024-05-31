import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

data = pd.read_csv('data/dataset.csv',sep=';')
data['log_value'] = np.log(data['Valor'])
data['log_area'] = np.log(data['Area'])
data['log_dist_beach'] = np.log(data['Dist_Praia'] + 1)
data['log_dist_pharmacy'] = np.log(data['Dist_Farmacia'] + 1)

y = data['log_value']
x = data[['log_area','log_dist_beach','log_dist_pharmacy']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2811)
x_train_with_constant = sm.add_constant(x_train)

model_statsmodels = sm.OLS(endog=y_train,exog=x_train_with_constant,hasconst=True).fit()
print('============== Summary Model 1 ==============')
print(model_statsmodels.summary())

y_new_model = data['log_value']
x_new_model = data[['log_area','log_dist_beach']]

x_train_new_model, x_test_new_model, y_train_new_model, y_test_new_model = train_test_split(x_new_model, y_new_model, test_size=0.2, random_state=2811)
x_train_with_constant_new_model = sm.add_constant(x_train_new_model)

model_statsmodels_new_model = sm.OLS(endog=y_train_new_model,exog=x_train_with_constant_new_model,hasconst=True).fit()
print('============== Summary Model 2 ==============')
print(model_statsmodels_new_model.summary())