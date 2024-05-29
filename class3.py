import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/dataset.csv',sep=';')

sns.set_palette("Accent")
sns.set_style("darkgrid")

data['log_value'] = np.log(data['Valor'])
data['log_area'] = np.log(data['Area'])
data['log_dist_beach'] = np.log(data['Dist_Praia'] + 1)
data['log_dist_pharmacy'] = np.log(data['Dist_Farmacia'] + 1)

ax = sns.histplot(data['log_Valor'])
ax.figure.set_size_inches(20,6)
ax.set_title('Frequency Distribution', fontsize=20)
ax.set_xlabel('Log Real State Prices', fontsize=16)
plt.show()

ax = sns.pairplot(data, y_vars='log_value', x_vars=['log_area', 'log_dist_beach', 'log_dist_pharmacy'], height=5)
ax.fig.suptitle('Dispersion beetwen Log Variables', fontsize=20, y=1.05)
plt.show()