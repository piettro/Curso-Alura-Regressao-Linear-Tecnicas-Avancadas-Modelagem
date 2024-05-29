import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data/dataset.csv',sep=';')

sns.set_palette("Accent")
sns.set_style("darkgrid")

ax = sns.boxplot(data=data['Valor'], orient='h', width=0.3)
ax.figure.set_size_inches(20, 5)
ax.set_title('Real State Prices', fontsize=20)
ax.set_xlabel('Reais', fontsize=16)
plt.show()

ax = sns.histplot(data['Valor'])
ax.figure.set_size_inches(20, 6)
ax.set_title('Frequency Distribution', fontsize=20)
ax.set_xlabel('Real State Prices (R$)', fontsize=16)
plt.show()

ax = sns.pairplot(data, y_vars='Valor', x_vars=['Area', 'Dist_Praia', 'Dist_Farmacia'], kind='reg', height=5)
ax.fig.suptitle('Dispersion beetwen Variables', fontsize=20, y=1.05)
plt.show()
