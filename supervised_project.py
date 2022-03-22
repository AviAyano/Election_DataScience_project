# -*- coding: utf-8 -*-
"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

election = pd.read_csv('/content/dataB.csv')
features = pd.read_csv('/content/dataC.csv')
feature = features[["yshuv","Totalpop16","AhuzOlim1990","Bagrut_1516","Eshkol_15","Shetah_shiput_13","DiabRate14_16"]]

feature.isna().sum()

feature.loc[feature.AhuzOlim1990.isna(), 'AhuzOlim1990'] = feature.AhuzOlim1990.mean()

feature.isna().sum()

election.shape

feature.rename(columns={'yshuv': 'yshuv_symbol'}, inplace=True)

election.rename(columns={'To': 'to'}, inplace=True)
election.rename(columns={'Symbol of a settlement': 'yshuv_symbol'}, inplace=True)

election.rename(columns={'To.1': 'TO'}, inplace=True)

election.columns

feature.columns

data = pd.merge(feature,election, on ='yshuv_symbol',how='inner')
data.shape

data.rename(columns={'Totalpop16': 'Totalpop'}, inplace=True)

data.columns

data['The Arabs Bloc'] = data['And the same'] 
data['The_Left_Bloc'] = data['Truth'] + data['Meretz'] + data['The Arabs Bloc']
data['The_Center_Bloc'] = data['Here'] 
data['The Ultra-Orthodox Bloc'] = data['third'] +data['Shot']
data['The_Right_Bloc'] =  data['TO'] + data['Tab'] + data['about'] + data['forgave'] + data['The Ultra-Orthodox Bloc']

data.columns

data.drop(['BJ','Insulations','Truth','third',
       'And the same', 'G', 'pure', 'Zets', 'Tab', 'to', 'Yaz', 'Will', 'Jack',
       'about', 'TO', 'forgave', 'Meretz', 'Naz', 'Ni', 'Hawk', 'P.', 'Here',
       'Twisted', 'And run', 'CAN', 'end', 'only', 'Shot'] , axis = 1, inplace = True )

data['Totalpop'] = data['Totalpop'] * 1000

data['The_Right_Bloc%'] = data.The_Right_Bloc / data.Kosher
data['The_Left_Bloc%'] = data.The_Left_Bloc / data.Kosher
data['The_Center_Bloc%'] = data.The_Center_Bloc / data.Kosher
data['Vote%'] = data['Kosher'] / data['Totalpop']
data['PopDensity%'] = data['Totalpop'] / data['Shetah_shiput_13']

data.head()

data.isna().sum() #count the number of NAN in each column

data.drop(['yshuv_symbol','BJ','Voters','Insulations'] , axis = 1, inplace = True ) #prepare to linear regression modeling

data.columns

data.drop(columns=['Voters', 'Kosher', 'The Arabs Bloc',	'The_Left_Bloc', 'The_Center_Bloc',	'The Ultra-Orthodox Bloc', 'The_Right_Bloc']).corr()

data.describe()

X = data.drop(columns=['Name of settlement', 'Voters', 'Kosher', 'The Arabs Bloc',	'The_Left_Bloc', 'The_Center_Bloc',	'The Ultra-Orthodox Bloc', 'The_Right_Bloc', 
                       'yshuv_symbol', 'The_Left_Bloc%', 'The_Right_Bloc%' ,'The_Center_Bloc%']).copy()
y = data['The_Right_Bloc%'].copy()

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
# %matplotlib inline

sns.pairplot(data, x_vars=['Totalpop', 'AhuzOlim1990', 'Bagrut_1516', 'Eshkol_15',
       'Shetah_shiput_13', 'DiabRate14_16', 'Vote%', 'PopDensity%'
       ], y_vars= 'The_Right_Bloc')

reg = np.polyfit(data['Totalpop'],data['The_Right_Bloc'],deg=1)
reg

trend = np.polyval(reg,data['Totalpop'])
plt.scatter(data['Totalpop'],data['The_Right_Bloc'])
plt.plot(data['Totalpop'],trend, 'r')

reg = np.polyfit(data['PopDensity%'],data['The_Right_Bloc'],deg=1)
reg

trend = np.polyval(reg,data['PopDensity%'])
plt.scatter(data['PopDensity%'],data['The_Right_Bloc'],color='g')
plt.plot(data['PopDensity%'],trend, 'k')

scaler =  MinMaxScaler()
X = scaler.fit_transform(X)
# y = scaler.fit_transform(y.to_frame())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2 )

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

LR = LinearRegression()

pd.DataFrame(X_train).isna().sum()

LR.fit(X_train, y_train)

prediction = LR.predict(X_test)

from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error

mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
print("The LR model evaluation scores are:")
print(f'mse={mae}')
print(f'mse={mse}')
print(f'r2={r2}')

"""# KNN Algorithm"""

from sklearn.neighbors import KNeighborsClassifier  # להביא את המסווג 
from sklearn import metrics 
from sklearn import preprocessing

lab_enc = preprocessing.LabelEncoder()
# encoded = lab_enc.fit_transform(y_train)
k_num = np.arange(1,16,2)
list_k = []
for k in k_num:
  model = KNeighborsClassifier(k) #k=1,3,5,...
  y_train = lab_enc.fit_transform(y_train)
  model.fit(X_train,y_train ) #אימוון המודל
  result_knn_target = model.predict(X_test)
  y_test = lab_enc.fit_transform(y_test)
  list_k.append(metrics.accuracy_score(y_test, result_knn_target))
plt.subplots(1,1,figsize = (6,5))
plt.plot(k_num,list_k)
plt.xlabel("K")
plt.ylabel("Accuracy score")
plt.grid()
plt.show

lab_enc = preprocessing.LabelEncoder()
model = KNeighborsClassifier(5) #k=3
y_train = lab_enc.fit_transform(y_train)
model.fit(X_train,y_train ) 
esult_knn_target = model.predict(X_test)
y_test = lab_enc.fit_transform(y_test)

mae = mean_absolute_error(y_test, result_knn_target)
mse = mean_squared_error(y_test, result_knn_target)
r2 = r2_score(y_test, result_knn_target)
print("The KNN model evaluation scores are:",metrics.accuracy_score(y_test, result_knn_target))
print(f'mse={mae}')
print(f'mse={mse}')
print(f'r2={r2}')

